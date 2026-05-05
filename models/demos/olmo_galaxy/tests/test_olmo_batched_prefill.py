# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
OLMo-3.1-32B Batched Prefill (B32) Test.

Verifies that:
1. TtOlmoModelArgs.supports_batched_prefill is True
2. Batched prefill (batch_size=32, ISL=128) runs without crash
3. Batched output for user-0 has PCC >= 0.98 vs sequential B1 output

The key fix being tested is the QK-norm reshape bug:
  Before fix: q_flat = reshape([32, S, 5, 128] → [1, 1, S, 640])  # wrong - total elem mismatch
  After fix:  q_flat = reshape([32, S, 5, 128] → [1, 1, 32*S, 640])  # correct

Run with:
    export ARCH_NAME=wormhole_b0 && export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd)
    source python_env/bin/activate
    export HF_MODEL=~/models/OLMo-3.1-32B-Think
    export LINE_RS=1
    pytest models/demos/olmo_galaxy/tests/test_olmo_batched_prefill.py -xvs
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.olmo_galaxy.tt.llama_attention import TtLlamaAttention
from models.demos.olmo_galaxy.tt.llama_ccl import TT_CCL
from models.demos.olmo_galaxy.tt.llama_common import gather_cos_sin, get_rot_transformation_mat, precompute_freqs_yarn
from models.demos.olmo_galaxy.tt.olmo_model_config import TtOlmoModelArgs
from models.demos.olmo_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup

SEQ_LEN = 128
BATCH_SIZE = 32
LAYER_NUM = 0
PCC_THRESHOLD = 0.98  # bfloat8_b + multi-device


def _make_attention(mesh_device, model_args):
    """Instantiate TtLlamaAttention and return (tt_attention, tt_ccl, rot_mats)."""
    dtype = ttnn.bfloat8_b
    prefetcher_setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=1, mode="prefill")
    mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])
    tt_ccl = TT_CCL(
        mesh_device,
        model_args,
        prefetcher_setup.worker_sub_device_id,
        mode="prefill",
        is_qwen=False,
        is_olmo=True,
    )

    head_dim = model_args.head_dim
    trans_mat = ttnn.from_torch(
        get_rot_transformation_mat(dhead=head_dim),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    transformation_mats = {"decode": trans_mat, "prefill": trans_mat}

    state_dict = model_args.load_state_dict()
    model_args.WEIGHTS_DTYPE = dtype
    tt_attention = TtLlamaAttention(
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=LAYER_NUM,
        dtype=dtype,
        transformation_mats=transformation_mats,
        configuration=model_args,
        prefetcher_setup=prefetcher_setup,
        tt_ccl=tt_ccl,
    )

    ttnn_cos, ttnn_sin, _ = precompute_freqs_yarn(
        dim=head_dim,
        end=model_args.max_seq_len * 2,
        theta=model_args.rope_theta,
        scaling_factor=model_args.rope_scaling_factor,
        original_max_position_embeddings=model_args.original_max_position_embeddings,
        beta_fast=model_args.yarn_beta_fast,
        beta_slow=model_args.yarn_beta_slow,
        attention_factor=model_args.yarn_attention_factor,
    )
    position_ids = torch.arange(SEQ_LEN)
    cos_gathered, sin_gathered = gather_cos_sin(position_ids, ttnn_cos, ttnn_sin)
    rot_mats = [
        ttnn.from_torch(
            cos_gathered,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
        ttnn.from_torch(
            sin_gathered,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
    ]
    return tt_attention, tt_ccl, rot_mats


def _run_attention(tt_attention, model_args, rot_mats, x_torch, batch_size):
    """Run forward_prefill and return torch tensor [B, 1, S, D].

    For batch_size=1: output tensor is [1, 1, S, D].
    For batch_size=32: TTNN output seq dim is B*S (flattened), so we reshape to [B, 1, S, D].
    """
    tt_input = model_args.prepare_residual_tensor_prefill(x_torch)
    tt_out = tt_attention.forward_prefill(tt_input, rot_mats, user_id=0, batch_size=batch_size)
    dim = model_args.dim
    total_seq = batch_size * SEQ_LEN
    out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            model_args.mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape
        ),
    )
    # out shape after concat: [1, num_row_devices, total_seq, dim]
    # Take only row-device-0 (all row devices hold same reduced value after fast_reduce_nc)
    out = out[:, :1, :total_seq, :dim]  # [1, 1, B*S, D]
    if batch_size > 1:
        out = out.reshape(batch_size, 1, SEQ_LEN, dim)  # [B, 1, S, D]
    return out


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        }
    ],
    indirect=True,
)
class TestOlmoBatchedPrefill:
    def test_supports_batched_prefill_flag(self, mesh_device, device_params):
        """TtOlmoModelArgs.supports_batched_prefill must be True."""
        model_args = TtOlmoModelArgs(mesh_device, max_batch_size=32, max_seq_len=512)
        assert getattr(
            model_args, "supports_batched_prefill", False
        ), "supports_batched_prefill must be True — set it in TtOlmoModelArgs"
        logger.info("supports_batched_prefill = True ✓")

    def test_batched_prefill_no_crash(self, mesh_device, device_params, reset_seeds, ensure_gc):
        """Batched prefill B32/ISL=128 runs without shape errors or NaNs.

        Note: batched prefill packs 32 users into one forward call producing a total
        seq_len of 32*128=4096. In production, paged attention routes each user's K/V
        to its own cache block. Here we use non-paged attention with max_seq_len > 4096
        so fill_cache does not OOB (KV cache correctness is not the test objective).
        """
        if not os.getenv("HF_MODEL"):
            pytest.skip("HF_MODEL not set")

        # max_seq_len must cover the total batched seq_len (BATCH_SIZE * SEQ_LEN = 4096)
        model_args = TtOlmoModelArgs(mesh_device, max_batch_size=BATCH_SIZE, max_seq_len=BATCH_SIZE * SEQ_LEN * 2)
        model_args.n_layers = 1
        model_args.use_prefetcher = False

        tt_attention, tt_ccl, rot_mats = _make_attention(mesh_device, model_args)

        try:
            torch.manual_seed(42)
            x = torch.randn(BATCH_SIZE, SEQ_LEN, model_args.dim) * 0.1

            logger.info(f"Running batched prefill: B={BATCH_SIZE}, ISL={SEQ_LEN}")
            out = _run_attention(tt_attention, model_args, rot_mats, x, BATCH_SIZE)

            assert out.shape == (BATCH_SIZE, 1, SEQ_LEN, model_args.dim), f"Unexpected shape: {out.shape}"
            assert not torch.isnan(out).any(), "NaN in batched prefill output"
            assert not torch.isinf(out).any(), "Inf in batched prefill output"
            logger.info(f"Batched prefill smoke test PASSED, output shape: {out.shape}")
        finally:
            tt_ccl.close()

    def test_batched_prefill_pcc_vs_single(self, mesh_device, device_params, reset_seeds, ensure_gc):
        """
        User-0 output from batched B32 must match sequential B1 (PCC >= 0.98).

        This validates the QK-norm fix: batch and seq dims are correctly merged/split
        around the distributed RMSNorm in forward_prefill.
        """
        if not os.getenv("HF_MODEL"):
            pytest.skip("HF_MODEL not set")

        model_args = TtOlmoModelArgs(mesh_device, max_batch_size=BATCH_SIZE, max_seq_len=BATCH_SIZE * SEQ_LEN * 2)
        model_args.n_layers = 1
        model_args.use_prefetcher = False

        tt_attention, tt_ccl, rot_mats = _make_attention(mesh_device, model_args)

        try:
            torch.manual_seed(0)
            x_all = torch.randn(BATCH_SIZE, SEQ_LEN, model_args.dim) * 0.1

            # Batched B32: all users in one forward pass
            logger.info("Running batched B32 prefill...")
            batched_out = _run_attention(tt_attention, model_args, rot_mats, x_all, BATCH_SIZE)
            user0_batched = batched_out[0:1]  # [1, 1, S, D]

            # Sequential B1: only user 0
            logger.info("Running sequential B1 prefill for user 0...")
            single_out = _run_attention(tt_attention, model_args, rot_mats, x_all[0:1], 1)
            # single_out: [1, 1, S, D]

            passing, pcc_msg = comp_pcc(user0_batched, single_out, PCC_THRESHOLD)
            logger.info(f"Batched B32 vs B1 PCC for user 0: {pcc_msg}")
            assert passing, f"Batched prefill PCC {pcc_msg} < {PCC_THRESHOLD}"
            logger.info("PCC test PASSED!")
        finally:
            tt_ccl.close()
