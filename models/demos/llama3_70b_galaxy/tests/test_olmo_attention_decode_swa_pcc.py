# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
OLMo single-layer attention decode PCC test: sliding_window=4096 vs None.

APPROACH
--------
The float32 reference KV cache diverges from the device's bfloat8 cache every
step, making step-vs-reference comparison noisy. Instead, this test compares
two consecutive TT runs that use identical random inputs:

  Run A: sliding_window_size=None   → outputs_A  (baseline, the "fix")
  Run B: sliding_window_size=4096   → outputs_B  (the bug)

For positions 0..4095, full-attention == SWA mathematically, so both runs go
through exactly the same bfloat8 quantization path and should agree tightly.
Any PCC drop between A and B before step 4096 is the bug.

After step 4096, some divergence is mathematically expected (SWA truly differs
from full attention) — so we only assert agreement up to the window boundary.

OLMo POST-NORM architecture note
---------------------------------
In decode mode (llama_decoder.py lines 272-332) the decoder does:
    attn_in = x_res_dram   (raw residual — NO pre-attention norm applied)
    attn_out = attention.forward(attn_in)
    attn_normed = ff_norm(attn_out)   ← post-attention norm
    ...

So TtLlamaAttention receives the raw residual as input, which is what we feed here.

Run:
    export HF_MODEL=~/models/models--allenai--Olmo-3.1-32B-Think
    pytest models/demos/llama3_70b_galaxy/tests/test_olmo_attention_decode_swa_pcc.py -v -s
"""

import torch
import pytest
from loguru import logger
import ttnn

from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
from models.demos.llama3_70b_galaxy.tt.llama_attention import TtLlamaAttention
from models.demos.llama3_70b_galaxy.tt.llama_rope import TtLlamaRotarySetup
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL
from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig
from models.common.utility_functions import comp_pcc


NUM_DECODE_STEPS = 600
BATCH_SIZE = 32
MAX_SEQ_LEN = 1024  # must be >= NUM_DECODE_STEPS
PAGE_BLOCK_SIZE = 64
PAGE_MAX_NUM_BLOCKS = (MAX_SEQ_LEN // PAGE_BLOCK_SIZE) * BATCH_SIZE
LAYER_NUM = 0  # layer 0 has sliding_window=4096 for OLMo
SEED = 7


def _setup_device(mesh_device, model_args):
    """Create sub-device and CCL (OLMo: no prefetcher)."""
    all_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])
    all_sub_device = ttnn.SubDevice([all_core_range_set])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([all_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])
    tt_ccl = TT_CCL(mesh_device, model_args, worker_sub_device_id, mode="decode", is_olmo=True)
    return tt_ccl


def _make_attention(
    mesh_device, model_args, state_dict, rope_setup, paged_attention_config, tt_ccl, use_sliding_window_decode: bool
):
    """Create a TtLlamaAttention with fresh KV cache and the chosen sliding window mode."""
    dtype = ttnn.bfloat8_b
    transformation_mats = rope_setup.get_both_trans_mats()
    tt_attn = TtLlamaAttention(
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=LAYER_NUM,
        dtype=dtype,
        transformation_mats=transformation_mats,
        configuration=model_args,
        paged_attention_config=paged_attention_config,
        use_paged_kv_cache=False,  # attention owns the paged KV cache
        prefetcher_setup=None,
        tt_ccl=tt_ccl,
    )
    tt_attn._use_sliding_window_decode = use_sliding_window_decode
    return tt_attn


def _run_one_pass(mesh_device, model_args, model_config, rope_setup, page_table_tt, use_sliding_window_decode: bool):
    """
    Run NUM_DECODE_STEPS decode steps and return list of per-step output tensors (CPU).
    Reinitializes TT_CCL and TtLlamaAttention each call for a fresh KV cache.
    """
    state_dict = model_args.load_state_dict()
    paged_attention_config = PagedAttentionConfig(block_size=PAGE_BLOCK_SIZE, max_num_blocks=PAGE_MAX_NUM_BLOCKS)

    tt_ccl = _setup_device(mesh_device, model_args)
    tt_attn = _make_attention(
        mesh_device, model_args, state_dict, rope_setup, paged_attention_config, tt_ccl, use_sliding_window_decode
    )

    label = f"sw={'4096' if use_sliding_window_decode else 'None'}"
    logger.info(f"[{label}] Starting {NUM_DECODE_STEPS}-step decode run")

    outputs = []
    torch.manual_seed(SEED)

    for step in range(NUM_DECODE_STEPS):
        pt_input = (torch.rand(BATCH_SIZE, 1, model_args.dim) * 2) - 1

        current_pos_torch = torch.full((BATCH_SIZE,), step, dtype=torch.int32)
        current_pos_tt = ttnn.from_torch(
            current_pos_torch,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=model_args.cluster_shape),
        )
        rot_mats = rope_setup.get_rm_rot_mats(current_pos_torch)

        # OLMo decode: raw residual goes directly to attention (no pre-norm)
        tt_residual = model_args.prepare_residual_tensor_decode(
            pt_input.clone(), model_config["DECODE_RESIDUAL_MEMCFG"]
        )
        attn_in = ttnn.to_memory_config(
            ttnn.to_memory_config(tt_residual, ttnn.DRAM_MEMORY_CONFIG),
            model_config["SHARDED_ATTN_INPUT_RING_MEMCFG"],
        )
        ttnn.deallocate(tt_residual)

        tt_out = tt_attn.forward(
            attn_in,
            current_pos=current_pos_tt,
            rot_mats=rot_mats,
            user_id=0,
            mode="decode",
            page_table=page_table_tt,
        )
        out_cpu = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )[:, 0:1, :BATCH_SIZE, : model_args.dim].view(BATCH_SIZE, 1, model_args.dim)

        outputs.append(out_cpu)

        if step < 20 or step % 50 == 0:
            logger.info(f"[{label}] step={step:4d} done")

    tt_ccl.close()
    return outputs


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": True}],
    indirect=True,
)
def test_olmo_attention_decode_swa_vs_none(mesh_device, reset_seeds, ensure_gc):
    """
    Compare two identical OLMo single-layer decode runs — one with sliding_window=None
    (fix) and one with sliding_window=4096 (original, potentially buggy) — using the
    same random inputs and same SEED.

    For positions 0..4095: SWA == full attention mathematically.
    Both runs go through identical bfloat8 quantization, so outputs must agree tightly.
    Any PCC drop BEFORE step 4096 is the decode sliding-window bug.

    After step 4095, mathematical divergence is expected and the test does not assert.
    """
    model_args = TtOlmoModelArgs(mesh_device, max_batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    model_args.n_layers = 1
    model_args.use_prefetcher = False
    model_config = model_args.get_model_config()

    rope_setup = TtLlamaRotarySetup(
        mesh_device,
        BATCH_SIZE,
        model_args.head_dim,
        MAX_SEQ_LEN,
        model_args.rope_theta,
        model_args.use_scaled_rope,
        model_args.rope_scaling_factor,
    )

    # Shared page table: user i → sequential blocks [i*bpu .. (i+1)*bpu)
    blocks_per_user = MAX_SEQ_LEN // PAGE_BLOCK_SIZE
    page_table_torch = torch.arange(BATCH_SIZE * blocks_per_user, dtype=torch.int32).reshape(
        BATCH_SIZE, blocks_per_user
    )
    page_table_tt = ttnn.from_torch(
        page_table_torch,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -2), mesh_shape=model_args.cluster_shape),
    )

    # Run A: sliding_window=None (fix)
    outputs_none = _run_one_pass(
        mesh_device, model_args, model_config, rope_setup, page_table_tt, use_sliding_window_decode=False
    )
    # Run B: sliding_window=4096 (bug candidate)
    outputs_swa = _run_one_pass(
        mesh_device, model_args, model_config, rope_setup, page_table_tt, use_sliding_window_decode=True
    )

    # Compare outputs step by step — only assert for positions < sliding_window
    sliding_window = model_args.get_sliding_window_size(LAYER_NUM)  # 4096
    pcc_threshold = 0.99  # should be near-identical since both use same bfloat8 path

    first_failure_step = None
    for step, (out_none, out_swa) in enumerate(zip(outputs_none, outputs_swa)):
        passing, pcc_val = comp_pcc(out_none, out_swa, pcc_threshold)
        if step < 20 or step % 50 == 0:
            logger.info(f"step={step:4d} | PCC(None vs 4096)={pcc_val} | {'PASS' if passing else 'FAIL'}")
        if not passing and step < sliding_window and first_failure_step is None:
            first_failure_step = step
            logger.error(
                f"BUG FOUND: sliding_window=4096 diverges from sliding_window=None "
                f"at step={step} (pos {step} < window {sliding_window}): PCC={pcc_val}"
            )

    if first_failure_step is not None:
        pytest.fail(
            f"sliding_window=4096 diverges from None at step={first_failure_step} "
            f"(before window boundary {sliding_window}). This is the decode sliding-window bug."
        )
    else:
        logger.info(
            f"PASS: sliding_window=4096 and None agree (PCC>{pcc_threshold}) "
            f"for all {min(NUM_DECODE_STEPS, sliding_window)} steps below the window boundary."
        )
