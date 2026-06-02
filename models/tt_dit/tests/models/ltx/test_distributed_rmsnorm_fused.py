# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
A/B parity test for the LTX distributed RMSNorm fused-op swap.

``DistributedRMSNorm.forward`` has two code paths, selected by the ``LTX_FUSED_AGRMS`` env var:
  * default        -> wan ``rmsnorm_pre_allgather`` -> ``all_gather`` -> ``rmsnorm_post_allgather``
  * LTX_FUSED_AGRMS -> the single fused ``ttnn.all_gather_rms_norm`` op (this PR)

This test builds the norm EXACTLY as the LTX model does (same ctor kwargs / call signature), feeds the
same sharded activation, and runs ``.forward`` both ways, asserting the two outputs match. It covers the
two ways LTX invokes the norm:
  * "block"  : norm1/2/3, audio_norm* -> non-affine, num_heads_per_device=1 (output keeps input shape).
  * "qk"     : norm_q/norm_k          -> affine (weight), num_heads_per_device = num_heads/TP (head-split,
               output reshaped to (1, num_heads, B*N, head_dim)).

RoPE is NOT part of either norm path (LTX applies it as a separate rotary op afterward), so it is not
exercised here. Mesh is (2,4) with the LTX SP/TP axis assignments; the gather runs over the TP axis.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.layers.normalization import DistributedRMSNorm
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.tensor import bf16_tensor_2dshard
from models.tt_dit.utils.test import line_params
from tests.ttnn.utils_for_testing import assert_with_pcc

NUM_HEADS = 32  # LTX video and audio both use 32 attention heads


@pytest.mark.parametrize("device_params", [line_params], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize(
    "sp_axis, tp_axis",
    [(1, 0), (0, 1)],
    ids=["sp1tp0_tp2", "sp0tp1_tp4"],
)
@pytest.mark.parametrize("dim", [4096, 2048], ids=["video4096", "audio2048"])
@pytest.mark.parametrize("case", ["block", "qk"], ids=["block", "qk"])
@pytest.mark.parametrize("seq_len", [512], ids=["seq512"])
def test_distributed_rmsnorm_fused_parity(mesh_device, sp_axis, tp_axis, dim, case, seq_len):
    if mesh_device.get_num_devices() < 8:
        pytest.skip("needs 8 devices for a 2x4 mesh")

    tp = tuple(mesh_device.shape)[tp_axis]
    sp = tuple(mesh_device.shape)[sp_axis]
    affine = case == "qk"
    num_heads_per_device = (NUM_HEADS // tp) if case == "qk" else 1
    eps = 1e-6
    B = 1

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    # Build the norm exactly as LTX does (mesh_axis = TP axis; the reduction-dim hidden is TP-sharded).
    norm = DistributedRMSNorm(
        embedding_dim=dim,
        norm_eps=eps,
        norm_elementwise_affine=affine,
        bias=False,
        mesh_axis=tp_axis,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
    )

    torch.manual_seed(1234)
    if affine:
        # Random weight (gamma), loaded the same way LTX loads norm_q/norm_k weights.
        torch_weight = (torch.rand(dim) * 2 - 1).to(torch.bfloat16)
        norm.load_torch_state_dict({"weight": torch_weight})

    # Activation: (1, B, N, dim), sequence sharded over the SP axis, hidden over the TP axis (= the norm's
    # mesh_axis). Each device holds (1, B, N/sp, dim/tp), matching the norm's expected last dim dim/tp.
    x = (torch.randn(1, B, seq_len, dim) * 4 - 1).to(torch.bfloat16)
    x_tt = bf16_tensor_2dshard(x, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})

    def run(fused: bool):
        prev = os.environ.get("LTX_FUSED_AGRMS")
        os.environ["LTX_FUSED_AGRMS"] = "1" if fused else "0"
        try:
            return norm(x_tt, num_heads_per_device=num_heads_per_device)
        finally:
            if prev is None:
                os.environ.pop("LTX_FUSED_AGRMS", None)
            else:
                os.environ["LTX_FUSED_AGRMS"] = prev

    out_ref = run(fused=False)  # wan pre/AG/post
    out_fused = run(fused=True)  # ttnn.all_gather_rms_norm

    # Output sharding: block -> (1,B,N,dim) [seq/SP, hidden/TP]; qk head-split -> (1,H,M,E) [heads/TP, seq/SP].
    if case == "qk":
        concat_dims = [0, 0]
        concat_dims[sp_axis] = 2  # seq (M)
        concat_dims[tp_axis] = 1  # heads
    else:
        concat_dims = [0, 0]
        concat_dims[sp_axis] = 2  # seq
        concat_dims[tp_axis] = 3  # hidden
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(concat_dims))

    ref = ttnn.to_torch(out_ref, mesh_composer=composer).float()
    fused = ttnn.to_torch(out_fused, mesh_composer=composer).float()

    assert list(ref.shape) == list(fused.shape), f"shape mismatch: {tuple(ref.shape)} vs {tuple(fused.shape)}"
    max_abs = (ref - fused).abs().max().item()
    logger.info(
        f"distributed_rmsnorm parity[{case} dim={dim} tp={tp} sp={sp} H/dev={num_heads_per_device} "
        f"shape={tuple(fused.shape)}] max_abs_err={max_abs:.4f}"
    )
    # Both paths implement the same RMSNorm (+ gamma, + head-split); differences are only kernel numerics.
    assert_with_pcc(ref, fused, pcc=0.999)
