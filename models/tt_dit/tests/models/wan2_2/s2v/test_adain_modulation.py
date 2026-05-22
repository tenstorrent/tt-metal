# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""S2V AdaIN modulation parity (ColParallel chunk-on-shard)."""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from .....layers.linear import prepare_chunked_linear_output
from .....models.transformers.wan2_2.s2v.audio_utils import AdaLayerNormZero
from .....parallel.manager import CCLManager
from .....utils.check import assert_quality
from .....utils.tensor import bf16_tensor, local_device_to_torch
from .....utils.test import line_params, ring_params


@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology"),
    [
        # 2x4 BH-LB (tp_factor=2): chunk-on-shard with 2 chips. Less interesting
        # than 4x8 but verifies the math works at the smaller TP factor too.
        pytest.param((2, 4), (2, 4), 1, 0, 2, line_params, ttnn.Topology.Linear, id="bh_2x4sp1tp0"),
        # 4x8 BH-GLX (tp_factor=4): the production configuration; tightest test.
        pytest.param((4, 8), (4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, id="bh_4x8sp1tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_adain_projection_s2v(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
) -> None:
    """ColParallelLinear chunk + E-matmul expansion parity."""
    torch.manual_seed(0)
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    adain_dim = 1024
    dim = 5120  # 14B model dim
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    T_video = 16  # canonical S2V

    # ---- Build torch reference ----
    audio_emb_torch = torch.randn(1, 1, T_video, adain_dim, dtype=torch.float32)
    W_torch = torch.randn(2 * dim, adain_dim, dtype=torch.float32) * 0.02
    b_torch = torch.randn(2 * dim, dtype=torch.float32) * 0.02

    silu = torch.nn.functional.silu(audio_emb_torch)
    proj_ref = silu @ W_torch.transpose(0, 1) + b_torch
    shift_ref, scale_ref = proj_ref.chunk(2, dim=-1)

    # ---- Build AdaLayerNormZero on device, load via state-dict path ----
    adain = AdaLayerNormZero(
        dim=dim,
        adain_dim=adain_dim,
        mesh_device=mesh_device,
        tp_mesh_axis=tp_axis,
    )
    adain.load_torch_state_dict({"linear.weight": W_torch.clone(), "linear.bias": b_torch.clone()})

    # ---- Run on-device projection ----
    audio_emb_dev = bf16_tensor(audio_emb_torch, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    proj_dev = adain.linear(ttnn.silu(audio_emb_dev))
    shift_pf_dev, scale_pf_dev = ttnn.chunk(proj_dev, 2, dim=-1)

    ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    shift_gather = ccl.all_gather_persistent_buffer(shift_pf_dev, dim=-1, mesh_axis=tp_axis)
    scale_gather = ccl.all_gather_persistent_buffer(scale_pf_dev, dim=-1, mesh_axis=tp_axis)
    shift_tt = local_device_to_torch(shift_gather).float()
    scale_tt = local_device_to_torch(scale_gather).float()
    logger.info(f"shift gathered shape: {tuple(shift_tt.shape)}")

    assert_quality(shift_tt, shift_ref, pcc=0.99)
    assert_quality(scale_tt, scale_ref, pcc=0.99)

    # ---- Exercise the full E @ extended pipeline ----
    # Per-token modulation: each spatial token of frame t picks up row t's
    # (shift, scale) projection; const/pad tokens pick up identity (0, 1).
    noisy_len = T_video * 64  # 16 frames × 64 hw tokens/frame
    padded_N = noisy_len
    hw_per_frame = noisy_len // T_video

    E_torch = torch.zeros(padded_N, T_video + 1, dtype=torch.float32)
    for i in range(noisy_len):
        E_torch[i, i // hw_per_frame] = 1.0
    E_tt = bf16_tensor(
        E_torch.unsqueeze(0).unsqueeze(0).contiguous(),
        device=mesh_device,
        mesh_axis=sp_axis,
        shard_dim=2,
        layout=ttnn.TILE_LAYOUT,
    )
    zero_row_tt = bf16_tensor(
        torch.zeros(1, 1, 1, dim, dtype=torch.float32),
        device=mesh_device,
        mesh_axis=tp_axis,
        shard_dim=3,
        layout=ttnn.TILE_LAYOUT,
    )
    one_row_tt = bf16_tensor(
        torch.ones(1, 1, 1, dim, dtype=torch.float32),
        device=mesh_device,
        mesh_axis=tp_axis,
        shard_dim=3,
        layout=ttnn.TILE_LAYOUT,
    )
    scale_pf_p1_dev = ttnn.add(scale_pf_dev, 1.0)
    shift_ext = ttnn.concat([shift_pf_dev, zero_row_tt], dim=-2)
    scale_ext = ttnn.concat([scale_pf_p1_dev, one_row_tt], dim=-2)
    shift_full_dev = ttnn.matmul(E_tt, shift_ext)
    scale_full_dev = ttnn.matmul(E_tt, scale_ext)

    shift_full_tp = ccl.all_gather_persistent_buffer(shift_full_dev, dim=-1, mesh_axis=tp_axis)
    shift_full_full = ccl.all_gather_persistent_buffer(shift_full_tp, dim=-2, mesh_axis=sp_axis)
    scale_full_tp = ccl.all_gather_persistent_buffer(scale_full_dev, dim=-1, mesh_axis=tp_axis)
    scale_full_full = ccl.all_gather_persistent_buffer(scale_full_tp, dim=-2, mesh_axis=sp_axis)
    shift_full_host = local_device_to_torch(shift_full_full).float()
    scale_full_host = local_device_to_torch(scale_full_full).float()

    shift_full_ref = shift_ref.squeeze(0).squeeze(0).repeat_interleave(hw_per_frame, dim=0)
    scale_full_ref = (scale_ref + 1.0).squeeze(0).squeeze(0).repeat_interleave(hw_per_frame, dim=0)
    shift_full_ref = shift_full_ref.unsqueeze(0).unsqueeze(0)
    scale_full_ref = scale_full_ref.unsqueeze(0).unsqueeze(0)

    logger.info(f"shift_full device shape: {tuple(shift_full_host.shape)} (tp_factor={tp_factor})")
    assert_quality(shift_full_host, shift_full_ref, pcc=0.99)
    assert_quality(scale_full_host, scale_full_ref, pcc=0.99)


def test_prepare_chunked_linear_output_s2v() -> None:
    """CPU check of the chunk-on-shard permutation rule."""
    torch.manual_seed(0)
    dim = 5120
    in_features = 1024
    tp_factor = 4
    chunks = 2
    out_features = chunks * dim  # 10240

    W = torch.randn(out_features, in_features, dtype=torch.float32) * 0.02
    b = torch.randn(out_features, dtype=torch.float32) * 0.02
    state = {"linear.weight": W.clone(), "linear.bias": b.clone()}
    prepare_chunked_linear_output(state, prefix="linear", device_count=tp_factor, chunks=chunks)
    W_p = state["linear.weight"]
    b_p = state["linear.bias"]

    x = torch.randn(2, in_features, dtype=torch.float32)
    ref_proj = x @ W.transpose(0, 1) + b
    ref_shift, ref_scale = ref_proj.chunk(2, dim=-1)

    # Simulate ColParallel: transpose then shard last axis.
    W_post = W_p.transpose(0, 1)
    proj = x @ W_post + b_p
    out_per_chip = out_features // tp_factor
    dim_per_chip = dim // tp_factor

    shift_assembled = torch.empty(*proj.shape[:-1], dim)
    scale_assembled = torch.empty(*proj.shape[:-1], dim)
    for chip in range(tp_factor):
        local = proj[..., chip * out_per_chip : (chip + 1) * out_per_chip]
        ls, lsc = torch.chunk(local, 2, dim=-1)
        shift_assembled[..., chip * dim_per_chip : (chip + 1) * dim_per_chip] = ls
        scale_assembled[..., chip * dim_per_chip : (chip + 1) * dim_per_chip] = lsc

    err_s = (shift_assembled - ref_shift).abs().max().item()
    err_sc = (scale_assembled - ref_scale).abs().max().item()
    logger.info(f"shift max abs err: {err_s:.6e}")
    logger.info(f"scale max abs err: {err_sc:.6e}")
    assert err_s < 1e-4, f"shift max err {err_s} too large"
    assert err_sc < 1e-4, f"scale max err {err_sc} too large"
