# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Op-level smoke test for windowed (block-diagonal) attention via
ttnn.transformer.scaled_dot_product_attention(..., cu_window_seqlens=...).

This intentionally lives next to the other SDPA op tests (instead of only under
models/demos/qwen25_vl/) so that any change to the shared SDPA kernel helpers
(e.g. dataflow_common.hpp / write_block) exercises the windowed writer kernel in
a pre-merge / per-commit gate. Device kernels are JIT-built at op invocation, so
running the op on hardware is the only way to catch a kernel-signature break -
which is exactly what slipped through in #45015 and broke Qwen2.5-VL nightly.

Correctness is checked against torch SDPA with the equivalent block-diagonal
window mask.
"""

import os
import torch
import pytest
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def windowed_mask(seq_len, cu_window_seqlens):
    """Block-diagonal mask: token i attends only to tokens in the same window."""
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float32)
    for i in range(1, len(cu_window_seqlens)):
        start, end = cu_window_seqlens[i - 1], cu_window_seqlens[i]
        mask[start:end, start:end] = 0.0
    return mask


@pytest.mark.parametrize(
    "seq_len, chunk, cu_window_seqlens",
    [
        (128, 32, [0, 64, 128]),  # two equal tile-aligned windows
        (128, 32, [0, 32, 96, 128]),  # three uneven windows
        (256, 32, [0, 64, 128, 256]),  # larger sequence
        (96, 64, [0, 33, 64, 96]),  # sequence padded to chunk size; windowed mask owns padding
        (129, 64, [0, 32, 97, 129]),  # partial final tile plus chunk padding
        # Aggressive K-range narrowing: each Q chunk's windows cover a small fraction of the 8
        # K chunks, so a wrong [k_lo, k_hi) (missing or extra keys) craters PCC rather than hiding
        # behind a nearly-dense range.
        (1024, 128, [0, 128, 256, 384, 512, 640, 768, 896, 1024]),  # 8 chunk-aligned windows
        (1024, 128, [0, 200, 480, 730, 1024]),  # uneven windows straddling chunk boundaries
    ],
    ids=[
        "s128_w2",
        "s128_w3",
        "s256_w3",
        "s96_padded_chunk",
        "s129_partial_tile",
        "s1024_w8_aligned",
        "s1024_w4_straddle",
    ],
)
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize(
    "dtype, pcc_threshold",
    [
        (ttnn.bfloat16, 0.99),
        # bfloat8_b mirrors the dtype Qwen2.5-VL actually feeds the op
        # (vision_attention.py typecasts q/k/v to bf8 before the call); looser
        # PCC accounts for the reduced input precision.
        (ttnn.bfloat8_b, 0.98),
    ],
    ids=["bf16", "bf8"],
)
# Both dest-accumulation modes are covered: fp32_dest_acc_en selects different compute paths
# (False -> streaming on Blackhole, True -> standard), so both must stay correct.
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32acc", "no_fp32acc"])
def test_windowed_sdpa_smoke(
    device, dtype, pcc_threshold, num_heads, seq_len, chunk, cu_window_seqlens, fp32_dest_acc_en
):
    torch.manual_seed(42)
    b, dh = 1, 128
    scale = dh**-0.5

    q = torch.randn(b, num_heads, seq_len, dh, dtype=torch.bfloat16)
    k = torch.randn(b, num_heads, seq_len, dh, dtype=torch.bfloat16)
    v = torch.randn(b, num_heads, seq_len, dh, dtype=torch.bfloat16)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        exp_approx_mode=False,
        q_chunk_size=chunk,
        k_chunk_size=chunk,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
    )

    q_tt = ttnn.from_torch(q, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    k_tt = ttnn.from_torch(k, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    v_tt = ttnn.from_torch(v, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    cu_tt = ttnn.from_torch(
        torch.tensor(cu_window_seqlens, dtype=torch.int32),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )

    out_tt = ttnn.transformer.scaled_dot_product_attention(
        q_tt,
        k_tt,
        v_tt,
        is_causal=False,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        cu_window_seqlens=cu_tt,
    )
    out = ttnn.to_torch(out_tt).to(torch.float32)

    mask = windowed_mask(seq_len, cu_window_seqlens).unsqueeze(0).unsqueeze(0)
    gt = torch.nn.functional.scaled_dot_product_attention(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), attn_mask=mask, scale=scale
    )

    passing, pcc = comp_pcc(gt, out, pcc_threshold)
    logger.info(f"windowed SDPA dtype={dtype} s={seq_len} heads={num_heads} windows={cu_window_seqlens} pcc={pcc}")
    assert passing, f"PCC below threshold: {pcc}"
    assert out.shape == gt.shape, f"shape mismatch: {out.shape} vs {gt.shape}"


@pytest.mark.parametrize(
    "seq_len, chunk, cu_window_seqlens, num_shards",
    [
        # Windows aligned to shard boundaries: every shard's rows sit inside one window.
        (256, 32, [0, 64, 128, 192, 256], 4),
        # Windows that straddle shard boundaries: shard 1 (64..127) spans windows [0,96) and [96,160).
        # This is the case a per-block SDPA loop cannot express and the offset exists for.
        (256, 32, [0, 96, 160, 256], 4),
        # Uneven windows, 2 shards, and a window shorter than the chunk.
        (128, 32, [0, 32, 96, 128], 2),
    ],
    ids=["aligned_4shard", "straddling_4shard", "uneven_2shard"],
)
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("offset_as_tensor", [False, True], ids=["scalar", "tensor"])
def test_windowed_sdpa_q_token_offset(
    device, seq_len, chunk, cu_window_seqlens, num_shards, num_heads, offset_as_tensor
):
    """Each Q shard attends over the full K/V with GLOBAL window boundaries.

    This is the sequence-parallel shape: Q holds `seq_len // num_shards` contiguous rows and is indexed
    locally, while K/V and `cu_window_seqlens` stay global. `windowed_q_token_offset` tells the on-device
    mask generator where the shard starts, so a row's window is decided by its global position.

    Concatenating the shards must reproduce the unsharded result exactly -- attention is row-independent
    given the mask, so splitting Q changes no arithmetic. Compared against the same torch reference the
    unsharded test uses.
    """
    torch.manual_seed(42)
    b, dh = 1, 128
    scale = dh**-0.5
    shard_rows = seq_len // num_shards
    assert shard_rows % 32 == 0, "offset must be tile-aligned"

    q = torch.randn(b, num_heads, seq_len, dh, dtype=torch.bfloat16)
    k = torch.randn(b, num_heads, seq_len, dh, dtype=torch.bfloat16)
    v = torch.randn(b, num_heads, seq_len, dh, dtype=torch.bfloat16)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        exp_approx_mode=False,
        q_chunk_size=chunk,
        k_chunk_size=chunk,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    # K/V are global and shared by every shard; only Q is sliced.
    k_tt = ttnn.from_torch(k, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    v_tt = ttnn.from_torch(v, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    cu_tt = ttnn.from_torch(
        torch.tensor(cu_window_seqlens, dtype=torch.int32),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )

    shards = []
    for shard in range(num_shards):
        offset = shard * shard_rows
        q_shard = q[:, :, offset : offset + shard_rows, :].contiguous()
        out_tt = ttnn.transformer.scaled_dot_product_attention(
            ttnn.from_torch(q_shard, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
            k_tt,
            v_tt,
            is_causal=False,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            cu_window_seqlens=cu_tt,
            # Two ways to supply the same value. The scalar is baked into the program; the tensor is read
            # on device at dispatch, which is what lets one shared program serve differently-offset
            # devices when it is sharded on the sequence-parallel axis. They must agree exactly.
            windowed_q_token_offset=0 if offset_as_tensor else offset,
            windowed_q_token_offset_tensor=(
                ttnn.from_torch(
                    torch.tensor([offset], dtype=torch.int32),
                    device=device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.uint32,
                )
                if offset_as_tensor
                else None
            ),
        )
        got = ttnn.to_torch(out_tt).to(torch.float32)
        assert got.shape[-2] == shard_rows, f"shard {shard} returned {got.shape[-2]} rows, expected {shard_rows}"
        shards.append(got)

    out = torch.cat(shards, dim=-2)

    mask = windowed_mask(seq_len, cu_window_seqlens).unsqueeze(0).unsqueeze(0)
    gt = torch.nn.functional.scaled_dot_product_attention(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), attn_mask=mask, scale=scale
    )

    passing, pcc = comp_pcc(gt, out, 0.99)
    logger.info(
        f"windowed SDPA q-offset s={seq_len} shards={num_shards} heads={num_heads} "
        f"windows={cu_window_seqlens} pcc={pcc}"
    )
    assert passing, f"PCC below threshold: {pcc}"


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "seq_len, chunk, cu_window_seqlens",
    [
        # Windows straddle the shard cuts (shards are 64 rows), so every device resolves a different
        # window set from a different offset -- the strongest test of per-coordinate extraction.
        (256, 32, [0, 96, 160, 256]),
    ],
    ids=["straddling"],
)
@pytest.mark.parametrize("num_heads", [8])
def test_windowed_sdpa_q_offset_tensor_on_mesh(mesh_device, seq_len, chunk, cu_window_seqlens, num_heads):
    """The offset tensor's actual use case: ONE SDPA call over a mesh, Q sharded on the sequence.

    The serial test above proves each offset value is honored; this proves the per-device plumbing.
    Every device runs the SAME cached program, so the offsets must diverge through data: Q is sharded
    on dim 2 across the mesh, K/V and cu_window_seqlens are replicated, and the 1-element offset
    tensor is sharded so device d's local value is d * shard_rows. If per-coordinate extraction or
    accessor binding broke (e.g. every device reading device 0's offset), devices 1..3 would mask
    against the wrong windows and the composed PCC craters.

    Skips (via the mesh_device fixture) on machines with fewer than 4 devices.
    """
    torch.manual_seed(42)
    b, dh = 1, 128
    scale = dh**-0.5
    num_shards = mesh_device.get_num_devices()
    shard_rows = seq_len // num_shards
    assert shard_rows % 32 == 0, "offset must be tile-aligned"

    q = torch.randn(b, num_heads, seq_len, dh, dtype=torch.bfloat16)
    k = torch.randn(b, num_heads, seq_len, dh, dtype=torch.bfloat16)
    v = torch.randn(b, num_heads, seq_len, dh, dtype=torch.bfloat16)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
        exp_approx_mode=False,
        q_chunk_size=chunk,
        k_chunk_size=chunk,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    q_tt = ttnn.from_torch(
        q,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )
    k_tt = ttnn.from_torch(k, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate)
    v_tt = ttnn.from_torch(v, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate)
    cu_tt = ttnn.from_torch(
        torch.tensor(cu_window_seqlens, dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=replicate,
    )
    offsets_tt = ttnn.from_torch(
        torch.arange(num_shards, dtype=torch.int32) * shard_rows,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    out_tt = ttnn.transformer.scaled_dot_product_attention(
        q_tt,
        k_tt,
        v_tt,
        is_causal=False,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        cu_window_seqlens=cu_tt,
        windowed_q_token_offset_tensor=offsets_tt,
    )
    out = ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2)).to(torch.float32)

    mask = windowed_mask(seq_len, cu_window_seqlens).unsqueeze(0).unsqueeze(0)
    gt = torch.nn.functional.scaled_dot_product_attention(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), attn_mask=mask, scale=scale
    )

    passing, pcc = comp_pcc(gt, out, 0.99)
    logger.info(
        f"windowed SDPA mesh q-offset s={seq_len} devices={num_shards} heads={num_heads} "
        f"windows={cu_window_seqlens} pcc={pcc}"
    )
    assert passing, f"PCC below threshold: {pcc}"
