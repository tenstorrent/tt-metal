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
window mask. With is_causal=True the mask is also lower-triangular inside each
window (packed variable-length causal sequences, issue #57920).
"""

import torch
import pytest
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def windowed_mask(seq_len, cu_window_seqlens, is_causal=False):
    """Block-diagonal mask: token i attends only to tokens in the same window (and, if causal, only to
    tokens at or before i)."""
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float32)
    for i in range(1, len(cu_window_seqlens)):
        start, end = cu_window_seqlens[i - 1], cu_window_seqlens[i]
        mask[start:end, start:end] = 0.0
    if is_causal:
        mask = mask + torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask


def to_cu_tensor(cu_window_seqlens, device, **kwargs):
    return ttnn.from_torch(
        torch.tensor(cu_window_seqlens, dtype=torch.int32),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        **kwargs,
    )


def sdpa_configs(device, q_chunk, k_chunk, fp32_dest_acc_en=True):
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        exp_approx_mode=False,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
    )
    return program_config, compute_kernel_config


def reference_sdpa(q, k, v, mask, scale):
    """torch SDPA in fp32, with GQA K/V heads repeated up to the Q head count."""
    rep = q.shape[1] // k.shape[1]
    k, v = (t.to(torch.float32).repeat_interleave(rep, dim=1) for t in (k, v))
    return torch.nn.functional.scaled_dot_product_attention(
        q.to(torch.float32), k, v, attn_mask=mask.unsqueeze(0).unsqueeze(0), scale=scale
    )


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
@pytest.mark.parametrize("is_causal", [False, True], ids=["bidir", "causal"])
def test_windowed_sdpa_smoke(
    device, dtype, pcc_threshold, num_heads, seq_len, chunk, cu_window_seqlens, fp32_dest_acc_en, is_causal
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
        is_causal=is_causal,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        cu_window_seqlens=cu_tt,
    )
    out = ttnn.to_torch(out_tt).to(torch.float32)

    mask = windowed_mask(seq_len, cu_window_seqlens, is_causal).unsqueeze(0).unsqueeze(0)
    gt = torch.nn.functional.scaled_dot_product_attention(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), attn_mask=mask, scale=scale
    )

    passing, pcc = comp_pcc(gt, out, pcc_threshold)
    logger.info(
        f"windowed SDPA causal={is_causal} dtype={dtype} s={seq_len} heads={num_heads} "
        f"windows={cu_window_seqlens} pcc={pcc}"
    )
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
@pytest.mark.parametrize("is_causal", [False, True], ids=["bidir", "causal"])
def test_windowed_sdpa_q_token_offset(
    device, seq_len, chunk, cu_window_seqlens, num_shards, num_heads, offset_as_tensor, is_causal
):
    """Each Q shard attends over the full K/V with GLOBAL window boundaries.

    This is the sequence-parallel shape: Q holds `seq_len // num_shards` contiguous rows and is indexed
    locally, while K/V and `cu_window_seqlens` stay global. `windowed_q_token_offset` tells the on-device
    mask generator where the shard starts, so a row's window is decided by its global position.

    Concatenating the shards must reproduce the unsharded result exactly -- attention is row-independent
    given the mask, so splitting Q changes no arithmetic. Compared against the same torch reference the
    unsharded test uses. In causal mode the diagonal is at the GLOBAL row, so a shard with Sq < Sk is
    still well defined.
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
            is_causal=is_causal,
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

    mask = windowed_mask(seq_len, cu_window_seqlens, is_causal).unsqueeze(0).unsqueeze(0)
    gt = torch.nn.functional.scaled_dot_product_attention(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), attn_mask=mask, scale=scale
    )

    passing, pcc = comp_pcc(gt, out, 0.99)
    logger.info(
        f"windowed SDPA q-offset causal={is_causal} s={seq_len} shards={num_shards} heads={num_heads} "
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
@pytest.mark.parametrize("is_causal", [False, True], ids=["bidir", "causal"])
def test_windowed_sdpa_q_offset_tensor_on_mesh(mesh_device, seq_len, chunk, cu_window_seqlens, num_heads, is_causal):
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
        is_causal=is_causal,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        cu_window_seqlens=cu_tt,
        windowed_q_token_offset_tensor=offsets_tt,
    )
    out = ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2)).to(torch.float32)

    mask = windowed_mask(seq_len, cu_window_seqlens, is_causal).unsqueeze(0).unsqueeze(0)
    gt = torch.nn.functional.scaled_dot_product_attention(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), attn_mask=mask, scale=scale
    )

    passing, pcc = comp_pcc(gt, out, 0.99)
    logger.info(
        f"windowed SDPA mesh q-offset causal={is_causal} s={seq_len} devices={num_shards} heads={num_heads} "
        f"windows={cu_window_seqlens} pcc={pcc}"
    )
    assert passing, f"PCC below threshold: {pcc}"


@pytest.mark.parametrize(
    "windows",
    [[512, 512], [300, 212, 512], [128, 896], [1024]],
    ids=["w512x2", "w300_212_512", "w128_896", "w1024"],
)
@pytest.mark.parametrize(
    "q_chunk, k_chunk",
    [(None, None), (128, 128), (64, 256), (256, 64)],
    ids=["default_chunks", "q128_k128", "q64_k256", "q256_k64"],
)
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32acc", "no_fp32acc"])
def test_windowed_causal_sdpa_gqa(device, windows, q_chunk, k_chunk, fp32_dest_acc_en):
    """Issue #57920: packed variable-length causal sequences (Qwen3-Embedding-4B: 32 Q / 8 KV heads,
    d=128, causal, last-token pooling). Window lengths that are not tile multiples put both window
    boundaries and the diagonal through the middle of tiles.

    Besides the global PCC, each window's LAST row -- the one last-token pooling reads -- is checked
    on its own, so a defect confined to one window's tail cannot hide behind the other rows.
    """
    torch.manual_seed(0)
    b, nqh, nkv, dh = 1, 32, 8, 128
    seq_len = sum(windows)
    cu_window_seqlens = [0] + torch.tensor(windows).cumsum(0).tolist()
    scale = dh**-0.5

    q = torch.randn(b, nqh, seq_len, dh, dtype=torch.bfloat16)
    k = torch.randn(b, nkv, seq_len, dh, dtype=torch.bfloat16)
    v = torch.randn(b, nkv, seq_len, dh, dtype=torch.bfloat16)

    kwargs = {}
    if q_chunk is not None:
        kwargs["program_config"], kwargs["compute_kernel_config"] = sdpa_configs(
            device, q_chunk, k_chunk, fp32_dest_acc_en
        )
    elif not fp32_dest_acc_en:
        pytest.skip("default chunks run with the default compute config only")

    tt = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_tt = ttnn.transformer.scaled_dot_product_attention(
        tt(q),
        tt(k),
        tt(v),
        is_causal=True,
        scale=scale,
        cu_window_seqlens=to_cu_tensor(cu_window_seqlens, device),
        **kwargs,
    )
    out = ttnn.to_torch(out_tt).to(torch.float32)
    gt = reference_sdpa(q, k, v, windowed_mask(seq_len, cu_window_seqlens, is_causal=True), scale)
    assert out.shape == gt.shape, f"shape mismatch: {out.shape} vs {gt.shape}"

    passing, pcc = comp_pcc(gt, out, 0.99)
    logger.info(f"windowed causal SDPA windows={windows} chunks=({q_chunk},{k_chunk}) pcc={pcc}")
    assert passing, f"PCC below threshold: {pcc}"

    last_rows = [end - 1 for end in cu_window_seqlens[1:]]
    passing, pcc = comp_pcc(gt[:, :, last_rows, :], out[:, :, last_rows, :], 0.99)
    assert passing, f"last-token rows PCC below threshold: {pcc}"


@pytest.mark.parametrize(
    "seq_len, q_chunk, k_chunk, cu_window_seqlens",
    [
        # Windows shorter than a tile, including a 1-token window.
        (256, 64, 64, [0, 5, 32, 33, 60, 256]),
        # Empty windows (repeated boundaries) between real ones.
        (256, 64, 128, [0, 100, 100, 200, 200, 256]),
        # Many tiny windows in one chunk: the diagonal and many window edges share tiles.
        (128, 128, 32, [0, 7, 19, 31, 45, 64, 90, 101, 128]),
        # Unpadded length not a tile multiple, and a window that ends inside the last partial tile.
        (1000, 128, 128, [0, 333, 666, 1000]),
        # Long windows spanning several chunks on both axes, so most Q chunks skip K chunks on both sides.
        (2048, 128, 128, [0, 700, 1500, 2048]),
    ],
    ids=["sub_tile", "empty_windows", "dense_edges", "s1000_partial", "s2048_long"],
)
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32acc", "no_fp32acc"])
# Bidirectional runs the same layouts so a failure can be pinned on the causal overlay or not.
@pytest.mark.parametrize("is_causal", [False, True], ids=["bidir", "causal"])
def test_windowed_sdpa_edges(device, seq_len, q_chunk, k_chunk, cu_window_seqlens, fp32_dest_acc_en, is_causal):
    torch.manual_seed(1234)
    b, nh, dh = 1, 4, 64
    scale = dh**-0.5
    q, k, v = (torch.randn(b, nh, seq_len, dh, dtype=torch.bfloat16) for _ in range(3))
    program_config, compute_kernel_config = sdpa_configs(device, q_chunk, k_chunk, fp32_dest_acc_en)

    tt = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_tt = ttnn.transformer.scaled_dot_product_attention(
        tt(q),
        tt(k),
        tt(v),
        is_causal=is_causal,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        cu_window_seqlens=to_cu_tensor(cu_window_seqlens, device),
    )
    out = ttnn.to_torch(out_tt).to(torch.float32)
    gt = reference_sdpa(q, k, v, windowed_mask(seq_len, cu_window_seqlens, is_causal), scale)

    passing, pcc = comp_pcc(gt, out, 0.99)
    logger.info(f"windowed SDPA edges causal={is_causal} s={seq_len} windows={cu_window_seqlens} pcc={pcc}")
    assert passing, f"PCC below threshold: {pcc}"
    # Every row attends at least to itself, so no row may be all-masked: the output must be finite.
    assert torch.isfinite(out).all(), "non-finite output rows"


@pytest.mark.parametrize("is_causal", [False, True], ids=["bidir", "causal"])
def test_windowed_sdpa_output_concat_heads(device, is_causal):
    torch.manual_seed(7)
    b, nqh, nkv, seq_len, dh = 1, 8, 2, 512, 128
    cu_window_seqlens = [0, 150, 300, 512]
    scale = dh**-0.5
    q = torch.randn(b, nqh, seq_len, dh, dtype=torch.bfloat16)
    k = torch.randn(b, nkv, seq_len, dh, dtype=torch.bfloat16)
    v = torch.randn(b, nkv, seq_len, dh, dtype=torch.bfloat16)
    program_config, compute_kernel_config = sdpa_configs(device, 128, 128, fp32_dest_acc_en=False)

    tt = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_tt = ttnn.transformer.scaled_dot_product_attention(
        tt(q),
        tt(k),
        tt(v),
        is_causal=is_causal,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        cu_window_seqlens=to_cu_tensor(cu_window_seqlens, device),
        output_concat_heads=True,
    )
    out = ttnn.to_torch(out_tt).to(torch.float32)
    gt = reference_sdpa(q, k, v, windowed_mask(seq_len, cu_window_seqlens, is_causal), scale)
    gt = gt.permute(0, 2, 1, 3).reshape(b, 1, seq_len, nqh * dh)
    assert out.shape == gt.shape, f"shape mismatch: {out.shape} vs {gt.shape}"

    passing, pcc = comp_pcc(gt, out, 0.99)
    logger.info(f"windowed SDPA concat-heads causal={is_causal} pcc={pcc}")
    assert passing, f"PCC below threshold: {pcc}"
