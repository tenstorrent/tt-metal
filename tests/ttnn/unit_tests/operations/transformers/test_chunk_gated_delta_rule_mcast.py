# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness gate for the GDN scan shared-input multicast.

The scan phase of ttnn.transformer.chunk_gated_delta_rule multicasts its six shared V-independent
inputs (kd, q_decay, intra, k_dec_t, dl, t_inv) from one sender core per head into the sibling
V-block cores' CBs instead of every sibling re-reading identical DRAM pages. The multicast forwards
the exact bytes the sender read into the same CB indices, so the op's outputs must be BIT-IDENTICAL
with the multicast on and off — any difference is a bug, not numerical noise.

QWEN_GDN_SCAN_MCAST is read where ChunkGdnScanParams is built and stored as a hashed attribute
(`use_mcast`), so toggling it between calls in one process compiles two distinct cached scan
programs — the test asserts this via the program-cache entry count, which also makes the A/B
non-vacuous (on a build that stops reading or hashing the env, both runs would use one program
and the cache assertion fails).
"""

import pytest
import torch

import ttnn

CHUNK = 32  # the fused op's supported chunk size (Ct=1)


def _scan_nv(device, bh, vt):
    """Replicates distribute_scan's row-aligned NV selection (largest divisor of vt whose 1xNV
    head rectangles fit the padded grid)."""
    grid = device.compute_with_storage_grid_size()
    for cand in range(vt, 0, -1):
        if vt % cand != 0 or cand > grid.x:
            continue
        if bh <= (grid.x // cand) * grid.y:
            return cand
    return 1


def _const_tiles(device, chunk_size=CHUNK):
    """The fused op's constant tiles (mirrors qwen36 fused_chunk.build_fused_const_tiles)."""
    c = chunk_size
    eye = torch.eye(c, dtype=torch.float32)
    tril = torch.tril(torch.ones(c, c, dtype=torch.float32))
    ones = torch.ones(c, c, dtype=torch.float32)
    ii = torch.arange(32).unsqueeze(1)
    jj = torch.arange(32).unsqueeze(0)
    lo_i, lo_j = ii < 16, jj < 16
    qtl = (lo_i & lo_j).float()
    qbr = (~lo_i & ~lo_j).float()
    qbl = (~lo_i & lo_j).float()
    masks = torch.cat([qtl, qbr, qbl], dim=1)  # [32, 96]

    def _up(t):
        return ttnn.from_torch(t.reshape(1, 1, *t.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    return (_up(eye), _up(tril), _up(ones), _up(masks))


def _run_op(device, tensors, const_tiles, initial_state):
    q, k, v, g, beta = tensors
    eye, tril, ones, masks = const_tiles
    o, fs = ttnn.transformer.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=CHUNK,
        eye=eye,
        tril=tril,
        ones=ones,
        masks=masks,
    )
    o_t = ttnn.to_torch(o)
    fs_t = ttnn.to_torch(fs)
    ttnn.deallocate(o)
    ttnn.deallocate(fs)
    return o_t, fs_t


@pytest.mark.parametrize(
    "batch, num_k_heads, num_v_heads, want_mcast",
    [
        (1, 4, 12, True),  # TP-4-like per-device shape: BH=12 -> NV=4 on a 13x10 grid, fan-out 3
        (1, 16, 48, True),  # single-device Qwen3.6 shape: BH=48 -> NV=2, fan-out 1
        (2, 16, 48, False),  # batched prefill: BH=96 -> NV=1, multicast degenerates to plain reader
    ],
)
@pytest.mark.parametrize("with_initial_state", [False, True])
def test_scan_mcast_bit_exact(device, monkeypatch, batch, num_k_heads, num_v_heads, want_mcast, with_initial_state):
    torch.manual_seed(20260819)
    B, T, Dk, Dv = batch, 256, 128, 128
    BH = B * num_v_heads

    grid = device.compute_with_storage_grid_size()
    if BH > grid.x * grid.y:
        pytest.skip(f"BH={BH} exceeds the {grid.x}x{grid.y} compute grid (scan needs a core per head)")
    nv = _scan_nv(device, BH, Dv // 32)
    if want_mcast and nv == 1:
        pytest.skip(f"grid {grid.x}x{grid.y} gives NV=1 for BH={BH}: multicast path not exercised")

    # Neutralize ambient GDN debug/profiling knobs that would bypass or fork the scan path.
    monkeypatch.setenv("QWEN_GDN_PHASED", "1")
    monkeypatch.delenv("QWEN_GDN_SCAN_SERIAL", raising=False)
    # QWEN_GDN_DUMP is read once via a function-local static; delenv helps only if the op has not
    # run yet in this process — kept for hygiene.
    monkeypatch.delenv("QWEN_GDN_DUMP", raising=False)

    # Realistic-shaped inputs; bit-exactness holds for any values, but keep them in the op's
    # numeric regime (L2-normalized keys upstream, beta in (0,1), g <= 0).
    q = torch.randn(B, T, num_k_heads, Dk, dtype=torch.bfloat16)
    k = torch.randn(B, T, num_k_heads, Dk, dtype=torch.bfloat16)
    v = torch.randn(B, T, num_v_heads, Dv, dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn(B, T, num_v_heads, dtype=torch.float32))
    g = -torch.nn.functional.softplus(torch.randn(B, T, num_v_heads, dtype=torch.float32)) * 0.5

    def dev(t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    tensors = (
        dev(q, ttnn.bfloat16),
        dev(k, ttnn.bfloat16),
        dev(v, ttnn.bfloat16),
        dev(g, ttnn.float32),
        dev(beta, ttnn.float32),
    )
    s0 = None
    if with_initial_state:
        s0_t = 0.1 * torch.randn(B, num_v_heads, Dk, Dv, dtype=torch.float32)
        s0 = dev(s0_t, ttnn.float32)
    const_tiles = _const_tiles(device)

    monkeypatch.setenv("QWEN_GDN_SCAN_MCAST", "1")
    o_on, fs_on = _run_op(device, tensors, const_tiles, s0)
    n_on = device.num_program_cache_entries()

    monkeypatch.setenv("QWEN_GDN_SCAN_MCAST", "0")
    o_off, fs_off = _run_op(device, tensors, const_tiles, s0)
    n_off = device.num_program_cache_entries()

    # The toggle must recompile exactly the scan prim (use_mcast is a hashed attribute); everything
    # else is a cache hit. A delta of 0 means the env was not read per-call or not hashed — and the
    # bit-exact comparison below would be vacuously comparing the op against itself.
    assert n_off - n_on == 1, (
        f"QWEN_GDN_SCAN_MCAST toggle compiled {n_off - n_on} new programs (expected exactly the "
        "scan prim): env not read per-call or use_mcast not in the program-cache key"
    )

    # Bit-exact: the multicast delivers the same bytes to the same CB slots the plain reader fills.
    assert torch.equal(o_on, o_off), "scan multicast changed o (must be bit-identical)"
    assert torch.equal(fs_on, fs_off), "scan multicast changed final_state (must be bit-identical)"
