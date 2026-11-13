# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit and perf tests for ``ttnn.experimental.deepseek.compress_pool_decode``.

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_compress_pool_decode_op.py
    python -m tracy -r -p -v -m pytest models/experimental/deepseek_v41_flash/tests/decode/test_compress_pool_decode_op.py -k perf
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v4_flash.tt.common import width_sharded_l1_config
from models.experimental.deepseek_v4_flash.tt.decode.attention import (
    _apply_rope,
    _interleaved_rotate_matrix,
    _one_row_per_user,
    make_rope_table,
)
from models.experimental.deepseek_v4_flash.tt.layers import DeepSeekV4RMSNorm

HEAD_DIM = 512
ROPE_DIM = 64
EPS = 1e-6
PCC_THRESHOLD = 0.9999

FUSED_KV_CORES = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(7, 9))})


def _torch_rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma


def _torch_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int = ROPE_DIM) -> torch.Tensor:
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    pairs = rope.reshape(*rope.shape[:-1], -1, 2)
    rotated = torch.stack([-pairs[..., 1], pairs[..., 0]], dim=-1).reshape_as(rope)
    return torch.cat([nope, rope * cos + rotated * sin], dim=-1)


def _make_rope_tables(device, rope_dim: int = ROPE_DIM, seed: int = 123):
    gen = torch.Generator().manual_seed(seed)
    angles = torch.rand(1, rope_dim // 2, generator=gen) * 6.28
    cos_host, sin_host = make_rope_table(angles.cos(), angles.sin())
    cos_tt = ttnn.from_torch(
        cos_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    sin_tt = ttnn.from_torch(
        sin_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return (
        cos_tt,
        sin_tt,
        cos_host.bfloat16().float().reshape(1, rope_dim),
        sin_host.bfloat16().float().reshape(1, rope_dim),
    )


def _make_row_tensor(x: torch.Tensor, device, mem_mode: str) -> ttnn.Tensor:
    if mem_mode == "dram":
        mem = ttnn.DRAM_MEMORY_CONFIG
    elif mem_mode == "l1_16":
        mem = width_sharded_l1_config(1, HEAD_DIM, device, num_cores=16, tile_height=1)
    elif mem_mode == "l1_kv_grid":
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(FUSED_KV_CORES, [1, HEAD_DIM // 16], ttnn.ShardOrientation.ROW_MAJOR),
        )
    elif mem_mode == "l1_8":
        mem = width_sharded_l1_config(1, HEAD_DIM, device, num_cores=8, tile_height=1)
    else:
        raise ValueError(f"unknown mem_mode {mem_mode}")
    return ttnn.from_torch(
        x.reshape(1, 1, 1, HEAD_DIM),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem,
    )


def _slot_tensor(idx: int, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.tensor([idx], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _baseline_ttnn_latent_and_rope(
    kv_host: torch.Tensor,
    gate_host: torch.Tensor | None,
    win_kv_host: torch.Tensor | None,
    win_gate_host: torch.Tensor | None,
    win_slot_idx: int,
    prev_slot_idx: int,
    norm_weight_host: torch.Tensor,
    cos_tt: ttnn.Tensor | None,
    sin_tt: ttnn.Tensor | None,
    device,
    do_pool: bool = True,
):
    """Exact TTNN baseline matching ``DeepSeekV41Attention._latent`` + ``_apply_rope``."""
    kv_tt = _make_row_tensor(kv_host, device, "dram")
    norm = DeepSeekV4RMSNorm(norm_weight_host.reshape(-1), EPS, device, sharded=True)
    if win_kv_host is not None:
        assert gate_host is not None and win_gate_host is not None
        gate_tt = _make_row_tensor(gate_host, device, "dram")
        win_kv_tt = ttnn.from_torch(
            win_kv_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        win_gate_tt = ttnn.from_torch(
            win_gate_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        slot_tt = _slot_tensor(win_slot_idx, device)
        ttnn.experimental.paged_update_cache(win_kv_tt, _one_row_per_user(kv_tt), update_idxs_tensor=slot_tt)
        ttnn.experimental.paged_update_cache(win_gate_tt, _one_row_per_user(gate_tt), update_idxs_tensor=slot_tt)
        if not do_pool:
            return None, None
        k0 = ttnn.slice(win_kv_tt, [0, 0, prev_slot_idx, 0], [1, 1, prev_slot_idx + 1, HEAD_DIM])
        g0 = ttnn.slice(win_gate_tt, [0, 0, prev_slot_idx, 0], [1, 1, prev_slot_idx + 1, HEAD_DIM])
        k1 = ttnn.to_layout(kv_tt, ttnn.TILE_LAYOUT)
        g1 = ttnn.to_layout(gate_tt, ttnn.TILE_LAYOUT)
        w0 = ttnn.sigmoid(ttnn.subtract(g0, g1))
        pooled = ttnn.add(k1, ttnn.multiply(w0, ttnn.subtract(k0, k1)))
    else:
        pooled = kv_tt

    latent = norm(pooled)
    if cos_tt is None or sin_tt is None:
        return latent, None
    rot = ttnn.from_torch(
        _interleaved_rotate_matrix(cos_tt.shape[-1]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    roped = _apply_rope(latent, cos_tt, sin_tt, rot, cos_tt.shape[-1])
    return latent, roped


@pytest.mark.parametrize("mem_mode", ("dram", "l1_16", "l1_kv_grid", "l1_8"))
@pytest.mark.parametrize("do_pool", (True, False))
@pytest.mark.parametrize("with_rope", (True, False))
@pytest.mark.parametrize("rope_dim", (64, 32))
def test_compress_pool_decode_ratio2(device, reset_seeds, mem_mode, do_pool, with_rope, rope_dim):
    if not do_pool and (with_rope or rope_dim != 64):
        pytest.skip("do_pool=False only updates win_kv/win_gate")
    if rope_dim != 64 and not with_rope:
        pytest.skip("rope_dim irrelevant without RoPE")

    torch.manual_seed(42)
    kv_host = torch.randn(1, HEAD_DIM).bfloat16().float()
    gate_host = torch.randn(1, HEAD_DIM).bfloat16().float()
    win_kv_host = torch.randn(1, 1, 2, HEAD_DIM).bfloat16().float()
    win_gate_host = torch.randn(1, 1, 2, HEAD_DIM).bfloat16().float()
    norm_weight_host = (1.0 + 0.1 * torch.randn(1, 1, 1, HEAD_DIM)).bfloat16().float()

    win_slot_idx = 1 if do_pool else 0
    prev_slot_idx = 1 - win_slot_idx

    kv_tt = _make_row_tensor(kv_host, device, mem_mode)
    gate_tt = _make_row_tensor(gate_host, device, mem_mode)
    win_kv_tt = ttnn.from_torch(
        win_kv_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    win_gate_tt = ttnn.from_torch(
        win_gate_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    win_slot_tt = _slot_tensor(win_slot_idx, device)
    norm_w_tt = ttnn.from_torch(
        norm_weight_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    cos_tt, sin_tt, cos_ref, sin_ref = (
        _make_rope_tables(device, rope_dim=rope_dim) if with_rope else (None, None, None, None)
    )

    outputs = ttnn.experimental.deepseek.compress_pool_decode(
        kv_tt,
        norm_w_tt,
        epsilon=EPS,
        gate=gate_tt,
        win_kv=win_kv_tt,
        win_gate=win_gate_tt,
        win_slot=win_slot_tt,
        cos=cos_tt,
        sin=sin_tt,
        do_pool=do_pool,
    )

    # Verify in-place window updates (exact bit-for-bit on updated & untouched rows).
    expected_win_kv = win_kv_host.clone()
    expected_win_kv[0, 0, win_slot_idx] = kv_host.reshape(-1)
    expected_win_gate = win_gate_host.clone()
    expected_win_gate[0, 0, win_slot_idx] = gate_host.reshape(-1)

    actual_win_kv = ttnn.to_torch(win_kv_tt).float()
    actual_win_gate = ttnn.to_torch(win_gate_tt).float()
    assert torch.equal(actual_win_kv, expected_win_kv), "win_kv mismatch"
    assert torch.equal(actual_win_gate, expected_win_gate), "win_gate mismatch"

    if not do_pool:
        assert outputs == []
        return

    k0 = win_kv_host[0, 0, prev_slot_idx : prev_slot_idx + 1]
    g0 = win_gate_host[0, 0, prev_slot_idx : prev_slot_idx + 1]
    w0 = torch.sigmoid(g0 - gate_host)
    pooled_ref = kv_host + w0 * (k0 - kv_host)
    latent_ref = _torch_rms_norm(pooled_ref, norm_weight_host.reshape(1, HEAD_DIM))

    if with_rope:
        assert len(outputs) == 2
        latent_tt, roped_tt = outputs
        assert roped_tt.layout == ttnn.ROW_MAJOR_LAYOUT
        assert list(roped_tt.shape) == [1, 1, 1, HEAD_DIM]
        roped_ref = _torch_rope(latent_ref, cos_ref, sin_ref, rope_dim=rope_dim)
        actual_roped = ttnn.to_torch(roped_tt).float().reshape(1, HEAD_DIM)
        passing, pcc = comp_pcc(roped_ref, actual_roped, PCC_THRESHOLD)
        logger.info(f"ratio=2 ({mem_mode}, rope={rope_dim}) roped PCC: {pcc}")
        assert passing, f"roped PCC {pcc} < {PCC_THRESHOLD}"
    else:
        assert len(outputs) == 1
        latent_tt = outputs[0]

    assert latent_tt.layout == ttnn.TILE_LAYOUT
    assert list(latent_tt.shape) == [1, 1, 1, HEAD_DIM]
    actual_latent = ttnn.to_torch(latent_tt).float().reshape(1, HEAD_DIM)
    passing, pcc = comp_pcc(latent_ref, actual_latent, PCC_THRESHOLD)
    logger.info(f"ratio=2 ({mem_mode}, with_rope={with_rope}) latent PCC: {pcc}")
    assert passing, f"latent PCC {pcc} < {PCC_THRESHOLD}"

    # Also cross-check against TTNN baseline chain when rope_dim == 64.
    if with_rope and rope_dim == 64 and mem_mode == "dram":
        base_latent, base_roped = _baseline_ttnn_latent_and_rope(
            kv_host,
            gate_host,
            win_kv_host,
            win_gate_host,
            win_slot_idx,
            prev_slot_idx,
            norm_weight_host,
            cos_tt,
            sin_tt,
            device,
        )
        passing_l, pcc_l = comp_pcc(
            ttnn.to_torch(base_latent).float(), actual_latent.reshape(1, 1, 1, HEAD_DIM), PCC_THRESHOLD
        )
        passing_r, pcc_r = comp_pcc(
            ttnn.to_torch(base_roped).float(), actual_roped.reshape(1, 1, 1, HEAD_DIM), PCC_THRESHOLD
        )
        assert passing_l and passing_r, f"vs baseline latent={pcc_l}, roped={pcc_r}"


@pytest.mark.parametrize(
    "rows,prev_idx,win_idx",
    (
        (2, 0, 1),
        (64, 0, 1),
        (64, 32, 33),  # same tile in second 32-row tile block
        (64, 31, 32),  # crossing 32x32 tile boundary
        (64, 63, 0),  # wrap across 32x32 tile boundary
    ),
)
def test_compress_pool_decode_multi_session_slots(device, reset_seeds, rows, prev_idx, win_idx):
    torch.manual_seed(7 + prev_idx + win_idx)
    kv_host = torch.randn(1, HEAD_DIM).bfloat16().float()
    gate_host = torch.randn(1, HEAD_DIM).bfloat16().float()
    win_kv_host = torch.randn(1, 1, rows, HEAD_DIM).bfloat16().float()
    win_gate_host = torch.randn(1, 1, rows, HEAD_DIM).bfloat16().float()
    norm_weight_host = (1.0 + 0.1 * torch.randn(1, 1, 1, HEAD_DIM)).bfloat16().float()

    kv_tt = _make_row_tensor(kv_host, device, "l1_16")
    gate_tt = _make_row_tensor(gate_host, device, "l1_16")
    win_kv_tt = ttnn.from_torch(
        win_kv_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    win_gate_tt = ttnn.from_torch(
        win_gate_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    win_slot_tt = _slot_tensor(win_idx, device)
    prev_slot_tt = _slot_tensor(prev_idx, device)
    norm_w_tt = ttnn.from_torch(
        norm_weight_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cos_tt, sin_tt, cos_ref, sin_ref = _make_rope_tables(device, rope_dim=ROPE_DIM, seed=99)

    latent_tt, roped_tt = ttnn.experimental.deepseek.compress_pool_decode(
        kv_tt,
        norm_w_tt,
        epsilon=EPS,
        gate=gate_tt,
        win_kv=win_kv_tt,
        win_gate=win_gate_tt,
        win_slot=win_slot_tt,
        prev_slot=prev_slot_tt,
        cos=cos_tt,
        sin=sin_tt,
        do_pool=True,
    )

    # All other rows (including prev_idx when prev_idx != win_idx) must remain bit-exact!
    expected_win_kv = win_kv_host.clone()
    expected_win_kv[0, 0, win_idx] = kv_host.reshape(-1)
    expected_win_gate = win_gate_host.clone()
    expected_win_gate[0, 0, win_idx] = gate_host.reshape(-1)
    assert torch.equal(ttnn.to_torch(win_kv_tt).float(), expected_win_kv)
    assert torch.equal(ttnn.to_torch(win_gate_tt).float(), expected_win_gate)

    k0 = win_kv_host[0, 0, prev_idx : prev_idx + 1]
    g0 = win_gate_host[0, 0, prev_idx : prev_idx + 1]
    w0 = torch.sigmoid(g0 - gate_host)
    pooled_ref = kv_host + w0 * (k0 - kv_host)
    latent_ref = _torch_rms_norm(pooled_ref, norm_weight_host.reshape(1, HEAD_DIM))
    roped_ref = _torch_rope(latent_ref, cos_ref, sin_ref, rope_dim=ROPE_DIM)

    p_lat, pcc_lat = comp_pcc(latent_ref, ttnn.to_torch(latent_tt).float().reshape(1, HEAD_DIM), PCC_THRESHOLD)
    p_rop, pcc_rop = comp_pcc(roped_ref, ttnn.to_torch(roped_tt).float().reshape(1, HEAD_DIM), PCC_THRESHOLD)
    logger.info(f"slots ({prev_idx} -> {win_idx}, rows={rows}): latent={pcc_lat}, roped={pcc_rop}")
    assert p_lat and p_rop


@pytest.mark.parametrize("mem_mode", ("dram", "l1_16", "l1_kv_grid"))
@pytest.mark.parametrize("with_rope", (True, False))
def test_compress_pool_decode_ratio1(device, reset_seeds, mem_mode, with_rope):
    torch.manual_seed(11)
    kv_host = torch.randn(1, HEAD_DIM).bfloat16().float()
    norm_weight_host = (1.0 + 0.1 * torch.randn(1, 1, 1, HEAD_DIM)).bfloat16().float()

    kv_tt = _make_row_tensor(kv_host, device, mem_mode)
    norm_w_tt = ttnn.from_torch(
        norm_weight_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cos_tt, sin_tt, cos_ref, sin_ref = (
        _make_rope_tables(device, rope_dim=ROPE_DIM) if with_rope else (None, None, None, None)
    )

    outputs = ttnn.experimental.deepseek.compress_pool_decode(
        kv_tt,
        norm_w_tt,
        epsilon=EPS,
        cos=cos_tt,
        sin=sin_tt,
    )

    latent_ref = _torch_rms_norm(kv_host, norm_weight_host.reshape(1, HEAD_DIM))
    if with_rope:
        assert len(outputs) == 2
        latent_tt, roped_tt = outputs
        roped_ref = _torch_rope(latent_ref, cos_ref, sin_ref, rope_dim=ROPE_DIM)
        p_r, pcc_r = comp_pcc(roped_ref, ttnn.to_torch(roped_tt).float().reshape(1, HEAD_DIM), PCC_THRESHOLD)
        assert p_r, f"ratio=1 roped PCC {pcc_r}"
    else:
        assert len(outputs) == 1
        latent_tt = outputs[0]

    p_l, pcc_l = comp_pcc(latent_ref, ttnn.to_torch(latent_tt).float().reshape(1, HEAD_DIM), PCC_THRESHOLD)
    assert p_l, f"ratio=1 latent PCC {pcc_l}"


def test_compress_pool_decode_sequential_and_cache(device, reset_seeds):
    """Simulate 8 consecutive decode steps (positions 0..7) alternating do_pool=False/True + program cache reuse."""
    torch.manual_seed(2026)
    win_kv_host = torch.zeros(1, 1, 2, HEAD_DIM)
    win_gate_host = torch.zeros(1, 1, 2, HEAD_DIM)
    norm_weight_host = (1.0 + 0.1 * torch.randn(1, 1, 1, HEAD_DIM)).bfloat16().float()

    win_kv_tt = ttnn.from_torch(
        win_kv_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    win_gate_tt = ttnn.from_torch(
        win_gate_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    norm_w_tt = ttnn.from_torch(
        norm_weight_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    steady_cache_entries = None
    for pos in range(8):
        kv_host = torch.randn(1, HEAD_DIM).bfloat16().float()
        gate_host = torch.randn(1, HEAD_DIM).bfloat16().float()
        slot = pos % 2
        do_pool = (pos % 2) == 1

        kv_tt = _make_row_tensor(kv_host, device, "l1_16")
        gate_tt = _make_row_tensor(gate_host, device, "l1_16")
        slot_tt = _slot_tensor(slot, device)
        cos_tt, sin_tt, cos_ref, sin_ref = _make_rope_tables(device, rope_dim=ROPE_DIM, seed=1000 + pos)

        outputs = ttnn.experimental.deepseek.compress_pool_decode(
            kv_tt,
            norm_w_tt,
            epsilon=EPS,
            gate=gate_tt,
            win_kv=win_kv_tt,
            win_gate=win_gate_tt,
            win_slot=slot_tt,
            cos=cos_tt if do_pool else None,
            sin=sin_tt if do_pool else None,
            do_pool=do_pool,
        )

        win_kv_host[0, 0, slot] = kv_host.reshape(-1)
        win_gate_host[0, 0, slot] = gate_host.reshape(-1)

        assert torch.equal(ttnn.to_torch(win_kv_tt).float(), win_kv_host)
        assert torch.equal(ttnn.to_torch(win_gate_tt).float(), win_gate_host)

        if do_pool:
            if steady_cache_entries is None:
                # Both do_pool=False (pos=0) and do_pool=True (pos=1) are now in the program cache.
                steady_cache_entries = device.num_program_cache_entries()
            else:
                assert device.num_program_cache_entries() == steady_cache_entries
            k0 = win_kv_host[0, 0, 0:1]
            g0 = win_gate_host[0, 0, 0:1]
            w0 = torch.sigmoid(g0 - gate_host)
            pooled_ref = kv_host + w0 * (k0 - kv_host)
            latent_ref = _torch_rms_norm(pooled_ref, norm_weight_host.reshape(1, HEAD_DIM))
            roped_ref = _torch_rope(latent_ref, cos_ref, sin_ref, rope_dim=ROPE_DIM)

            latent_tt, roped_tt = outputs
            p_l, pcc_l = comp_pcc(latent_ref, ttnn.to_torch(latent_tt).float().reshape(1, HEAD_DIM), PCC_THRESHOLD)
            p_r, pcc_r = comp_pcc(roped_ref, ttnn.to_torch(roped_tt).float().reshape(1, HEAD_DIM), PCC_THRESHOLD)
            assert p_l and p_r, f"pos={pos}: latent={pcc_l}, roped={pcc_r}"
        else:
            assert outputs == []
            if steady_cache_entries is not None:
                assert device.num_program_cache_entries() == steady_cache_entries


@pytest.mark.parametrize("mode", ("fused_pool_rope", "fused_nopool", "fused_ratio1", "baseline_pool_rope"))
def test_compress_pool_decode_perf(device, reset_seeds, mode):
    """Warm dispatches of compress_pool_decode (vs baseline chain) for Tracy timing:

    python -m tracy -r -p -v -m pytest models/experimental/deepseek_v41_flash/tests/decode/test_compress_pool_decode_op.py -k perf
    """
    torch.manual_seed(55)
    kv_host = torch.randn(1, HEAD_DIM).bfloat16().float()
    gate_host = torch.randn(1, HEAD_DIM).bfloat16().float()
    win_kv_host = torch.randn(1, 1, 2, HEAD_DIM).bfloat16().float()
    win_gate_host = torch.randn(1, 1, 2, HEAD_DIM).bfloat16().float()
    norm_weight_host = (1.0 + 0.1 * torch.randn(1, 1, 1, HEAD_DIM)).bfloat16().float()

    kv_tt = _make_row_tensor(kv_host, device, "l1_kv_grid")
    gate_tt = _make_row_tensor(gate_host, device, "l1_kv_grid")
    win_kv_tt = ttnn.from_torch(
        win_kv_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    win_gate_tt = ttnn.from_torch(
        win_gate_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    win_slot_tt = _slot_tensor(1, device)
    norm_w_tt = ttnn.from_torch(
        norm_weight_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cos_tt, sin_tt, _, _ = _make_rope_tables(device, rope_dim=ROPE_DIM)

    if mode == "baseline_pool_rope":
        norm = DeepSeekV4RMSNorm(norm_weight_host.reshape(-1), EPS, device, sharded=True)
        rot = ttnn.from_torch(
            _interleaved_rotate_matrix(ROPE_DIM),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(10):
            kv_rm = _one_row_per_user(kv_tt)
            gate_rm = _one_row_per_user(gate_tt)
            ttnn.experimental.paged_update_cache(win_kv_tt, kv_rm, update_idxs_tensor=win_slot_tt)
            ttnn.experimental.paged_update_cache(win_gate_tt, gate_rm, update_idxs_tensor=win_slot_tt)
            k0 = ttnn.slice(win_kv_tt, [0, 0, 0, 0], [1, 1, 1, HEAD_DIM])
            g0 = ttnn.slice(win_gate_tt, [0, 0, 0, 0], [1, 1, 1, HEAD_DIM])
            k1 = ttnn.to_layout(ttnn.to_memory_config(kv_tt, ttnn.DRAM_MEMORY_CONFIG), ttnn.TILE_LAYOUT)
            g1 = ttnn.to_layout(ttnn.to_memory_config(gate_tt, ttnn.DRAM_MEMORY_CONFIG), ttnn.TILE_LAYOUT)
            w0 = ttnn.sigmoid(ttnn.subtract(g0, g1))
            pooled = ttnn.add(k1, ttnn.multiply(w0, ttnn.subtract(k0, k1)))
            lat = norm(pooled)
            rop = _apply_rope(lat, cos_tt, sin_tt, rot, ROPE_DIM)
            for t in (kv_rm, gate_rm, k0, g0, k1, g1, w0, pooled, lat, rop):
                ttnn.deallocate(t)
        ttnn.synchronize_device(device)
        return

    for _ in range(10):
        if mode == "fused_pool_rope":
            outs = ttnn.experimental.deepseek.compress_pool_decode(
                kv_tt,
                norm_w_tt,
                epsilon=EPS,
                gate=gate_tt,
                win_kv=win_kv_tt,
                win_gate=win_gate_tt,
                win_slot=win_slot_tt,
                cos=cos_tt,
                sin=sin_tt,
                do_pool=True,
            )
        elif mode == "fused_nopool":
            outs = ttnn.experimental.deepseek.compress_pool_decode(
                kv_tt,
                norm_w_tt,
                epsilon=EPS,
                gate=gate_tt,
                win_kv=win_kv_tt,
                win_gate=win_gate_tt,
                win_slot=win_slot_tt,
                do_pool=False,
            )
        else:
            outs = ttnn.experimental.deepseek.compress_pool_decode(
                kv_tt,
                norm_w_tt,
                epsilon=EPS,
                cos=cos_tt,
                sin=sin_tt,
            )
        for o in outs:
            ttnn.deallocate(o)
    ttnn.synchronize_device(device)
