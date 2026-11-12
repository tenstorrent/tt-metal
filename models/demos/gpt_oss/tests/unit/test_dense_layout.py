# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the batched dense decode MoE layout (tt/experts/dense_layout.py) with every expert padded to
whole tiles (gpt-oss-20b at TP 4: 720 -> 736 columns), and of the routed prefill's page table over it
(tt/experts/stream_pages.py dense_page_table):
  * the padded layout holds each device's shard of every expert at its place and zeros in the padded columns / rows;
  * the dense decode math (decode_forward_dense: biases, SwiGLU-OAI with clamp, routing weights, down, down bias,
    summed over the TP devices) on the padded layout equals the per-expert MoE: padded columns add exactly zero;
  * reading tile (expert, column, k) through the page table returns that tile of the padded tensors in a simulated
    DRAM width-sharded buffer (tt_metal page mapping: shard i on bank i, row stride = shard width, partial last shard);
  * the decode programs' core split covers the 736-tile activation exactly (fused_decode/config.py DENSE_MOE_*).

Loads the modules from their files, so no ttnn is imported: runs on any host with torch.
"""

import ast
import importlib.util
from pathlib import Path

import pytest
import torch

_TT = Path(__file__).resolve().parents[2] / "tt"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dl = _load("gpt_oss_dense_layout", _TT / "experts" / "dense_layout.py")
sp = _load("gpt_oss_stream_pages", _TT / "experts" / "stream_pages.py")

TILE = 32
E20, H20, I20, TP20, BANKS = 32, 2880, 2880, 4, 8


def _config_constants():
    """The DENSE_MOE_* integer constants of fused_decode/config.py (parsed: the module imports ttnn)."""
    tree = ast.parse((_TT / "fused_decode" / "config.py").read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.startswith("DENSE_MOE_") and isinstance(node.value, ast.Constant):
                out[name] = node.value.value
    return out


def _old_side_by_side(w, tp):
    """The unpadded layout as experts/weights.py built it before the padding (gate / up)."""
    E, H, I = w.shape
    il = I // tp
    return torch.cat(
        [w[:, :, d * il : (d + 1) * il].permute(1, 0, 2).reshape(H, E * il) for d in range(tp)], dim=-1
    ).reshape(1, 1, H, tp * E * il)


def test_padded_width():
    assert dl.padded_expert_width(I20 // TP20) == 736
    assert dl.padded_expert_width(768) == 768


@pytest.mark.parametrize("E,H,I,tp", [(3, 64, 2880, 4), (2, 32, 96, 2)])
def test_padded_layout(E, H, I, tp):
    gen = torch.Generator().manual_seed(0)
    il = I // tp
    il_pad = dl.padded_expert_width(il)
    w = torch.randn(E, H, I, generator=gen)
    wd = torch.randn(E, I, H, generator=gen)
    b = torch.randn(E, I, generator=gen)
    gate = dl.gate_up_columns(w, tp, il_pad)[0, 0].reshape(H, tp, E, il_pad)  # device d = column block d
    down = dl.down_rows(wd, tp, il_pad)[0, 0].reshape(tp, E, il_pad, H)
    bias = dl.bias_rows(b, tp, il_pad, 5)[0, 0]
    assert bias.shape == (5, tp * E * il_pad) and torch.equal(bias[0], bias[4])
    bias = bias[0].reshape(tp, E, il_pad)
    for d in range(tp):
        for e in range(E):
            assert torch.equal(gate[:, d, e, :il], w[e, :, d * il : (d + 1) * il])
            assert torch.equal(down[d, e, :il], wd[e, d * il : (d + 1) * il])
            assert torch.equal(bias[d, e, :il], b[e, d * il : (d + 1) * il])
    # padded columns / rows are zero
    assert not gate[..., il:].any() and not down[:, :, il:].any() and not bias[..., il:].any()
    # without padding: the old layout exactly
    if il % TILE == 0:
        assert torch.equal(dl.gate_up_columns(w, tp, il), _old_side_by_side(w, tp))
    db = dl.down_bias_rows(torch.randn(E, H, generator=gen), tp)[0, 0]
    assert db.shape == (E, tp * H) and not db[:, H:].any()
    ex = dl.expand_matrix(E, il_pad)[0, 0]
    assert torch.equal(ex.sum(0), torch.ones(E * il_pad)) and all(ex[e, e * il_pad + il_pad - 1] == 1 for e in range(E))


def _swiglu_oai(g, u, limit=7.0, alpha=1.702):
    g = g.clamp(max=limit)
    u = u.clamp(-limit, limit)
    return (u + 1) * g * torch.sigmoid(alpha * g)


def test_swiglu_of_zero_is_zero():
    z = torch.zeros(4, dtype=torch.float64)
    assert torch.equal(_swiglu_oai(z, z), z)
    assert torch.equal(_swiglu_oai(z, torch.full_like(z, 100.0)), z)  # up clamped, gate 0


def test_padded_columns_add_nothing():
    """decode_forward_dense on the padded layout (per device: x @ gate + b, x @ up + b, SwiGLU, x routing weight of
    the column's expert (w @ expand), @ down, + w @ down_bias; summed over devices) == the per-expert MoE."""
    gen = torch.Generator().manual_seed(1)
    E, H, I, tp, T, k = 4, 64, 2880, 4, 6, 2
    il = I // tp
    il_pad = dl.padded_expert_width(il)
    f = torch.float64
    gate_w = torch.randn(E, H, I, generator=gen, dtype=f) * 0.2
    up_w = torch.randn(E, H, I, generator=gen, dtype=f) * 0.2
    down_w = torch.randn(E, I, H, generator=gen, dtype=f) * 0.2
    gate_b = torch.randn(E, I, generator=gen, dtype=f)
    up_b = torch.randn(E, I, generator=gen, dtype=f)
    down_b = torch.randn(E, H, generator=gen, dtype=f)
    x = torch.randn(T, H, generator=gen, dtype=f) * 3
    # routing weights: top-k softmax, zeros elsewhere
    logits = torch.randn(T, E, generator=gen, dtype=f)
    top = logits.topk(k, dim=-1)
    rw = torch.zeros(T, E, dtype=f).scatter(1, top.indices, torch.softmax(top.values, -1))

    ref = torch.zeros(T, H, dtype=f)
    for e in range(E):
        act = _swiglu_oai(x @ gate_w[e] + gate_b[e], x @ up_w[e] + up_b[e])
        ref += rw[:, e : e + 1] * (act @ down_w[e] + down_b[e])

    n = E * il_pad
    gate = dl.gate_up_columns(gate_w, tp, il_pad)[0, 0]
    up = dl.gate_up_columns(up_w, tp, il_pad)[0, 0]
    down = dl.down_rows(down_w, tp, il_pad)[0, 0]
    gb = dl.bias_rows(gate_b, tp, il_pad, T)[0, 0]
    ub = dl.bias_rows(up_b, tp, il_pad, T)[0, 0]
    db = dl.down_bias_rows(down_b, tp)[0, 0]
    expand = dl.expand_matrix(E, il_pad)[0, 0].to(f)
    out = torch.zeros(T, H, dtype=f)
    for d in range(tp):
        cols = slice(d * n, (d + 1) * n)
        act = _swiglu_oai(x @ gate[:, cols] + gb[:, cols], x @ up[:, cols] + ub[:, cols]) * (rw @ expand)
        pad = torch.zeros(E, il_pad, dtype=torch.bool)
        pad[:, il:] = True
        assert not act[:, pad.reshape(-1)].any(), "a padded column is non-zero"
        out += act @ down[cols] + rw @ db[:, d * H : (d + 1) * H]
    assert torch.allclose(out, ref, rtol=1e-10, atol=1e-9)


def _sharded_banks(t, shard_tiles, banks, tile_bytes_fn):
    """Simulated DRAM width-sharded buffer of a [K, N] tensor (tt_metal buffer page mapping): bank b holds tile
    columns [b W, (b + 1) W), tile (k, c) at index k W + c of the shard (a partial last shard keeps the stride W, its
    missing tiles filled with 0xFF bytes). Returns the per-bank uint8 bytes."""
    K, N = t.shape
    kt, nt = K // TILE, N // TILE
    W = shard_tiles
    assert W * banks >= nt
    out = []
    for b in range(banks):
        tiles = []
        for kk in range(kt):
            for c in range(W):
                n = b * W + c
                tiles.append(tile_bytes_fn(t, kk, n) if n < nt else None)
        size = len(tile_bytes_fn(t, 0, 0))
        out.append(torch.cat([x if x is not None else torch.full((size,), 0xFF, dtype=torch.uint8) for x in tiles]))
    return out


def _tile_f32(t, k, n):
    return t[k * TILE : (k + 1) * TILE, n * TILE : (n + 1) * TILE].contiguous().reshape(-1).view(torch.uint8)


@pytest.mark.parametrize("E,H,I,tp,banks,wpb", [(4, 96, 160, 2, 4, 3), (3, 64, 2880, 4, 8, 3), (5, 128, 96, 1, 4, 1)])
def test_dense_page_table_reads_padded_tiles(E, H, I, tp, banks, wpb):
    """Rebuild every expert's padded gate / up / down slice of one device from tiles read through the page table."""
    gen = torch.Generator().manual_seed(2)
    il = I // tp
    il_pad = dl.padded_expert_width(il)
    n = E * il_pad
    gate_all = dl.gate_up_columns(torch.randn(E, H, I, generator=gen), tp, il_pad)[0, 0]
    up_all = dl.gate_up_columns(torch.randn(E, H, I, generator=gen), tp, il_pad)[0, 0]
    down_all = dl.down_rows(torch.randn(E, I, H, generator=gen), tp, il_pad)[0, 0]
    d = tp - 1  # the last device's shard
    gate, up, down = gate_all[:, d * n : (d + 1) * n], up_all[:, d * n : (d + 1) * n], down_all[d * n : (d + 1) * n]
    gu_w = dl.dram_shard_tiles(n // TILE, banks, wpb)
    dn_w = dl.dram_shard_tiles(-(-H // TILE), banks, wpb)
    tb = TILE * TILE * 4  # float32 tiles
    g_banks = _sharded_banks(gate, gu_w, banks, _tile_f32)
    u_banks = _sharded_banks(up, gu_w, banks, _tile_f32)
    d_banks = _sharded_banks(down, dn_w, banks, _tile_f32)
    # gate and up share one address space per bank: up at gate + up_delta
    up_delta = len(g_banks[0]) + 4 * tb
    gu = [torch.cat([g, torch.zeros(up_delta - len(g), dtype=torch.uint8), u]) for g, u in zip(g_banks, u_banks)]
    table = sp.dense_page_table(E, il_pad, H, gu_w, dn_w, tb, up_delta)
    n_gu, n_dn = il_pad // TILE, H // TILE
    assert table.shape == (E, 2 * (2 * n_gu + n_dn))
    ge, ue, de = sp.table_entries(table, n_gu, n_dn)

    def rebuild(bufs, entries, k_tiles, stride):
        """[k_tiles * 32, cols * 32] from tile k of every entry (k at offset + k * stride)."""
        rows = []
        for kk in range(k_tiles):
            raw = sp.read_tiles(bufs, entries, kk, stride, tb).reshape(-1, tb)
            tiles = [r.view(torch.float32).reshape(TILE, TILE) for r in raw]
            rows.append(torch.cat(tiles, dim=1))
        return torch.cat(rows, dim=0)

    for e in range(E):
        cols = slice(e * il_pad, (e + 1) * il_pad)
        assert torch.equal(rebuild(gu, ge[e], H // TILE, gu_w * tb), gate[:, cols])
        assert torch.equal(rebuild(gu, ue[e], H // TILE, gu_w * tb), up[:, cols])
        assert torch.equal(rebuild(d_banks, de[e], n_gu, dn_w * tb), down[cols])
        assert not gate[:, cols][:, il:].any() and not down[cols][il:].any()


def test_dense_page_table_gpt_oss_20b():
    """gpt-oss-20b at TP 4: 23 gate / up columns and 90 down columns per expert; every (bank, offset) is in range,
    DRAM aligned, and the table addresses each tile of the [2880, 32 x 736] / [32 x 736, 2880] tensors once."""
    il_pad = dl.padded_expert_width(I20 // TP20)
    n_tiles = E20 * il_pad // TILE
    gu_w = dl.dram_shard_tiles(n_tiles, BANKS, 3)
    dn_w = dl.dram_shard_tiles(H20 // TILE, BANKS, 3)
    assert (gu_w, dn_w) == (93, 12)
    for tb in (sp.BFP4_BYTES, sp.BFP8_BYTES):
        table = sp.dense_page_table(E20, il_pad, H20, gu_w, dn_w, tb)
        ge, ue, de = sp.table_entries(table, 23, 90)
        assert ge.shape == (E20, 23, 2) and de.shape == (E20, 90, 2) and torch.equal(ge, ue)
        assert int(table.reshape(-1, 2)[:, 0].max()) < BANKS
        assert all(int(v) % 64 == 0 for v in table.reshape(-1, 2)[:, 1]) and (gu_w * tb) % 64 == 0
        # every gate tile (k, n) once: bank * shard + (offset / tb) over all k
        seen = set()
        for k in range(H20 // TILE):
            for b, o in ge.reshape(-1, 2).tolist():
                seen.add((b, o // tb + k * gu_w))
        assert len(seen) == (H20 // TILE) * n_tiles
        # last bank: a partial shard (85 of 93 columns), its last real column at the right place
        b, o = ge[E20 - 1, 22].tolist()
        assert (b, o // tb) == ((n_tiles - 1) // gu_w, (n_tiles - 1) % gu_w)
        assert all(o // tb + (H20 // TILE - 1) * gu_w < gu_w * (H20 // TILE) for _, o in ge.reshape(-1, 2).tolist())
        # down: expert e's K rows start at tile row 23 e
        assert torch.equal(de[:, 0, 1], torch.arange(E20) * 23 * dn_w * tb)


def test_dense_page_table_rejects_partial_tiles():
    with pytest.raises(AssertionError, match="not whole tiles"):
        sp.dense_page_table(E20, I20 // TP20, H20, 90, 12, sp.BFP4_BYTES)


def test_dram_shard_tiles_fit_the_dram_sharded_matmul():
    """dense_layout.dram_shard_tiles meets matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp's
    conditions: shard % workers == 0, shard == workers * ceil(N / (banks * workers)), no padding-only bank."""
    wpb = _config_constants()["DENSE_MOE_WORKERS_PER_BANK"]
    assert 1 <= wpb <= 3  # validate_num_workers_per_dram_bank
    for n in (E20 * 736 // TILE, H20 // TILE, E20 * 720 // TILE):
        s = dl.dram_shard_tiles(n, BANKS, wpb)
        assert s % wpb == 0 and s == wpb * -(-n // (BANKS * wpb))
        assert s * BANKS >= n and (s * BANKS - n) // s == 0
    assert dl.dram_shard_tiles(E20 * 720 // TILE, BANKS, 3) == 90  # the unpadded layout's shard (unchanged)


def test_core_split_covers_736_tiles():
    c = _config_constants()
    il_pad = dl.padded_expert_width(I20 // TP20)
    split = dl.dense_core_split(
        H20, il_pad, E20, c["DENSE_MOE_CORES"], c["DENSE_MOE_HIDDEN_CORES"], c["DENSE_MOE_ELTWISE_CORES"], 11 * 10
    )
    assert split["act_tiles"] == 736 and split["act"] * c["DENSE_MOE_CORES"] == 736
    assert split["hidden"] * c["DENSE_MOE_HIDDEN_CORES"] == H20 // TILE
    assert split["eltwise"] * c["DENSE_MOE_ELTWISE_CORES"] == 736
    assert (split["act"], split["hidden"], split["eltwise"]) == (16, 2, 8)
    # dense_swiglu_reader.cpp: NT % 8 == 0 and each core's first column tile / 8 fits a byte
    assert split["act"] % 8 == 0 and (c["DENSE_MOE_CORES"] - 1) * split["act"] // 8 < 256
    # the DRAM-sharded matmuls' in0 blocks (DenseDecodePlan dram_sharded: largest divisor of the shard <= 16)
    assert 736 % split["act"] == 0 and (H20 // TILE) % split["hidden"] == 0
    with pytest.raises(AssertionError, match="do not split"):
        dl.dense_core_split(H20, il_pad, E20, 45, 45, 90, 110)
