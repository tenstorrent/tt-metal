# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Table-addressed weights of ``unified_routed_expert_moe`` (``weight_pages``): the op reads every gate / up / down weight
tile through a host-built page table from byte streams instead of from per-expert weight tensors.

Table layout (uint32, one row per local expert e): the gate columns' entries (weights_n_gate_up), then up's (same
count), then down's (weights_n_down); each entry is (dram_bank_id, byte_offset) of the column's tile k = 0, relative to
the source tensor's buffer address in that bank. Tile k of the column is at byte_offset + k * k_stride.

Host tests (torch only, no device): the BFP packer below produces the device's tile bytes for values on the BFP grid,
and reading tile (e, column, k) through a built table returns that tile's bytes.

Device tests (one Blackhole chip): random weights on the BFP grid, run the op once with per-expert weight tensors and
once with the same tiles scattered over the 8 DRAM banks of two width-sharded uint8 streams (random bank per column,
random gaps and garbage between them, gaps between a column's tiles). The two outputs must be bit-identical, and close
to an fp32 torch reference. Covers bfloat4_b tile-transposed tiles (gpt-oss K-GPTQ), bfloat8_b, and 32 local experts
with a 768-wide intermediate (gpt-oss-20b TP=4: every device holds a quarter of every expert).
"""

import pytest
import torch

try:
    from loguru import logger

    import ttnn
    from tests.ttnn.utils_for_testing import comp_pcc
    from tt_py_test_utils_common.utility_functions import is_blackhole
except ImportError:  # host-only run without a ttnn build: only the host tests below can run
    ttnn = None

TILE = 32
FACE = 16
EMB = 2880  # gpt-oss-20b hidden size
K_GU = EMB // TILE  # gate / up K tiles
N_D = EMB // TILE  # down output tile columns
N_GU = 24  # gate / up tile columns per expert: a 768-wide TP quarter of the 2880 intermediate, padded
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0
BITS = {"bf4": 3, "bf8": 7}  # BFP magnitude bits (bfloat4_b, bfloat8_b)


# ----------------------------------------------------------------------------------------------- host-only helpers


def tile_bytes(bits):
    """Bytes of one BFP tile: 64 shared exponents, then 1024 values of (bits + 1) bits."""
    return 64 + 1024 * (bits + 1) // 8


def tile_transpose(w):
    """Transpose every 32x32 tile of w [K, N] in place (the tile grid is unchanged). Its own inverse."""
    K, N = w.shape
    return w.reshape(K // TILE, TILE, N // TILE, TILE).permute(0, 3, 2, 1).reshape(K, N)


def _faces(tiles):
    """[n, 32, 32] -> [n, 4, 16, 16] faces in tile order (face 0: rows 0-15 x cols 0-15, 1: cols 16-31, 2, 3)."""
    n = tiles.shape[0]
    return tiles.reshape(n, 2, FACE, 2, FACE).permute(0, 1, 3, 2, 4).reshape(n, 4, FACE, FACE)


def _unfaces(f):
    n = f.shape[0]
    return f.reshape(n, 2, 2, FACE, FACE).permute(0, 1, 3, 2, 4).reshape(n, TILE, TILE)


def _exp_fp32(x):
    """Biased FP32 exponent field of x (0 for 0)."""
    return (x.float().contiguous().view(torch.int32) >> 23) & 0xFF


def bfp_quantize(tiles, bits):
    """[n, 32, 32] -> the nearest values on the BFP grid (one exponent per 16-value face row, `bits` magnitude
    bits), which a BFP pack stores exactly."""
    f = _faces(tiles.float())
    E = _exp_fp32(f.abs().amax(-1, keepdim=True))
    scale = torch.pow(2.0, (E - 127 - (bits - 1)).float())
    q = torch.clamp(torch.round(f.abs() / scale), max=(1 << bits) - 1)
    return _unfaces(torch.where(E > 0, torch.sign(f) * q * scale, torch.zeros_like(f)))


def pack_bfp(tiles, bits):
    """[n, 32, 32] values on the BFP grid -> [n, tile_bytes(bits)] uint8, the device's BFP tile bytes (tt_metal
    blockfloat_common.cpp, is_exp_a = false): 64 exponent bytes (face-major, one per face row), then per face row the
    16 values as (sign << bits) | magnitude, packed into 32-bit words with element 0 in the low bits."""
    f = _faces(tiles.float())
    E = _exp_fp32(f).amax(-1)  # [n, 4, 16]
    scale = torch.pow(2.0, (E.float() - 127 - (bits - 1)).unsqueeze(-1))
    mag = f.abs() / scale
    q = torch.round(mag)
    assert torch.equal(q, mag) and (q <= (1 << bits) - 1).all(), "value not on the BFP grid"
    sign = ((f < 0) & (q > 0)).to(torch.int64)
    code = (sign << bits) | q.to(torch.int64)
    per_word = 32 // (bits + 1)
    code = code.reshape(*code.shape[:-1], FACE // per_word, per_word)
    words = (code << (torch.arange(per_word) * (bits + 1))).sum(-1).reshape(f.shape[0], -1)
    wbytes = torch.stack([(words >> (8 * i)) & 0xFF for i in range(4)], -1).reshape(f.shape[0], -1)
    return torch.cat([E.reshape(E.shape[0], -1), wbytes], -1).to(torch.uint8)


def unpack_bfp(raw, bits):
    """Inverse of pack_bfp: [n, tile_bytes(bits)] uint8 -> [n, 32, 32] float."""
    n = raw.shape[0]
    E = raw[:, :64].to(torch.int64).reshape(n, 4, FACE)
    words = (raw[:, 64:].to(torch.int64).reshape(n, -1, 4) * torch.tensor([1, 1 << 8, 1 << 16, 1 << 24])).sum(-1)
    per_word = 32 // (bits + 1)
    code = (words.unsqueeze(-1) >> (torch.arange(per_word) * (bits + 1))) & ((1 << (bits + 1)) - 1)
    code = code.reshape(n, 4, FACE, FACE)
    q = (code & ((1 << bits) - 1)).float()
    sign = torch.where((code >> bits) & 1 == 1, -1.0, 1.0)
    return _unfaces(sign * q * torch.pow(2.0, (E.float() - 127 - (bits - 1)).unsqueeze(-1)))


def on_bfp_grid(stored, bits):
    """stored [K, N] (the layout the device tiles) -> the same matrix with every tile moved onto the BFP grid."""
    K, N = stored.shape
    tiles = stored.reshape(K // TILE, TILE, N // TILE, TILE).permute(0, 2, 1, 3).reshape(-1, TILE, TILE)
    q = bfp_quantize(tiles, bits)
    return q.reshape(K // TILE, N // TILE, TILE, TILE).permute(0, 2, 1, 3).reshape(K, N)


def column_tile_bytes(stored, bits):
    """stored [K, N] -> [N / 32, K / 32, tile_bytes] uint8: column n's tiles k = 0.. as the device stores them."""
    K, N = stored.shape
    tiles = stored.reshape(K // TILE, TILE, N // TILE, TILE).permute(2, 0, 1, 3).reshape(-1, TILE, TILE)
    return pack_bfp(tiles, bits).reshape(N // TILE, K // TILE, -1)


def scatter_columns(columns, k_stride, banks, gen):
    """columns: [C, K, tile_bytes] uint8 -> (bank_bytes [banks, L] uint8, entries [C, 2] int64 (bank, byte offset)).

    Each column goes to a random bank, in random order, after a random gap (whole 64-byte units); its tiles sit
    k_stride bytes apart. Every byte no tile covers is random, so a misaddressed read returns garbage."""
    C, K, tb = columns.shape
    assert k_stride % 64 == 0 and k_stride >= tb
    span = (K - 1) * k_stride + tb
    span = (span + 63) // 64 * 64
    entries = torch.zeros(C, 2, dtype=torch.int64)
    cursor = [0] * banks
    for c in torch.randperm(C, generator=gen).tolist():
        b = int(torch.randint(banks, (1,), generator=gen))
        cursor[b] += 64 * int(torch.randint(4, (1,), generator=gen))
        entries[c] = torch.tensor([b, cursor[b]])
        cursor[b] += span
    length = max(cursor) + 64
    bank_bytes = torch.randint(256, (banks, length), dtype=torch.uint8, generator=gen)
    for c in range(C):
        b, off = entries[c].tolist()
        for k in range(K):
            bank_bytes[b, off + k * k_stride : off + k * k_stride + tb] = columns[c, k]
    return bank_bytes, entries


def page_table(gate_entries, up_entries, down_entries):
    """Per-expert entries ([E, n_gu, 2], [E, n_gu, 2], [E, n_d, 2]) -> the op's weight_pages rows [E, 2 * (2 n_gu +
    n_d)]: gate's entries, then up's, then down's."""
    E = gate_entries.shape[0]
    return torch.cat([gate_entries, up_entries, down_entries], dim=1).reshape(E, -1)


def read_via_table(bank_bytes, table, e, entry, k, k_stride, tb):
    """Bytes of tile k of the column whose entry index (in the expert's row) is `entry`, read the way the kernel does."""
    bank, off = table[e, 2 * entry].item(), table[e, 2 * entry + 1].item()
    return bank_bytes[bank, off + k * k_stride : off + k * k_stride + tb]


def width_sharded_rows(bank_bytes):
    """[banks, L] -> [L / 64, banks * 64]: the uint8 ROW_MAJOR tensor whose DRAM width-sharded layout (one 64-byte
    page per shard row, shard b on bank b) holds bank b's bytes contiguously (the decode expert streams' layout)."""
    banks, L = bank_bytes.shape
    assert L % 64 == 0
    return torch.cat([bank_bytes[b].reshape(L // 64, 64) for b in range(banks)], dim=-1)


def _expert_streams(stored, bits, gu_stride, d_stride, banks, gen):
    """stored: per expert {gate, up, down} [K, N] on the BFP grid -> (gate/up bank bytes, down bank bytes, table)."""
    E = len(stored)
    gu_cols = torch.cat([column_tile_bytes(stored[e][r], bits) for e in range(E) for r in ("gate", "up")])
    d_cols = torch.cat([column_tile_bytes(stored[e]["down"], bits) for e in range(E)])
    gu_bytes, gu_entries = scatter_columns(gu_cols, gu_stride, banks, gen)
    d_bytes, d_entries = scatter_columns(d_cols, d_stride, banks, gen)
    n_gu = stored[0]["gate"].shape[1] // TILE
    n_d = stored[0]["down"].shape[1] // TILE
    gu_entries = gu_entries.reshape(E, 2, n_gu, 2)
    table = page_table(gu_entries[:, 0], gu_entries[:, 1], d_entries.reshape(E, n_d, 2))
    return gu_bytes, d_bytes, table


def _random_stored(E, k_gu, n_gu, n_d, bits, gen):
    def w(k, n, s):
        return on_bfp_grid(torch.randn(k * TILE, n * TILE, generator=gen) * s, bits)

    return [{"gate": w(k_gu, n_gu, 0.08), "up": w(k_gu, n_gu, 0.08), "down": w(n_gu, n_d, 0.05)} for _ in range(E)]


@pytest.mark.parametrize("fmt", ["bf4", "bf8"])
def test_bfp_pack_round_trip(fmt):
    """Host only: values on the BFP grid pack and unpack exactly (byte count = the device tile size)."""
    bits = BITS[fmt]
    gen = torch.Generator().manual_seed(1)
    tiles = bfp_quantize(torch.randn(6, TILE, TILE, generator=gen) * 0.1, bits)
    tiles[0, 3] = 0.0  # an all-zero face row
    raw = pack_bfp(tiles, bits)
    assert raw.shape == (6, tile_bytes(bits))
    assert torch.equal(unpack_bfp(raw, bits), tiles)
    assert tile_bytes(3) == 576 and tile_bytes(7) == 1088


def test_tile_transpose_is_involution():
    """Host only: tile_transpose transposes each tile and is its own inverse."""
    w = torch.randn(64, 96)
    assert torch.equal(tile_transpose(tile_transpose(w)), w)
    assert torch.equal(tile_transpose(w)[32:64, 64:96], w[32:64, 64:96].T)


@pytest.mark.parametrize("fmt, gu_extra, d_extra", [("bf4", 0, 128), ("bf8", 64, 0)])
def test_page_table_reads_back_every_tile(fmt, gu_extra, d_extra):
    """Host only: through the built table, tile (e, column, k) of every role is exactly that tile's bytes, the columns
    are spread over several banks, and the width-sharded tensor holds each bank's bytes contiguously."""
    bits = BITS[fmt]
    tb = tile_bytes(bits)
    gen = torch.Generator().manual_seed(3)
    E, k_gu, n_gu, n_d, banks = 3, 4, 3, 5, 8
    stored = _random_stored(E, k_gu, n_gu, n_d, bits, gen)
    gu_stride, d_stride = tb + gu_extra, tb + d_extra
    gu_bytes, d_bytes, table = _expert_streams(stored, bits, gu_stride, d_stride, banks, gen)
    assert table.shape == (E, 2 * (2 * n_gu + n_d))
    for e in range(E):
        for role, base, n_cols, src, stride, k_tiles in (
            ("gate", 0, n_gu, gu_bytes, gu_stride, k_gu),
            ("up", n_gu, n_gu, gu_bytes, gu_stride, k_gu),
            ("down", 2 * n_gu, n_d, d_bytes, d_stride, n_gu),
        ):
            expect = column_tile_bytes(stored[e][role], bits)
            for n in range(n_cols):
                for k in range(k_tiles):
                    got = read_via_table(src, table, e, base + n, k, stride, tb)
                    assert torch.equal(got, expect[n, k]), (e, role, n, k)
                    tile = unpack_bfp(got.unsqueeze(0), bits)[0]
                    assert torch.equal(tile, stored[e][role][k * TILE : (k + 1) * TILE, n * TILE : (n + 1) * TILE])
    assert len(set(table[:, 0::2].flatten().tolist())) > 1, "expected columns on several banks"
    rows = width_sharded_rows(gu_bytes)
    for b in range(banks):
        assert torch.equal(rows[:, b * 64 : (b + 1) * 64].reshape(-1), gu_bytes[b])


# ------------------------------------------------------------------------------------------------------ device test


def _swigluoai_ffn(x, w, b):
    """Clamped SwiGLU-OAI expert FFN with biases (gpt-oss), fp32. Weights (in, out)."""
    gate = (x @ w["gate"] + b["gate"]).clamp(max=SWIGLU_LIMIT)
    up = (x @ w["up"] + b["up"]).clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    return ((up + 1.0) * gate * torch.sigmoid(SWIGLU_ALPHA * gate)) @ w["down"] + b["down"]


SINGLE_CHIP_MESH_PARAMS = (
    [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")] if ttnn is not None else []
)


def _run_weight_pages(mesh_device, fmt, transposed, counts, max_tokens, indexed=False):
    """indexed: x is a token-ordered ROW_MAJOR input read through x_row_index (buffer row r = x row index[r]) and the
    output is ROW_MAJOR bf16 (the gpt-oss prefill call); otherwise x is the ROW_MAJOR dispatch buffer itself."""
    bits = BITS[fmt]
    wdtype = ttnn.bfloat4_b if fmt == "bf4" else ttnn.bfloat8_b
    tb = tile_bytes(bits)
    E = len(counts)
    banks = mesh_device.dram_grid_size().x
    gen = torch.Generator().manual_seed(17)

    # stored = the matrices the device tiles (tile-transposed when `transposed`); logical (in, out) weights from them
    stored = _random_stored(E, K_GU, N_GU, N_D, bits, gen)
    logical = [{r: tile_transpose(s) if transposed else s for r, s in st.items()} for st in stored]
    biases = [
        {
            "gate": (torch.randn(N_GU * TILE, generator=gen) * 0.5).bfloat16().float(),
            "up": (torch.randn(N_GU * TILE, generator=gen) * 0.5).bfloat16().float(),
            "down": (torch.randn(EMB, generator=gen) * 0.5).bfloat16().float(),
        }
        for _ in range(E)
    ]
    rows_total = E * max_tokens
    if indexed:
        x_in = torch.randn(512, EMB, generator=gen).bfloat16().float()
        row_index = torch.randint(x_in.shape[0], (rows_total,), generator=gen)
        x = x_in[row_index]  # buffer row r
    else:
        x_in = x = torch.randn(rows_total, EMB, generator=gen).bfloat16().float()

    dram = ttnn.DRAM_MEMORY_CONFIG

    def dev(t, dtype, layout):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh_device, memory_config=dram)

    def idx(values):
        return dev(torch.tensor(values, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    offsets, tt_counts, global_idx = idx([e * max_tokens for e in range(E)]), idx(counts), idx(list(range(E)))
    tt_x = dev(x_in, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    bias_lists = {
        r: [dev(biases[e][r].reshape(1, -1), ttnn.bfloat16, ttnn.TILE_LAYOUT) for e in range(E)]
        for r in ("gate", "up", "down")
    }
    common = dict(
        activation=ttnn.RoutedExpertActivation.SwiGluOai,
        gate_biases=bias_lists["gate"],
        up_biases=bias_lists["up"],
        down_biases=bias_lists["down"],
        weights_tile_transposed=transposed,
    )
    if indexed:
        common.update(x_row_index=idx(row_index.tolist()), output_row_major=True)

    # 1) per-expert weight tensors (DRAM interleaved)
    w_lists = {r: [dev(stored[e][r], wdtype, ttnn.TILE_LAYOUT) for e in range(E)] for r in ("gate", "up", "down")}
    out_tensors = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        tt_x,
        offsets,
        tt_counts,
        global_idx,
        w_lists["gate"],
        w_lists["up"],
        w_lists["down"],
        max_dispatched_tokens_per_expert=max_tokens,
        **common,
    )
    out_tensors = ttnn.to_torch(out_tensors).float().reshape(-1, EMB)
    for ws in w_lists.values():
        for w in ws:
            ttnn.deallocate(w)

    # 2) the same tiles scattered over two width-sharded uint8 byte streams, read through the page table
    gu_stride, d_stride = tb, tb + 128  # gate/up tiles back to back (as the decode streams), down tiles with gaps
    gu_bytes, d_bytes, table = _expert_streams(stored, bits, gu_stride, d_stride, banks, gen)

    def stream(bank_bytes):
        rows = width_sharded_rows(bank_bytes)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
                (rows.shape[0], 64),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        return ttnn.from_torch(
            rows, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=mem
        )

    gu_src, d_src = stream(gu_bytes), stream(d_bytes)
    tt_table = dev(table.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    out_pages = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        tt_x,
        offsets,
        tt_counts,
        global_idx,
        [],
        [],
        [],
        max_dispatched_tokens_per_expert=max_tokens,
        weight_pages=tt_table,
        gate_up_weight_source=gu_src,
        down_weight_source=d_src,
        gate_up_k_stride=gu_stride,
        down_k_stride=d_stride,
        weights_n_gate_up=N_GU,
        weights_n_down=N_D,
        weights_k_gate_up=K_GU,
        weights_k_down=N_GU,
        weights_dtype=wdtype,
        **common,
    )
    out_pages = ttnn.to_torch(out_pages).float().reshape(-1, EMB)

    # rows outside the expert regions' counts are never written: compare the written rows only
    for e, c in enumerate(counts):
        rows = slice(e * max_tokens, e * max_tokens + c)
        assert torch.equal(out_pages[rows], out_tensors[rows]), f"expert {e}: table-addressed output differs"
        if c == 0:
            continue
        with torch.no_grad():
            ref = _swigluoai_ffn(x[rows], logical[e], biases[e])
        assert torch.isfinite(out_pages[rows]).all(), f"expert {e}: NaN / Inf"
        _, pcc = comp_pcc(ref, out_pages[rows], 0.98)
        logger.info(f"{fmt} transposed={transposed} expert {e} ({c} tokens): PCC {pcc:.6f}")
        assert pcc >= 0.98, f"expert {e}: PCC {pcc}"


@pytest.mark.skipif(ttnn is None or not is_blackhole(), reason="unified_routed_expert op is Blackhole-only")
@pytest.mark.parametrize(
    "fmt, transposed",
    [("bf4", True), ("bf8", False), ("bf8", True)],
    ids=["bf4-transposed", "bf8", "bf8-transposed"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_weight_pages_match_weight_tensors(mesh_device, device_params, fmt, transposed):
    """4 experts with different token counts (one partial tile-row, one empty expert)."""
    _run_weight_pages(mesh_device, fmt, transposed, counts=[128, 45, 0, 256], max_tokens=256)


@pytest.mark.skipif(ttnn is None or not is_blackhole(), reason="unified_routed_expert op is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_weight_pages_32_local_experts(mesh_device, device_params):
    """gpt-oss-20b TP shape: 32 local experts, each a 768-wide quarter, bfloat4_b tile-transposed tiles, with the
    prefill call's x_row_index input and ROW_MAJOR output."""
    counts = [(37 * e) % 161 for e in range(32)]  # 0 .. 160 tokens, expert 0 empty
    _run_weight_pages(mesh_device, "bf4", True, counts=counts, max_tokens=160, indexed=True)
