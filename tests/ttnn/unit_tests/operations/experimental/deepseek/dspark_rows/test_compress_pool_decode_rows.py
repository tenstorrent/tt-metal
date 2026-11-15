# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``compress_pool_decode`` on a DSpark verify block of 4 rows (positions p .. p + 3 of one session) vs 4 one-row
calls at p .. p + 3, for compress ratios 1 and 2 and both parities of p, with every fused write (index key,
comp_kv, dense_kv, key_row) on: outputs and every cache must be bit-exact, and the RoPE'd rows / keys match a torch
reference. Plus device-time perf at production shapes (s = 1 vs s = 4):

    pytest tests/ttnn/unit_tests/operations/experimental/deepseek/dspark_rows/test_compress_pool_decode_rows.py -k "not perf"
    python -m tracy -r -p -v --no-web-server -t 8123 -o <out> -m pytest <this file> -k perf -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v4_flash.tt.decode.attention import make_rope_table

DH = 512
DK = 128
ROPE_DIM = 64
TP = 4
EPS = 1e-6
N_BLOCKS = 4
BLOCK_ROWS = 64
DENSE_ROWS = 512
ROWS = 4


def _dram(x, layout, dtype=ttnn.bfloat16, tile=None, device=None):
    return ttnn.from_torch(
        x, dtype=dtype, layout=layout, tile=tile, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _ints(vals, device):
    return _dram(torch.tensor(vals, dtype=torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.int32, device=device)


def _gathered(kv: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    """[n, Dh] kv (and gate) -> the TP-gathered rows [n, W]: per rank its Dh / TP kv columns, then its gate's."""
    if gate is None:
        return kv
    n, c = kv.shape[0], DH // TP
    return torch.stack([kv.reshape(n, TP, c), gate.reshape(n, TP, c)], dim=2).reshape(n, 2 * DH)


def _torch_rope(x, cos, sin):
    nope, rope = x[..., :-ROPE_DIM], x[..., -ROPE_DIM:]
    pairs = rope.reshape(*rope.shape[:-1], -1, 2)
    rotated = torch.stack([-pairs[..., 1], pairs[..., 0]], dim=-1).reshape_as(rope)
    return torch.cat([nope, rope * cos + rotated * sin], dim=-1)


def _rms(x, g):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g


class _State:
    """The session's device state the op updates: window (ratio 2) and the closed groups' caches."""

    def __init__(self, host: dict, device, ratio: int):
        self.win_kv = _dram(host["win_kv"], ttnn.TILE_LAYOUT, device=device) if ratio == 2 else None
        self.win_gate = _dram(host["win_gate"], ttnn.TILE_LAYOUT, device=device) if ratio == 2 else None
        self.idx_key_cache = _dram(host["idx_key_cache"], ttnn.TILE_LAYOUT, device=device)
        self.comp_kv = _dram(host["comp_kv"], ttnn.ROW_MAJOR_LAYOUT, device=device)
        self.dense_kv = _dram(host["dense_kv"], ttnn.TILE_LAYOUT, device=device)

    def torch(self) -> dict:
        out = {k: ttnn.to_torch(getattr(self, k)) for k in ("idx_key_cache", "comp_kv", "dense_kv")}
        if self.win_kv is not None:
            out["win_kv"] = ttnn.to_torch(self.win_kv)
            out["win_gate"] = ttnn.to_torch(self.win_gate)
        return out


def _setup(device, ratio: int, seed: int = 0, depth: int = 2):
    gen = torch.Generator().manual_seed(seed)

    def bf(*shape, scale=1.0, offset=0.0):
        return (offset + scale * torch.randn(*shape, generator=gen)).bfloat16().float()

    host = {
        "kv": bf(ROWS, DH),
        "gate": bf(ROWS, DH) if ratio == 2 else None,
        "win_kv": bf(1, 1, depth, DH),
        "win_gate": bf(1, 1, depth, DH),
        "norm": bf(1, 1, 1, DH, scale=0.1, offset=1.0),
        "index_wk": bf(DH, DK, scale=DH**-0.5),
        "index_norm": bf(1, 1, 1, DK, scale=0.1, offset=1.0),
        "idx_key_cache": bf(N_BLOCKS, 1, BLOCK_ROWS, DK),
        "comp_kv": bf(N_BLOCKS, 1, BLOCK_ROWS, DH),
        "dense_kv": bf(1, 1, DENSE_ROWS, DH),
        "angles": torch.rand(ROWS, ROPE_DIM // 2, generator=gen) * 6.28,
    }
    host["page_table"] = torch.stack([torch.arange(N_BLOCKS), torch.tensor([1, 3, 0, 2])]).to(torch.int32)
    dev = {
        "norm": _dram(host["norm"], ttnn.TILE_LAYOUT, device=device),
        "index_wk": _dram(host["index_wk"], ttnn.TILE_LAYOUT, device=device),
        "index_norm": _dram(host["index_norm"], ttnn.TILE_LAYOUT, device=device),
        "page_table": _dram(host["page_table"], ttnn.ROW_MAJOR_LAYOUT, ttnn.int32, device=device),
        "page_row": _ints([1], device),
    }
    return host, dev


def _rope_rows(host, idxs):
    """cos / sin [1, 1, n, Rd] of the closed groups (one random angle set per group)."""
    cs = [make_rope_table(host["angles"][i : i + 1].cos(), host["angles"][i : i + 1].sin()) for i in idxs]
    cos = torch.cat([c.reshape(1, ROPE_DIM) for c, _ in cs]).reshape(1, 1, -1, ROPE_DIM)
    sin = torch.cat([s.reshape(1, ROPE_DIM) for _, s in cs]).reshape(1, 1, -1, ROPE_DIM)
    return cos.bfloat16().float(), sin.bfloat16().float()


def _closing(ratio: int, p: int) -> list[int]:
    return [j for j in range(ROWS) if (p + j + 1) % ratio == 0]


def _call(
    kv,
    dev,
    st,
    ratio,
    interleave,
    win_slot,
    pool,
    cos=None,
    sin=None,
    entry=None,
    dense_entry=None,
    key_row=None,
    start_odd=False,
    index_key=True,
    prev_slot=None,
):
    kw = {}
    if ratio == 2:
        kw.update(gate=kv, win_kv=st.win_kv, win_gate=st.win_gate, win_slot=win_slot)
        if prev_slot is not None:
            kw["prev_slot"] = prev_slot
    if pool and not index_key:
        kw.update(cos=cos, sin=sin)
    elif pool:
        kw.update(
            cos=cos,
            sin=sin,
            index_wk=dev["index_wk"],
            index_norm_weight=dev["index_norm"],
            idx_key_cache=st.idx_key_cache,
            page_table=dev["page_table"],
            entry=entry,
            page_row=dev["page_row"],
            key_row=key_row,
            dense_kv=st.dense_kv,
            dense_entry=dense_entry,
            comp_kv=st.comp_kv,
        )
    if start_odd:
        kw["start_odd"] = True
    return ttnn.experimental.deepseek.compress_pool_decode(
        kv, dev["norm"], epsilon=EPS, do_pool=pool, kv_gate_interleave=interleave, **kw
    )


@pytest.mark.parametrize("cs_tile", ("32x32", "4x32"))
@pytest.mark.parametrize(
    "ratio, p, depth",
    (
        (1, 126, 0),  # entries 126..129: crosses a page-table block and a dense_kv tile
        (1, 61, 0),
        (2, 253, 2),  # odd: rows 0 (with the window's row) and 2 close -> entries 126, 127; row 3 stays open
        (2, 254, 2),  # even: rows 1 and 3 close -> entries 127, 128 (a block boundary)
        (2, 1025, 2),  # odd, entries past the 512 dense rows: dense_entry -1 skips dense_kv
        # DSpark's window: spec_window_rows (32) slots, slot = pos % 32, prev_slot = the slot of p - 1.
        (2, 253, 32),
        (2, 254, 32),
        (2, 31, 32),  # slots 31, 0, 1, 2 (wrap), row 0 pools with slot 30
        (2, 63, 64),  # slots 63, 0, 1, 2: the block's rows cross the window's tile rows
        (2, 62, 64),
    ),
)
def test_compress_pool_decode_rows(device, ratio, p, depth, cs_tile):
    host, dev = _setup(device, ratio, seed=ratio * 1000 + p, depth=max(depth, 2))
    deep = depth > 2
    prev = lambda q: _ints([q % depth], device) if deep else None  # slot of position q
    interleave = DH // TP if ratio == 2 else 0
    gathered = _gathered(host["kv"], host["gate"])  # [4, W]
    closing = _closing(ratio, p)
    n = len(closing)
    entries = [(p + j) // ratio for j in closing]
    entries = [e % (N_BLOCKS * BLOCK_ROWS) for e in entries]  # keep inside the test's 4 blocks
    dense_rows = [(p + j) // ratio if (p + j) // ratio < DENSE_ROWS else -1 for j in closing]
    win_slots = [(p + j) % max(depth, 2) for j in range(ROWS)]
    cos_h, sin_h = _rope_rows(host, range(n))

    # 4 one-row calls.
    seq = _State(host, device, ratio)
    seq_latent, seq_roped, seq_key = [], [], []
    key_row1 = _dram(torch.zeros(1, 1, 1, DK), ttnn.ROW_MAJOR_LAYOUT, device=device)
    for j in range(ROWS):
        kv1 = _dram(gathered[j].reshape(1, 1, 1, -1), ttnn.ROW_MAJOR_LAYOUT, device=device)
        pool = j in closing
        g = closing.index(j) if pool else None
        args = {}
        if pool:
            args = dict(
                cos=_dram(cos_h[:, :, g : g + 1], ttnn.TILE_LAYOUT, device=device),
                sin=_dram(sin_h[:, :, g : g + 1], ttnn.TILE_LAYOUT, device=device),
                entry=_ints([entries[g]], device),
                dense_entry=_ints([dense_rows[g]], device),
                key_row=key_row1,
            )
        outs = _call(
            kv1, dev, seq, ratio, interleave, _ints([win_slots[j]], device), pool, prev_slot=prev(p + j - 1), **args
        )
        if pool:
            seq_latent.append(ttnn.to_torch(outs[0]).reshape(1, DH))
            seq_roped.append(ttnn.to_torch(outs[1]).reshape(1, DH))
            seq_key.append(ttnn.to_torch(key_row1).reshape(1, DK))

    # One 4-row call.
    blk = _State(host, device, ratio)
    tile4 = ttnn.Tile((ROWS, 32))
    kv4 = _dram(gathered.reshape(1, 1, ROWS, -1), ttnn.TILE_LAYOUT, tile=tile4, device=device)
    cs_tile_arg = ttnn.Tile((4, 32)) if cs_tile == "4x32" else None
    key_row4 = _dram(torch.zeros(1, 1, n, DK), ttnn.ROW_MAJOR_LAYOUT, device=device)
    latent4, roped4 = _call(
        kv4,
        dev,
        blk,
        ratio,
        interleave,
        _ints(win_slots, device),
        True,
        cos=_dram(cos_h, ttnn.TILE_LAYOUT, tile=cs_tile_arg, device=device),
        sin=_dram(sin_h, ttnn.TILE_LAYOUT, tile=cs_tile_arg, device=device),
        entry=_ints(entries, device),
        dense_entry=_ints(dense_rows, device),
        key_row=key_row4,
        start_odd=p % 2 == 1,
        prev_slot=prev(p - 1),
    )
    assert list(latent4.shape) == [1, 1, n, DH] and list(roped4.shape) == [1, 1, n, DH]
    latent4 = ttnn.to_torch(latent4).reshape(n, DH)
    roped4 = ttnn.to_torch(roped4).reshape(n, DH)
    key4 = ttnn.to_torch(key_row4).reshape(n, DK)

    # Bit-exact vs the one-row calls.
    assert torch.equal(latent4, torch.cat(seq_latent)), "latent rows differ from the one-row calls"
    assert torch.equal(roped4, torch.cat(seq_roped)), "roped rows differ from the one-row calls"
    assert torch.equal(key4, torch.cat(seq_key)), "key rows differ from the one-row calls"
    s_seq, s_blk = seq.torch(), blk.torch()
    for name in s_seq:
        assert torch.equal(s_seq[name], s_blk[name]), f"{name} differs from the one-row calls"

    # vs torch, per closed group.
    kv, gate = host["kv"], host["gate"]
    for g, j in enumerate(closing):
        if ratio == 2:
            stored = (p - 1) % depth if deep else win_slots[0] ^ 1
            k0 = kv[j - 1] if j > 0 else host["win_kv"][0, 0, stored]
            g0 = gate[j - 1] if j > 0 else host["win_gate"][0, 0, stored]
            x = kv[j] + torch.sigmoid(g0 - gate[j]) * (k0 - kv[j])
        else:
            x = kv[j]
        latent = _rms(x, host["norm"].reshape(DH))
        roped = _torch_rope(latent, cos_h[0, 0, g], sin_h[0, 0, g])
        key = _torch_rope(
            _rms(latent @ host["index_wk"], host["index_norm"].reshape(DK)), cos_h[0, 0, g], sin_h[0, 0, g]
        )
        _, pcc_r = comp_pcc(roped, roped4[g], 0.9999)
        _, pcc_k = comp_pcc(key, key4[g], 0.999)
        logger.info(f"ratio {ratio} p {p} group {g} (row {j}): roped PCC {pcc_r:.6f}, key PCC {pcc_k:.6f}")
        assert pcc_r >= 0.9999 and pcc_k >= 0.999
    if ratio == 2:
        # The window ends as the one-row steps leave it: slot win_slots[j] holds the last row j written there.
        exp, exp_g = host["win_kv"].clone(), host["win_gate"].clone()
        for j in range(ROWS):
            exp[0, 0, win_slots[j]] = kv[j]
            exp_g[0, 0, win_slots[j]] = gate[j]
        assert torch.equal(s_blk["win_kv"], exp) and torch.equal(s_blk["win_gate"], exp_g)


@pytest.mark.parametrize(
    "mode",
    ("r2_s1_pool", "r2_s1_nopool", "r2_s4_even", "r2_s4_odd", "r1_s1", "r1_s4", "r1_s1_nokey", "r1_s4_nokey"),
)
def test_compress_pool_decode_rows_perf(device, mode):
    """Production shapes (TP4 rank): ratio 2 reads the gathered [kv | gate] row(s) (interleave 128), ratio 1
    the kv row(s); every fused write is on. Warmup, signpost, 20 timed calls."""
    from tracy import signpost

    ratio = 2 if mode.startswith("r2") else 1
    rows = 4 if "_s4" in mode else 1
    depth = 32  # DSpark's spec_window_rows: slot = pos % 32, prev_slot = the slot of p - 1
    host, dev = _setup(device, ratio, depth=depth)
    st = _State(host, device, ratio)
    interleave = DH // TP if ratio == 2 else 0
    gathered = _gathered(host["kv"], host["gate"])
    p = 255 if mode.endswith("odd") else 254
    index_key = not mode.endswith("nokey")
    if rows == 1:
        p = 255 if mode != "r2_s1_nopool" else 254
    closing = _closing(ratio, p)[:rows] if rows > 1 else ([0] if (p + 1) % ratio == 0 else [])
    n = len(closing)
    pool = n > 0
    if rows == 1:
        kv = _dram(gathered[0].reshape(1, 1, 1, -1), ttnn.ROW_MAJOR_LAYOUT, device=device)
        win_slot = _ints([p % depth], device)
    else:
        kv = _dram(gathered.reshape(1, 1, 4, -1), ttnn.TILE_LAYOUT, tile=ttnn.Tile((4, 32)), device=device)
        win_slot = _ints([(p + j) % depth for j in range(4)], device)
    args = {}
    if pool:
        cos_h, sin_h = _rope_rows(host, range(n))
        args = dict(
            cos=_dram(cos_h, ttnn.TILE_LAYOUT, device=device),
            sin=_dram(sin_h, ttnn.TILE_LAYOUT, device=device),
            entry=_ints([100 + g for g in range(n)], device),
            dense_entry=_ints([100 + g for g in range(n)], device),
            key_row=_dram(torch.zeros(1, 1, n, DK), ttnn.ROW_MAJOR_LAYOUT, device=device),
        )
        if rows > 1:
            args["start_odd"] = p % 2 == 1
        if not index_key:
            args = dict(cos=args["cos"], sin=args["sin"], index_key=False, start_odd=args.get("start_odd", False))
    args["prev_slot"] = _ints([(p - 1) % depth], device)
    outs = _call(kv, dev, st, ratio, interleave, win_slot, pool, **args)
    ttnn.synchronize_device(device)
    signpost(mode)
    for _ in range(20):
        for o in outs:
            ttnn.deallocate(o)
        outs = _call(kv, dev, st, ratio, interleave, win_slot, pool, **args)
    ttnn.synchronize_device(device)


@pytest.mark.parametrize("rows", (1, 4))
@pytest.mark.parametrize("ratio", (2, 1))
def test_compress_pool_decode_rows_program_cache(device, ratio, rows):
    """Steps with new slots / entries / RoPE rows (and a dense_entry of -1) reuse one program."""
    depth = 32
    host, dev = _setup(device, ratio, depth=depth)
    st = _State(host, device, ratio)
    interleave = DH // TP if ratio == 2 else 0
    gathered = _gathered(host["kv"], host["gate"])
    counts = []
    for step, p in enumerate((255, 259, 263)):  # same parity: one program
        n = len(_closing(ratio, p)) if rows > 1 else 1
        if rows == 1:
            kv = _dram(gathered[step].reshape(1, 1, 1, -1), ttnn.ROW_MAJOR_LAYOUT, device=device)
            win_slot = _ints([p % depth], device)
        else:
            kv = _dram(gathered.reshape(1, 1, 4, -1), ttnn.TILE_LAYOUT, tile=ttnn.Tile((4, 32)), device=device)
            win_slot = _ints([(p + j) % depth for j in range(4)], device)
        cos, sin = _rope_rows(host, [(step + g) % ROWS for g in range(n)])
        _call(
            kv,
            dev,
            st,
            ratio,
            interleave,
            win_slot,
            True,
            cos=_dram(cos, ttnn.TILE_LAYOUT, device=device),
            sin=_dram(sin, ttnn.TILE_LAYOUT, device=device),
            entry=_ints([40 * step + g for g in range(n)], device),
            dense_entry=_ints([(40 * step + g) if step != 1 else -1 for g in range(n)], device),
            key_row=_dram(torch.zeros(1, 1, n, DK), ttnn.ROW_MAJOR_LAYOUT, device=device),
            start_odd=rows > 1,
            prev_slot=_ints([(p - 1) % depth], device),
        )
        counts.append(device.num_program_cache_entries())
    logger.info(f"ratio {ratio} rows {rows}: program cache entries {counts}")
    assert counts[0] == counts[1] == counts[2], f"the op recompiled: {counts}"
