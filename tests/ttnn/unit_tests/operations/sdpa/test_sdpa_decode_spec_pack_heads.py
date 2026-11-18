# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""``spec_pack_heads`` paged SDPA decode against the unpacked ``spec_multi_pos_tiles`` call.

Both calls take the same Q ([1, G, Tg*32, DH], candidate j of group g on row-tile j, its NH heads on rows 0..NH-1),
the same paged K/V cache, page table, positions and program config. The packed call puts a group's Tg x NH q rows
into one tile. The valid output rows (rows 0..NH-1 of each candidate row-tile) must be equal byte for byte, and the
packed rows must be close to an fp32 torch reference. The padding rows of Q hold random values: neither call may let
them reach a valid row.

Every page table has spare columns past the last position: the spec SDPA reads whole K chunks, which can run past
a row's last written block.
"""

import math
import time

import pytest
import torch
from loguru import logger

import ttnn

pytestmark = pytest.mark.use_module_device

BLOCK = 64
HD = 256
TILE = 32
SPARE_BLOCKS = 4  # at least one 128-position dynamic K chunk of spare columns per page-table row

# Per group, the first candidate's position; candidate j of a group sits at base + j. Edges: candidates straddle a
# 32-row tile (28..31, 29..32, 30..33), a 128-position chunk edge (124..127, 125..128, 127..130, which leaves
# candidate 0 of that group with a fully masked last chunk) and the first positions (0..3, 1..4).
EDGES = [127, 125, 28, 29, 30, 124, 1, 0]
MID = [2047, 4093, 517, 1000, 2048, 3001, 4094, 1919]

# (id, Tg, NH, groups, first positions, k_chunk_size); k_chunk_size 0 is the in-kernel dynamic chunk.
CASES = [
    # The Qwen3.8-27B TP=8 verify: Tg 4, 3 local q heads, one group per user.
    ("tp8_u8_edges", 4, 3, 8, EDGES, 0),
    ("tp8_u8_mid", 4, 3, 8, MID, 0),
    ("tp8_u4_edges", 4, 3, 4, EDGES[:4], 0),
    ("tp8_u2_mid", 4, 3, 2, MID[:2], 0),
    ("tp8_u1_32k", 4, 3, 1, [32765], 0),
    ("tp8_u2_chunk128", 4, 3, 2, [125, 2047], 128),
    # 6 heads (TP=4): candidate 2's rows 12..17 cross row 16, the face boundary, so they move as two row runs.
    ("nh6_u2", 4, 6, 2, [29, 1021], 0),
    # Odd head counts: the stride rounds up to an even row count (5 -> 6, 1 -> 2).
    ("nh5_u2", 4, 5, 2, [30, 517], 0),
    ("nh1_u4", 4, 1, 4, EDGES[:4], 0),
    # Two candidates per group, 8 heads each: rows 0..7 and 8..15.
    ("tg2_nh8_u4", 2, 8, 4, [127, 31, 1000, 0], 0),
]


def _cfg(device, groups, k_chunk_size):
    """The verify's program config: the grid split across the groups, at most 64 cores per group."""
    grid = device.compute_with_storage_grid_size()
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        q_chunk_size=0,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
        max_cores_per_head_batch=min(64, max(1, (grid.x * grid.y) // groups)),
    )


class _Case:
    """Host and device inputs of one case. Group g owns nb shuffled blocks."""

    def __init__(self, device, Tg, NH, groups, bases, cache_dtype, seed, nb=None):
        g = torch.Generator().manual_seed(seed)
        self.Tg, self.NH, self.G = Tg, NH, groups
        self.nb = nb if nb is not None else math.ceil((max(bases) + Tg) / BLOCK) + SPARE_BLOCKS
        nblocks = groups * self.nb
        k = torch.randn(nblocks, 1, BLOCK, HD, generator=g).bfloat16()
        v = torch.randn(nblocks, 1, BLOCK, HD, generator=g).bfloat16()
        self.page_table = torch.randperm(nblocks, generator=g).reshape(groups, self.nb).to(torch.int32)
        self.pos = torch.tensor([bases[i] + j for i in range(groups) for j in range(Tg)], dtype=torch.int32)
        # Every row random, the padding rows NH..31 of each candidate row-tile included.
        self.q = torch.randn(1, groups, Tg * TILE, HD, generator=g).bfloat16()

        self.k_tt = ttnn.from_torch(k, dtype=cache_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.v_tt = ttnn.from_torch(v, dtype=cache_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.pt_tt = ttnn.from_torch(self.page_table, dtype=ttnn.int32, device=device)
        self.pos_tt = ttnn.from_torch(self.pos, dtype=ttnn.int32, device=device)
        self.q_tt = ttnn.from_torch(self.q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def call(self, cfg, pack, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            self.q_tt,
            self.k_tt,
            self.v_tt,
            page_table_tensor=self.pt_tt,
            cur_pos_tensor=self.pos_tt,
            scale=HD**-0.5,
            program_config=cfg,
            memory_config=memory_config,
            spec_multi_pos_tiles=self.Tg,
            spec_pack_heads=self.NH if pack else 0,
        )

    def valid_rows(self, out_tt):
        """[1, G, Tg*32, HD] -> [G, Tg, NH, HD] bf16: rows 0..NH-1 of every candidate row-tile."""
        out = ttnn.to_torch(out_tt)
        ttnn.deallocate(out_tt)
        return out.reshape(self.G, self.Tg, TILE, HD)[:, :, : self.NH, :].contiguous()

    def torch_reference(self):
        """fp32 attention of each candidate over [0, its position], from the cache as stored (bf8 rounding included)."""
        k_dev = ttnn.to_torch(self.k_tt).float()
        v_dev = ttnn.to_torch(self.v_tt).float()
        q = self.q.reshape(self.G, self.Tg, TILE, HD)[:, :, : self.NH, :].float()
        ref = torch.zeros(self.G, self.Tg, self.NH, HD)
        for gi in range(self.G):
            blocks = self.page_table[gi].long()
            k_u = k_dev[blocks].reshape(-1, HD)
            v_u = v_dev[blocks].reshape(-1, HD)
            for j in range(self.Tg):
                p = int(self.pos[gi * self.Tg + j])
                s = (q[gi, j] @ k_u[: p + 1].T) * HD**-0.5
                ref[gi, j] = torch.softmax(s, dim=-1) @ v_u[: p + 1]
        return ref

    def free(self):
        for t in (self.k_tt, self.v_tt, self.pt_tt, self.pos_tt, self.q_tt):
            ttnn.deallocate(t)


def _pcc(a, b):
    a = a.flatten().double() - a.double().mean()
    b = b.flatten().double() - b.double().mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(torch.equal(a, b))


def _assert_same_bytes(unpacked, packed, where):
    same = unpacked.view(torch.int16) == packed.view(torch.int16)
    assert bool(same.all()), (
        f"{where}: packed differs from unpacked in {int((~same).sum())} of {same.numel()} values "
        f"(max abs {(unpacked.float() - packed.float()).abs().nan_to_num().max()})"
    )


@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bf8", "bf16"])
@pytest.mark.parametrize("cid, Tg, NH, groups, bases, k_chunk_size", CASES, ids=[c[0] for c in CASES])
def test_spec_pack_heads_matches_unpacked(device, cid, Tg, NH, groups, bases, k_chunk_size, cache_dtype):
    case = _Case(device, Tg, NH, groups, bases, cache_dtype, seed=len(cid))
    cfg = _cfg(device, groups, k_chunk_size)
    unpacked = case.valid_rows(case.call(cfg, pack=False))
    packed = case.valid_rows(case.call(cfg, pack=True))
    _assert_same_bytes(unpacked, packed, cid)

    ref = case.torch_reference()
    for gi in range(groups):
        pcc = _pcc(packed[gi].float(), ref[gi])
        assert pcc > 0.99, f"{cid} group {gi} (first position {bases[gi]}): packed vs torch PCC {pcc:.5f}"
    case.free()


def test_spec_pack_heads_program_cache(device):
    """Packed and unpacked are two programs (spec_pack_heads is in the program hash); a second set of tensors of the
    same shapes (new addresses, new positions) hits both, and the outputs still match."""
    nb = math.ceil((max(MID) + 4) / BLOCK) + SPARE_BLOCKS + 1  # a cache shape no other test uses
    cfg = _cfg(device, 8, 0)
    first = _Case(device, 4, 3, 8, EDGES, ttnn.bfloat8_b, seed=7, nb=nb)
    before = device.num_program_cache_entries()
    unpacked = first.valid_rows(first.call(cfg, False))
    after_unpacked = device.num_program_cache_entries()
    packed = first.valid_rows(first.call(cfg, True))
    entries = device.num_program_cache_entries()
    assert after_unpacked == before + 1 and entries == after_unpacked + 1, (before, after_unpacked, entries)
    _assert_same_bytes(unpacked, packed, "miss")

    second = _Case(device, 4, 3, 8, MID, ttnn.bfloat8_b, seed=8, nb=nb)
    first.free()
    _assert_same_bytes(second.valid_rows(second.call(cfg, False)), second.valid_rows(second.call(cfg, True)), "hit")
    assert device.num_program_cache_entries() == entries, "same shapes with new tensors compiled a new program"
    second.free()


def test_spec_pack_heads_l1_output(device):
    """The Qwen3.8 verify writes the attention output to L1 interleaved."""
    case = _Case(device, 4, 3, 8, MID, ttnn.bfloat8_b, seed=9)
    cfg = _cfg(device, 8, 0)
    unpacked = case.valid_rows(case.call(cfg, False, memory_config=ttnn.L1_MEMORY_CONFIG))
    packed = case.valid_rows(case.call(cfg, True, memory_config=ttnn.L1_MEMORY_CONFIG))
    _assert_same_bytes(unpacked, packed, "L1 output")
    case.free()


@pytest.mark.parametrize(
    "Tg, NH, match",
    [(4, 9, "do not fit one"), (2, 17, "must be <=")],
    ids=["rows_do_not_fit", "heads_past_face"],
)
def test_spec_pack_heads_refuses_a_shape_that_does_not_pack(device, expect_error, Tg, NH, match):
    case = _Case(device, Tg, NH, 2, [100, 200], ttnn.bfloat8_b, seed=3)
    with expect_error(RuntimeError, match):
        case.call(_cfg(device, 2, 0), pack=True)
    case.free()


def test_spec_pack_heads_refuses_without_spec_mode(device, expect_error):
    case = _Case(device, 4, 3, 2, [100, 200], ttnn.bfloat8_b, seed=4)
    q_rows = ttnn.reshape(case.q_tt, (1, 8, TILE, HD))  # one candidate row-tile per batch row
    pt_rows = ttnn.from_torch(case.page_table.repeat_interleave(4, dim=0), dtype=ttnn.int32, device=device)
    with expect_error(RuntimeError, "requires spec_multi_pos_tiles"):
        ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_rows,
            case.k_tt,
            case.v_tt,
            page_table_tensor=pt_rows,
            cur_pos_tensor=case.pos_tt,
            scale=HD**-0.5,
            program_config=_cfg(device, 8, 0),
            spec_pack_heads=3,
        )
    ttnn.deallocate(pt_rows)
    case.free()


@pytest.mark.parametrize("first", [1990, 32765], ids=["2k", "32k"])
def test_spec_pack_heads_latency(device, first):
    """Per-call time at the 8-user TP=8 verify shape, unpacked and packed, calls back to back (reported, not a gate)."""
    case = _Case(device, 4, 3, 8, [first - 8 * i for i in range(8)], ttnn.bfloat8_b, seed=11)
    cfg = _cfg(device, 8, 0)
    n = 50
    times = {False: [], True: []}
    for pack in (False, True, False, True):
        for _ in range(3):
            ttnn.deallocate(case.call(cfg, pack, memory_config=ttnn.L1_MEMORY_CONFIG))
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.deallocate(case.call(cfg, pack, memory_config=ttnn.L1_MEMORY_CONFIG))
        ttnn.synchronize_device(device)
        times[pack].append((time.perf_counter() - t0) / n * 1e6)
    logger.info(
        f"spec SDPA, 8 groups x Tg 4 x 3 heads near position {first}: unpacked "
        f"{[round(t, 1) for t in times[False]]} us, packed {[round(t, 1) for t in times[True]]} us per call"
    )
    case.free()
