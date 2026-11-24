# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device checks of ``custom_ops/sdpa_decode_splitfree`` (the split-independent paged decode attention):

* ``test_splitfree_matches_torch``: PCC against a torch reference over random paged caches and page tables, S in
  {1K, 4K, 8K} / {32K, 128K} / {256K} (one case per group), B in {1, 8, 32}, K in {32, 64}, rows at the unit
  transitions (``n_chunks`` in {K, K+1, 2K, 2K+1}); the stock op's PCC logged beside it.
* ``test_splitfree_gqa_and_window``: two KV groups at 16K, the sliding window and the bounded ring, both merges;
  per merge (and unit) the combine split never changes a bit (CT in {1, 2, 4, DHt}).
* ``test_splitfree_row_bits_independent_of_batch_grid_ct_and_q``: a row's bytes are the same for B in
  {1, 4, 8, 16, 32}, on the device grid and an 8x8 grid, for CT in {1, 2, 4, DHt} (under both merges and the
  legacy unit too) and with a DRAM or an L1 height-sharded q; the stock op at the Galaxy pin (3 cores per head)
  keeps the row's bytes across B as well.
* ``test_splitfree_timing_vs_stock``: traced device time per call against the stock op at 16 and 32 cores per
  head, S from 1K to GEMMA4_SDPA_TEST_CACHE, B in {1, 8, 32}, the legacy op, K in {32, 64}, NSUB_MIN 2, both merges
  and READS 32 / 64; asserts the default's (K = 32, stats merge) envelope against the stock op.
* ``test_splitfree_partials_vs_torch``: where an error lives (the unit partials and both merges in torch).

Gated like ``tests/unit/test_attention.py::test_decode_sdpa_core_split_long_context``: ``GEMMA4_SDPA_TEST_CACHE``
must be set (the longest S, e.g. 262144), so the long device runs never start by accident:

    TT_VISIBLE_DEVICES=<chip> GEMMA4_SDPA_TEST_CACHE=262144 pytest models/demos/gemma4/tests/test_sdpa_decode_splitfree.py
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.custom_ops.sdpa_decode_splitfree import SdpaDecodeSplitfree

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]
SINGLE_CHIP_TRACE = [
    pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED, "trace_region_size": 64_000_000}, id="single-chip")
]
CACHE_ENV = "GEMMA4_SDPA_TEST_CACHE"
LADDER_S_ENV = "GEMMA4_SDPA_LADDER_S"  # comma list: the ladder's lengths (default: LADDER_S up to the cache)
LADDER_VARIANTS_ENV = "GEMMA4_SDPA_LADDER_VARIANTS"  # comma list of LADDER_VARIANTS names (default: all)
BLOCK = 64
PCC_MIN = 0.999
# the op's knobs (planner.py)
SPLIT_ENV = "GEMMA4_SDPA_SPLITFREE_SPLIT"
MIN_CHUNKS_ENV = "GEMMA4_SDPA_SPLITFREE_MIN_CHUNKS"
UNIT_ENV = "GEMMA4_SDPA_SPLITFREE_UNIT"
COMBINE_ENV = "GEMMA4_SDPA_SPLITFREE_COMBINE"
COMBINE_TILES_ENV = "GEMMA4_SDPA_SPLITFREE_COMBINE_TILES"
READS_ENV = "GEMMA4_SDPA_SPLITFREE_READS"
LEGACY = {UNIT_ENV: "4096", COMBINE_ENV: "pairwise"}  # the op before the position-derived unit (same bits)
# the scratch of a case covers the default, K = 64 and the legacy unit
SCRATCH_ENVS = ({}, {SPLIT_ENV: "64"}, {UNIT_ENV: "4096"})
LADDER_S = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
LADDER_VARIANTS = {
    "stock16": None,
    "stock32": None,
    "legacy": LEGACY,
    "k32": {SPLIT_ENV: "32"},
    "k32-pairwise": {SPLIT_ENV: "32", COMBINE_ENV: "pairwise"},
    "k64": {SPLIT_ENV: "64"},
    "k64-min2": {SPLIT_ENV: "64", MIN_CHUNKS_ENV: "2"},
    "k32-reads32": {SPLIT_ENV: "32", READS_ENV: "32"},
    "k32-reads64": {SPLIT_ENV: "32", READS_ENV: "64"},
}

gated = pytest.mark.skipif(
    not os.environ.get(CACHE_ENV),
    reason=f"long-context device checks of sdpa_decode_splitfree: set {CACHE_ENV}=<tokens> to run them",
)


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _transition_positions(split, S, chunk=256):
    """``cur_pos`` below ``S`` at the unit-size transitions of ``K = split``: the first and last position of
    ``n_chunks`` in {K, K+1, 2K, 2K+1} (the unit grows from ``n_chunks = K + 1`` and ``2K + 1`` on)."""
    out = []
    for n in (split, split + 1, 2 * split, 2 * split + 1):
        out += [p for p in ((n - 1) * chunk, n * chunk - 1) if p < S]
    return out


class _Case:
    """One paged cache of ``S`` tokens on the device (random data, bfloat8_b), its torch mirror and the scratch."""

    def __init__(self, mesh_device, S, H_q, H_kv, D, dtype=ttnn.bfloat8_b, seed=0):
        torch.manual_seed(seed)
        self.device, self.S, self.H_q, self.H_kv, self.D = mesh_device, S, H_q, H_kv, D
        self.num_blocks = S // BLOCK
        self.k_cache = ttnn.from_torch(
            torch.randn(self.num_blocks, H_kv, BLOCK, D, dtype=torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.v_cache = ttnn.from_torch(
            torch.randn(self.num_blocks, H_kv, BLOCK, D, dtype=torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # the reference reads what the device reads (the bfp8 rounding)
        self.k_ref = ttnn.to_torch(self.k_cache).float()
        self.v_ref = ttnn.to_torch(self.v_cache).float()
        specs = [(H_kv, H_q // H_kv, D, 1024), (H_kv, H_q // H_kv, D, None)]
        tiles = max(
            SdpaDecodeSplitfree.scratch_tiles(mesh_device, 32, specs, S, kv_dtype=dtype, env=env)
            for env in SCRATCH_ENVS
        )
        self.scratch = SdpaDecodeSplitfree.allocate_scratch(mesh_device, tiles)

    def inputs(self, B, cur_pos, tables, q, q_sharded=False):
        """Device tensors of one call: ``q [B, H_q, D]`` (the TILE layout pads the heads to 32 rows, as the
        model's Q arrives; ``q_sharded``: L1 height-sharded, one user per core, shard [32, D] -- the fused head
        ops' layout the op reads in place), ``cur_pos [B]``, ``tables [B, blocks]``."""
        tt_q = ttnn.from_torch(
            q.reshape(1, B, self.H_q, self.D), device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        if q_sharded:
            grid = self.device.compute_with_storage_grid_size()
            mem = ttnn.create_sharded_memory_config(
                shape=(32, self.D),
                core_grid=ttnn.num_cores_to_corerangeset(B, grid, True),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            sharded = ttnn.to_memory_config(tt_q, mem)
            tt_q.deallocate(True)
            tt_q = sharded
        tt_pos = ttnn.from_torch(
            cur_pos.to(torch.int32), device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
        )
        tt_tab = ttnn.from_torch(
            tables.to(torch.int32), device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
        )
        return tt_q, tt_pos, tt_tab

    def run(self, tt_q, tt_pos, tt_tab, **kw):
        return SdpaDecodeSplitfree.op(tt_q, self.k_cache, self.v_cache, tt_pos, tt_tab, self.scratch, scale=1.0, **kw)

    def golden(self, q, cur_pos, tables, **kw):
        return SdpaDecodeSplitfree.golden(
            q.reshape(1, -1, self.H_q, self.D), self.k_ref, self.v_ref, tables, cur_pos, scale=1.0, **kw
        )

    def random_tables(self, B, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.stack([torch.randperm(self.num_blocks, generator=g) for _ in range(B)])


def _to_rows(out, B, H_q):
    return ttnn.to_torch(out)[0, :B, :H_q, :].contiguous()


def _free(*tensors):
    for t in tensors:
        t.deallocate(True)


def _model_compute_config(mesh_device):
    """The compute config the model's stock decode SDPA takes (tt/attention/decode.py): HiFi2 with fp32 dest
    accumulation (the op default leaves the dest in bf16, which the long-context scores cannot afford)."""
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _stock(case, tt_q, tt_pos, tt_tab, cap, compute_kernel_config):
    """The stock paged decode SDPA on the case's caches at ``cap`` cores per head."""
    grid = case.device.compute_with_storage_grid_size()
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        q_chunk_size=32,
        k_chunk_size=64,
        exp_approx_mode=False,
        max_cores_per_head_batch=cap,
    )
    return ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_q,
        case.k_cache,
        case.v_cache,
        cur_pos_tensor=tt_pos,
        page_table_tensor=tt_tab,
        scale=1.0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=pc,
        compute_kernel_config=compute_kernel_config,
    )


def _traced_ms(mesh_device, call, n_calls=10, replays=10):
    """Traced device wall per call (ms): ``n_calls`` calls in one trace (one live output), ``replays`` replays,
    median of 3 (an eager call is host-bound: two ProgramDescriptors with per-core runtime args per call)."""
    out = call()
    ttnn.synchronize_device(mesh_device)
    out.deallocate(True)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    last = None
    for i in range(n_calls):
        o = call()
        if i < n_calls - 1:
            o.deallocate(True)
        else:
            last = o
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    means = []
    for _ in range(3):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(replays):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        means.append((time.perf_counter() - t0) / (replays * n_calls) * 1e3)
    ttnn.release_trace(mesh_device, tid)
    last.deallocate(True)
    return sorted(means)[1]


@gated
@pytest.mark.parametrize("lengths", [(1024, 4096, 8192), (32768, 131072), (262144,)], ids=["short", "mid", "long"])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_splitfree_matches_torch(mesh_device, device_params, lengths, reset_seeds):
    """PCC > 0.999 against torch, overall and per live row, for every S of the case's ``lengths``, K in {32, 64}
    and B in {1, 8, 32}: random caches, random page tables per row, row 0 at S-1, a one-token row, a padding row,
    rows at the unit transitions of K below S (where a row's bits may change with its position), the rest random;
    one kv head, 4 q heads, head_dim 512 (the 12B / 31B global layer). The stock op's PCC (cap 16, the model's
    compute config) is logged beside."""
    ckc = _model_compute_config(mesh_device)
    for S in lengths:
        _matches_torch_at(mesh_device, S, ckc)


def _matches_torch_at(mesh_device, S, ckc):
    case = _Case(mesh_device, S, H_q=4, H_kv=1, D=512)
    for split in (32, 64):
        for B in (1, 8, 32):
            q = torch.randn(B, 4, 512, dtype=torch.bfloat16)
            cur_pos = torch.randint(0, S, (B,))
            fixed = ([S - 1, 0] + _transition_positions(split, S))[: max(1, B - 1)]
            cur_pos[: len(fixed)] = torch.tensor(fixed)
            if B > 1:
                cur_pos[B - 1] = -1  # a padding row
            tables = case.random_tables(B, seed=B + split)
            tt_q, tt_pos, tt_tab = case.inputs(B, cur_pos, tables, q)
            out = case.run(tt_q, tt_pos, tt_tab, env={SPLIT_ENV: str(split)})
            got = _to_rows(out, B, 4)
            ref = case.golden(q, cur_pos, tables)[0]
            live = [b for b in range(B) if cur_pos[b] >= 0]
            pcc = _pcc(got[live], ref[live])
            rows = {int(cur_pos[b]): _pcc(got[b], ref[b]) for b in live}
            stock = _stock(case, tt_q, tt_pos, tt_tab, 16, ckc)
            stock_pcc = _pcc(_to_rows(stock, B, 4)[live], ref[live])
            worst = min(rows, key=rows.get)
            logger.info(
                f"S={S} K={split} B={B}: pcc {pcc:.6f} over {len(live)} live rows (worst row at {worst}: "
                f"{rows[worst]:.6f}); stock cap 16 {stock_pcc:.6f}"
            )
            assert pcc > PCC_MIN, f"S={S} K={split} B={B}: pcc {pcc:.5f}"
            assert rows[worst] > PCC_MIN, f"S={S} K={split} B={B}: row at {worst} pcc {rows[worst]:.5f}"
            _free(out, stock, tt_q, tt_pos, tt_tab)
    _free(case.k_cache, case.v_cache, case.scratch)


@gated
@pytest.mark.parametrize(
    "S, H_q, H_kv, window, modulo, positions",
    [
        (16384, 4, 2, None, None, [16383, 0, 255, 256, 8191, 8192, -1, 12345]),
        (4096, 4, 2, 1024, None, [4095, 0, 1023, 1024, 1025, 2500, -1, 3000]),
        (4096, 2, 1, 1024, 2048, [4095, 0, 1023, 1024, 1025, 2500, -1, 3000]),
    ],
    ids=["gqa", "gqa-window", "window-ring"],
)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_splitfree_gqa_and_window(mesh_device, device_params, S, H_q, H_kv, window, modulo, positions, reset_seeds):
    """Two KV groups (the head-row segments of group 1) at 16K, where every row past 4095 spans several units under
    both the position's unit (K = 32: 8192 is the first transition) and the legacy fixed 4096 unit; the sliding
    window (the window mask, one chunk per unit) and the bounded ring (the virtual tile wrap before the page-table
    lookup) at 4K; head_dim 256, both merges. Per merge and unit the combine split never changes a bit: CT in
    {1, 2, 4, DHt} O tiles per job give identical live rows (the bit gate), at PCC_MIN against torch."""
    D = 256
    case = _Case(mesh_device, S, H_q=H_q, H_kv=H_kv, D=D)
    B = len(positions)
    q = torch.randn(B, H_q, D, dtype=torch.bfloat16)
    cur_pos = torch.tensor(positions)
    # with the ring every row's positions wrap into its first modulo / BLOCK blocks
    tables = case.random_tables(B, seed=7)
    live = [b for b in range(B) if positions[b] >= 0]
    kw = dict(num_kv_heads=H_kv, block_size=BLOCK, sliding_window=window, cache_position_modulo=modulo)
    ref = case.golden(q, cur_pos, tables, **kw)[0]
    tt_q, tt_pos, tt_tab = case.inputs(B, cur_pos, tables, q)
    # a windowed layer runs one chunk per unit whatever the unit knobs
    units = {"k32": {SPLIT_ENV: "32"}} if window else {"k32": {SPLIT_ENV: "32"}, "legacy": {UNIT_ENV: "4096"}}
    for combine in ("stats", "pairwise"):
        for unit_name, unit_env in units.items():
            bits = {}
            for ct in (1, 2, 4, D // 32):
                env = {**unit_env, COMBINE_ENV: combine, COMBINE_TILES_ENV: str(ct)}
                out = case.run(tt_q, tt_pos, tt_tab, env=env, **kw)
                bits[ct] = _to_rows(out, B, H_q)[live].view(torch.int16).clone()
                out.deallocate(True)
            pcc = _pcc(bits[1].view(torch.bfloat16), ref[live])
            logger.info(
                f"S={S} H_q={H_q} H_kv={H_kv} window={window} ring={modulo} {combine} {unit_name}: "
                f"CT {sorted(bits)} pcc {pcc:.6f}"
            )
            assert pcc > PCC_MIN, f"{combine} {unit_name}: pcc {pcc:.5f}"
            for ct in bits:
                assert torch.equal(bits[ct], bits[1]), f"{combine} {unit_name}: CT={ct} differs from CT=1"
    _free(tt_q, tt_pos, tt_tab)


@gated
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_splitfree_row_bits_independent_of_batch_grid_ct_and_q(mesh_device, device_params, reset_seeds, monkeypatch):
    """A row's output bytes (row 0 at S-1, then at 8192, the first K = 32 transition) are identical across the
    padded batch B in {1, 4, 8, 16, 32} (the other rows random live or padding), the device grid vs
    GEMMA4_TEST_GRID=8x8 (the unit follows the row's position, never the grid: the core mapping changes, the bits
    do not), CT in {1, 2, 4, DHt} and a DRAM vs an L1 height-sharded q (read in place), under the default (K = 32,
    the stats merge); CT never changes them under the pairwise merge and the legacy fixed unit either (the bit gate
    on one KV group at head_dim 512). The stock op at the Galaxy pin (3 cores per head) also gives the row the same
    bytes in every padded batch (the pin's purpose; the custom op needs no pin). S = GEMMA4_SDPA_TEST_CACHE."""
    S = int(os.environ.get(CACHE_ENV, "131072"))
    case = _Case(mesh_device, S, H_q=4, H_kv=1, D=512)
    ckc = _model_compute_config(mesh_device)
    q0 = torch.randn(4, 512, dtype=torch.bfloat16)
    table0 = case.random_tables(1, seed=11)[0]
    gen = torch.Generator().manual_seed(12)

    def row0(B, pos0, env=None, q_sharded=False, stock_cap=None):
        q = torch.randn(B, 4, 512, dtype=torch.bfloat16, generator=gen)
        q[0] = q0
        cur_pos = torch.randint(0, S, (B,), generator=gen)
        cur_pos[torch.rand(B, generator=gen) < 0.5] = -1  # padding rows among the live ones
        cur_pos[0] = pos0
        tables = case.random_tables(B, seed=100 + B)
        tables[0] = table0
        tt_q, tt_pos, tt_tab = case.inputs(B, cur_pos, tables, q, q_sharded=q_sharded)
        if stock_cap:
            out = _stock(case, tt_q, tt_pos, tt_tab, stock_cap, ckc)
        else:
            out = case.run(tt_q, tt_pos, tt_tab, env=env or {})
        bits = _to_rows(out, B, 4).view(torch.int16).clone()[0]
        _free(out, tt_q, tt_pos, tt_tab)
        return bits

    for pos0 in (S - 1, 8192):
        if pos0 >= S:
            continue
        bits = {("B", B): row0(B, pos0) for B in (1, 4, 8, 16, 32)}
        for ct in (1, 2, 4, 16):
            bits[("CT", ct)] = row0(8, pos0, env={COMBINE_TILES_ENV: str(ct)})
        for B in (1, 8, 32):
            bits[("sharded q", B)] = row0(B, pos0, q_sharded=True)
        monkeypatch.setenv("GEMMA4_TEST_GRID", "8x8")
        bits[("8x8", 1)] = row0(1, pos0)
        bits[("8x8", 32)] = row0(32, pos0)
        monkeypatch.delenv("GEMMA4_TEST_GRID")
        ref = case.golden(q0.unsqueeze(0), torch.tensor([pos0]), table0.unsqueeze(0))[0, 0]
        pcc = _pcc(bits[("B", 1)].view(torch.bfloat16), ref)
        logger.info(f"row at {pos0}: pcc {pcc:.6f}; {len(bits)} runs")
        assert pcc > PCC_MIN, f"row at {pos0}: pcc {pcc:.5f}"
        for key, b in bits.items():
            assert torch.equal(b, bits[("B", 1)]), f"row at {pos0}: {key} differs from the B=1 run"
        for variant, venv in (
            ("pairwise", {COMBINE_ENV: "pairwise"}),
            ("legacy", LEGACY),
            ("legacy stats", {UNIT_ENV: "4096"}),
        ):
            vbits = {ct: row0(8, pos0, env={**venv, COMBINE_TILES_ENV: str(ct)}) for ct in (1, 2, 4, 16)}
            for ct, b in vbits.items():
                assert torch.equal(b, vbits[1]), f"row at {pos0}: {variant} CT={ct} differs from CT=1"
        stock3 = {B: row0(B, pos0, stock_cap=3) for B in (1, 8, 32)}
        for B, b in stock3.items():
            assert torch.equal(b, stock3[1]), f"row at {pos0}: stock cap 3 differs at B={B}"


@gated
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP_TRACE, indirect=True)
def test_splitfree_timing_vs_stock(mesh_device, device_params, reset_seeds):
    """Traced device time per call (both programs of the op in one trace; ``_traced_ms``) against the stock op at
    16 (served) and 32 cores per head under the model's compute config, for S in LADDER_S up to
    GEMMA4_SDPA_TEST_CACHE (GEMMA4_SDPA_LADDER_S overrides) and the shapes B=1, B=8 / 32 with row 0 at S-1 and the
    others padding ("lone") or all rows at S-1 ("full"); one page table as wide as the cache (the served tables
    are max_seq_len wide, so one program per variant and batch). Variants (GEMMA4_SDPA_LADDER_VARIANTS narrows
    them): the legacy op (4096 unit, pairwise merge), K = 32 / 64, K = 64 with NSUB_MIN 2, the pairwise merge at
    K = 32, READS 32 / 64. Every time is logged before the envelope of the default (``k32``) against the stock op
    at cap 16 is asserted. A lone row -- the case the op exists for (tt-metal#59300) -- is no slower from 16K on
    and within 1.3x below (the combine program's fixed cost); one live row in a padded batch of 8 / 32 is no
    slower at any S; with every row live the op stays within 1.45x (the fp32 partials' DRAM traffic). Measured on
    one Blackhole Galaxy chip at 11212cc05e9 (2026-11-24): lone B=1 1.23x at 1K, 1.05x at 8K, 0.91x at 16K, 0.62x at
    256K; lone B=8 / 32 at most 0.84x; full B=8 1.08-1.25x, full B=32 1.09-1.36x (accepted on the served numbers:
    12B P300X2, release benchmarks, TPOT +2-4 % at 8 users)."""
    S_max = int(os.environ[CACHE_ENV])
    lengths = [int(s) for s in os.environ.get(LADDER_S_ENV, "").split(",") if s.strip()]
    lengths = lengths or [s for s in LADDER_S if s <= S_max]
    names = [n.strip() for n in os.environ.get(LADDER_VARIANTS_ENV, "").split(",") if n.strip()]
    names = names or list(LADDER_VARIANTS)
    assert all(n in LADDER_VARIANTS for n in names), f"{LADDER_VARIANTS_ENV}: one of {list(LADDER_VARIANTS)}"
    assert max(lengths) <= S_max, f"{LADDER_S_ENV} reaches past the {S_max}-token cache"
    case = _Case(mesh_device, S_max, H_q=4, H_kv=1, D=512)
    ckc = _model_compute_config(mesh_device)
    times = {}
    for B, fill in ((1, "lone"), (8, "lone"), (8, "full"), (32, "lone"), (32, "full")):
        q = torch.randn(B, 4, 512, dtype=torch.bfloat16)
        tables = case.random_tables(B, seed=B)
        for S in lengths:
            cur_pos = torch.full((B,), S - 1) if fill == "full" else torch.tensor([S - 1] + [-1] * (B - 1))
            tt_q, tt_pos, tt_tab = case.inputs(B, cur_pos, tables, q)
            for name in names:
                if name.startswith("stock"):
                    call = lambda cap=int(name[len("stock") :]): _stock(case, tt_q, tt_pos, tt_tab, cap, ckc)
                else:
                    call = lambda env=LADDER_VARIANTS[name]: case.run(tt_q, tt_pos, tt_tab, env=dict(env))
                times[(B, fill, S, name)] = _traced_ms(mesh_device, call)
            _free(tt_q, tt_pos, tt_tab)
            logger.info(
                f"B={B:2d} {fill:4s} S={S:6d}: "
                + ", ".join(f"{n} {times[(B, fill, S, n)] * 1e3:7.1f} us" for n in names)
            )
    misses = []
    for (B, fill, S, name), t in sorted(times.items()):
        if name != "k32" or (B, fill, S, "stock16") not in times:
            continue
        stock = times[(B, fill, S, "stock16")]
        if fill == "lone":
            bound = 1.3 if (B == 1 and S <= 8192) else 1.0
        else:
            bound = 1.45
        logger.info(f"B={B:2d} {fill:4s} S={S:6d}: k32 / stock16 = {t / stock:.3f} (bound {bound})")
        if t > bound * stock:
            misses.append(f"B={B} {fill} S={S}: {t * 1e3:.1f} us vs stock16 {stock * 1e3:.1f} us (bound {bound}x)")
    assert not misses, "k32 leaves its envelope: " + "; ".join(misses)


@gated
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_splitfree_partials_vs_torch(mesh_device, device_params, reset_seeds):
    """Where an error lives, at S = GEMMA4_SDPA_TEST_CACHE, B=1, 4 heads, head_dim 512, row 0 at S-1 on a random
    page table: the unit partials (m, l, O; the group's head rows the partials writer keeps) read back from the
    scratch against a torch flash model of the same units on fp32 scores (m_dev - m_ref is the device's score
    error at the row max: a bf16 ulp is 0.5 there, a tf32 ulp 1/16), the device partials merged in torch both ways
    (the pairwise correction chain and the stats-first merge the op runs by default) against the output and the
    golden, the output against the golden (asserted at PCC_MIN), and the stock op against the golden under the
    model's compute config (HiFi2, fp32 dest accumulation) and the op default (bf16 dest)."""
    S = int(os.environ[CACHE_ENV])
    case = _Case(mesh_device, S, H_q=4, H_kv=1, D=512)
    q = torch.randn(1, 4, 512, dtype=torch.bfloat16)
    cur_pos = torch.tensor([S - 1])
    tables = case.random_tables(1, seed=1)
    tt_q, tt_pos, tt_tab = case.inputs(1, cur_pos, tables, q)
    out = case.run(tt_q, tt_pos, tt_tab)
    got = _to_rows(out, 1, 4)[0].float()
    ref = case.golden(q, cur_pos, tables)[0, 0]
    plan = SdpaDecodeSplitfree.plan(mesh_device, tt_q, case.k_cache, tt_pos, tt_tab)
    logger.info(f"S={S}: {plan.describe()}")
    blocks = tables[0].long()
    K = case.k_ref[blocks, 0].reshape(S, 512)
    V = case.v_ref[blocks, 0].reshape(S, 512)
    s = q[0].float() @ K.T  # [4, S] fp32 scores
    scratch = ttnn.to_torch(case.scratch)
    dm, lr, opcc, parts = [], [], [], []
    for u in plan.live_units(S - 1):
        page0 = plan.slot(0, 0, u) * plan.tiles_per_slot
        tile = lambda t: scratch[(page0 + t) * 32 : (page0 + t) * 32 + 4]  # the 4 head rows of page t
        m_dev, l_dev = tile(0)[:, 0], tile(1)[:, 0]
        O_dev = torch.cat([tile(2 + c) for c in range(16)], dim=1)
        chunks = plan.unit_chunks(u, S - 1)
        lo, hi = chunks.start * plan.chunk, chunks.stop * plan.chunk
        su = s[:, lo:hi]
        m_ref = su.amax(1)
        p = torch.exp(su - m_ref[:, None])
        O_ref = p @ V[lo:hi]
        dm.append(m_dev - m_ref)
        lr.append(l_dev / p.sum(1))
        opcc.append(_pcc(O_dev, O_ref))
        parts.append((m_dev, l_dev, O_dev))
    dm, lr = torch.stack(dm), torch.stack(lr)
    logger.info(
        f"S={S}: {len(parts)} units; m_dev - m_ref: max |dm| {dm.abs().max():.4f}, mean {dm.mean():+.4f}, "
        f"rms {dm.pow(2).mean().sqrt():.4f}; l_dev / l_ref: min {lr.min():.4f} max {lr.max():.4f}; "
        f"pcc(O_dev, O_ref): min {min(opcc):.6f} mean {sum(opcc) / len(opcc):.6f}"
    )
    logger.info(f"S={S}: unit 0 dm per head {dm[0].tolist()}, l ratio {lr[0].tolist()}")
    # the pairwise correction chain
    m, l, O = parts[0]
    for mu, lu, Ou in parts[1:]:
        mn = torch.maximum(m, mu)
        a, au = torch.exp(m - mn), torch.exp(mu - mn)
        l, O, m = l * a + lu * au, O * a[:, None] + Ou * au[:, None], mn
    pairwise = O / l[:, None]
    # the stats-first merge: M first, w_u = exp(m_u - M), L = sum w_u l_u, O = sum w_u O_u (scale 1)
    ms = torch.stack([p[0] for p in parts])
    w = torch.exp(ms - ms.amax(0))
    L = (w * torch.stack([p[1] for p in parts])).sum(0)
    stats = sum(w[u][:, None] * parts[u][2] for u in range(len(parts))) / L[:, None]
    logger.info(
        f"S={S}: pcc(out, golden) {_pcc(got, ref):.6f}; device partials merged in torch: pairwise vs golden "
        f"{_pcc(pairwise, ref):.6f}, stats-first vs golden {_pcc(stats, ref):.6f}; pcc(out, pairwise) "
        f"{_pcc(got, pairwise):.6f}, pcc(out, stats-first) {_pcc(got, stats):.6f}; per-head pcc(out, golden) "
        f"{[round(_pcc(got[h], ref[h]), 5) for h in range(4)]}"
    )
    for name, ckc in (("model cfg (HiFi2, fp32 dest)", _model_compute_config(mesh_device)), ("op default", None)):
        for cap in (3, 16):
            o = _stock(case, tt_q, tt_pos, tt_tab, cap, ckc)
            logger.info(f"S={S}: stock cap {cap:2d} {name}: pcc vs golden {_pcc(_to_rows(o, 1, 4)[0], ref):.6f}")
            o.deallocate(True)
    assert _pcc(got, ref) > PCC_MIN, f"S={S}: pcc {_pcc(got, ref):.5f}"
