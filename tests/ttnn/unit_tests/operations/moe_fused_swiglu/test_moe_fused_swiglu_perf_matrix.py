# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""PERF MATRIX — two OPS x weight PLACEMENT against M, plus the wider cross on request.

    scripts/run_safe_pytest.sh --profile --run-all \
        tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_moe_fused_swiglu_perf_matrix.py
    ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/parse_perf_matrix.py \
        <out_prefix> <newest report dir> /tmp/moe_perf_matrix_manifest.json

THE DEFAULT MEASUREMENT is bf16 ROW-MAJOR activations, bfp4 weights, K (emb) 7168, N (hidden) 2048,
capacity 5120, 88 cores — the op's designed-for configuration, and also the `kimi_k26` shape the
routed-expert baseline was tuned against — swept over

    op             moe_fused_swiglu | routed_expert                  - the IMPLEMENTATION
    wplace         nd_shard | interleaved                            - the WEIGHT placement in DRAM
    count (M)      32 64 96 128 192 256 384 512 1024 2048 5120       - REAL tokens routed here

4 configurations x 11 M values. Every axis is still env-driven, so the wider cross this file was
built for is one variable away: `MOE_MATRIX_FORMATS=bf16_rm,bfp8_tile`,
`MOE_MATRIX_WDTYPES=bfp4,bfp8`, `MOE_MATRIX_HIDDEN=`, `MOE_MATRIX_EMBS=`, `MOE_MATRIX_COUNTS=`,
`MOE_MATRIX_OPS=`.

THE OP AXIS is what makes this a comparison rather than a self-report. `routed_expert` is
`ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn` — the standard DeepSeek-prefill
routed-expert FFN, ported from `mbezulj/2607-routed-expert-dram`. It computes the same
`silu(x@Wg) * (x@Wu) @ Wd` on the same 88 cores from the same tensors, so the two columns are
directly comparable. Three things are matched deliberately so the difference measured is the
implementation and not the harness:

  * GRID. The routed expert HARDCODES an 11x8 N-parallel grid (`kMaxGridX` / `MAX_GRID_Y` in its
    program factory) and takes no grid argument, so `GRID` below is set to the same (11, 8).
  * OUTPUT DTYPE. Both write bfp8 TILE DRAM. `moe_fused_swiglu` defaults to that; the routed
    expert would otherwise emit x's dtype (bf16, i.e. 2x the write bytes), so it is handed an
    explicit pre-allocated bfp8 output — which is also what `unified_routed_expert_moe` does in
    production.
  * ONE PROGRAM PER CONFIGURATION. Both read the token count DEVICE-side and size their own
    M-work from it, so all eleven M values run the same compiled program in both columns.

WHERE THEY DIFFER ARCHITECTURALLY, and it shows up in the `util` denominator: `moe_fused_swiglu`
holds all three weight sets L1-RESIDENT for the whole dispatch (see the L1 table below), so it
reads them from DRAM once. The routed expert chunks M into at most 32 tile-rows and RE-READS the
full weight set once per chunk (`adaptive_chunk::num_chunks`), which at capacity 5120 is 5 reads
of 24.8 MB. `read_bytes` is therefore op-aware; a shared denominator would have flattered it.

K IS PINNED TO 7168 BY DEFAULT rather than sweeping 7168 and 6144, because the op axis doubled the
configuration count and the dispatch budget below is what affords REPS 3 (and therefore the
run-to-run spread column that short-M numbers need to be readable at all). `MOE_MATRIX_EMBS=6144`
measures the other contraction as its own session.

ONE TEST PER CONFIGURATION, with M swept inside it, and that shape is load-bearing rather than
cosmetic: `count` is DEVICE-resident (the kernels read `counts[idx[local_expert_id]]` themselves) and
`input_m_tiles` is left at its default `capacity / 32`, so all eleven M values run the SAME compiled
program. Splitting M into pytest cells would buy 44 JIT builds and eleven tensor uploads per config
for no extra coverage, and would put a host-side allocation between dispatches that are supposed to
be back to back.

THE WEIGHT DTYPE IS AN L1 AXIS, NOT JUST A NUMERICS ONE, which is why N 2048 is a bfp4-only default.
Weight CBs are resident, so a wider dtype costs proportionally. Measured against the device's real
budget (1 461 376 B = `get_max_worker_l1_unreserved_size()` 1 532 032 minus `L1_CB_RESERVE` 70 656):

    K 7168  N 2048   bfp4 bf16_rm   1 445 504 B   fits, 15 872 B spare
    K 7168  N 2048   bfp4 bfp8_tile 1 359 872 B   fits
    K 7168  N 2048   bfp8 bf16_rm   1 562 240 B   OVER by 100 864 B  (W_down residency already off)
    K 7168  N 2048   bfp8 bfp8_tile 1 476 608 B   OVER by  15 232 B
    K 6144  N 2048   bfp4 both      1 262 592 .. 1 335 680 B   fit, >= 125 696 B spare
    K 6144  N 2048   bfp8 both      1 354 752 .. 1 427 840 B   fit, >=  33 536 B spare
    K 7168  N 1024   all four       997 184 .. 1 219 520 B     fit, >= 241 856 B spare

So the bfp8 arm cannot be measured at K 7168 / N 2048 at all, and a weight-dtype comparison needs
either `MOE_MATRIX_HIDDEN=1024` (both dtypes at the graded K) or `MOE_MATRIX_EMBS=6144` (both dtypes
at the graded N). Any (K, N) may be passed; a cell the op refuses SKIPS carrying the op's own byte
numbers, so the hole is reported rather than silently dropped or mistaken for a crash.

GRID SELECTION. This op's `core_grid` is (x, y) = (COLUMNS, ROWS), so 8 rows of 11 is written
(11, 8). Omitting `core_grid` uses the complete compute-with-storage grid. This benchmark explicitly
pins (11, 8); the transposed spelling (8, 11) is a different geometry, not an alternate spelling.

WEIGHT PLACEMENT IS A CALLER'S CHOICE, NOT A KNOB, and it is worth up to 11 %. The op reads whatever
it is handed and takes the coalesced path only when `nd_shard_n_tiles` can prove a contiguous run;
an interleaved weight is silently CORRECT and merely takes the uncoalesced one-request-per-tile
stream. That is exactly why a placement axis needs a self-check — `assert_placement` asserts the
READER's own predicate in BOTH directions, so a config that failed to shard cannot be reported as a
legitimate nd_shard number, and an "interleaved" arm cannot accidentally measure shards. The bytes
read are identical either way, so `read_bytes` (the util denominator) does not depend on placement.

EACH OP GETS ITS OWN ND-SHARD WIDTH, and handing one the other's would be a silent mismeasurement
rather than an error in only one direction. Both want the same SHAPE of shard — one tile-row tall,
on the DRAM bank grid, ROW_MAJOR, ROUND_ROBIN_1D, so consecutive K-rows rotate banks — because both
discovered the same thing: a core pinned to one bank saturates near 30 GB/s regardless of request
size, while the same bytes with the bank rotating reach ~370. They differ in WIDTH, because each
splits N across its grid differently, and the width is what makes a core's K-row slice exactly one
shard hence one NoC request:

    moe_fused_swiglu   gate/up `hn_pad`, down `ec_max`      - from its own Blocking()
    routed_expert      ceil(n_tiles / 11) for BOTH          - hidden and emb over its 11 columns

At K 7168 / N 2048 the routed expert's widths are [32, 192] for gate/up and [32, 672] for down. It
VALIDATES those host-side (`check_shard_width` in its program factory) and refuses a mismatch, so
this cannot silently degrade — but the failure would be a skipped cell, so the widths are derived
here from the same `ceil(n_tiles / GRID_X)` rule its factory uses rather than hardcoded.

Correctness is NOT asserted here beyond the output shape — that is the golden suite's job. This file
exists to produce one clean device-kernel duration per cell.

DISPATCH BUDGET. `count` is invisible in the profiler CSV, so the mapping comes from a manifest
written in dispatch order; the parser refuses to report if its length disagrees with the CSV's row
count. Keep a profiled session near ~150 dispatches: tracy's HOST-side `process_ops_logs` holds the
whole trace in pandas and needs ~50 MB of RSS per profiled dispatch. The default here is
4 configs x (1 warmup + 11 counts x 3 reps) = 136, which is what affords REPS 3 and therefore a real
run-to-run spread column. Widening any axis costs that headroom: adding back the second K, or a
second weight dtype, doubles to 272 and must drop to REPS 1 — and past ~150 the session must be
CHUNKED with `MOE_MATRIX_OPS` / `MOE_MATRIX_WDTYPES` / `MOE_MATRIX_WPLACES` / `MOE_MATRIX_FORMATS` /
`MOE_MATRIX_EMBS`, then every (report, manifest) pair passed to the parser at once.
"""

import json
import os

import pytest
import torch

import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import (
    nd_shard_n_tiles,
    weight_memory_configs,
)

TILE = 32
NUM_GLOBAL_EXPERTS, NUM_LOCAL_EXPERTS, LOCAL_EXPERT_ID, GLOBAL_EXPERT_ID = 256, 8, 3, 137

#: (x, y) = (COLUMNS, ROWS). This benchmark explicitly pins 8 rows of 11; callers may omit the
#: override to use the full compute-with-storage grid.
GRID = (11, 8)

HIDDEN = int(os.environ.get("MOE_MATRIX_HIDDEN", 2048))
CAPACITY = int(os.environ.get("MOE_MATRIX_CAPACITY", 5120))

#: The swept M — real tokens routed to the local expert. 32 is one tile-row (the shortest program the
#: op can run) and 5120 == capacity is the `full` fill bucket; 384 is the first value past
#: capacity/16, so both the `balanced` and the `partial` fill bucket are represented.
COUNTS = [int(c) for c in os.environ.get("MOE_MATRIX_COUNTS", "32,64,96,128,192,256,384,512,1024,2048,5120").split(",")]

_FORMATS = {"bf16_rm": (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), "bfp8_tile": (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)}
_WDTYPES = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}
_WPLACES = ("nd_shard", "interleaved")
_OPS = ("moe_fused_swiglu", "routed_expert")

#: The routed expert's N-parallel grid WIDTH, and the reason it is a literal here: its program
#: factory fixes `kMaxGridX = 11` and asserts the device is at least that wide rather than adapting,
#: so `per_core_N = ceil(n_tiles / 11)` — the ND-shard width it validates — is a constant of the op
#: and not a function of the grid this file asks for. It is checked against GRID below.
RE_GRID_X = 11

#: The routed expert's CB-sized MAXIMUM chunk in tile-rows (`kMaxChunkMTiles`, = per_core_M 4 x its
#: GRID_Y 8). The kernel picks the real chunk from the device-side count but never exceeds this, so
#: it is what sets how many times the weight set is re-read. See `read_bytes`.
RE_MAX_CHUNK_TILES = 32


def _axis(env, default):
    return [v.strip() for v in os.environ.get(env, default).split(",") if v.strip()]


#: K, the contraction dim — `emb` in the op's own vocabulary. A SWEPT axis, not a scalar, so both
#: supported values land in ONE profiled session and therefore one manifest: the two K are compared
#: against each other, and splitting them across sessions would put a device reset and a fresh
#: profiler sync between the halves of that comparison.
#: K, the contraction dim — `emb` in the op's own vocabulary. PINNED to 7168 by default now that the
#: op axis exists: two ops x two placements already spends the dispatch budget that buys REPS 3, and
#: 7168 is the shape both implementations were tuned at (`kimi_k26` on the routed-expert side). Pass
#: `MOE_MATRIX_EMBS=6144` for the other supported contraction, or `7168,6144` to put both in one
#: session at the cost of dropping to REPS 1.
EMBS = [int(e) for e in _axis("MOE_MATRIX_EMBS", "7168")]
FORMATS = _axis("MOE_MATRIX_FORMATS", "bf16_rm")
WPLACES = _axis("MOE_MATRIX_WPLACES", "nd_shard,interleaved")
WDTYPES = _axis("MOE_MATRIX_WDTYPES", "bfp4")
OPS = _axis("MOE_MATRIX_OPS", "moe_fused_swiglu,routed_expert")

REPS = int(os.environ.get("MOE_MATRIX_REPS", 3))
#: Warmup dispatches per configuration, excluded from the reported statistics. ONE IS THE MINIMUM,
#: not the default: the first dispatch of a configuration is where the JIT build lands and where the
#: op decides whether the cell fits L1, so it can neither be skipped nor counted.
WARMUP = max(1, int(os.environ.get("MOE_MATRIX_WARMUP", 1)))

#: Dispatches between `ttnn.ReadDeviceProfiler` drains. LOAD-BEARING under `--profile`: the op emits
#: ~125 zone records per core per dispatch against a 12 000-record device-side buffer, and an
#: overflow makes tracy ABORT report generation entirely ("Device data missing: Op N not present")
#: rather than emit a partial CSV.
READ_PROFILER_EVERY = int(os.environ.get("MOE_MATRIX_PROFILER_READ_EVERY", 3))

MANIFEST = os.environ.get("MOE_MATRIX_MANIFEST", "/tmp/moe_perf_matrix_manifest.json")

# Truncate at IMPORT — once per session, not per test. The eight configuration tests share one
# session and must APPEND to one manifest, but a manifest left by a PREVIOUS session must not
# survive into this one: stale entries would shift the whole CSV mapping by however many they are,
# and the parser's length check is what turns that into a refusal instead of a wrong curve.
if os.path.exists(MANIFEST):
    os.remove(MANIFEST)

CONFIGS = [(e, f, p, w, o) for e in EMBS for f in FORMATS for p in WPLACES for w in WDTYPES for o in OPS]


def re_num_chunks(count, max_chunk_tiles=RE_MAX_CHUNK_TILES):
    """How many times the routed expert reads the FULL weight set for `count` tokens.

    Mirrors `adaptive_chunk::num_chunks` exactly — a run of full `max_chunk_tiles` chunks then one
    tail chunk — because each chunk re-reads gate/up/down from DRAM. At the default capacity that is
    1 for every M up to 1024 and 5 at 5120 (160 tile-rows = 5 * 32).

    ASSUMES the op's L1 guard did not lower the CB-sized max below `RE_MAX_CHUNK_TILES`. All three
    bfp4 shapes in this matrix compile with per_core_M=4, so the max stands. On a larger shape where
    the guard did shrink it, this term would UNDER-count the weight re-reads and overstate `util` —
    the `us` columns, which are what this file exists to produce, are unaffected either way.
    """
    count_t = (count + TILE - 1) // TILE
    if count_t < 1:
        return 0
    full = count_t // max_chunk_tiles
    return full + (1 if count_t - full * max_chunk_tiles > 0 else 0)


def read_bytes(count, emb, hidden, w_tile, input_format, op="moe_fused_swiglu"):
    """DRAM bytes the op must read for one dispatch: the weight sets + ONE read of the real tokens.

    THE WEIGHT TERM IS OP-DEPENDENT, and that is the one place these two implementations are not
    reading the same bytes. `moe_fused_swiglu` keeps all three weight sets L1-RESIDENT for the whole
    dispatch, so it reads them from DRAM exactly once and the term is count-independent. The routed
    expert chunks M to at most 32 tile-rows and re-reads the full set per chunk, so its term scales
    with `re_num_chunks`. Sharing one denominator would have quietly credited it with a utilisation
    it never achieved at M 5120, where it reads 5x24.8 MB rather than 24.8.

    Either way the term follows the weight dtype's tile size, which is why `w_tile` is a parameter —
    a denominator left at bfp4's 576 B would overstate a bfp8 run's utilisation by ~1.8x. The
    activation is read at its own format's granularity: row-major reads exactly the real rows, while
    a tiled input can only be read a whole tile-row at a time, so a count that is not a multiple of
    32 pays for the padding it shares a tile with.
    """
    weights = 3 * (emb * hidden // 1024) * w_tile
    if op == "routed_expert":
        weights *= re_num_chunks(count)
    if input_format == "bf16_rm":
        return weights + count * emb * 2.0
    return weights + ((count + TILE - 1) // TILE) * TILE * emb * 1.0625


def re_nd_shard_config(device, n_dim):
    """The ROUTED EXPERT's preferred weight placement for an N of `n_dim`.

    Same shard SHAPE as `moe_fused_swiglu`'s — one tile-row tall on the DRAM bank grid, ROW_MAJOR,
    the default ROUND_ROBIN_1D — so shard id is `k * shard_grid_n + gx` and consecutive K-rows land
    in different banks. The WIDTH is the op's own N split, `ceil(n_tiles / RE_GRID_X)`, which is what
    makes one core's K-row slice exactly one shard and therefore one NoC request. Its program
    factory validates this width and refuses a mismatch.
    """
    n_tiles = n_dim // TILE
    per_core_n = (n_tiles + RE_GRID_X - 1) // RE_GRID_X
    dram = device.dram_grid_size()
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=ttnn.Shape([TILE, per_core_n * TILE]),
            grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))]),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def weight_configs(device, emb, hidden, wplace, op):
    """`(gate_up, down)` memory configs for this op's preferred placement.

    The nd_shard arm is PER-OP: each implementation's shard width is its own N split (see the module
    docstring), and handing one op the other's width is either a host-side refusal or — worse — a
    number measured on a slower path than the column claims.
    """
    if wplace == "interleaved":
        return ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    if wplace != "nd_shard":
        raise ValueError(f"unknown wplace {wplace!r}; expected one of {_WPLACES}")
    if op == "moe_fused_swiglu":
        return weight_memory_configs(device, emb, hidden, core_grid=GRID)
    if op == "routed_expert":
        # gate/up are (emb, hidden) so their N is `hidden`; down is (hidden, emb) so its N is `emb`.
        return re_nd_shard_config(device, hidden), re_nd_shard_config(device, emb)
    raise ValueError(f"unknown op {op!r}; expected one of {_OPS}")


def assert_placement(tt_w, wplace):
    """Assert the READER's own predicate, in BOTH directions.

    An interleaved weight is silently CORRECT — just uncoalesced — so a placement that failed to
    apply would otherwise be reported as a legitimate number for the wrong path. Checking the
    reverse direction matters just as much: the `interleaved` arm exists to measure the uncoalesced
    stream, and it would be worthless if it were quietly handed shards.
    """
    widths = [nd_shard_n_tiles(w) for w in tt_w]
    if wplace == "nd_shard":
        assert all(w > 0 for w in widths), f"asked for nd_shard but the reader sees interleaved: {widths}"
    else:
        assert all(w == 0 for w in widths), f"asked for interleaved but the reader sees shards: {widths}"
    return widths


def _counts_tensor(count, device):
    counts = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    counts[GLOBAL_EXPERT_ID] = count
    return ttnn.from_torch(
        counts, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _idx_table(device):
    idx = torch.tensor([(11 + 37 * i) % NUM_GLOBAL_EXPERTS for i in range(NUM_LOCAL_EXPERTS)], dtype=torch.int32)
    idx[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID
    return ttnn.from_torch(
        idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _build_static(device, emb, input_format, wplace, wdtype_name, op):
    """x and the three weight sets — built ONCE per configuration, shared by every M.

    x is `randn` over all `capacity` rows with no per-count sentinel: one tensor serves every M value,
    and nothing here depends on the phantom rows' contents because this file asserts only the output
    shape. Whether the phantom rows are correctly ignored is the golden suite's job.

    The weight TENSORS are identical across ops — both want gate/up as (emb, hidden) and down as
    (hidden, emb) in TILE layout — so only the memory config varies by op.
    """
    torch.manual_seed(42)
    dt, lay = _FORMATS[input_format]
    x = torch.randn((1, 1, CAPACITY, emb), dtype=torch.float32).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=dt, layout=lay, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    del x

    w_dtype = _WDTYPES[wdtype_name]
    gate_up_mc, down_mc = weight_configs(device, emb, HIDDEN, wplace, op)
    tt_w = [
        ttnn.from_torch(
            torch.randn(s, dtype=torch.bfloat16),
            dtype=w_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mc,
        )
        for s, mc in (((emb, HIDDEN), gate_up_mc), ((emb, HIDDEN), gate_up_mc), ((HIDDEN, emb), down_mc))
    ]
    widths = assert_placement(tt_w, wplace)
    return tt_x, tt_w, widths


@pytest.mark.parametrize("emb, input_format, wplace, wdtype_name, op", CONFIGS)
def test_perf_matrix(device, emb, input_format, wplace, wdtype_name, op):
    grid = device.compute_with_storage_grid_size()
    if GRID[0] > int(grid.x) or GRID[1] > int(grid.y):
        pytest.skip(f"device grid is {grid.x}x{grid.y}, smaller than {GRID[0]}x{GRID[1]}")
    if op == "routed_expert":
        # The routed expert takes no grid argument — it fixes its own 11x8. Comparing its number to
        # a moe_fused_swiglu number measured on a DIFFERENT grid would be meaningless, so require
        # the two to agree rather than silently reporting them side by side.
        assert GRID == (RE_GRID_X, 8), (
            f"routed_expert hardcodes an {RE_GRID_X}x8 grid; GRID is {GRID[0]}x{GRID[1]}, so the two "
            f"op columns would not be measured on the same cores"
        )
        if input_format != "bf16_rm":
            pytest.skip(f"routed_expert column is the x_is_row_major path; {input_format} not measured here")

    w_tile = ttnn.tile_size(_WDTYPES[wdtype_name])
    tt_x, tt_w, widths = _build_static(device, emb, input_format, wplace, wdtype_name, op)
    tt_idx = _idx_table(device)
    # Every counts tensor uploaded UP FRONT, so the timed loop contains nothing but the op and no
    # host-side allocation lands between two dispatches of the same cell.
    tt_counts = {c: _counts_tensor(c, device) for c in COUNTS}

    # The routed expert's output, pre-allocated ONCE for the same reason the counts are: it would
    # otherwise emit x's dtype (bf16 TILE) and pay 2x the write bytes, which is not what
    # moe_fused_swiglu does and not what `unified_routed_expert_moe` does in production. Both columns
    # therefore write bfp8 TILE DRAM. moe_fused_swiglu allocates its own output per call; that is a
    # HOST-side allocation and cannot enter DEVICE KERNEL DURATION, which is the reported quantity.
    tt_out = (
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, CAPACITY, emb]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        if op == "routed_expert"
        else None
    )

    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else []
    label = f"{op}/K{emb}/{input_format}/{wplace}/{wdtype_name}"

    def dispatch(count, rep, warmup):
        if op == "routed_expert":
            out = ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn(
                tt_x,
                tt_w[0],
                tt_w[1],
                tt_w[2],
                tt_counts[count],
                tt_idx,
                LOCAL_EXPERT_ID,
                output=tt_out,
                x_is_row_major=True,
            )
            assert list(out.shape) == [1, 1, CAPACITY, emb]
            # `out` IS tt_out (the op returns the buffer it was handed), so it must not be freed
            # here — the next dispatch of this cell writes into it again.
        else:
            out = moe_fused_swiglu(
                tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts[count], tt_idx, LOCAL_EXPERT_ID, core_grid=GRID
            )
            assert list(out.shape) == [1, 1, CAPACITY, emb]
            ttnn.deallocate(out)
        manifest.append(
            {
                "op": op,
                "format": input_format,
                "wplace": wplace,
                "weight_dtype": wdtype_name,
                "w_tile": w_tile,
                "grid": f"{GRID[0]}x{GRID[1]}",
                "emb": emb,
                "hidden": HIDDEN,
                "capacity": CAPACITY,
                "count": count,
                "rep": rep,
                "warmup": warmup,
                "read_bytes": read_bytes(count, emb, HIDDEN, w_tile, input_format, op),
            }
        )

    # The FIRST dispatch is where the op decides whether this cell fits L1, and it refuses with the
    # computed numbers rather than letting the allocator throw. A refusal is a documented hole in the
    # matrix, not a failure of this harness, so it is surfaced as a skip carrying the op's own
    # message — see the module docstring for which cells that is and why.
    try:
        dispatch(COUNTS[len(COUNTS) // 2], 0, True)
    except RuntimeError as e:
        if "L1 per core" not in str(e):
            raise
        pytest.skip(f"{label} does not fit L1 at K {emb} / N {HIDDEN} on {GRID[0]}x{GRID[1]}:\n{e}")
    for w in range(1, WARMUP):
        dispatch(COUNTS[len(COUNTS) // 2], w, True)
    ttnn.ReadDeviceProfiler(device)

    print(f"[matrix] {label} reader shard widths (N tiles/shard, 0=interleaved) = {widths}", flush=True)

    since_read = 0
    for count in COUNTS:
        for rep in range(REPS):
            dispatch(count, rep, False)
            since_read += 1
            if since_read >= READ_PROFILER_EVERY:
                ttnn.ReadDeviceProfiler(device)  # a no-op when the profiler is off
                since_read = 0
        print(
            f"[matrix] {label} K={emb} N={HIDDEN} cap={CAPACITY} M={count} "
            f"read_MB={read_bytes(count, emb, HIDDEN, w_tile, input_format, op) / 1e6:.3f} done",
            flush=True,
        )
    ttnn.ReadDeviceProfiler(device)

    json.dump(manifest, open(MANIFEST, "w"))
    print(f"[matrix] manifest: {MANIFEST} ({len(manifest)} dispatches)", flush=True)

    # Free the configuration's working set before the next one builds its own: every configuration
    # shares one module-scoped device, and x alone is up to 73 MB at K 7168 / capacity 5120.
    for t in (tt_x, *tt_w, tt_idx, *tt_counts.values(), *([tt_out] if tt_out is not None else [])):
        ttnn.deallocate(t)
