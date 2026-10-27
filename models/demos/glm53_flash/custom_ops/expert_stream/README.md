# expert_stream: the routed-expert block as DRAM-bank-streaming generic ops

The chip's slice of the hybrid placement (8x4 Galaxy: column c's 72 experts x row r's 256-wide intermediate slice;
gate|up `[72, 4096, 512]`, down `[72, 256, 4096]`, bfp8) for one 32-token block, run as two `ttnn.generic_op`
programs (`op.py`: `ExpertStream.__call__` / `.golden`, kernels in `kernels/`), after `models/demos/gpt_oss/tt/fused_decode/stream.py`. Evidence and
numbers: `doc/mesh/expert_kernel_sweep.md` section 9.

## Contract

| tensor | shape / dtype / layout | role |
|---|---|---|
| `x` | `[1, 1, 32, K=4096]` BF16 TILE (DRAM or L1 interleaved) | the block's tokens (M <= 32 rows, zero-padded) |
| `routing` | `[1, 1, 32, E=72]` BF16 TILE | the router's dense matrix restricted to the chip's experts: `w[t, e]` or 0 |
| `w13` | `[1, 1, rows13, 256]` BFP8 TILE, DRAM width-sharded over the 8 banks (`stream_memory_config`) | `gate_up_stream_layout`: bank `b` = (gate tile column `b`, up tile column `b`) of every expert, expert-major, K-contiguous |
| `w2` | `[1, 1, rows2, 256]` BFP8 TILE, same sharding | `down_stream_layout`: bank `b` = output tile columns `[16 b, 16 b + 16)` of every expert, column-major, K-contiguous |
| `act` | `[1, E, 32, I=256]` BF16 TILE DRAM | written by the gate|up op at the active experts' tiles: `D_e @ clamped_silu_glu(x W_g, x W_u)` |
| `out` | `[1, 1, 32, H=4096]` BF16 TILE DRAM | written by the down op: `sum_e act[e] @ W2[e]` (zeros where no expert is active) |

`act` and `out` are pre-allocated by the caller (`ExpertStream.alloc(device, act_l1)`), persistent or allocated
inside the trace; `act` in L1 interleaved (1.2 MB) and `x` in L1 are the defaults of the model path (~8 % faster).
The routing matrix is scanned by every kernel (reader and writer on the data-movement cores, the count reaches the
compute threads through the BRISC mailboxes), so the active-expert set is data-dependent at runtime and the calls
are trace-safe. On a mesh the same program runs on every chip with that chip's weights and gathered tokens.

## Dispatch ops (T > 32 gathered tokens, weights read once)

| tensor | shape / dtype / layout | role |
|---|---|---|
| `x_rm` | `[1, 1, T, K]` BF16 ROW_MAJOR (L1) | `to_layout(x, ROW_MAJOR)`: the writer gathers token rows (8 KB each) |
| `routing` | `[1, 1, T, E]` BF16 TILE | as above (T a multiple of 32, <= 256) |
| `counts` | `[1, 1, 1, E]` BF16 TILE | `routing_counts(routing) = sum_t (routing != 0)`: the per-expert row counts every kernel enumerates the passes from |
| `act` | `[1, P, 32, I]` BF16 TILE (L1) | pass k's activations (expert ascending, 32 routed rows per pass; `P = max_passes(T, E) = T/4 + E`) |
| `table` | `[1, 1, P, 64]` UINT16 ROW_MAJOR (L1) | pass k's `[n_rows, e, t_0 .. t_31]`, written by the gate|up op (bank 0), read by the down op |
| `out` | `[1, 1, T, H]` BF16 TILE DRAM | the expert sum per token |

`ExpertDispatch(device, E, K, I, H, T)`: `prepare(x, routing)` (the two helper ops), `alloc(device)`,
`forward(x_rm, routing, counts, w13, w2, act, table, out)`; `golden` = `expert_dispatch_golden` (per-pass act, y,
passes). The gate|up dispatch keeps an expert's (gate, up) pair resident over its passes with one pair per bank
(the 8x4) and re-streams per pass with more pairs; the down dispatch runs 8 readers per bank (2 output columns),
phase 1 = `h_k = act[k] @ W2[e][:, cols]` per pass into a resident BF16 buffer, phase 2 = per token row tile the
contributions (<= 8 per token) gathered into slot tiles and summed in fp32 DST (identity-tile matmuls, HiFi4).
Numbers: `doc/mesh/expert_kernel_sweep.md` section 10.

### EP mode (`ep=True`: orchestration v2, every chip owns `E / D` full-width experts)

The routing arrives as the top-k lists instead of a dense matrix: `routing` = the weight tiles `[1, 1, T, 32]` BF16
TILE, `counts` = the index tiles `[1, 1, T, 32]` BF16 TILE (global expert ids as exact BF16 integers, columns >= K
padded with `IDX_PAD`; `ep_index_tiles`), plus `base` = the chip's expert base id as a `[1, 1, 1, 16]` UINT16
ROW_MAJOR page (`ep_base_tensor`; sharded per chip on a mesh). Every kernel counts the rows of the chip's experts
`[base, base + E)` from the index list itself (no `ne` / `sum` helper ops), the writer gathers the rows and builds
`D` from the weights. `w13_dtype` selects bfp8 (1088 B tiles) or bfp4 (576 B) gate|up weights; the down weights
stay bfp8. `tt/moe.py` `TtEPExperts` (`GLM53_MOE_LAYOUT=ep`) wraps it with the mesh dispatch / combine.

## Kernels and core assignment

`ExpertGateUpStream(device, E, K, I, groups)`: `groups` worker cores per DRAM bank (default 12 -> 96 cores), placed
from the bank's optimal NOC0 worker core outwards; core (bank `b`, group `g`) streams the active experts
`g, g + groups, ...` (round-robin over the active list, so the groups stay balanced for any routing):

* `reader_gate_up.cpp` (NCRISC, NOC0): per expert the bank's gate column then up column (`K/32` tiles each, one
  contiguous range: `expert_stride = 2 * K/32 * 1088` bytes), 8-tile (8.7 KB) NOC packets, three column buffers
  with one column in flight ahead of compute (transaction ids).
* `writer_gate_up.cpp` (BRISC, NOC1): loads `x` (K/32 tiles) once, per expert builds `D_e = diag(w[:, e])` and
  writes the finished `[32, 32]` tile to `act` tile `(e, b)`.
* `compute_gate_up.cpp` (TRISC, HiFi2, fp32 DST): `g = x W_g`, `u = x W_u` (K/32 tile products into DST 0 / 1),
  `clamped_silu_glu_tile(0, 1, 2)` (the fused SFPU op, limit 10 = `swiglu_limit`), pack, then `D_e @ h`.

`ExpertDownStream(device, E, I, H, readers)`: `readers` cores per bank (default 2 -> 16 cores), each owning
`cols = 16 / readers` output tile columns (<= 8, the fp32 DST capacity):

* `reader_down.cpp`: per active expert the core's `cols` column segments (`I/32` tiles = 8.7 KB each, contiguous),
  three expert buffers with one in flight.
* `writer_down.cpp`: per active expert loads `act[e]` (8 tiles); writes the `cols` output tiles once at the end.
* `compute_down.cpp`: `out[:, c] += act[e] @ W2[e][:, c]` for every active expert with the `cols` output tiles
  resident in DST (fp32), packed once.

Wider per-chip slices (I = 1024 on a (2,4) block, 2048 on one chip): `ppb = I / 32 / banks` column pairs per bank
(the compute loops the pairs per expert, the expert's `D` tile shared), down segments of `I/32` K tiles read in
packets that divide the segment, readers per bank chosen so that an expert's block stays <= 128 tiles (2 x 8
columns at I = 256, 4 x 4 at 1024, 8 x 2 at 2048; `_down_readers`).

Dispatch kernels: `reader_gate_up_dispatch.cpp` (the LPT expert list, pairs as the streaming unit),
`writer_gate_up_dispatch.cpp` (row gather, `D` per pass, pass table), `compute_gate_up_dispatch.cpp` (tilize +
the block compute per pass), `reader_down_dispatch.cpp`, `writer_down_dispatch.cpp` (activations per pass, the
metadata page, the gather slots), `compute_down_dispatch.cpp`; `dispatch_common.hpp` (counts -> passes, LPT).

Compile-time args carry CB ids, tile counts, `E`, `M = 32`, the group count / columns per core, strides and the
`TensorAccessorArgs` of the interleaved tensors; runtime args carry buffer addresses, the bank id, the NOC virtual
channel and the core's group / column offset (`op.py`).

## Tests

`tests/custom_ops/test_expert_dispatch.py`: the dispatch ops at T = 64 / 256, I = 256 / 1024, skewed routing with
multi-pass experts, per-pass act and pass table vs the torch golden, the whole dispatch as a trace.
`tests/custom_ops/test_expert_stream.py`: random weights at the model shapes, 8 / 32 / 256 tokens (256 = 8 blocks),
2 or 8 routed experts per token, PCC >= 0.9999 of `act` and `out` vs the torch golden on bfp8-rounded weights, and
one block captured as a trace and replayed (device time per block). `GLM53_STREAM_GROUPS` /
`GLM53_STREAM_DOWN_READERS` set the core counts, `GLM53_STREAM_X_L1` / `GLM53_STREAM_ACT_L1` the placements,
`GLM53_STREAM_TRACE_PICKS` the routed experts per token of the trace test. The mesh variants ((1,4) / (2,4): each
chip its own expert slice, the tokens replicated) are gated on `GLM53_STREAM_MESH_TESTS=1` and have not been run
yet (no multi-chip mesh was opened in this pass). `tests/test_moe_block.py` runs the block through
`TtRoutedExperts(stream=True)` with real layer-4 weights and routing next to the sparse and dense paths.

## Results (chip 2 of the 8x4, Blackhole, traced; `doc/mesh/expert_kernel_sweep.md` section 9)

| case | us per 32-token block | note |
|---|---|---|
| real routing, 28 of 72 active (4 users/row), `test_moe_block.py` | **341** (372 with x / act in DRAM) | sparse block 683, dense 720; PCC 0.999926 vs torch, 0.999972 vs the sparse block |
| random routing, 43 of 72 active, unit test | 476 | 302 GB/s = 59 % of DRAM peak |
| random routing, 69 of 72 active, unit test | 707 | 326 GB/s = 64 % |
| real routing, 256 tokens (31 active) as 8 blocks | 301 x 8 = 2410 per layer | dense one-pass 1256: the block form re-streams the weights per block |

Best configuration: 12 groups (96 gate|up cores), 2 down readers per bank (16 cores), x and act in L1; 48 gate|up
cores are 1.28x slower, 120 bring nothing (the 8 DRAM banks are the limit), 4 down readers halve the contiguous
segment per core and lose. The target at M = 32 (<= 400 us) is met; the M = 256 target (<= 1.0 ms) needs the
per-expert dispatch form (gather the rows routed to each expert on device, stream each active expert once, fp32
row partials in L1): the design is in section 9 of the sweep doc, expected 0.6-0.8 ms per layer per chip.

## Streaming knobs (decode ledger item (d), 2026-10-26; `plan.py`)

`GLM53_EXPERT_PLAN=legacy` (default: today's program) | `explicit` (the knobs below over the legacy values) |
`roofline` (`plan.plan_expert_stream` for `GLM53_EXPERT_PLAN_GRID` (13x10), `GLM53_EXPERT_PLAN_ACTIVE` (expected
active experts), `GLM53_EXPERT_PLAN_T`, the formats, with the explicit knobs on top). Knobs -> `ExpertDispatch`
kwargs: `GLM53_STREAM_GROUPS` (cores per bank), `GLM53_STREAM_UNITS` (pair-unit LPT), `GLM53_STREAM_UNIT_PAIRS`
(`unit_pairs`: pairs per unit of the units kernels = `ppb / unit_pairs` cores per bank per expert; 1 = today's
units, 8 = one unit per expert at I = 2048), `GLM53_STREAM_PREFETCH` (`prefetch_pairs`: pairs in flight ahead of
the compute, ring = prefetch + 1 pair slots; bfp8 pairs at K = 4096 allow 1, bfp4 2-3), `GLM53_STREAM_PACKET_TILES`
(weight tiles per NOC packet), `GLM53_DISPATCH_DOWN_READERS`, `GLM53_STREAM_GATHER_ROTATE` (`gather_rotate`: every
gate|up core starts its row gather, and every down core its activation read, at a different row / tile: the H1
"convoy" hypothesis of the 2x slower units mode at few experts, `plan.fit_gather_convoy`). Kernel compile-time args:
readers 12 = ring slots, 13 = pairs per unit; gate|up writers 16 = rotate, 17 = pairs per unit; down writer 20 =
rotate; units compute 7 = pairs per unit (`dispatch_common.hpp::for_each_unit_chunk`, `rotated_row`). None of them
changes a result (reordered reads, a different work split): `tests/test_moe_roofline_mesh.py::test_stream_knobs_chip0`
asserts bit identity per variant and logs the us table that calibrates `plan.py`.

## Knobs

`GLM53_MOE_LAYOUT=ep` (the EP32 expert path, `TtEPExperts`), `GLM53_EXPERT_W13_DTYPE=bfp4` (C12: the routed gate|up
format over the model's, applied in `TtMoE` and in `TtMoE.expert_cache_complete`; the tensorbin names carry the
dtype), `GLM53_EP_PLACEMENT=<placement.json>` (item (e): the expert -> chip placement from `plan.plan_ep_placement`,
the router's id table, cache tag `_p<hash>`), `GLM53_STREAM_DISPATCH` (1: 32 < T <= 256 gathered tokens go through the dispatch ops, 0: 32-token blocks),
`GLM53_DISPATCH_DOWN_READERS` (8). `GLM53_STREAM_EXPERTS=1` selects the stream block in `tt/moe.py` (`TtRoutedExperts(stream=True)`, weight cache tag
`_stream`: the bank-major layouts are built once from the per-chip slices and uploaded width-sharded over the DRAM
banks through `upload_cached(..., rows=0, cols=1, memory_config=stream_memory_config(rows))`); `GLM53_STREAM_GROUPS`
(12), `GLM53_STREAM_DOWN_READERS` (2), `GLM53_STREAM_X_L1` (1), `GLM53_STREAM_ACT_L1` (1).

## Regions-sourced weights (`regions_src=True`, 2026-10-27; `regions.py`, `doc/mesh/regions_decode.md`)

Every op takes a `RegionsSource(gate, up, down, banks)` -- the regions prefill path's per-expert DRAM-interleaved
`[K, I]` / `[K, I]` / `[I, H]` lists (`tt/moe.py` `_build_dispatch_path`) -- in place of the stacked `w13` / `w2`,
and reads them in place: a DRAM-interleaved tile tensor puts tile column `n` in bank `n % banks` at the K stride
`ppb = I/32/banks` tiles, so at `ppb = 1` (the 8x4) the gate|up stream is byte-for-byte the stacked one from a per-expert
base address (`REGIONS_SRC` kernel variants; the addresses as the readers' common runtime args: gate `0..E-1`, up
`E..2E-1`, down `0..E-1`); the down block is read as `seg_tiles` packets of the core's `cols` tiles (k-major; the
down computes index `k * cols + c`); the writers map `(bank, q)` to act column `bank + banks * q` and the down core's
columns to `bank + banks * (r * cols + c)` (`kernels/regions_src.hpp`; the down writers take `[h_tiles, banks]` /
`[banks]` as compile-time args after the accessor args). Results are bit-identical to the stacked ops. `ppb > 1`
reads single-tile gate|up packets; the pair-unit kernels are stacked-only. The planner `regions.plan_regions_stream`
(bank locality: `I/32` and `H/32` multiples of the bank count -- not Wormhole's 12 at these widths) sizes the packets;
`tests/test_regions_decode_host.py` checks its address walk against the page map, `tests/custom_ops/test_expert_stream.py
::test_expert_stream_regions_src` the A/B on a chip. Model knob: `GLM53_REGIONS_DECODE=off | stream | regions`
(`tt/moe.py`): `stream` serves decode (block op) and 33..256 gathered tokens (dispatch ops) from the regions lists --
the one-copy configuration, no stacked upload; `regions` forces the regions kernel (A/B).
