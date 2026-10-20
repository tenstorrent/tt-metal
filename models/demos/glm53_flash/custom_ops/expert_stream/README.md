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

## Knobs

`GLM53_STREAM_DISPATCH` (1: 32 < T <= 256 gathered tokens go through the dispatch ops, 0: 32-token blocks),
`GLM53_DISPATCH_DOWN_READERS` (8). `GLM53_STREAM_EXPERTS=1` selects the stream block in `tt/moe.py` (`TtRoutedExperts(stream=True)`, weight cache tag
`_stream`: the bank-major layouts are built once from the per-chip slices and uploaded width-sharded over the DRAM
banks through `upload_cached(..., rows=0, cols=1, memory_config=stream_memory_config(rows))`); `GLM53_STREAM_GROUPS`
(12), `GLM53_STREAM_DOWN_READERS` (2), `GLM53_STREAM_X_L1` (1), `GLM53_STREAM_ACT_L1` (1).
