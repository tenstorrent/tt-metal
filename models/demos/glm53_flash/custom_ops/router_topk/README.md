# `router_topk` -- fused exact-fp32 MoE router top-k for decode (model-local `ttnn.generic_op`)

One program for everything `tt/moe.py::TtRouter.forward` does after the gate matmul when T <= 32
(`GLM53_ROUTER_FUSED=1`): HF noaux_tc, n_group = 1, from the fp32 sigmoid scores `s` (the fused-sigmoid gate
matmul output, `GLM53_ROUTER_FUSED_SIGMOID=1`) and the fp32 correction bias:

    choice = s + bias ; idx = top-K(choice) ; w = s[idx] / sum(s[idx]) * route_scale

It replaces the 16-op composite (`ttnn.add`, FillPad, `ttnn.topk`, FillPad, slice, `ttnn.gather`, FillPad, sum,
rdiv, mul, typecast, 3 x untilize, scatter, tilize: 150.5 us device time on one core-bound chain in the
rows-4-5 block decode profile) with one 32-core program.

## Contract

| tensor | shape (logical) | dtype / layout / memory | role |
|---|---|---|---|
| `scores` | `[1,1,T<=32,E]` (padded `[32,E]`) | fp32 TILE interleaved | sigmoid gate scores (read) |
| `bias` | `[1,1,1,E]` (padded `[32,E]`) | fp32 TILE interleaved | `e_score_correction_bias`, row 0 (read; `TtRouter.bias`) |
| `dense` | `[1,1,T,E]` | bf16 TILE interleaved, pre-allocated | optional: `w` at the picked expert columns, 0 elsewhere |
| `w` | `[1,1,T,K]` (padded `[32,32]`) | fp32 or bf16 TILE interleaved, pre-allocated | optional: normalized weights, descending choice order |
| `idx` | `[1,1,T,K]` (padded `[32,32]`) | uint32 or fp32 (integral) TILE interleaved, pre-allocated | optional: expert ids in the same order |

EP layouts (decode ledger C10, 2026-10-26; `tests/test_moe_roofline_host.py`, `tests/test_moe_roofline_mesh.py`):

| tensor | shape | dtype / layout / memory | role |
|---|---|---|---|
| `x` (input) | `[1,1,T,H]` | bf16 TILE interleaved | the MoE input, untilized per token row by the reader kernel (NCRISC) |
| `xw` | `[T,1,1,H+K]` | bf16 ROW_MAJOR L1, pre-allocated (`allocate_xw`) | token row t = `[x[t] \| w[t] (bf16)]`: the `all_to_all_dispatch` input, no untilize / concat ops |
| `idx` (rows) | `[T,1,1,K]` | uint16 ROW_MAJOR L1 (`allocate_idx_rows`) | the dispatch metadata ids, no typecast / untilize ops |
| `idx` (tiles) | `[1,1,T,32]` | bf16 TILE (`allocate_idx_tiles`) | the single-chip EP op's index tiles: ids as bf16 integers, `IDX_PAD` (4096) beyond K |
| `perm` (input) | `[1,1,1,E]` | uint16 ROW_MAJOR | the EP placement's global -> slot table (`custom_ops/expert_stream/plan.py`), applied to the ids of every idx layout; the dense row keeps the global columns |

Writer compile-time args 10-15: `X_MODE`, `PERM_MODE`, `CB_XROW`, `H/32`, the xw page bytes (`2 (H + K)` aligned to
16), the bf16 bits of `IDX_PAD`; accessors `(scores, bias, dense, w, idx, xw, perm)` from 16; common runtime args
`[scores, bias, dense, w, idx, T, xw, perm]`. Reader (`kernels/reader_router_topk.cpp`, X_MODE only): 2 x H/32 reads
of 64 B (the face-row pair holding the row, 64 B aligned both sides), the row's halves copied into the contiguous
`2 H`-byte row of `CB_XROW`, the K-wide tail zeroed; the writer fills the tail with the bf16 weights and sends the
page. Rows `>= T` write nothing into the row-major pages (they have exactly T pages). Wiring:
`TtRouter.forward(ep_mode="rm" | "tiles")`, knobs `GLM53_ROUTER_EP_ROWS=1` / `GLM53_ROUTER_EP_TILES=1` in `TtMoE`.

`E % 32 == 0` (288), `K <= 16` (8), `T` is a common runtime arg: rows `T..31` (padded token rows) get zeros in
every output (the composite gives them the bias top-8 with real weights; the expert paths ignore / skip zero
rows). At least one output. Padding columns of `w` / `idx` (`K..31`) are written as zeros. `allocate_dense` /
`allocate_w` / `allocate_idx` give outputs in DRAM; under trace the outputs are allocated inside the trace
(the router does this per call) and `bias` is persistent. Works on a MeshDevice (same program on every chip:
the router is replicated).

### Semantics vs the composite (`GLM53_ROUTER_TIE_BREAK=topk`)

* `choice = s + bias`: the compute kernel adds the two fp32 tiles with operands unpacked straight into the fp32
  dest (`UnpackToDestFp32`) and the SFPU `add_binary_tile`, the path `ttnn.add` takes on fp32 tensors
  (measured bit-exact vs torch: 0 / 9216 mismatches), so the ranking keys are the composite's bits.
* Ranking: exact fp32 order (monotonic unsigned keys of the bit patterns). Ties: **lowest column first, and a
  later equal value never displaces an earlier one** (stable). `ttnn.topk` (the composite) is a bitonic network
  whose tie order is neither lowest- nor highest-index-first and which can even keep a *different* tied column
  (probe `_logs/agents/routerop/probe_topk.log`: row of values `arange % 3` -> ttnn `[2, 5, 11, 8, 17, 20, 23, 14]`,
  stable `[2, 5, 8, 11, 14, 17, 20, 23]`). On real router inputs exact fp32 ties of `sigmoid + bias` do not occur
  (the unit test checks the ids bit-exact against the composite on real layer-4 scores); the tie case is covered
  against the stable torch golden only.
* Weights: soft fp32 on the RISC-V (correctly rounded IEEE), HF order `(s / sum) * scale`; the composite does
  `s * (scale / sum)` through SFPU reciprocal: both within ~2.3e-7 relative of torch. bf16 outputs are
  round-to-nearest-even of the fp32 weight.

## Program

32 cores (rows 0..31 of the tile row, row-major over the compute grid: `RouterTopk.cores`), one token row each.
Per core: writer (BRISC, `kernels/writer_router_topk.cpp`) and compute (`kernels/compute_router_topk.cpp`).

| CB | size | content |
|---|---|---|
| `c_0` S | 1 fp32 tile | the row's E scores packed 9 x 32: column tile `ct`, face half `h`, element `i` at word `h*256 + ct*16 + i` (each 64 B face-row piece of a score tile is contiguous) |
| `c_1` B | 1 fp32 tile | the biases at the same positions (row 0 of the bias tiles) |
| `c_2` CH | 1 fp32 tile | `S + B` from the compute kernel |
| `c_3` SCR | 2 KB | writer staging (dense row pieces, w / idx face rows), 64 B aligned |

Writer: 18 + 18 reads of 64 B (`scores` page `ct` offset `(face + h) * 1024 + (row % 16) * 64`, `bias` page `ct` offset
`h * 1024`), push S and B; wait CH; insertion top-K (strict compares) over the 288 keys; `sum`, 8 divides and
multiplies in soft fp32; zero the staging area and write: dense 18 x 32 B (`dense` page `ct`, offset
`(face + h) * 512 + (row % 16) * 32`, the L1 piece at the same offset modulo 64), w / idx two 64 B face rows (fp32,
uint32) or two 32 B face rows (bf16). `noc_async_write_barrier`, pop CH, `noc_async_full_barrier`.

Compile-time args (writer): `[CB_S, CB_B, CB_CH, CB_SCR, E, K, bits(scale), w_mode (0 none / 1 fp32 / 2 bf16),
idx_mode (0 none / 1 uint32 / 2 fp32), dense_mode (0 / 1)] + TensorAccessorArgs(scores, bias, dense, w, idx)`
(absent outputs carry the scores accessor as a placeholder). Compute: `[CB_S, CB_B, CB_CH]`; config HiFi4,
`fp32_dest_acc_en`, `UnpackToDestFp32` for S and B. Runtime args: writer per core `[row]`, common
`[scores, bias, dense, w, idx addresses, T]`; compute none.

## Measured (chip 0, layer 4 KDA+MoE, B = 32 users, one traced replay under Tracy, `tests/test_perf_layer.py`)

| router after the gate matmul (`32 x 4096 x 288`, 17.8 us, both paths) | ops | device us | op-to-op gaps us |
|---|---|---|---|
| composite (`GLM53_ROUTER_FUSED=0`): add, topk, FillPad, gather, FillPad, sum, rdiv, mul, typecast, untilize x3, scatter, tilize | 14 | 142.1 | 14.0 |
| fused (`GLM53_ROUTER_FUSED=1`): `router_topk` on 32 cores | 1 | 13.2 | 0.25 |

Layer step: 72 -> 59 ops, 10318 -> 10180 us device time, 10.49 -> 10.35 ms/step wall (`_logs/agents/routerop/report_{base,fused}/`).
The step is dominated by the expert path (`SparseMatmul` 4.2 + 2.2 ms at B = 32 on one chip); the router's remaining cost is the
gate matmul plus this op (31 us).

Test: `tests/custom_ops/test_router_topk.py` (real layer-4 scores through the real gate weights from the T256
dump, T = 32 / 4, all three output modes vs the composite and the torch golden; a tie case; trace replay;
op timing). Wiring: `TtRouter._forward_fused` (`GLM53_ROUTER_FUSED=1`, default off).
