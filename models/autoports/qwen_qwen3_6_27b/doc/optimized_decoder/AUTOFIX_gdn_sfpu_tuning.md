# Precise SFPU recurrence tuning candidates

Source-only helper work, 2026-11-12. The stage owner reports that compact17's corrected intake passes the isolated own-input recurrence probe at approximately1e-7 relative error, and correction18 passes the long decoder sequence with minimum PCC0.995407 at3.231ms. Those measurements establish the starting point; they are not measurements of the candidates below.

All candidates reside in `doc/optimized_decoder/candidates/`, expose the same `recurrent_sfpu_compact` interface and `READER`, `WRITER`, `COMPUTE` strings, and include corrected64-byte BF16 face intake and FP32 direct unpack. Each file is standalone and cumulative. No main decoder, frozen candidate, or shared TTNN source was changed. No hardware or C++ compilation was performed by this helper.

| Candidate | Change | Source justification | Required experiment |
|---|---|---|---|
| `gdn_sfpu_hoist_init.py` | Hoist `copy_init` out of every `load` | `api/compute/tile_move_copy.h` separates copy init from repeated copy calls. `ttnn/.../unary/device/kernels/compute/eltwise_sfpu.cpp:32` initializes once before its copy/SFPU loop. Every recurrence CB has the same FP32 tile format. | Own-input probe, then full decoder latency |
| `gdn_sfpu_scalar_mul.py` | Specialize scalar normalization with known-zero mean to scalar multiply | Reuses native Blackhole `_broadcast_scalar_from_dest_`, row-band addressing, and independent `SFPMUL` operations from `ckernel_sfpu_binary_bcast.h:715`. Eliminates zero-tile unpack, zero-scalar broadcast, and32 adds per tile. | Probe state/output against prior variant, then latency |
| `gdn_sfpu_row_copy.py` | Replicate delta row directly instead of multiplying a full ones tile by that row | Reuses native row-broadcast `SFPTRANSP` mapping: extract row0 into LREG4, copy it into four row registers, transpose back, store the same four-row pattern into all eight bands. | Probe with distinct per-column values before decoder latency |
| `gdn_sfpu_fused_outer.py` | Fuse column multiplication and state addition in one SFPU traversal | Native column broadcast supplies exact K values and delta data. Separate `SFPMUL` then `SFPADD` preserve the original product/add rounding points; no FMA. | Probe normal/zero-state/zero-KV plus repeated steps, then latency |

## Operation accounting

The normalized compact baseline performs53 tile packs/head: exp1, Q/K transpose8, decayed state16, prediction4, delta4, updated state16, readout4. These candidates leave that pack count unchanged. They target the substantial per-tile setup and DST traffic before attempting a schedule with more interdependent changes.

Baseline `load()` issues193 copy-init/copy pairs/head, excluding transpose. Hoisting reduces initialization to three calls/head plus one initial call: before each Q/K preparation and immediately after both transposes. SFPU math does not switch the unpacker into a different operation. The explicit reinitialization after transpose restores the copy configuration before decay and contractions.

Scalar specialization removes20 input tile copies/head in normalized mode:16 zero-mean copies for decay and4 for beta scaling. Raw normalization removes another8 zero-mean copies. Row replication removes16 ones-tile copies/head, leaving157 ordinary copies/head in normalized mode. Its bit-moving broadcast also removes multiplication by1, which is exact for finite normal inputs. The fused outer candidate removes the intermediate product DST store/reload required by a separate full-tile addition traversal while retaining distinct product/add arithmetic.

The specialized scalar and outer kernels use only the existing inline compute source mechanism. They reuse internal SFPU helpers from the checkout's Blackhole implementation, so their compile compatibility is tied to that source version. They preserve all external allocation, reader/writer, state ownership, and output synchronization behavior. Zero/one constant CBs remain allocated so the candidate isolates compute changes.

## Other scheduling hypotheses

Four value-column workers per head are legal because each state column and output column is independent; no cross-core reduction is needed. Each worker would read four state tiles, the shared Q/K vectors and gates, and one V tile, then update disjoint state/output pages. This is promising for B1, where48 heads cannot occupy110 cores. At B32,1536 heads already fill the device; splitting would quadruple Q/K/gate preparation and per-head scheduling while preserving state traffic and total arithmetic. It is untried, not rejected, and should follow measured compute results if the small-batch profile warrants it.

Eliminating the decayed-state CB by recalculating decay inside prediction and update would save16 packs/head but duplicate16 decay operations and their scalar loads. A retained-DST column schedule could avoid those duplicates, but requires a careful full-DST register allocation and changes pack/math overlap. These are hand-off hypotheses, not claimed improvements.

## Validation performed and gates outstanding

Python AST checks passed for all four files. Host checks verified balanced inline C++ braces, the three explicit copy-init sites, corrected BF16 intake, explicit FP32 unpack modes, and the absence of `SFPMAD` in the fused-outer candidate. The parent should measure candidates sequentially against compact17/correction18, retaining only numerically valid latency improvements. Compile/API errors require diagnosis rather than rejection of the underlying optimization. Full repeated-run, long-sequence, trace, and watcher gates remain necessary for the ultimately selected code.
