# Integrated causal QSA scorer contract

This CPU-only audit inspected the integrated source and completed artifacts on
2026-11-14. It did not open devices, rerun measurements, or change implementation
or tests. No scorer-local buffer-count, output-initialization, persistent-state,
or trace-input ownership defect was found. The runtime evidence below covers
exact real-operand scores/selection and reduced-layer two-slot serving behavior;
it is not a substitute for the full-stack accuracy, batch-32, or performance gates.

## Integration and numerical contract

`MeshQSA._selected_ids` in
[`multichip_decoder.py`](../../tt/multichip_decoder.py), lines 1031–1036, calls
`causal_indexer_scores(..., packed_decode=True)` and immediately feeds its output
to the existing `fused_selected_token_ids`. This integration applies to prefill
and decode wherever MeshQSA calls the overridden selector. The scorer changes
neither projection weights nor cache dtype, residual layout, RoPE, selected-KV
gather, or attention arithmetic. LM-head geometry 20/3/2 is a separate terminal
policy and is not used by this scorer.

[`causal_indexer_scores.py`](../../tt/causal_indexer_scores.py) validates:

- BF16 tiled interleaved DRAM query `[1,4,Q,128]`, with `1 <= Q <= 32`;
- BF16 tiled compressed keys `[1,1,K,128]`, tile-aligned `32 <= K <= 65536`;
- FP32 tiled absolute positions `[1,1,Q,1]`, physically `[1,1,32,32]`;
- same device/mesh for all tensors and FP32 output with exact padded shape.

The position domain is nonnegative integer token positions below `4*K`. The
reader converts exact FP32 integer bits on the device and computes
`complete = (position + 1) // 4`. This matches the selector's complete-group rule
in [`fused_decoder.py`](../../tt/fused_decoder.py), lines 1254–1270, and the
four-token compression/write-group logic at lines 1476–1479. The reader handles
position zero explicitly. Fractional/negative positions are outside the model
contract; they are not a supported inactive-row encoding.

Q1 uses packed per-head dots `[1,1,4,K]`, physically `[1,1,32,K]`. Q>1 retains
`[1,4,Q,K]`, physically `[1,4,32,K]`. Public raw `causal_indexer_dots` retains the
original layout even for Q1. The packed Q1 tail is ReLU, sum over dimension 2,
then multiply by `128**-0.5`; the other tail retains dimension-1 head summation.
The packed form avoids three physical dot-tile rows and the head transpose
without replacing the common final arithmetic with a scalar RISC computation.

Both compute kernels retain the same dot calculation:

1. Broadcast one BF16 query row to DST0 with
   `unary_bcast_init/ROW/uninit`, then `copy_init(3)` and copy the BF16 key tile
   to DST1.
2. Multiply with SFPU; the first width tile seeds DST2, and width tiles 1, 2,
   and 3 are added sequentially in FP32 to DST2.
3. Apply `sfpu_reduce<SUM, Float32, REDUCE_ROW>` to DST2 and pack FP32.

See [`causal_score_compute.cpp`](../../tt/kernels/causal_score_compute.cpp),
lines 26–53, and
[`causal_score_packed_q1_compute.cpp`](../../tt/kernels/causal_score_packed_q1_compute.cpp).
This is the accurate reduction helper's four-tile fold followed by intra-tile
SFPU reduction, not a TF32 matrix dot or the previously replaced explicit
half-width tree. Both kernels use FP32 destination accumulation and disable
approximate math. The BF16 inputs have the same values as the baseline's BF16
to FP32 conversion; no new reduced-precision operand boundary is introduced.

The readers populate only row-zero BF16 elements at face offsets 0–15 and
256–271 before each CB2 push. The ROW broadcast API consumes those elements;
other CB2 cells are intentionally not initialized because they are not operands.
This does not apply to output tensors: their complete physical tiles are
initialized, as described below.

## Complete output writes and circular-buffer accounting

Let `T = K/32`, `Lq = floor((position[q]+1)/4)`, and
`A(j) = number of query rows with 32*(j % T) < Lq`.

| Path | Job ownership | CB3 key tiles per active job | CB2 query tiles | CB4 result tiles | Output writes |
|---|---|---:|---:|---:|---|
| Original Q1..32 layout | `j = core_index; j < 4*T; j += num_cores`; head=`j/T` | 4 if A(j)>0 | 4*A(j) | A(j) | One complete FP32 tile at output index j |
| Packed Q1 | `j = core_index; j < T; j += num_cores` | 4 if 32*j<L0 | 16 if active | 4 if active | One complete FP32 tile at output index j |

Every output job belongs to exactly one core. Readers push exactly the operand
counts that compute pops; compute pushes exactly the result counts that writer
pops. An inactive job pushes/pops no operands or results but still writes its
complete zero output tile. A zero-live invocation therefore has no dangling
compute wait or unmatched tile-register acquisition.

CB ownership is explicit in the Python launch descriptor:

| CB | Capacity/format | Ownership and lifetime |
|---|---|---|
| 0 | 1 FP32-sized page | Reader writes integer live counts; compute reads through mailbox-synchronized `read_tile_value`; retained until compute finishes |
| 1 | 4 BF16 tiles | Reader-private query scratch; never published to compute |
| 2 | 8 BF16 tiles | Reader publishes four row broadcasts per active head/query; compute pops each after use |
| 3 | 8 BF16 tiles | Reader publishes four key tiles per active job; compute retains all four until every consumer for the job finishes |
| 4 | 2 FP32 tiles | Compute result stream; writer consumes one tile per active head/query |
| 5 | 1 FP32 tile | Writer-private output assembly; zeroed for every output job |
| 6 | 1 FP32 tile | Writer-private position read |

The original writer, lines 29–41, zeroes all **1024** FP32 cells, fills only
valid `(query,key)` cells, then writes all 4096 bytes and waits for the NOC write
barrier before reusing scratch. Consequently padded query rows and invalid key
columns are zero even when the scratch formerly held another row/head.

The packed writer, lines 25–38, likewise zeroes all 1024 cells, then places each
head's dot results into rows 0–3. Rows 4–31 and invalid columns remain zero. A
single worker owns all four head rows, eliminating partial-write races. It also
waits for each NOC write before reusing scratch.

Sources:
[`original reader`](../../tt/kernels/causal_score_reader.cpp), lines 25–65;
[`original writer`](../../tt/kernels/causal_score_writer.cpp), lines 29–41;
[`packed reader`](../../tt/kernels/causal_score_packed_q1_reader.cpp), lines 26–63;
[`packed writer`](../../tt/kernels/causal_score_packed_q1_writer.cpp), lines 25–38.
The packed reader copies each query broadcast into CB2 before overwriting its
private query scratch for the next head. Key CB3 remains live across all four
heads and is popped only afterwards. Each active head/query has exactly one
DST acquire/commit/wait/release sequence; no runtime-dependent branch splits it.

The last active key tile can contain physically allocated future keys. Their
rows may be computed, but writer masking excludes them, and row reduction does
not mix distinct key rows. The selector independently bounds reads by the same
complete-group count, so zeros in invalid output columns cannot become selected
future keys.

## Trace lifetime and full-generator preparation

The scorer stores no mutable Python invocation cache, no global output tensor,
no semaphore pool, and no reusable device-state owner. Its generic-op bindings
contain only query, keys, positions, and output. Production calls allocate a
fresh output; the optional raw-output interface validates geometry and rejects
an output sharing an input buffer address. No output is assigned back into a
cache or retained by the scorer. All seven CBs are program-local resources.

Eager execution may allocate temporary dot and tail tensors. Capture records
the operations and their addresses; replay does not execute the Python wrapper
or allocate a new output. Dots are consumed by ReLU/head-sum, scores by the
selector, and selected IDs by the following KV gather in the same ordered graph.
No scorer-local allocation exemption or broad corruptible-allocation scope is
present. Temporary storage must remain governed by the surrounding trace
allocation lifetime; the scorer does not claim an independent exemption.

The new runtime positions affect only device control flow. They do not change
host arguments, tensor dimensions, CB sizes, compute descriptors, or generic-op
cache signatures. Warming at zero live groups still prepares the same native
program that later reads nonzero groups; the extra iterations are runtime loops
in that compiled program, not additional host-dispatched ops.

[`generator.py`](../../tt/generator.py), lines 399–447, prepares all physical
prefill widths `(1,2,4,8,16,32)`, then whole-model B1 decode or factored decode,
then state helpers and sampling, before capture. It resets warmup-mutated state
and synchronizes before capturing factored traces. Logical nonaligned prompt
lengths are decomposed into dyadic physical chunks by `prefill_spans` at lines
669–679; adding Q1 packing does not restrict logical prompt lengths. Prefill
traces are captured at first use only after this preparation; B1 whole-model
decode is likewise captured from prepared state.

[`factored_decode.py`](../../tt/factored_decode.py), lines 70–95, binds each
attention segment to one fixed slot, its persistent token input, and that
slot's cache. Attention is Q1 per slot even at logical batch 32. Only MoE
activations are packed into multirow buckets. Thus the packed-Q1 scorer is used
for every active slot; it does not reinterpret 32 independent users as 32
contiguous positions in one request. Lines 106–143 warm every configured slot
and MoE bucket before recording the finite trace set. Lines 145–160 select
already-captured per-slot attention segments and submit them nonblocking on
CQ0. Inactive slots have no attention replay, so their scorer/cache/position
state is not touched. Changes in scheduler row order do not rebind scorer
storage or force a new scorer capture.

## Completed evidence inspected

- [causal_packed_v4.json](causal_packed_v4.json) and
  [its log](causal_packed_v4.log): `passed=true`, `packed_decode=true`, serial
  four-chip real captured operands. All **36** rows report exact live scores,
  exact ordered selected IDs, and `passed=true`. Coverage is Q=1/3/17/32 and
  starts 0/3/127/2047/2048/32767/205504/262112; start 3 is repeated after the
  longer positions for each Q, so this is 32 distinct shape/start combinations
  plus four repeated-input checks. Q1 exercises packed output; larger Q
  exercises the original output layout with ROW broadcast. These are final
  score/selection comparisons, not a separately recorded raw-dot bitwise test.
- [serving_watcher_v1.json](serving_watcher_v1.json),
  [console log](serving_watcher_v1.log), and
  [watcher log](serving_watcher_v1_watcher.log): all **16** checks pass on real
  reduced layers 0/1/3, context capacity 128 and **two fixed slots**.
  Allocation tracking is enabled with program-cache allocations included
  (`skip_program_cache="0"`), and program-cache entries remain 2170 from
  preparation to completion. The console records watcher activation; the stage
  owner ran this with `TT_METAL_WATCHER=10`.
- The serving artifact covers mixed prompts `[3,33]`, reordered slots, resumed
  absolute starts `[2,17]`, fixed/inactive slots, dormant feedback preservation,
  changed/unchanged page tables, and cross-request trace reuse. The dedicated
  steady-state feedback check reports zero token, position, and page-table host
  refreshes. Overall counters include deliberate scheduler/reset tests and are
  not steady-state benchmark counters.
- Its positive control records rejection of one deliberately live unsafe device
  buffer, followed by final clean replay. See
  [positive_control.log](trace_alloc_tracking/positive_control.log).
  Twenty-two captures have zero releases before teardown; the unload test
  releases them explicitly. The artifact's source manifest includes the scorer
  Python integration and factored/generator files; it does not independently
  list every C++ kernel source, so retain the linked current native sources and
  the compilation/run logs alongside this evidence.

## Limits and remaining stage evidence

The scorer has no host token/position readback, but that source property does
not establish the entire generator's performance or fallback status. This
report does not claim full-48-layer correctness or throughput from a component
probe. The full-stack benchmark running during this audit is intentionally not
used as completed evidence.

The two-slot watcher artifact validates the factored mechanism with current
code. Batch-32 source routing was inspected; **current integrated B32 hardware
coverage is not supplied by that artifact**. A prior B32 run cannot silently
stand in for current-code evidence if this change invalidates its measured path.
The stage owner retains responsibility for the applicable B32 gate.

Capacity 65536 compressed groups corresponds to the supported 262144-token
maximum. Exact near-end scorer cases are not a full-model long-context memory
or prefill/decode smoke. This optimization changes neither cache dtype nor cache
layout, so it does not itself require a new maximum-context capacity search.

The output and tail still scale with allocated compressed-key capacity even
for short prefixes. Long-prefix performance remains a measured workload
question. `causal_packed_v4.json` includes score-plus-selector timings, not
full-model token-out timings or isolated kernel durations.

The source audit noticed stale experimental/not-integrated docstrings. The
stage owner corrected only those docstrings during the full-stack measurement;
executable code was unchanged. `selected_sources_v1.json` preserves the source
at launch and `work_log.md` records this documentation-only difference.
