# g6_clone: forecast

## Summary

Feasibility: feasible as one device program

| Id | Severity | Risk | Would show as | Handled by |
|---|---|---|---|---|
| R1 | high | values pass through a conversion or an engine and change bits | wrong bits in float32 low mantissa, subnormals, NaN payloads or integers above 2^24; randn data within a tolerance | spec rule 1 and the identity paragraph; suite: random-bit-pattern generator and bitwise exact grading (brief 1, 7), regression test "bit patterns" (brief 6) |
| R2 | high | the output aliases the input, or the call changes the input | right values, wrong storage; a later write shows in both | spec rule 2 and the Output line; runner asserts dtype, layout and placement (brief 1); regression test "new storage" (brief 6) |
| R3 | medium | tile padding of the input reaches a logical element, or logical rows of a partial tile are misaddressed | wrong values at [1,1,1,62080], [1,5120], [1,1] only | spec rule 6 and Shape handling; INPUTS holds the three shapes (brief 4); loose case "poisoned padding" (brief 5) |
| R4 | low | the work collapses on few cores for a one-tile-row tensor or a few-row ROW_MAJOR tensor | correct values, at the speed of a few cores, on p4 and [1,1,1,62080] | spec rule 9 and Work geometry; INPUTS holds p4 (brief 4) |
| R5 | medium | ROW_MAJOR pages of 4, 32 or 64 B against the 64 B DRAM alignment, or sub-page reads | hang or wrong elements for [1], [8], [16] int32 / uint32 | INPUTS holds every ROW_MAJOR production shape (brief 4); left to the planner |
| R6 | medium | a program cache key without dtype or layout replays the program of another case | wrong data on the second of two cases of one shape: [1,1] float32 and [1,1] bfloat16 TILE, [16] int32 and [16] uint32 | mandated rule BB-07; INPUTS holds both pairs (brief 4) |
| R7 | low | an extra DRAM pass or an intermediate tensor | correct values; twice the DRAM traffic on p1 to p4 | spec rule 4 |
| R8 | low | per-call overhead dominates the many tiny cases | correct values; nothing inside the kernel moves it | mandated rules BB-17 and BB-18; lesson RB-036 |

## Risks

### R1: values pass through a conversion or an engine and change bits

- Why it bites here: the contract's gate is bit-exact equality (`sources/local/contract.md:73`).
  A float32 operand of the matrix engine keeps 10 mantissa bits; the oracle's float32 precision
  row puts that cut at relative RMS 0.0004 on randn data, inside the default float32 tolerance
  (`oracle/oracle_report.json`, emulated (storage and operands only), not confirmed on a device).
  A float32 detour of int32 or uint32 loses 2^24 + 1 (oracle probe). A path through unpacker and
  packer may flush subnormals or rewrite NaN payloads (INFERENCE: not measured here).
- Symptom: exact comparison fails on a few elements, or nothing fails when inputs are randn and
  the grading is by tolerance.
- Caught by: bitwise exact grading of every cell with random bit patterns as inputs (brief 1, 7);
  regression test "bit patterns".
- What the spec does: the identity paragraph and rule 1 (bit-identical for every bit pattern, no
  conversion or arithmetic on the way).
- Left to: golden suite (the generator and the exact grading), planner (the data path).

### R2: the output aliases the input

- Why it bites here: S1 of reference.md: the result is new storage. A tensor returned as the
  input itself, or as a view on its buffer, has bit-identical values in every cell.
- Symptom: no golden cell fails; in the model a later in-place write to one tensor changes the
  other.
- Caught by: regression test "new storage" (brief 6) and the verifier's reading against rule 2.
- What the spec does: the Output line (newly allocated buffer) and rule 2.
- Left to: golden suite, verifier.

### R3: tile padding reaches a logical element, or partial tiles are misaddressed

- Why it bites here: three TILE production shapes have partial tiles: [1,1,1,62080] (1940 tiles,
  1 logical row of 32 each), [1,5120] (160 tiles, 1 row), [1,1] (1 element) (tensix.md,
  Geometry). Input padding is garbage in the model and zeros in a host-made test tensor
  (`sources/local/contract.md:74`). A copy of whole tiles is immune; a design that moves only
  logical rows (tensix.md, Candidate regimes) must address them right.
- Symptom: wrong elements in these three shapes only.
- Caught by: the three shapes in INPUTS; the loose case with poisoned padding (brief 5).
- What the spec does: Shape handling and rule 6.
- Left to: planner.

### R4: the work collapses on few cores

- Why it bites here: p4 [1,1,32,62080] and [1,1,1,62080] are one tile row of 1940 tiles; the
  ROW_MAJOR [16,3,2560] is 48 pages (tensix.md, Geometry). A split by tile rows or by large
  blocks puts them on one or a few cores. No case of the brief fails on a single-core op: the
  resident set of one unit of work is 8192 B at two float32 tiles, 0.01 of the 1347776 B the
  model leaves free (`tile_budget.py resident x=2:float32 --l1-bytes 1347776`; INFERENCE).
- Symptom: correct values, slow on the flagged case p4.
- Caught by: nothing mechanical; the verifier's reading against rule 9.
- What the spec does: Work geometry and rule 9 (the cores rule; its condition leaves out inputs
  of one tile or one row).
- Left to: planner, verifier.

### R5: small ROW_MAJOR pages against the DRAM alignment

- Why it bites here: the int32 and uint32 cases have pages of 4 B ([1]), 32 B ([8]), 64 B
  ([16]) and 128 B ([32], [1,1,1,32]); DRAM pages are stored at a 64 B stride
  (`tt_metal/hw/inc/internal/tt-1xx/blackhole/core_config.h:42`), so a page of 4 or 32 B is
  followed by an alignment gap that is not data. A transfer sized or aligned for L1 (16 B,
  `tt_metal/hw/inc/internal/tt-1xx/blackhole/core_config.h:43`) can be wrong for DRAM (INFERENCE
  from the two alignments).
- Symptom: a hang with a NoC error under the debug build, or wrong elements, only in these cases.
- Caught by: all eight int32 and uint32 production shapes in INPUTS (brief 4), once their dtype
  is supported.
- What the spec does: Shape handling names the row sizes; nothing more: the transfer is the
  planner's.
- Left to: planner.

### R6: an incomplete program cache key

- Why it bites here: two pairs of production cases share a shape and differ in dtype only:
  [1,1] float32 and [1,1] bfloat16 TILE (tile of 4096 B against 2048 B), [16], [8] and [1] int32
  against uint32 ROW_MAJOR (same bytes). A key without dtype replays the first program for the
  second case (BB-07).
- Symptom: wrong data on the second case of a pair, only when both run in one process.
- Caught by: both cases of each pair in INPUTS (brief 4); mandated rule BB-07.
- What the spec does: nothing beyond the mandated rule.
- Left to: implementer, verifier.

### R7: an extra DRAM pass

- Why it bites here: the op is a pure move (math.md, Counts: 0 arithmetic); every extra full
  pass over a tensor adds its whole size again: 12582912 B at p1.
- Symptom: correct values, twice the traffic.
- Caught by: nothing mechanical.
- What the spec does: rule 4.
- Left to: verifier.

### R8: per-call overhead on the tiny cases

- Why it bites here: 10 of the 29 cases are one page or one tile ([1], [8], [16], [32],
  [1,1,1,32] uint32, [1], [8], [16] int32, [1,1] float32 and bfloat16); their call is two page
  transfers. The
  fixed cost of a program dominates.
- Symptom: correct values; a kernel that cannot get faster.
- Caught by: nothing in the suite (a case of fewer than 32 tiles is not measured by the perf
  passes).
- What the spec does: nothing beyond the mandated rules BB-17 (few core ranges) and BB-18
  (kernel text that fits beside the preceding programs).
- Left to: planner; lessons.md, RB-036.

## Ways to pass dishonestly

| Id | Shortcut | Why the suite would pass | Countermeasure |
|---|---|---|---|
| X1 | the entry point calls another device op that copies, or a chain of ops | values are right; programs per call are not graded | spec rule 3 (the op's own program); mandated rules BB-01 and BB-02 |
| X2 | read the tensor to the host and write it back | values are right | BB-01 ("read tensor values on the host") |
| X3 | return input_tensor or a view on its buffer | every value is identical | spec rule 2; regression test "new storage" (brief 6) |
| X4 | one core moves everything | "the golden suite passes on a single core too" | spec rule 9, the cores rule |
| X5 | move float32 through an engine and pass on tolerance | randn values keep most bits | bitwise exact grading with random bit patterns (brief 1, 7); spec rule 1 |
| X6 | host-side to_layout, tilize or untilize around a TILE-only kernel for ROW_MAJOR cases | values are right | spec rule 5 |
| X7 | gate the ROW_MAJOR or small-page cases out by a tagger, EXCLUSIONS or a size axis | refused cells are expected failures | the Phase 0 paragraph closes each hatch by name; spec rule 8 |

## Bottleneck forecast

    python3 -B "$EXPLORATOR"/scripts/tile_budget.py counts --read x=16,12,128,128:float32 --write out=16,12,128,128:float32

| Shape | Bound by | Because | Changes when |
|---|---|---|---|
| p1 [16,12,128,128] float32 TILE | DRAM traffic | r = 0: no engine work against a machine balance of 0.290 core-cycles per byte at an assumed 110 cores, dtype float32; 12582912 B read and written | never: no arithmetic at any size |
| p2 [8,12,128,128] float32 TILE; [1,12,128,128] float32 | DRAM traffic | as p1; 6291456 B and 786432 B each way | never |
| p3 [1,512,5120] bfloat16 TILE and the other bfloat16 TILE shapes from 32 to 1280 tiles | DRAM traffic | as p1, dtype bfloat16; 5242880 B each way at p3 | below a few tiles: per-call overhead |
| p4 [1,1,32,62080] bfloat16 TILE | DRAM traffic | as p1; 3973120 B each way; one tile row of 1940 tiles (R4) | never |
| [1,1,1,62080] and [1,5120] bfloat16 TILE | DRAM traffic | as p1 on the tiled bytes (3973120 B and 327680 B); the logical bytes are 1 row of 32 per tile | when only logical rows move: DRAM traffic, counted in transactions (row 0 of a tile lies in two 16 x 16 faces: 3880 pieces of 32 B each way, INFERENCE from the face layout) |
| [16,3,2560], [8,3,2560], [1,3,2560] bfloat16 ROW_MAJOR | DRAM traffic | 48, 24, 3 pages of 5120 B each way | at 3 pages: per-call overhead |
| [1], [8], [16], [32], [1,1,1,32] int32 / uint32 ROW_MAJOR; [1,1] float32 and bfloat16 TILE | per-call overhead | one page or one tile each way | never in production |

## Prerequisites and blockers

- None. The op is one device program of NoC transfers; no primitive is missing (tensix.md, Gaps),
  no question blocks the spec.

## Not forecast

- How close a real copy comes to the DRAM rate on Blackhole with the whole grid busy (no
  measurement for a saturated copy is recorded).
- Whether the device's unpacker and packer preserve subnormals and NaN payloads (R1 rests on an
  inference; the rule forbids the path whatever the answer).
- The start gap and kernel-config ring behaviour of this program inside the traced models
  (BB-17, BB-18).
