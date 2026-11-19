# Selected norm and recurrence kernel contracts

Source-only audit of [delivery194](../../tt/optimized_decoder.py), SHA256
`315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22`.
No correctness defect requiring a root fix was found in the selected kernels.
This is bounded implementation review, not a stage-review verdict or a claim
that final-source gates have finished. No hardware was run or implementation
changed for this audit.

After stripping only their integration namespace prefixes, Python AST comparison
found the following definitions identical to the independently probed helpers.
Embedded C++ strings were compared byte for byte; Python formatting was ignored.

| Selected implementation | Compared definitions | Independent evidence |
| --- | --- | --- |
| `_norm_zero_meanhidden_rmsnorm_sfpu` | All nine kernel strings and host function match [zero-mean helper](candidates/hidden_rmsnorm_exact_tree_zero_mean.py) | [norm184 probe](norm184_probe.json): 54 cases, zero BF16 mismatches |
| `_fused_delta_row_recurrent_sfpu_compact` | All three kernels, compute configuration, partition and host functions, and five compile-time control flags match [delta-row helper](candidates/gdn_sfpu_fused_delta_row.py) | [delta168 probe](delta168_probe.json): 54 cases, zero output/state bit mismatches |

The default GDN policy selects raw packed BF16 Q/K/V, internal Q/K normalization,
BF16 persistent state, and the fused decay/prediction recurrence. The selected
norm handles compact hidden input. Other policy branches are outside this audit.

## Hidden RMSNorm

The input is interleaved DRAM BF16 TILE `[1,B,5120]`, `1 <= B <= 32`;
folded gamma is interleaved DRAM FP32 TILE `[1,1,5120]`. Output is BF16 with the
input shape. All buffers must belong to the same device. Output and scratch
must not alias inputs or each other. The helper checks distinct base addresses,
shapes, dtypes and memory layouts; it assumes valid, nonoverlapping allocations.

Arithmetic preserves the accurate staged reference's exact reduction topology:

1. For each of 1024 positions, square the five BF16-widened inputs in FP32 and
   accumulate chunks in increasing order, with separate multiply/add rounds.
2. Reconstruct the same face-major dense tile: folded channel `block*32+col`
   becomes dense `[block,col]`. Perform native SFPU COL reduction, then ROW
   reduction, multiply by FP32 `1/5120`, add FP32 epsilon and apply native rsqrt.
3. Preserve `x + (-0)`, multiply by the inverse RMS, multiply separately by FP32
   gamma, and use explicit native FP32-to-BF16 round-to-nearest-even before pack.

All FP32 circular buffers use `UnpackToDestFp32`; FP32 destination accumulation
alone is insufficient because ordinary source unpack can truncate to TF32.
The specialized zero mean sets the same negative-zero register consumed by
native `_finish_col_normalize_row_band_`. It does not delete the zero subtraction.
[CPU/source checks](candidates/hidden_norm_zero_mean_cpu.json) cover all 65,280
finite BF16 encodings including signed zero and all referenced inverse addresses.
These checks and the finite-input device probe are not exhaustive NaN/Inf tests.

Scratch is FP32 TILE `[1,32,1024]` (128 KiB) plus `[1,32,32]` (4 KiB), normally
interleaved L1; supplied interleaved DRAM scratch is also supported. The three
generic operations must execute in order and buffers must remain alive through
their consumers, including trace replay. Tensor arguments register their addresses
with each generic operation. The selected decoder allocates scratch internally.

Fold uses 32 independent column-block owners. Inverse uses one owner per active
user, writing disjoint rows; the final user zeros inactive inverse rows. Output
owns all 160 pages exactly once across up to 80 cores. Its inverse reader fetches
only the two left faces at byte offsets 0 and 2048. Right faces are uninitialized
locally but never referenced by SFPU arithmetic; direct FP32 unpack only copies
them. Output writer explicitly zeros inactive BF16 token rows. Every input is
pushed after its read barrier, and every writer waits for remote completion
before releasing its source buffer.

The earlier distributed norm with a changed sum tree failed a whole-sequence
gate despite small isolated error. That variant is not selected; changing the
fold/reduction order invalidates this contract.

## Compact GDN recurrence

Each global head owns a disjoint 128-by-128 state matrix. There are `48*B` value
heads; Q/K head index is `(global_head % 48) // 3`. Packed row-major BF16 input
uses one 20,480-byte page per user, with Q/K/V byte offsets 0/4096/8192 and
256 bytes per head vector. Gates are TILE `[B,1,48]`; persistent state is TILE
`[48*B,128,128]`, BF16 or FP32. Inputs are interleaved DRAM. The internally
allocated output is FP32 DRAM TILE `[1,48*B,128]`.

Raw Q/K normalization squares and folds the four 32-element segments in order,
then uses SFPU ROW reduction, epsilon `1e-6`, rsqrt and multiplication. Q scaling
by FP32 `128**-0.5` is a separate operation. Prepared Q/K replicates each column
with the native broadcast operation and packs four segments into disjoint
eight-column slots. Head partitioning is contiguous and exhaustive; reuse starts
at a core boundary or global triple boundary, and ends at a core boundary or the
end of a triple. Duplicate reader Q/K entries are consumed even when reused.

Per value-column block, state decay, multiplication by K and accumulation are
separate FP32 operations. Four state segments accumulate in increasing order,
then use the native COL reduction for prediction. Delta is `(V-prediction)*beta`
with separate subtract and multiply. The row-only delta transform preserves the
live row and broadcasts it without reassociation. State update is separately
rounded `delta*K`, then addition to decayed state. Q readout uses the current
FP32 update, **before** BF16 persistent-state rounding, with the same four-segment
accumulation order and COL reduction. No fused multiply-add substitutes for
these operation pairs.

BF16 persistence copies each live FP32 value and applies the native integer RNE
sequence: `(bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000`, with 32-bit
wrapping. Live FP32 registers remain unchanged for Q readout. FP32-state mode
stores the FP32 update directly. [Persistence/balance evidence](persistbalance164_probe.json)
and [CPU checks](candidates/gdn_fused_persistence_balance_cpu.json) cover this
predecessor; [delta-row CPU checks](candidates/gdn_fused_delta_row_cpu.json) and
the delta168 device probe cover the selected final recurrence helper.

Full DST synchronization and eight FP32 DST tiles are required. State occupies
DST0..3, delta/work DST4, K DST5, accumulator DST6 and decay/beta/Q DST7 according
to phase. Nothing survives `tile_regs_release`. All FP32 copied buffers use
direct FP32 unpack; BF16 state buffers use their proper BF16 format. Explicit
pack format switches and reduction masks separate persistent-state and output
packing. Native SFPU register layout, broadcast, transpose and typecast contracts
are architecture-sensitive dependencies, not interchangeable approximations.

Reader inputs and state/output queues actually hold **two heads**: CB0..2 have
8 tiles, CB3..4 have 2, state CB5 and updated CB11 have 32, output CB12 has 8.
The reader clears both slots of CB0..4 before use and only overwrites live values
after reservation. CB15 is reader-private scratch. Prepared Q/K CB13/14 hold
one tile each. The compute kernel alone consumes input queues; the writer alone
consumes published state/output queues. All state for a head is read before
that head is updated, and ownership prevents overlap with another head/core.
The writer waits for both updated state and completed output, performs its
per-head remote-completion barrier, then pops both queues. The experimental
departed-write variant is not selected. Final partial output-tile padding is
written from the reduction-masked zero row.

Two inherited comments at delivery lines 4857 and 5458 call the reader rings
one head deep. Each reserve/push/pop does handle one head, but the comments
literally describe ring capacity, which is two heads. Actual capacities and
two-slot initialization agree, so this is a documentation issue, not a dataflow
defect. Optional externally supplied recurrence output is only
shape-checked: its FP32/TILE/DRAM format, device and nonaliasing are caller
preconditions. The selected decoder uses internal allocation and satisfies them.
Calls mutating the same recurrent state must be serialized, including replay.

## Scope of evidence

[Integrated193 B32/512](integrated193_l0_sequence.json) exercised the selected
combination for 512 decode steps (minimum per-user PCC `0.9955427456116543`);
[B1](integrated193_l0_b1.json) and [B32 short](integrated193_l0_b32.json) provide
additional whole-decoder evidence. These gates test model accuracy, not bitwise
identity with Hugging Face. The exact probes compare candidate and predecessor
on identical device inputs. [Delivery manifest](delivery194_manifest.json)
records the final source and its selection provenance. Final-source validation
is owned by the parent and must be reported from its completed gate artifacts.
