# Root review notes

## FP32 identity-only in-place numerator

Reviewed the first `SDPA_V3_L1_INPLACE` draft on September 18, 2026.
This authorizes bounded smoke testing, not numerical/performance acceptance.

- `sdpa_inner_loop_step` takes `prev` and `cur` by reference. Swapping only
  their output aliases propagates through the outer whole-state swap, while
  sum/max CBs retain their original ping-pong ownership.
- Both numerator banks have exactly 32 FP32 tiles. Waiting/popping a whole
  fronted bank and reserving its whole capacity returns to the same physical
  base without erasing its bytes. The alternate output bank remains empty.
  This argument depends on the fixed private geometry and exact CB capacities.
- Non-final row publication totals 32 tiles. Final normalization consumes rows
  progressively; existing write offsets compensate for advancing pointers.
  Changed-max fallback consumes the previously retained bank normally.
- The full-Q guard checks all 256 finite BF16 maxima and broadcasts the same
  decision to all three RISCs. No inference from only one representative row.
- Correction-CB publication is retained even when its payload is unnecessary;
  it also provides an ordering fence for prior packs.
- Blackhole `reconfigure_packer_l1_acc` stalls configuration against PACK.
  Keep these calls around softmax/denominator writes: accumulation must never
  leak into an overwrite operation. Smoke tests must verify that subsequent
  format/width reconfiguration retains the intended accumulation setting.
- Local ISA documentation describes FP32 near-memory accumulation, distinct
  from TF32 ingress through FPU operands. It is not an IEEE rounding proof.
  The algorithm changes row-zero partial-sum association and needs actual
  original-reference accuracy tests, coherent-state/cancellation cases and
  eager/replay equality.

Required boundaries: one/two/odd-three K chunks, distinct Q jobs, final flush,
and identity→changed-maximum→identity transitions. Resident repeated KV strongly
favors this branch; distinct-input timing must be separate.

### First smoke exposed an accumulation-mode leak

The original split-drain loop enables and disables L1 accumulation only for
partial PV products after `kt_sub == 0`. The in-place draft enabled it for the
first partial product too, but retained the old conditional disable. Therefore
the following last-row softmax write could inherit accumulation rather than
overwrite. Root identified this after normal inputs failed badly while constant
V remained exact. Proposed repair: disable after every in-place partial PV,
including the first, before the next softmax operation. Device confirmation is
still required. Constant V alone cannot catch this: numerator and denominator
can share the same corrupted weights and still produce the constant value.

### Extension to denominator and publication-wait correction

The follow-on draft also reuses the eight-tile FP32 denominator bank. Phase-1
sum packing is compiled out in FP32 streaming, so its early reserve does not
write the alternate bank. A per-RISC flag is reset at each step and suppresses
the outer cleanup pop of the now-empty old-sum alias. Resident and fullchip
wrappers both call the same standard outer loop.

Root found that the shortcut bypassed `salad_correct_fused`'s explicit
correction-CB `wait_front`, not merely its arithmetic. Retaining push/pop alone
does not retain that handshake. Requested explicit `wait_front(sbh)` before
the shortcut's pop, including the numerator-only path. No promotion from
earlier tests that omitted this wait; requalify the corrected candidate.

## Group-two compensation

Independent reviewer audits the actual shared group-two implementation;
see `review/GROUP2_REVIEW.md`. Root additionally checked that a nonboundary
local group holds one already-rounded BF16 PV chunk, not a BF16 running sum.
This is why two-chunk grouping deserves testing while larger groups failed
the scalar filter. Neither scalar model establishes attention accuracy.

## Early FP32 identity scan

Rechecked the frozen `fp32/early_guard.hpp` after hardware recovery. The
UNPACK thread scans each previous row only after that row's max publication
and after issuing the next QK operation. It keeps a private provisional flag;
only the final whole-Q decision is broadcast to MATH/PACK. All 256 row maxima
must match and be finite. The final tile row is scanned at the original guard
site. The change therefore moves scalar guard work into available overlap,
without changing the accepted predicate or bypassing publication waits.

The specialization asserts one Q tile row per QK subblock and rejects the
alternate `SDPA_FP32_PIPELINE` loop, which does not execute these progressive
scans. In-place CB reuse remains restricted to Q256/K512/D128, noncausal,
non-ring, no attention sink and ordinary output handling. These restrictions
are important: device qualification here does not authorize broader dispatch.

## Recovered unchanged-A control

`unchanged-A-resident-v2.json` is a fresh unchanged baseline on reservation
224379: 137.965306 ms, 1.992370 useful TFLOP/s/core, 14 exact trace replays.
Selected sources and original device inputs were unchanged. The 20.276227%
L2 is for a 512-times-repeated KV tile, not a distinct normal 256K sample;
do not present it as representative model accuracy. Timing is a compute-side
control, not an end-to-end model speedup.

For common V, an offset can hide variation loss in both global and row L2.
Qualification should also report error normalized by the reference with its
known common mode removed. This is a diagnostic, not an invented extra hard
acceptance threshold. Shared `numerics.py` remains frozen.
