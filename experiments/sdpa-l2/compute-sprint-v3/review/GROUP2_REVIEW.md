# Independent group-two review

This is a source/design review, not device approval or a performance result.
The implementation remains owned by `../compensated/`.

## Mathematical state

Let `R = float32(H + L)` be protected two-BF16 state, `U` the local BF16
contribution, `X` the current BF16 PV chunk, and `a` the unchanged canonical
maximum correction. Both `R` and `U` use the previous global maximum.

- Bootstrap: `H=X`, `L=0`, `U=0`.
- Odd-indexed, unchanged-maximum, non-final step: `U=X`.
- Even-indexed or final unchanged-maximum step:
  `split(float32(R + float32(U+X)))`, then `U=0`.
- Any changed/nonfinite maximum:
  `split(fma(float32(R+U), a, X))`, then `U=0`.

Here `split(y)` stores BF16-rounded high and BF16-rounded FP32 residual. The
actual instruction rounding and fused MAD must be checked in generated code.

Because every even step and every changed-max step clears `U`, every odd
nonboundary store encounters an empty local state. Consequently group size two
does not perform a rounded BF16 running accumulation inside the group: its
local operation is an exact copy. This is the key difference from grouping
three or more chunks. The arithmetic association and the number of protected
residual roundings still change; neither bit equivalence nor a universal 5%
relative error bound follows from this argument.

Changing maxima must correct **both** protected and local state before adding
the new contribution. Final partial groups must flush, including an odd final
index. A changed maximum on an odd step clears local, so the subsequent even
fold safely consumes zero. Denominator remains canonical.

## Publication and lifetime obligations

1. CB8 protected high/low is initialized and fronted afresh for every distinct
   Q job. CB9 local state is explicitly zeroed for every Q job; no reliance on
   prior final-branch contents.
2. CB8 stays fronted during K steps. In-place absolute writes must use its
   physical base, independent of normal read-front advancement at final row
   normalization. Root reads and writes must not race previous/next rows.
3. CB9 PV writes only high slots. Local resides in the old low slots; old
   first-iteration residual-zeroing code must not accidentally clear live local.
4. Existing correction-CB publication fences prior PV packs. Even identity
   branches must preserve the publication token unless replaced by a proven
   fence. The v1 explicit extra PACK_DONE fence was removed only because this
   release/acquire already orders the PV chunk.
5. Root updates and local clears must be retired before a later K reads them;
   same-final-K normalization must see root updates, not merely rely on a
   publication that occurs in the following K iteration.
6. CB9 push/pop counts and physical wrap must balance once per K. CB8 is popped
   only as final normalization consumes rows. Max/sum ping-pong remains separate
   from fixed root/scratch identity.
7. Normal DST release clears its half. All retained operands must be copied
   within the acquired lifetime; never rely on correction/local DST surviving
   an ordinary release.

## Independent scalar probe

`state_probe.py` and `state-probe.json` cover 8,192 recurrences over
1/2/3/4/5/16/64/512 chunks, four correction patterns and four chunk distributions.
They assert local-empty and final-flush invariants, model FP32 intermediates and
BF16 nearest-ties-away residual splitting, and leave denominator canonical.

All invariants passed. Largest constant-V pre-output-pack ratio change was
0.007076%, at 512 identity-correction chunks. Final BF16 packing is not modeled,
and this number is not a hardware SDPA error. Scores, PV arithmetic, exp and
reciprocal are also outside this recurrence model. Hardware qualification must
use original-input FP64 reference and the per-case v3 budget.

## Numerical acceptance helper

Reviewed shared `../numerics.py`: percentage-point floor, finite rejection,
exact-zero-reference absolute gate, nonzero-row quantiles and zero-row
diagnostics match the v3 contract. Constant-reference PCC is undefined rather
than spuriously reported as one. No blocking issue identified.

## Integration review before the first device run

Read `group2/compute_streaming.hpp`, `operations.hpp`, and `state.hpp` against
the frozen v2 early-guard header. No blocking source defect was identified for
the fixed noncausal geometry. In particular:

- Existing first-K scratch-low zeroing resets local, while the new root
  initializer zeroes all protected slots each Q.
- Root read indices follow its advancing read pointer during final row
  normalization; root write indices remain global relative to its wrapped
  physical write base. Scratch writes remain relative to its row-advanced
  write pointer.
- The current sum/scratch push precedes normalization and publishes root
  writes in the same K step. Later correction publication precedes the next
  numerator read. Bootstrap has an explicit prior-PV publication fence.
- Only max/sum ping-pong after each K; output CB identities are restored to
  root8/scratch9. Root and scratch pop totals balance at final normalization.
- Compensated denominator macro initialization follows the custom SFPI body,
  reinstating canonical replay state.

Approval is for bounded compile/K1 then K2/K3 multi-Q testing, not production
correctness. Generated SFPI fused MAD and rounding still need inspection.

## Direct-PV-local sibling

Reviewed the separate `group2_direct` diff. Every even-indexed step flushes and
clears local; consequently every odd-indexed step starts with local empty for
every row, regardless of intervening changed-max decisions. Odd PV can therefore
write directly into the local low plane, eliminating the odd identity-step copy
without introducing a validity bitmap.

On an odd changed-max or final step, the helper explicitly uses zero for the
**old** local state and reads the low-plane **new** chunk exactly once. It folds
and clears low. Even steps retain the original two-input group fold. Both
materialized-V pack sites add the low-plane column offset; their doubled physical
row stride and first-partial-overwrite/subsequent-partial-accumulate behavior are
unchanged. Publication, CB ownership and denominator paths remain unchanged.
No source blocker was identified for bounded K2/K3/transition tests. This is not
a numerical or performance approval.

## Empty-local fast fallback

Reviewed `group2_valid` before its bounded test. Four booleans are reset at each
Q job on all three RISCs and passed through the inner-loop call. Their index is
the global row-group `salad_row`, not the popped-relative write index. The
mailbox-synchronized identity decision updates each thread's flag identically:
only odd, unchanged, non-final steps retain valid local state.

The helper first handles the odd nonboundary identity no-op. Its next branch
handles **all** empty-local folds, both identity and changed maxima, using the
frozen paired compensated replay. It copies the current chunk from scratch and
packs the high/low result to root. Only after that branch returns can any helper
read a physical old-local operand. Thus stale local slots may remain physically
uncleared without being read when invalid. Remaining true-local identity/changed
folds retain the previous group-two math. Removing `root+0` may affect signed
zero; it does not authorize changed correction/rounding or nondeterminism.

The independent `validity_probe.py` deliberately carries stale local storage
across folds and Q jobs. 9,216 modeled row-group recurrences (2,304 four-group
scenarios) matched the original group-two model. A negative control demonstrates the required even
identity check: chunks[1,2,3], corrections[1,0.5,1] correctly produce5.5;
incorrectly reusing stale local after the odd changed-max step produces7.5.
This supports the state-machine argument, not hardware acceptance.

## Even-identity replay review

The retained18-op helper uses DST planes hi0/hi1, lo0/lo1, local0/local1,
chunk0/chunk1. Frozen root-add macros produce L1/L2; local-plus-chunk produces
L4/L5; the final root-plus-group produces L0/L3. The original round, residual
subtract and store macro tail follows. DST offsets0/64/128/192 and paired
dependency spacing match the two-tile layout. Only the final store macro
advances ADDR_MOD7; the final three NOPs and SFPU drain remain.

Its replay slots0..17 overlap denominator slots15..17, so the caller's
denominator replay reinitialization before use is essential and retained.
The empty-local helper uses the frozen paired seven-slot layout instead.
No numerical coefficients, rounding templates, fidelity, denominator recurrence,
or reader/writer/input-slot geometry changed.

The final62-case B device qualification reproduces every original grouped
output hash on the43 matched cases and adds19 held-out cases. This is concrete
evidence for the reviewed transformations at Q256/K512/D128, not a proof for
unsupported ring, causal or arbitrary-shape instantiations.
