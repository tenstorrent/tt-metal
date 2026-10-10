# Compact GDN gates: implementation and queued physical validation

October 10, 2026, 20:58 UTC snapshot. The candidate is implemented and CPU
preflight passed; no physical result or measured speedup exists yet.

The measured compact decoder keeps Q/K/V and z packed but expands its 64
a/b projection channels from `[1,B,64]` to `[B,1,64]`. That creates B padded
tile planes for the subsequent small gate operations. The new
`compact_gdn_gates=True` layer policy retains `[1,B,12]` log-decay and beta,
including all user rows through exponentiation. Native sigmoid, FP32 typecast,
bias addition, softplus, multiplication and exponentiation are unchanged.

The direct preparation reader optionally reads gate rows from one tiled
plane. It handles the second face, odd BF16 user rows and Blackhole DRAM
alignment explicitly; FP32 decay rows are 64-byte aligned. Default gate layout,
FP32 recurrent state, recurrence, convolution and epilogue remain unchanged.
The new policy defaults false and is selected only by the isolated test; no
serving environment switch or precision-artifact promotion was added.

Engineering target: **0.5-1.5 ms per B16 full-model step**, approximately 1-3%
from the measured 49.913-ms compact baseline. This is an estimate pending the
matched profile and hardware timing. It overlaps the roadmap's remaining gate
layout/fusion allowance and must not be added to that allowance again.

## Validation and queue

- Frozen CPU preflight: **585 passed, 1 skipped, 91 subtests passed**; physical
  test collected and launcher shell syntax passed. Precommit passed after
  formatting. CPU checks cover all-user arithmetic/shape preservation, the
  adapter's exponentiation of every compact row, invalid layout selection, and
  missing/forged physical coverage. They do not establish native kernel math.
- Physical component cases: B1/16/17/31/32 in L1 and DRAM, with independent
  inputs on four ranks. Compare all four prepared operands bit-for-bit against
  the existing public-gate path, including extreme gates and user boundaries.
  A/B/A allocations, changed-input traces, immutable inputs and stable addresses
  are checked. Five 100-replay timing samples per before/candidate/after arm
  include gate math and preparation, with a 3% control-drift limit.
- Real-weight B16/B32 tests run two sessions through 64 changing-input updates.
  They compare FP32 state, convolution history and projected output on all ranks
  at steps 1/2/4/8/16/32/64. Persistent allocations precede both trace captures.
  This is an exact comparison with the existing compact decoder, not an
  independent dense-reference, 4K-step or full-model GPQA result.
- Unit `qwen38-gdn-compact-gates-v1-20261010.service`, PID402442, invocation
  `37fb5d460eed4792841823a8ba71d855`, was live and waiting at capture. It follows
  the exact register-resident experiment invocation
  `9a87b6620b6c4892b407a17f7c54f04e`, after the existing GPQA/profile,
  projection, prefill and long-horizon queue. No predecessor was reordered.
- Test timeout 1,620 seconds, wrapper 1,800 seconds, 16 GiB host RAM, eight CPU
  equivalents, shared device lock. Controller lifetime is 28 hours. Persistent
  across client disconnect, not host reboot. Source hashes are verified before
  hardware starts; correctness does not automatically promote the model.

Source, control and output directories use the `gdn-compact-gates-*-v1` names
under `/home/ttuser/qwen38-artifacts-20261007`. The nine changed source/harness
files match the frozen manifest. Existing native installs, active GPQA source
and earlier frozen experiments were untouched.

The captured GPQA progress is 168 correct of 180 completed, no truncations,
18 questions remaining. That partial receipt is not a final score. The current
measured performance remains **20.035 TSU at B16/32K/TP4**.
