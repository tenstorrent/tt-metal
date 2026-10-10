# Epilogue unused-row experiment: implementation and persistent queue

The qualified compact profile attributes 2.853 ms of kernel time per model
step to the GDN epilogue. Its reader initializes two FP32 and BF16 input-ring
windows: 48 KiB of stores per worker invocation. Every head then overwrites
only row zero. The arithmetic reduces across columns independently per row;
the compact writer emits row zero and separately clears destination padding.

The default remains `input_padding="zero"`. A new standalone opt-in `"skip"`
omits the input clear, and `"poison"` fills both ring slots with quiet NaNs
before overwriting live rows. The latter tests row isolation and is never the
timed candidate. Nonzero modes reject public full-tile output. Compute, BF16
rounding, input/output ownership and writer padding behavior are unchanged.
No model or serving policy enables either new mode.

The engineering estimate is **1-2 ms per full step**, approximately **2-4%**
or **20.44-20.87 TSU** from the qualified 20.035 baseline. This is unmeasured,
may be reduced by overlap with compute, and is not added to the running combined
GDN experiment as a claimed gain. Thirty TSU remains unachieved.

## Gates and queue

Frozen CPU validation passed **709 tests, 104 subtests, one skipped** in 4.29 s.
Hardware test collection passed. Black, isort, clang-format and other applicable
pre-commit checks passed before freezing. The initial attempt had 13 CPU test
failures because the repository's `expect_error` fixture requires a message
argument; all occurred before hardware launch. The corrected second attempt
uses fresh directories and preserves the first attempt's logs and JUnit.

The hardware plan is:

- Eighteen four-rank cases: B16/B32 with public, compact and packed-offset
  gates in L1/DRAM; B1/B17/B31 with packed gates in both placements.
- Exact native-epilogue normalization and multiplied output, with zero output
  padding, NaN-poisoned input rows, A/B/A allocation rebinding, stable input
  addresses and changed-input trace replay for both skip and poison variants.
- B32 must exercise both input CB slots and wrap to a reused slot. Only
  zero/skip/zero brackets receive timing credit, subject to the existing 3%
  control-drift gate.
- Actual BFP8 layer weights with independent sessions, FP32 recurrent state,
  and 4096 changing-input updates at B16 and B32 for both skip and poison
  versus zero padding. State, convolution history and projected output must
  match exactly on all four ranks. This is not independent dense-reference or
  full-model/GPQA qualification.

`qwen38-gdn-epilogue-padding-v2-20261010.service`, invocation
`2cb03803508147cca833c466023b1a52`, PID 500939, waits for exact combined-followup
invocation `8e20a69d96884f5a8f9346d08d638e64` to complete and release hardware.
This preserves the current full-model comparison and conditional G0/API/GPQA
priority. At the October 10 22:23:23 UTC capture, all three units were live;
the new unit was waiting with `hardware_started=false`.

The unit has a 28-hour lifetime, 32-GiB host-memory and eight-CPU limits;
hardware execution is bounded at 1800 s and uses `/tmp/tt-device.lock`.
It survives disconnect, not reboot. Allow roughly 5-15 minutes of hardware
time after its predecessors finish, subject to compilation. Source, launch
commands, tests, unit properties and the preserved failed preflight are included.
The live combined run completed its first full-model arm at 22:17:20 UTC and
was measuring the candidate at 22:23 UTC; neither its speedup nor this new
padding experiment is qualified by the queue receipts.
