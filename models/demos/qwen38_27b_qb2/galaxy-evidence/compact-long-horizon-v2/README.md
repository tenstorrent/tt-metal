# Compact GDN 4K changing-input gate

October 10, 2026 UTC. The physical run is queued, not passed. The current
full-model measurement/qualification source is unchanged.

The compact GDN block has passed 64 changing-input updates. This separate test
extends that comparison to 4096 updates at both B16 and B32 on four physical
ranks, using layer-0 BFP8 weights and four changing BF16 input tensors. Each
variant owns private FP32 recurrent state and BF16 convolution history.
Persistent inputs, state, reset tensors and both graphs are prepared before
either trace captures scratch addresses.

The control is `single_step_flat_prepare_epilogue`; the candidate is
`single_step_compact_gdn`. At every power-of-two step from 1 through 4096,
readbacks must be finite and the complete recurrent state, convolution history
and projected outputs must have identical hashes on every rank. The original
64-step integration gate retains its coverage and default. The new gate is
an exact comparison to the qualified path, not an independent dense FP32
reference, a performance measurement, or full-model GPQA qualification.

The receipt validator rejects missing batches, missing steps/ranks/state,
nonfinite results, and missing independent-session setup. Frozen preflight:
**569 CPU tests and 69 subtests passed**, one unrelated skip; the physical test
collected successfully. The first staging attempt failed six CPU tests because
the repository's `expect_error` fixture requires a message argument. No unit or
hardware was launched from that attempt. Its preflight log is retained.

Persistent unit: `qwen38-compact-long-horizon-v2-20261010.service`.
Launch PID: 258205; invocation `474d01e736f645718edb4dc861d2fbc6`.
It follows the exact prefill-attention-v2 invocation and its clean receipt,
then takes the shared hardware lock. The existing compact controls,
qualification/profile, projection, and prefill jobs were not interrupted.
The test has a 30-minute hard bound; expected hardware time is 2-10 minutes
after its predecessors finish. The waiting unit has a 28-hour lifetime,
96 GiB host-memory cap and 8-CPU quota. It survives disconnect, not reboot.

The source snapshot's complete model/config set and hashes match
compact-gdn-source-v3. Only test/controller files changed. Native installation,
weights, precision and active sources are unchanged. All outputs remain in
the task-owned host-disk directory. No serving promotion is configured.

Retained evidence includes exact launch arguments, invocation, source manifest,
successful and failed CPU preflight logs, JUnit results, waiting-state receipt,
and both staging helpers. `test_duration_s` in eventual hardware results will
include host orchestration and readbacks; it must not be reported as decode TSU.

Source provenance check found one inherited difference from the published
checkout: the frozen convolution reader predates clang-format. The retained
diff contains only line wrapping; all other model/config hashes match. The
exact frozen reader is archived with its hash for replay. Active and follow-up
model sources still match each other exactly; neither was edited.

The retained compact-v3 before-control snapshot completed its32K cell at
16.583725 TSU /60.300082ms and95.996245s median TTFT. Its16K cell was still
pending, so the arm and full comparison were incomplete. This repeats the
qualified baseline; no compact full-model speedup is established by it.
