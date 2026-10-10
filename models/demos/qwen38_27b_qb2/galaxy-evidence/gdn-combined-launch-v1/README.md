# Combined resident-state and compact-gate model experiment

A new explicit policy, `precision_single_step_compact_gdn_resident_gates_bfp8_all.json`,
selects both tested optimizations at B16/B32. The original qualified policy file
is unchanged. Weights/KV remain BFP8, activations BF16 and recurrent state FP32;
smaller decode buckets and prefill retain their existing paths. This is an
experiment, not a replacement for the GPQA-qualified launch.

Frozen-source CPU preflight passed **693 tests, 95 subtests, one skipped**.
The physical combined-versus-qualified-compact comparison then passed **4096
changing-input steps at B16 and B32**, with exact recurrent state, convolution
history and projected output hashes on all four ranks. This integration test
compares implementations; standalone resident-state dense-reference evidence
is recorded separately in the prior follow-up report.

Persistent comparison `qwen38-gdn-combined-v1-20261010.service`, invocation
`15bebec92d7649909517ba9a4a5a3778`, runs the 4K gate followed by full 64-layer
before/candidate/after measurements at B16/32K and B16/16K. Each arm uses one
warmup plus three natural-prompt measurements, with fresh prefill and 128 output
tokens. The candidate is compared to the qualified compact policy, not the older
flat-boundary baseline. Source, precision, inputs and timing receipts are
reconciled before any speed claim. Predicted saving is approximately **0.94 ms**,
or **20.42 TSU at 32K** if additive; neither is a measured full-model result yet.

Persistent follow-up `qwen38-gdn-combined-followup-v1-20261010.service`, invocation
`8e20a69d96884f5a8f9346d08d638e64`, waits for a clean matched comparison with at
least 1% B16/32K gain. Only then does it run eight-replica G0/API validation and
full 198-question GPQA using the exact frozen candidate. No extra full profile
is scheduled for this incremental change. Both units have 24-hour limits,
256-GiB/16-CPU bounds and the existing hardware lock; they survive disconnect,
not reboot. A gain or queue receipt alone does not promote the candidate.

Launch estimates: 35-60 minutes for the full comparison; a further 70-100 minutes
for conditional G0/API/GPQA, including setup. Estimates are not deadlines or
promises. The retained 22:03:07 UTC snapshot shows the comparison loading its
first full-model arm and the follow-up waiting. Exact commands, source manifest,
CPU JUnit and 4K hardware receipts are included here.
