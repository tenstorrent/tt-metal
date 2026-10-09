# Restored chunked-prefill state diagnostic

Queued Oct 9, **09:35:14 UTC**, after BFP8 GPQA and its independent completion
audit. This restores the earlier test whose predecessor failed before hardware
execution. The old failed receipts remain intact.

Unit `qwen38-chunked-state-v2-20261009.service` waits for the exact auditor
invocation `0213267c7b54413792a6882351321291` to finish successfully. That auditor
itself waits for the BFP8 GPQA controller to finish and release all workers.
The diagnostic then acquires `/tmp/tt-device.lock`; it cannot overlap GPQA.
Eleven-hour outer bound, 160 GiB, eight CPU quota, one-hour pytest bound.
Survives disconnect, not reboot. No serving setting is changed.

This uses the previously G0-qualified **native BFP4** model source and policy,
not the BFP8 candidate. Preflight verified the exact original runtime hashes,
precision and disabled chunked-prefill capability. The 27 tests and six
subtests passed, with one expected hardware skip.

The hardware test compares eleven full-vocabulary outputs from isolated and
interleaved requests with identical chunk boundaries, changed physical slots,
and mixed continuation/decode scheduling. It requires PCC >= 0.999, relative
RMS <= 0.02 and greedy-token agreement, using teacher-forced host logits.
Passing would validate this adapter state scenario; it would not qualify the
plugin scheduler, device sampler, BFP8 policy, long-context accuracy or speed.
Those integration checks must precede enabling chunked prefill for Tau3.

The motivation is the existing pilot's four 1,200-second task timeouts, where
completed agent calls consumed 1,034–1,149 seconds. No score is changed by this
diagnostic and no timed-out task is presumed correct. Tau3 remains 3/12.

Remote source, controller and result paths are respectively
`chunked-state-source-v2`, `chunked-state-control-v2`, `chunked-state-v2`
under `/home/ttuser/qwen38-artifacts-20261007`. Launch, frozen source manifest,
preflight XML/logs, initial queue and service identity are preserved here.
