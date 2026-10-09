# Persistent decoder precision comparisons

Both controls completed with clean shutdown at **09:25:38 UTC, Oct 9**.
The BFP4/HiFi2 control reproduced every saved full-logit metric from the
BFP4/LoFi baseline. Higher arithmetic fidelity alone provided no measured
improvement on this prompt. With decoder BFP8/HiFi2, mean relative logit RMS
error across the eight positions fell from **38.65% to 9.16%** (76.3% lower).
Top-20 overlap improved from 11–17 to 17–20; all eight top-1 predictions agree
with HF under both policies. Initial prefill layer-63 input RMS error fell
from 88.56% to 22.49%. These are cumulative full-decoder errors, not individual
layer errors or task scores. The short teacher-forced probe supports testing
BFP8 on GPQA; it does not establish the reason for every benchmark failure.

The matched [full GPQA follow-up](../decoder-gpqa-v1/README.md) uses BFP8.
Runtime source, reference and policy hashes, both complete comparisons, the
final queue, hardware test XML and logs are retained here. Controls took
326/346 seconds, including model loading; these are not inference timings.

After the completed [reference comparison](../hf-reference-recovery-v2/README.md),
two matched-input experiments were launched at **09:14:25 UTC, Oct 9**:

1. Keep decoder weights BFP4; change decoder projection arithmetic to HiFi2.
2. Keep HiFi2 arithmetic; change decoder weights to BFP8.

Both retain the BFP8/HiFi2 head, BFP8 KV, FP32 recurrent state, native recurrence
and accurate-full-tile attention. Each loads one full TP4 replica and compares
the same eight BF16 HF teacher-forced positions and all decoder inputs. They
are diagnostics, not GPQA, performance measurements or serving qualification.
The existing G0 still authenticates the baseline source/partition/precision;
the explicit candidate policy is separately recorded as unqualified. Validation
rejects changes to the head, recurrence, cache or nonprojection settings.

The first controller failed at import because its frozen bundle omitted
`run_chunked_prefill_followup.py`. It created no result directory and opened no
hardware. The failed unit and log are preserved. The corrected v2 bundle
includes the helper, tests its imports and passed **34 host tests**, with the
hardware test correctly skipped in preflight.

Unit `qwen38-hf-controls-v2-20261009.service` runs both controls sequentially,
with a two-hour outer bound, 160 GiB RAM, eight CPU quota and the shared hardware
lock. Each test has a 40-minute pytest bound and 50-minute process-group bound.
A failed test or unproven cleanup stops the sequence. Source, controller,
launch, service identity and initial queue receipt are retained here. It
survives disconnect, not reboot. Neither policy is promoted or GPQA-qualified.

Higher fidelity/weight precision may reduce throughput; there is no performance
claim from these tests. A candidate that reduces numerical error still needs
eight-replica G0, the unchanged full 198-question GPQA gate and measured serving
performance before promotion. See the [capacity estimate](../decoder-precision-controls-v1/README.md).
