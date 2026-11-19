# Three-slot convolution scheduling candidate

Source-only AutoFix hypothesis on frozen `candidates/integrated134.py`.
No hardware execution, numerical result or performance improvement is claimed.

The current untilize-once convolution assigns `B*10` independent 1,024-channel
tasks to up to 110 cores. At B32, 100 cores receive three tasks and ten receive
two. The two-slot implementation processes a group of two, waits for both
writer completion tokens, then pops that entire input group. Only after this
can the reader fetch the third task. This creates a potential scheduling bubble
on the 100 busiest cores even though the third task is independent.

The candidate increases all ten BF16 CB rings from two to three pages. Reader
code is unchanged. Compute groups up to three tasks, and writer's physical
input-slot selection uses `(task-start)%3`. The third input can now be fetched
while earlier tasks compute and write. No arithmetic instruction, input layout,
untilize call, task ordering, persistent history allocation or output publication
rule changes. Writer retains a NoC completion barrier for **each** task; this
does not eliminate those required barriers. It removes the intermediate
whole-group release prerequisite for fetching task three.

CB allocation grows from 40 to 60 KiB/core. At B1 and B3 each core handles one
task, so no scheduling improvement is expected. Increasing buffer depth can
also hurt through additional memory pressure or scheduling differences; only
device measurements can select this candidate.

## Artifacts and source checks

- Helper: `candidates/conv_decode_sfpu_untilize_pipeline3.py`.
- Frozen full decoder: `candidates/conv_pipeline3_on134.py`.
- Generator: `candidates/make_conv_pipeline3_candidate.py`; defaults to
  `integrated134.py`, refuses overwrite, replaces only the active namespaced
  convolution call and appends a separately namespaced helper.
- Check: `candidates/check_conv_pipeline3_cpu.py`.
- Check results: `candidates/conv_pipeline3_cpu.json`.

The checker verifies that numerical compute source is identical after only the
group-bound substitutions, and that reader source is identical apart from
comments. Its finite-state model enumerates all enabled interleavings of input
publication, computation, writer issue, writer completion, and group release
for 1,2,3,4,5,6,7,10,20,40,80,160,320 tasks. No reachable deadlock or overwrite of
live input/output slots occurs. Input pages are popped only by compute, after
all writer acknowledgements for the group; the writer's read pointer remains
at the physical ring base. A final partial group is safe because no task follows
it. This model validates the intended protocol, not device execution.

Commands already run without device access:

```bash
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/check_conv_pipeline3_cpu.py
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/make_conv_pipeline3_candidate.py --output conv_pipeline3_on134.py
python -m py_compile models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/conv_decode_sfpu_untilize_pipeline3.py models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/conv_pipeline3_on134.py
```

## Recommended serialized device checks

Use the parent's authorized device queue. The existing isolated precision
probe supports this helper without modifying the harness:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_conv_precision_probe \
  --candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/conv_decode_sfpu_untilize_pipeline3.py \
  --baseline models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/integrated134.py \
  --batch 1 3 32 --steps 4 --require-baseline-exact \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/conv_pipeline3_probe.json
```

Then select frozen `conv_pipeline3_on134.py` through the usual evidence decoder
class/kwargs override and compare warmed traced B32 whole-layer latency to
integrated134 with identical weights/configuration. Require unchanged output
and history, changed-input repeat coverage, and the existing long-sequence
acceptance gate before keeping it. Kernel-only latency can diagnose the result;
it does not replace whole-decoder latency. If there is no consistent gain,
retain the two-slot implementation and record the measured rejection.

The separate grouped-Q/K reader elision was **not recreated**: parent evidence
for `grouped126` already shows no consistent whole-path gain. The present
candidate changes only convolution scheduling and can be combined independently
with recurrence SFPU work if separately measured and correct.
