# Persistent capture of the unchanged full BFP8 GPQA run

The user requested letting all 198 questions finish and capturing the result.
The original evaluator remains untouched. Its exact invocation was verified
live at 10:53 UTC; recent server logs confirmed the five remaining requests
were generating, rather than stalled. At the capture launch it had advanced
to 196/198 completed, 170 correct and zero truncations. This is not a final
score and cannot reach the unchanged 177/198 gate.

The existing independent audit already waits for the exact original run to
finish and shut down its workers. A second persistent, read-only follower now
waits for both original service invocations and completed receipts. It saves:

- The original full summary, sampling protocol and scored response metadata.
- The independent raw-response hash/cutoff audit and terminal service state.
- Final deployment/source/precision and worker-shutdown evidence.
- Per-question matched outcomes against native BFP4 and the BFP8-head control.
- Client TTFT, decode rate, request duration and output-length distributions,
  including mean, median, p90 and p95, plus full-interval aggregate output rate.
- Final server/evaluator logs, compressed, and a byte-count/SHA256 capture manifest.

It preserves the original scores. One sampled run per policy is not proof of
numerical causality; client metrics are not isolated device-kernel throughput.
Private benchmark prompts and response text remain in the original protected
run directory and are not copied into this publication capture.

- Unit: `qwen38-decoder-gpqa-capture-v1-20261009.service`.
- PID at launch: 1601319; invocation `75fb7d86dbbe47d9ad4af9985bac6014`.
- Output: `/home/ttuser/qwen38-artifacts-20261007/decoder-gpqa-capture-v1`.
- Frozen source: adjacent `decoder-gpqa-capture-source-v1`.
- Limits: one CPU, 512 MiB, nine-hour observation deadline, ten-hour service bound.
- Survives session disconnect; does not resume after reboot.

The terminal reader was validated read-only against the completed head-control
run by adapting only its artifact paths and exact service identities. It
reproduced 166/198, checked audit/receipt hashes and matching protocols, computed
paired/timing data and generated all ten valid metadata/log artifacts. This is
a collector test, not a new BFP8 eval. Its exact adaptations and source hashes
are in `completed-control-validation.json`. The current frozen reader was not
modified during that validation.

The capture job was subsequently verified alive and waiting. `status-at-collection.json`
is that snapshot, not a terminal result. All source, staging and validation
scripts are retained as gzip to preserve their original bytes. Local result
collection can use the retained `collect-decoder-gpqa-result-v1.py.gz` script
after completion; it writes no destination until both exact jobs are terminal
and the complete result validates, and refuses an existing destination.
