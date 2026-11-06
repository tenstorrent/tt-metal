# Retained-token capacity

Default CI retention is4GiB per engine process, with one-second export and1,024events/chunk. The local live pair's64MiB limit is not the CI configuration. Current default fits the calibrated workloads; no configuration or runtime change is justified now.

| B8 workload | Conservative bytes/decode step | Full-cap estimate |12h at measured34.05steps/s |12h at40steps/s |
|---|---:|---:|---:|---:|
| SWE32768,C2 |1048 |36.51MiB |1.60GiB |1.85GiB |
| GPQA126976,C4 |1080 |145.35MiB |1.64GiB |1.90GiB |

Estimates include both submission/completion, all8 physical token rows,19-digit clocks,8-digit step IDs,6-digit positions/IDs, full remap and4KiB/s framing allowance. Actual empty-frame maximum is1826bytes. Measured live B8 peak one-second rate35steps/s; B1 peak40. Sustained12h break-even is90.96/88.26steps/s before additional prefill records. These are scenario calculations, not a guaranteed bound on arbitrary traffic or disk stalls. Prior calls share the same file: reserve at least36.51/145.35MiB before a new cap-sized call (plus prefill and safety margin). At the calibrated rate, prior consumption must be below approximately3.964/3.858GiB respectively. A64MiB cap cannot retain the full GPQA call.

The1,000,000-event in-memory limit is reusable after successful export, not a lifetime limit. At34.05steps/s, about68.1events/s is far below nominal1,024events/s export capacity; an entirely stalled exporter fills the buffer after about4.08h of decode alone. JSON size is not Python heap size. Pending association state is separately bounded64steps.

Crucially, this is prefix retention, not a rolling tail. File exhaustion replaces the entire next export chunk with a loss marker, clears remaining buffered events and stops all future retention. A capped call arriving after exhaustion is lost. Buffer overflow instead drops newest events until a drain frees room and leaves sticky loss status; submit/complete pairs may be partial. Disk failure preserves only already-written data, marks failure and stops retries; missing final status is incomplete evidence. None silently proves a full capped segment.

Three existing host tests confirmed file-cap terminal markers, buffer overflow and short-write/failure behavior. No devices, runtime edits or dispatch. If a normal run's measured prior bytes/projected traffic exceed the cap, the minimum repair is an existing GEMMA4_PHASE_MAX_FILE_BYTES configuration increase with disk reserve, confirmed by host serialization and cap tests. Do not introduce eviction that destroys the late-call evidence being sought. Normal-run final loss/pending checks remain mandatory.

Evidence: calibration.json includes exact source artifact names, measured file/event/frame sizes and scenario formulas; analyze.py reproduces it; bounds-checks.log records tests. Accepted retention change remains accepted; this audit does not classify the missing SWE response.
