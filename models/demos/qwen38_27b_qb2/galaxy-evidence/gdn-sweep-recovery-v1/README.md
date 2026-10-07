# Matched TP4 sweep: preserved failure and recovery evidence

The baseline graphs in [baseline-preview/index.html](baseline-preview/index.html)
contain 12 measured native-recurrence points with input throughput, decode
throughput, per-user decode speed, TTFT and output throughput over the whole
request. They are partial baseline measurements, not candidate speedups.

- `baseline-v1.json` and `baseline-v1.xml.gz`: original monolithic sweep,
  including the B32 / 32K prefill allocator failure.
- `fresh-b32-v1.json`, `fresh-b32-v1.xml.gz`, `launch-fresh-b32-v1.json`:
  the same configuration also fails in a fresh process, without preceding
  sweep points. It closes the devices cleanly.
- `unit-v3.xml.gz`: 203 CPU tests and 40 subtests pass, real tokenizer skipped
  because this preflight did not set `MODEL_WEIGHTS_DIR`.
- `startup-v2.xml.gz`: startup correctly stops before device work when the
  real-tokenizer test finds its missing baseline fixture in source snapshot v7.
- `unit-v4.xml.gz` and `gdn-sweep-recovery-unit-v4.log`: all 204 CPU tests and
  40 subtests pass after copying that unchanged fixture into snapshot v8.
- `collection-v3.xml.gz`: omitted `pytest.ini` allowed collection outside the
  source tree, into an unrelated stale weights mount. No device test ran.
- `unit-v5.xml.gz` and `collection-v1.xml.gz`: all 204 CPU tests and 40 subtests
  pass with the real tokenizer, and hardware-test collection passes through
  the safe runner with explicit root/config paths and restored `pytest.ini`.
- `launch-gdn-perf-sweeps-v2.json` through `launch-gdn-perf-sweeps-v4.json`:
  exact persistent commands, snapshot file hashes and native runtime revision.

The v4 recovery run uses the preserved v1 native results only after checking
source/policy hashes, runtime knobs and measurement accounting. The native
and single-step recurrence variants each receive up to six hours; the outer
systemd unit has a 14-hour limit. Each DRAM allocation failure is recorded,
then device cleanup must complete before the controller starts a fresh
process for remaining cells. Other failures stop the sweep. `completed_with_oom`
means the loop finished with explicit unsupported measurements, not that all
configurations passed. No candidate results or full-Galaxy results are included
in this snapshot.
