# Parent-owned d810 matched selection queue

Prepared source-only while the parent owns the active long benchmark. No hardware launch, TTNN/Torch import, oracle tensor load, large tensor hashing, or benchmark occurred during preparation. No git/staging/shared-doc edit.

From repo root, host command/source validation only:

```bash
bash models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_d810_selection_queue/queue.sh
bash models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_d810_selection_queue/queue.sh --with-cached-controls
```

After all other device/CPU work closes, parent explicitly starts:

```bash
bash models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_d810_selection_queue/queue.sh --run
# Or include six ordinary cached comparator controls after all nine timings:
bash models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_d810_selection_queue/queue.sh --run --with-cached-controls
```

Timing order is block1:44f87,85ad,d810; block2:d810,85ad,44f87; block3:85ad,44f87,d810. Every fresh process invokes the unchanged standard `benchmark_fused_snapshot.py`, layer0/S256/context512/10prefill/60decode, default seed42. All instrumentation/topology overrides are unset; capture/watcher/tracking are off. Source/command pins checked before and after every process. Each job has1800s timeout plus60s TERM-to-KILL grace and `fuser /dev/tenstorrent/0` immediately before launch; busy/failed query stops the queue. No monitor or other child runs concurrently with a measurement.

Outputs are fresh `E/v6_selection_{44f87|85ad|d810}_repeat{1|2|3}.json` and `.log`, plus harness provenance. All15 possible job paths were absent during preparation. Optional controls use pending manifest-compatible `v6_transpose_layer0_seed{913|42|154}` and `v6_transpose_interleaved_layer0_seed{913|42|154}` names. Each ordinary `stress_fused_snapshot.py` uses prompt128/decode8192/continuation33/repeats1, tracking1/skip-program-cache0. Oracle paths are **read from prior provenance**, including seed154's actual `formatted_linear_seed154_8192.pt` (not a guessedv6name). Seed913/42 use the recordedv6_layer0 paths. No reference state/output injection or final fresh-HF acceptance claim is introduced.

Stop this queue between jobs by creating `models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_d810_selection_queue/STOP_SELECTION_QUEUE`. It never reads/removes/reuses `doc/fused_decoder/STOP_QUEUE`, which remains the final suite's independent stop mechanism. The stop flag does not interrupt an active job; that job's logs/result are validated and preserved before stopping. Existing run status/output/log/provenance files cause a refusal; there is no implicit resume, overwrite or automatic retry. After a stopped/failed run, parent must explicitly prepare new paths or review remaining jobs; preserved evidence is never erased to rerun this script.

`run_status.json` is exclusively created only by `--run`, then atomically updated with per-job exact command, source pins, timestamps, exit status, logs/hash, validation/hash, completed samples and pending jobs. A nonzero exit, timeout, source mutation, malformed result or gate failure stops further launches and records `failed_preserved`. Combined stdout/stderr logs are written with exclusive creation. Timeout failures retain all partial artifacts; the script performs no reset/recovery.

After each benchmark, host checks exact measured source/provenance,10/60sample counts, positive finite samples/medians, all4PCC fields≥0.995, deterministic output, runtime auditPASS, clean serial environment and workload. Optional controls invoke the unchanged pinned `check_comparator_endurance` function via AST extraction (avoids importing broad audit entrypoints), retaining every source-bound8192+33/seed/tracking/oracle gate. No final-comparator ranking is performed; all repeat controls must remain available for the later best-correct assessment, including inconvenient results. No source is selected or copied into production.

`queue_plan.json` is the reviewable exact command inventory. `host_preparation_checks.json` records all command/source validations, and `pinned_inputs.sha256` binds runner/plan/README/harnesses/snapshots/checker/prior oracle-provenance. Validation does not callfuser or spawn a device process. Parent owns launching the queue and final performance interpretation.
