# Completed device-session accounting correction

The old ledger overstated completed session ownership time by **2,936.643 seconds (48m56.643s)** after union deduplication. This audit found no missing historical completed interval to add. It did not run hardware, add work to reach a time budget, or change any device evidence.

| Comparison | Completed union seconds | Hours |
|---|---:|---:|
| Original10:49:47ledger | 20,429.465 | 5.674851389 |
| Same original records, excluding transcript-derived pairs | 17,492.822 | 4.859117222 |
| Fresh corrected scan, including subsequently completed direct jobs | 19,021.927 | 5.283868611 |

The fresh scan adds1,529.105seconds of later completed work relative to the corrected old cutoff. That ordinary increase is separate from the historical correction. The8hownership timebox remains unmet. These are device-session ownership durations including in-session setup/host work, not summed active kernel time or evidence that another Stop Rule is satisfied.

## Proven false pair and deduplication

`logs/autodebug_v4_profiler_launcher.log` is an OpenAI Codex investigation transcript. Atline337 it quotes the native open `2026-11-13 03:14:26.399`; atline3271 a different tool excerpt quotes close `04:45:47.248`. The parser incorrectly paired these as one5,480.849second session. `qualified_v5_with_layer34/logs/autodebug_v4_profiler_launcher.log` contains the same false pair. Counting one continuous union across them is invalid even though their duplicated spans are deduplicated.

The preserved direct job log [profile_0.log](profile_v4_failed_01/profile_0.log) proves the separate actual boundaries:

- lines12/3052:03:14:26.399→03:14:57.527,31.128s.
- lines3094/3779:03:56:13.164→03:56:37.550,24.386s.
- lines3821/3859:03:58:39.308→03:59:01.613,22.305s.
- lines3901/5294:04:45:21.291→04:45:47.248,25.957s.

Other real device sessions overlap parts of the false span, so the correct reduction after re-union is2,936.643s, not the full5,480.849s. The [machine audit](v6_device_time_transcript_fix_audit.json) enumerates **every one of the nine** previously accepted transcript-derived records: the two false spans above, plus seven pairs exactly corroborated by direct job logs and already counted there. Excluding all transcript contributions loses no independently proved old session. Endpoint-conflict scanning found only those two contradictory accepted records.

## The82original unknown records

All76marked as having another completed endpoint were independently matched to preserved **direct** session evidence. The machine audit records those matches. They add no time.

Five records describe the same failed06:02:47.665profiler job (one direct-log unmatched open and two copies each of transcript overlap/unclosed markers). The direct log ends in `terminate called` atline7699 and `Aborted (core dumped)` at7736. It has no later native driver close, UMD cluster-close or cluster-destructor marker for this job. Host Tracy postprocessing timestamps are not device endpoints. There is no supported completed interval to reconstruct; it remains excluded.

The remaining record was the then-live stress job in `logs/stress_endurance_0.log`: open10:49:41.824 atline252, now native close10:50:33.465 atline285. Its51.641s is correctly recovered by a normal fresh parser pass after completion, not by inventing an endpoint at the old ledger generation time. The fresh ledger has only the one failed direct-job unknown record.

Native source confirms the ownership boundary: `tt_metal/llrt/tt_cluster.cpp`:605 logs driver closing immediately before `driver_->close_device()`. We retain the existing conservative close-start endpoint; cluster shutdown completion or unrelated process termination is not substituted to inflate time.

## Parser correction and verification

Old [ledger](v6_device_time_ledger_1050.json) SHA256 remains `002a0a72efc3911892b4024002619616d4bb0593e0e2382ff41e88770932eb8f`. Old parser preserved [here](v6_device_time_parser_before_transcript_fix.py.snapshot), SHA256 `07a859b58909d3674425a0d7b0f994387a0af4bb7d73f2b74e73a83f76716cb5`.

Updated [parser](../../../tests/summarize_fused_device_time.py), SHA256 `936eae54f6a73eb3c62c670318184d8a923e17227100d44b6909956ad5c6baeb`, excludes launcher/queue filenames, agent transcript headers, and wrapper tail-excerpt markers **before** pairing. Even exact quoted pairs contribute zero; preserved direct logs supply their evidence. Excluded files and hashes are now reported separately. Thirteen current transcript/summary files are classified, four with no native markers. A future genuine job log named as a launcher/queue must be preserved separately under a direct-job path; conservative exclusion is intentional.

The [source diff](v6_device_time_parser_transcript_fix.diff) and machine audit prove `utc`, `iso`, `union_intervals` and the entire native pairing-loop body AST unchanged. No active/ambiguous session becomes countable. New [regressions](../../../tests/test_device_time_ledger.py), SHA256 `dfaccb0a1ccfe53893dcd26bb01f4f5c7519f64308f17f31a9b4319d3008df68`, pass all6tests: cross-job Codex excerpts, launcher/queue names, renamed tail excerpts, clipping/overlap/duplicate union, live/abort/ambiguous exclusion, and separate direct-job durations. Pinned Black23.10.1/py310, autoflake2.3.1 and isort5.13.2 pass both files.

Commands:

```bash
PYTHONPATH=. python_env/bin/python -m unittest models.autoports.qwen_qwen3_6_35b_a3b.tests.test_device_time_ledger -v
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/summarize_fused_device_time.py \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_device_time_ledger_transcript_corrected.json
```

The new [ledger](v6_device_time_ledger_transcript_corrected.json) is a dated accounting snapshot, not a replacement for future final qualification accounting. Preserve it; use a new output path when more direct jobs complete.
