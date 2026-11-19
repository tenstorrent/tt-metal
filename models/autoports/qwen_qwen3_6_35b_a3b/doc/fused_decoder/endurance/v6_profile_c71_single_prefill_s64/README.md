# Validated c71 profile: cold S64 prefill and traced decode

Capture and report validation **passed** for linear-attention layer 0 on physical Blackhole chip 0. This is a complete first-call/cold S64 prefill table and 16 warmed trace replays at position 64. It is separate from clean warmed S256 latency; profiler/device-time sums must not be substituted for that benchmark. No stage-wide completion or full-attention qualification is claimed here.

Measured decoder SHA256: `c71ea01a4b38458f0f53808d288d9a39ca5563a13cb5e97f71e701391670c3f2` (`../../selected_fused_decoder_v6.py.snapshot`, resolved exact path in correctness/provenance). Helper SHA256: `c16d7b30bf83c4f2c2a4e8837562c15eb41ac96623255198dba034201e7d2617`. Native device close completed at 10:45:39 UTC; final CSV export completed at 10:46:51 UTC on 2026-11-13.

| Phase | Passes | Ops/pass | Device kernel sum/pass (μs) | Report gap/pass (μs) | Table | CSV |
|---|---:|---:|---:|---:|---|---|
| first-call/cold full S64 | 1 | 19137 | 188000.827000 | 4194659.701 | [text](prefill_perf_report.txt) | [CSV](prefill_perf_report.csv) |
| 16 warmed trace replays at position64 | 16 | 207 | 1411.229812 | 305372.908 | [text](decode_perf_report.txt) | [CSV](decode_perf_report.csv) |

Device time is the sum of measured kernel durations; it excludes gaps. Gaps include first-call compilation, instrumentation, serial drains, and host reporting overhead. The large gap values are not clean runtime performance. CSV `Device Time [μs]` is already microseconds; decode totals are divided by 16, and no terminal TOTAL row is added to the CSV sums. Raw sums and report sums agree.

## Completeness and correctness

All four exact signposts occur once in order. All 20 decode sessions contain 207 identical ordered operations, global call IDs, program hashes, attributes, and operand metadata; measured sessions are 5–20. Every executed operation has positive finite kernel/FW durations and ordered FW endpoints. Every measured phase row has usable overlap data under the report library’s cycle-to-time method, and overlap is zero. There are no malformed/truncated rows, dropped-marker warnings, instrumentation failures, or missing measured timings. The finalized raw CSV contains 23,709 rows: 4 signposts, 23,498 executed rows, and 207 untimed trace-capture metadata rows.

Prefill PCC `0.9999928705454029`, prefill-state PCC `0.9999986529417728`, eager/traced decode PCC `0.9999981480232413`; determinism and device-only runtime audit passed. The original ≥0.995 gates remain unchanged. Metadata caching is disabled by the actual capture provenance; watcher and allocation tracking are absent.

Untimed rows 19352–19558 are **outside both measured windows** and exactly match the captured trace’s later replay metadata. Native `tools/tracy/process_ops_logs.py:1433–1464` emits host capture metadata and replay `traceOps` separately, clearing trace fields on the host row. These rows are preserved, identified explicitly, and never assigned invented durations. Every executed replay is timed. Their presence is not silently ignored as generic missing data.

Exporter warnings are recorded in [profiler_completeness.json](profiler_completeness.json): mixed-type pandas warnings were checked with a strict string CSV reader and explicit numeric validation; the optional WASM convenience copy referenced a missing generated path, while the real 40.7 MB `.tracy` artifact exists in `raw/reports/2026_11_13_10_46_38`. The motherboard-name/subset-device warnings concern discovery and are retained. Report “unclassified operation” warnings concern advice/model categories, not missing measured time.

The initial host validator expected a `No overlap detected` console sentence. tt-perf-report suppresses that sentence on this single-grid branch, leaving zero overlap cells blank. The initial script and failure explanation are preserved in `validate_capture_host.initial.py.snapshot` and `validator_initial_report_text_assumption.json`. The final validator requires full finite raw overlap coverage through `tt_perf_report.overlap.measure_overlap` and zero report overlap; the numerical gate was not relaxed.

## Measured bottlenecks

### prefill

| Operation | Count/pass | Device μs/pass | Share of kernel sum |
|---|---:|---:|---:|
| GenericOpDeviceOperation | 8 | 71245.974 | 37.90% |
| UntilizeWithUnpaddingDeviceOperation | 3787 | 38991.276 | 20.74% |
| BinaryNgDeviceOperation | 4933 | 24569.538 | 13.07% |
| TilizeWithValPaddingDeviceOperation | 3972 | 20273.646 | 10.78% |
| AccumulationDeviceOperation | 132 | 11093.081 | 5.90% |
| SliceDeviceOperation | 4386 | 5680.907 | 3.02% |
| GroupedMatmulDeviceOperation | 8 | 4892.501 | 2.60% |
| UntilizeDeviceOperation | 378 | 3483.232 | 1.85% |

### decode

| Operation | Count/pass | Device μs/pass | Share of kernel sum |
|---|---:|---:|---:|
| GenericOpDeviceOperation | 7 | 511.343 | 36.23% |
| BinaryNgDeviceOperation | 65 | 249.975 | 17.71% |
| SparseMatmulDeviceOperation | 2 | 172.742 | 12.24% |
| FastReduceNCDeviceOperation | 13 | 64.052 | 4.54% |
| UnaryDeviceOperation | 17 | 49.623 | 3.52% |
| TilizeWithValPaddingDeviceOperation | 10 | 47.771 | 3.39% |
| MatmulDeviceOperation | 2 | 41.154 | 2.92% |
| TernaryDeviceOperation | 3 | 39.781 | 2.82% |

Prefill has eight fused corrected-matmul GenericOps, replacing the earlier 768 per-K generic calls. Their combined kernel time is 71,245.974 μs (37.90% of the cold S64 kernel sum). Thousands of remaining untilize/tilize/slice and binary operations remain visible; source attribution points to explicit ordered triangular-state and prefix arithmetic. This capture does not prove those operations removable under the unchanged arithmetic gates.

Decode has seven GenericOps. The first measured replay’s actual operand metadata confirms packed attention/QZ and router read interleaved DRAM activations; the output projection reads WIDTH_SHARDED L1. Thus the source-derived placement assessment is now corroborated by the measured path. All four ordered projections consume original BF16 TILE activations directly, with no coefficient-layout conversion sequence between their producer and the GenericOp. Other decode movement remains (10 tilize-with-padding, 7 untilize-with-unpadding, 5 interleaved-to-sharded, 4 sharded-to-interleaved, and 1 gather per replay); necessity must be assessed at each remaining source boundary rather than claiming the entire decoder is movement-free.

The tt-perf-report modeled-op DRAM roofline lines are 1.7% prefill and 8.3% decode. These are incomplete models: GenericOp corrected arithmetic/streaming kernels and several other operations lack standard advice/model classification. They cannot establish ≥70% roofline or the optimization Stop Rule. The trustworthy conclusions here are measured operation durations, order, shapes, placement, completeness, and zero overlap.

## Reproduction and artifacts

Parent capture command: `bash ../v6_profile_c71_single_prefill_s64.sh` from the repository root using its full repo-relative path. It uses the isolated single-prefill helper, full logical S64/context512, `--no-op-info-cache`, native device CSV dumping, and disabled legacy device GUI push. Realtime profiling stays active. The command refuses an existing artifact directory.

Host validation/report command:

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_profile_c71_single_prefill_s64/validate_capture_host.py
```

The validator refuses an existing final completeness record. Exact tt-perf-report commands are in [report_commands.json](report_commands.json). This used tt-perf-report 1.5.0. All work by this agent was host-only; the parent ran the device capture.

- [Completeness and hashes](profiler_completeness.json), [raw validation](raw_capture_validation.json), [numeric summary](perf_summary.json).
- [Raw preserved CSV](ops.csv), [original source path](ops_source.txt), original `.tracy` and device CSV under `raw/reports/2026_11_13_10_46_38`.
- [Compact topology](op_topology.md), [full prefill sequence](prefill_op_sequence.csv), [first measured decode sequence](decode_op_sequence.csv).
- `*_perf_report.txt`, `*_perf_report.csv`, console logs, stacked CSV and PNG generated by the report tool; [artifact hashes](artifact_manifest.json).
- `correctness.json` and `correctness.json.provenance.json`, parent log `../v6_profile_c71_single_prefill_s64.log`.

Older overflowing captures remain rejected. Their truncated rows are not combined with this capture.
