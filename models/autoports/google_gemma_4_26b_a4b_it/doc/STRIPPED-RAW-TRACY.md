# Stripped raw Tracy captures — google_gemma_4_26b_a4b_it (fuse-advise)

Only `tt-perf-report` outputs are published for this cell. GitHub rejects blobs over 100 MB and
this cell's raw captures reach **280 MB** (~2.1 GB total), so they are not in this branch's history
at all — the files were brought across without the cell branch's history, not deleted in a later
commit, because a push validates every reachable blob.

**Nothing was lost.** The complete tree is on machine A at branch
`skillexp-cell/fuse-advise/gemma4`.

Kept: `*perf_report*`, `*perf_summary*`. Stripped: `tracy_ops_times.csv`,
`profile_log_device.csv`, `tracy_profile_log_host.tracy`, `ops.csv`, `ops_perf_results.csv`
— profiler inputs, regenerable by re-running the capture.

Stripped: 95 files. Largest omitted:

```
functional_decoder/tracy/decode_trace_device_current.csv
functional_decoder/tracy/decode_trace_latency.csv
functional_decoder/tracy/decode_trace_latency.txt
functional_decoder/tracy/prefill_full_attention_batch1.csv
functional_decoder/tracy/prefill_full_attention_batch1.txt
functional_decoder/tracy/prefill_sliding_attention_batch1.csv
functional_decoder/tracy/prefill_sliding_attention_batch1.txt
functional_decoder/tracy/provenance.json
fused_decoder/tracy/decode_capture_raw_sliding_batch1.csv
```
