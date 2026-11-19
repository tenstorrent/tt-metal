# Functional decoder measurements

Status: in progress. These measurements do not establish overall stage acceptance.
The indexed-attention measurement below predates the shared residual-write
rounding correction; a final measurement of the corrected path is required.
The target is the pinned Qwen3.8-Flash-Next checkpoint and unchanged HF context
262144 on one Blackhole device, 1x1 mesh, 110 worker cores. Hardware jobs run
serially on visible chip 0; CPU reference and checkpoint preparation may run
concurrently. Host timings are diagnostic only.

| Layer kind | Prompt | Cache capacity | Warmed prefill kernel sum | Warmed traced decode kernel sum | Kernel + inter-op gaps, prefill / decode |
|---|---:|---:|---:|---:|---:|
| Indexed attention, real layer 3 | 32 | 262144 | 257.783 ms | 148.134 ms | 273.498 / 148.402 ms |
| GDN, layer 0 | Pending final numerical regression | — | Pending | Pending | — |
| GDN + PLE, layer 1 | Pending resident precision validation | — | Pending | Pending | — |

The indexed-attention row uses `tracy/indexed_attention/ops.csv` and the
signpost-filtered `prefill_perf_report.csv` / `decode_perf_report.csv`.
The measured column is **Device Time [μs]**, divided by 1000 for milliseconds;
the separate gap column is **Op-to-Op Gap [μs]**. There are 1155 prefill and
335 decode operations. Human-readable tables are
`tracy/indexed_attention/prefill_perf_report.txt` and
`tracy/indexed_attention/decode_perf_report.txt`. Source path and SHA are in
`provenance.json`; exact totals are in `latency_summary.json` beside them.

The runner performs warmup, synchronization, prefill start signpost, the
device-only public prefill pass, one synchronization, and end signpost. Decode
is captured after warmup, replayed once for warming, restored to its prefix
state, then measured from a subsequent replay. Watcher is disabled for this
run. Correctness and allocation-tracker evidence are separate artifacts.

From the checkout, using the pinned `$SNAPSHOT`:

```bash
env -u TT_METAL_WATCHER -u TT_METAL_LOGS_PATH TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto TT_METAL_TRACE_ALLOC_TRACKING=0 PYTHONPATH=. timeout 1800 python_env/bin/python -m tracy -r -p -v --no-web-server --op-support-count 10000 --tracy-tools-folder build/tools/profiler/bin -o models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/tracy/indexed_attention/final_raw models/autoports/qwen_qwen3_8_flash_next/tests/profile_decoder.py --snapshot "$SNAPSHOT" --layer 3 --seq-len 32 --output models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/tracy/indexed_attention/final_profile.json
```

`final_profile.log` records the complete command output. Each table is generated
with the following command, replacing `PREFILL`/`prefill` with `DECODE`/`decode`
for the second window:

```bash
python_env/bin/tt-perf-report models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/tracy/indexed_attention/ops.csv --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END --no-advice --no-summary --no-color --arch blackhole > models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/tracy/indexed_attention/prefill_perf_report.txt
python_env/bin/tt-perf-report models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/tracy/indexed_attention/ops.csv --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END --csv models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/tracy/indexed_attention/prefill_perf_report.csv --no-advice --no-summary --no-color --arch blackhole > models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/tracy/indexed_attention/prefill_perf_report.console.log
```

Two earlier profiler attempts are retained. `--device-trace-profiler` records
whole-trace timing rather than the per-operation data required by this report
path. Standard mode with the default 1000-program capacity then lost records;
`--op-support-count 10000` produced the complete report. These earlier failures
are not used for performance claims. The report tool's help lists
`blackhole/p150` as alternative aliases; the accepted argument is `blackhole`.
