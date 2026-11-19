# Final reduced terminal capture queue

Prepared commands only; no device/profiler work was launched during this audit.
Use the device owner's serialized lane after the selected production source has
passed its tracked correctness checks. This profiles real layers 0/1/3, B1,
capacity 262144, current LM head and row assembly, current token-feedback helper,
and both common samplers. It constructs the current single prefill workspace,
but the decode-only profile neither executes nor measures prefill. No all-layer
Tracy run is authorized by this recipe.

The helper warms only its measured eager B1 decode and exact sampler wrappers.
It never calls full serving prepare or captures traces in `--profile-ops-only`.
Warmup profiler drains precede measured signposts. The six markers remain
`PERF_MODEL`, `PERF_MODEL_END`, `PERF_SPLIT_GREEDY`, `PERF_SPLIT_GREEDY_END`,
`PERF_FORCE_ARGMAX`, `PERF_FORCE_ARGMAX_END`. Both samplers must produce the
same expected greedy token on frozen real logits. Host argmax is diagnostic
outside every measured region. This is operation attribution, not traced
throughput, final full-model performance, or an alternative sampling policy.

The installed `python -m tracy --help` and
`python -m tt_perf_report.perf_report --help` were checked without hardware.
Their options below match the successful `terminal_tracy_v3` recipe. Its
strict `validate_terminal_profile.py` accepts variable operation counts, so the
new head/row assembly/feedback need no threshold changes. It still requires
exact host/device/final-operation equality, positive preserved durations,
equal counts on all four ranks, and exactly the six matching signposts.

Run from the checkout; these paths must be new. If a capture fails, retain its
JSON, log, `.logs`, raw Tracy/device CSVs and any partial report. Select a new
versioned directory for a later attempt; do not delete or overwrite v1/v2/v3 or
other historical evidence. The helper itself refuses an existing output JSON.

```bash
set -e
source python_env/bin/activate
source /home/ttuser/qwen38-flash/lane-env.sh
export TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto
export TT_VISIBLE_DEVICES=0,1,2,3
export QWEN_CHECKPOINT=/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa
export QWEN_PLE_CACHE=/home/ttuser/qwen38-flash/packed_ple_lloyd_lsq_group10_scale8_wide
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset TT_METAL_WATCHER TT_METAL_TRACE_PROFILER TT_METAL_STREAMING_PROFILER
export TT_METAL_PROFILE_PERF_COUNTERS=0
export TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_TRACEBACKS=0
export TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
terminal_root=models/autoports/qwen_qwen3_8_flash_next/doc/full_model
terminal_raw="$terminal_root/terminal_final_tracy_v1"
terminal_report="$terminal_root/terminal_final_perf_v1"
test ! -e "$terminal_raw"
test ! -e "$terminal_report"
test ! -e "$terminal_root/terminal_final_ops_v1.json"
test ! -e "$terminal_root/terminal_final_ops_v1.log"
TT_METAL_DEVICE_PROFILER=1 timeout --signal=INT --kill-after=60 3600 \
  python -u -m tracy -r -p -v --no-web-server --op-support-count 8192 \
  -o "$terminal_raw" \
  -m models.autoports.qwen_qwen3_8_flash_next.tests.profile_full_model_terminal \
  --profile-ops-only --iterations 1 --warmup 2 --capacity 262144 \
  --output "$terminal_root/terminal_final_ops_v1.json" \
  > "$terminal_root/terminal_final_ops_v1.log" 2>&1
```

Only after the capture succeeds, validate its own JSON, select its single
generated CSV, and preserve a byte-identical report input. The shell variables
above remain in scope; no report-generation command opens a device.

```bash
python - "$terminal_root/terminal_final_ops_v1.json" "$terminal_raw" "$terminal_report" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import sys
result, raw, destination = map(Path, sys.argv[1:])
evidence = json.loads(result.read_text())
assert evidence['passed'] and evidence['layers'] == [0, 1, 3]
assert evidence['batch_size'] == 1 and evidence['capacity'] == 262144
assert evidence['profile_contract']['no_live_traces']
assert evidence['source_sha256']
sources = list(raw.glob('reports/**/ops_perf_results*.csv'))
assert len(sources) == 1, sources
destination.mkdir(exist_ok=False)
copied = destination / 'ops_perf_results.csv'
shutil.copyfile(sources[0], copied)
assert hashlib.file_digest(sources[0].open('rb'), 'sha256').digest() == hashlib.file_digest(copied.open('rb'), 'sha256').digest()
PY
python "$terminal_root/validate_terminal_profile.py" \
  "$terminal_raw" "$terminal_report/ops_perf_results.csv" \
  "$terminal_report/coverage.json" > "$terminal_report/coverage.log" 2>&1
for terminal_region in model:PERF_MODEL split_greedy:PERF_SPLIT_GREEDY force_argmax:PERF_FORCE_ARGMAX
do
  terminal_name=${terminal_region%%:*}
  terminal_marker=${terminal_region#*:}
  python -m tt_perf_report.perf_report "$terminal_report/ops_perf_results.csv" \
    --start-signpost "$terminal_marker" --end-signpost "${terminal_marker}_END" \
    --no-color --csv "$terminal_report/${terminal_name}_perf_report.csv" \
    > "$terminal_report/${terminal_name}_perf_report.console.log" 2>&1
  python -m tt_perf_report.perf_report "$terminal_report/ops_perf_results.csv" \
    --start-signpost "$terminal_marker" --end-signpost "${terminal_marker}_END" \
    --no-color --no-summary > "$terminal_report/${terminal_name}_perf_report.txt" 2>&1
done
```

Advice remains enabled. Keep CSV status chatter separate from the rendered
text tables. Compare the new largest LM-head, assembly and sampler operations
with the retained v3 report and clean traced timing, respecting units: raw
device durations are ns, filtered `Device Time` is microseconds. Merged-rank
kernel sums are not single-rank elapsed latency. Reconcile advice against
the preserved precision/rejection ledger; do not select a lower-fidelity
configuration from advice alone. A dominant sampler still requires terminal
work, not a documentation waiver.

Source audit after production factoring found no live `prefill_outputs`
references outside frozen diagnostic sources. Acceptance's direct `_capture`
and this profiler are B1-only. B>1 direct `_capture` in the trace-memory and
factored comparison probes is explicitly routed through the independently
owned `whole_decode_control` frozen loader; it is historical comparison, not
production serving. The serving owner maintains those controls. The terminal
scope CPU fake now warms `token_feedback`, matching the actual helper; five
scope/drain/capacity tests pass and Python compilation passes. The profiler report now
hashes all model tt sources, common sampler sources and its own helper so a
later source change cannot be confused with this capture.

The explicit C262144 now matches final acceptance allocation and QSA score width.
The CLI default128 remains for historical reproduction. The scorer still makes
one merged-head decode block at either capacity, so8192 op support is retained;
strict coverage remains mandatory. This short-history reduced capture does not
measure maximum-position selector cost. See [performance boundary audit](performance_boundary_audit.md).
