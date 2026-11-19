#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
artifact=models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder
waiter="$TT_MODEL_BRINGUP_ROOT/scripts/wait_for"
for layer in 0 3; do
    [[ ! -f "$artifact/STOP_QUEUE" ]] || exit 3
    destination="$artifact/tracy/layer_$layer"
    mkdir -p "$destination"
    "$waiter" --cmd "timeout 1800 env -u TT_METAL_WATCHER -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE python -m tracy -r -p -v -o $destination/raw -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_decoder --layer $layer --length 256 --context 512 --profile --output $destination/correctness.json" --file "$artifact/logs/profile_$layer.log" --timeout 1830 --tail 15
    python - "$destination" <<'PY'
import pathlib, shutil, sys
p=pathlib.Path(sys.argv[1])
files=list((p/'raw').rglob('ops_perf_results_*.csv'))
assert files, 'Tracy postprocessed ops CSV missing'
source=max(files,key=lambda x:x.stat().st_mtime)
shutil.copy2(source,p/'ops.csv')
(p/'ops_source.txt').write_text(str(source)+'\n')
print(source)
PY
    for phase in prefill decode; do
        signpost="PERF_${phase^^}"
        tt-perf-report "$destination/ops.csv" --start-signpost "$signpost" --end-signpost "${signpost}_END" --csv "$destination/${phase}_perf_report.csv" --no-advice > "$destination/${phase}_perf_report.console.log" 2>&1
        tt-perf-report "$destination/ops.csv" --start-signpost "$signpost" --end-signpost "${signpost}_END" --no-summary --no-advice --no-color > "$destination/${phase}_perf_report.txt" 2>&1
    done
done
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.summarize_perf
