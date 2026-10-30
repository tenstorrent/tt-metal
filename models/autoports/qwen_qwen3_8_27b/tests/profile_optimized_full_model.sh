#!/usr/bin/env bash
set -euo pipefail
export USER=mvasiljev TT_METAL_HOME=/workspace TT_METAL_CACHE=/workspace/.cache/tt-metal-cache PYTHONPATH="/workspace:${PYTHONPATH:-}"
export TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000 TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1
unset TT_METAL_WATCHER TT_METAL_WATCHER_DEBUG_DELAY
batch=$1
base=models/autoports/qwen_qwen3_8_27b/doc/optimized_full_model/tracy/b${batch}
mkdir -p "$base"
python -m tracy -r -p -v --no-web-server -o "$base" -m models.autoports.qwen_qwen3_8_27b.tests.profile_full_model_reduced --batch "$batch" --buffered --output "$base/scope.json"
python -m models.autoports.qwen_qwen3_8_27b.tests.collect_multichip_perf "$base"
python - "$base" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'whole_layer_timing.json'
d=json.loads(p.read_text())
d['scope']='Reduced full-model variant: real layers0/3 plus embedding, terminal and canonical sampler. NOT all64layers or a single-layer measurement.'
d['time_basis']='Maximum complete per-device reduced-stack firmware envelope, all operations and gaps included; per-chip cycles / MHz. Never combine clocks across ASICs.'
p.with_name('reduced_stack_timing.json').write_text(json.dumps(d,indent=2)+'\n')
p.unlink()
PY
