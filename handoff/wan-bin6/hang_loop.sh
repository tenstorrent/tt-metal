#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Repeatedly reset the board and run the Wan test, capturing triage for every hang.
#
#   handoff/wan-bin6/hang_loop.sh [num_runs] [res] [chunk|nochunk] [out_dir]
#   defaults:                      5          720p  chunk           ~/wan_hang_reports
#
# Per run, <out_dir>/run_NN/ gets: links.txt (after reset), test.log, and on a hang triage.txt,
# triage_callstacks_2.txt (second snapshot, for frozen-vs-polling), host_stacks.txt (gdb) and
# summary.txt (analyze_triage.py). One line per run is appended to <out_dir>/index.txt and printed.
set -uo pipefail

runs="${1:-5}"
res="${2:-720p}"
mode="${3:-chunk}"
out="${4:-$HOME/wan_hang_reports}"
stall_s=180        # max silent seconds once the pipeline is running on device
setup_stall_s=900  # before "Pipeline allocation" (mesh open, weight load)

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
cd "$repo"
source python_env/bin/activate
mkdir -p "$out"

export TT_METAL_HOME="$repo"
export PYTHONPATH="$repo:$here"
export NO_PROMPT=1
export TT_DIT_CACHE_DIR="${TT_DIT_CACHE_DIR:-/tmp/TT_DIT_CACHE_wan_bin6}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}"  # /mnt/models/vbench here is empty and read-only
export TORCH_HOME="${TORCH_HOME:-$VBENCH_CACHE_DIR/torch_hub}"
export TT_MESH_GRAPH_DESC_PATH="$repo/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto"

plugin=()
[[ "$mode" == "chunk" ]] && plugin=(-p vae_chunk_patch)
test_id="models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py::test_pipeline_inference[blackhole-resolution_${res}-4x8sp1tp0nl2_linear_is_fsdp0-True]"
build="$(git rev-parse --short HEAD) $(git branch --show-current) pr60190_build=$(strings build/lib/libtt_metal.so | grep -c harvested_dram_views)"

note() { echo "$*" | tee -a "$out/index.txt"; }
note "=== $(date '+%F %T') start: $runs x $res $mode, build: $build, TT_MM_THROTTLE_PERF=${TT_MM_THROTTLE_PERF:-unset}"

first=$(( $(ls -d "$out"/run_* 2>/dev/null | wc -l) + 1 ))
for (( i = first; i < first + runs; i++ )); do
    d="$out/run_$(printf %02d "$i")"
    mkdir -p "$d"

    # Never reset or run while someone else is using the chips.
    others="$(ps -eo user=,cmd= | grep -E '[p]ytest|[t]trun|[t]t-smi' | awk -v me="$USER" '$1 != me' || true)"
    if [[ -n "$others" ]]; then
        note "run $i: ABORT, other users on the chips: $(echo "$others" | head -2 | cut -c1-120)"
        exit 1
    fi

    timeout 900 tt-smi -glx_reset > "$d/reset.log" 2>&1
    python -I "$here/check_links.py" 2>/dev/null | grep '^T' > "$d/links.txt"
    links="$(wc -l < "$d/links.txt") chips with down ports"

    start=$(date +%s)
    timeout 2400 python -m pytest "${plugin[@]}" --timeout 2100 "$test_id" -x > "$d/test.log" 2>&1 &
    tpid=$!
    sleep 5
    pid="$(pgrep -P "$tpid" -f pytest | head -1)"
    pid="${pid:-$tpid}"

    result=""
    while kill -0 "$tpid" 2>/dev/null; do
        sleep 15
        # After CLIP the device work is done; VBench scores on CPU and can be silent for minutes.
        grep -q "CLIP scores" "$d/test.log" && continue
        limit=$setup_stall_s
        grep -q "Pipeline allocation" "$d/test.log" && limit=$stall_s
        age=$(( $(date +%s) - $(stat -c %Y "$d/test.log") ))
        (( age > limit )) || continue

        step="$(tail -c 2000 "$d/test.log" | tr '\r' '\n' | grep -oE 'Denoise step [0-9]+/[0-9]+|Pipeline allocation run' | tail -1)"
        timeout 900 python tools/tt-triage.py --llm-output --disable-progress --skip-version-check > "$d/triage.txt" 2>&1
        timeout 600 python tools/tt-triage.py --llm-output --disable-progress --skip-version-check --run=dump_callstacks > "$d/triage_callstacks_2.txt" 2>&1
        timeout 180 gdb -p "$pid" -batch -ex "thread apply all bt 15" > "$d/host_stacks.txt" 2>&1
        python -I "$here/analyze_triage.py" "$d/triage.txt" "$d/triage_callstacks_2.txt" > "$d/summary.txt" 2>&1
        strag="$(grep -m1 '^Straggler' "$d/summary.txt" || grep -m1 'host-side' "$d/summary.txt")"
        frozen="$(grep -m1 'frozen compute' "$d/summary.txt" | sed 's/.*: //')"
        result="HANG after '${step}' | ${strag} | frozen: ${frozen:-n/a}"
        kill "$pid" "$tpid" 2>/dev/null; sleep 10; kill -9 "$pid" "$tpid" 2>/dev/null
        break
    done
    wait "$tpid" 2>/dev/null

    if [[ -z "$result" ]]; then
        summary="$(grep -E '[0-9]+ (passed|failed|error)' "$d/test.log" | tail -1 | tr -d '=')"
        clip="$(grep -oE 'CLIP scores: .*' "$d/test.log" | tail -1)"
        err="$(grep -m1 -oE '(Out of Memory|TT_FATAL|TT_THROW|AssertionError|RuntimeError)[^|]{0,120}' "$d/test.log")"
        result="FINISHED: ${summary:-no pytest summary} | ${clip:-no CLIP} | ${err:-}"
    fi
    note "run $i [$(date '+%T'), $(( ($(date +%s) - start) / 60 )) min, links: $links]: $result"
done
note "=== $(date '+%F %T') done"
