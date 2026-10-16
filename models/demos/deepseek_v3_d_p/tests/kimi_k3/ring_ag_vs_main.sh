#!/bin/bash
# Ring all-gather / ring joint SDPA on this branch vs another tree (e.g. an origin/main worktree), with the same probe
# (test_zz_ring_probe.py) on the same machine. One process per configuration: the experiment switches are read once
# per process (static locals in the C++ factories).
#
# Configurations:
#   lat10  K3 warm latent ring_mla (24 heads x 640 q rows per chip), 10x10 attention + CCL column x=10
#   latC   same, layout C: 11x9 attention on rows 1-9, CCL on the eth-nearest top-row cores (branch only)
#   exp10  expanded K/V ring_joint_sdpa (24 heads, q32/k640, ring-bound), 10x10 + CCL column, and its gather alone
#   expC   same, layout C (branch only; stock code rejects 11x9 + a CCL row: "SDPA coregrid overlaps with AllGather")
#
# usage: ring_ag_vs_main.sh [-m MAIN_WORKTREE] [-p PYTHON_ENV] [-o OUT_DIR] [-r REPEATS]
#   MAIN_WORKTREE: a built tree to compare against; the probe is copied into it (untracked) if missing.
#   PYTHON_ENV: venv to activate (default: <tree>/python_env).
# Both trees must be built (build_Release, installed into build/lib). Device runs go through scripts/run_safe_pytest.sh.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BRANCH=$(git -C "$HERE" rev-parse --show-toplevel)
PROBE=models/demos/deepseek_v3_d_p/tests/kimi_k3/test_zz_ring_probe.py
MAIN=""
PYENV=""
OUT=${TMPDIR:-/tmp}/ring_ag_vs_main
REPEATS=3
while getopts m:p:o:r: opt; do
    case $opt in
        m) MAIN=$(cd "$OPTARG" && pwd) ;;
        p) PYENV=$OPTARG ;;
        o) OUT=$OPTARG ;;
        r) REPEATS=$OPTARG ;;
        *) exit 2 ;;
    esac
done
mkdir -p "$OUT"

LAT="RING_PROBE_SKIP_GATHER=1 RING_PROBE_VARIANTS=24x640 RING_PROBE_REPEATS=$REPEATS RING_PROBE_GATHERS=ring"
EXP="RING_PROBE_SKIP_GATHER=1 RING_PROBE_VARIANTS=1x32 RING_PROBE_REPEATS=$REPEATS RING_PROBE_GATHERS= RING_PROBE_EXPANDED=24 RING_PROBE_EXP_GATHERS=ring"
BRANCH_ENV="RING_AG_PREFETCH_PACKETS=8"
LAYOUT_C="RING_PROBE_PLACEMENTS=11x9@0,0r RING_SDPA_GRID_ORIGIN_Y=1 RING_AG_PLACE_ETH_FUSED=1"

run() {  # tree name env...
    local tree=$1 name=$2
    shift 2
    (
        cd "$tree" || exit 1
        # shellcheck disable=SC1091
        source "${PYENV:-$tree/python_env}/bin/activate"
        export TT_METAL_HOME=$tree TT_METAL_RUNTIME_ROOT=$tree PYTHONPATH=$tree/ttnn:$tree TT_METAL_CACHE=$OUT/cache_$name
        env "$@" scripts/run_safe_pytest.sh "$PROBE" -s > "$OUT/$name.log" 2>&1
    )
    echo "$name exit=$?" >&2
}

# us of the fused op, us of the gather alone, output checksum
fused_us() { grep -oE "(ring_mla heads=24 q_rows=640|expanded ring_joint_sdpa heads=24): +[0-9.]+" "$1" | head -1 | grep -oE "[0-9.]+$"; }
gather_us() { grep -oE "(RESULT gather +ring|RESULT expanded gather ring heads=24): best +[0-9.]+" "$1" | head -1 | grep -oE "[0-9.]+$"; }
checksum() { grep -oE "CHECKSUM (expanded 24: out|[0-9x@,rc]+ 24x640:) sum [-0-9.]+" "$1" | head -1 | grep -oE "[-0-9.]+$"; }

ROWS=()
measure() {  # label tree name env...
    local label=$1 tree=$2 name=$3
    shift 3
    run "$tree" "$name" "$@"
    local log=$OUT/$name.log status
    status=$(grep -oE "SAFE_PYTEST_RESULT: [A-Z]+" "$log" | tail -1 | cut -d' ' -f2)
    ROWS+=("$(printf '%-6s %-8s %10s %10s %14s %s' "$name" "$label" "$(fused_us "$log")" "$(gather_us "$log")" "$(checksum "$log")" "${status:-?}")")
}

for c in lat exp; do
    base=$([ $c = lat ] && echo "$LAT" || echo "$EXP")
    if [ -n "$MAIN" ]; then
        [ -f "$MAIN/$PROBE" ] || cp "$BRANCH/$PROBE" "$MAIN/$PROBE"
        # shellcheck disable=SC2086
        measure main "$MAIN" main_${c}10 $base
    fi
    # shellcheck disable=SC2086
    measure branch "$BRANCH" ${c}10 $base $BRANCH_ENV
    # shellcheck disable=SC2086
    measure branch "$BRANCH" ${c}C $base $BRANCH_ENV $LAYOUT_C
done

printf '%-6s %-8s %10s %10s %14s %s\n' config tree fused_us gather_us checksum status
printf '%s\n' "${ROWS[@]}"
echo "logs: $OUT"
