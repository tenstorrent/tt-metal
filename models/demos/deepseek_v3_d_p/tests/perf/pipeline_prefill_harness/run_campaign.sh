#!/usr/bin/env bash
# The whole PP=4 vs single-rank campaign, end to end, on whatever galaxy you are sitting on.
#
#   ./run_campaign.sh              # everything: captures + 16 cells + warm latency + summary (~2 h)
#   ./run_campaign.sh --check      # preflight only, touches no chips
#   ./run_campaign.sh --summary    # re-print the tables from an existing run, no device needed
#   ./run_campaign.sh --no-captures    # skip the two Tracy captures (~10 min)
#
# This is what produced the tables in docs/MISTRAL4_PP4_BRINGUP_RERUN.md. Four phases:
#   1. pp4_deep + 1rank_deep single-layer Tracy captures  -> the per-layer budget (§2)
#   2. the 16-cell matrix, traced                         -> throughput (§1.1)
#   3. the ttft cells AGAIN with FORCE=1                  -> warm latency (§1.2)
#   4. summarize_prefill_campaign.py                              -> all three tables
#
# WHY PHASE 3 EXISTS. run_matrix.sh runs each ttft cell once, cold: first touch, JIT, allocator
# warm-up all land inside a single-request measurement. Cold pp4@25,600 reads 6.5 s against 1.17 s
# warm. Throughput cells do not care (they discard the first 8 intervals) but latency cells are one
# shot, so they have to be re-run on a warm machine. Skipping phase 3 does not give you a slow
# result, it gives you a WRONG one.
#
# EVERYTHING IS PER-HOST. Output goes to mistral4_perf_$(hostname) and
# mistral4_perf_profile_$(hostname), so two galaxies can be compared and neither overwrites the
# other. The staging dir the drivers write first (tests/perf/pipeline_prefill_harness/logs/) is NOT per-host and IS
# overwritten by the next campaign -- the per-host copies are the durable ones, and are what
# --summary reads.
set -u
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The analyzers and the binding generator are NOT in this directory: they are reusable and live
# one level up in tests/perf/, next to summarize_device_perf.py. Only the drivers live here.
TOOLS="$(cd "$S/.." && pwd)"
source "$S/env.sh"
cd "$TT_METAL_HOME" || exit 1
H="$(hostname)"
RES="${CAMPAIGN_OUT:-$TT_METAL_HOME/mistral4_perf_$H}"
PROF="${CAMPAIGN_PROFILE_OUT:-$TT_METAL_HOME/mistral4_perf_profile_$H}"
export M4_PROFILE_OUT="$PROF"
# OUT is deliberately NOT exported. run_single_layer_profile.sh resolves its own output as
# ${OUT:-$M4_PROFILE_OUT}, so exporting OUT here hijacks it and the Tracy captures land in the
# matrix's results directory instead of the profile one -- where summarize_prefill_campaign.py does not
# look, so the layer-budget table silently comes out empty. run_matrix.sh already defaults to the
# same per-host path, so OUT is passed per-invocation below purely to honour CAMPAIGN_OUT.
CHUNKS="${DEEP_CHUNKS:-8}"

summarize(){ "$PY" "$TOOLS/summarize_prefill_campaign.py" "$RES" "$PROF"; }

case "${1:-run}" in
  --check)   exec "$S/preflight.sh" ;;
  --summary) summarize; exit 0 ;;
  --no-captures) SKIP_CAP=1 ;;
  run) SKIP_CAP=0 ;;
  *) echo "usage: $0 [--check|--summary|--no-captures]"; exit 1 ;;
esac

echo "=== campaign on $H  ($(date -Is)) ==="
echo "    results : $RES"
echo "    profile : $PROF"
echo

# A wrong per-galaxy binding does NOT error -- it builds stages that are not columns and reports
# plausible, wrong numbers. Refuse rather than measure garbage.
TOPO=models/demos/common/prefill/runners/topology_configuration
BIND="$TOPO/pipeline_prefill_request_intragalaxy_4rank_8x1_torus_y.$H.yaml"
if [ ! -f "$BIND" ]; then
  echo "FAIL: no rank binding for $H. The [8,1] column->device map is per-galaxy and a wrong one is"
  echo "      silently wrong, not an error. Generate both, with the galaxy idle:"
  echo "        \$PY $TOOLS/gen_pipeline_binding.py"
  echo "        \$PY $TOOLS/gen_pipeline_binding.py --profile"
  exit 1
fi

if [ "${SKIP_CAP:-0}" = "0" ]; then
  for mode in pp4_deep 1rank_deep; do
    echo "=== [1/4] capture $mode ($CHUNKS chunks)  $(date -Is) ==="
    DEEP_CHUNKS="$CHUNKS" "$S/run_single_layer_profile.sh" "$mode"; echo "    $mode rc=$?"
  done
else
  echo "=== [1/4] captures SKIPPED ==="
fi

echo "=== [2/4] 16-cell matrix  $(date -Is) ==="
OUT="$RES" "$S/run_matrix.sh"; echo "    matrix rc=$?"

echo "=== [3/4] warm latency re-run  $(date -Is) ==="
OUT="$RES" MODES=ttft FORCE=1 "$S/run_matrix.sh"; echo "    warm ttft rc=$?"

echo "=== [4/4] summary  $(date -Is) ==="
summarize | tee "$RES/SUMMARY_$H.txt"
echo
echo "wrote $RES/SUMMARY_$H.txt"

BUSY=0; for d in /dev/tenstorrent/[0-9]*; do [ -n "$(fuser "$d" 2>/dev/null)" ] && BUSY=$((BUSY+1)); done
[ "$BUSY" -eq 0 ] && echo "chips: all released" \
  || echo "chips: WARNING $BUSY still held -- check for a stuck rank before handing the machine over"
