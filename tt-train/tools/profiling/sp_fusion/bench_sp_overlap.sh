#!/usr/bin/env bash
# Usage: [IMPL=composed] [BWD_IMPL=same] [OVERLAP=backward|split|off] [CCL=rows=1|columns=1|...] [BATCH=1] [MEMEFF=0] bench_sp_overlap.sh <ring|line> [max_steps=6] [extra train.py args...]
# bench_sp_train.sh (same run, same RESULT/PHASES lines) plus the two-stream knobs of agent S: sp_linear_backward_impl,
# sp_overlap (backward = two queues + CCL sub-device; split = the CCL region reserved but one queue, the reference for
# backward; off = whole grid, one queue) and the CCL sub-device shape (ignored for off).
# Log: $SPFUSE/logs/bench_sp_ovl_<impl>[-bwd<impl>]_<overlap>_<ccl>_<topo>_b<B>[_memeff].log
set -uo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/env.sh"
TOPO="${1:?ring|line}"; STEPS="${2:-6}"; shift 2 2>/dev/null || shift $#
IMPL="${IMPL:-composed}"; BWD_IMPL="${BWD_IMPL:-same}"; OVERLAP="${OVERLAP:-backward}"; CCL="${CCL:-rows=1}"
BATCH="${BATCH:-1}"; MEMEFF="${MEMEFF:-0}"
case "$TOPO" in
  ring) MGD="$TT_METAL_HOME/tt-train/configs/mgd/bh_galaxy_1_4_ring_ring.textproto" ;;
  line) MGD="$TT_METAL_HOME/tt-train/configs/mgd/bh_galaxy_1_4_line_line.textproto" ;;
  *) echo "topo must be ring|line"; exit 2 ;;
esac
case "$OVERLAP" in backward|split|off) ;; *) echo "OVERLAP must be backward|split|off"; exit 2;; esac
ROWS=0; COLS=0; [ "$OVERLAP" = off ] && CCL=none
case "$CCL" in rows=*) ROWS="${CCL#rows=}" ;; columns=*) COLS="${CCL#columns=}" ;; none) ;; *) echo "CCL must be rows=<n> or columns=<n>"; exit 2 ;; esac
TAG="$IMPL"; [ "$BWD_IMPL" != same ] && TAG="${TAG}-bwd${BWD_IMPL}"; TAG="${TAG}_${OVERLAP}_${CCL//=/}_${TOPO}_b${BATCH}"; [ "$MEMEFF" = 1 ] && TAG="${TAG}_memeff"
SRC="$TT_METAL_HOME/tt-train/configs/training_configs/training_llama8b_tp4_sp.yaml"
CFG="$SPFUSE/bench_llama8b_tp4_sp_ovl_${TAG}.yaml"
KNOBS="  sp_linear_impl: $IMPL\n  sp_linear_backward_impl: $BWD_IMPL\n  sp_overlap: \"$OVERLAP\"\n  sp_ccl_rows: $ROWS\n  sp_ccl_columns: $COLS"
sed -e "s/^  enable_sp: true/  enable_sp: true\n$KNOBS/" \
    -e "s/^  gradient_accumulation_steps: .*/  gradient_accumulation_steps: 1/" \
    -e "s/^  batch_size: .*/  batch_size: $BATCH/" "$SRC" > "$CFG"
grep -q "sp_linear_impl: $IMPL" "$CFG" || { echo "failed to derive $CFG"; exit 2; }
if [ "$MEMEFF" = 1 ]; then
  MODEL_SRC="$TT_METAL_HOME/tt-train/configs/model_configs/llama8b.yaml"; MODEL_CFG="$SPFUSE/bench_llama8b_memeff.yaml"
  sed -e "s/^  runner_type: .*/  runner_type: memory_efficient/" "$MODEL_SRC" > "$MODEL_CFG"
  grep -q "runner_type: memory_efficient" "$MODEL_CFG" || { echo "failed to derive $MODEL_CFG"; exit 2; }
  sed -i -e "s#^  model_config: .*#  model_config: \"$MODEL_CFG\"#" "$CFG"
fi
NAME="bench_sp_ovl_${TAG}"
"$SPFUSE/devrun.sh" "$NAME" 1200 5400 -- \
  "cd tt-train && TT_MESH_GRAPH_DESC_PATH=$MGD TTML_NAIVE_PROFILER=1 $PY sources/examples/train/train.py -c $CFG --max-steps $STEPS --fresh $*"
RC=$?
LOG="$SPFUSE/logs/$NAME.log"
grep -o "optimizer_step_done timestamp_us=[0-9]*" "$LOG" | awk -F= -v tag="$TAG" '
  NR > 1 { d = ($2 - prev) / 1e6; printf "step %d: %.3f s\n", NR - 1, d; if (NR > 3) { sum += d; n++ } }
  { prev = $2 }
  END { if (n) printf "RESULT %s: mean step %.3f s over %d steps (after 2 warm-up steps)\n", tag, sum / n, n;
        else print "RESULT " tag ": not enough steps completed, see " ENVIRON["LOG"] }' LOG="$LOG"
python3 "$SPFUSE/S_summarize.py" "$LOG" | tail -1 | sed 's/^/PHASES /'
exit $RC
