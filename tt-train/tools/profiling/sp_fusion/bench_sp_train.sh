#!/usr/bin/env bash
# Usage: [BATCH=1] [PROFILE=0] [MEMEFF=0] bench_sp_train.sh <composed|fused|nocomm> <ring|line> [max_steps=6] [extra train.py args...]
#   BATCH   per-device batch size written into the derived config (default 1)
#   PROFILE 1 = run under the tt-metal device profiler (python -m tracy; captures the child's output, so the
#               watchdog gets idle == hard budget) and copy the ops CSV to logs/<name>_ops.csv
#   MEMEFF  1 = activation recompute (model config runner_type: memory_efficient); needed for batch >= 4 on this box
# A short llama8b tp4 SP training run (1x4 mesh, tp on cluster_axis 1, seq 2048, no grad accumulation) through
# devrun.sh, then per-step wall time from the naive profiler's optimizer_step_done markers; step 1 is the JIT compile
# and step 2 warms caches, so RESULT averages steps 3..N. Log: $SPFUSE/logs/bench_sp_train_<impl>_<topo>_b<B>[_memeff][_prof].log
set -uo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/env.sh"
IMPL="${1:?composed|fused|nocomm}"; TOPO="${2:?ring|line}"; STEPS="${3:-6}"; shift 3 2>/dev/null || shift $#
case "$IMPL" in composed|fused|nocomm) ;; *) echo "impl must be composed|fused|nocomm"; exit 2;; esac
BATCH="${BATCH:-1}"; PROFILE="${PROFILE:-0}"; MEMEFF="${MEMEFF:-0}"
case "$TOPO" in
  ring) MGD="$TT_METAL_HOME/tt-train/configs/mgd/bh_galaxy_1_4_ring_ring.textproto" ;;
  line) MGD="$TT_METAL_HOME/tt-train/configs/mgd/bh_galaxy_1_4_line_line.textproto" ;;
  *) echo "topo must be ring|line"; exit 2 ;;
esac
SRC="$TT_METAL_HOME/tt-train/configs/training_configs/training_llama8b_tp4_sp.yaml"
SUFFIX="b${BATCH}"; [ "$MEMEFF" = 1 ] && SUFFIX="${SUFFIX}_memeff"
CFG="$SPFUSE/bench_llama8b_tp4_sp_${IMPL}_${SUFFIX}.yaml"
sed -e "s/^  enable_sp: true/  enable_sp: true\n  sp_linear_impl: $IMPL/" \
    -e "s/^  gradient_accumulation_steps: .*/  gradient_accumulation_steps: 1/" \
    -e "s/^  batch_size: .*/  batch_size: $BATCH/" "$SRC" > "$CFG"
grep -q "sp_linear_impl: $IMPL" "$CFG" || { echo "failed to derive $CFG"; exit 2; }
if [ "$MEMEFF" = 1 ]; then
  MODEL_SRC="$TT_METAL_HOME/tt-train/configs/model_configs/llama8b.yaml"; MODEL_CFG="$SPFUSE/bench_llama8b_memeff.yaml"
  sed -e "s/^  runner_type: .*/  runner_type: memory_efficient/" "$MODEL_SRC" > "$MODEL_CFG"
  grep -q "runner_type: memory_efficient" "$MODEL_CFG" || { echo "failed to derive $MODEL_CFG"; exit 2; }
  sed -i -e "s#^  model_config: .*#  model_config: \"$MODEL_CFG\"#" "$CFG"
fi
NAME="bench_sp_train_${IMPL}_${TOPO}_${SUFFIX}"; [ "$PROFILE" = 1 ] && NAME="${NAME}_prof"
if [ "$PROFILE" = 1 ]; then
  "$SPFUSE/devrun.sh" "$NAME" 7200 7200 -- \
    "cd tt-train && TT_MESH_GRAPH_DESC_PATH=$MGD TTML_NAIVE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=200000 $PY -m tracy -r -v -p --no-op-info-cache -n $NAME sources/examples/train/train.py -c $CFG --max-steps $STEPS --fresh $*"
  RC=$?
  CSV=$(ls -t "$TT_METAL_HOME"/generated/profiler/reports/"$NAME"/ops_perf_results_*.csv "$TT_METAL_HOME"/generated/profiler/reports/"$NAME"/*/ops_perf_results_*.csv 2>/dev/null | head -1)
  [ -n "$CSV" ] && cp "$CSV" "$SPFUSE/logs/${NAME}_ops.csv" && echo "ops csv -> $SPFUSE/logs/${NAME}_ops.csv ($(wc -l < "$CSV") rows)"
else
  "$SPFUSE/devrun.sh" "$NAME" 1200 5400 -- \
    "cd tt-train && TT_MESH_GRAPH_DESC_PATH=$MGD TTML_NAIVE_PROFILER=1 $PY sources/examples/train/train.py -c $CFG --max-steps $STEPS --fresh $*"
  RC=$?
fi
LOG="$SPFUSE/logs/$NAME.log"
grep -o "optimizer_step_done timestamp_us=[0-9]*" "$LOG" | awk -F= -v impl="$IMPL" -v topo="$TOPO" -v batch="$BATCH" -v memeff="$([ "$MEMEFF" = 1 ] && echo ' memeff' || echo '')" '
  NR > 1 { d = ($2 - prev) / 1e6; printf "step %d: %.3f s\n", NR - 1, d; if (NR > 3) { sum += d; n++ } }
  { prev = $2 }
  END { if (n) printf "RESULT %s %s batch %s%s: mean step %.3f s over %d steps (after 2 warm-up steps)\n", impl, topo, batch, memeff, sum / n, n;
        else print "RESULT " impl " " topo " batch " batch memeff ": not enough steps completed, see " ENVIRON["LOG"] }' LOG="$LOG"
# Per-phase split from the naive profiler markers (ms, mean of steps 3..N)
python3 - "$LOG" <<'PY'
import re, sys
ev = [(m.group(1), int(m.group(2))) for m in re.finditer(r"\[NAIVE_PROFILER\] ([a-z_]+_done) timestamp_us=(\d+)", open(sys.argv[1]).read())]
names = []
for n, _ in ev:
    if n in names: break
    names.append(n)
if names and len(ev) >= 3 * len(names):
    steps = [ev[i:i + len(names)] for i in range(0, len(ev) - len(names) + 1, len(names))]
    acc = {}
    for si in range(2, len(steps)):
        for k, (n, t) in enumerate(steps[si]):
            prev = steps[si][k - 1][1] if k else steps[si - 1][-1][1]
            acc.setdefault(n, []).append((t - prev) / 1e3)
    print("PHASES " + "  ".join(f"{n.replace('_done','')}={sum(v)/len(v):.1f}ms" for n, v in acc.items()))
PY
exit $RC
