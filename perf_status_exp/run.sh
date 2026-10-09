#!/bin/bash
# run.sh <tag> <mode> <pytest-file-relative> <-k expr> [SWEEP_ISL]
# mode base: main build host + untouched 768453d91be tree (.base_root).
# mode m7 / m8 / new: this worktree's build + kernels from .m7_root (merge, no step 5),
#   .m8_root (merge + step 5) or the worktree itself (current HEAD).
W=/localdev/mbezulj/tt-metal/.claude/worktrees/agent-abb2aba9472191522
TAG=$1; MODE=$2; TFILE=$3; KEXPR=$4
export SWEEP_ISL=${5:-256}
RUNS=/localdev/mbezulj/logs/merge8_runs.txt
if grep -q 'RC=124' $RUNS 2>/dev/null; then echo "SKIP after timeout $TAG $MODE" >> $RUNS; exit 1; fi
source /localdev/mbezulj/tt-metal/python_env/bin/activate
export TT_VISIBLE_DEVICES=0 CCACHE_DISABLE=1 LOGURU_LEVEL=DEBUG
unset TT_METAL_DEVICE_PROFILER
case $MODE in
  base) R=$W/.base_root; export PYTHONPATH=$R ;;
  m7) R=$W/.m7_root ;;
  m8) R=$W/.m8_root ;;
  new) R=$W ;;
  *) echo "bad mode"; exit 2 ;;
esac
export TT_METAL_CACHE=$W/.jitcache_$MODE
if [ "$MODE" != "base" ]; then
  export PYTHONPATH=$W/ttnn:$R
  export LD_LIBRARY_PATH=$W/build/lib:$LD_LIBRARY_PATH
fi
export TT_METAL_RUNTIME_ROOT=$R
cd $R
LOG=/localdev/mbezulj/logs/merge8_${TAG}_${MODE}_$(date +%Y%m%d_%H%M%S).log
echo "CWD $(pwd) ISL=$SWEEP_ISL K=$KEXPR" > $LOG
python -c "
import ttnn
print('TTNN_FILE', ttnn.__file__)
print('GRID', ttnn.UNIFIED_ROUTED_EXPERT_CORE_GRID)
for l in open('/proc/self/maps'):
    if '_ttnn' in l and ' r-xp ' in l: print('SO', l.split()[-1])
" >> $LOG 2>&1
if [ -n "$KEXPR" ]; then
  timeout ${TMO:-600} pytest $R/$TFILE -k "$KEXPR" -v -s >> $LOG 2>&1
else
  timeout ${TMO:-600} pytest $R/$TFILE -v -s >> $LOG 2>&1
fi
RC=$?
echo "RC=$RC" >> $LOG
echo "RC=$RC $TAG $MODE $LOG" >> $RUNS
