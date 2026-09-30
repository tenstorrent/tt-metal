exec 9>/data/kmabee/runs_sasha/queue.lock; flock 9
source /data/kmabee/runs_sasha/env.sh; waitchips
cd $W && echo "=== mm knobs $(date +%T) sha=$(git rev-parse --short HEAD) $(tdp)"
run () { local out=$1; shift; env TT_METAL_HOME=$W PYTHONPATH=$W/ttnn:$W timeout 7200 $PY $O/mm_sweep/mm_sweep.py "$@" --tdp "$(tdp)" --out $O/mm_sweep/$out < /dev/null >> $O/mm_sweep/${out%.jsonl}.log 2>&1; echo "rc=$? $out $* $(date +%T)"; }
# 1) L1-sharded 1D at M=256 (2k) and 416 with the output unshard cost timed
run ws.jsonl --m 256 416 512 --proj qkv_s qkv_g o_s o_g gate up down --families model 1d_ws --filter _ks4 _ks7 _ks8 _ks12 _ks14 _ks21
# 2) compute knobs
for V in "" "--no-l1acc" "--no-fp32" "--no-fp32 --no-l1acc"; do
  run knobs.jsonl --m 256 1024 --proj gate down qkv_g --families model 2d --filter _k7 _k12 _k14 _k24 $V
done
echo "=== mm knobs done $(date +%T)"
