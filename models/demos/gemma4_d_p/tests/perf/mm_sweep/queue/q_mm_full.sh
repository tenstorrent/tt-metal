exec 9>/data/kmabee/runs_sasha/queue.lock; flock 9
source /data/kmabee/runs_sasha/env.sh; waitchips
cd $W && echo "=== mm full $(date +%T) sha=$(git rev-parse --short HEAD) $(tdp)"
KB="_k7 _k8 _k12 _k14 _k16 _k21 _k24 _k28 _ks4 _ks7 _ks8 _ks12 _ks14 _ks21"
run () { env TT_METAL_HOME=$W PYTHONPATH=$W/ttnn:$W timeout 7200 $PY $O/mm_sweep/mm_sweep.py "$@" --filter $KB --tdp "$(tdp)" --out $O/mm_sweep/full.jsonl < /dev/null >> $O/mm_sweep/full.log 2>&1; echo "rc=$? $* $(date +%T)"; }
P="qkv_s qkv_g o_s o_g gate down"
run --m 256 --proj $P --families model default 1d 2d 2d_t 1d_ws 2d_bs dram_sharded
run --m 1024 --proj $P --families model default 2d 2d_t 2d_bs
run --m 512 --proj $P --families model default 1d 2d 2d_t 2d_bs dram_sharded
run --m 416 832 1248 --proj $P --families model default 1d 2d 2d_t
echo "=== mm full done $(date +%T)"
