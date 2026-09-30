exec 9>/data/kmabee/runs_sasha/queue.lock; flock 9
source /data/kmabee/runs_sasha/env.sh
cd $W && [ -z "$(git status --short --untracked-files=no)" ] || { echo "tree dirty"; exit 1; }
echo "=== autov2 merge $(date +%T)"
git checkout -q -B kmabee/combined-main-mm-oob 87ab006f63e && git merge -q --no-edit origin/rmillerTT/mm-oob || { echo "merge FAIL"; git merge --abort; exit 1; }
echo "HEAD=$(git rev-parse --short HEAD)"
(cmake build_Release > /dev/null 2>&1 && cmake --build build_Release --target install -j 64 > $O/build_autov2.log 2>&1) && echo "build ok $(date +%T)" || { echo "build FAIL"; grep -m5 error: $O/build_autov2.log; exit 1; }
purge
export TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-t3/$(git rev-parse --short HEAD)
$PY -c "import ttnn; print('has v2', hasattr(ttnn.CONFIG,'matmul_auto_config_v2'))"
waitchips
env TT_METAL_HOME=$W PYTHONPATH=$W/ttnn:$W timeout 7200 $PY $O/mm_sweep/mm_sweep.py --m 256 416 512 832 1024 1248 \
  --proj qkv_s qkv_g o_s o_g gate up down --families model default auto_v2 --tdp "$(tdp)" --out $O/mm_sweep/autov2.jsonl < /dev/null > $O/mm_sweep/autov2.log 2>&1
echo "sweep rc=$? $(date +%T)"
git checkout -q --detach 87ab006f63e && (cmake --build build_Release --target install -j 64 > $O/build_back.log 2>&1) && echo "back to 87ab006f63e build ok $(date +%T)" || echo "rebuild back FAIL"
purge
echo "=== autov2 done $(date +%T)"
