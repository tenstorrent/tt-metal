# SDPA final-chunk math util across chunk sizes (PR 58135 test + local sweep test, applied as a patch).
exec 9>/data/kmabee/runs_sasha/queue.lock; flock 9
source /data/kmabee/runs_sasha/env.sh
cd $W && git apply $O/sdpa_sweep.patch || exit 1
trap 'cd $W && git apply -R $O/sdpa_sweep.patch; echo reverted' EXIT
TID="tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_gemma4_sdpa_chunk_sweep"
run () { local L=$1; shift; waitchips
  (cd $W && env "$@" TT_METAL_HOME=$W PYTHONPATH=$W/ttnn:$W timeout 1500 $PY -m pytest "$TID" -sv -p no:cacheprovider < /dev/null > $O/sdpa/$L.log 2>&1)
  echo "$L rc=$? $(grep -o 'G4S_RESULT.*' $O/sdpa/$L.log | tail -1) $(grep -m1 -oE '(RuntimeError|TT_THROW|TT_FATAL|AssertionError)[^\n]{0,200}' $O/sdpa/$L.log)"; }
echo "=== sdpa sweep $(date +%T) sha=$(git -C $W rev-parse --short HEAD) $(tdp)"
G="G4S_MODEL=gemma4_global"; P="G4S_LOFI=1 G4S_SEG=1"
# calibration against the PR's 55.4% (HiFi2, no segments)
run g_c8192_q96k256_pr $G G4S_CHUNK=8192 G4S_Q=96 G4S_K=256
# production global configs
run g_c2048_q64k256s3 $G $P G4S_CHUNK=2048 G4S_Q=64 G4S_K=256 G4S_KSPLITS=3
run g_c4096_q128k256s3 $G $P G4S_CHUNK=4096 G4S_Q=128 G4S_K=256 G4S_KSPLITS=3
run g_c8192_q96k256 $G $P G4S_CHUNK=8192 G4S_Q=96 G4S_K=256
# #57836 sizes
for qk in "32 256" "32 416" "32 512"; do set -- $qk; run g_c3328_q$1k$2 $G $P G4S_CHUNK=3328 G4S_Q=$1 G4S_K=$2; done
for qk in "64 256" "64 320"; do set -- $qk; run g_c6656_q$1k$2 $G $P G4S_CHUNK=6656 G4S_Q=$1 G4S_K=$2; done
for qk in "96 256" "96 288"; do set -- $qk; run g_c9984_q$1k$2 $G $P G4S_CHUNK=9984 G4S_Q=$1 G4S_K=$2; done
# 8192 with the #57836 style q (64 -> 128 units/8 heads) for reference
run g_c8192_q64k256 $G $P G4S_CHUNK=8192 G4S_Q=64 G4S_K=256
# sliding (production q128 k128 HiFi2)
S="G4S_MODEL=gemma4_swa"
for C in 2048 4096 8192 3328 6656 9984; do run s_c${C}_q128k128 $S G4S_CHUNK=$C G4S_Q=128 G4S_K=128; done
for C in 3328 6656 9984; do run s_c${C}_q64k128 $S G4S_CHUNK=$C G4S_Q=64 G4S_K=128; done
echo "=== sdpa done $(date +%T)"
