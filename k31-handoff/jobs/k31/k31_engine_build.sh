#!/bin/bash
#SBATCH -N 1
#SBATCH -t 4:00:00
# k31_engine_build.sh: offline build of /data/$USER/tt-llm-engine (eed816d4, cloned from Phil's tree via a bundle):
# its bundled tt-metal (e49ac055, a worktree of /data/$USER/tt-metal) without Python bindings or host Tracy, then the
# migration layer (migration_endpoint, migration_worker, _migration_client.so), as M3_MIGRATION_RUNBOOK.md 0.1 b.
# exabox reaches no GitHub: CPM caches are copies whose hash dirs are relinked (CPM hashes args with tree paths),
# and googletest comes from the CPM cache instead of its URL.
set -uo pipefail
E=/data/$USER/tt-llm-engine M=$E/tt-metal
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.file:///nonexistent-offline/.insteadOf GIT_CONFIG_VALUE_0=https://github.com/

# relink_loop <label> <cmd...>: run a configure command; when it fails on a CPM package whose cache dir is missing,
# link that dir to the copy beside it and rerun.
relink_loop() {
  local label=$1; shift
  for round in $(seq 1 40); do
    out=$("$@" 2>&1); rc=$?
    if [ $rc = 0 ]; then echo "[$label] configure ok after $round round(s)"; return 0; fi
    line=$(grep -E "CPM: (.*: )?Adding package .* to .*\.cpmcache/" <<<"$out" | tail -1)
    path=$(sed -E 's/.* to (.*)\)$/\1/' <<<"$line")
    if [ -z "$path" ] || [ -e "$path" ]; then echo "[$label] round $round: nothing to relink (rc=$rc)"; tail -40 <<<"$out"; return 1; fi
    src=$(find "$(dirname "$path")" -mindepth 1 -maxdepth 1 -type d ! -name "$(basename "$path")" | head -1)
    [ -n "$src" ] || { echo "[$label] no copy to link for $path"; tail -40 <<<"$out"; return 1; }
    ln -sfn "$(basename "$src")" "$path"; echo "[$label] round $round: linked $path -> $(basename "$src")"
  done
  return 1
}

cd $M || exit 1
echo "[engine] tt-metal $(git log -1 --format='%h %cs')"
relink_loop tt-metal ./build_metal.sh --build-type Release --without-python-bindings --disable-profiler --configure-only || exit 1
./build_metal.sh --build-type Release --without-python-bindings --disable-profiler 2>&1 | tail -5
ls $M/build_Release/lib/libtt_metal.so || { echo "[engine] no libtt_metal.so"; exit 1; }
[ -e $M/build ] || ln -sfn build_Release $M/build

cd $E || exit 1
B=$E/disaggregation/migration/build_RelWithDebInfo
GTEST=$(ls -d $M/.cpmcache/gtest/*/ | head -1)
cfg=(cmake -S $E/disaggregation/migration -B $B -DCMAKE_BUILD_TYPE=RelWithDebInfo -DTT_METAL_DIR=$M
     -DTT_METAL_BUILD_DIR=$M/build_Release -DBUILD_DEVICE_TESTS=OFF -DWORKER_SYNTHETIC_ONLY=OFF
     -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${GTEST%/})
export MPI_HOME=${MPI_HOME:-/opt/openmpi-v5.0.7-ulfm}
relink_loop migration "${cfg[@]}" || exit 1
cmake --build $B -j "$(nproc)" 2>&1 | tail -5
ls -la $B/bin/migration_endpoint $B/bin/migration_worker $B/python/_migration_client*.so || exit 1
readelf -d $B/bin/migration_worker | grep -E "RUNPATH|RPATH"
echo "[engine] build ok $(git -C $E log -1 --format=%h) / tt-metal $(git -C $M log -1 --format=%h)"
