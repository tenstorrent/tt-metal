#!/bin/bash
# glxq_rank.sh CMD...: one MPI rank of a multi-Galaxy job (MULTI_GALAXY_HARNESS.md), started by tt-run on its host.
# tt-run passes every TT_* variable to the ranks, so the job's paths travel as TT_GLXQ_JOB (the job directory, on /data)
# and TT_GLXQ_TREE (the tree). Each rank gets
#   - its own TT_METAL_HOME: a symlink farm of the tree with its own generated/ (outputs don't collide across ranks);
#   - its own kernel cache: TT_GLXQ_JOB/cache/r<rank> (kept: the PC mapping and hang decoding read the ELFs);
#   - its own log: TT_GLXQ_JOB/ranks/r<rank>_<host>.log, plus r<rank>/info.txt (host, start, env, exit code).
# mpirun's tagged output carries only one status line per rank.
set -uo pipefail
r=${TT_RUN_RANK:?TT_RUN_RANK unset: start this through tt-run}
J=${TT_GLXQ_JOB:?}
T=${TT_GLXQ_TREE:?}
h=${GLXQ_HOST:-$(hostname -s)}  # GLXQ_HOST: glxq.sbatch stub mode only
F=$J/ranks/r$r
mkdir -p "$F/tree/generated" "$J/cache/r$r"
for e in "$T"/*; do
    n=$(basename "$e")
    [ "$n" = generated ] || ln -sfn "$e" "$F/tree/$n"
done
export TT_METAL_HOME=$F/tree TT_METAL_RUNTIME_ROOT=$F/tree TT_METAL_CACHE=$J/cache/r$r
cd "$F/tree" || exit 97
{
    echo "rank $r host $h start $(date -u +%FT%TZ) cmd: $*"
    env | grep -E '^(TT_|LD_LIBRARY_PATH=|PATH=|OMPI_COMM_WORLD_RANK=|PMIX_RANK=)' | sort
} > "$F/info.txt"
"$@" > "$J/ranks/r${r}_$h.log" 2>&1
rc=$?
echo "rank $r host $h exit $rc $(date -u +%FT%TZ)" >> "$F/info.txt"
echo "[glxq] rank $r on $h: exit $rc ($(sed 's/\x1b\[[0-9;]*m//g' "$J/ranks/r${r}_$h.log" | grep -aoE 'Tests: [0-9]+ executed, [0-9]+ failed|[0-9]+ passed|[0-9]+ failed' | tail -1))"
exit $rc
