#!/bin/bash
# /data inode relief for a running rf_glx.sbatch job (coordinator, 2026-11-05: /data at 97% of its inodes): every
# kernel cache the job writes under $J is deleted as soon as the last step that uses it has finished, after its router
# builds (ELFs with the JIT's -g, generated headers: what the breakdown mapping and the slot rows need) are kept as one
# tarball in $J/results/<cache>_routers.tgz. Polls the job log; exits when the job has ended.
#   rf_glx_cache_cleaner.sh JOBID
J=/data/aagarwal/frs-glx/jobs/$1
L=/data/aagarwal/frs-glx/logs/frs-glx-rf-$1.log
declare -A LAST=(  # cache: the step after which no step uses it (rf_glx.sbatch's step order)
    [cache_main_a]=a_main_nx [cache_port_a]=a_port_nx [bdcache_main_a]=a_main_s_fullring [bdcache_port_a]=a_port_s_fullring
    [cache_mainh_a]=a_mainh_vc02m4352 [cache_porth_a]=a_porth_vc02m4352
    [cache_mainh_a_mvc]=a_mainh_B_main_match [cache_porth_a_mvc]=a_porth_B_new_match
    [cache_porth_b_mvc]=b_porth_B_new_base [cache_mainh_b_mvc]=b_mainh_B_main_base
    [cache_porth_b]=b_porth_ew4352vc2 [cache_mainh_b]=b_mainh_ew4352vc2
    [bdcache_port_b]=b_port_s_fullring [bdcache_main_b]=b_main_s_fullring [cache_port_b]=b_port_linear [cache_main_b]=b_main_linear)
done_step() { grep -qE "^$1: exit |^!!! $1[: ]|^skipped $1 " $L 2>/dev/null; }
while :; do
    for c in "${!LAST[@]}"; do
        d=$J/$c
        [ -d $d ] && done_step ${LAST[$c]} || continue
        t=$J/results/${c}_routers.tgz
        if [ ! -s $t ]; then
            # the breakdown caches keep the router ELFs (mapping); the others only the generated headers (configs)
            case $c in bdcache_*) pat=(-name '*.elf' -o -name '*.h' -o -name '*.hpp') ;; *) pat=(-name '*.h' -o -name '*.hpp') ;; esac
            (cd $d && find . -path '*/kernels/fabric_erisc_router*/*' \( "${pat[@]}" \) -print |
                tar czf $t.part -T - && mv $t.part $t) || { echo "$(date -u +%T) $c: tar failed, kept"; continue; }
        fi
        n=$(find $d | wc -l)
        rm -rf "${d:?}" && echo "$(date -u +%T) $c: router builds kept in $(basename $t) ($(du -k $t | cut -f1) KB), $n entries deleted"
    done
    squeue -j $1 -h 2>/dev/null | grep -q . || { echo "$(date -u +%T) job $1 ended"; break; }
    sleep 60
done
