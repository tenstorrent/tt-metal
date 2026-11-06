#!/bin/bash
# Single-ERISC pass form A/B on one P150 LoudBox (an lb2 job; never a09u32), coordinator 2026-11-05 (local, never committed).
# Three arms, every run TT_FABRIC_NO_BW_TELEMETRY=1 (run_port), each its own fresh kernel cache per pass:
#   main  d2bd12fcf1f, main's router as shipped (pr_perf_runs.sh's staged tree /data/aagarwal/lb2/trees/pr_main_d2bd)
#   A     ba486eb70ba as built (origin/aagarwal/fabric-router-mainline tip): Blackhole's single-ERISC passes out of line
#         over their state records (staged /data/aagarwal/lb2/trees/single_ab_ba48, built in /home/aagarwal/tt-metal-single-ab)
#   B     A with router_single.hpp = router_single.B.hpp (arm_b_inline.patch: Wormhole's inline form, SINGLE_ERISC_PASS
#         FORCE_INLINE and inline_passes = true, on Blackhole too): a farm of A (mk_arm_b.py), A's host binaries, at
#         /home/aagarwal/tt-metal-single-in (a path as long as A's, so no path length differs between the arms' builds)
# Both staged trees carry only fabric_exp/port_tune/local_patches/test_tt_fabric_no_bw_telemetry.patch (the stock
# test_tt_fabric ignores TT_FABRIC_NO_BW_TELEMETRY); this job unpacks them at their build paths (the binaries' RUNPATH)
# and checks sha256, MD5SUMS and COMMIT, as pr_perf_runs.sh. A path that holds something this harness didn't put there
# (no .lb2_pr_owned) is never touched.
# Runs (suite: yaml, env; one ERISC = TT_METAL_DISABLE_FABRIC_TWO_ERISC=1; mux runs one ERISC by itself on Blackhole):
#   pass a, per arm main, A, B: data-checked validation in one-ERISC mode, mesh_1e (test_fabric_mesh_sanity_common.yaml)
#     and val1d_1e (v4/val1d_m1.yaml: val1d without rings, mux, telemetry and trimming); then the perf suites in order
#     4352_1e 1d_1e rse mux, each with arms main, A, B
#   pass b: the perf suites reversed, each with arms B, A, main
# Labels cen${SAB_PASS:-sab}<a|b>_<main|A|B>_<suite> (census_report.py's format): results/<label>.log, .csv, _tests.csv.
# Pass a also keeps each arm's router builds (ELF + generated headers of every fabric_erisc_router build the JIT made):
# results/sab_routers_<arm>.tar.zst. A hang (past the timeout: killed, captured into results/<label>_hangcap) or a run
# with no result line stops the job; a failed run is reported and the job goes on; executed + skipped tests are
# checked against the expected count (a node with fewer routing planes skips multi-link tests). Every run also sets
# TT_METAL_LOG_JIT_TELEMETRY=1 (logging only: the host logs, at exit, its least kernel-config-buffer headroom per core
# type; the program-size check is skipped for fabric programs on mock clusters, so this is where the fit shows).
#   single_ab_runs.sh           DRY=1: print the runs only (no unpack, no device)
cd /home/aagarwal/tt-metal || exit 1
CON=fabric_exp/consolidation
out=fabric_exp/results
D=fabric_exp/pr_perf/single_ab
DRY=${DRY:-0}
P=${SAB_PASS:-sab}
G=/data/aagarwal/lb2/trees
declare -A TREE=([main]=/home/aagarwal/tt-metal-main [A]=/home/aagarwal/tt-metal-single-ab [B]=/home/aagarwal/tt-metal-single-in)
declare -A TAR=([main]=$G/pr_main_d2bd.tar.zst [A]=$G/single_ab_ba48.tar.zst)
declare -A WANT=([main]=d2bd12fcf1f [A]=ba486eb70ba)
E1="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
# suite -> yaml timeout expected-tests env
suite_of() {
    case $1 in
        mesh_1e) echo "$SAN/test_fabric_mesh_sanity_common.yaml 300 96 $E1" ;;
        val1d_1e) echo "fabric_exp/v4/val1d_m1.yaml 300 143 $E1" ;;
        4352_1e) echo "fabric_exp/exp_2d_4352.yaml 300 10 $E1" ;;
        1d_1e) echo "fabric_exp/exp_1d_4352.yaml 300 16 $E1" ;;
        rse) echo "fabric_exp/pr_perf/pr_ring_se.yaml 300 24 $E1" ;;
        mux) echo "fabric_exp/pr_perf/pr_mux.yaml 420 33" ;;
        *) return 1 ;;
    esac
}
VALS=(mesh_1e val1d_1e)
SUITES=(4352_1e 1d_1e rse mux)
ARMS=(main A B)
for s in "${VALS[@]}" "${SUITES[@]}"; do y=$(suite_of $s | cut -d' ' -f1); [ -f $y ] || { echo "!!! no $y"; exit 1; }; done
for r in main A; do [ -f ${TAR[$r]} ] && [ -f ${TAR[$r]}.sha256 ] || { echo "!!! no ${TAR[$r]}"; exit 1; }; done
for f in mk_arm_b.py router_single.A.hpp router_single.B.hpp; do [ -f $D/$f ] || { echo "!!! no $D/$f"; exit 1; }; done

unpack() {  # router (pr_perf_runs.sh's)
    local r=$1 T=${TREE[$1]} f=${TAR[$1]}
    if [ -e $T ] && [ ! -e $T/.lb2_pr_owned ]; then echo "!!! $T exists and is not this harness's: not touched"; exit 1; fi
    (cd $(dirname $f) && sha256sum -c --quiet $(basename $f).sha256) || { echo "!!! $f fails its sha256"; exit 1; }
    rm -rf "${T:?}.new" && mkdir -p $T.new && tar -I zstd -xf $f -C $T.new || { echo "!!! unpack of $f failed"; exit 1; }
    (cd $T.new && md5sum -c --quiet MD5SUMS) || { echo "!!! $r: binaries differ from MD5SUMS"; exit 1; }
    case $(cat $T.new/COMMIT) in ${WANT[$r]}*) ;; *) echo "!!! $r tree at $(cat $T.new/COMMIT), want ${WANT[$r]}"; exit 1 ;; esac
    ln -sfn build_Release $T.new/build
    echo "created by $(hostname) job ${SLURM_JOB_ID:-?} $(date -u +%FT%TZ) from $f" > $T.new/.lb2_pr_owned
    rm -rf "${T:?}" && mv $T.new $T || { echo "!!! could not put the tree at $T"; exit 1; }
    echo "$r: $(cat $T/COMMIT | cut -c1-11) + $(grep -c '^[+-][^+-]' $T/PATCH) patch lines at $T ($(du -sh $T | cut -f1))"
}
arm_b() {  # B = farm of A with router_single.B.hpp; an existing B is replaced only when it is a farm mk_arm_b.py made
    local T=${TREE[B]}
    python3 -B $D/mk_arm_b.py ${TREE[A]} $T || { echo "!!! arm B not made"; exit 1; }
    echo "created by $(hostname) job ${SLURM_JOB_ID:-?} $(date -u +%FT%TZ): farm of ${TREE[A]}" > $T/.lb2_pr_owned
    echo "B: $(sha256sum $T/tt_metal/fabric/hw/inc/edm_fabric/router/router_single.hpp | cut -c1-12) router_single.hpp; A: $(sha256sum ${TREE[A]}/tt_metal/fabric/hw/inc/edm_fabric/router/router_single.hpp | cut -c1-12)"
}
cache() { echo /home/aagarwal/.tt-metal-cache-sab-$1-$2; }

if [ "$DRY" != 1 ]; then
    [ "$(hostname)" != bh-lb-120-a09u32 ] || { echo "!!! a09u32 holds the build worktrees: run this on an lb2 node"; exit 1; }
    source $CON/vc1_capture_lib.sh
    source fabric_exp/port_slots/port_lib.sh
    mkdir -p $out
    for r in main A; do unpack $r; done
    arm_b
    for pass in a b; do for r in "${ARMS[@]}"; do c=$(cache $r $pass); rm -rf "${c:?}"; mkdir -p $c; done; done
fi
one() {  # pass router suite
    local pass=$1 r=$2 s=$3 y t n env label rc got
    read -r y t n env <<< "$(suite_of $s)"
    label=cen${P}${pass}_${r}_$s
    if [ "$DRY" = 1 ]; then echo "   $label: ${TREE[$r]} $y (timeout $t, $n tests) ${env:-}"; return 0; fi
    PT=${TREE[$r]} C=$(cache $r $pass)
    run_port $label $y $t $env TT_METAL_LOG_JIT_TELEMETRY=1 TT_METAL_RUNTIME_ROOT=$PT; rc=$?
    [ $rc = 2 ] && { echo "!!! $label hung (captured): stopping"; exit 1; }
    # the kernels this run compiled must come from the arm's own tree (B: the farm, whose router_single.hpp is B's)
    local roots=$(clean $out/$label.log | grep -a "compile cmd:" | grep -aoE " -I[^ ]*/tt_metal/hw/inc " | sort | uniq -c)
    local nr=$(grep -c . <<< "$roots") ok=$(grep -cF -- " -I$PT/tt_metal/hw/inc " <<< "$roots")
    if [ -n "$roots" ] && { [ "$nr" != 1 ] || [ "$ok" != 1 ]; }; then
        echo "!!! $label compiled from another root:"; echo "$roots"; echo "!!! stopping"; exit 1
    fi
    [ -z "$roots" ] || echo "   $label: $(awk '{print $1}' <<< "$roots") compiles, all from $PT"
    got=$(clean $out/$label.log | grep -aoE 'Tests: [0-9]+ executed' | tail -1 | grep -oE '[0-9]+')
    [ -n "$got" ] || { echo "!!! $label: no result line: stopping"; exit 1; }
    if [ "$got" != "$n" ]; then
        echo "!!! $label: $got executed, want $n; skips: $(clean $out/$label.log | grep -aoE 'Test groups: [0-9]+ total, [0-9]+ ran, [0-9]+ skipped' | tail -1)"
    fi
    [ $rc = 0 ] || echo "!!! $label failed (rc $rc): going on"
    return 0
}
keep_routers() {  # arm: pass a's router builds (ELF + generated headers), for the offline image check
    local c=$(cache $1 a) f=$out/sab_routers_$1.tar.zst
    (cd $c && find . -path '*/kernels/fabric_erisc_router*/*' \( -name '*.elf' ! -name '*.xip.elf' -o -name '*.h' -o -name '*.hpp' \) -print |
        sort | tar -I 'zstd -T8 -19' -cf $OLDPWD/$f -T -) && echo "   $1: $(tar -I zstd -tf $f | grep -c '\.elf$') router ELFs kept in $f ($(du -h $f | cut -f1))"
}
echo "##### pass a: validation (${VALS[*]}) per arm, then ${SUITES[*]} with arms ${ARMS[*]} (job at ${SECONDS} s)"
for r in "${ARMS[@]}"; do for s in "${VALS[@]}"; do one a $r $s; done; done
for s in "${SUITES[@]}"; do for r in "${ARMS[@]}"; do one a $r $s; done; done
[ "$DRY" = 1 ] || for r in "${ARMS[@]}"; do keep_routers $r; done
echo "##### pass b: suites reversed, arms reversed (job at ${SECONDS} s)"
for ((i = ${#SUITES[@]} - 1; i >= 0; i--)); do for ((j = ${#ARMS[@]} - 1; j >= 0; j--)); do one b ${ARMS[$j]} ${SUITES[$i]}; done; done
echo "ALL_DONE (job at ${SECONDS} s)"
