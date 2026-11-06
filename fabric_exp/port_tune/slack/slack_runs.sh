#!/bin/bash
# Slack study device runs (NOTES.md; local patches, never committed): per-side headroom at saturation of main's router
# and the port, by a fixed spin of D cycles per packet on one ERISC (tx<D>: E0, after each packet sent; rx<D>: E1, after
# each packet received and handled; base: the plain router).
#   Trees: main /home/aagarwal/tt-metal-main at d2bd12fcf1f (its only local change: test_tt_fabric.cpp's
#          TT_FABRIC_NO_BW_TELEMETRY switch); port /home/aagarwal/tt-metal-port-slack at e7fe1899049 + exactly
#          port_tune/local_patches/test_tt_fabric_no_bw_telemetry.patch. Each tree's own host build.
#   Roots: /home/aagarwal/tt-metal-slack-roots/<router>_<arm> (mk_roots.py: symlink farms of the tree with only the
#          spin's call sites and slack_study_spin.hpp edited; base runs <router>_plain). The router JIT-compiles from the
#          root, so arms differ only in device code (offline_check.sh / img_check.py: IMG_CHECK_OK before any run).
#   Rows: slack_rows.yaml. Router bandwidth telemetry off (TT_FABRIC_NO_BW_TELEMETRY=1). One JIT cache per router and
#          arm (~/.tt-metal-cache-slack-<router>-<arm>, new for this study; the JIT rebuilds on any source change).
#   slack_runs.sh PASS STEP...   STEP = <router>_<arm> (e.g. main_base port_tx4; base2 = a mid-pass base run), in order.
# Results: fabric_exp/results/slack/s<TAG><PASS>_<router>_<arm>.{log,csv,_detail.csv}. Stops at the first failed or hung run,
# and before a run that would not finish inside the job's 1500 s (prints the steps not run).
# Broker job on bh-lb-120-a09u32 (workspace /home/aagarwal, timeout 1500 s):
#   bash /home/aagarwal/tt-metal/fabric_exp/port_tune/slack/slack_runs.sh a main_base port_base ...
# DRY=1: the checks, then the runs it would do.
set -uo pipefail
cd /home/aagarwal/tt-metal
S=fabric_exp/port_tune/slack
R=/home/aagarwal/tt-metal-slack-roots
out=/home/aagarwal/tt-metal/fabric_exp/results/slack
RUN_TIMEOUT=${RUN_TIMEOUT:-420}
BUDGET=1440
PASS=$1; shift
TAG=${TAG:-}  # a set's tag in the result labels (s<TAG><pass>_...): "" = the wall-clock ladder (jobs 833-837), m = the
              # mcycle settling A/B (NOTES.md section 10)
STEPS=("$@")
declare -A TREE=([main]=/home/aagarwal/tt-metal-main [port]=/home/aagarwal/tt-metal-port-slack)
declare -A SHA=([main]=d2bd12fcf1f [port]=e7fe1899049)
(while sleep 30; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
echo "slack_runs: $(hostname), set '${TAG}', pass $PASS, steps: ${STEPS[*]}"

# --- checks before any device run
[ "$(hostname)" = bh-lb-120-a09u32 ] || { echo "!!! expected bh-lb-120-a09u32"; exit 1; }
case $PASS in a|b) ;; *) echo "!!! pass must be a or b"; exit 1 ;; esac
for r in main port; do
    T=${TREE[$r]}
    [ "$(git -C $T rev-parse --short=11 HEAD)" = ${SHA[$r]} ] || { echo "!!! $T is not at ${SHA[$r]}"; exit 1; }
    [ "$(git -C $T diff --name-only)" = tests/tt_metal/tt_fabric/test_infra/test_tt_fabric.cpp ] \
        || { echo "!!! $T differs from HEAD beyond test_tt_fabric.cpp"; exit 1; }
    grep -q TT_FABRIC_NO_BW_TELEMETRY $T/tests/tt_metal/tt_fabric/test_infra/test_tt_fabric.cpp \
        || { echo "!!! $T's test_tt_fabric does not read TT_FABRIC_NO_BW_TELEMETRY"; exit 1; }
    [ $T/build_Release/test/tt_metal/tt_fabric/test_infra/test_tt_fabric -nt $T/tests/tt_metal/tt_fabric/test_infra/test_tt_fabric.cpp ] \
        || { echo "!!! $T's test binary is older than its patched source"; exit 1; }
    echo "tree $r: $T at ${SHA[$r]}; binary md5 $(md5sum < $T/build_Release/test/tt_metal/tt_fabric/test_infra/test_tt_fabric | cut -c1-12)," \
        "libtt_metal md5 $(md5sum < $T/build_Release/tt_metal/libtt_metal.so | cut -c1-12)"
done
norm() { grep -vE '^(index |@@ )' | sed -n '/^diff --git/,/^-- $/p' | grep -v '^-- $'; }
diff <(git -C ${TREE[port]} diff HEAD | norm) <(norm < fabric_exp/port_tune/local_patches/test_tt_fabric_no_bw_telemetry.patch) > /dev/null \
    || { echo "!!! the port tree's local diff is not exactly the telemetry-switch patch"; exit 1; }
python3 - $R $S/roots_manifest.json "${STEPS[@]}" <<'PY' || exit 1
# every step's root: edited files exactly as the manifest (written with the offline check), the rest symlinks into its tree
import hashlib, json, os, sys
R, man, steps = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3:]
tree = {"main": "/home/aagarwal/tt-metal-main", "port": "/home/aagarwal/tt-metal-port-slack"}
for st in steps:
    router, arm = st.split("_", 1)
    name = f"{router}_{'plain' if arm.startswith('base') else arm}"
    root = os.path.join(R, name)
    got = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d != "generated" and not os.path.islink(os.path.join(dp, d))]
        for f in fn:
            p = os.path.join(dp, f)
            if os.path.islink(p):
                if not os.readlink(p).startswith(tree[router] + "/"):
                    sys.exit(f"!!! {p} points outside {tree[router]}")
            else:
                got[os.path.relpath(p, root)] = hashlib.md5(open(p, "rb").read()).hexdigest()
    if got != man[name]:
        sys.exit(f"!!! {root}: edited files differ from roots_manifest.json")
print(f"roots: {len(steps)} steps, edited files as roots_manifest.json")
PY
[ -n "${DRY:-}" ] || mkdir -p $out

run() {  # router arm
    local router=$1 arm=$2 T=${TREE[$1]} label=s${TAG}${PASS}_$1_$2 t0=$SECONDS rc
    local carm=$arm; [[ $arm == base* ]] && carm=base  # base, base2 (a mid-pass base): the plain root
    local root=$R/${router}_$([ $carm = base ] && echo plain || echo $arm)
    local B=$T/build_Release cache=/home/aagarwal/.tt-metal-cache-slack-$router-$carm log=$out/$label.log
    local G=$root/generated/fabric
    echo "=== $label: root $root, cache $cache"
    [ -n "${DRY:-}" ] && return 0
    rm -f $G/bandwidth_summary_results_blackhole.csv $G/bandwidth_results_blackhole.csv \
        $G/bandwidth_summary_results_blackhole_upload.csv
    (cd $root && env $(env | grep -o '^TT_METAL_STREAMING_PROFILER[A-Z_]*' | sed 's/^/-u /') -u TT_METAL_SLOW_DISPATCH_MODE \
        TT_METAL_HOME=$root TT_METAL_RUNTIME_ROOT=$root TT_METAL_CACHE=$cache TT_FABRIC_NO_BW_TELEMETRY=1 \
        LD_LIBRARY_PATH=$B/tt_metal:$B/tt_stl:$B/lib:$B/tt_metal/third_party/umd/lib \
        timeout -k 30 $RUN_TIMEOUT $B/test/tt_metal/tt_fabric/test_infra/test_tt_fabric \
        --test_config /home/aagarwal/tt-metal/$S/slack_rows.yaml > $log 2>&1 < /dev/null)
    rc=$?
    sed -i 's/\x1b\[[0-9;]*m//g' $log
    cp $G/bandwidth_summary_results_blackhole.csv $out/$label.csv 2>/dev/null
    cp $G/bandwidth_results_blackhole.csv $out/${label}_detail.csv 2>/dev/null
    echo "$label: exit $rc after $((SECONDS - t0)) s; $(grep -ac 'Running Test:' $log) tests;" \
        "summary rows $(($(wc -l < $out/$label.csv 2>/dev/null || echo 1) - 1))"
    python3 - $out/$label.csv <<'PY'
import csv, sys
try:
    for r in csv.DictReader(open(sys.argv[1])):
        print(f"   {r['test_name']:24s} links {r['num_links']} {r['packet_size']:>5s} B: "
              f"{1.35e9 / float(r['avg_packets_per_s']):7.2f} cycles/packet, {float(r['avg_bandwidth_gigabytes_per_s']):6.2f} GB/s"
              f" (iterations {r['iterations']}, std {float(r['bw_std_dev_gigabytes_per_s']):.3f})")
except FileNotFoundError:
    pass
PY
    grep -aE "TT_FATAL|TT_THROW|terminate called|Timed out|timeout" $log | head -3 | sed 's/^/   | /'
    [ $rc = 0 ] || { echo "!!! $label: rc $rc (124 = run timeout: a hang): STOPPING"; return 1; }
    return 0
}

done_n=0
longest=150  # the longest run so far (s), at least 150: a step starts only if one that long still fits the budget
for st in "${STEPS[@]}"; do
    if [ -z "${DRY:-}" ] && [ $((SECONDS + longest + 30)) -gt $BUDGET ]; then
        echo "=== stopping at ${SECONDS} s (budget): NOT RUN: ${STEPS[*]:$done_n}"; exit 3
    fi
    t0=$SECONDS
    run ${st%%_*} ${st#*_} || { echo "NOT RUN: ${STEPS[*]:$((done_n + 1))}"; exit 1; }
    [ $((SECONDS - t0)) -gt $longest ] && longest=$((SECONDS - t0))
    done_n=$((done_n + 1))
done
echo "slack_runs done: pass $PASS, ${#STEPS[@]} steps in ${SECONDS} s"
