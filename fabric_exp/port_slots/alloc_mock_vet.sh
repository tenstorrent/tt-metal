#!/bin/bash
# Mock vet of the staged allocator's device A/B (SLOT_ALLOCATOR.md section 12.14; CPU only, no device): for every suite
# of yamls/alloc_manifest.json and both arms (staged: no switch; tier: TT_FABRIC_EXP_TIER_SLOTS=1), one mock run per
# distinct fabric_setup of the suite (its first test, benchmark off) on the mock P150 (fabric_exp/a09u32_cluster_desc.yaml;
# vc1: two ranks through tt-run, each rank's mock the same descriptor), with the port-alloc build
# (/home/aagarwal/tt-metal-port-alloc: c53b5a5cbbd + alloc_tier_switch.patch) and TT_FABRIC_EXP_LOG_SLOTS=1. Each run
# must reach "Launching programs". The distinct slot rows each (suite, arm) logged go to
# yamls/alloc_expected_rows.json, which the device job (alloc_runs.sh) checks every run against.
#   [V=<out dir>] [J=<parallel runs>] [DESC=<cluster descriptor>] [M=<suite manifest>] [OUT=<expected rows json>]
#   alloc_mock_vet.sh [SUITE...]   (default: every suite of M)
set -uo pipefail
cd /home/aagarwal/tt-metal
PT=/home/aagarwal/tt-metal-port-alloc
D=fabric_exp/port_slots
V=${V:-/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/allocvet}
J=${J:-4}
DESC=${DESC:-/home/aagarwal/tt-metal/fabric_exp/a09u32_cluster_desc.yaml}  # the Galaxy jam rows: the single BH Galaxy
PY=/home/aagarwal/tt-metal/python_env/bin/python
RB=$PWD/fabric_exp/port_tune/cov/vc1/p150x8_2x2_split_rank_bindings.yaml
M=${M:-$D/yamls/alloc_manifest.json}
OUT=${OUT:-$D/yamls/alloc_expected_rows.json}  # the suites run here replace their entries; the others are kept
mkdir -p $V/yamls
suites=${*:-$(python3 -c "import json; print(' '.join(json.load(open('$M'))))")}
lib=$PT/build_Release
one() {  # suite arm setup-index yaml
    local s=$1 a=$2 i=$3 y=$4 out=$V/$1_$2_$3 cmd mock arm=() env
    rm -rf "${out:?}"; mkdir -p $out
    env=$(python3 -c "import json,sys; print(json.load(open('$M'))[sys.argv[1]]['env'])" $s)
    [ $a = tier ] && arm=(TT_FABRIC_EXP_TIER_SLOTS=1)
    cmd=($lib/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $y)
    mock=(TT_METAL_MOCK_CLUSTER_DESC_PATH=$DESC)
    case $s in vc1_*)
        printf 'rank_to_cluster_mock_cluster_desc:\n  0: "%s"\n  1: "%s"\n' $DESC $DESC > $out/mock_mapping.yaml
        cmd=($PY $PT/ttnn/ttnn/distributed/ttrun.py --mock-cluster-rank-binding $out/mock_mapping.yaml --rank-binding $RB
             --mpi-args --oversubscribe "${cmd[@]}")
        mock=() ;;
    esac
    (cd $PT && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$PT TT_METAL_RUNTIME_ROOT=$PT \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib \
        TT_METAL_CACHE=$out/cache TT_METAL_LOGS_PATH=$out/logs "${mock[@]}" \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_EXP_LOG_SLOTS=1 $env "${arm[@]}" \
        timeout 1500 nice -n 10 "${cmd[@]}" > $out/run.log 2>&1)
    sed 's/\x1b\[[0-9;]*m//g' $out/run.log > $out/run.clean
    local st=ok
    grep -aq "Launching programs" $out/run.clean || st="NO LAUNCH ($(grep -aE 'TT_FATAL|TT_THROW|error' $out/run.clean | head -1 | cut -c1-200))"
    grep -aqE "build failed|Failed to generate binaries" $out/run.clean && st="KERNEL BUILD FAILED"
    grep -aoE 'VC[0-9] sender slots \{[^}]*\} receiver slots \{[^}]*\} \(channel buffer [0-9]+ B\)' $out/run.clean | sort -u > $out/rows.txt
    printf '%-10s %-6s %s %s (%s rows)\n' $s $a $i "$st" $(wc -l < $out/rows.txt) > $out/status.txt
    rm -rf $out/cache
}
# one mock yaml per distinct fabric_setup: the first test with that setup, benchmark off
jobs_list=$V/jobs.txt; : > $jobs_list
for s in $suites; do
    python3 - $s $M $V/yamls <<'PY' >> $jobs_list
import json, sys, yaml
s, m, out = sys.argv[1:4]
src = json.load(open(m))[s]["yaml"]
d = yaml.safe_load(open("/home/aagarwal/tt-metal/" + src))
seen = {}
for t in d["Tests"]:
    k = json.dumps(t.get("fabric_setup"), sort_keys=True)
    if k not in seen:
        seen[k] = dict(t)
for i, t in enumerate(seen.values()):
    t["benchmark_mode"] = False
    y = f"{out}/{s}_{i}.yaml"
    yaml.safe_dump({k: v for k, v in d.items() if k != "Tests"} | {"Tests": [t]}, open(y, "w"), sort_keys=False)
    print(s, i, y)
PY
done
export -f one; export V M PT lib DESC PY RB
while read -r s i y; do for a in staged tier; do echo "$s $a $i $y"; done; done < $jobs_list |
    xargs -P $J -L 1 bash -c 'one "$0" "$1" "$2" "$3"'
cat $V/*/status.txt | sort | tee $V/summary.txt
python3 - $V $M $OUT <<'PY'
import glob, json, os, sys
v, m, out = sys.argv[1:4]
exp = {}
for d in sorted(glob.glob(os.path.join(v, "*_*_*/rows.txt"))):
    s, a, i = os.path.basename(os.path.dirname(d)).rsplit("_", 2)
    exp.setdefault(s, {}).setdefault(a, set()).update(l.strip() for l in open(d) if l.strip())
exp = {s: {a: sorted(r) for a, r in arms.items()} for s, arms in exp.items()}
kept = json.load(open(out)) if os.path.exists(out) else {}
json.dump(kept | exp, open(out, "w"), indent=1)
for s, arms in exp.items():
    for a, r in arms.items():
        print(f"{s:10} {a:6} " + " | ".join(x.split(" (channel")[0] for x in r))
PY
bad=$(grep -vc " ok (" $V/summary.txt)
[ "$bad" = 0 ] && echo ALLOC_VET_OK || { echo "ALLOC_VET: $bad not ok"; exit 1; }
