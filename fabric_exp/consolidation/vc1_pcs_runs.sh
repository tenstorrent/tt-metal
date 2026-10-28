#!/bin/bash
# PC sampling of a router variant on the VC1 split (two 2x2 meshes, ttrun.py; RESULTS §98 method):
#   vc1_pcs_runs.sh VARIANT-DIR|tree TAG LEVEL ROW...      (LEVEL: O3, O2, auto; ROW: ew16, ew4352, mcast)
# A fresh kernel cache: the short compile row (vc1/pcs_vc1_compile.yaml) builds the routers and logs their compile
# commands (for the -g twins: pcs_sections.py twins <that log> ...); each ROW (vc1/pcs_vc1_<row>.yaml, one long row of
# vc1_split.yaml's benchmarks) then runs while pc_sampler.py reads every router ERISC's PC on all eight chips.
# Labels vc1pcs_<tag>_<row>: the run log (_13129088_trial.log), the samples (_samples.bin/.json), and a trial CSV
# (_13129088_trial.csv) with the row's rate as the pcs tools read it: the slowest sender's packets per second.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
VAR=$1; TAG=$2; LEVEL=$3; shift 3
C=/home/aagarwal/.tt-metal-cache-vc1pcs-$TAG
VC1CAP_RUN_TIMEOUT=150
rm -rf "${C:?}"; mkdir -p $C
_vc1cap_trap_on
if [ "$VAR" != tree ]; then python3 $CON/swap.py apply $VAR $BK || { echo STOPPED; exit 1; }; fi
lab=vc1pcs_${TAG}_compile
echo "=== $lab: opt $LEVEL, cache $C ==="
VC1CAP_YAML=fabric_exp/vc1/pcs_vc1_compile.yaml _vc1cap_run $out/${lab}_13129088_trial.log $C $LEVEL
echo "exit $VC1CAP_RC after ${VC1CAP_SECS} s; $VC1CAP_FIN finishes (2 expected); $VC1CAP_BAD error lines"
[ $VC1CAP_RC = 0 ] && [ "$VC1CAP_BAD" = 0 ] || { echo STOPPED; exit 1; }
for r in "$@"; do
    lab=vc1pcs_${TAG}_${r}
    log=$out/${lab}_13129088_trial.log
    rm -f $log $out/${lab}_samples.bin $out/${lab}_samples.json $out/${lab}_13129088_trial.csv \
        generated/fabric/bandwidth_results_blackhole.csv
    echo "=== $lab ==="
    timeout 300 python_env/bin/python $CON/pc_sampler.py $out/${lab}_samples --start-log $log --max-s 60 2>&1 \
        | grep --line-buffered -E '^sampler: [0-9]+ samples|start pattern never|Error|error|Traceback' &
    SP=$!
    VC1CAP_YAML=fabric_exp/vc1/pcs_vc1_${r}.yaml _vc1cap_run $log $C $LEVEL
    echo "exit $VC1CAP_RC after ${VC1CAP_SECS} s; $VC1CAP_FIN finishes (2 expected); $VC1CAP_BAD error lines"
    wait $SP
    [ $VC1CAP_RC = 0 ] && [ "$VC1CAP_BAD" = 0 ] || { echo "!!! $r failed"; break; }
    python3 - generated/fabric/bandwidth_results_blackhole.csv $out/${lab}_13129088_trial.csv $r <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
pps = min(float(r["packets_per_second"]) for r in rows)
with open(sys.argv[2], "w") as f:
    f.write("test_name,avg_packets_per_s\n%s,%f\n" % (rows[0]["test_name"], pps))
print(f"{sys.argv[3]}: {len(rows)} senders, slowest {1.35e9 / pps:.2f} cycles/packet")
EOF
done
if [ "$VAR" != tree ]; then python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }; fi
_vc1cap_trap_off
rm -f $M
echo ALL_DONE
