#!/bin/bash
# v4se single-ERISC on VC1 (2026-10-28; after se4_runs.sh, job 122: VC0 functional in both single-ERISC modes). Per
# MODE given, v4se on the VC1 split (vc1_split.yaml, the rows of main's single-ERISC baseline, se_runs.sh) and the
# inter-mesh rows (vc1_inter.yaml), hang capture armed on both, fresh caches, depths logged:
#   two   the default (two ERISCs per router), the same-job reference
#   aer1  TT_METAL_DISABLE_MULTI_AERISC=1 (router on ERISC 0)
#   fab1  TT_METAL_DISABLE_FABRIC_TWO_ERISC=1 (router on ERISC 1)
# A hang stops the job; failures do not. Labels s1<mode>vc1[i]_v4se_default_p1_o3.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
[ $# -gt 0 ] || { echo "usage: se4vc1_runs.sh MODE..."; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
L="TT_FABRIC_EXP_LOG_SLOTS=1"
declare -A MODE_ENV=([two]="" [aer1]="TT_METAL_DISABLE_MULTI_AERISC=1" [fab1]="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1")
procs() {  # log: which ERISCs the router binaries were built for
    sed 's/\x1b\[[0-9;]*m//g' $1 | grep -ao 'kernels/fabric_erisc_router_v2/[0-9]*/[a-z_]*erisc' | awk -F/ '{print $NF}' \
        | sort | uniq -c | tr '\n' ' '
}
python3 $CON/swap.py apply fabric_exp/variants/v4se $BK || { echo STOPPED; exit 1; }
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
for mode in "$@"; do
    [ -n "${MODE_ENV[$mode]+x}" ] || { echo "unknown mode $mode"; stop; }
    E=${MODE_ENV[$mode]}
    echo "##### mode $mode: ${E:-default} (job at ${SECONDS} s)"
    C=/home/aagarwal/.tt-metal-cache-s1vc1-$mode
    l=s1${mode}vc1_v4se_default_p1_o3
    VC1CAP_EXTRA_ENV="$L $E" run_vc1_cap $l $C O3 || echo "!!! VC1 split failed in mode $mode"
    [ -d $out/${l}_hangcap ] && stop
    echo "   VC1 router binaries: $(procs $out/${l}_13129088_vc1.log)"
    l=s1${mode}vc1i_v4se_default_p1_o3
    VC1CAP_EXTRA_ENV="$L $E" VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_inter.yaml VC1CAP_SPLIT_FINISHES=16 \
        run_vc1_cap $l $C O3 || echo "!!! VC1 inter rows failed in mode $mode"
    [ -d $out/${l}_hangcap ] && stop
done
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
