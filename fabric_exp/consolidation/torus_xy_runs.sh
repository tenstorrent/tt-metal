#!/bin/bash
# XY-torus rows on the P150x8 (user, 2026-11-03; run on an extra LB through lb2_run.sh): main's router as shipped
# against v4rxg3, same job, in two shapes. Modeled on report_size_runs.sh (same settings: main_fw = kernel v1, flags 0,
# firmware keepalive; a variant = swapped into the v2 slot, study settings, keepalive 1000; -O3, telemetry off).
#   torus_xy_runs.sh PASS SHAPE...      SHAPE: 2x4 (auto-discovered descriptor) | 4x2 (study descriptor
#                                       fabric_exp/torus/p150_x8_4x2_torus_xy_mgd.textproto, TT_MESH_GRAPH_DESC_PATH)
#   ROUTERS (default "main_fw v4rxg3"; reversed for pass b). TXY_VAL=1 (default) first runs the data-checked validation
#   (fabric_exp/torus/val_torus_xy_p150.yaml) for every router and shape, and checks the derived axes the control plane
#   logs ("axis topologies derived as": 2x4 Y=LINE X=RING, 4x2 Y=RING X=LINE) and ENABLE_DEADLOCK_AVOIDANCE in the
#   generated router args (all four directions); any mismatch or failure stops the job before the benchmark rows.
# Rows: fabric_exp/torus/torusxy_<shape>_sizes.yaml (8 rows x 5 sizes x write/scatter). Labels txy<PASS>_<router>_<shape>
# (.log, .csv), validation txyv<PASS>_<router>_<shape>. Mock-vetted first (torus_xy_mock.sh). A hang or failure stops.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
P=$1; shift
[ -n "$P" ] && [ $# -gt 0 ] || { echo "usage: [ROUTERS=...] [TXY_VAL=0] torus_xy_runs.sh PASS SHAPE..."; exit 1; }
V1="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
T=fabric_exp/torus
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
shape_env() { [ $1 = 4x2 ] && echo "TT_MESH_GRAPH_DESC_PATH=$PWD/$T/p150_x8_4x2_torus_xy_mgd.textproto"; }
want_axes() { case $1 in 2x4) echo "Y=LINE X=RING" ;; 4x2) echo "Y=RING X=LINE" ;; esac; }
one() {  # router label yaml cache shape
    local r=$1 l=$2 y=$3 c=$4 s=$5 rc
    case $r in
        main_fw) KA_ENV= run_cap $l $y $c 900 $V1 $(shape_env $s); rc=$? ;;
        *) python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop
           run_cap $l $y $c 900 $(shape_env $s); rc=$?
           python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; } ;;
    esac
    [ $rc = 0 ] || { echo "!!! $l: rc $rc"; stop; }
}
check_val() {  # label cache shape: derived axes and deadlock avoidance per direction
    local l=$1 c=$2 s=$3 axes
    axes=$(sed 's/\x1b\[[0-9;]*m//g' $out/$l.log | grep -aoE "axis topologies derived as Y=[A-Z]+ X=[A-Z]+" | sort -u | sed 's/.* as //')
    echo "   $l: axes '${axes:-not logged}' (want '$(want_axes $s)')"
    echo "axes=${axes:-not logged}" > $out/${l}_checks.txt   # read by report/scripts/build_torus_xy.py
    [ "$axes" = "$(want_axes $s)" ] || { echo "!!! $l: derived axes '$axes' != '$(want_axes $s)'"; stop; }
    python3 - $c $out/${l}_checks.txt <<'EOF' || stop
import collections, glob, re, sys
da = collections.defaultdict(collections.Counter)
for f in glob.glob(sys.argv[1] + "/*/kernels/*router*/*/named_ct_arg_map_generated.h"):
    s = open(f).read()
    d, e = re.search(r'\{"MY_DIRECTION",(\d+)\}', s), re.search(r'\{"ENABLE_DEADLOCK_AVOIDANCE",(\d+)\}', s)
    if d and e:
        da["EWNS"[int(d.group(1))]][e.group(1)] += 1
txt = ", ".join(f"{k}: {dict(v)}" for k, v in sorted(da.items()))
print("   deadlock avoidance by direction: " + txt)
open(sys.argv[2], "a").write(f"deadlock_avoidance={txt}\n")
sys.exit(0 if set(da) == set("EWNS") and all(set(v) == {"1"} for v in da.values()) else 1)
EOF
}
if [ "${TXY_VAL:-1}" = 1 ]; then
    for s in "$@"; do
        for r in ${ROUTERS:-main_fw v4rxg3}; do
            c=/home/aagarwal/.tt-metal-cache-txyv-$r-$s
            rm -rf "${c:?}"; mkdir -p $c
            echo "##### validation $s on $r (job at ${SECONDS} s)"
            one $r txyv${P}_${r}_$s $T/val_torus_xy_p150.yaml $c $s
            check_val txyv${P}_${r}_$s $c $s
        done
    done
fi
for s in "$@"; do
    y=$T/torusxy_${s}_sizes.yaml
    [ -f $y ] || { echo "no $y"; exit 1; }
    for r in ${ROUTERS:-main_fw v4rxg3}; do
        c=/home/aagarwal/.tt-metal-cache-txy-$r-$s
        mkdir -p $c
        echo "##### $s on $r (job at ${SECONDS} s)"
        one $r txy${P}_${r}_$s $y $c $s
    done
done
for s in "$@"; do for r in ${ROUTERS:-main_fw v4rxg3}; do
    awk -F, -v t="txy${P}_${r}_$s" 'NR > 1 && $7 == 4352 && $3 == "NOC_UNICAST_WRITE" {printf "%s: %s %.1f cycles/packet\n", t, $1, 1.35e9 / $10}' \
        $out/txy${P}_${r}_$s.csv
done; done
echo ALL_DONE
