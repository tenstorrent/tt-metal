#!/bin/bash
# Check this galaxy's ethernet links against the BH galaxy cabling spec before a multi-stage run.
# usage: link_check.sh [--retrain]
#   default    read-only: report missing channels, change nothing
#   --retrain  let run_cluster_validation retrain missing links (up to 5 rounds) and report what is left
# Exit 0 when no channel is missing, 1 otherwise. run_cluster_validation itself exits 0 with missing
# links unless --hard-fail is set, and --hard-fail aborts before retraining, so this script parses its
# output instead. A link that trains with one of its two channels hangs multi-stage runs intermittently
# (see the TORUS_X investigation in K3_PREFILL_4X4_SINGLE_GALAXY_PLAN.md): treat any missing channel as
# a bad reset. tt-smi -r can leave channels untrained; tt-smi -glx_reset (needs IPMI access) usually
# brings them back.
REPO=$(cd "$(dirname "$0")/../../../../.." && pwd)
cd "$REPO"
export TT_METAL_HOME=$PWD LD_LIBRARY_PATH=$PWD/build/lib
OUT=${LINK_CHECK_OUT:-/tmp/link_check_$USER}
mkdir -p "$OUT"
DEPLOY=$OUT/deployment.textproto
cat > "$DEPLOY" <<DEP
hosts: {
  hall: "lab"
  aisle: "A"
  rack: 1
  shelf_u: 1
  node_type: "BH_GALAXY_XY_TORUS"
  host: "$(hostname)"
}
DEP
RETRAIN=--skip-retrain; [ "${1:-}" = "--retrain" ] && RETRAIN=""
LOG=$OUT/validation.log
timeout 1500 flock -w 300 /tmp/tt-device.lock ./build/tools/scaleout/run_cluster_validation $RETRAIN \
  --cabling-descriptor-path tools/tests/scaleout/cabling_descriptors/bh_galaxy_xy_torus.textproto \
  --deployment-descriptor-path "$DEPLOY" --output-path "$OUT/report" > "$LOG" 2>&1
rc=$?
# Net missing against extra channels per ASIC pair: some galaxies carry a link on different channel ids
# (and QSFP ports) than the reference cabling spec, which shows up as N missing plus N extra channels on
# the same ASIC pair. Only a pair with more missing than extra channels is actually degraded.
python3 - "$LOG" "$rc" <<'PY'
import re, sys, collections
text = re.sub(r"\x1b\[[0-9;]*m", "", open(sys.argv[1]).read())
rc = int(sys.argv[2])
blocks = re.findall(r"found (\d+) (missing|extra) channel connections:\n((?:\s+- .*\n)*)", text)
if not blocks:
    print(f"run_cluster_validation produced no connection report (rc={rc}); see {sys.argv[1]}")
    sys.exit(0 if rc == 0 and "All Detected Links are healthy" in text else 1)
ep = re.compile(r"tray_id=(\d+), asic_channel=AsicChannel\{asic_location=(\d+), channel_id=(\d+)\}")
net = collections.Counter(); chans = collections.defaultdict(lambda: {"missing": [], "extra": []})
# A retrain run prints the lists once per round; the last missing and last extra lists are the end state.
last = {}
for _, kind, body in blocks:
    last[kind] = body
for kind, body in last.items():
    for line in body.strip().splitlines():
        a, b = ep.findall(line)[:2]
        pair = tuple(sorted([(int(a[0]), int(a[1])), (int(b[0]), int(b[1]))]))
        net[pair] += 1 if kind == "missing" else -1
        chans[pair][kind].append(int(a[2]))
bad = {p: n for p, n in net.items() if n > 0}
rewired = [p for p in chans if chans[p]["missing"] and net[p] <= 0]
for p in sorted(rewired):
    (t1, a1), (t2, a2) = p
    print(f"  ok (rewired)  tray {t1} asic {a1} <-> tray {t2} asic {a2}: spec ch {chans[p]['missing']}, live ch {chans[p]['extra']}")
if not bad:
    print("LINKS OK: no degraded ASIC pairs"); sys.exit(0)
for p, n in sorted(bad.items()):
    (t1, a1), (t2, a2) = p
    print(f"  DEGRADED      tray {t1} asic {a1} <-> tray {t2} asic {a2}: {n} channel(s) missing {chans[p]['missing']}")
print(f"LINKS DEGRADED: {len(bad)} ASIC pair(s); reset (tt-smi -r, or tt-smi -glx_reset) and check again")
sys.exit(1)
PY
