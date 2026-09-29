#!/usr/bin/env bash
# P2 goodput (CPU only): 16x[2,4] (our effs) vs 16x[4,2] with the torus-carve effs (p2t2 = v2 dispatch + v2 combine,
# p2t3 = v2 dispatch only) and a "mixed" set (2 of 4 [4,2] stages per galaxy on the torus). Outputs goodput/<name>_w<W>.json.
#   NODE=... bash tools/goodput_p2.sh
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd)
R=$(dirname "$T")
O=$R/goodput
NODE=${NODE:-$(ls ~/.vscode-server/cli/servers/*/server/node | head -1)}
W_=${WORKERS:-16}
ringc() { python3 -c "import json;print(round(0.30248237322064786*json.load(open('$T/cal_effs_ours_2x4_w$1.json'))['2x4']['dense']['ring_c']/0.3575944448870825,6))"; }
run() { local n=$1 W=$2; shift 2; echo "== $n W=$W $*"
  nice -n 10 "$NODE" "$T/run_goodput.js" --budget "$W" --slo 10,3 --workers "$W_" --out "$O/$n.json" "$@" | tail -n +2; }
for W in 4096 8192; do
  P="{\"ringC\":$(ringc $W)}"
  B=$T/cal_effs_ours_2x4_w$W.json
  for v in ${VARIANTS:-p2t2 p2t2_native p2mix p2mix_native} ${EXTRA:-}; do
    f=$T/cal_effs_ours_4x2_${v}_w$W.json
    [ -f "$f" ] || { echo "skip $v (no $f)"; continue; }
    run ${v}_w$W $W --topo 2x4,4x2 --features near,full --effs "$B,$f" --pipe "$P"
  done
done
