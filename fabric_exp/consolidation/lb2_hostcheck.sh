#!/bin/bash
# Second-LB harness (LB2_HARNESS.md): read-only host check of an extra P150x8 LB, run by lb2.sbatch (LB2_MODE=hostcheck)
# in the node-local tree. No reset, no kernels on the device:
#   1. topo_dump.py: the UMD cluster descriptor (opens the devices, reads ARC and Ethernet status) -> NODEDIR/cluster_desc.yaml;
#      chip count, PCIe map, links per chip pair, the VC1 split's two 2x2 groups
#   2. wiring against a09u32 (fabric_exp/a09u32_cluster_desc.yaml): the same (chip, channel) <-> (chip, channel) link set?
#   3. read_eth_speed.py: every Ethernet core's port / training status and speed -> NODEDIR/eth_baseline.txt (the
#      baseline lb2.sbatch's link check compares against); every connected channel UP / PASS / 400G; ports down
#   4. CPU-only mock runs on this node's descriptor (TT_METAL_MOCK_CLUSTER_DESC_PATH; no device): the torus rows
#      (pcs3/torusns, torusns_scatter) and the VC1 split (two ranks via ttrun.py with the study's rank bindings, both
#      vc1_alloc.yaml and vc1_inter.yaml), main's router; each must get past fabric setup
#   lb2_hostcheck.sh NODE JOBDIR      exit 0 only when 1-3 pass and the wiring is identical; mock results are reported
set -uo pipefail
NODE=$1; J=$2
NB=/data/aagarwal/lb2/nodes/$NODE
cd /home/aagarwal/tt-metal || exit 1
mkdir -p $NB
P=python_env/bin/python
rc=0
echo "=== 1. topology (UMD cluster descriptor)"
timeout 300 $P fabric_exp/consolidation/topo_dump.py $NB/cluster_desc.yaml 2>&1 | grep -v -E '^\s*$' | tail -30 || rc=1
cp $NB/cluster_desc.yaml $J/ 2>/dev/null
echo "=== 2. wiring vs bh-lb-120-a09u32"
$P - $NB/cluster_desc.yaml fabric_exp/a09u32_cluster_desc.yaml <<'EOF' || rc=1
import sys, yaml
def links(p):
    d = yaml.safe_load(open(p))
    s = set()
    for a, b in d["ethernet_connections"]:
        s.add(frozenset([(a["chip"], a["chan"]), (b["chip"], b["chan"])]))
    return d, s
dn, ln = links(sys.argv[1])
da, la = links(sys.argv[2])
print(f"this node: {len(dn['arch'])} chips {sorted(set(dn['arch'].values()))}, {len(ln)} links; a09u32: {len(da['arch'])} chips, {len(la)} links")
if ln == la:
    print("wiring: IDENTICAL to a09u32 at (chip, channel) level")
    sys.exit(0)
for l in sorted(map(sorted, ln - la)):
    print(f"   only here: {l}")
for l in sorted(map(sorted, la - ln)):
    print(f"   only on a09u32: {l}")
print("!!! wiring differs from a09u32")
sys.exit(1)
EOF
echo "=== 3. ethernet links (read_eth_speed.py)"
timeout 300 $P -I fabric_exp/read_eth_speed.py 8 > $J/eth_hostcheck.txt 2>&1 || { tail -5 $J/eth_hostcheck.txt; rc=1; }
$P - $J/eth_hostcheck.txt $NB/cluster_desc.yaml <<'EOF' || rc=1
import collections, re, sys, yaml
cores = collections.defaultdict(list)
for l in open(sys.argv[1]):
    m = re.match(r"chip (\d+) eth \(\s*(\d+),\s*(\d+)\): port_status=(\d+) train_status=(\d+) train_speed=(\d+)", l)
    if m:
        cores[int(m.group(1))].append((int(m.group(2)), int(m.group(3)), *map(int, m.group(4, 5, 6))))
d = yaml.safe_load(open(sys.argv[2]))
used = collections.defaultdict(set)
for a, b in d["ethernet_connections"]:
    used[a["chip"]].add(a["chan"]); used[b["chip"]].add(b["chan"])
n = sum(len(v) for v in cores.values())
up = [(c, x) for c, v in cores.items() for x in v if x[2] == 1]
good = [(c, x) for c, x in up if x[3] == 2 and x[4] == 400]
print(f"{len(cores)} chips, {n} Ethernet cores read; {len(up)} UP, {len(good)} UP / PASS / 400G")
print("speeds of UP cores: " + str(dict(collections.Counter(x[4] for c, x in up))))
bad = 0
for c in sorted(cores):
    for ch, x in enumerate(cores[c]):  # soc descriptor order = channel order
        if ch in used[c] and not (x[2] == 1 and x[3] == 2 and x[4] == 400):
            print(f"   !!! chip {c} chan {ch} eth ({x[0]},{x[1]}) is a mesh link but port {x[2]} train {x[3]} speed {x[4]}")
            bad += 1
        elif x[2] == 2:
            print(f"   port DOWN (not a mesh link): chip {c} chan {ch} eth ({x[0]},{x[1]})")
nl = sum(len(v) for v in used.values())
print(f"mesh link ends (cluster descriptor): {nl}; of them not UP / PASS / 400G: {bad}")
sys.exit(1 if bad or len(cores) != 8 else 0)
EOF
[ $rc = 0 ] && cp $J/eth_hostcheck.txt $NB/eth_baseline.txt && echo "baseline -> $NB/eth_baseline.txt"
echo "=== 4. mock runs on this node's descriptor (CPU only)"
MC=/home/aagarwal/.tt-metal-cache-lb2mock
rm -rf "${MC:?}"; mkdir -p $MC
for row in torusns torusns_scatter; do
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$PWD TT_METAL_MOCK_CLUSTER_DESC_PATH=$NB/cluster_desc.yaml \
        TT_METAL_CACHE=$MC TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0 TT_FABRIC_NO_BW_TELEMETRY=1 \
        timeout 600 ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $PWD/fabric_exp/pcs3/$row.yaml \
        > $J/mock_$row.log 2>&1
    r=$?
    echo "mock $row: exit $r; $(sed 's/\x1b\[[0-9;]*m//g' $J/mock_$row.log | grep -aE 'TT_FATAL|TT_THROW' | head -2 | cut -c1-200)"
done
printf 'rank_to_cluster_mock_cluster_desc:\n  0: "%s"\n  1: "%s"\n' $NB/cluster_desc.yaml $NB/cluster_desc.yaml > $NB/mock_mapping.yaml
for y in vc1_alloc vc1_inter; do
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$PWD TT_METAL_CACHE=$MC TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0 \
        TT_FABRIC_NO_BW_TELEMETRY=1 timeout 600 $P ttnn/ttnn/distributed/ttrun.py --mock-cluster-rank-binding $NB/mock_mapping.yaml \
        --rank-binding fabric_exp/vc1/p150x8_2x2_split_rank_bindings.yaml \
        ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $PWD/fabric_exp/vc1/$y.yaml > $J/mock_$y.log 2>&1
    r=$?
    echo "mock VC1 $y (2 ranks, no --oversubscribe): exit $r; $(sed 's/\x1b\[[0-9;]*m//g' $J/mock_$y.log | grep -acE 'Running Test') test starts;" \
        "$(sed 's/\x1b\[[0-9;]*m//g' $J/mock_$y.log | grep -aE 'TT_FATAL|TT_THROW|not enough slots|placement' | head -2 | cut -c1-200)"
done
rm -rf "${MC:?}"
echo "host check rc $rc"
exit $rc
