#!/bin/bash
# mock_pair.sh NAME MGD [FSD]: CPU-only control-plane check of MGD on pair NAME (mk_pair.sh output): tt-run in MGD mode
# with the pair's mock cluster descriptors and the FSD (default: the published exabox FSD, which the control plane
# filters to the pair's hosts), fabric_unit_tests' FactoryDescriptorControlPlaneFixture (RELAXED health, FSD ingested):
# the descriptor matches the hardware with no downed links, and every mapped node resolves to a real chip.
S=${PAIRS_WORK:-/home/aagarwal/tt-metal/fabric_exp/galaxy/pairs/work}  # pair dirs and mock outputs
T=/home/aagarwal/tt-metal-mainline
n=$1; mgd=$2; fsd=${3:-/data/scaleout_configs/tt-cluster-config-sources/exabox-latest-main/factory_system_descriptor.textproto}
d=$S/pair_$n; tag=$(basename $mgd .textproto); out=$d/mock_${tag}_$(basename $(dirname $fsd))
mkdir -p $out
for v in $(env | grep -oE '^TT_[A-Za-z0-9_]+'); do unset $v; done
cd $T
env TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_HOME=$T TT_METAL_CACHE=$out/cache TT_METAL_LOGS_PATH=$out \
    timeout ${MOCK_TIMEOUT:-900} taskset -c 48-63 /home/aagarwal/tt-metal/python_env/bin/python $T/ttnn/ttnn/distributed/ttrun.py \
    --mesh-graph-descriptor $mgd --mock-cluster-rank-binding $d/cd_mapping.yaml --factory-system-descriptor $fsd \
    --mpi-args "--oversubscribe" ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
    --gtest_filter="FactoryDescriptorControlPlaneFixture.AMatchingDescriptorReportsNoDownedLinks:FactoryDescriptorControlPlaneFixture.EveryMappedNodeResolvesToARealChip" \
    > $out/log.txt 2>&1
rc=$?
res=$(sed 's/\x1b\[[0-9;]*m//g' $out/log.txt | grep -aoE '\[  (PASSED|FAILED|SKIPPED) +\] [0-9]+ test' | sort | uniq -c | tr '\n' ' ')
fat=$(sed 's/\x1b\[[0-9;]*m//g' $out/log.txt | grep -aoE '(TT_FATAL|TT_THROW)[^|]{0,200}' | head -1)
echo "$n $tag: rc $rc; $res${fat:+| first fatal: $fat}"
rm -rf $out/cache
