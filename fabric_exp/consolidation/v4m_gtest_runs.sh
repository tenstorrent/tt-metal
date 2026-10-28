#!/bin/bash
# fabric_unit_tests on v4m (variants/v4): the 1D sparse-multicast gtests (no suite we run sends sparse multicast), then
# every Fabric1DFixture test. If a sparse-multicast test fails on v4m, the same tests on the tree's own router (v1),
# to tell a v4 bug from a test or host problem. Same router settings as the suites (jobs_lib.sh run).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
GT=build/test/tt_metal/tt_fabric/fabric_unit_tests
gtest() {  # label kernel-version cache filter
    echo "=== $1: $4"
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=$2 TT_FABRIC_NO_BW_TELEMETRY=1 \
        TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 TT_METAL_CACHE=$3 TT_METAL_FABRIC_OPT_LEVEL=O3 \
        timeout 600 $GT --gtest_filter="$4" > fabric_exp/results/$1.log 2>&1
    local rc=$?
    grep -aE '^\[ +(OK|FAILED|PASSED|SKIPPED) +\]|tests? ran|FAILED TEST' fabric_exp/results/$1.log | sed 's/^/   /'
    echo "$1: exit $rc"
    return $rc
}
C=/home/aagarwal/.tt-metal-cache-v4m-gtest
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply $V/v4 $BK || { echo STOPPED; exit 1; }
gtest v4m_gtest_sparse 2 $C 'Fabric1DFixture.TestSparseMcastPerPage*'; SP=$?
gtest v4m_gtest_1d 2 $C 'Fabric1DFixture.*'
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
if [ $SP -ne 0 ]; then
    C=/home/aagarwal/.tt-metal-cache-v1-gtest
    rm -rf "${C:?}"; mkdir -p $C
    gtest v1_gtest_sparse 1 $C 'Fabric1DFixture.TestSparseMcastPerPage*'
fi
rm -f $M
