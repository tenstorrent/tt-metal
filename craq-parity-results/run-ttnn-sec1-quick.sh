#!/usr/bin/env bash
set -euo pipefail
REPO="${TT_METAL_HOME:-${SLURM_SUBMIT_DIR:-/data/rsong/tt-metal2}}"
CRAQ="${CRAQ_SIM:-/data/rsong/craq-sim}"
BUILD="$REPO/build_Debug"
RESULTS="$REPO/craq-parity-results/ttnn-sec1-quick-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RESULTS"
SIM="$RESULTS/sim"
mkdir -p "$SIM"
cp "$CRAQ/src/_out/release_bh/libttsim.so" "$SIM/"
cp "$REPO/tt_metal/soc_descriptors/blackhole_140_arch.yaml" "$SIM/soc_descriptor.yaml"
ln -sfn build_Debug "$REPO/build"
export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" ARCH_NAME=blackhole
export TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DISABLE_SFPLOADMACRO=1 TT_METAL_DRAM_BACKED_CQ=1
export TT_METAL_SIMULATOR_CQ_WAIT_CLOCKS=10000
export TT_METAL_SIMULATOR="$SIM/libttsim.so" TT_METAL_SIMULATOR_HOME="$SIM"
# Galaxy nodes may inherit WH mock cluster desc from other jobs; section-1 single-card tests must not use it.
unset TT_METAL_MOCK_CLUSTER_DESC_PATH
export LD_LIBRARY_PATH="$BUILD/tt_metal:$BUILD/tt_metal/third_party/umd/lib:$BUILD/ttnn:$BUILD/tt_stl:$BUILD/lib"
cd "$REPO"
echo "host=$(hostname -s)" | tee "$RESULTS/env.txt"
env | grep -E 'TT_METAL|ARCH_NAME|LD_LIBRARY' | sort | tee -a "$RESULTS/env.txt"

echo "=== MCQ filter (expect skip, no segfault) ===" | tee "$RESULTS/mcq.log"
./build/test/ttnn/unit_tests_ttnn --gtest_filter='MultiCommandQueueSingleDeviceFixture.*' 2>&1 | tee -a "$RESULTS/mcq.log"
rc1=${PIPESTATUS[0]}
echo "mcq exit=$rc1" | tee -a "$RESULTS/mcq.log"

echo "=== RegionWriteRead filter ===" | tee "$RESULTS/region.log"
./build/test/ttnn/unit_tests_ttnn_tensor --gtest_filter='*RegionWriteReadTest*' 2>&1 | tee -a "$RESULTS/region.log"
rc2=${PIPESTATUS[0]}
grep -cE '\[  FAILED  \].*RegionWriteReadTest' "$RESULTS/region.log" | tee -a "$RESULTS/summary.txt" || true
grep -cE '\[       OK \].*RegionWriteReadTest' "$RESULTS/region.log" | tee -a "$RESULTS/summary.txt" || true
echo "region exit=$rc2" | tee -a "$RESULTS/summary.txt"
