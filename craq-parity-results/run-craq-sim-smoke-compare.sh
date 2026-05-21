#!/usr/bin/env bash
#SBATCH --job-name=craq-smoke
#SBATCH --partition=bh_sc5_B2B9_D12
#SBATCH --nodelist=bh-glx-b06u08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=/data/rsong/tt-metal2/craq-parity-results/slurm-craq-smoke-%j.out
#SBATCH --error=/data/rsong/tt-metal2/craq-parity-results/slurm-craq-smoke-%j.err

set -euo pipefail

REPO=/data/rsong/tt-metal2
CRAQ=/data/rsong/craq-sim
BUILD="$REPO/build_Debug"
OUT="$REPO/craq-parity-results/craq-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
SIM="$OUT/sim_wh"
mkdir -p "$SIM" "$OUT"

# shellcheck source=scripts/lib/require-bh-glx-compute.sh
source "$REPO/scripts/lib/require-bh-glx-compute.sh"
require_bh_glx_compute

cp "$CRAQ/src/_out/release_wh/libttsim.so" "$SIM/"
cp "$REPO/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml" "$SIM/soc_descriptor.yaml"
ln -sfn build_Debug "$REPO/build"

export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" ARCH_NAME=wormhole_b0
export TT_METAL_DISABLE_SFPLOADMACRO=1 TT_METAL_DRAM_BACKED_CQ=1
export TT_METAL_SIMULATOR_CQ_WAIT_CLOCKS=10000
export TT_METAL_SIMULATOR="$SIM/libttsim.so" TT_METAL_SIMULATOR_HOME="$SIM"
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$REPO/tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml"
export LD_LIBRARY_PATH="$BUILD/tt_metal:$BUILD/tt_metal/third_party/umd/lib:$BUILD/ttnn:$BUILD/tt_stl:$BUILD/lib"

T3K_MOCK="$REPO/tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml"
WH_ENV="ARCH_NAME=wormhole_b0 TT_METAL_SIMULATOR=$SIM/libttsim.so TT_METAL_SIMULATOR_HOME=$SIM TT_METAL_MOCK_CLUSTER_DESC_PATH=$T3K_MOCK TT_METAL_DRAM_BACKED_CQ=1 TT_METAL_SIMULATOR_CQ_WAIT_CLOCKS=10000"
MOCK_ENV="env -u TT_METAL_SIMULATOR -u TT_METAL_SIMULATOR_HOME ARCH_NAME=wormhole_b0 TT_METAL_MOCK_CLUSTER_DESC_PATH=$T3K_MOCK TT_METAL_SLOW_DISPATCH_MODE=1"

{
  echo "craq-sim HEAD: $(git -C "$CRAQ" log -1 --oneline)"
  echo "libttsim sha256: $(sha256sum "$SIM/libttsim.so" | awk '{print $1}')"
  echo "eth_io local EINTR: $(grep -c EINTR "$CRAQ/src/eth_io.cpp" || true)"
  echo "started: $(date -u -Iseconds)"
} | tee "$OUT/manifest.txt"

run_one() {
  local name="$1"
  shift
  local log="$OUT/${name}.log"
  echo "=== RUN $name ===" | tee -a "$OUT/summary.txt"
  set +e
  local start end dur rc
  start=$(date +%s)
  bash -lc "cd '$REPO' && $*" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e
  end=$(date +%s)
  dur=$((end - start))
  echo "$name rc=$rc duration=${dur}s" | tee -a "$OUT/summary.txt"
  echo -e "${name}\t${rc}\t${dur}" >>"$OUT/results.tsv"
}

: >"$OUT/results.tsv"
echo -e "test\trc\tduration_sec" >>"$OUT/results.tsv"

cd "$REPO"

run_one "fabric_control_T3k" \
  "$MOCK_ENV ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter='ControlPlaneFixture.*T3k*'"

run_one "eth_direct_send" \
  "$WH_ENV TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth --gtest_filter='MeshDeviceFixture.ActiveEthKernelsDirectSendAllConnectedChips'"

run_one "fabric_worker_edm" \
  "$WH_ENV ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter='*WorkerFabricEdmDatapath*:*EdmFabric*'"

run_one "fabric_Fabric2D_unicast" \
  "$WH_ENV TT_METAL_SLOW_DISPATCH_MODE=1 TT_MESH_GRAPH_DESC_PATH=tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_1x8_mesh_graph_descriptor.textproto ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter='*Fabric2DFixture.TestUnicast*'"

echo "finished $(date -u -Iseconds)" | tee -a "$OUT/summary.txt"
echo "results: $OUT"
