#!/usr/bin/env bash
#
# Run the full emule ASAN sanitizer test suite (all check families) in one shot.
#
# Usage:
#   ./run_sanitizer_tests.sh                 # run every sanitizer test
#   ./run_sanitizer_tests.sh --gtest_filter="MeshDeviceFixture.CB_Boundary_*"   # override / narrow
#   TT_METAL_DIR=/path/to/tt-metal ./run_sanitizer_tests.sh                     # point at another checkout
#
# Any extra args are forwarded verbatim to unit_tests_api (a later --gtest_filter
# overrides the default below).
set -euo pipefail

TT_METAL_DIR="${TT_METAL_DIR:-/localdev/smeydanshahi/tt-metal}"
BUILD_DIR="${BUILD_DIR:-$TT_METAL_DIR/build_emule}"
BIN="$BUILD_DIR/test/tt_metal/unit_tests_api"

if [[ ! -x "$BIN" ]]; then
    echo "error: $BIN not found — build it first:" >&2
    echo "  cmake --build $BUILD_DIR -j\$(nproc) --target unit_tests_api" >&2
    exit 1
fi

# Emule device env (MeshDeviceFixture GTEST_SKIPs the whole suite without these).
export TT_METAL_HOME="$TT_METAL_DIR"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml"
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1

# One family glob per sanitizer test file. Keep in sync with the per-file "To run"
# headers and run_regression_wormhole.sh Tier 3a.
FILTER="MeshDeviceFixture.Host_UAF_*"          # Use-After-Free        (test_tensor_bad_acess.cpp)
FILTER+=":MeshDeviceFixture.Host_Alignment_*"  # Host L1/DRAM align    (test_host_alignment.cpp)
FILTER+=":MeshDeviceFixture.Metadata_*"        # Metadata Overflow     (test_metadata_size.cpp)
FILTER+=":MeshDeviceFixture.OOB_Tensor_*"      # OOB Write             (test_write_outside_tensor.cpp)
FILTER+=":MeshDeviceFixture.Tensor_Padding_*"  # Tensor Padding        (test_padded_write.cpp)
FILTER+=":MeshDeviceFixture.Semaphore_*"       # Illegal Semaphore     (test_semaphore_write.cpp)
FILTER+=":MeshDeviceFixture.CB_Boundary_*"     # CB Boundary           (test_write_beyond_res_pages.cpp)
FILTER+=":MeshDeviceFixture.CB_Reservation_*"  # CB Reservation        (test_cb_pages.cpp)
FILTER+=":MeshDeviceFixture.NoC_Barrier_*"     # NoC pending on pop    (test_noc_without_barrier.cpp)
FILTER+=":MeshDeviceFixture.Noc*"              # NOC Transfer Align    (test_alignment_writes.cpp)
FILTER+=":MeshDeviceFixture.Dirty_CB_*"        # Dirty CB              (test_cb_leak.cpp)
FILTER+=":MeshDeviceFixture.Object_Intent_*"   # Object Intent         (test_valid_mem_wrong_alloc.cpp)

echo "Running emule sanitizer suite: $BIN"
exec "$BIN" --gtest_filter="$FILTER" "$@"
