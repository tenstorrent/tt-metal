#!/usr/bin/env bash
# Full D2HStreamService validation: C++ gtests, cross-process MPI, Python pytest.
# Run on a Tenstorrent device node (e.g. srun on a Galaxy allocation).
#
# Usage (repo root):
#   srun --jobid=<JOB> -w <NODE> bash -lc 'cd /data/bzhang/tt-metal-tensor-socket && ./scripts/run_d2h_stream_service_all_tests.sh'

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

reset_device() {
  if command -v tt-smi >/dev/null 2>&1; then
    echo "=== tt-smi -r ==="
    tt-smi -r
  else
    echo "WARNING: tt-smi not found; skipping reset" >&2
  fi
}

run_phase() {
  echo ""
  echo "=== $1 ==="
  shift
  "$@"
}

reset_device

run_phase "C++ D2HStreamServiceTest" \
  ./build_Release/test/ttnn/unit_tests_ttnn_tensor --gtest_filter='D2HStreamServiceTest.*'

reset_device

run_phase "C++ cross-process (MPI)" \
  mpirun --oversubscribe -np 2 \
  ./build_Release/test/tt_metal/distributed/cross_process_d2h_stream_service_test

reset_device

run_phase "Python pytest" \
  ./scripts/run_d2h_stream_service_pytest.sh -q

echo ""
echo "All D2H stream service tests passed."
