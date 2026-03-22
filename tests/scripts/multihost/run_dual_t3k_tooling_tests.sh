#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Dedicated test suite for tooling and MPI infrastructure tests on dual T3K.
# This runs:
#   1. ttrun env passthrough multihost pytest (validates tt-run launch infrastructure)
#   2. ULFM fault tolerance tests (MPI fault detection / recovery control plane)
#   3. Single-node ULFM gap tests (exercise fast-fail, watchdog, terminate handler)
#   4. Python unit tests: ttrun exit code interpretation
#   5. Python unit tests: mpi_fault.py failure paths
#   6. Triage tool unit tests

set -eo pipefail

# Exit immediately if ARCH_NAME is not set or empty
if [ -z "${ARCH_NAME}" ]; then
  echo "Error: ARCH_NAME is not set. Exiting." >&2
  exit 1
fi

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

cd "$TT_METAL_HOME"
export PYTHONPATH="$TT_METAL_HOME"

fail=0
start_time=$(date +%s)

echo "LOG_METAL: Running run_dual_t3k_tooling_tests"

# Repo root for absolute paths / JUnit
repo_root="$(pwd)"
mkdir -p "${repo_root}/generated/test_reports"

# ── 1. ttrun env passthrough multihost pytest ──────────────────────────
#
# Run from /tmp with PYTHONPATH unset so repo-root ``ttnn/`` does not shadow
# the installed package.  --confcutdir keeps ``tests/ttnn/conftest.py``
# without pulling in the repo-root conftest.
echo "LOG_METAL: Running ttrun env passthrough multihost pytest (import-isolated)"
(cd /tmp && env -u PYTHONPATH pytest --override-ini "addopts=--import-mode=importlib -vv -rA --durations=0" \
  --confcutdir="${repo_root}/tests/ttnn" \
  --junitxml="${repo_root}/generated/test_reports/most_recent_tests_ttrun_env_passthrough_tooling.xml" \
  -m multihost \
  "${repo_root}/tests/ttnn/distributed/test_ttrun_env_passthrough.py") ; fail=$((fail + $?))

# ── 2. ULFM fault tolerance tests ─────────────────────────────────────
echo "LOG_METAL: Running ULFM fault tolerance tests"
"${repo_root}/tests/tt_metal/multihost/run_fault_tolerance_tests.sh" ; fail=$((fail + $?))

# ── 3. Single-node ULFM gap tests (section 7.8) ───────────────────────
#
# These test the ULFM control-plane paths that don't require actual
# multi-host hardware: fast-fail exit code 70, MPI_Finalize watchdog,
# std::set_terminate handler, and agree() consensus.
echo "LOG_METAL: Running single-node ULFM gap tests"
"${repo_root}/tests/tt_metal/multihost/run_single_node_ulfm_tests.sh" ; fail=$((fail + $?))

# ── 4. Python unit tests: ttrun exit code interpretation ──────────────
#
# Pure Python tests (no MPI runtime needed) that verify ExitCategory,
# interpret_exit_code(), PRRTE version detection, and log output quality.
echo "LOG_METAL: Running ttrun exit code interpretation tests"
(cd /tmp && env -u PYTHONPATH python3 -m pytest \
  --override-ini "addopts=--import-mode=importlib -vv -rA --durations=0" \
  --confcutdir="${repo_root}/tests/ttnn" \
  --junitxml="${repo_root}/generated/test_reports/test_ttrun_exit_codes.xml" \
  "${repo_root}/tests/ttnn/distributed/test_ttrun_exit_codes.py") ; fail=$((fail + $?))

# ── 5. Python unit tests: mpi_fault.py failure paths ─────────────────
#
# Tests the Python ULFM wrapper (install_ulfm_handler, ulfm_guard,
# MPIRankFailureError) with mocked mpi4py — no real MPI runtime needed.
echo "LOG_METAL: Running mpi_fault.py Python tests"
(cd /tmp && env -u PYTHONPATH python3 -m pytest \
  --override-ini "addopts=--import-mode=importlib -vv -rA --durations=0" \
  --confcutdir="${repo_root}/tests/ttnn" \
  --junitxml="${repo_root}/generated/test_reports/test_mpi_fault_python.xml" \
  "${repo_root}/tests/ttnn/distributed/test_mpi_fault_python.py") ; fail=$((fail + $?))

# ── 6. Triage tool unit tests ─────────────────────────────────────────
#
# test_parse_inspector_logs_paths.py: pure Python, no hardware needed.
# test_triage.py: requires ttexalens + real hardware; failures are
#   reported but do not block the tooling suite (ttexalens may not be
#   installed in all environments).
echo "LOG_METAL: Running triage unit tests (parse_inspector_logs_paths)"
(cd /tmp && env -u PYTHONPATH python3 -m pytest \
  --override-ini "addopts=--import-mode=importlib -vv -rA --durations=0" \
  --confcutdir="${repo_root}/tools/tests/triage" \
  --junitxml="${repo_root}/generated/test_reports/test_parse_inspector_logs_paths.xml" \
  "${repo_root}/tools/tests/triage/test_parse_inspector_logs_paths.py") ; fail=$((fail + $?))

echo "LOG_METAL: Running triage integration tests (test_triage, requires ttexalens + inspector)"
# test_triage.py is a full integration test: it starts a hang application,
# connects to the live Inspector RPC or reads from generated/inspector/, and
# exercises the triage tool against real hardware.  This requires:
#   - A compiled hang-app binary in build/
#   - TT_METAL_INSPECTOR=1 and the Inspector RPC running
#   - ttexalens installed with its full C extension stack
# These conditions are not met in every CI environment, so failures here are
# treated as warnings and do NOT block the tooling suite.
(cd /tmp && env -u PYTHONPATH python3 -m pytest \
  --override-ini "addopts=--import-mode=importlib -vv -rA --durations=0" \
  --confcutdir="${repo_root}/tools/tests/triage" \
  --junitxml="${repo_root}/generated/test_reports/test_triage.xml" \
  "${repo_root}/tools/tests/triage/test_triage.py") ; triage_exit=$?
if [[ $triage_exit -ne 0 ]]; then
  echo "LOG_METAL: WARNING: test_triage.py exited $triage_exit (needs inspector + hang-app binary; non-blocking)"
fi

# ── Done ───────────────────────────────────────────────────────────────
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "LOG_METAL: run_dual_t3k_tooling_tests $duration seconds to complete"

if [[ $fail -ne 0 ]]; then
  exit 1
fi
