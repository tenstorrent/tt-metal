#!/bin/bash
# Test 2: ReduceToOneB1 chain — 10 blocks (mirrors a 10-token decode of the
# 10-MoE-stage single-pod pipeline; exercises program cache + semaphore
# lifecycle across iterations).

case "${1:-}" in
  -h|--help)
    cat <<EOF
Usage: $(basename "$0") [-h|--help]

Launches the 10-block ReduceToOneB1 chain test on the 16-rank single-pod
cluster (4 hosts × 4 ranks/host). Each rank does TEN back-to-back
reduce-to-one trees (8→1) on its own (4, 2) submesh — mirrors a 10-token
decode worth of MoE-end traffic. Exercises program cache + semaphore
lifecycle across iterations on top of what run_1block.sh covers.

Pytest target:
  test_fake_moe_traffic.py::test_fake_moe_chain_real_reduce_to_one_4x2_single_pod
    [10blocks-h7168-root_0_1-1link-4x2_grid-fabric_2d_torus_y]

Fabric / dispatch:
  FABRIC_2D_TORUS_Y, fabric_router_config(15232), fast dispatch
  worker_l1_size=1431568

Expected: PASSED=16/16. Wallclock ≈ 90-120 s.

Required environment:
  TT_METAL_HOME    Repo root. Default: /data/llong/tt-metal

Optional environment:
  SINGLE_POD_HOSTS Space- or comma-separated 4-host list. Default in _hosts.sh.
                   *** OVERRIDE THIS for a different cluster. ***

  PYTEST_TIMEOUT   Per-test timeout (seconds). Default: 240.

Pre-flight:
  Same as run_1block.sh — bundle is auto-bootstrapped on first launch
  (via bootstrap_pipeline_dir.sh). Reset chips first; run
  recover_hung_run.sh + reset_chips.sh after any hang.

Examples:
  bash $0
  SINGLE_POD_HOSTS="h1 h2 h3 h4" bash $0
  PYTEST_TIMEOUT=480 bash $0

See also:
  RUNBOOK.md (in the parent directory) for full prerequisites and troubleshooting.
EOF
    exit 0
    ;;
  "") ;;
  *)
    echo "[error] unexpected argument: $1" >&2
    echo "Run with --help for usage." >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cat <<'EOF'
[run] ─────────── ReduceToOneB1 — 10 blocks per rank ───────────
[run] each of the 16 ranks: 10 back-to-back ReduceToOneB1 calls on its own (4, 2) submesh
[run]                       (mirrors a 10-token decode worth of MoE-end traffic)
[run] fabric:  FABRIC_2D_TORUS_Y, fabric_router_config(15232), worker_l1_size=1431568
[run] expect:  PASSED=16/16
[run] ────────────────────────────────────────────────────────
EOF
TEST="test_fake_moe_chain_real_reduce_to_one_4x2_single_pod[10blocks-h7168-root_0_1-1link-4x2_grid-fabric_2d_torus_y]" \
  TEST_FILE="test_fake_moe_traffic.py" \
  EXTRA_ENV="" \
  PYTEST_TIMEOUT="240" \
  "$SCRIPT_DIR/_run_common.sh"
