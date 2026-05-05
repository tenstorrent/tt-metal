#!/bin/bash
# Test 1: ReduceToOneB1 chain — single block (1 reduce-to-one tree per rank).
# Per-token wire workload of the single-pod MoE pipeline.

case "${1:-}" in
  -h|--help)
    cat <<EOF
Usage: $(basename "$0") [-h|--help]

Launches the single-block ReduceToOneB1 chain test on the 16-rank single-pod
cluster (4 hosts × 4 ranks/host). Each rank does ONE 3-level reduce-to-one
tree (8→1) on its own (4, 2) submesh.

Pytest target:
  test_fake_moe_traffic.py::test_fake_moe_chain_real_reduce_to_one_4x2_single_pod
    [1block-h7168-root_0_1-1link-4x2_grid-fabric_2d_torus_y]

Fabric / dispatch:
  FABRIC_2D_TORUS_Y, fabric_router_config(15232), fast dispatch
  worker_l1_size=1431568

Expected: PASSED=16/16. Wallclock ≈ 60-90 s.

Required environment:
  TT_METAL_HOME    Repo root. Default: /data/llong/tt-metal

Optional environment:
  SINGLE_POD_HOSTS Space- or comma-separated 4-host list. Default in _hosts.sh:
                     bh-glx-110-c07u02 bh-glx-110-c07u08
                     bh-glx-110-c08u02 bh-glx-110-c08u08
                   *** OVERRIDE THIS for a different cluster. ***

  PYTEST_TIMEOUT   Per-test timeout (seconds). Default: 240.

Pre-flight:
  - The runner auto-bootstraps the pipeline-config bundle on first launch
    (via bootstrap_pipeline_dir.sh — wraps per-host PCIe device discovery
    and rank-binding generation). No manual generator step required.
  - Reset chips before the first run and after any hang:
      ./reset_chips.sh

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
[run] ─────────── ReduceToOneB1 — 1 block per rank ───────────
[run] each of the 16 ranks: 1 ReduceToOneB1 (3-level tree, 8→1) on its own (4, 2) submesh
[run] fabric:  FABRIC_2D_TORUS_Y, fabric_router_config(15232), worker_l1_size=1431568
[run] expect:  PASSED=16/16
[run] ──────────────────────────────────────────────────────
EOF
TEST="test_fake_moe_chain_real_reduce_to_one_4x2_single_pod[1block-h7168-root_0_1-1link-4x2_grid-fabric_2d_torus_y]" \
  TEST_FILE="test_fake_moe_traffic.py" \
  EXTRA_ENV="" \
  PYTEST_TIMEOUT="240" \
  "$SCRIPT_DIR/_run_common.sh"
