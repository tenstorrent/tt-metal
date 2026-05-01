#!/bin/bash
# Test 1: ReduceToOneB1 chain — single block (1 reduce-to-one tree per rank).
# Per-token wire workload of the single-pod MoE pipeline.
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
