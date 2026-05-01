#!/bin/bash
# Test 2: ReduceToOneB1 chain — 10 blocks (mirrors a 10-token decode of the
# 10-MoE-stage single-pod pipeline; exercises program cache + semaphore
# lifecycle across iterations).
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
