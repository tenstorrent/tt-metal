#!/bin/bash
# Test 3: Pipeline framework smoke (fake-MoE substitution).
# Slow dispatch — required by the blitz_decode pipeline framework.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cat <<'EOF'
[run] ─────────── single-pod pipeline framework smoke (fake-MoE) ───────────
[run] rank → pipeline stage:
[run]    rank  0       EmbeddingStage             (real, with synthetic embedding tensor)
[run]    ranks 1-3     PassthroughStage(ACT)      (Dense slots — stubbed)
[run]    ranks 4-13    PassthroughStage(ACT)      (MoE slots — stubbed via FakeMoeDecoder)
[run]    rank 14       FakeLMHeadStage            (no-compute LMHead stub)
[run]    rank 15       PassthroughStage(TOKEN)
[run] fabric:  FABRIC_2D_TORUS_Y, fabric_router_config(15232), worker_l1_size=1431568
[run] expect:  PASSED=16/16  (sockets, fabric, mesh-graph, slow dispatch reachability)
[run] ──────────────────────────────────────────────────────────────────
EOF
TEST="test_single_pod_pipeline_fake_moe" \
  TEST_FILE="test_single_pod_pipeline_fake_moe.py" \
  EXTRA_ENV="TT_METAL_SLOW_DISPATCH_MODE=1" \
  PYTEST_TIMEOUT="600" \
  "$SCRIPT_DIR/_run_common.sh"
