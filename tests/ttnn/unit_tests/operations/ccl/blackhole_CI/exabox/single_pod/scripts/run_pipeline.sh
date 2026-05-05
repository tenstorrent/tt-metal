#!/bin/bash
# Test 3: Pipeline framework smoke (fake-MoE substitution).
# Slow dispatch — required by the blitz_decode pipeline framework.

case "${1:-}" in
  -h|--help)
    cat <<EOF
Usage: $(basename "$0") [-h|--help]

Launches the single-pod pipeline framework smoke test on the 16-rank
cluster. Each of the 16 ranks acts as one stage of the Blitz pipeline:

  rank  0       EmbeddingStage             (real, with synthetic embedding)
  ranks 1-3     PassthroughStage(ACT)      (Dense slots — stubbed)
  ranks 4-13    PassthroughStage(ACT)      (MoE slots — fake-MoE)
  rank 14       FakeLMHeadStage            (no-compute LMHead stub)
  rank 15       PassthroughStage(TOKEN)

This validates:
  - sockets, fabric routing
  - tt-run, mesh-graph descriptor
  - slow-dispatch reachability
  - kernel scaffolding + FIFO sizing
  *NOT* MoE/LMHead numerics — those are stubbed out so the framework can
  run end-to-end without the broken MoE op synthetic-weight path.

Pytest target:
  test_single_pod_pipeline_fake_moe.py::test_single_pod_pipeline_fake_moe

Fabric / dispatch:
  FABRIC_2D_TORUS_Y, fabric_router_config(15232), worker_l1_size=1431568
  *** SLOW DISPATCH *** (TT_METAL_SLOW_DISPATCH_MODE=1) — required by the
  pipeline framework (see _vendored/pipeline.py). Cannot share a process
  with sub-device CCL ops like ttnn.broadcast / all_to_all_*.

Expected: PASSED=16/16. Wallclock ≈ 5-10 minutes (slow dispatch + pipeline init).

Required environment:
  TT_METAL_HOME    Repo root. Default: /data/llong/tt-metal

Optional environment:
  SINGLE_POD_HOSTS Space- or comma-separated 4-host list. Default in _hosts.sh.
                   *** OVERRIDE THIS for a different cluster. ***

  PYTEST_TIMEOUT   Per-test timeout (seconds). Default: 600.

Pre-flight:
  - Auto-bootstraps the pipeline-config bundle on first launch
    (via bootstrap_pipeline_dir.sh — wraps per-host PCIe device discovery
    and rank-binding generation). No manual generator step required.
  - Reset chips first; this test is the most sensitive to stale device state.

Examples:
  bash $0
  SINGLE_POD_HOSTS="h1 h2 h3 h4" bash $0
  PYTEST_TIMEOUT=1200 bash $0

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
