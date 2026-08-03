#!/usr/bin/env bash
# Trace-replay device-time perf for high_bw_all_gather, launched as two MPI ranks under tt-run.
#
#   rank 0 = RANK0_HOST (mesh 0): traced all_gather loop -> D2D send
#   rank 1 = RANK1_HOST (mesh 1): D2D recv -> traced all_gather loop (receiver lease left GRANTED,
#                                 mirroring prefill_runner._lease_reclaim so the captured collective
#                                 does not deadlock on replay against the resident receiver connection)
#
# PROFILE=1 wraps each rank in `python -m tracy -p -r`; per-host CSVs land in
# generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv (filter OP CODE HighBwAllGatherDeviceOperation).
#
# Case A (single galaxy, no D2D) is a plain local pytest, not this 2-rank launcher:
#   pytest -svv <TEST> -k test_allgather_trace_perf_single_glx
#
#   RANK0_HOST=bh-glx-120-c05u08 RANK1_HOST=bh-glx-120-c05u02 PROFILE=1 ./run_allgather_trace_perf.sh
set -euo pipefail

RANK0_HOST="${RANK0_HOST:-bh-glx-120-c05u08}"
RANK1_HOST="${RANK1_HOST:-bh-glx-120-c05u02}"
TCP_IFACE="${TCP_IFACE:-ens5f0np0}"
PROFILE="${PROFILE:-0}"

TT_METAL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
export TT_METAL_HOME PYTHONPATH="$TT_METAL_HOME"
# Per-host local JIT cache (/tmp), same rationale as run_allgather_d2d_probe.sh.
export TT_METAL_CACHE="${PP_TT_METAL_CACHE:-/tmp/tt-metal-cache-trace-perf-${USER}}"
cd "$TT_METAL_HOME"
source python_env/bin/activate

BINDING="models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_rank_binding_2rank_d2d.yaml"
TEST="tests/ttnn/unit_tests/operations/experimental/deepseek_prefill/test_allgather_trace_perf.py"

RUNNER=(pytest -svv "$TEST" -k test_allgather_trace_perf_d2d)
if [ "$PROFILE" = "1" ]; then
  RUNNER=(python -m tracy -p -r -v -m "${RUNNER[@]}")
fi

exec python3 ttnn/ttnn/distributed/ttrun.py \
  --tcp-interface "$TCP_IFACE" \
  --rank-binding "$BINDING" \
  --mpi-args "--host ${RANK0_HOST}:1,${RANK1_HOST}:1 --map-by slot --bind-to none --tag-output --allow-run-as-root -x PATH -x LD_LIBRARY_PATH" \
  "${RUNNER[@]}"
