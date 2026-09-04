#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# One-command re-run of the Llama 3.1 8B prefill bring-up test ladders.
#
#   ./run_bringup.sh            # everything, in recipe order (D -> M -> P)
#   ./run_bringup.sh d1         # one stage
#   ./run_bringup.sh d2 d3      # several
#
# Each FABRIC CONFIG runs in its OWN pytest process, and so does each mesh shape. Once a process has
# brought fabric up (or deliberately not, for the fabric-less (1,1) case), a later mesh with a
# different fabric config in that same process dies with
#   "Fabric Router Sync: Timeout after 10000 ms ... expected status 0xa2b2c2d2"
# — the routers never complete the ethernet handshake. Observed both ways round: (1,1) then (1,4), and
# (8,4) then (1,4). This grouping is a correctness requirement, not a preference.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
UNIT="models/demos/llama3_1_8b_d_p/tests/unit"
TORCH_T="models/demos/llama3_1_8b_d_p/tests/torch"
TESTS="models/demos/llama3_1_8b_d_p/tests"
PY="${PYTHON:-/opt/venv/bin/python}"
LOGDIR="${BRINGUP_LOG_DIR:-$HERE/.bringup_runs}"
mkdir -p "$LOGDIR"

cd "$ROOT"
FAILED=()

run() {  # run <label> <pytest args...>
    local label="$1"; shift
    local log="$LOGDIR/${label}.log"
    echo "=== $label ==="
    if timeout "${BRINGUP_TIMEOUT:-5400}" "$PY" -m pytest "$@" -q --no-header -p no:randomly > "$log" 2>&1; then
        grep -E "passed|failed" "$log" | tail -1
    else
        echo "FAILED (see $log)"
        grep -E "FAILED|Error|assert|Timeout" "$log" | grep -viE "metal \||meshdevice" | tail -5
        FAILED+=("$label")
    fi
}

stage_d1() {
    # Host-only: no device, no conftest fixtures needed.
    run d1_reference "$TORCH_T/test_llama_reference.py" "$TORCH_T/test_golden_cache.py" --noconftest
}

stage_d2_d3() {
    # The decoder test suite. Written in D2, made to pass in D3 — same table, run the same way.
    # (1,1) runs with fabric DISABLED, so it cannot share a process with any fabric mesh.
    run d3_single_chip "$UNIT/test_norm_vs_ref.py" "$UNIT/test_dense_mlp_vs_ref.py" \
        "$UNIT/test_rope_vs_ref.py" "$UNIT/test_attention_vs_ref.py" \
        "$UNIT/test_decoder_layer_vs_ref.py" -k "1x1"
    run d3_mesh_norm_mlp "$UNIT/test_norm_vs_ref.py" "$UNIT/test_dense_mlp_vs_ref.py" -k "8x4"
    run d3_mesh_rope "$UNIT/test_rope_vs_ref.py" -k "8x4"
    run d3_mesh_kv "$UNIT/test_kv_cache_write_vs_ref.py" "$UNIT/test_kv_cache_gqa_sp_vs_ref.py" -k "8x4"
    run d3_mesh_ring "$UNIT/test_ring_joint_sp_vs_ref.py" "$UNIT/test_ring_joint_cache_read_sp_vs_ref.py" -k "8x4"
    run d3_mesh_chunked "$UNIT/test_attention_chunked_vs_ref.py" -k "8x4"
    run d3_mesh_layer "$UNIT/test_decoder_layer_vs_ref.py" -k "8x4"
}

stage_m() {
    # Host-only padding math first (no device), then the mesh model suite.
    run m3_host "$UNIT/test_lm_head_vs_ref.py::test_per_device_vocab_padding_math" --noconftest
    run m3_mesh_embed "$UNIT/test_parallel_embedding_vs_ref.py" -k "8x4"
    run m3_mesh_lm_head "$UNIT/test_lm_head_vs_ref.py" -k "8x4"
    run m3_mesh_model "$UNIT/test_model_sp_vs_ref.py" -k "8x4"
}

stage_p() {
    # P1/P2 need a real checkpoint (HF_MODEL); they skip cleanly without one.
    if [ -n "${HF_MODEL:-}" ]; then
        PREFILL_CHUNKED=0 && export PREFILL_CHUNKED=0 && run p1_kv_pcc_oneshot "$TESTS/galaxy_prefill_kv_pcc.py"
        export PREFILL_CHUNKED=1 && run p2_kv_pcc_chunked "$TESTS/galaxy_prefill_kv_pcc.py"
        unset PREFILL_CHUNKED
    else
        echo "=== p1/p2 skipped (set HF_MODEL to a Llama-3.1-8B-Instruct checkpoint) ==="
    fi
    run p4_kv_table_host "$TESTS/test_kv_cache_table.py::test_chunk_size_bytes_matches_tile_geometry" --noconftest
    run p4_kv_table "$TESTS/test_kv_cache_table.py" -k "8x4"
}

declare -A STAGES=([d1]=stage_d1 [d2]=stage_d2_d3 [d3]=stage_d2_d3 [m]=stage_m [p]=stage_p)

if [ $# -eq 0 ]; then
    stage_d1; stage_d2_d3; stage_m; stage_p
else
    seen=""
    for s in "$@"; do
        fn="${STAGES[${s,,}]:-}"
        [ -z "$fn" ] && { echo "unknown stage '$s' (d1 d2 d3 m p)"; exit 2; }
        case " $seen " in *" $fn "*) continue;; esac
        seen="$seen $fn"; "$fn"
    done
fi

echo
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "ALL GREEN"
else
    echo "FAILED GROUPS: ${FAILED[*]}"
    exit 1
fi
