#!/bin/bash
# Full-torus validation of pjosipovic/unrolled-v-tp-sharded-kv with scripts/run_safe_pytest.sh (see VALIDATION.md).
# Run from the repo root with python_env active. One line per step; logs in $LOGS.
cd "$(git rev-parse --show-toplevel)" || exit 1
LOGS=${LOGS:-$PWD/validation_logs}
mkdir -p "$LOGS"
export OMP_NUM_THREADS=$(nproc) LOGURU_LEVEL=INFO TT_METAL_HOME=$PWD PYTHONPATH=$PWD PREFILL_TORUS_XY_CERTIFIED=1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
unset TT_METAL_CCACHE_KERNEL_SUPPORT TT_METAL_DEVICE_PROFILER
RJS=tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py
MLA=models/demos/deepseek_v3_d_p/tests/test_mla.py
SV=models/demos/deepseek_v3_d_p/tests/scratch_validation

run() {
    local tag=$1
    shift
    scripts/run_safe_pytest.sh --run-all "$@" > "$LOGS/$tag.log" 2>&1
    echo "$tag: rc=$? $(grep -E '[0-9]+ (passed|failed|skipped|error)' "$LOGS/$tag.log" | tail -1)"
}

run rjs_gate $RJS -k test_ring_joint_attention_chunked_expanded_k_perf_check
grep -oE "perf check kimi[^:]*: math_util=[0-9.]+%.*duration=[0-9.]+ ms" "$LOGS/rjs_gate.log" | sed 's/^/  /'
run rjs_acc_k3 $RJS -k "expanded_k_production_accuracy and kimi_k3_expanded"
run rjs_acc_k27 $RJS -k "expanded_k_production_accuracy and kimi_k2_7_expanded"
grep -ohE 'chunk +[0-9]+: min per-head PCC [0-9.]+' "$LOGS"/rjs_acc_k*.log | awk '{print $6}' | sort -n | head -1 | sed 's/^/  min per-head PCC /'
run rjs_det $RJS -k expanded_k_production_determinism
run rjs_linear $RJS -k expanded_k_linear_accuracy
run rjs_cachehit $RJS -k cache_hit_uses_new_semaphores
run rotacc $SV/rotacc.py
run mla_rot $MLA -k "torus-xy-8x4 and cpu and (rot- or maxedge-1u) and metadata and no_determinism and sp_only and (k3 or k2_7)"
run mla_tp $MLA -k "torus-xy-8x4 and func and deep-50k+5k and metadata and no_determinism and tp_sharded and (k3 or k2_7)"
run mla_rt_gate $MLA -k "test_mla_chunked_perf_check"
grep -oE 'kimi_k3 chunked 50k\+5k realtime perf: [^,]*, [^,]*' "$LOGS/mla_rt_gate.log" | sed 's/^/  /'
run minimal_matmul tests/ttnn/nightly/unit_tests/operations/experimental/test_minimal_matmul.py
run high_bw_all_gather tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather_fabric.py
echo "done $(date +%T), logs in $LOGS"
