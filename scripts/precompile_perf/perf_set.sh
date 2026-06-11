# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Curated per-op perf-measurement manifest for up-front precompile A/B.
#
# Sourced by run_op_perf.sh. Each record is one operation, with the test
# target(s) in the "main" tree and the "nightly" tree, plus an optional pytest
# -k filter used to deselect trace-driven cases (precompile cannot warm a
# traced command sequence, so those must be excluded from both arms).
#
# Record format (tab-separated):
#   op_name <TAB> main_target <TAB> nightly_target <TAB> k_filter
#
# - main_target / nightly_target: one or more paths RELATIVE to the tree base
#   (space-separated for multiple). "-" means this op has no test in that tree.
# - k_filter: a pytest -k expression applied to BOTH arms, or "-" for none.
#
# Tree bases (do not include in the records):
PERF_MAIN_BASE="tests/ttnn/unit_tests/operations"
PERF_NIGHTLY_BASE="tests/ttnn/nightly/unit_tests/operations"

# conv3d's test_conv3d_sweep_shapes is a ~1536-case sweep that would dominate
# the warm-run wall. It is pruned by default; set PERF_INCLUDE_CONV3D_SWEEP=1 to
# fold it back in (it is cheap to COLLECT — _fast_conv3d stand-in — but long to RUN).
_conv3d_kfilter="not test_conv3d_sweep_shapes"
[[ "${PERF_INCLUDE_CONV3D_SWEEP:-0}" == "1" ]] && _conv3d_kfilter="-"

# op                        main target(s)                                  nightly target(s)                          k-filter
perf_ops() {
cat <<EOF
conv2d	conv/test_conv2d.py	conv/test_conv2d.py	-
conv3d	conv/test_conv3d.py	conv/test_conv3d.py	${_conv3d_kfilter}
matmul	matmul/test_matmul.py	matmul/test_matmul.py matmul/test_matmul2.py	-
reductions	reduce	reduction	-
layernorm	fused/test_layer_norm.py	fused/test_layernorm.py	-
rms_norm	fused/test_rms_norm.py	fused/test_rmsnorm.py	-
groupnorm	fused/test_group_norm.py	fused/test_group_norm.py	-
eltwise	eltwise	eltwise	-
tilize	data_movement/test_tilize.py	-	not deepseek_v3_mla_tilize_trace_mode
tilize_with_val_padding	data_movement/test_tilize_with_val_padding.py	-	-
EOF
}

# Trace-driven cases that MUST stay deselected (precompile can't warm a traced
# command stream). The whole-directory selections (reductions, eltwise) are
# re-scanned for trace markers at run time by run_op_perf.sh's pre-flight; this
# list is the known set so the catalog and the runner agree.
#   main:    matmul/test_experimental.py::test_ttnn_linear  (graph tracer; not in our matmul target)
#            data_movement/test_tilize.py::test_deepseek_v3_mla_tilize_trace_mode  (metal trace; filtered above)
#   nightly: matmul/test_rs_matmul_1d_gather_in0.py  (metal trace; not in our matmul target)
#            sdpa/test_sdpa_chunked.py, sdpa/test_sdpa_prefill.py  (metal trace; not in our set)
#            experimental/test_moe_compute_single_card.py  (metal trace; not in our set)
PERF_TRACE_MARKERS='begin_trace_capture|execute_trace|trace_region_size|tracer.trace'
