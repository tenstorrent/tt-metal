#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export TT_METAL_HOME="$(pwd)"
export TT_METAL_RUNTIME_ROOT="$(pwd)"
export PYTHONPATH="${TT_METAL_HOME}:${PYTHONPATH:-}"
source python_env/bin/activate

export TT_METAL_DEVICE_PROFILER=1
export ENABLE_TRACY=1
export TT_METAL_PROFILER_MID_RUN_DUMP=1
export TTNN_RUN_GEMM_FLOPS_BENCHMARK=1

mkdir -p generated

BENCH=tests/ttnn/unit_tests/benchmarks/test_benchmark.py

pytest "${BENCH}::test_matmul_2d_host_perf" -xvs --timeout=7200

ARCH=$(python -c "from models.common.utility_functions import is_blackhole; print('bh' if is_blackhole() else 'wh')")
mkdir -p tech_reports/GEMM_FLOPS/data
cp generated/matmul_benchmark_report.csv "tech_reports/GEMM_FLOPS/data/${ARCH}.csv"

python tech_reports/GEMM_FLOPS/plot_util_grid.py
python tech_reports/GEMM_FLOPS/plot_utilization.py
python tech_reports/GEMM_FLOPS/plot_scatter_tracing.py
python tech_reports/GEMM_FLOPS/plot_scatter_performance.py
python tech_reports/GEMM_FLOPS/plot_bar.py
python tech_reports/GEMM_FLOPS/plot_aspect_ratio_by_dtype.py
