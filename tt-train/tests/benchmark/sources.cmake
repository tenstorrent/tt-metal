# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

set(BENCHMARK_SOURCES
    benchmark/matmuls_benchmark.cpp
    benchmark/from_vector_benchmark.cpp
    benchmark/adamw_benchmark.cpp
    benchmark/frobenius_normalize_benchmark.cpp
    benchmark/log_softmax_benchmark.cpp
    benchmark/polynorm_fusion_benchmark.cpp
    benchmark/rmsnorm_benchmark.cpp
    benchmark/gram_matmul_benchmark.cpp
    benchmark/mla_q_rope_benchmark.cpp
    benchmark/mla_kv_assemble_benchmark.cpp
    benchmark/transpose_matmul_benchmark.cpp
    benchmark/qkv_causal_conv1d_silu_benchmark.cpp
)
