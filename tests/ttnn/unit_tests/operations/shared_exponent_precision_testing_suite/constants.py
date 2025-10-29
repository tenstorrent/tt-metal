# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


class ResultKeys:
    PCC = "pcc"
    MAX_ABS_ERROR = "max_abs_error"
    MEAN_ABS_ERROR = "mean_abs_error"
    MAX_REL_ERROR = "max_rel_error"
    MEAN_REL_ERROR = "mean_rel_error"
    ULP_MEAN = "ulp_mean"
    ULP_MAX = "ulp_max"
    ULP_PERCENTILES = "ulp_percentiles"
    ULP_PERCENTILE_50 = "50"
    ULP_PERCENTILE_90 = "90"
    ULP_PERCENTILE_99 = "99"
    ALLCLOSE_1E_2 = "allclose_1e-2"
    ALLCLOSE_1E_3 = "allclose_1e-3"


class Filepaths:
    RESULTS_DIRECTORY = "bfloat8_experiment_results"
    RAW_RESULTS_JSON_FILENAME = "raw_results.json"
    RAW_RESULTS_MARKDOWN_FILENAME = "raw_results.md"
    WORST_CASES_ANALYSIS_FILENAME = "worst_cases_analysis.md"
    PATTERN_IMPACT_ANALYSIS_FILENAME = "pattern_impact_analysis.md"


class ShapeType:
    SINGLE_TILE = "single_tile"
    MULTI_TILE = "multi_tile"
    RECTANGULAR = "rectangular"


class OperationType:
    MATMUL = "matmul"
    MATMUL_TT = "matmul_tt"
    SOFTMAX = "softmax"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    EXP = "exp"
    LOG = "log"
    SQRT = "sqrt"


class MatmulTTConfig:
    TILE_W = "tile_w"
    TRANSPOSE = "transpose"
