# SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


class ResultKeys:
    PCC_KEY = "pcc"
    MAX_ABS_ERROR_KEY = "max_abs_error"
    MEAN_ABS_ERROR_KEY = "mean_abs_error"
    MAX_REL_ERROR_KEY = "max_rel_error"
    MEAN_REL_ERROR_KEY = "mean_rel_error"
    ULP_MEAN_KEY = "ulp_mean"
    ULP_MAX_KEY = "ulp_max"
    ULP_PERCENTILES_KEY = "ulp_percentiles"
    ULP_PERCENTILE_50_KEY = "50"
    ULP_PERCENTILE_90_KEY = "90"
    ULP_PERCENTILE_99_KEY = "99"
    ALLCLOSE_1E_2_KEY = "allclose_1e-2"
    ALLCLOSE_1E_3_KEY = "allclose_1e-3"


class Filepaths:
    RESULTS_DIRECTORY = "bfloat8_experiment_results"
    RAW_RESULTS_JSON_FILENAME = "raw_results.json"
    RAW_RESULTS_MARKDOWN_FILENAME = "raw_results.md"
    WORST_CASES_ANALYSIS_FILENAME = "worst_cases_analysis.md"
    PATTERN_IMPACT_ANALYSIS_FILENAME = "pattern_impact_analysis.md"


class ShapeType:
    SINGLE_TILE_KEY = "single_tile"
    MULTI_TILE_KEY = "multi_tile"
    RECTANGULAR_KEY = "rectangular"


class OperationType:
    MATMUL_KEY = "matmul"
    MATMUL_TT_KEY = "matmul_tt"
    ADD_KEY = "add"
    SUB_KEY = "sub"
    MUL_KEY = "mul"
    DIV_KEY = "div"
    SUM_KEY = "sum"
    MEAN_KEY = "mean"
    MAX_KEY = "max"
    MIN_KEY = "min"
    RELU_KEY = "relu"
    SIGMOID_KEY = "sigmoid"
    TANH_KEY = "tanh"
    EXP_KEY = "exp"
    LOG_KEY = "log"
    SQRT_KEY = "sqrt"


class MatmulTTConfig:
    TILE_W_KEY = "tile_w"
    TRANSPOSE_KEY = "transpose"
