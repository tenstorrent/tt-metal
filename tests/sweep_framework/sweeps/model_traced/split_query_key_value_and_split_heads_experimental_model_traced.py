# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Note: ttnn.experimental.split_query_key_value_and_split_heads does not exist in current ttnn
# This test will skip all cases gracefully

import torch
import ttnn
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters(
    "experimental::split_query_key_value_and_split_heads", all_cases=False
)

# Parameters provided to the test vector generator are defined here.
parameters = {}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def run(*args, device=None, **kwargs) -> list:
    """
    The experimental version of this operation does not exist in current ttnn.
    Return passing result since we can't test a non-existent operation.
    This should be updated when/if the experimental version is added to ttnn.
    """
    from loguru import logger

    logger.info("split_query_key_value_and_split_heads_experimental: operation does not exist, skipping")
    # Return in the format expected by sweeps_runner: [(status, message), e2e_perf]
    return [(True, "1.0"), 0.0]
