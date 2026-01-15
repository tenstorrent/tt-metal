# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Shared constants for the sweep framework.

This module contains constants that are used across multiple modules
in the sweep framework to ensure consistency and avoid duplication.
"""

# Lead models are models that are prioritized for sweep testing.
# These patterns are matched against the source path in traced operations
# to identify which vectors belong to lead model workloads.
#
# Used by:
#   - sweeps_parameter_generator.py: To filter vector generation for lead models only
#   - master_config_loader.py: To filter configurations when loading from master JSON
#
# To add a new lead model, add the model directory name pattern here.
# Example: "llama3" would match source paths containing "llama3"
LEAD_MODELS = [
    "deepseek_v3",
]
