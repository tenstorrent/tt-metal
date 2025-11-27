# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Modular MLP module.

This package provides a refactored, modular MLP implementation with:
- Separated config computation (mlp_config.py)
- Encapsulated CCL strategies (ccl_strategies.py)
- Clean forward pass implementation (modular_mlp.py)
- Static analysis tools (trace_dependencies.py)

Usage:
    from models.common.modules.mlp import ModularMLP, MLPConfig

    # Or use the factory function for compatibility with existing code
    from models.common.modules.mlp import create_mlp_config_from_model_args
"""

from models.common.modules.mlp.modular_attempt.mlp_config import (
    MLPConfig,
    MLPLayerConfig,
    MLPProgramConfigs,
    MLPMemoryConfigs,
    HardwareTopology,
    create_mlp_config_from_model_args,
)

from models.common.modules.mlp.modular_attempt.ccl_strategies import (
    CCLStrategy,
    SingleChipStrategy,
    LinearTopologyStrategy,
    GalaxyTopologyStrategy,
    create_ccl_strategy,
)

from models.common.modules.mlp.modular_attempt.modular_mlp import ModularMLP

__all__ = [
    # Config
    "MLPConfig",
    "MLPLayerConfig",
    "MLPProgramConfigs",
    "MLPMemoryConfigs",
    "HardwareTopology",
    "create_mlp_config_from_model_args",
    # CCL Strategies
    "CCLStrategy",
    "SingleChipStrategy",
    "LinearTopologyStrategy",
    "GalaxyTopologyStrategy",
    "create_ccl_strategy",
    # MLP Module
    "ModularMLP",
]
