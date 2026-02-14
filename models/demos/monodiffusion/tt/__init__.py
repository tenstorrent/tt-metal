# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MonoDiffusion TTNN Implementation
Following vanilla_unet module structure
"""

from models.demos.monodiffusion.tt.model import (
    TtMonoDiffusion,
    create_monodiffusion_from_configs,
    create_monodiffusion_from_parameters,
)
from models.demos.monodiffusion.tt.config import (
    TtMonoDiffusionLayerConfigs,
    TtMonoDiffusionConfigBuilder,
    create_monodiffusion_configs_from_parameters,
)
from models.demos.monodiffusion.tt.common import (
    load_reference_model,
    create_monodiffusion_preprocessor,
    concatenate_skip_connection,
    compute_pcc,
    MONODIFFUSION_PCC_TARGET,
)

__all__ = [
    # Model classes
    "TtMonoDiffusion",
    "create_monodiffusion_from_configs",
    "create_monodiffusion_from_parameters",
    # Configuration
    "TtMonoDiffusionLayerConfigs",
    "TtMonoDiffusionConfigBuilder",
    "create_monodiffusion_configs_from_parameters",
    # Utilities
    "load_reference_model",
    "create_monodiffusion_preprocessor",
    "concatenate_skip_connection",
    "compute_pcc",
    "MONODIFFUSION_PCC_TARGET",
]
