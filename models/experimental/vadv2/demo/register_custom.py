# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Register custom VAD transforms and datasets with mmdet3d.
Import this module before using the VAD model to ensure all custom classes are registered.

Usage:
    import models.experimental.vadv2.demo.register_custom
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import custom modules to trigger registration decorators
from models.experimental.vadv2.demo.nuscenes_vad_dataset import VADCustomNuScenesDataset  # noqa: F401
from models.experimental.vadv2.demo.transforms_vad_3d import (  # noqa: F401
    CustomLoadMultiViewImageFromFiles,
    CustomLoadPointsFromFile,
    CustomObjectRangeFilter,
    CustomObjectNameFilter,
    PadMultiViewImage,
    NormalizeMultiviewImage,
    PhotoMetricDistortionMultiViewImage,
    CustomCollect3D,
    RandomScaleImageMultiViewImage,
    CustomPointsRangeFilter,
    DefaultFormatBundle,
    DefaultFormatBundle3D,
    CustomDefaultFormatBundle3D,
)

print("✓ VAD custom transforms and datasets registered successfully")
