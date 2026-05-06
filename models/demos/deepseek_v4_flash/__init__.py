# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V4 Flash TTNN bringup utilities and primary runtime contract."""

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.converter import convert_hf_checkpoint
from models.demos.deepseek_v4_flash.mesh_config import MeshConfig, ModeConfig
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest, validate_tt_manifest
from models.demos.deepseek_v4_flash.runtime import (
    DeepSeekRuntimeBlocked,
    HardwareMeshProbeResult,
    HostBoundaryViolation,
    RuntimeBlocker,
    RuntimeBlockerReport,
    RuntimeHostBoundaryGuard,
    TtDeepSeekV4FlashRuntime,
)

__all__ = [
    "DeepSeekV4FlashConfig",
    "DeepSeekRuntimeBlocked",
    "HardwareMeshProbeResult",
    "HostBoundaryViolation",
    "MeshConfig",
    "ModeConfig",
    "RuntimeBlocker",
    "RuntimeBlockerReport",
    "RuntimeHostBoundaryGuard",
    "TtDeepSeekV4FlashRuntime",
    "convert_hf_checkpoint",
    "load_tt_manifest",
    "validate_tt_manifest",
]
