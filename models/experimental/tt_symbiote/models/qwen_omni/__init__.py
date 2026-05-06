# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni-MoE–specific helpers (HF integration patches, mesh defaults, etc.)."""

from models.experimental.tt_symbiote.models.qwen_omni.distributed_config import (
    QwenOmniDistributedConfig,
    QwenOmniReplicatedMeshTensorConfig,
    qwen_omni_replicated_concat_dim0_tensor_config,
)
from models.experimental.tt_symbiote.models.qwen_omni.hf_generation_compat import (
    apply_qwen3_omni_talker_prepare_inputs_fix,
)

__all__ = [
    "apply_qwen3_omni_talker_prepare_inputs_fix",
    "QwenOmniDeviceInit",
    "QwenOmniDistributedConfig",
    "QwenOmniReplicatedMeshTensorConfig",
    "qwen_omni_replicated_concat_dim0_tensor_config",
]


def __getattr__(name: str):
    # Lazy: importing ``device_management`` pulls ``run_config``; package ``__init__`` runs before
    # ``run_config`` can finish loading ``distributed_config`` from this package.
    if name == "QwenOmniDeviceInit":
        from models.experimental.tt_symbiote.utils.device_management import QwenOmniDeviceInit as _QwenOmniDeviceInit

        return _QwenOmniDeviceInit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
