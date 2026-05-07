# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Stable import path for Omni mesh helpers (implementations live in ``qwen_omni_modules``).

``DistributedConfig`` / ``DistributedTensorConfig`` resolve lazily from ``run_config`` so they stay
aligned when Gr00t (or other code) replaces those names at runtime.
"""

from models.experimental.tt_symbiote.models.qwen_omni.qwen_omni_modules import (
    QwenOmniDistributedConfig,
    QwenOmniReplicatedMeshTensorConfig,
    distributed_config_col_sharded_last_dim,
    ensure_qwen_omni_normalrun_to_torch_slice,
    qwen_omni_maybe_slice_replicated_mesh_compose,
    qwen_omni_replicated_concat_dim0_tensor_config,
)

__all__ = [
    "DistributedConfig",
    "DistributedTensorConfig",
    "QwenOmniDistributedConfig",
    "QwenOmniReplicatedMeshTensorConfig",
    "distributed_config_col_sharded_last_dim",
    "ensure_qwen_omni_normalrun_to_torch_slice",
    "qwen_omni_maybe_slice_replicated_mesh_compose",
    "qwen_omni_replicated_concat_dim0_tensor_config",
]


def __getattr__(name: str):
    if name == "DistributedConfig":
        from models.experimental.tt_symbiote.core import run_config

        return run_config.DistributedConfig
    if name == "DistributedTensorConfig":
        from models.experimental.tt_symbiote.core import run_config

        return run_config.DistributedTensorConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
