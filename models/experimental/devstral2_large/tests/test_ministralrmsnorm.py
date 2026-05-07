# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: Hugging Face ``Ministral3RMSNorm`` vs ``TtDevstral2LargeRMSNorm`` on Devstral-2-123B shards.

Loads **only** the requested HF weight tensors via ``model.safetensors.index.json`` (one or few
shards), builds the meta-format dict used by TT (``map_hf_to_meta_keys`` / HF RoPE path), and
compares against TT. Avoids full checkpoint load.

Covers layer ``input_layernorm`` and ``post_attention_layernorm``. Root ``model.norm`` is not tested.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pytest
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.ministral3.modeling_ministral3 import Ministral3RMSNorm

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_large.tt.tt_ministralrmsnorm import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    TtDevstral2LargeRMSNorm,
)
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta_no_qkv_permute, standardize_hf_keys
from models.tt_transformers.tt.ccl import TT_CCL

DEVSTRAL2_LARGE_REPO_ID = "mistralai/Devstral-2-123B-Instruct-2512"


def _text_cfg_from_hf_cfg(hf_cfg) -> Ministral3Config:
    inner = getattr(hf_cfg, "text_config", None)
    out = inner if inner is not None else hf_cfg
    if not isinstance(out, Ministral3Config):
        raise TypeError(f"Expected Ministral3Config, got {type(out)!r}")
    return out


@dataclass
class _Ministral3NormArgsShim:
    """Subset of :class:`ModelArgs` for :class:`TtMinistralRMSNorm` in unit tests.

    ``is_distributed_norm`` / ``ccl_topology`` mirror :meth:`ModelArgs.is_distributed_norm` and
    :meth:`ModelArgs.ccl_topology`.
    """

    mesh_device: object
    dim: int
    norm_eps: float
    dummy_weights: bool = False
    rms_norm_add_unit_offset: bool = False

    def get_state_dict_prefix(self, module_name: str, layer_num: int | None, is_vision: bool = False):
        if layer_num is None:
            return ""
        return f"layers.{layer_num}."

    def weight_cache_path(self, dtype):
        return None

    def is_distributed_norm(self, mode):
        """Force **replicated** activations + ``ttnn.rms_norm`` for PCC vs HF (not tensor-parallel norm)."""
        return False

    def ccl_topology(self):
        """Match ``ModelArgs.ccl_topology`` (Ring on P150 x4 / x8 fabric)."""
        md = self.mesh_device
        if md is None or md.get_num_devices() <= 1:
            return ttnn.Topology.Linear
        ct = ttnn.cluster.get_cluster_type()
        if ct in (
            ttnn.cluster.ClusterType.P300_X2,
            ttnn.cluster.ClusterType.P150_X4,
            ttnn.cluster.ClusterType.P150_X8,
        ):
            return ttnn.Topology.Ring
        if ct in (ttnn.cluster.ClusterType.T3K, ttnn.cluster.ClusterType.GALAXY):
            return ttnn.Topology.Ring if md.get_num_devices() >= 8 else ttnn.Topology.Linear
        return ttnn.Topology.Linear


def _load_hf_tensors_for_keys(repo_id: str, keys: list[str]) -> dict[str, torch.Tensor]:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import safe_open as safetensors_safe_open

    try:
        index_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors.index.json",
            local_files_only=os.getenv("CI") == "true",
        )
    except Exception as exc:
        raise RuntimeError(f"Could not download model.safetensors.index.json: {exc}") from exc

    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    out: dict[str, torch.Tensor] = {}
    for key in keys:
        if key not in weight_map:
            raise KeyError(f"Key {key!r} not in weight_map for {repo_id}")
        shard_name = weight_map[key]
        shard_path = hf_hub_download(
            repo_id=repo_id,
            filename=shard_name,
            local_files_only=os.getenv("CI") == "true",
        )
        with safetensors_safe_open(shard_path, framework="pt", device="cpu") as sf:
            if key not in sf.keys():
                raise KeyError(f"Key {key!r} missing from shard {shard_name}")
            out[key] = sf.get_tensor(key).clone()
    return out


def _hf_key_for_variant(norm_variant: str) -> str:
    if norm_variant == "layer_pre":
        return "model.layers.0.input_layernorm.weight"
    if norm_variant == "layer_post":
        return "model.layers.0.post_attention_layernorm.weight"
    raise ValueError(norm_variant)


def _build_meta_state_dict(hf_partial: dict[str, torch.Tensor], text_cfg: Ministral3Config) -> dict:
    sd = dict(hf_partial)
    sd = standardize_hf_keys(sd)
    head_dim = getattr(text_cfg, "head_dim", None) or text_cfg.hidden_size // text_cfg.num_attention_heads
    return convert_hf_to_meta_no_qkv_permute(
        sd,
        head_dim=int(head_dim),
        n_heads=text_cfg.num_attention_heads,
        n_kv_heads=text_cfg.num_key_value_heads,
    )


@torch.no_grad()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "mesh_device",
    [
        # Align with models/tt_transformers/conftest.py: 4-device Quietbox / 1×4 fabric is N150x4 (WH)
        # or P150x4 (Blackhole). Without these keys, MESH_DEVICE=N150x4 falls through to
        # len(ttnn.get_device_ids()), which only matches 1×4 when exactly four boards are visible.
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "P150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (128,))
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 30000000,
            "num_command_queues": 1,
            "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("norm_variant", ("layer_pre", "layer_post"))
def test_ministral3_rmsnorm_pcc_devstral2_large_partial_weights(
    mesh_device,
    seq_len,
    batch_size,
    norm_variant,
):
    try:
        hf_cfg = AutoConfig.from_pretrained(
            DEVSTRAL2_LARGE_REPO_ID,
            trust_remote_code=True,
            local_files_only=os.getenv("CI") == "true",
        )
    except Exception as exc:
        pytest.skip(f"Could not load Hugging Face config: {exc}")

    text_cfg = _text_cfg_from_hf_cfg(hf_cfg)
    hf_key = _hf_key_for_variant(norm_variant)

    try:
        hf_partial = _load_hf_tensors_for_keys(DEVSTRAL2_LARGE_REPO_ID, [hf_key])
    except Exception as exc:
        pytest.skip(f"Could not download Norm shard(s) from Hub: {exc}")

    try:
        meta_sd = _build_meta_state_dict(hf_partial, text_cfg)
    except Exception as exc:
        pytest.skip(f"HF→meta conversion failed: {exc}")

    w = hf_partial[hf_key]
    ref_norm = Ministral3RMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps)
    ref_norm.weight.data.copy_(w)
    ref_norm.eval()

    args_shim = _Ministral3NormArgsShim(
        mesh_device=mesh_device,
        dim=text_cfg.hidden_size,
        norm_eps=float(text_cfg.rms_norm_eps),
    )

    torch_in = torch.randn(batch_size, 1, seq_len, text_cfg.hidden_size, dtype=torch.bfloat16)
    ref_out = ref_norm(torch_in)

    dtype = ttnn.bfloat16
    tt_ccl = TT_CCL(mesh_device)
    post_attention = norm_variant == "layer_post"

    tt_norm = TtDevstral2LargeRMSNorm(
        mesh_device,
        args_shim,
        meta_sd,
        weight_cache_path=None,
        layer_num=0,
        tt_ccl=tt_ccl,
        post_attention=post_attention,
    )

    tt_in = ttnn.from_torch(
        torch_in,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_out = tt_norm(tt_in, Mode.PREFILL)
    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    if tt_torch.shape != ref_out.shape:
        tt_torch = tt_torch.reshape(ref_out.shape)

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(ref_out, tt_torch, pcc_required)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC ({norm_variant}): {pcc_message}")
    assert passing, f"PCC below {pcc_required}: {pcc_message}"
