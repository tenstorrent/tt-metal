# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: Hugging Face ``Ministral3DecoderLayer`` vs ``TtMinistral3DecoderLayer`` (layer 0, partial Hub weights).

Loads attention, MLP, and both RMSNorm tensors for ``model.layers.0`` via ``model.safetensors.index.json``,
converts to meta keys with :func:`convert_hf_to_meta_no_qkv_permute`, and compares full prefill layer output.
Does not load the full checkpoint.

Prefill PCC forces **replicated** RMSNorm (``is_distributed_norm`` → false) so wide hidden does not use
``rms_norm_pre_all_gather`` on multi-chip Blackhole (L1 circular-buffer overflow); same rationale as
``test_ministralrmsnorm``.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.ministral3.modeling_ministral3 import (
    Ministral3DecoderLayer,
    Ministral3RotaryEmbedding,
)

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_large.tt.tt_ministral3_decoder_layer import TtMinistral3DecoderLayer
from models.experimental.devstral2_large.tt.tt_ministralrmsnorm import DEVSTRAL2_LARGE_L1_SMALL_SIZE
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta_no_qkv_permute, standardize_hf_keys
from models.tt_transformers.tt.model_config import ModelArgs, PrecisionSetting, TensorGroup

DEVSTRAL2_LARGE_REPO_ID = "mistralai/Devstral-2-123B-Instruct-2512"


def _text_cfg_from_hf_cfg(hf_cfg) -> Ministral3Config:
    inner = getattr(hf_cfg, "text_config", None)
    out = inner if inner is not None else hf_cfg
    if not isinstance(out, Ministral3Config):
        raise TypeError(f"Expected Ministral3Config, got {type(out)!r}")
    return out


def _decoder_layer_hf_keys(layer: int = 0) -> list[str]:
    p = f"model.layers.{layer}"
    return [
        f"{p}.self_attn.q_proj.weight",
        f"{p}.self_attn.k_proj.weight",
        f"{p}.self_attn.v_proj.weight",
        f"{p}.self_attn.o_proj.weight",
        f"{p}.mlp.gate_proj.weight",
        f"{p}.mlp.up_proj.weight",
        f"{p}.mlp.down_proj.weight",
        f"{p}.input_layernorm.weight",
        f"{p}.post_attention_layernorm.weight",
    ]


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


def _to_bf16_host_if_fp8(t: torch.Tensor) -> torch.Tensor:
    fp8_dtypes = tuple(
        dt
        for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz")
        if (dt := getattr(torch, name, None)) is not None
    )
    if fp8_dtypes and t.dtype in fp8_dtypes:
        return t.to(torch.bfloat16)
    return t


def _force_mlp_weight_dtypes_bf16(model_args: ModelArgs) -> None:
    """Stage FFN weights as BF16 on device (matches BF16 HF reference numerics)."""
    dec = model_args.model_config["DECODERS_OPTIMIZATIONS"]
    for conf in dec.decoder_optimizations.values():
        tp = conf._opt_settings["TensorPrecision"]
        tp[TensorGroup.FF1_FF3] = PrecisionSetting.BF16
        tp[TensorGroup.FF2] = PrecisionSetting.BF16


def _force_replicated_rmsnorm_for_layer_pcc(model_args: ModelArgs) -> None:
    """Match HF single-process RMSNorm and avoid ``rms_norm_pre_all_gather`` on wide prefill.

    For ``dim > 4096`` and multi-chip prefill, :meth:`ModelArgs.is_distributed_norm` enables the
    distributed norm path. On Blackhole that uses ``rms_norm_pre_all_gather`` with a sharded program
    whose static circular buffers exceed L1 (~1.5 MiB) for ~12k hidden — see ``tt_ministralrmsnorm``
    module docstring and :mod:`test_ministralrmsnorm`.

    Call **before** constructing :class:`TtMinistral3DecoderLayer` so embedded RMSNorm modules capture
    the overridden predicate.
    """
    model_args.is_distributed_norm = lambda mode: False  # type: ignore[method-assign]


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


@pytest.fixture
def trust_remote_ministral(monkeypatch):
    from models.tt_transformers.tt import model_config as mc

    orig_set = mc.ModelArgs._set_hf_params

    def _set_hf_params_trust(self, checkpoint_dir: str):
        self.trust_remote_code_hf = True
        return orig_set(self, checkpoint_dir)

    monkeypatch.setattr(mc.ModelArgs, "_set_hf_params", _set_hf_params_trust)

    def _get_hf_model_cls_devstral_safe(self):
        from transformers import AutoModelForCausalLM
        from transformers.models.auto.modeling_auto import AutoModelForImageTextToText

        if not self.is_multimodal:
            return AutoModelForCausalLM
        if type(self.hf_config) in AutoModelForImageTextToText._model_mapping:
            return AutoModelForImageTextToText
        raise ValueError(
            f"Test supports multimodal configs in AutoModelForImageTextToText only; got {type(self.hf_config)}"
        )

    monkeypatch.setattr(mc.ModelArgs, "get_hf_model_cls", _get_hf_model_cls_devstral_safe)


@pytest.fixture
def devstral2_123b_dummy_config_path(monkeypatch):
    from models.tt_transformers.tt import model_config as mc

    short_name = DEVSTRAL2_LARGE_REPO_ID.strip("/").split("/")[-1]
    monkeypatch.setitem(mc.ModelArgs.LOCAL_HF_PARAMS, short_name, DEVSTRAL2_LARGE_REPO_ID)


@torch.no_grad()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "mesh_device",
    [
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
def test_ministral3_decoder_layer_pcc_devstral2_large_partial_weights(
    mesh_device,
    seq_len,
    batch_size,
    monkeypatch,
    trust_remote_ministral,
    devstral2_123b_dummy_config_path,
):
    monkeypatch.setenv("HF_MODEL", DEVSTRAL2_LARGE_REPO_ID)
    monkeypatch.setenv("MAX_PREFILL_CHUNK_SIZE", "128")

    try:
        hf_cfg = AutoConfig.from_pretrained(
            DEVSTRAL2_LARGE_REPO_ID,
            trust_remote_code=True,
            local_files_only=os.getenv("CI") == "true",
        )
    except Exception as exc:
        pytest.skip(f"Could not load Hugging Face config: {exc}")

    text_cfg = _text_cfg_from_hf_cfg(hf_cfg)
    text_cfg._attn_implementation = "eager"

    all_keys = _decoder_layer_hf_keys(0)
    try:
        hf_partial = _load_hf_tensors_for_keys(DEVSTRAL2_LARGE_REPO_ID, all_keys)
    except Exception as exc:
        pytest.skip(f"Could not download decoder-layer shard(s) from Hub: {exc}")

    for k in all_keys:
        hf_partial[k] = _to_bf16_host_if_fp8(hf_partial[k])

    try:
        meta_sd = _build_meta_state_dict(hf_partial, text_cfg)
    except Exception as exc:
        pytest.skip(f"HF→meta conversion failed: {exc}")

    layer_prefix = "model.layers.0."
    hf_layer_sd = {k[len(layer_prefix) :]: hf_partial[k].clone() for k in all_keys}

    ref_layer = Ministral3DecoderLayer(text_cfg, layer_idx=0).eval()
    ref_layer.load_state_dict(hf_layer_sd, strict=True)
    ref_layer.to(dtype=torch.bfloat16)

    hidden_size = text_cfg.hidden_size
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    rotary = Ministral3RotaryEmbedding(text_cfg).eval()
    position_embeddings = rotary(x, position_ids=position_ids)
    causal_mask = create_causal_mask(
        config=text_cfg,
        inputs_embeds=x,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )

    ref_out = ref_layer(
        x,
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        position_embeddings=position_embeddings,
    )

    dtype = ttnn.bfloat16
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max(512, seq_len),
        dummy_weights=True,
        use_hf_rope=True,
        cache_hf=False,
    )

    _layer0_opt = model_args.decoders_optimizations.decoder_optimizations[0]
    _layer0_opt.tensor_dtype_settings[TensorGroup.WQKV] = PrecisionSetting.BF16
    _layer0_opt.tensor_dtype_settings[TensorGroup.WO] = PrecisionSetting.BF16
    _layer0_opt.tensor_dtype_settings[TensorGroup.ACTIVATION] = PrecisionSetting.BF16
    _layer0_opt.tensor_dtype_settings[TensorGroup.KV_CACHE] = PrecisionSetting.BF16
    _force_mlp_weight_dtypes_bf16(model_args)
    _force_replicated_rmsnorm_for_layer_pcc(model_args)

    rope_params = text_cfg.rope_parameters or {}
    if not isinstance(rope_params, dict):
        rope_params = dict(rope_params)

    tt_ccl = TT_CCL(mesh_device)
    transformation_mats = {"decode": None, "prefill": None}

    tt_layer = TtMinistral3DecoderLayer(
        model_args,
        mesh_device,
        tt_ccl,
        dtype,
        meta_sd,
        layer_num=0,
        weight_cache_path=model_args.weight_cache_path(dtype),
        transformation_mats=transformation_mats,
        llama_4_scaling_beta=rope_params.get("llama_4_scaling_beta"),
        original_max_position_embeddings=rope_params.get("original_max_position_embeddings"),
    )

    cos, sin = position_embeddings
    cos_tt = ttnn.from_torch(
        cos.unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    sin_tt = ttnn.from_torch(
        sin.unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    rot_mats = [cos_tt, sin_tt]

    x_tt = ttnn.from_torch(
        x.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=model_args.get_residual_mem_config(Mode.PREFILL, None),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    pos_tt = ttnn.from_torch(
        position_ids.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = tt_layer(
        x_tt,
        current_pos=None,
        rot_mats_global=rot_mats,
        mode=Mode.PREFILL,
        position_ids=pos_tt,
    )

    # ``TtMinistral3DecoderLayer`` may all-gather TP shards so each device already holds
    # ``model_args.dim`` activations; ``ConcatMeshToTensor(dim=-1)`` would wrongly stack
    # ``num_devices`` copies (e.g. 1×4 → last dim 49152 vs expected 12288). A mesh tensor
    # still cannot use ``to_torch`` without a composer—read one device's buffer when replicated.
    out_last = int(tt_out.shape[-1])
    if out_last == model_args.dim:
        tt_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
    else:
        tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    tt_torch = tt_torch.reshape(ref_out.shape)

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(ref_out, tt_torch, pcc_required)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC (decoder layer): {pcc_message}")
    assert passing, f"PCC below {pcc_required}: {pcc_message}"
