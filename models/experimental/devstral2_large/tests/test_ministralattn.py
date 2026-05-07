# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: Hugging Face ``Ministral3Attention`` vs ``TtDevstral2LargeAttention`` on Devstral-2-123B shards.

Loads ``model.layers.0.self_attn.{q,k,v,o}_proj.weight`` via ``model.safetensors.index.json`` (one
or few shards), builds the meta dict used by TT (``standardize_hf_keys`` +
``convert_hf_to_meta_no_qkv_permute``), and compares prefill output against TT. Does not load the
full checkpoint or instantiate the full HF model.

Rotary cos/sin come from HF ``Ministral3RotaryEmbedding`` (same tables as production ``use_hf_rope``).
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
from transformers.models.ministral3.modeling_ministral3 import Ministral3Attention, Ministral3RotaryEmbedding

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_large.tt.tt_ministralattn import TtDevstral2LargeAttention
from models.experimental.devstral2_large.tt.tt_ministralrmsnorm import DEVSTRAL2_LARGE_L1_SMALL_SIZE
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta_no_qkv_permute, standardize_hf_keys
from models.tt_transformers.tt.model_config import ModelArgs, PrecisionSetting, TensorGroup

DEVSTRAL2_LARGE_REPO_ID = "mistralai/Devstral-2-123B-Instruct-2512"


def _text_cfg_from_hf_cfg(hf_cfg) -> Ministral3Config:
    inner = getattr(hf_cfg, "text_config", None)
    out = inner if inner is not None else hf_cfg
    if not isinstance(out, Ministral3Config):
        raise TypeError(f"Expected Ministral3Config, got {type(out)!r}")
    return out


def _attn_hf_keys(layer: int = 0) -> list[str]:
    p = f"model.layers.{layer}.self_attn"
    return [
        f"{p}.q_proj.weight",
        f"{p}.k_proj.weight",
        f"{p}.v_proj.weight",
        f"{p}.o_proj.weight",
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
def test_ministral3_attention_pcc_devstral2_large_partial_weights(
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
    # ``Ministral3Attention.forward`` dispatches via ``config._attn_implementation``; keep explicit
    # ``eager`` so the reference uses ``eager_attention_forward`` (matches TT prefill SDPA style).
    text_cfg._attn_implementation = "eager"
    attn_keys = _attn_hf_keys(0)
    try:
        hf_partial = _load_hf_tensors_for_keys(DEVSTRAL2_LARGE_REPO_ID, attn_keys)
    except Exception as exc:
        pytest.skip(f"Could not download attention shard(s) from Hub: {exc}")

    for k in attn_keys:
        hf_partial[k] = _to_bf16_host_if_fp8(hf_partial[k])

    try:
        meta_sd = _build_meta_state_dict(hf_partial, text_cfg)
    except Exception as exc:
        pytest.skip(f"HF→meta conversion failed: {exc}")

    ref_attn = Ministral3Attention(text_cfg, layer_idx=0).eval()
    assert isinstance(
        ref_attn, Ministral3Attention
    ), f"PCC reference must be HF Ministral3Attention; got {type(ref_attn).__module__}.{type(ref_attn).__name__}"
    ref_attn.q_proj.weight.data.copy_(hf_partial[attn_keys[0]])
    ref_attn.k_proj.weight.data.copy_(hf_partial[attn_keys[1]])
    ref_attn.v_proj.weight.data.copy_(hf_partial[attn_keys[2]])
    ref_attn.o_proj.weight.data.copy_(hf_partial[attn_keys[3]])
    # Default HF Linear weights are float32; activations are bf16 to match TT / ``torch_in``.
    ref_attn.to(dtype=torch.bfloat16)

    rotary = Ministral3RotaryEmbedding(text_cfg).eval()
    hidden_size = text_cfg.hidden_size
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    position_embeddings = rotary(x, position_ids=position_ids)
    causal_mask = create_causal_mask(
        config=text_cfg,
        inputs_embeds=x,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )

    ref_out, _ = ref_attn(
        x,
        position_embeddings=position_embeddings,
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=None,
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

    # Default decoder optimisations use BFP8 for ``WQKV`` / ``WO`` / ``KV_CACHE`` and implicit activations.
    # HF reference is BF16 throughout. We patch layer 0 **before** ``Attention`` construction.
    #
    # Critical: ``forward_prefill`` casts Q to ``activation_dtype`` but casts K/V to ``kv_cache_dtype``
    # (``keys_BKSD.dtype``). If only WQKV/WO/ACTIVATION are BF16 while ``KV_CACHE`` stays BFP8, SDPA sees
    # mixed-precision Q (BF16) vs K/V (BFP8) — PCC stays ~0.6–0.7 vs full BF16 HF.
    _layer0_opt = model_args.decoders_optimizations.decoder_optimizations[0]
    _layer0_opt.tensor_dtype_settings[TensorGroup.WQKV] = PrecisionSetting.BF16
    _layer0_opt.tensor_dtype_settings[TensorGroup.WO] = PrecisionSetting.BF16
    _layer0_opt.tensor_dtype_settings[TensorGroup.ACTIVATION] = PrecisionSetting.BF16
    _layer0_opt.tensor_dtype_settings[TensorGroup.KV_CACHE] = PrecisionSetting.BF16

    rope_params = text_cfg.rope_parameters or {}
    if not isinstance(rope_params, dict):
        rope_params = dict(rope_params)

    tt_ccl = TT_CCL(mesh_device)
    transformation_mats = {"decode": None, "prefill": None}

    tt_attn = TtDevstral2LargeAttention(
        mesh_device,
        tt_ccl,
        model_args,
        meta_sd,
        model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        transformation_mats=transformation_mats,
        configuration=model_args,
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
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    pos_tt = ttnn.from_torch(
        position_ids.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = tt_attn.forward_prefill(x_tt, rot_mats, position_ids=pos_tt)

    # Line mesh (e.g. 1x4): stitch device shards along the hidden dimension (same as devstral-small PCC tests).
    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    tt_torch = tt_torch.reshape(ref_out.shape)

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(ref_out, tt_torch, pcc_required)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC (attention): {pcc_message}")
    assert passing, f"PCC below {pcc_required}: {pcc_message}"
