# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: Hugging Face ``Ministral3Model`` vs ``TtMinistral3Model`` (Devstral-2 large stack, real Hub weights).

Loads **partial** tensors from the Hub (``model.embed_tokens``, ``model.norm``, ``model.layers.0`` only),
converts them to meta keys, and compares **prefill** last hidden states to a HF reference built from the
same tensors. ``ModelArgs`` is configured for **one decoder layer** (``num_hidden_layers=1``) so the test
never loads the full 123B checkpoint into host RAM; ``dummy_weights=True`` keeps :class:`ModelArgs` from
calling ``from_pretrained`` on the whole model.

``TtMinistral3Model`` all-gathers embedding activations to full hidden width before the first RMSNorm when
the mesh shards width. Prefill attention requires **``seq_len % 128 == 0``** (``tt_transformers`` fused
prefill), so this test uses ``128`` tokens like ``test_ministral3_decoder_layer``.
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
from transformers.models.ministral3.modeling_ministral3 import Ministral3Model

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_large.tt.tt_ministral3_model import TtMinistral3Model
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


def _ministral3_single_layer_hub_keys(layer: int = 0) -> list[str]:
    return ["model.embed_tokens.weight", "model.norm.weight", *_decoder_layer_hf_keys(layer)]


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
    # :func:`standardize_hf_keys` ties ``lm_head`` to ``model.embed_tokens`` when ``lm_head`` is missing:
    # it **copies** embed to ``lm_head.weight`` and **deletes** ``model.embed_tokens.weight``. The base
    # ``Ministral3Model`` / ``Embedding`` path still needs ``tok_embeddings`` after HF→meta mapping, so
    # keep a duplicate ``lm_head`` tensor (same trick as ``test_ministral3_decoder_layer`` callers that
    # include embed in a larger dict, or the old random-init model test).
    if "lm_head.weight" not in sd and "model.embed_tokens.weight" in sd:
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"].detach().clone()
    sd = standardize_hf_keys(sd)
    head_dim = getattr(text_cfg, "head_dim", None) or text_cfg.hidden_size // text_cfg.num_attention_heads
    return convert_hf_to_meta_no_qkv_permute(
        sd,
        head_dim=int(head_dim),
        n_heads=text_cfg.num_attention_heads,
        n_kv_heads=text_cfg.num_key_value_heads,
    )


def _set_hf_params_devstral_one_text_layer(self, checkpoint_dir: str):
    """Like :meth:`ModelArgs._set_hf_params`, but force ``num_hidden_layers=1`` for partial-weight PCC."""

    def merge_text_config(base_config: dict) -> dict:
        text_config = dict(base_config.get("text_config", {}) or {})
        text_config.update({k: v for k, v in base_config.items() if k not in ("text_config", "vision_config")})
        return text_config

    def merge_vision_config(base_config: dict) -> dict:
        vision_config = dict(base_config.get("vision_config", {}) or {})
        vision_config.update({k: v for k, v in base_config.items() if k not in ("text_config", "vision_config")})
        return vision_config

    self.trust_remote_code_hf = True
    src = self.LOCAL_HF_PARAMS[self.model_name] if self.dummy_weights else self.CKPT_DIR
    self.hf_config = AutoConfig.from_pretrained(
        src,
        trust_remote_code=True,
        local_files_only=os.getenv("CI") == "true",
    )
    config = self.hf_config.to_dict()
    if "text_config" in config or "vision_config" in config:
        merged_text_config = merge_text_config(config)
        merged_text_config["num_hidden_layers"] = 1
        lt = merged_text_config.get("layer_types")
        if isinstance(lt, (list, tuple)) and len(lt) > 1:
            merged_text_config["layer_types"] = list(lt)[:1]
        self._set_params_from_dict(merged_text_config)

        if "Mistral-Small-3.1-24B-Instruct-2503" in self.model_name:
            self._set_vision_params(config["vision_config"])
        else:
            if "vision_config" in config:
                merged_vision_config = merge_vision_config(config)
                self._set_vision_params({"vision_config": merged_vision_config})

        self.is_multimodal = "vision_config" in config or self.is_vision()
    else:
        config["num_hidden_layers"] = 1
        self._set_params_from_dict(config)
        self.is_multimodal = False

    if hasattr(self.hf_config, "text_config"):
        self.hf_config.text_config.num_hidden_layers = 1
        if hasattr(self.hf_config.text_config, "layer_types"):
            ltg = getattr(self.hf_config.text_config, "layer_types", None)
            if isinstance(ltg, (list, tuple)) and len(ltg) > 1:
                self.hf_config.text_config.layer_types = list(ltg)[:1]
    else:
        self.hf_config.num_hidden_layers = 1


def _force_mlp_weight_dtypes_bf16(model_args: ModelArgs) -> None:
    dec = model_args.model_config["DECODERS_OPTIMIZATIONS"]
    for conf in dec.decoder_optimizations.values():
        tp = conf._opt_settings["TensorPrecision"]
        tp[TensorGroup.FF1_FF3] = PrecisionSetting.BF16
        tp[TensorGroup.FF2] = PrecisionSetting.BF16


def _force_replicated_rmsnorm(model_args: ModelArgs) -> None:
    model_args.is_distributed_norm = lambda mode: False  # type: ignore[method-assign]


@pytest.fixture
def trust_remote_ministral(monkeypatch):
    from models.tt_transformers.tt import model_config as mc

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


@pytest.fixture
def devstral_one_text_layer_hf_params(monkeypatch):
    from models.tt_transformers.tt import model_config as mc

    monkeypatch.setattr(mc.ModelArgs, "_set_hf_params", _set_hf_params_devstral_one_text_layer)


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
def test_ministral3_model_pcc_devstral2_large_partial_weights_one_layer(
    mesh_device,
    seq_len,
    batch_size,
    monkeypatch,
    trust_remote_ministral,
    devstral2_123b_dummy_config_path,
    devstral_one_text_layer_hf_params,
):
    monkeypatch.setenv("HF_MODEL", DEVSTRAL2_LARGE_REPO_ID)
    monkeypatch.setenv("MAX_PREFILL_CHUNK_SIZE", "128")

    dtype = ttnn.bfloat16
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max(512, seq_len),
        dummy_weights=True,
        use_hf_rope=True,
        cache_hf=False,
    )

    text_cfg = _text_cfg_from_hf_cfg(model_args.hf_config)
    assert text_cfg.num_hidden_layers == 1, "fixture must trim to one layer for partial Hub weights"

    hub_keys = _ministral3_single_layer_hub_keys(0)
    try:
        hf_partial = _load_hf_tensors_for_keys(DEVSTRAL2_LARGE_REPO_ID, hub_keys)
    except Exception as exc:
        pytest.skip(f"Could not download model shard(s) from Hub: {exc}")

    for k in hub_keys:
        hf_partial[k] = _to_bf16_host_if_fp8(hf_partial[k])

    try:
        meta_sd = _build_meta_state_dict(hf_partial, text_cfg)
    except Exception as exc:
        pytest.skip(f"HF→meta conversion failed: {exc}")

    hf_sd: dict[str, torch.Tensor] = {}
    for k, v in hf_partial.items():
        short = k[len("model.") :] if k.startswith("model.") else k
        hf_sd[short] = v

    text_cfg._attn_implementation = "eager"
    hf_model = Ministral3Model(text_cfg).eval().to(torch.bfloat16)
    hf_model.load_state_dict(hf_sd, strict=True)

    _layer0_opt = model_args.decoders_optimizations.decoder_optimizations[0]
    _layer0_opt.tensor_dtype_settings[TensorGroup.WQKV] = PrecisionSetting.BF16
    _layer0_opt.tensor_dtype_settings[TensorGroup.WO] = PrecisionSetting.BF16
    _layer0_opt.tensor_dtype_settings[TensorGroup.ACTIVATION] = PrecisionSetting.BF16
    _layer0_opt.tensor_dtype_settings[TensorGroup.KV_CACHE] = PrecisionSetting.BF16
    _force_mlp_weight_dtypes_bf16(model_args)
    _force_replicated_rmsnorm(model_args)

    torch.manual_seed(42)
    gen = torch.Generator(device="cpu").manual_seed(42)
    input_ids = torch.randint(0, text_cfg.vocab_size, (batch_size, seq_len), dtype=torch.long, generator=gen)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    inputs_embeds = hf_model.embed_tokens(input_ids)
    causal_mask = create_causal_mask(
        config=text_cfg,
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )

    ref_out = hf_model(
        input_ids=input_ids,
        attention_mask=causal_mask,
        position_ids=position_ids,
        use_cache=False,
    ).last_hidden_state

    tt_ccl = TT_CCL(mesh_device)
    transformation_mats = {"decode": None, "prefill": None}
    tt_model = TtMinistral3Model(
        model_args,
        mesh_device,
        tt_ccl,
        dtype,
        meta_sd,
        weight_cache_path=model_args.weight_cache_path(dtype),
        transformation_mats=transformation_mats,
    )

    tokens_4d = input_ids.reshape(1, 1, 1, -1).to(torch.int32)
    input_ids_tt = ttnn.from_torch(
        tokens_4d,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = tt_model(
        input_ids=input_ids_tt,
        mode=Mode.PREFILL,
        batch_size=batch_size,
    ).last_hidden_state

    out_last = int(tt_out.shape[-1])
    if out_last == model_args.dim:
        tt_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
    else:
        tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    tt_torch = tt_torch.reshape(ref_out.shape)

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(ref_out, tt_torch, pcc_required)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC (Ministral3Model partial Hub weights): {pcc_message}")
    assert passing, f"PCC below {pcc_required}: {pcc_message}"
