# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: HF ``Ministral3Model`` (text backbone) vs ``TtMinistral3Model`` for Devstral-2 Large.

Why this test is parameterized
------------------------------

Devstral-2 Large is 88 decoder layers, hidden=12288, intermediate=28672 → ~123 B params
(~1.385 B params per layer). On a T3K (8 × Wormhole-B0, ~12 GB DRAM per chip = ~96 GB
total useful DRAM) the per-precision weight footprint is:

    bf16 weights         : ~247 GB total  → ~30.9 GB/chip   (does NOT fit; pure BF16 OOMs)
    BFP8 weights         : ~131 GB total  → ~16.4 GB/chip   (does NOT fit)
    BFP4 FF + BFP8 attn  : ~83  GB total  → ~10.4 GB/chip   (fits, tight)
    BFP4 everywhere      : ~70  GB total  → ~8.7  GB/chip   (fits comfortably)

Two knobs let this test scale with hardware capacity rather than fail with OOM:

1. ``DEVSTRAL2_TT_LAYERS`` (default 8) - cap the number of decoder layers used
   both in the HF reference and in ``TtMinistral3Model`` for apples-to-apples PCC.
   Only this many layers' worth of safetensors shards are downloaded from the Hub,
   so host RAM stays bounded.

2. Precision is BFP4 (FF1/FF2/FF3) + BFP8 (WQKV/WO/KV_CACHE) by default - the
   only combination that lets all 88 layers fit on T3K. The bf16 overrides used
   by the partial-weights one-layer test are NOT applied here. Override with
   ``DEVSTRAL2_TT_PRECISION=bf16`` if you actually want bf16 weights for a tiny
   (~few-layer) run.

3. ``DEVSTRAL2_PCC_THRESHOLD`` (default 0.85 for BFP4, 0.95 for BF16) - PCC
   targets relax for lower-precision weights. Override with a float.

Opt-in for the full 88-layer attempt: ``DEVSTRAL2_TT_LAYERS=88``. Requires both
~250 GB free host RAM (HF reference is built in bf16) and the BFP4 FF + BFP8
attn precision combination to fit T3K DRAM.

Weights are read from the HF hub (``DEVSTRAL2_LARGE_REPO_ID``) using only the
shards that contain the layers we are testing.
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.integrations.finegrained_fp8 import Fp8Dequantize
from transformers.masking_utils import create_causal_mask
from transformers.models.ministral3.modeling_ministral3 import Ministral3Model

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_large.tests.test_ministral3_model import (
    DEVSTRAL2_LARGE_REPO_ID,
    _build_meta_state_dict,
    _decoder_layer_hf_keys,
    _force_replicated_rmsnorm,
    _load_hf_tensors_for_keys,
    _text_cfg_from_hf_cfg,
    _to_bf16_host_if_fp8,
)
from models.experimental.devstral2_large.tt.tt_ministral3_model import TtMinistral3Model
from models.experimental.devstral2_large.tt.tt_ministralrmsnorm import DEVSTRAL2_LARGE_L1_SMALL_SIZE
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    ModelArgs,
    PrecisionSetting,
    TensorGroup,
)

_ORIGINAL_FP8_DEQUANTIZE_ONE = Fp8Dequantize._dequantize_one


def _dequantize_one_compat(self, quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if scales.ndim == 0:
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if quantized.dtype == torch.int8 or (fp4_dtype is not None and quantized.dtype == fp4_dtype):
            quantized_fp32 = self._unpack_fp4(quantized)
        else:
            quantized_fp32 = quantized.to(torch.float32)
        out_dtype = scales.dtype if scales.dtype.is_floating_point and scales.element_size() >= 2 else torch.bfloat16
        scale = scales.to(torch.float32)
        return (quantized_fp32 * scale).to(out_dtype)
    return _ORIGINAL_FP8_DEQUANTIZE_ONE(self, quantized, scales)


Fp8Dequantize._dequantize_one = _dequantize_one_compat


def _set_hf_params_devstral_n_text_layers(n_layers: int):
    """Return an ``_set_hf_params`` override that trims ``num_hidden_layers`` to ``n_layers``."""

    def _set_hf_params(self, checkpoint_dir: str):  # noqa: ARG001 - signature mirrors ModelArgs._set_hf_params
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
            merged_text_config["num_hidden_layers"] = n_layers
            lt = merged_text_config.get("layer_types")
            if isinstance(lt, (list, tuple)) and len(lt) > n_layers:
                merged_text_config["layer_types"] = list(lt)[:n_layers]
            self._set_params_from_dict(merged_text_config)
            if "vision_config" in config:
                merged_vision_config = merge_vision_config(config)
                self._set_vision_params({"vision_config": merged_vision_config})
            self.is_multimodal = "vision_config" in config or self.is_vision()
        else:
            config["num_hidden_layers"] = n_layers
            self._set_params_from_dict(config)
            self.is_multimodal = False

        if hasattr(self.hf_config, "text_config"):
            self.hf_config.text_config.num_hidden_layers = n_layers
            ltg = getattr(self.hf_config.text_config, "layer_types", None)
            if isinstance(ltg, (list, tuple)) and len(ltg) > n_layers:
                self.hf_config.text_config.layer_types = list(ltg)[:n_layers]
        else:
            self.hf_config.num_hidden_layers = n_layers

    return _set_hf_params


def _parse_precision(name: str) -> str:
    name = (name or "").strip().lower()
    if name in ("", "bfp4", "bfp4_bfp8", "default"):
        return "bfp4"
    if name in ("bfp8",):
        return "bfp8"
    if name in ("bf16",):
        return "bf16"
    raise ValueError(f"Unknown DEVSTRAL2_TT_PRECISION={name!r}; expected one of bfp4, bfp8, bf16.")


def _apply_precision_overrides(model_args: ModelArgs, precision: str) -> None:
    """Set every decoder layer's weight precision to a single, fitting profile.

    The defaults from ``DecodersPrecision.accuracy`` for non-Llama/Mistral-7B models are BF16
    everywhere (see ``model_config.ModelOptimizations.accuracy``), which OOMs Devstral-2 Large
    on T3K. This forces a profile that actually fits.
    """
    if precision == "bf16":
        setting = {
            TensorGroup.FF1_FF3: PrecisionSetting.BF16,
            TensorGroup.FF2: PrecisionSetting.BF16,
            TensorGroup.WQKV: PrecisionSetting.BF16,
            TensorGroup.WO: PrecisionSetting.BF16,
            TensorGroup.KV_CACHE: PrecisionSetting.BF16,
            TensorGroup.ACTIVATION: PrecisionSetting.BF16,
        }
    elif precision == "bfp8":
        setting = {
            TensorGroup.FF1_FF3: PrecisionSetting.BFP8,
            TensorGroup.FF2: PrecisionSetting.BFP8,
            TensorGroup.WQKV: PrecisionSetting.BFP8,
            TensorGroup.WO: PrecisionSetting.BFP8,
            TensorGroup.KV_CACHE: PrecisionSetting.BFP8,
            TensorGroup.ACTIVATION: None,
        }
    else:  # bfp4
        setting = {
            TensorGroup.FF1_FF3: PrecisionSetting.BFP4,
            TensorGroup.FF2: PrecisionSetting.BFP4,
            TensorGroup.WQKV: PrecisionSetting.BFP8,
            TensorGroup.WO: PrecisionSetting.BFP8,
            TensorGroup.KV_CACHE: PrecisionSetting.BFP8,
            TensorGroup.ACTIVATION: None,
        }

    dec = model_args.model_config["DECODERS_OPTIMIZATIONS"]
    for conf in dec.decoder_optimizations.values():
        tp = conf._opt_settings["TensorPrecision"]
        for group, value in setting.items():
            tp[group] = value


def _default_pcc_for_precision(precision: str) -> float:
    return {"bf16": 0.95, "bfp8": 0.90, "bfp4": 0.85}[precision]


def _log_memory_budget(n_layers: int, precision: str, mesh_device) -> None:
    """Print a quick on-device weight footprint estimate so callers see the math."""
    bytes_per_p = {"bf16": 2.0, "bfp8": 1.0625, "bfp4": 0.5625}
    # ~1.385 B params per Ministral3 layer (Q/K/V/O + W1/W2/W3 + 2 RMSNorms)
    params_per_layer = 1.384e9
    embed_params = 131072 * 12288  # vocab × hidden
    n_dev = mesh_device.get_num_devices()
    if precision == "bfp4":
        # MLP (BFP4) ~= 1.057e9 params, attn (BFP8) ~= 0.327e9 params
        per_layer_bytes = 1.057e9 * bytes_per_p["bfp4"] + 0.327e9 * bytes_per_p["bfp8"]
    else:
        per_layer_bytes = params_per_layer * bytes_per_p[precision]
    weights_total = n_layers * per_layer_bytes + embed_params * 2  # embed stays bf16
    per_chip = weights_total / n_dev
    logger.info(
        f"Devstral2 Large weight footprint @ n_layers={n_layers}, precision={precision}: "
        f"{weights_total / 1e9:.2f} GB total / {n_dev} chips = {per_chip / 1e9:.2f} GB per chip "
        f"(Wormhole-B0 DRAM is ~12 GB; budget ~10.5 GB usable). "
        f"Sliding head-room shrinks with seq_len, KV cache, and trace region."
    )


@pytest.fixture
def trust_remote_ministral(monkeypatch):
    from models.tt_transformers.tt import model_config as mc

    short_name = DEVSTRAL2_LARGE_REPO_ID.strip("/").split("/")[-1]
    monkeypatch.setitem(mc.ModelArgs.LOCAL_HF_PARAMS, short_name, DEVSTRAL2_LARGE_REPO_ID)

    def _get_hf_model_cls(self):
        from transformers import AutoModelForCausalLM
        from transformers.models.auto.modeling_auto import AutoModelForImageTextToText

        if not self.is_multimodal:
            return AutoModelForCausalLM
        if type(self.hf_config) in AutoModelForImageTextToText._model_mapping:
            return AutoModelForImageTextToText
        raise ValueError(
            f"Test supports multimodal configs in AutoModelForImageTextToText only; got {type(self.hf_config)}"
        )

    monkeypatch.setattr(mc.ModelArgs, "get_hf_model_cls", _get_hf_model_cls)


@pytest.fixture
def devstral_n_text_layers(monkeypatch):
    """Trim the HF config to ``DEVSTRAL2_TT_LAYERS`` decoder layers so HF and TT agree."""
    n_layers = int(os.environ.get("DEVSTRAL2_TT_LAYERS", "8"))
    if n_layers < 1:
        pytest.skip("DEVSTRAL2_TT_LAYERS must be >= 1")
    from models.tt_transformers.tt import model_config as mc

    monkeypatch.setattr(mc.ModelArgs, "_set_hf_params", _set_hf_params_devstral_n_text_layers(n_layers))
    return n_layers


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
@pytest.mark.timeout(0)
def test_ministral3_model_pcc_devstral2_large(
    mesh_device,
    seq_len,
    batch_size,
    monkeypatch,
    trust_remote_ministral,
    devstral_n_text_layers,
):
    """Run ``DEVSTRAL2_TT_LAYERS`` (default 8) layers end-to-end with a fitting precision profile.

    Weights for the trimmed layer range are pulled from the Hub shard-by-shard (no full
    checkpoint load). HF reference and TtMinistral3Model both see the same N layers.
    """
    n_layers = devstral_n_text_layers
    precision = _parse_precision(os.environ.get("DEVSTRAL2_TT_PRECISION", "bfp4"))
    pcc_required = float(os.environ.get("DEVSTRAL2_PCC_THRESHOLD", _default_pcc_for_precision(precision)))

    monkeypatch.setenv("HF_MODEL", DEVSTRAL2_LARGE_REPO_ID)
    monkeypatch.setenv("MAX_PREFILL_CHUNK_SIZE", "128")

    dtype = ttnn.bfloat16
    if precision == "bfp4":
        optimizations = lambda ma: DecodersPrecision.performance(num_decoders=ma.n_layers, model_name=ma.model_name)
    else:
        optimizations = None  # accuracy default; per-tensor overrides applied below
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max(512, seq_len),
        dummy_weights=True,
        use_hf_rope=True,
        cache_hf=False,
        optimizations=optimizations,
    )
    assert (
        model_args.n_layers == n_layers
    ), f"_set_hf_params override should have trimmed to {n_layers} layers (got {model_args.n_layers})"

    _log_memory_budget(n_layers, precision, mesh_device)

    text_cfg = _text_cfg_from_hf_cfg(model_args.hf_config)

    # Download only the keys we need: embed + final norm + the N decoder layers.
    hub_keys: list[str] = ["model.embed_tokens.weight", "model.norm.weight"]
    for layer_idx in range(n_layers):
        hub_keys.extend(_decoder_layer_hf_keys(layer_idx))
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

    _apply_precision_overrides(model_args, precision)
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
    logger.info(f"HF reference output shape: {ref_out.shape}")

    # Free the HF reference before we upload TT weights; on T3K + N=8 this saves ~22 GB host RAM.
    del hf_model, inputs_embeds, causal_mask

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
    logger.info(f"TtMinistral3Model built: {model_args.n_layers} decoder layers, precision={precision}")

    tokens_4d = input_ids.reshape(1, 1, 1, -1).to(torch.int32)
    input_ids_tt = ttnn.from_torch(
        tokens_4d,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Do NOT pass position_ids as a TTNN tensor. TtMinistral3Model.forward immediately
    # calls ttnn.to_torch(position_ids).long() — that's a device→host read that triggers
    # a MeshCommandQueue::finish across all 8 chips and stalls for many minutes behind the
    # just-enqueued weight uploads. The same forward auto-builds contiguous positions on
    # host when position_ids is None (matches torch.arange(seq_len) for this prefill).
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

    passing, pcc_message = comp_pcc(ref_out, tt_torch, pcc_required)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(
        f"PCC (Ministral3Model vs TtMinistral3Model, {model_args.n_layers} layers, precision={precision}): "
        f"{pcc_message}"
    )
    assert passing, f"PCC below {pcc_required}: {pcc_message}"
