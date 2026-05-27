# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Optional

import ttnn
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors_file

from models.experimental.voxtraltts.reference.voxtral_config import DEFAULT_VOXTRAL_MODEL, load_voxtral_config
from models.experimental.voxtraltts.tt.text_decoder_layer import (
    permute_voxtral_text_qk_for_hf_rope,
    remap_voxtral_text_state_dict,
)
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    MathFidelitySetting,
    ModelArgs,
    ModelOptimizations,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
)


def voxtral_text_default_optimizations(model_args):
    """Decode perf: BFP8 weights + HiFi2 decode matmuls; activations stay BF16 for E2E PCC."""
    opt = ModelOptimizations(
        {
            "TensorPrecision": {
                TensorGroup.FF1_FF3: PrecisionSetting.BFP8,
                TensorGroup.FF2: PrecisionSetting.BFP8,
                TensorGroup.WQKV: PrecisionSetting.BFP8,
                TensorGroup.KV_CACHE: PrecisionSetting.BFP8,
                TensorGroup.WO: PrecisionSetting.BFP8,
            },
            "OpFidelity": {
                OpGroup.LI_FF1_FF3: MathFidelitySetting.HIFI2_FP16,
                OpGroup.LI_FF2: MathFidelitySetting.HIFI2_FP16,
                OpGroup.LI_QKV_DECODE: MathFidelitySetting.HIFI2,
                OpGroup.LI_QKV_PREFILL: MathFidelitySetting.HIFI4,
                OpGroup.LI_O_DECODE: MathFidelitySetting.HIFI2,
                OpGroup.LI_O_PREFILL: MathFidelitySetting.HIFI2,
                OpGroup.SDPA_DECODE: MathFidelitySetting.HIFI2,
                OpGroup.SDPA_PREFILL: MathFidelitySetting.HIFI4,
            },
        }
    )
    return DecodersPrecision(model_args.n_layers, model_args.model_name, opt)


def voxtral_text_high_accuracy_optimizations(model_args):
    """BF16 weights + HiFi4 matmuls for text backbone (logits PCC tests)."""
    opt = ModelOptimizations(
        {
            "TensorPrecision": {
                TensorGroup.FF1_FF3: PrecisionSetting.BF16,
                TensorGroup.FF2: PrecisionSetting.BF16,
                TensorGroup.WQKV: PrecisionSetting.BF16,
                TensorGroup.KV_CACHE: PrecisionSetting.BF16,
                TensorGroup.WO: PrecisionSetting.BF16,
            },
            "OpFidelity": {
                OpGroup.LI_FF1_FF3: MathFidelitySetting.HIFI4,
                OpGroup.LI_FF2: MathFidelitySetting.HIFI4,
                OpGroup.LI_QKV_DECODE: MathFidelitySetting.HIFI4,
                OpGroup.LI_QKV_PREFILL: MathFidelitySetting.HIFI4,
                OpGroup.SDPA_DECODE: MathFidelitySetting.HIFI4,
                OpGroup.SDPA_PREFILL: MathFidelitySetting.HIFI4,
                OpGroup.LI_O_DECODE: MathFidelitySetting.HIFI4,
                OpGroup.LI_O_PREFILL: MathFidelitySetting.HIFI4,
            },
        }
    )
    return DecodersPrecision(model_args.n_layers, model_args.model_name, opt)


def voxtral_text_hf_matched_optimizations(model_args):
    """BFP8 weights + HiFi2 matmuls + HiFi4_FP32 SDPA — matches HF bf16 compute path end-to-end.

    BFP8 + HiFi2 approximates HF's bf16×bf16 matmul precision; HiFi4_FP32 for SDPA
    matches HF's fp32 softmax promotion (dst_full_sync_en=False). This combination
    minimises hidden-state divergence across the autoregressive decode loop for best
    end-to-end waveform PCC against the CPU reference.
    """
    opt = ModelOptimizations(
        {
            "TensorPrecision": {
                TensorGroup.FF1_FF3: PrecisionSetting.BFP8,
                TensorGroup.FF2: PrecisionSetting.BFP8,
                TensorGroup.WQKV: PrecisionSetting.BFP8,
                TensorGroup.KV_CACHE: PrecisionSetting.BFP8,
                TensorGroup.WO: PrecisionSetting.BFP8,
            },
            "OpFidelity": {
                OpGroup.LI_FF1_FF3: MathFidelitySetting.HIFI2_FP16,
                OpGroup.LI_FF2: MathFidelitySetting.HIFI2_FP16,
                OpGroup.LI_QKV_DECODE: MathFidelitySetting.HIFI2,
                OpGroup.LI_QKV_PREFILL: MathFidelitySetting.HIFI4,
                OpGroup.LI_O_DECODE: MathFidelitySetting.HIFI2,
                OpGroup.LI_O_PREFILL: MathFidelitySetting.HIFI2,
                OpGroup.SDPA_DECODE: MathFidelitySetting.HIFI4_FP32,
                OpGroup.SDPA_PREFILL: MathFidelitySetting.HIFI4_FP32,
            },
        }
    )
    return DecodersPrecision(model_args.n_layers, model_args.model_name, opt)


# Backward-compatible alias used by text-model PCC tests.
voxtral_text_logits_pcc_optimizations = voxtral_text_high_accuracy_optimizations


def _load_safetensors_state_dict(model_name_or_path: str) -> dict[str, torch.Tensor]:
    model_path = Path(model_name_or_path)
    if model_path.is_dir():
        safetensor_files = sorted(model_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")
    else:
        snapshot_dir = Path(
            snapshot_download(
                repo_id=model_name_or_path,
                local_files_only=os.getenv("CI") == "true",
            )
        )
        safetensor_files = sorted(snapshot_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors files found in snapshot for {model_name_or_path}")

    state_dict: dict[str, torch.Tensor] = {}
    for path in safetensor_files:
        state_dict.update(load_safetensors_file(str(path)))
    return state_dict


def get_VoxtralTTArgs(preloaded_state_dict: Optional[dict[str, torch.Tensor]] = None):
    class VoxtralTTArgs(ModelArgs):
        def __init__(self, *args, model_name_or_path: str = DEFAULT_VOXTRAL_MODEL, **kwargs):
            self._voxtral_model_name_or_path = model_name_or_path
            prev_hf_model = os.environ.get("HF_MODEL")
            os.environ["HF_MODEL"] = model_name_or_path
            try:
                kwargs.setdefault("use_hf_rope", True)
                super().__init__(*args, **kwargs)
                # Decode: fuse SiLU into w1 DRAM-sharded matmul (SwiGLU gate); mul becomes w1_silu * w3.
                self.mlp_w1_fuse_silu_decode = True
            finally:
                if prev_hf_model is None:
                    os.environ.pop("HF_MODEL", None)
                else:
                    os.environ["HF_MODEL"] = prev_hf_model

        def _set_hf_params(self, checkpoint_dir):
            cfg = load_voxtral_config(self._voxtral_model_name_or_path)
            config_dict = {
                "hidden_size": cfg.dim,
                "num_attention_heads": cfg.n_heads,
                "num_key_value_heads": cfg.n_kv_heads,
                "num_hidden_layers": cfg.n_layers,
                "rms_norm_eps": cfg.norm_eps,
                "vocab_size": cfg.vocab_size,
                "head_dim": cfg.head_dim,
                "intermediate_size": cfg.hidden_dim,
                "max_position_embeddings": cfg.max_position_embeddings,
                "rope_theta": cfg.rope_theta,
                "model_type": cfg.model_type,
                "tie_word_embeddings": cfg.tied_embeddings,
            }
            self._set_params_from_dict(config_dict)
            self.model_name = Path(self._voxtral_model_name_or_path).name or "Voxtral-4B-TTS-2603"
            self.state_dict_text_prefix = ""
            self.state_dict_vision_prefix = ""
            self.is_multimodal = False

        def create_tokenizer(self):
            return None

        def create_processor(self):
            return None

        def load_state_dict(self):
            if preloaded_state_dict is not None:
                state_dict = dict(preloaded_state_dict)
            else:
                state_dict = _load_safetensors_state_dict(self._voxtral_model_name_or_path)

            state_dict = remap_voxtral_text_state_dict(state_dict)
            state_dict = permute_voxtral_text_qk_for_hf_rope(
                state_dict,
                num_heads=self.n_heads,
                num_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim,
                hidden_size=self.dim,
            )

            if "output.weight" not in state_dict and "tok_embeddings.weight" in state_dict:
                state_dict["output.weight"] = state_dict["tok_embeddings.weight"]

            return state_dict

        def get_mlp_ff1_w1_prg_config(self, mode: Mode, seq_len: int = 1, prefetcher=None):
            """Voxtral decode: w1 matmul with fused SiLU (w3 uses plain ff1_3 config)."""
            if not getattr(self, "mlp_w1_fuse_silu_decode", False) or mode != Mode.DECODE:
                return self.get_mlp_ff1_3_prg_config(mode, seq_len, prefetcher)
            if prefetcher is not None or self.is_galaxy:
                return self.get_mlp_ff1_3_prg_config(mode, seq_len, prefetcher)
            return self.dram_matmul_config(
                m=self.tile_padded_batch_rows,
                k=self.dim,
                n=self.hidden_dim // self.cluster_shape[1],
                num_cores=self.mlp_core_grid.num_cores,
                fused_activation=ttnn.UnaryOpType.SILU,
            )

    return VoxtralTTArgs
