# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging
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
    Prefetcher,
    PrecisionSetting,
    TensorGroup,
)

logger = logging.getLogger(__name__)


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


def voxtral_text_fast_optimizations(model_args):
    """EXPERIMENTAL — BFP4 FFN weights + HiFi2 fidelity.

    WARNING: BFP4 weights cause a libtt_metal.so crash during inference for this model
    as of this commit.  Do NOT use in production.  Kept here for future investigation
    when BFP4 support is stabilised.

    When BFP4 is stable, BFP4 FF1/FF2/FF3 would halve the DRAM bandwidth vs BFP8 for
    the MLP projections (3072×9216 each), giving ~15-20% overall speedup.  Attention
    weights stay at BFP8.  LoFi fidelity on FFN degrades TTS quality (PCC < 0.1).
    """
    opt = ModelOptimizations(
        {
            "TensorPrecision": {
                TensorGroup.FF1_FF3: PrecisionSetting.BFP4,  # halves DRAM BW for FFN gate/up
                TensorGroup.FF2: PrecisionSetting.BFP4,  # halves DRAM BW for FFN down
                TensorGroup.WQKV: PrecisionSetting.BFP8,  # keep attn quality
                TensorGroup.KV_CACHE: PrecisionSetting.BFP8,
                TensorGroup.WO: PrecisionSetting.BFP8,  # keep attn output quality
            },
            "OpFidelity": {
                OpGroup.LI_FF1_FF3: MathFidelitySetting.HIFI2_FP16,  # same as default (no LoFi)
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
                # Prefill: L1 interleaved activations (Matmul/LayerNorm/SDPA Tracy labels; lower DRAM BW).
                # Disable with VOXTRAL_TEXT_PREFILL_L1=0 if prefill OOMs on very long sequences.
                self.prefill_activations_l1 = os.environ.get("VOXTRAL_TEXT_PREFILL_L1", "1") != "0"
                self._apply_voxtral_decode_mlp_dram_grid_overrides()
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

        def _voxtral_decode_width_dram_matmul_grid(self, k: int) -> ttnn.CoreGrid | None:
            """Pick a core grid for decode DRAM-sharded matmul when **M is one tile** (e.g. batch×seq → 32 rows).

            Geometry (decode down-proj example): ``A [M,K] @ B [K,N]`` with ``M=32`` → **1 tile in M**.
            Height / block sharding on M cannot split a single M-tile, so parallelism is **width-only**
            (shard along **N** / output columns). ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`` still
            requires ``k % (TILE_SIZE * num_cores) == 0``, i.e. ``num_cores`` divides ``K_tiles = k / 32``.
            So for ``K=4096`` (128 K-tiles) you cannot use **96** cores: 128 is not divisible by 96 — the
            ideal “one N-tile per core” (96 N-tiles) hits a **K-blocking** limit; use the **largest** valid
            core count (often **64** on an 8×8-capable mesh) instead.

            Returns a ``CoreGrid`` with ``y*x == num_cores`` fitting ``max_grid_size``, or ``None``.
            """
            tile = ttnn.TILE_SIZE
            if k % tile != 0:
                return None
            k_tiles = k // tile
            cap = self.max_grid_size
            max_cores = cap.x * cap.y

            def divisors_desc(n: int) -> list[int]:
                out: list[int] = []
                i = 1
                while i * i <= n:
                    if n % i == 0:
                        out.append(n // i)
                        if i != n // i:
                            out.append(i)
                    i += 1
                return sorted(out, reverse=True)

            for num_cores in divisors_desc(k_tiles):
                if num_cores > max_cores:
                    continue
                # Prefer larger y first so BH can pick 8×12 over 12×8 when both fit.
                for y in range(min(num_cores, cap.y), 0, -1):
                    if num_cores % y:
                        continue
                    x = num_cores // y
                    if x <= cap.x:
                        return ttnn.CoreGrid(y=y, x=x)
            return None

        def _apply_voxtral_decode_mlp_dram_grid_overrides(self) -> None:
            """Widen decode MLP DRAM matmul grids using **width-only** core counts (see ``_voxtral_decode_width_dram_matmul_grid``).

            Replaces the generic ``find_grid_k_n`` cap (8×8, gcd(K_tiles,N_tiles)) which often picks **32** cores
            even when **K_tiles** allows more width parallelism on single-tile-M decode.

            Disable with ``VOXTRAL_DECODE_WIDE_MLP_GRID=0``.
            """
            if os.environ.get("VOXTRAL_DECODE_WIDE_MLP_GRID", "1") == "0":
                return
            if self.is_galaxy or self.prefetcher is not None:
                return
            prev_mlp, prev_mlp2 = self.mlp_core_grid, self.mlp2_core_grid

            # w2 decode: k = hidden_dim / mesh column (see ``get_mlp_ff2_prg_config``).
            k_ff2 = self.hidden_dim // self.cluster_shape[1]
            g2 = self._voxtral_decode_width_dram_matmul_grid(k_ff2)
            if g2 is not None and g2.num_cores > self.mlp2_core_grid.num_cores:
                self.mlp2_core_grid = g2

            # w1 / w3 decode: k = full ``dim`` (see ``get_mlp_ff1_3_prg_config`` / fused w1 path).
            g1 = self._voxtral_decode_width_dram_matmul_grid(self.dim)
            if g1 is not None and g1.num_cores > self.mlp_core_grid.num_cores:
                self.mlp_core_grid = g1

            if self.mlp_core_grid != prev_mlp or self.mlp2_core_grid != prev_mlp2:
                logger.info(
                    "Voxtral decode width-sharded DRAM matmul grids (M=1 tile): mlp_core_grid %s → %s, "
                    "mlp2_core_grid %s → %s",
                    prev_mlp,
                    self.mlp_core_grid,
                    prev_mlp2,
                    self.mlp2_core_grid,
                )

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

        def _prefill_l1_mem(self, mode: Mode) -> bool:
            return bool(getattr(self, "prefill_activations_l1", False)) and mode == Mode.PREFILL

        def get_residual_mem_config(self, mode: Mode, prefetcher: Prefetcher = None):
            if self._prefill_l1_mem(mode):
                return ttnn.L1_MEMORY_CONFIG
            return super().get_residual_mem_config(mode, prefetcher)

        def get_mlp_input_mem_config(self, mode: Mode, prefetcher: Prefetcher = None):
            if self._prefill_l1_mem(mode):
                return ttnn.L1_MEMORY_CONFIG
            return super().get_mlp_input_mem_config(mode, prefetcher)

        def get_mlp_ff1_3_mem_config(self, mode: Mode, prefetcher: Prefetcher = None):
            if self._prefill_l1_mem(mode):
                return ttnn.L1_MEMORY_CONFIG
            return super().get_mlp_ff1_3_mem_config(mode, prefetcher)

        def get_mlp_ff2_mem_config(self, mode: Mode, prefetcher: Prefetcher = None):
            if self._prefill_l1_mem(mode):
                return ttnn.L1_MEMORY_CONFIG
            return super().get_mlp_ff2_mem_config(mode, prefetcher)

        def get_mlp_act_mem_config(self, mode: Mode):
            if self._prefill_l1_mem(mode):
                return ttnn.L1_MEMORY_CONFIG
            return super().get_mlp_act_mem_config(mode)

        def get_mlp_ff2_all_reduce_mem_config(self, mode: Mode, tensor: ttnn.Tensor):
            if self._prefill_l1_mem(mode):
                return ttnn.L1_MEMORY_CONFIG
            return super().get_mlp_ff2_all_reduce_mem_config(mode, tensor)

        def get_attn_qkv_mm_mem_config(self, mode: Mode, prefetcher: Prefetcher = None):
            if self._prefill_l1_mem(mode):
                return ttnn.L1_MEMORY_CONFIG
            return super().get_attn_qkv_mm_mem_config(mode, prefetcher)

        def get_attn_qkv_all_reduce_output_mem_config(
            self, mode: Mode, mesh_cols: int = 1, prefetcher: Prefetcher = None
        ):
            if self._prefill_l1_mem(mode):
                return ttnn.L1_MEMORY_CONFIG
            return super().get_attn_qkv_all_reduce_output_mem_config(mode, mesh_cols, prefetcher)

        def get_attn_create_head_input_mem_config(self, mode: Mode):
            if self._prefill_l1_mem(mode):
                return ttnn.L1_MEMORY_CONFIG
            return super().get_attn_create_head_input_mem_config(mode)

        def get_attn_create_head_output_mem_config(self, mode: Mode, prefetcher: Prefetcher = None):
            if self._prefill_l1_mem(mode):
                return ttnn.L1_MEMORY_CONFIG
            return super().get_attn_create_head_output_mem_config(mode, prefetcher)

        def get_attn_sdpa_output_mem_config(
            self, mode: Mode, batch_size_per_device_group: int = 1, prefetcher: Prefetcher = None
        ):
            if self._prefill_l1_mem(mode):
                return ttnn.L1_MEMORY_CONFIG
            return super().get_attn_sdpa_output_mem_config(mode, batch_size_per_device_group, prefetcher)

        def get_attn_concat_heads_output_mem_config(self, mode: Mode, prefetcher: Prefetcher = None):
            if self._prefill_l1_mem(mode):
                return ttnn.L1_MEMORY_CONFIG
            return super().get_attn_concat_heads_output_mem_config(mode, prefetcher)

        def get_attn_wo_output_mem_config(self, mode: Mode, prefetcher: Prefetcher = None):
            if self._prefill_l1_mem(mode):
                return ttnn.L1_MEMORY_CONFIG
            return super().get_attn_wo_output_mem_config(mode, prefetcher)

        def get_attn_dense_output_mem_config(self, mode: Mode, prefetcher: Prefetcher = None):
            if self._prefill_l1_mem(mode):
                return ttnn.L1_MEMORY_CONFIG
            return super().get_attn_dense_output_mem_config(mode, prefetcher)

        def get_attn_all_gather_output_mem_config(self, mode: Mode, prefetcher: Prefetcher = None):
            if self._prefill_l1_mem(mode):
                return ttnn.L1_MEMORY_CONFIG
            return super().get_attn_all_gather_output_mem_config(mode, prefetcher)

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
