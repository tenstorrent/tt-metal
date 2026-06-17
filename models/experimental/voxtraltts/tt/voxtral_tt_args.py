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

from models.common.utility_functions import is_blackhole
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


def voxtral_text_hf_aligned_optimizations(model_args):
    """Max accuracy + HF-aligned attention: BF16 weights, HiFi4 matmuls, fp32 SDPA softmax.

    Identical to ``voxtral_text_high_accuracy_optimizations`` except SDPA runs at
    ``HIFI4_FP32`` (``dst_full_sync_en=False``) so the attention softmax is promoted to
    fp32, matching the HF reference's fp32 softmax. This minimises last-hidden divergence
    across the autoregressive decode loop (which feeds the acoustic head), reducing the
    free-run acoustic-code flips that come from text-hidden drift rather than FSQ rounding.
    """
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
                OpGroup.SDPA_DECODE: MathFidelitySetting.HIFI4_FP32,
                OpGroup.SDPA_PREFILL: MathFidelitySetting.HIFI4_FP32,
                OpGroup.LI_O_DECODE: MathFidelitySetting.HIFI4,
                OpGroup.LI_O_PREFILL: MathFidelitySetting.HIFI4,
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
                allow_patterns=["*.safetensors"],
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
                # Decode SwiGLU gate SiLU fusion. Unfusing it (folding SiLU into the w1*w3 bfp8 mul)
                # is faster (w1 147->110us) BUT perturbs the gated activation enough to flip the
                # step-1 semantic near-tie in the free-run AR loop, collapsing free-run waveform PCC
                # 0.77 -> 0.40 (teacher-forced stays 0.9994; bisected — w2/wo 1D are innocent). So
                # SiLU is FUSED by default to preserve free-run quality. Unfuse with VOXTRAL_W1_FUSE_SILU=0.
                self.mlp_w1_fuse_silu_decode = os.environ.get("VOXTRAL_W1_FUSE_SILU", "1") == "1"
                # MLP w1/w3 (3072x9216) interleaved-weight 1D path (~88us vs DRAM-sharded ~110us).
                # OFF by default: the 1D w1 config cannot fuse SiLU (interleaved weight + DS fused-SiLU
                # config => "Only L1 buffers can have a CB"), so enabling it forces SiLU unfused, which
                # drops free-run PCC (see above). Enable with VOXTRAL_MLP_1D=1 (auto-unfuses SiLU).
                self.mlp_interleaved_weights = os.environ.get("VOXTRAL_MLP_1D", "0") == "1"
                # w1/w3 1D-interleaved weights are incompatible with fusing SiLU into the w1 DS
                # dram_matmul. If both are requested, the 1D path wins and SiLU is folded into the mul.
                if self.mlp_interleaved_weights and self.mlp_w1_fuse_silu_decode:
                    logger.info(
                        "VOXTRAL: mlp_interleaved_weights (w1/w3 1D) forces SiLU unfused; ignoring W1_FUSE_SILU."
                    )
                    self.mlp_w1_fuse_silu_decode = False
                # MLP w2 down-proj (9216x3072) interleaved-weight path: weights DRAM-interleaved
                # (mlp.py gate) + L1-interleaved w2_in/w2_out so the decode matmul is a fully
                # interleaved 1D mcast (~76us on 48 cores) instead of DRAM-sharded (~103us) — sweep
                # winner test_matmul_32x9216x3072_sweep (1.36x on w2). ON by default; disable with
                # VOXTRAL_MLP_FF2_1D=0.
                self.mlp_ff2_interleaved_weights = os.environ.get("VOXTRAL_MLP_FF2_1D", "1") == "1"
                # Attention output proj wo (4096x3072) interleaved-weight path: weights DRAM-interleaved
                # (attention.py gate) + L1-interleaved wo_in/wo_out so the decode matmul is a fully
                # interleaved 1D mcast (~38us on 48 cores) instead of DRAM-sharded (~51us) — sweep winner
                # test_matmul_32x4096x3072_sweep (1.32x on wo). ON by default; disable with VOXTRAL_ATTN_WO_1D=0.
                self.attn_wo_interleaved_weights = os.environ.get("VOXTRAL_ATTN_WO_1D", "1") == "1"
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

        def _voxtral_worker_shard_cap(self) -> ttnn.CoreGrid:
            """Worker-only core grid for width-sharded decode activations (excludes BH dispatch cols)."""
            cap = self.max_grid_size
            if is_blackhole():
                # COL dispatch occupies the eastern column(s); ``find_grid``'s 12-wide BH cap can
                # pick 2×12 for 24 K-tiles and trip writer_unary_sharded on dispatch cores.
                return ttnn.CoreGrid(y=cap.y, x=min(8, max(1, cap.x - 1)))
            return cap

        def _voxtral_find_worker_grid(self, k_tiles: int) -> ttnn.CoreGrid:
            worker_cap = self._voxtral_worker_shard_cap()
            max_rows = int(worker_cap.y)
            max_cols = int(worker_cap.x)
            max_cores = max_rows * max_cols
            target = min(32, max_cores)

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

            for cores in divisors_desc(k_tiles):
                if cores > max_cores:
                    continue
                for rows in range(min(cores, max_rows), 0, -1):
                    if cores % rows:
                        continue
                    cols = cores // rows
                    if cols <= max_cols:
                        return ttnn.CoreGrid(y=rows, x=cols)
            return ttnn.CoreGrid(y=1, x=1)

        def dram_shard_core_grid_for_k(self, k: int) -> ttnn.CoreGrid:
            """Worker-safe width-shard grid (BH COL dispatch columns excluded)."""
            if k % ttnn.TILE_SIZE != 0:
                return super().dram_shard_core_grid_for_k(k)
            grid = self._voxtral_find_worker_grid(k // ttnn.TILE_SIZE)
            cap_cores = self._voxtral_worker_shard_cap().x * self._voxtral_worker_shard_cap().y
            if grid.num_cores <= cap_cores:
                return grid
            return self._dram_shard_core_grid_capped(k, cap_cores)

        def _dram_shard_core_grid_capped(self, k: int, cap_cores: int) -> ttnn.CoreGrid:
            k_tiles = k // ttnn.TILE_SIZE
            worker_cap = self._voxtral_worker_shard_cap()
            cap_cores = min(cap_cores, worker_cap.x * worker_cap.y)
            divisors = [c for c in range(1, min(cap_cores, k_tiles) + 1) if k_tiles % c == 0]
            if not divisors:
                return ttnn.CoreGrid(y=1, x=1)
            target = min(32, cap_cores)
            divisors.sort(key=lambda c: abs(c - target))
            num_cores = divisors[0]
            for y in range(min(num_cores, worker_cap.y), 0, -1):
                if num_cores % y:
                    continue
                x = num_cores // y
                if x <= worker_cap.x:
                    return ttnn.CoreGrid(y=y, x=x)
            return ttnn.CoreGrid(y=1, x=min(num_cores, worker_cap.x))

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
            # Interleaved 1D w2 path: w2_out stays L1-interleaved (the MLP output reshard at the end
            # of mlp.py forward normalizes it to get_mlp_output_mem_config, so the residual is unaffected).
            if self._ff2_interleaved_decode(mode, prefetcher):
                return ttnn.L1_MEMORY_CONFIG
            return super().get_mlp_ff2_mem_config(mode, prefetcher)

        def _ff2_interleaved_decode(self, mode: Mode, prefetcher: Prefetcher = None) -> bool:
            """True when the opt-in interleaved-weight 1D w2 path applies (decode, single-device)."""
            return (
                getattr(self, "mlp_ff2_interleaved_weights", False)
                and mode == Mode.DECODE
                and prefetcher is None
                and not self.is_galaxy
            )

        def get_mlp_binary_mult_mem_config(self, mode: Mode):
            # Feed the 1D w2 matmul an L1-INTERLEAVED in0 (mcast_in0) instead of width-sharded.
            if self._ff2_interleaved_decode(mode, None):
                return ttnn.L1_MEMORY_CONFIG
            return super().get_mlp_binary_mult_mem_config(mode)

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

        def _wo_interleaved_decode(self, mode: Mode, prefetcher: Prefetcher = None) -> bool:
            """True when the opt-in interleaved-weight 1D wo path applies (decode, single-device)."""
            return (
                getattr(self, "attn_wo_interleaved_weights", False)
                and mode == Mode.DECODE
                and prefetcher is None
                and not self.is_galaxy
                and not getattr(self, "use_fused_all_gather_matmul", False)
            )

        def get_attn_wo_output_mem_config(self, mode: Mode, prefetcher: Prefetcher = None):
            if self._prefill_l1_mem(mode):
                return ttnn.L1_MEMORY_CONFIG
            # Interleaved 1D wo path: wo_out stays L1-interleaved (the attention output reshard at
            # attention.py:862 normalizes it to get_attn_dense_output_mem_config before the residual).
            if self._wo_interleaved_decode(mode, prefetcher):
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

        def get_mlp_ff1_3_prg_config(self, mode: Mode, seq_len: int = 1, prefetcher=None):
            """Voxtral DECODE: 1D mcast w1/w3 matmul (needs DRAM-interleaved weights), opt-in.

            in0/out stay width-sharded on ``mlp_core_grid`` (no reshard); only family DS->1D.
            PREFILL falls through to the base 2D matmul_config (unchanged).
            """
            if not getattr(self, "mlp_interleaved_weights", False) or prefetcher is not None or self.is_galaxy:
                return super().get_mlp_ff1_3_prg_config(mode, seq_len, prefetcher)
            if mode == Mode.PREFILL:
                # Base prefill config overrides per_core_N with a dram_shard_grid_width value tuned
                # for DRAM-SHARDED weights -> inconsistent with DRAM-interleaved weights (CB error).
                # Build a standard 2D matmul_config (per_core_N derived from the compute grid).
                return self.matmul_config(
                    m=min(seq_len, self.prefill_len_cutoff),
                    k=self.dim // self.cluster_shape[0],
                    n=self.hidden_dim // self.cluster_shape[1],
                    grid_size=self.mlp1_3_grid(seq_len),
                )
            grid = self.mlp_core_grid
            nc = grid.num_cores
            kt = self.dim // ttnn.TILE_SIZE
            nt = (self.hidden_dim // self.cluster_shape[1]) // ttnn.TILE_SIZE
            per_core_n = (nt + nc - 1) // nc
            in0_block_w = self.find_largest_divisor(kt // nc)
            out_subblock_w = next((w for w in (4, 3, 2, 1) if per_core_n % w == 0), 1)
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=out_subblock_w,
                per_core_M=1,
                per_core_N=per_core_n,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )

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

        def _voxtral_ff2_1d_grid(self, nt: int) -> tuple:
            """(gx, gy) for the 1D w2 matmul: ~48 cores dividing Nt, fitting the device grid.

            Prefers a wide row (large gx) to match the sweep winner (8x6 = 48 cores on Blackhole).
            """
            cap = self.max_grid_size
            for cores in (48, 40, 32, 24, 16, 12, 8):
                if nt % cores:
                    continue
                for gx in range(min(cores, cap.x), 0, -1):
                    if cores % gx == 0 and cores // gx <= cap.y:
                        return gx, cores // gx
            return cap.x, cap.y

        def get_mlp_ff2_prg_config(self, mode: Mode, seq_len: int = 1, prefetcher=None):
            """Voxtral DECODE: fully-interleaved 1D mcast w2 down-proj (9216x3072), opt-in.

            w2_in is fed L1-interleaved (get_mlp_binary_mult_mem_config) and w2_out stays
            L1-interleaved (get_mlp_ff2_mem_config) — the natural layout for mcast_in0. Weights are
            DRAM-interleaved (mlp.py gate). Sweep winner on Blackhole: 8x6 (48 cores), in0_block_w=4,
            per_core_N=2 -> ~76us vs ~103us DRAM-sharded (1.36x). PREFILL falls through unchanged.
            """
            if not self._ff2_interleaved_decode(mode, prefetcher):
                return super().get_mlp_ff2_prg_config(mode, seq_len, prefetcher)
            kt = (self.hidden_dim // self.cluster_shape[1]) // ttnn.TILE_SIZE  # 288
            nt = self.dim // ttnn.TILE_SIZE  # 96
            gx, gy = self._voxtral_ff2_1d_grid(nt)
            nc = gx * gy
            per_core_n = (nt + nc - 1) // nc
            in0_block_w = self.find_largest_divisor(kt, max_divisor=4)  # full-K per core (interleaved in0)
            out_subblock_w = next((w for w in (4, 3, 2, 1) if per_core_n % w == 0), 1)
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=out_subblock_w,
                per_core_M=1,
                per_core_N=per_core_n,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )

        def get_attn_wo_program_config(self, mode: Mode, seq_len: int = 1, prefetcher=None):
            """Voxtral DECODE: fully-interleaved 1D mcast attn output proj wo (4096x3072), opt-in.

            wo_in is fed L1-interleaved (get_attn_gather_users_mem_config) and wo_out stays
            L1-interleaved (get_attn_wo_output_mem_config). Weights DRAM-interleaved (attention.py gate).
            Sweep winner on Blackhole: 8x6 (48 cores), in0_block_w=4, per_core_N=2 -> ~38us vs ~51us
            DRAM-sharded (1.32x). PREFILL falls through unchanged.
            """
            if not self._wo_interleaved_decode(mode, prefetcher):
                return super().get_attn_wo_program_config(mode, seq_len, prefetcher)
            kt = ((self.n_heads * self.head_dim) // self.num_devices) // ttnn.TILE_SIZE  # 128
            nt = self.dim // ttnn.TILE_SIZE  # 96
            gx, gy = self._voxtral_ff2_1d_grid(nt)  # reuse the w2 grid picker (8x6 = 48 cores)
            nc = gx * gy
            per_core_n = (nt + nc - 1) // nc
            in0_block_w = self.find_largest_divisor(kt, max_divisor=4)  # full-K per core (interleaved in0)
            out_subblock_w = next((w for w in (4, 3, 2, 1) if per_core_n % w == 0), 1)
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=out_subblock_w,
                per_core_M=1,
                per_core_N=per_core_n,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )

    return VoxtralTTArgs
