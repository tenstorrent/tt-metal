# SPDX-FileCopyrightText: (c) 2026 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
"""
Higgs Audio v2 ModelArgs.

We piggy-back on ``tt_transformers.tt.model_config.ModelArgs`` for the text
backbone (Llama-3.2-3B-Instruct), then layer Higgs-specific fields on top:
audio vocab, codebook geometry, audio special-token IDs, dual-FFN state-dict
naming.

Key design choice: pass ``dummy_weights=True`` so ``ModelArgs._set_hf_params``
reads its config from ``model_params/Llama-3.2-3B-Instruct/config.json``
(local, no network, no HF_MODEL lookup). The actual Higgs weights are loaded
later via tt.reference.load_higgs_v2_state_dict and handed directly to each
sub-module's constructor.

Verified: tt_metal/models/tt_transformers/tt/model_config.py:2861 reads
LOCAL_HF_PARAMS[self.model_name] when dummy_weights is True.
"""
from __future__ import annotations

import os

from loguru import logger

from models.demos.audio.higgs_audio_v2.tt.reference import HiggsAudioV2Config
from models.tt_transformers.tt.common import rope_scaling_model_factory
from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    MathFidelitySetting,
    ModelArgs,
    ModelOptimizations,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
)


def _higgs_accuracy_optimizations_settings() -> ModelOptimizations:
    """Custom accuracy preset ported from PR #40907.

    All linear weights (WQKV, WO, FF1/FF3, FF2) and KV cache promoted to BF16;
    every attention + MLP matmul + SDPA at HiFi4. Stock
    ``ModelOptimizations.accuracy("Llama-3.2-3B-Instruct")`` would instead use
    BFP8 attention + HiFi2 MLP because it routes Llama-3 through a memory-
    conservative branch — that's the noise floor we keep hitting at ~0.81
    teacher-forced active accuracy.
    """
    settings = {
        "TensorPrecision": {
            TensorGroup.WQKV: PrecisionSetting.BF16,
            TensorGroup.KV_CACHE: PrecisionSetting.BF16,
            TensorGroup.WO: PrecisionSetting.BF16,
            TensorGroup.FF1_FF3: PrecisionSetting.BF16,
            TensorGroup.FF2: PrecisionSetting.BF16,
        },
        "OpFidelity": {
            OpGroup.LI_QKV_DECODE: MathFidelitySetting.HIFI4,
            OpGroup.LI_QKV_PREFILL: MathFidelitySetting.HIFI4,
            OpGroup.SDPA_DECODE: MathFidelitySetting.HIFI4,
            OpGroup.SDPA_PREFILL: MathFidelitySetting.HIFI4,
            OpGroup.LI_O_DECODE: MathFidelitySetting.HIFI4,
            OpGroup.LI_O_PREFILL: MathFidelitySetting.HIFI4,
            OpGroup.LI_FF1_FF3: MathFidelitySetting.HIFI4,
            OpGroup.LI_FF2: MathFidelitySetting.HIFI4,
        },
    }
    inst = ModelOptimizations(settings)
    inst.__name__ = "accuracy"
    return inst


def _higgs_accuracy_decoders_precision(num_layers: int, model_name: str) -> DecodersPrecision:
    """Wrap the Higgs accuracy ModelOptimizations in a per-layer
    DecodersPrecision that ``tt_transformers`` Attention/MLP expect.
    """
    inst = DecodersPrecision(num_layers, model_name, _higgs_accuracy_optimizations_settings())
    inst.__name__ = "accuracy"
    return inst


# Pinned base text model. Matches the closed PR's HiggsModelArgs choice and
# matches the Higgs Audio v2 backbone (Llama-3.2-3B has 28 layers, 3072 hidden,
# 24 attn heads, 8 KV heads, 8192 intermediate — same as Higgs's text side).
BASE_TEXT_MODEL = "Llama-3.2-3B-Instruct"


class HiggsModelArgs(ModelArgs):
    """tt_transformers ModelArgs pre-configured for Higgs Audio v2 backbone.

    Construction order matters:
    1. We MUST set ``os.environ["HF_MODEL"]`` before ``super().__init__``
       because ``ModelArgs.__init__`` raises if HF_MODEL is unset.
       The value is arbitrary as long as the basename matches a key in
       ``LOCAL_HF_PARAMS`` (which it does for "meta-llama/Llama-3.2-3B-Instruct").
    2. We pass ``dummy_weights=True`` so ``_set_hf_params`` reads the local
       ``model_params/Llama-3.2-3B-Instruct/config.json`` instead of trying
       to hit the network or read CKPT_DIR.
    3. After super().__init__ completes, we patch in Higgs-specific fields
       parsed from the real Higgs config.json.
    """

    def __init__(
        self,
        mesh_device,
        higgs_config: HiggsAudioV2Config,
        max_batch_size: int = 1,
        max_seq_len: int = 1024,
        optimizations=None,
    ):
        # tt_transformers expects HF_MODEL env var. The basename is what
        # selects LOCAL_HF_PARAMS — keep the meta-llama/ prefix for parity
        # with tests/pipeline_reorg/*.yaml which use the same string.
        prior_hf_model = os.environ.get("HF_MODEL")
        os.environ["HF_MODEL"] = f"meta-llama/{BASE_TEXT_MODEL}"
        if optimizations is None:
            # n_layers/model_name aren't on self yet — ModelArgs reads from
            # the loaded Llama-3.2-3B config. Use known values for the
            # Llama-3.2-3B backbone.
            optimizations = _higgs_accuracy_decoders_precision(
                num_layers=higgs_config.num_hidden_layers,
                model_name=BASE_TEXT_MODEL,
            )
        try:
            super().__init__(
                mesh_device=mesh_device,
                instruct=True,
                dummy_weights=True,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                optimizations=optimizations,
                cache_hf=False,
                use_hf_rope=True,  # Higgs uses HF-style RoPE (llama3 NTK scaling)
            )
        finally:
            if prior_hf_model is None:
                os.environ.pop("HF_MODEL", None)
            else:
                os.environ["HF_MODEL"] = prior_hf_model

        # The Llama-3.2-3B-Instruct config.json uses the old transformers-4.x
        # top-level ``rope_theta``/``rope_scaling`` schema, but transformers 5.x
        # ``LlamaConfig`` only surfaces those when nested under
        # ``rope_parameters``. So ModelArgs sees ``self.rope_theta = None``
        # after _set_params_from_dict. We re-populate from the Higgs config
        # (which has rope_parameters populated) — and Higgs's rope params
        # genuinely differ from stock Llama-3.2-3B anyway (Higgs uses
        # original_max_position_embeddings=1024, factors 0.125/0.5; Llama uses
        # 8192, 1.0/4.0).
        scaling_params = dict(higgs_config.rope_scaling or {})
        self.rope_theta = float(higgs_config.rope_theta)
        self.original_max_context_len = scaling_params.get("original_max_position_embeddings")
        self.rope_scaling = (
            rope_scaling_model_factory(scaling_params, original_max_context_len=self.original_max_context_len)
            if scaling_params
            else None
        )

        # Higgs-specific fields, layered on top of the Llama-3.2-3B base.
        self.higgs_config = higgs_config
        self.audio_num_codebooks = higgs_config.num_codebooks
        self.audio_codebook_size = higgs_config.codebook_size
        self.audio_vocab_size = higgs_config.audio_vocab_size
        self.audio_bos_token_id = higgs_config.audio_bos_token_id
        self.audio_delay_token_id = higgs_config.audio_delay_token_id
        self.audio_token_id = higgs_config.audio_token_id
        self.audio_stream_bos_id = higgs_config.audio_stream_bos_id
        self.audio_stream_eos_id = higgs_config.audio_stream_eos_id

        # Sanity: the base Llama-3.2-3B config we just loaded had better
        # match the Higgs backbone shape, otherwise the state-dict load
        # below will silently miswire.
        assert (
            self.dim == higgs_config.hidden_size
        ), f"hidden_size mismatch: base Llama-3.2-3B={self.dim}, Higgs={higgs_config.hidden_size}"
        assert (
            self.n_layers == higgs_config.num_hidden_layers
        ), f"num_hidden_layers mismatch: base={self.n_layers}, Higgs={higgs_config.num_hidden_layers}"
        assert (
            self.n_heads == higgs_config.num_attention_heads
        ), f"num_attention_heads mismatch: base={self.n_heads}, Higgs={higgs_config.num_attention_heads}"
        assert (
            self.n_kv_heads == higgs_config.num_key_value_heads
        ), f"num_kv_heads mismatch: base={self.n_kv_heads}, Higgs={higgs_config.num_key_value_heads}"

        logger.info(
            f"HiggsModelArgs ready: dim={self.dim} layers={self.n_layers} "
            f"heads={self.n_heads}/{self.n_kv_heads} audio_vocab={self.audio_vocab_size}"
        )


__all__ = ["HiggsModelArgs", "BASE_TEXT_MODEL"]
