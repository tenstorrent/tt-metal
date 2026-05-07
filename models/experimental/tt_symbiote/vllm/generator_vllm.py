# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""vLLM adapter for Gemma4-31B via tt_symbiote.

Bridges the vLLM serving interface (initialize_vllm_model / prefill_forward /
decode_forward / allocate_kv_cache) to the HuggingFace Gemma-4 model whose
decoder layers, norms, embedding, and text-model wrapper have been replaced
with TTNN equivalents through tt_symbiote's module replacement machinery.

The module replacement pattern and warmup sequence follow test_gemma4.py.
The boilerplate (DIAG, watchdog, _to_host_tensor, prefill/decode forward,
warmup loops, KV cache hand-off) lives in
models.experimental.tt_symbiote.vllm.symbiote_adapter_base.SymbioteAdapterBase;
this module only carries the Gemma-4-specific load + replacement + dual-paged
KV cache build.
"""

import logging
from typing import Optional

import torch
from tqdm import tqdm

from models.experimental.tt_symbiote.vllm.symbiote_adapter_base import SymbioteAdapterBase

logger = logging.getLogger(__name__)


class SymbioteGemma4ForCausalLM(SymbioteAdapterBase):
    """vLLM-compatible adapter for Gemma4-31B running on TT hardware via tt_symbiote.

    Implements the four-method contract expected by TTModelLoader / TTModelRunner:
        - initialize_vllm_model (classmethod, called once at startup)
        - prefill_forward (variable-length prompt encoding)
        - decode_forward (single-token autoregressive step)
        - allocate_kv_cache (returns the opaque KV cache object)

    All four are inherited from SymbioteAdapterBase. This subclass only
    overrides _build_model_and_kv_cache (the model-specific HF load + module
    replacement + dual-paged KV cache allocation).
    """

    MODEL_KEY = "GEMMA4"
    WATCHDOG_PREFILL_KERNEL_HINT = "gemma4_attention"

    # Sequence lengths primed during warmup. The benchmark sweep uses ISL=128
    # today; 1024 is a forward-looking entry that mirrors the Gemma-3 T3K
    # default returned by get_warmup_prefill_supported_seq_lens().
    WARMUP_PREFILL_SEQ_LENS = (128, 1024)

    @classmethod
    def _build_model_and_kv_cache(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        tt_data_parallel,
        optimizations: Optional[str] = None,
        **kwargs,
    ):
        """Load HF Gemma4 and replace modules with TTNN equivalents.

        Follows the loading and replacement logic from test_gemma4.py: two-pass
        module replacement, weight preprocessing, device transfer, and dual
        paged KV cache allocation. Returns (model, kv_cache, model_device);
        the base then runs model.eval(), grad disable, and the final
        device-property patch.
        """
        from transformers import AutoModelForCausalLM

        from models.experimental.tt_symbiote.modules.gemma4_attention import (
            TTNNGemma4PagedAttentionKVCache,
        )
        from models.experimental.tt_symbiote.modules.gemma4_modules import (
            TTNNGemma4DecoderLayer,
            TTNNGemma4LMHead,
            TTNNGemma4ScaledEmbedding,
        )
        from models.experimental.tt_symbiote.models.gemma4_text import TTNNGemma4TextModel
        from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm
        from models.experimental.tt_symbiote.utils.device_management import set_device
        from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict

        model_name = hf_config._name_or_path

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Capture model_device BEFORE module replacement -- after replacement,
        # TTNN modules lack _parameters so model.parameters() would fail.
        model_device = next(model.parameters()).device

        # Gemma4 is multimodal: model.model is Gemma4Model (wrapper),
        # the actual text model lives at model.model.language_model.
        text_model = model.model.language_model
        decoder_class = text_model.layers[0].__class__
        norm_class = text_model.layers[0].input_layernorm.__class__
        embed_class = text_model.embed_tokens.__class__
        text_model_class = text_model.__class__

        # Exclude vision_tower and embed_vision modules from replacement
        # (their norms have incompatible dims for multi-device sharding).
        exclude_vision = {name for name, _ in model.named_modules() if "vision_tower" in name or "embed_vision" in name}

        # Pass 1: decoder layers, norms, embedding, and lm_head.
        nn_to_ttnn = {
            decoder_class: TTNNGemma4DecoderLayer,
            norm_class: TTNNDistributedRMSNorm,
            embed_class: TTNNGemma4ScaledEmbedding,
            torch.nn.Linear: TTNNGemma4LMHead,
        }
        modules = register_module_replacement_dict(
            model, nn_to_ttnn, model_config=None, exclude_replacement=exclude_vision
        )

        # Pass 2: text model wrapper (handles input_ids -> embedding on device,
        # iterates layers without ModuleList slicing which breaks TTNNModule).
        nn_to_ttnn_model = {text_model_class: TTNNGemma4TextModel}
        modules.update(register_module_replacement_dict(model, nn_to_ttnn_model, model_config=None))

        set_device(model, mesh_device)

        logger.info(f"Preprocessing {len(modules)} TTNN modules weights...")
        for name, mod in tqdm(modules.items(), desc="Preprocessing & moving weights"):
            mod.preprocess_weights()
            mod.move_weights_to_device()

        # Allocate dual paged-attention KV cache (sliding + global).
        text_config = model.config.text_config
        global_indices = {i for i, lt in enumerate(text_config.layer_types) if lt == "full_attention"}
        kv_cache = TTNNGemma4PagedAttentionKVCache(
            text_config=text_config,
            global_layer_indices=global_indices,
            device=mesh_device,
        )
        kv_cache.to_device(mesh_device)

        return model, kv_cache, model_device
