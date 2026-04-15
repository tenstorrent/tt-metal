# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""vLLM adapter for Gemma4-31B via tt_symbiote.

Bridges the vLLM serving interface (initialize_vllm_model / prefill_forward /
decode_forward / allocate_kv_cache) to the HuggingFace model whose decoder
layers, norms, embedding, and text-model wrapper have been replaced with TTNN
equivalents through tt_symbiote's module replacement machinery.

The module replacement pattern and warmup sequence follow test_gemma4.py.
"""

import logging
from typing import Optional

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SymbioteGemma4ForCausalLM:
    """vLLM-compatible adapter for Gemma4-31B running on TT hardware via tt_symbiote.

    Implements the four-method contract expected by TTModelLoader / TTModelRunner:
        - initialize_vllm_model (classmethod, called once at startup)
        - prefill_forward (variable-length prompt encoding)
        - decode_forward (single-token autoregressive step)
        - allocate_kv_cache (returns the opaque KV cache object)
    """

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_multimodal": False,
    }

    def __init__(self, model, mesh_device, kv_cache, hf_config):
        self.model = model
        self.mesh_device = mesh_device
        self.kv_cache = kv_cache
        self.hf_config = hf_config

    # ------------------------------------------------------------------
    # Class method: model loading & weight conversion
    # ------------------------------------------------------------------

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=4096,
        tt_data_parallel=1,
        optimizations: Optional[str] = None,
    ):
        """Load HF Gemma4, replace modules with TTNN equivalents, and warm up.

        Follows the exact loading and replacement logic from test_gemma4.py:
        two-pass module replacement, weight preprocessing, device transfer,
        dual paged KV cache allocation, and warmup via model.generate().
        """
        import transformers

        major = int(transformers.__version__.split(".")[0])
        assert major >= 5, f"Gemma4 requires transformers>=5.0.0, found {transformers.__version__}"

        from transformers import AutoModelForCausalLM, AutoTokenizer

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
        logger.info(f"Loading HF model: {model_name}")

        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)

        # Capture model_device BEFORE module replacement -- after replacement,
        # TTNN modules lack _parameters so model.parameters() would fail.
        model_device = next(model.parameters()).device

        # Prepare tokenizer and warmup inputs while the model is still pure PyTorch
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        warmup_prompt = "<|turn>user\nHello<turn|>\n<|turn>model\n"
        warmup_inputs = tokenizer(warmup_prompt, return_tensors="pt").to(model_device)
        if "token_type_ids" in warmup_inputs:
            del warmup_inputs["token_type_ids"]

        # Gemma4 is multimodal: model.model is Gemma4Model (wrapper),
        # the actual text model lives at model.model.language_model
        text_model = model.model.language_model
        decoder_class = text_model.layers[0].__class__
        norm_class = text_model.layers[0].input_layernorm.__class__
        embed_class = text_model.embed_tokens.__class__
        text_model_class = text_model.__class__

        # Exclude vision_tower and embed_vision modules from replacement
        # (their norms have incompatible dims for multi-device sharding)
        exclude_vision = {name for name, _ in model.named_modules() if "vision_tower" in name or "embed_vision" in name}

        # Pass 1: decoder layers, norms, embedding, and lm_head
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
        # iterates layers without ModuleList slicing which breaks TTNNModule)
        nn_to_ttnn_model = {text_model_class: TTNNGemma4TextModel}
        modules.update(register_module_replacement_dict(model, nn_to_ttnn_model, model_config=None))

        set_device(model, mesh_device)

        logger.info(f"Preprocessing {len(modules)} TTNN modules weights...")
        for name, mod in tqdm(modules.items(), desc="Preprocessing & moving weights"):
            mod.preprocess_weights()
            mod.move_weights_to_device()

        # Allocate dual paged-attention KV cache (sliding + global)
        text_config = model.config.text_config
        global_indices = {i for i, lt in enumerate(text_config.layer_types) if lt == "full_attention"}
        kv_cache = TTNNGemma4PagedAttentionKVCache(
            text_config=text_config,
            global_layer_indices=global_indices,
            device=mesh_device,
        )
        kv_cache.to_device(mesh_device)

        model.eval()
        torch.set_grad_enabled(False)

        # Patch model.device so HF generate() can resolve the device after
        # TTNN replacement removed all standard torch parameters.
        try:
            _ = model.device
        except (AttributeError, StopIteration):
            pass
        type(model).device = property(lambda self: model_device)

        # Warmup: two generate() passes to trigger trace capture on decoder layers.
        # Pass 1 (eager): short generation to warm up TTNN kernels
        # Pass 2 (traced): slightly longer to capture and verify traces
        logger.info("Running warmup forward passes for trace capture...")

        model.generate(**warmup_inputs, max_new_tokens=2, past_key_values=kv_cache, use_cache=True)
        kv_cache.reset()

        model.generate(**warmup_inputs, max_new_tokens=4, past_key_values=kv_cache, use_cache=True)
        kv_cache.reset()
        logger.info("Warmup complete.")

        return cls(model, mesh_device, kv_cache, hf_config)

    # ------------------------------------------------------------------
    # Prefill: variable-length prompt encoding
    # ------------------------------------------------------------------

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        """Encode a full prompt sequence and populate the KV cache.

        Args:
            tokens: input token IDs, shape [batch, seq_len]
            page_table: vLLM page table (unused; adapter manages its own KV cache)
            kv_cache: opaque KV cache object (our self.kv_cache, passed back by runner)
            prompt_lens: actual prompt lengths per batch element
            **kwargs: additional TTModelInput fields (absorbed)
        """
        batch_size = tokens.shape[0]
        seq_len = tokens.shape[1]

        input_ids = tokens.view(batch_size, seq_len)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        self.kv_cache.reset()

        with torch.no_grad():
            outputs = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self.kv_cache,
                use_cache=True,
            )

        logits = outputs.logits
        if hasattr(logits, "to_torch"):
            import ttnn

            logits = ttnn.to_torch(logits)

        return logits

    # ------------------------------------------------------------------
    # Decode: single-token autoregressive step
    # ------------------------------------------------------------------

    def decode_forward(self, tokens, start_pos, page_table, kv_cache, **kwargs):
        """Generate logits for one decode step.

        Args:
            tokens: current token IDs, shape [batch, 1]
            start_pos: cache position for each sequence in the batch
            page_table: vLLM page table (unused; adapter manages its own KV cache)
            kv_cache: opaque KV cache object
            **kwargs: additional TTModelInput fields (enable_trace, read_from_device, etc.)
        """
        batch_size = tokens.shape[0]
        input_ids = tokens.view(batch_size, 1)

        if isinstance(start_pos, int):
            cache_position = torch.tensor([start_pos], dtype=torch.long)
        else:
            cache_position = start_pos

        with torch.no_grad():
            outputs = self.model.forward(
                input_ids=input_ids,
                past_key_values=self.kv_cache,
                use_cache=True,
                cache_position=cache_position,
            )

        logits = outputs.logits
        if hasattr(logits, "to_torch"):
            import ttnn

            logits = ttnn.to_torch(logits)

        return logits

    # ------------------------------------------------------------------
    # Warmup stubs: TTNN kernels are warmed in initialize_vllm_model
    # via model.generate(), so TTModelRunner warmup is a no-op.
    # ------------------------------------------------------------------

    def warmup_model_prefill(self, enable_trace=False, **kwargs):
        logger.info(
            "warmup_model_prefill: TTNN kernels already warmed "
            "via model.generate() in initialize_vllm_model; skipping."
        )

    def warmup_model_decode(self, enable_trace=False, **kwargs):
        logger.info(
            "warmup_model_decode: TTNN kernels already warmed "
            "via model.generate() in initialize_vllm_model; skipping."
        )

    # ------------------------------------------------------------------
    # KV cache allocation (no-op: adapter owns the cache)
    # ------------------------------------------------------------------

    def allocate_kv_cache(self, kv_cache_shape=None, dtype=None, num_layers=None):
        """Return the pre-allocated dual paged-attention KV cache.

        TTModelRunner calls this once and passes the result into every
        prefill_forward / decode_forward invocation unchanged.
        """
        return self.kv_cache
