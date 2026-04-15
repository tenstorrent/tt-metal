# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for Gemma4 31B model with TTNN backend.

Gemma4 31B is a dense multimodal model with:
- 60 heterogeneous decoder layers (50 sliding-window + 10 global attention)
- Sliding attention: 32Q/16KV heads, head_dim=256, window=1024
- Global attention: 32Q/4KV heads, head_dim=512, K=V sharing, partial RoPE
- GeGLU FFN (intermediate_size=21504)
- V-norm (RMSNorm without learnable scale) on all layers
- Logit soft-capping (30.0)
"""

import os

import pytest
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
import transformers
from models.experimental.tt_symbiote.core.run_config import TracedRun

assert transformers.__version__.startswith(
    "5."
), "This test requires transformers version 5.0.0.dev0. Try: `pip install git+https://github.com/huggingface/transformers.git`"


MESH_DEVICE_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 200000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("max_new_tokens", [128], ids=["128tok"])
def test_gemma4(mesh_device, max_new_tokens):
    """Test Gemma4 31B model with TTNN acceleration.

    Gemma4 31B is a dense model with 60 heterogeneous decoder layers:
    - 50 sliding-window attention layers (16 KV heads, head_dim=256, window=1024)
    - 10 global attention layers (4 KV heads, head_dim=512, K=V sharing, partial RoPE)
    - GeGLU FFN on all layers
    - Layer pattern: 5 sliding then 1 global, repeated 10 times
    """

    model_name = "google/gemma-4-31B"
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # NOTE: Gemma4 is multimodal (Gemma4ForConditionalGeneration). If AutoModelForCausalLM
    # fails because the model is registered as a conditional-generation model, use instead:
    #   from transformers import AutoModelForImageTextToText, AutoProcessor
    #   model = AutoModelForImageTextToText.from_pretrained(model_name, ...)
    #   processor = AutoProcessor.from_pretrained(model_name, ...)
    # and construct text-only inputs via processor(text=..., return_tensors="pt").
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)

    # Phase 2: TTNN module replacements (follows Ling 3-pass pattern)
    from models.experimental.tt_symbiote.modules.gemma4_attention import (
        TTNNGemma4PagedAttentionKVCache,
    )
    from models.experimental.tt_symbiote.modules.gemma4_modules import (
        TTNNGemma4ScaledEmbedding,
        TTNNGemma4DecoderLayer,
        TTNNGemma4LMHead,
    )
    from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm
    from models.experimental.tt_symbiote.models.gemma4_text import TTNNGemma4TextModel

    # Get HF classes from the loaded model
    # Gemma4 is multimodal: model.model is Gemma4Model (wrapper), text model is at language_model
    text_model = model.model.language_model
    decoder_class = text_model.layers[0].__class__  # Gemma4TextDecoderLayer
    norm_class = text_model.layers[0].input_layernorm.__class__  # Gemma4RMSNorm
    embed_class = text_model.embed_tokens.__class__  # Gemma4TextScaledWordEmbedding
    text_model_class = text_model.__class__  # Gemma4TextModel

    print(f"Decoder class: {decoder_class.__name__}")
    print(f"Norm class: {norm_class.__name__}")
    print(f"Embed class: {embed_class.__name__}")

    # Gemma4 tokenizer does not have a chat_template set, so we manually
    # construct the prompt using Gemma4's turn tokens: <|turn> and <turn|>
    # The tokenizer auto-prepends <bos>, so we don't include it in the string.
    prompt = "<|turn>user\nWhat is your favorite condiment?<turn|>\n<|turn>model\n"
    # Use first parameter's device (model.device may not exist for multi-modal models)
    model_device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device)

    # Some tokenizers produce token_type_ids that the model does not expect
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    # Three-pass replacement matching Ling pattern.
    # Gemma4RMSNorm is shared between language_model and vision_tower, but
    # vision tower norms have incompatible dims for 8-device sharding.
    # Exclude all vision_tower and embed_vision modules from replacement.
    exclude_vision = {name for name, _ in model.named_modules() if "vision_tower" in name or "embed_vision" in name}

    # Pass 1: decoder layers, final norm, embedding
    nn_to_ttnn = {
        decoder_class: TTNNGemma4DecoderLayer,
        norm_class: TTNNDistributedRMSNorm,
        embed_class: TTNNGemma4ScaledEmbedding,
        torch.nn.Linear: TTNNGemma4LMHead,
    }
    modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=None, exclude_replacement=exclude_vision)

    # Pass 2: model wrapper — replaces Gemma4TextModel with TTNNGemma4TextModel
    # which handles input_ids→embedding conversion (trace-safe) and iterates
    # layers without ModuleList slicing.
    nn_to_ttnn_model = {
        text_model_class: TTNNGemma4TextModel,
    }
    modules.update(register_module_replacement_dict(model, nn_to_ttnn_model, model_config=None))

    # Pass 3: lm_head — replace the CPU nn.Linear (5120 -> 262144) with TTNN.
    # Done as a direct replacement since nn.Linear is too broad for the dict pattern.

    set_device(model, mesh_device)

    print(f"Preprocessing {len(modules)} TTNN modules weights...")
    for k, v in tqdm(modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()

    # Create paged KV cache for Gemma4 (dual cache: sliding + global)
    text_config = model.config.text_config
    global_indices = {i for i, lt in enumerate(text_config.layer_types) if lt == "full_attention"}
    kv_cache = TTNNGemma4PagedAttentionKVCache(
        text_config=text_config,
        global_layer_indices=global_indices,
        device=mesh_device,
    )
    kv_cache.to_device(mesh_device)

    print("Running inference...")
    model.eval()
    torch.set_grad_enabled(False)

    # HF generate() accesses self.device (a read-only property from ModuleUtilsMixin).
    # After TTNN replacement, the property may fail if no torch parameters remain reachable
    # in the default iteration order. Override with a concrete property.
    try:
        _ = model.device
    except (AttributeError, StopIteration):
        pass
    # Force-set device via class override to ensure HF generate() can resolve it.
    type(model).device = property(lambda self: model_device)

    # Warmup run
    outputs = model.generate(**inputs, max_new_tokens=2, past_key_values=kv_cache, use_cache=True)
    kv_cache.reset()

    # Warmup run with traceing enabled
    outputs = model.generate(**inputs, max_new_tokens=4, past_key_values=kv_cache, use_cache=True)
    kv_cache.reset()

    # Actual traced run
    DispatchManager.clear_timings()
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, past_key_values=kv_cache, use_cache=True)
    ttnn.synchronize_device(mesh_device)

    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])
    print(f"Gemma4 OUTPUT: {generated_text}")

    assert len(generated_text.strip()) > 0, "Generated output should not be empty"

    DispatchManager.save_stats_to_file("gemma4_timing_stats.csv")
    TracedRun.release_all()
