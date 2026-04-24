# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test for dots.ocr model on N300 (1x2 Wormhole mesh) with col-sharded residual flow.

Phase 6: Full col-sharded pipeline with traced execution and performance profiling.
"""

import os
import time

import pytest
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.experimental.tt_symbiote.core.run_config import (
    DispatchManager,
    TracedRun,
)
from models.experimental.tt_symbiote.models.dots_ocr import TTNNDotsOCRModel
from models.experimental.tt_symbiote.modules.dots_ocr_decoder_layer import TTNNDotsOCRDecoderLayer
from models.experimental.tt_symbiote.modules.embedding import TTNNEmbedding
from models.experimental.tt_symbiote.modules.attention import (
    PagedAttentionConfig,
    TTNNPagedAttentionKVCache,
)
from models.experimental.tt_symbiote.modules.linear import TTNNLinearIColShardedWRowSharded
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm
from models.experimental.tt_symbiote.utils.device_management import (
    DeviceInit,
    set_device,
)
from models.experimental.tt_symbiote.utils.module_replacement import (
    register_module_replacement_dict,
)


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

DOTS_OCR_LOCAL_PATH = "/home/salnahari/.cache/huggingface/hub/models--rednote-hilab--dots.ocr/snapshots/c0111ce6bc07803dbc267932ffef0ae3a51dc951"


def create_paged_kv_cache(model_config, device, batch_size=1):
    """Create a paged attention KV cache for dots.ocr.

    Args:
        model_config: Model configuration
        device: TTNN device
        batch_size: Batch size

    Returns:
        TTNNPagedAttentionKVCache instance
    """
    head_dim = getattr(model_config, "head_dim", model_config.hidden_size // model_config.num_attention_heads)
    config = PagedAttentionConfig(
        block_size=64,
        max_num_blocks=32,
        batch_size=batch_size,
    )
    return TTNNPagedAttentionKVCache(
        num_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=head_dim,
        config=config,
        device=None,
    ).to_device(device)


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 300000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
def test_dots_ocr_n300(mesh_device):
    """Test dots.ocr on N300 (1x2 mesh) with col-sharded residual flow."""
    model_name = DOTS_OCR_LOCAL_PATH

    print("Loading dots.ocr from local cache...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    decoder_class = model.model.layers[0].__class__
    norm_class = model.model.layers[0].input_layernorm.__class__
    embed_class = model.model.embed_tokens.__class__

    # Pass 1: Replace decoder layers, final norm, and embedding
    nn_to_ttnn = {
        decoder_class: TTNNDotsOCRDecoderLayer,
        norm_class: TTNNDistributedRMSNorm,
        embed_class: TTNNEmbedding,
    }
    modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)

    # Convert ModuleList to plain list — Qwen2Model.forward slices self.layers[:n],
    # which constructs a new ModuleList and fails isinstance(TTNNModule, nn.Module).
    layers_list = list(model.model.layers)
    del model.model._modules["layers"]
    model.model.layers = layers_list

    # Pass 2: Replace lm_head
    nn_to_ttnn2 = {
        nn.Linear: TTNNLinearIColShardedWRowSharded,
    }
    modules2 = register_module_replacement_dict(model, nn_to_ttnn2, model_config=None)

    # Pass 3: Replace Qwen2Model with model wrapper that uses LayerStack
    qwen2_model_class = model.model.__class__
    nn_to_ttnn_model = {
        qwen2_model_class: TTNNDotsOCRModel,
    }
    modules_model = register_module_replacement_dict(model, nn_to_ttnn_model, model_config=None)

    type(model).device = property(lambda self: torch.device("cpu"))

    messages = [
        {"role": "user", "content": "What is optical character recognition and how does it work?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    set_device(model, mesh_device, device_init=DeviceInit)

    all_modules = {**modules, **modules2, **modules_model}
    print(f"Preprocessing {len(all_modules)} TTNN modules weights...")
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()

    # Create paged KV cache
    paged_cache = create_paged_kv_cache(model.config, mesh_device, batch_size=1)

    print("Running inference with paged attention...")
    model.eval()
    torch.set_grad_enabled(False)

    # Warmup run without trace
    outputs = model.generate(**inputs, max_new_tokens=2, use_cache=True, past_key_values=paged_cache)
    paged_cache.reset()
    # Actual run with trace
    outputs = model.generate(**inputs, max_new_tokens=4, use_cache=True, past_key_values=paged_cache)
    paged_cache.reset()

    DispatchManager.clear_timings()
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True, past_key_values=paged_cache)
    ttnn.synchronize_device(mesh_device)
    end_time = time.time()

    decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])
    print(f"dots.ocr N300 OUTPUT: {decoded}")

    total_time = end_time - start_time
    num_generated_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
    prompt_tokens = inputs["input_ids"].shape[-1]
    tokens_per_second = num_generated_tokens / total_time
    ms_per_token = total_time / num_generated_tokens * 1000

    print(f"\n{'='*60}")
    print(f"dots.ocr N300 Performance Summary (TRACED)")
    print(f"{'='*60}")
    print(f"Prompt tokens:        {prompt_tokens}")
    print(f"Generated tokens:     {num_generated_tokens}")
    print(f"Total time:           {total_time:.3f} s")
    print(f"Throughput:           {tokens_per_second:.1f} tok/s")
    print(f"Avg time per token:   {ms_per_token:.1f} ms/tok")
    print(f"{'='*60}\n")

    assert len(decoded.strip()) > 0, "Generated output should not be empty"

    DispatchManager.save_stats_to_file("dots_ocr_n300_timing_stats.csv")
    TracedRun.release_all()


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 300000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
def test_dots_ocr_n300_vision(mesh_device):
    """Test dots.ocr multimodal (image + text) on N300 with vision encoder on CPU."""
    from PIL import Image
    from transformers import Qwen2VLImageProcessor

    model_name = DOTS_OCR_LOCAL_PATH

    print("Loading dots.ocr model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Save CPU embedding weight BEFORE module replacement
    cpu_embed_weight = model.model.embed_tokens.weight.data.clone()

    # Build exclusion set: protect ALL vision_tower and merger modules from nn.Linear replacement
    vision_exclude = set()
    for name, _ in model.named_modules():
        if name.startswith("vision_tower"):
            vision_exclude.add(name)

    decoder_class = model.model.layers[0].__class__
    norm_class = model.model.layers[0].input_layernorm.__class__
    embed_class = model.model.embed_tokens.__class__

    # Pass 1: Replace decoder layers, final norm, and embedding (no exclusion needed — vision uses different classes)
    nn_to_ttnn = {
        decoder_class: TTNNDotsOCRDecoderLayer,
        norm_class: TTNNDistributedRMSNorm,
        embed_class: TTNNEmbedding,
    }
    modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)

    # Convert ModuleList to plain list (same as text-only test)
    layers_list = list(model.model.layers)
    del model.model._modules["layers"]
    model.model.layers = layers_list

    # Pass 2: Replace lm_head — EXCLUDE vision_tower linears
    nn_to_ttnn2 = {
        nn.Linear: TTNNLinearIColShardedWRowSharded,
    }
    modules2 = register_module_replacement_dict(
        model, nn_to_ttnn2, model_config=None, exclude_replacement=vision_exclude
    )

    # Pass 3: Replace Qwen2Model with TTNNDotsOCRModel
    qwen2_model_class = model.model.__class__
    nn_to_ttnn_model = {
        qwen2_model_class: TTNNDotsOCRModel,
    }
    modules_model = register_module_replacement_dict(model, nn_to_ttnn_model, model_config=None)

    # Verify vision encoder was NOT replaced
    assert isinstance(
        model.vision_tower.blocks[0].attn.qkv, nn.Linear
    ), "Vision encoder qkv was incorrectly replaced! Should remain nn.Linear."
    assert isinstance(
        model.vision_tower.merger.mlp[0], nn.Linear
    ), "PatchMerger MLP linear was incorrectly replaced! Should remain nn.Linear."

    type(model).device = property(lambda self: torch.device("cpu"))

    # Monkey-patch forward to handle inputs_embeds-only prefill
    # DotsOCRForCausalLM.forward() asserts len(input_ids) >= 1, which crashes
    # when HF generate() passes input_ids=None with inputs_embeds
    _orig_forward = model.forward
    # trust_remote_code can duplicate DotsOCRForCausalLM in the MRO, so we
    # walk past all copies and grab Qwen2ForCausalLM.forward directly.
    from transformers import Qwen2ForCausalLM

    _parent_forward = Qwen2ForCausalLM.forward

    def _vision_forward(input_ids=None, **kwargs):
        if input_ids is None and kwargs.get("inputs_embeds") is not None:
            return _parent_forward(model, input_ids=input_ids, **kwargs)
        return _orig_forward(input_ids=input_ids, **kwargs)

    model.forward = _vision_forward

    # Prepare image input
    img_proc = Qwen2VLImageProcessor.from_pretrained(model_name)

    # Use a small synthetic test image (448x448) for initial bring-up
    import numpy as np

    test_image = Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8))

    image_result = img_proc(images=[test_image], return_tensors="pt")
    pixel_values = image_result["pixel_values"].to(torch.bfloat16)
    image_grid_thw = image_result["image_grid_thw"]

    # Calculate vision token count after spatial merge (2x2)
    t, h, w = image_grid_thw[0].tolist()
    num_vision_tokens = int(t * (h // 2) * (w // 2))

    # Build prompt with correct number of <|imgpad|> tokens
    imgpad = "<|imgpad|>"
    vision_placeholder = "<|vision_start|>" + imgpad * num_vision_tokens + "<|vision_end|>"
    prompt = f"<|im_start|>user\n{vision_placeholder}Describe this image.<|im_end|>\n<|im_start|>assistant\n"

    text_inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = text_inputs["input_ids"]

    actual_img_tokens = (input_ids == model.config.image_token_id).sum().item()
    assert (
        actual_img_tokens == num_vision_tokens
    ), f"Token count mismatch: expected {num_vision_tokens}, got {actual_img_tokens}"

    attention_mask = text_inputs.get("attention_mask")
    if "token_type_ids" in text_inputs:
        del text_inputs["token_type_ids"]

    # Set device and preprocess weights
    set_device(model, mesh_device, device_init=DeviceInit)

    all_modules = {**modules, **modules2, **modules_model}
    print(f"Preprocessing {len(all_modules)} TTNN modules weights...")
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()

    paged_cache = create_paged_kv_cache(model.config, mesh_device, batch_size=1)

    model.eval()
    torch.set_grad_enabled(False)

    # Pre-compute fused vision + text embeddings on CPU
    print("Running vision encoder on CPU...")
    with torch.no_grad():
        cpu_embeds = torch.nn.functional.embedding(input_ids, cpu_embed_weight)

        img_mask = input_ids == model.config.image_token_id
        vision_embeddings = model.vision_tower(pixel_values, image_grid_thw)

        true_indices = torch.nonzero(img_mask).squeeze()
        if len(true_indices) > vision_embeddings.size(0):
            true_indices = true_indices[: vision_embeddings.size(0)]
            new_img_mask = torch.zeros_like(img_mask)
            new_img_mask[true_indices[:, 0], true_indices[:, 1]] = True
            img_mask = new_img_mask

        cpu_embeds = cpu_embeds.masked_scatter(
            img_mask.unsqueeze(-1).expand_as(cpu_embeds),
            vision_embeddings.type(cpu_embeds.dtype),
        )

    print(f"Vision encoding done. Fused embeds shape: {cpu_embeds.shape}")

    # Generate: prefill uses pre-computed inputs_embeds, decode uses TTNNEmbedding via input_ids
    print("Running vision + text inference...")
    outputs = model.generate(
        input_ids=input_ids,
        inputs_embeds=cpu_embeds,
        attention_mask=attention_mask,
        max_new_tokens=64,
        use_cache=True,
        past_key_values=paged_cache,
    )
    ttnn.synchronize_device(mesh_device)

    decoded = tokenizer.decode(outputs[0][input_ids.shape[-1] :])
    print(f"dots.ocr Vision OUTPUT: {decoded}")

    assert len(decoded.strip()) > 0, "Generated output should not be empty"

    TracedRun.release_all()
