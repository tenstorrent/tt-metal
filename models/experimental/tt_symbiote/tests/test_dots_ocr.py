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
from models.experimental.tt_symbiote.modules.dots_ocr_vision import TTNNDotsOCRVisionTower
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
        max_num_blocks=256,
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
@pytest.mark.parametrize(
    "image_link",
    [
        "https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/demo/demo_image1.jpg",
        "https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/assets/chart.png",
    ],
)
@pytest.mark.parametrize("use_real_image", [True, False], ids=["real_image", "random_embeds"])
def test_dots_ocr_n300_vision(mesh_device, image_link, use_real_image):
    """Test dots.ocr multimodal (image + text) on N300 with vision encoder on CPU."""
    pytest.importorskip("qwen_vl_utils")
    from qwen_vl_utils import process_vision_info

    model_name = DOTS_OCR_LOCAL_PATH

    print("Loading dots.ocr model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Save CPU embedding weight BEFORE module replacement
    cpu_embed_weight = model.model.embed_tokens.weight.data.clone()

    # Pass 0: Replace entire vision tower with TTNN module
    vision_tower_class = model.vision_tower.__class__
    modules_vision = register_module_replacement_dict(
        model, {vision_tower_class: TTNNDotsOCRVisionTower}, model_config=None
    )

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

    # Pass 2: Replace lm_head (no exclusion needed — vision tower is already TTNN)
    nn_to_ttnn2 = {
        nn.Linear: TTNNLinearIColShardedWRowSharded,
    }
    modules2 = register_module_replacement_dict(model, nn_to_ttnn2, model_config=None)

    # Pass 3: Replace Qwen2Model with TTNNDotsOCRModel
    qwen2_model_class = model.model.__class__
    nn_to_ttnn_model = {
        qwen2_model_class: TTNNDotsOCRModel,
    }
    modules_model = register_module_replacement_dict(model, nn_to_ttnn_model, model_config=None)

    assert isinstance(
        model.vision_tower, TTNNDotsOCRVisionTower
    ), f"Vision tower should be TTNNDotsOCRVisionTower, got {type(model.vision_tower)}"

    type(model).device = property(lambda self: torch.device("cpu"))

    # Monkey-patch forward to handle inputs_embeds-only prefill
    _orig_forward = model.forward
    from transformers import Qwen2ForCausalLM

    _parent_forward = Qwen2ForCausalLM.forward

    def _vision_forward(input_ids=None, logits_to_keep=0, **kwargs):
        if input_ids is None and kwargs.get("inputs_embeds") is not None:
            return _parent_forward(model, input_ids=input_ids, logits_to_keep=logits_to_keep, **kwargs)
        return _orig_forward(input_ids=input_ids, logits_to_keep=logits_to_keep, **kwargs)

    model.forward = _vision_forward

    # Prepare image input using proper processor pipeline
    # Construct Qwen2_5_VLProcessor directly — DotsVLProcessor omits video_processor
    # in its __init__, which crashes on transformers >=4.52 strict type checks.
    import json
    from transformers import AutoImageProcessor, AutoVideoProcessor, Qwen2_5_VLProcessor

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    video_processor = AutoVideoProcessor.from_pretrained(model_name)
    with open(os.path.join(model_name, "chat_template.json")) as f:
        chat_template = json.load(f)["chat_template"]
    processor = Qwen2_5_VLProcessor(image_processor, _tokenizer, video_processor, chat_template=chat_template)
    processor.image_token = "<|imgpad|>"
    processor.image_token_id = 151665

    IMAGE_URL = image_link

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE_URL},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    image_grid_thw = inputs["image_grid_thw"]

    # Set device and preprocess weights
    set_device(model, mesh_device, device_init=DeviceInit)

    all_modules = {**modules_vision, **modules, **modules2, **modules_model}
    print(f"Preprocessing {len(all_modules)} TTNN modules weights...")
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()

    paged_cache = create_paged_kv_cache(model.config, mesh_device, batch_size=1)

    model.eval()
    torch.set_grad_enabled(False)

    # Pre-compute fused vision + text embeddings on CPU
    with torch.no_grad():
        cpu_embeds = torch.nn.functional.embedding(input_ids, cpu_embed_weight)
        img_mask = input_ids == model.config.image_token_id

        if use_real_image:
            print("Running vision encoder on TTNN...")
            vision_embeddings = model.vision_tower(pixel_values, image_grid_thw)
        else:
            num_image_tokens = img_mask.sum().item()
            print(f"Using random embeddings for {num_image_tokens} image tokens...")
            vision_embeddings = torch.randn(num_image_tokens, model.config.hidden_size, dtype=torch.bfloat16)

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

    # Warmup run without trace
    outputs = model.generate(
        input_ids=input_ids,
        inputs_embeds=cpu_embeds,
        attention_mask=attention_mask,
        max_new_tokens=2,
        use_cache=True,
        past_key_values=paged_cache,
    )
    paged_cache.reset()
    TracedRun.release_all()
    # Trace capture run
    outputs = model.generate(
        input_ids=input_ids,
        inputs_embeds=cpu_embeds,
        attention_mask=attention_mask,
        max_new_tokens=4,
        use_cache=True,
        past_key_values=paged_cache,
    )
    paged_cache.reset()

    # Actual vision run (timed)
    print("Running vision + text inference...")
    DispatchManager.clear_timings()
    start_time = time.time()
    outputs = model.generate(
        input_ids=input_ids,
        inputs_embeds=cpu_embeds,
        attention_mask=attention_mask,
        max_new_tokens=512,
        use_cache=True,
        past_key_values=paged_cache,
    )
    ttnn.synchronize_device(mesh_device)
    end_time = time.time()

    decoded = processor.decode(outputs[0][input_ids.shape[-1] :], skip_special_tokens=True)
    print(f"dots.ocr Vision OUTPUT: {decoded}")

    total_time = end_time - start_time
    num_generated_tokens = outputs.shape[-1] - input_ids.shape[-1]
    prompt_tokens = input_ids.shape[-1]
    tokens_per_second = num_generated_tokens / total_time
    ms_per_token = total_time / num_generated_tokens * 1000

    print(f"\n{'='*60}")
    print(f"dots.ocr N300 Vision Performance Summary (TRACED)")
    print(f"{'='*60}")
    print(f"Prompt tokens:        {prompt_tokens}")
    print(f"Generated tokens:     {num_generated_tokens}")
    print(f"Total time:           {total_time:.3f} s")
    print(f"Throughput:           {tokens_per_second:.1f} tok/s")
    print(f"Avg time per token:   {ms_per_token:.1f} ms/tok")
    print(f"{'='*60}\n")

    assert len(decoded.strip()) > 0, "Generated output should not be empty"

    DispatchManager.save_stats_to_file("dots_ocr_n300_vision_timing_stats.csv")
    TracedRun.release_all()
