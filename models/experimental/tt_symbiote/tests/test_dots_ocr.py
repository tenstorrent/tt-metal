# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""E2E TTNN pipeline tests for dots.ocr (text-only and vision+text)."""

import os
import time

import pytest
import torch
from transformers import AutoTokenizer

import ttnn
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.models.dots_ocr import TTNNDotsOCRPipeline


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

DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"


def _resolve_model_path():
    """Resolve dots.ocr model path: env var > HF cache > model ID for auto-download."""
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(DOTS_OCR_MODEL_ID)
    except Exception:
        return DOTS_OCR_MODEL_ID


DOTS_OCR_LOCAL_PATH = _resolve_model_path()


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
def test_dots_ocr_text(mesh_device):
    """Test standalone TTNN pipeline for dots.ocr (text-only, no vision)."""

    pipeline = TTNNDotsOCRPipeline.from_hf_model(
        model_path=DOTS_OCR_LOCAL_PATH,
        device=mesh_device,
    )

    tokenizer = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    messages = [
        {"role": "user", "content": "What is optical character recognition and how does it work?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]

    pipeline.warmup(input_ids)

    DispatchManager.clear_timings()
    start_time = time.time()
    generated_ids = pipeline.generate(input_ids, max_new_tokens=128)
    ttnn.synchronize_device(mesh_device)
    end_time = time.time()

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Pipeline TEXT OUTPUT: {text}")

    total_time = end_time - start_time
    num_tokens = len(generated_ids)
    tokens_per_second = num_tokens / total_time
    ms_per_token = total_time / num_tokens * 1000

    print(f"\n{'='*60}")
    print(f"dots.ocr Pipeline Text Performance Summary")
    print(f"{'='*60}")
    print(f"Generated tokens:     {num_tokens}")
    print(f"Total time:           {total_time:.3f} s")
    print(f"Throughput:           {tokens_per_second:.1f} tok/s")
    print(f"Avg time per token:   {ms_per_token:.1f} ms/tok")
    print(f"{'='*60}\n")

    assert len(text.strip()) > 0, "Generated output should not be empty"

    DispatchManager.save_stats_to_file("dots_ocr_text_timing_stats.csv")
    pipeline.release()


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
    ],
)
def test_dots_ocr_vision(mesh_device, image_link):
    """Test standalone TTNN pipeline for dots.ocr with vision (image + text)."""
    pytest.importorskip("qwen_vl_utils")
    from qwen_vl_utils import process_vision_info

    pipeline = TTNNDotsOCRPipeline.from_hf_model(
        model_path=DOTS_OCR_LOCAL_PATH,
        device=mesh_device,
    )

    import json
    from transformers import AutoImageProcessor, AutoVideoProcessor, Qwen2_5_VLProcessor

    image_processor = AutoImageProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    _tokenizer = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    video_processor = AutoVideoProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    with open(os.path.join(DOTS_OCR_LOCAL_PATH, "chat_template.json")) as f:
        chat_template = json.load(f)["chat_template"]
    processor = Qwen2_5_VLProcessor(image_processor, _tokenizer, video_processor, chat_template=chat_template)
    processor.image_token = "<|imgpad|>"
    processor.image_token_id = 151665

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_link},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    image_grid_thw = inputs["image_grid_thw"]

    pipeline.warmup(input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    DispatchManager.clear_timings()
    start_time = time.time()
    generated_ids = pipeline.generate(
        input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=512,
    )
    ttnn.synchronize_device(mesh_device)
    end_time = time.time()

    decoded = processor.decode(generated_ids, skip_special_tokens=True)
    print(f"Pipeline VISION OUTPUT: {decoded}")

    total_time = end_time - start_time
    num_tokens = len(generated_ids)
    tokens_per_second = num_tokens / total_time
    ms_per_token = total_time / num_tokens * 1000

    print(f"\n{'='*60}")
    print(f"dots.ocr Pipeline Vision Performance Summary")
    print(f"{'='*60}")
    print(f"Generated tokens:     {num_tokens}")
    print(f"Total time:           {total_time:.3f} s")
    print(f"Throughput:           {tokens_per_second:.1f} tok/s")
    print(f"Avg time per token:   {ms_per_token:.1f} ms/tok")
    print(f"{'='*60}\n")

    assert len(decoded.strip()) > 0, "Generated output should not be empty"

    DispatchManager.save_stats_to_file("dots_ocr_vision_timing_stats.csv")
    pipeline.release()


def _assert_l1_resident(ttnn_tensor, label):
    memory_config = ttnn_tensor.memory_config() if hasattr(ttnn_tensor, "memory_config") else None
    if memory_config is not None:
        is_l1 = memory_config.buffer_type == ttnn.BufferType.L1
        assert is_l1, f"{label}: Tensor is not resident in L1 memory: memory_config={memory_config}"
        return
    loc = ttnn_tensor.get_layout() if hasattr(ttnn_tensor, "get_layout") else getattr(ttnn_tensor, "layout", None)
    if hasattr(ttnn_tensor, "device") and hasattr(ttnn_tensor.device, "is_l1_resident_tensor"):
        # Prefer device-aware check if available
        is_l1 = ttnn_tensor.device.is_l1_resident_tensor(ttnn_tensor)
    else:
        is_l1 = loc == "L1"
    assert is_l1, f"{label}: Tensor is not resident in L1 memory: layout={loc}"


# Fix imports for unit tests (and minimize test flakiness if possible)
from models.experimental.tt_symbiote.modules.dots_ocr_decoder_layer import TTNNDotsOCRDecoderLayer
from models.experimental.tt_symbiote.models.dots_ocr import _create_paged_kv_cache
from models.experimental.tt_symbiote.utils.device_management import set_device


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
def test_dots_ocr_decode_one_layer_l1_boundaries(mesh_device):
    """Exercise one decoder layer in decode mode and require L1 attn/MLP boundaries.

    Also runs the same input through the HF reference layer and compares the
    output via PCC, so any silent numerical regression in the optimized decode
    path is caught at the single-layer boundary.
    """
    from tests.ttnn.utils_for_testing import assert_with_pcc
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    model_config = AutoConfig.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    model_config.num_hidden_layers = 1
    hf_model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True).to(dtype=torch.bfloat16).eval()
    model_config = hf_model.config
    # Keep a reference to the HF layer + rotary embedding for the PCC check.
    # ``from_torch`` only reads weights, so we can safely re-use the same
    # in-memory layer after building the TTNN version.
    hf_layer = hf_model.model.layers[0]
    hf_rotary_emb = hf_model.model.rotary_emb
    layer = TTNNDotsOCRDecoderLayer.from_torch(hf_layer)
    layer._unique_name = "model.layers.0"
    layer.override_children_module_names()

    set_device(layer, mesh_device, register_forward_hook=False, dump_visualization=False)
    layer.preprocess_weights()
    layer.move_weights_to_device()

    paged_cache = _create_paged_kv_cache(model_config, mesh_device, batch_size=1)
    hidden_states_torch = torch.randn(1, 1, model_config.hidden_size, dtype=torch.bfloat16)
    hidden_states = ttnn.from_torch(
        hidden_states_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )
    cache_position = ttnn.from_torch(
        torch.zeros(1, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    seen_boundaries = {"attn": False, "mlp": False}
    original_attn_forward = layer.self_attn.forward
    original_mlp_forward = layer.mlp.forward

    def checked_attn_forward(*args, **kwargs):
        attn_input = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        _assert_l1_resident(attn_input, "attention input")
        output = original_attn_forward(*args, **kwargs)
        _assert_l1_resident(output[0], "attention output")
        seen_boundaries["attn"] = True
        return output

    def checked_mlp_forward(hidden_states):
        _assert_l1_resident(hidden_states, "MLP input")
        output = original_mlp_forward(hidden_states)
        _assert_l1_resident(output, "MLP output")
        seen_boundaries["mlp"] = True
        return output

    layer.self_attn.forward = checked_attn_forward
    layer.mlp.forward = checked_mlp_forward

    output = layer.forward(hidden_states, past_key_value=paged_cache, cache_position=cache_position)[0]
    ttnn.synchronize_device(mesh_device)

    _assert_l1_resident(output, "decoder layer output")
    assert seen_boundaries == {"attn": True, "mlp": True}

    # ------------------------------------------------------------------
    # PCC verification against the HF reference at cache_position=0.
    # ------------------------------------------------------------------
    # Decode at pos 0 with an empty past: position_ids = [[0]], no mask.
    position_ids = torch.zeros((1, 1), dtype=torch.long)
    cos, sin = hf_rotary_emb(hidden_states_torch, position_ids)
    torch_output = hf_layer(
        hidden_states_torch,
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        position_embeddings=(cos, sin),
    )[0]

    num_devices = int(mesh_device.get_num_devices()) if hasattr(mesh_device, "get_num_devices") else 1
    if num_devices > 1:
        # TP4 decode keeps hidden activations sharded across the hidden dim.
        ttnn_output_torch = ttnn.to_torch(
            output,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1),
        )
    else:
        ttnn_output_torch = ttnn.to_torch(output)
    ttnn_output_torch = ttnn_output_torch.to(torch.bfloat16).reshape(torch_output.shape)
    # Random-weight bf16 + paged-SDPA vs. eager attention adds some numerical
    # drift relative to a fp32 reference; 0.99 is the tight-but-safe bar that
    # still catches the regressions this test guards against (silent
    # layout/sharding bugs, wrong RoPE, dropped residual, etc.).
    assert_with_pcc(torch_output, ttnn_output_torch, pcc=0.99)


from models.experimental.tt_symbiote.modules.dots_ocr_decoder_layer import TTNNDotsOCRLayerStack


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
def test_dots_ocr_decode_full_decoder_l1_boundaries(mesh_device):
    """Exercise all decoder layers in decode mode and require L1 attn/MLP boundaries.

    Also runs the same input through all 28 HF reference layers and compares
    the final output via PCC. This is the regression gate for accumulated
    multi-layer numerical drift (e.g. pushing BFP4 onto too many layers) that
    the single-layer PCC test cannot detect.
    """
    from tests.ttnn.utils_for_testing import assert_with_pcc
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    model_config = AutoConfig.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    assert model_config.num_hidden_layers == 28, "dots.ocr decoder should have 28 layers"
    hf_model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True).to(dtype=torch.bfloat16).eval()

    # Compute the HF reference output BEFORE building TTNN layers. The
    # ``TTNNDotsOCRDecoderLayer.from_torch`` + ``preprocess_weights`` path
    # mutates the HF layers in-place (e.g. permutes QKV weights to the
    # KV-group-interleaved layout that ``nlp_create_qkv_heads`` expects),
    # so running HF inference after that step crashes inside
    # ``apply_rotary_pos_emb`` with a head_dim/num_heads shape mismatch.
    hidden_states_torch = torch.randn(1, 1, model_config.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.zeros((1, 1), dtype=torch.long)
    cos, sin = hf_model.model.rotary_emb(hidden_states_torch, position_ids)
    torch_hidden = hidden_states_torch
    for hf_layer in hf_model.model.layers:
        torch_hidden = hf_layer(
            torch_hidden,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=(cos, sin),
        )[0]
        # The HF Qwen2DecoderLayer can drop the seq dim on its output for
        # seq_len=1 (returns ``(batch, hidden)`` instead of ``(batch, seq,
        # hidden)``); re-add it so the next layer's q_proj reshape lands
        # the right number of heads on the right axis.
        if torch_hidden.dim() == 2:
            torch_hidden = torch_hidden.unsqueeze(1)
    torch_output = torch_hidden

    decoder_layers = []
    for layer_idx, hf_layer in enumerate(hf_model.model.layers):
        layer = TTNNDotsOCRDecoderLayer.from_torch(hf_layer)
        layer._unique_name = f"model.layers.{layer_idx}"
        layer.override_children_module_names()
        decoder_layers.append(layer)
    del hf_model

    decoder_stack = TTNNDotsOCRLayerStack(decoder_layers)
    decoder_stack._unique_name = "model.layer_stack"
    set_device(decoder_stack, mesh_device, register_forward_hook=False, dump_visualization=False)
    decoder_stack.preprocess_weights()
    decoder_stack.move_weights_to_device()

    paged_cache = _create_paged_kv_cache(model_config, mesh_device, batch_size=1)
    hidden_states = ttnn.from_torch(
        hidden_states_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )
    cache_position = ttnn.from_torch(
        torch.zeros(1, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    seen_boundaries = {layer_idx: {"attn": False, "mlp": False} for layer_idx in range(len(decoder_layers))}

    for layer_idx, layer in enumerate(decoder_layers):
        original_attn_forward = layer.self_attn.forward
        original_mlp_forward = layer.mlp.forward

        def checked_attn_forward(*args, _layer_idx=layer_idx, _original_forward=original_attn_forward, **kwargs):
            attn_input = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            _assert_l1_resident(attn_input, f"layer {_layer_idx} attention input")
            output = _original_forward(*args, **kwargs)
            _assert_l1_resident(output[0], f"layer {_layer_idx} attention output")
            seen_boundaries[_layer_idx]["attn"] = True
            return output

        def checked_mlp_forward(hidden_states, _layer_idx=layer_idx, _original_forward=original_mlp_forward):
            _assert_l1_resident(hidden_states, f"layer {_layer_idx} MLP input")
            output = _original_forward(hidden_states)
            _assert_l1_resident(output, f"layer {_layer_idx} MLP output")
            seen_boundaries[_layer_idx]["mlp"] = True
            return output

        layer.self_attn.forward = checked_attn_forward
        layer.mlp.forward = checked_mlp_forward

    output = decoder_stack.forward(hidden_states, past_key_value=paged_cache, cache_position=cache_position)
    ttnn.synchronize_device(mesh_device)

    _assert_l1_resident(output, "decoder stack output")
    assert all(boundaries == {"attn": True, "mlp": True} for boundaries in seen_boundaries.values())

    # ------------------------------------------------------------------
    # PCC verification across all 28 layers at cache_position=0.
    # ------------------------------------------------------------------
    # ``torch_output`` was computed at the top of the test, before any
    # TTNN setup, so the HF reference is unaffected by ``from_torch`` /
    # ``preprocess_weights`` in-place mutations on the HF layers.
    num_devices = int(mesh_device.get_num_devices()) if hasattr(mesh_device, "get_num_devices") else 1
    if num_devices > 1:
        # TP4 decode keeps hidden activations sharded across the hidden dim.
        ttnn_output_torch = ttnn.to_torch(
            output,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1),
        )
    else:
        ttnn_output_torch = ttnn.to_torch(output)
    ttnn_output_torch = ttnn_output_torch.to(torch.bfloat16).reshape(torch_output.shape)

    # 28-layer accumulated bf16 + BFP4-weight + paged-SDPA drift is larger
    # than the single-layer case; 0.95 is the tight-but-survivable bar that
    # still catches the failures this test guards against (silent layout/
    # sharding bugs, wrong RoPE, accumulated BFP4 corruption like the OCR
    # demo's dropped 'K' in tabular data).
    assert_with_pcc(torch_output, ttnn_output_torch, pcc=0.95)
