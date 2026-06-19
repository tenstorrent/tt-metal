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

# from models.experimental.tt_symbiote.models.dots_ocr import TTNNDotsOCRPipeline
from models.experimental.tt_symbiote.models.dots_ocr import TTNNDotsOCRPipeline, _create_paged_kv_cache
from models.experimental.tt_symbiote.modules.dots_ocr_decoder_layer import (
    TTNNDotsOCRDecoderLayer,
    TTNNDotsOCRLayerStack,
)
from models.experimental.tt_symbiote.modules.dots_ocr_vision import TTNNDotsOCRVisionTower
from models.experimental.tt_symbiote.utils.device_management import set_device


MESH_DEVICE_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (2, 4),
    "T3K_DP8": (8, 1),
    "TG": (8, 4),
    "P100": (1, 1),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}

DOTS_OCR_DP_MESH_DEVICE_MAP = {
    "N300": (2, 1),
    "T3K": (8, 1),
}

DOTS_OCR_DP2_TP4_MESH_DEVICE_MAP = {
    "T3K": (2, 4),
}

DOTS_OCR_DP2_TP2_MESH_DEVICE_MAP = {
    "T3K": (2, 2),
}

DOTS_OCR_LAYOUT_PROMPT = "output all the text present in the image"


def _dots_ocr_parallelism_mode() -> str:
    return os.environ.get("DOTS_OCR_PARALLELISM", "").upper()


def _mesh_device_map_get(mapping, mesh_device, default=None):
    if mesh_device is None:
        return default
    for key, value in mapping.items():
        if key.upper() == mesh_device.upper():
            return value
    return default


def _assert_l1_resident(tensor, name: str):
    assert isinstance(tensor, ttnn.Tensor), f"{name} should be a TTNN tensor"
    assert tensor.memory_config().buffer_type == ttnn.BufferType.L1, f"{name} should reside in L1"


def _resolve_mesh_device_shape():
    mesh_device = os.environ.get("MESH_DEVICE")
    parallelism = _dots_ocr_parallelism_mode()
    if parallelism == "DP2_TP4":
        shape = _mesh_device_map_get(DOTS_OCR_DP2_TP4_MESH_DEVICE_MAP, mesh_device)
        if shape is None:
            raise ValueError("DOTS_OCR_PARALLELISM=DP2_TP4 is only supported for MESH_DEVICE=T3K")
        return shape
    if parallelism == "DP2_TP2":
        # DP=2, TP=2 on T3K -> (2, 2) = 4 devices. TP axis (last dim) = 2 makes
        # ``_tp_degree`` = 2; defaults match DP2_TP4 (``tp4_prefill`` prefill body
        # + ``col_parallel`` decode) for multimodal OCR quality.
        shape = _mesh_device_map_get(DOTS_OCR_DP2_TP2_MESH_DEVICE_MAP, mesh_device)
        if shape is None:
            raise ValueError("DOTS_OCR_PARALLELISM=DP2_TP2 is only supported for MESH_DEVICE=T3K")
        return shape
    if parallelism == "DP":
        return _mesh_device_map_get(
            DOTS_OCR_DP_MESH_DEVICE_MAP,
            mesh_device,
            _mesh_device_map_get(MESH_DEVICE_MAP, mesh_device, len(ttnn.get_device_ids())),
        )
    return _mesh_device_map_get(MESH_DEVICE_MAP, mesh_device, len(ttnn.get_device_ids()))


def _dots_ocr_mesh_dp_degree():
    sh = _resolve_mesh_device_shape()
    if not isinstance(sh, (tuple, list)) or len(sh) < 2:
        return 1
    parallelism = _dots_ocr_parallelism_mode()
    if parallelism == "DP2_TP4":
        return int(sh[0])
    if parallelism == "DP":
        return int(sh[0]) if int(sh[0]) > 1 else int(sh[1])
    if int(sh[0]) > 1 and int(sh[1]) > 1:
        return int(sh[0])
    return 1


def _dots_ocr_mesh_num_devices():
    sh = _resolve_mesh_device_shape()
    if isinstance(sh, int):
        return max(1, int(sh))
    if isinstance(sh, (tuple, list)):
        if len(sh) >= 2:
            return int(sh[0]) * int(sh[1])
        if len(sh) == 1:
            return int(sh[0])
    return 1


def _dots_ocr_device_params():
    dp = {"trace_region_size": 300000000, "num_command_queues": 1}
    if _dots_ocr_mesh_num_devices() > 1:
        dp["fabric_config"] = ttnn.FabricConfig.FABRIC_1D_RING
    else:
        dp["fabric_config"] = ttnn.FabricConfig.DISABLED
    return dp


def _dots_ocr_decode_one_layer_tp_schemes():
    """Decoder-layer TP scheme(s) for the L1-boundary unit test.

    ``col_parallel`` is the default production TP contract for decode: hidden
    dim is sharded across TP (1536/4 = 384 on a 1x4 mesh), QKV/gate-up use
    N-dim column-parallel matmuls, and RMSNorm follows
    ``DOTS_OCR_COL_PARALLEL_RMSNORM_MODE`` (default ``full_multicore``). Set
    ``row`` explicitly to compare the older row-parallel path.
    Set ``DOTS_OCR_DECODE_ONE_LAYER_TP_SCHEMES=both`` or a comma-separated list
    to compare schemes.
    """
    env = os.environ.get("DOTS_OCR_DECODE_ONE_LAYER_TP_SCHEMES", "col_parallel").strip()
    if env.lower() == "both":
        return ["row", "col_parallel"]
    schemes = [s.strip() for s in env.split(",") if s.strip()]
    return schemes or ["col_parallel"]


def _dots_ocr_decode_full_decoder_tp_schemes():
    """Decoder-stack TP scheme(s) for the full 28-layer L1-boundary test.

    Mirrors :func:`_dots_ocr_decode_one_layer_tp_schemes` but for the full
    decoder stack. ``col_parallel`` (default) is the production TP contract
    (hidden dim sharded across TP, 1536/4 = 384 on a 1x4 mesh). ``row`` keeps the
    same hidden-sharded input/output (distributed TP4 RMSNorm) but routes
    QKV/gate-up as K-dim row-parallel matmuls. Set
    ``DOTS_OCR_DECODE_FULL_DECODER_TP_SCHEMES=both`` or a comma-separated list
    to compare schemes.
    """
    env = os.environ.get("DOTS_OCR_DECODE_FULL_DECODER_TP_SCHEMES", "col_parallel").strip()
    if env.lower() == "both":
        return ["row", "col_parallel"]
    schemes = [s.strip() for s in env.split(",") if s.strip()]
    return schemes or ["col_parallel"]


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


def _assert_l1_resident(tensor, name: str):
    assert isinstance(tensor, ttnn.Tensor), f"{name} should be a TTNN tensor"
    assert tensor.memory_config().buffer_type == ttnn.BufferType.L1, f"{name} should reside in L1"


def _assert_ttnn_tensor(tensor, name: str):
    assert isinstance(tensor, ttnn.Tensor), f"{name} should be a TTNN tensor"


def _assert_vision_tensor_spec(
    tensor,
    name: str,
    *,
    shape: list[int],
    dtype: ttnn.DataType,
    buffer_type: ttnn.BufferType,
    sharded: bool | None = None,
):
    _assert_ttnn_tensor(tensor, name)
    assert list(tensor.shape) == shape, f"{name} shape: expected {shape}, got {list(tensor.shape)}"
    assert tensor.dtype == dtype, f"{name} dtype: expected {dtype}, got {tensor.dtype}"
    mem = tensor.memory_config()
    assert mem.buffer_type == buffer_type, f"{name} buffer: expected {buffer_type}, got {mem.buffer_type}"
    if sharded is not None:
        is_sharded = mem.is_sharded()
        assert is_sharded == sharded, f"{name} sharded: expected {sharded}, got {is_sharded}"


def _vision_block_token_shape(seq_len: int, hidden_size: int) -> list[int]:
    return [1, 1, seq_len, hidden_size]


def _vision_head_shape(seq_len: int, num_heads: int, head_dim: int) -> list[int]:
    return [1, num_heads, seq_len, head_dim]


def _dots_ocr_vision_one_layer_block_counts() -> list[int]:
    """Block counts for ``test_dots_ocr_vision_one_layer``.

    Default runs 1, 3, and 4 identical blocks. Override with
    ``DOTS_OCR_VISION_NUM_BLOCKS=3`` (or 4) to run a single count.
    """
    env = os.environ.get("DOTS_OCR_VISION_NUM_BLOCKS")
    if env is not None:
        return [int(env)]
    return [1, 3, 4]


def _dots_ocr_pipeline_batch_size():
    """Match ``TTNNDotsOCRPipeline`` batch to mesh size when DP is requested.

    DP sharding in the pipeline requires ``batch_size == dp_degree``. Plain
    ``T3K`` defaults to the hybrid ``(2, 4)`` mesh; ``T3K_DP8`` keeps the
    working ``(8, 1)`` DP8/TP1 mesh available explicitly.
    """
    n = _dots_ocr_mesh_dp_degree()
    return n if n > 1 else 1


def _dots_ocr_stack_input_ids_for_dp(input_ids: torch.Tensor) -> torch.Tensor:
    """Turn ``[1, S]`` into ``[B, S]`` by repeating the same prompt on each stream."""
    bs = _dots_ocr_pipeline_batch_size()
    if bs <= 1 or input_ids.shape[0] == bs:
        return input_ids
    if input_ids.shape[0] != 1:
        raise ValueError(f"DP batch stacking expects base shape [1, S], got {tuple(input_ids.shape)}")
    return input_ids.expand(bs, -1).contiguous()


@pytest.mark.parametrize(
    "device_params",
    [_dots_ocr_device_params()],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [_resolve_mesh_device_shape()],
    indirect=True,
)
def test_dots_ocr_text(mesh_device):
    """Test standalone TTNN pipeline for dots.ocr (text-only, no vision)."""

    pbatch = _dots_ocr_pipeline_batch_size()
    pipeline = TTNNDotsOCRPipeline.from_hf_model(
        model_path=DOTS_OCR_LOCAL_PATH,
        device=mesh_device,
        batch_size=pbatch,
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
    input_ids = _dots_ocr_stack_input_ids_for_dp(inputs["input_ids"])

    pipeline.warmup(input_ids)

    DispatchManager.clear_timings()
    start_time = time.time()
    generated_ids = pipeline.generate(input_ids, max_new_tokens=128)
    ttnn.synchronize_device(mesh_device)
    end_time = time.time()

    if isinstance(generated_ids[0], list):
        streams = [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated_ids]
        text = "\n--- stream ---\n".join(streams)
        num_tokens = sum(len(seq) for seq in generated_ids)
    else:
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        num_tokens = len(generated_ids)
    print(f"Pipeline TEXT OUTPUT: {text}")

    total_time = end_time - start_time
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
    [_dots_ocr_device_params()],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [_resolve_mesh_device_shape()],
    indirect=True,
)
@pytest.mark.parametrize(
    "image_link",
    [
        "https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/demo/demo_image1.jpg",
    ],
)
def test_dots_ocr_vision(mesh_device, image_link):
    """Test standalone TTNN pipeline for dots.ocr with vision (image + text).

    Default mesh comes from ``MESH_DEVICE``. Plain T3K is the hybrid ``(2, 4)``
    shape and ``T3K_DP8`` is the working ``(8, 1)`` DP8/TP1 shape.
    ``DOTS_OCR_PARALLELISM=DP`` remains available for DP-only shapes. For TP decode scheme comparisons, set
    ``DOTS_OCR_TP_DECODE_SCHEME=row``; otherwise ``col_parallel`` is used.
    """
    pytest.importorskip("qwen_vl_utils")
    from qwen_vl_utils import process_vision_info
    from PIL import Image
    import requests

    pbatch = _dots_ocr_pipeline_batch_size()
    pipeline = TTNNDotsOCRPipeline.from_hf_model(
        model_path=DOTS_OCR_LOCAL_PATH,
        device=mesh_device,
        batch_size=pbatch,
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

    # Load and crop the image
    image = Image.open(requests.get(image_link, stream=True).raw)
    original_width, original_height = image.size

    # Crop to 57.5% of original height from the top
    new_height = int(original_height * 0.575)
    top = 0
    bottom = new_height

    # Crop box: (left, top, right, bottom)
    image = image.crop((0, top, original_width, bottom))

    print(f"Cropped image from {original_width}x{original_height} to {original_width}x{new_height}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
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
        max_length=2800,
        return_tensors="pt",
    )

    input_ids = _dots_ocr_stack_input_ids_for_dp(inputs["input_ids"])
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    image_grid_thw = inputs["image_grid_thw"]

    pipeline.warmup(input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    DispatchManager.clear_timings()
    start_time = time.time()
    generated_ids = pipeline.generate(
        input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=180,
        stop_on_eos=False,
    )
    ttnn.synchronize_device(mesh_device)
    end_time = time.time()

    if isinstance(generated_ids[0], list):
        streams = [processor.decode(seq, skip_special_tokens=True) for seq in generated_ids]
        decoded = "\n--- stream ---\n".join(streams)
        num_tokens = sum(len(seq) for seq in generated_ids)
    else:
        decoded = processor.decode(generated_ids, skip_special_tokens=True)
        num_tokens = len(generated_ids)
    print(f"Pipeline VISION OUTPUT: {decoded}")

    total_time = end_time - start_time
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


@pytest.mark.parametrize(
    "device_params",
    [_dots_ocr_device_params()],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [_resolve_mesh_device_shape()],
    indirect=True,
)
@pytest.mark.parametrize("tp_decode_scheme", _dots_ocr_decode_one_layer_tp_schemes())
def test_dots_ocr_decode_one_layer_l1_boundaries(mesh_device, tp_decode_scheme):
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
    # hf_layer = hf_model.model.layers[0]
    # hf_rotary_emb = hf_model.model.rotary_emb
    # layer = TTNNDotsOCRDecoderLayer.from_torch(hf_layer, tp_decode_scheme=tp_decode_scheme)
    # layer._unique_name = "model.layers.0"
    # layer.override_children_module_names()

    # set_device(layer, mesh_device, register_forward_hook=False, dump_visualization=False)
    # layer.preprocess_weights()
    # layer.move_weights_to_device()

    # paged_cache = _create_paged_kv_cache(model_config, mesh_device, batch_size=1)
    hf_layer = hf_model.model.layers[0]
    hf_rotary_emb = hf_model.model.rotary_emb
    # Profile a specific decoder depth's dtype config (BFP4 for layers 0..6,
    # BFP8 for layers >=7). layer_idx drives _use_bfp8_decoder_weights AND the
    # paged-cache slot, so size the cache to the real depth.
    target_layer_idx = int(os.environ.get("DOTS_OCR_DECODE_ONE_LAYER_IDX", "8"))
    hf_layer.self_attn.layer_idx = target_layer_idx
    layer = TTNNDotsOCRDecoderLayer.from_torch(hf_layer, tp_decode_scheme=tp_decode_scheme)
    layer._unique_name = f"model.layers.{target_layer_idx}"
    layer.override_children_module_names()

    set_device(layer, mesh_device, register_forward_hook=False, dump_visualization=False)
    layer.preprocess_weights()
    layer.move_weights_to_device()

    model_config.num_hidden_layers = max(target_layer_idx + 1, 28)
    # head_parallel stores only this device's LOCAL KV heads (num_key_value_heads
    # // TP = 1 at TP=2); row / col_parallel store the full replicated KV-head set.
    _tp_last = int(mesh_device.shape[-1]) if (hasattr(mesh_device, "shape") and list(mesh_device.shape)) else 1
    cache_kv_heads = (
        model_config.num_key_value_heads // _tp_last
        if tp_decode_scheme == "head_parallel"
        else model_config.num_key_value_heads
    )
    paged_cache = _create_paged_kv_cache(model_config, mesh_device, batch_size=1, num_kv_heads=cache_kv_heads)

    hidden_states_torch = torch.randn(1, 1, model_config.hidden_size, dtype=torch.bfloat16)
    num_devices = int(mesh_device.get_num_devices()) if hasattr(mesh_device, "get_num_devices") else 1
    is_tp_mesh = num_devices > 1 and hasattr(mesh_device, "shape") and list(mesh_device.shape)[-1] > 1
    # ``row``, ``col_parallel`` and ``head_parallel`` all keep the decode
    # hidden/residual TP-sharded end to end (head_parallel's row-parallel
    # reduce_scatter o_proj returns the same hidden/TP shard as col_parallel's
    # N-sharded o_proj). So all three shard the stack input along hidden on a TP
    # mesh and concat (dim=-1) for PCC.
    uses_tp_shard = is_tp_mesh and tp_decode_scheme in ("row", "col_parallel", "head_parallel")
    if uses_tp_shard:
        input_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
    else:
        input_mapper = None
    hidden_state_kwargs = {"mesh_mapper": input_mapper} if input_mapper is not None else {}
    hidden_states = ttnn.from_torch(
        hidden_states_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        **hidden_state_kwargs,
    )
    if uses_tp_shard:
        assert int(hidden_states.shape[-1]) == model_config.hidden_size // int(mesh_device.shape[-1])
    elif is_tp_mesh:
        assert int(hidden_states.shape[-1]) == model_config.hidden_size
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
    if uses_tp_shard:
        assert int(output.shape[-1]) == model_config.hidden_size // int(mesh_device.shape[-1])
    elif is_tp_mesh:
        assert int(output.shape[-1]) == model_config.hidden_size
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

    if uses_tp_shard:
        ttnn_output_torch = ttnn.to_torch(
            output,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1),
        )
    elif num_devices > 1:
        # Pure-DP meshes replicate the single-token decode stream on each device.
        # ``ConcatMeshToTensor(dim=0)`` stacks replicas along batch, after which
        # we take the first slice as the canonical output.
        ttnn_output_torch = ttnn.to_torch(
            output,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )[:1]
    else:
        ttnn_output_torch = ttnn.to_torch(output)
    ttnn_output_torch = ttnn_output_torch.to(torch.bfloat16).reshape(torch_output.shape)
    # Random-weight bf16 + paged-SDPA vs. eager attention adds some numerical
    # drift relative to a fp32 reference; 0.99 is the tight-but-safe bar that
    # still catches the regressions this test guards against (silent
    # layout/sharding bugs, wrong RoPE, dropped residual, etc.).
    assert_with_pcc(torch_output, ttnn_output_torch, pcc=0.99)


@pytest.mark.parametrize(
    "device_params",
    [_dots_ocr_device_params()],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [_resolve_mesh_device_shape()],
    indirect=True,
)
@pytest.mark.parametrize(
    "num_decode_steps",
    [int(os.environ.get("DOTS_OCR_DECODE_STEPS", "1440"))],
)
def test_dots_ocr_decode_one_layer_sdpa_cost_sweep(mesh_device, num_decode_steps):
    """Benchmark one decoder layer across N decode steps to characterise SDPA
    cost as ``cache_position`` grows.

    Loops the same layer ``num_decode_steps`` times, incrementing the cache
    position each step (mirrors what the full vision pipeline does, but with
    a single layer so the profiler buffer can capture every SDPA invocation
    cleanly). Per-step wall-time is sampled at a few cache positions and
    printed at the end so the SDPA cost ramp from 0 -> num_decode_steps is
    visible without running the full ~100s vision pipeline.

    Default N = 1440 matches the demo OCR run length; override with
    ``DOTS_OCR_DECODE_STEPS=200`` for a faster iteration loop. There is no
    PCC check here (the ``..._l1_boundaries`` test covers correctness at
    pos 0 -- per-step accuracy at growing cache lengths is not the point
    of this benchmark).
    """
    import time

    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    model_config = AutoConfig.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    model_config.num_hidden_layers = 1
    hf_model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True).to(dtype=torch.bfloat16).eval()
    model_config = hf_model.config
    layer = TTNNDotsOCRDecoderLayer.from_torch(hf_model.model.layers[0])
    layer._unique_name = "model.layers.0"
    layer.override_children_module_names()
    del hf_model

    set_device(layer, mesh_device, register_forward_hook=False, dump_visualization=False)
    layer.preprocess_weights()
    layer.move_weights_to_device()

    paged_cache = _create_paged_kv_cache(model_config, mesh_device, batch_size=1)
    hidden_states = ttnn.from_torch(
        torch.randn(1, 1, model_config.hidden_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Sample wall-time at these cache_position values. The first step is
    # always sampled (pos 0 floor); the rest are spread roughly evenly to
    # capture the ramp.
    sample_positions = sorted(
        set(
            [0, 1, 50, 100, 200, 400, 800, 1200, num_decode_steps - 1]
            if num_decode_steps > 200
            else [0, 1, 25, 50, 100, num_decode_steps - 1]
        )
    )
    sample_positions = [p for p in sample_positions if 0 <= p < num_decode_steps]
    timings: dict[int, float] = {}

    print(
        f"\n[sdpa-sweep] Running {num_decode_steps} decode steps on layer 0; "
        f"sampling wall-time at positions: {sample_positions}",
        flush=True,
    )

    for step_idx in range(num_decode_steps):
        cache_position = ttnn.from_torch(
            torch.tensor([step_idx], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if step_idx in sample_positions:
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            output = layer.forward(hidden_states, past_key_value=paged_cache, cache_position=cache_position)[0]
            ttnn.synchronize_device(mesh_device)
            timings[step_idx] = (time.perf_counter() - t0) * 1e6  # us
            print(
                f"[sdpa-sweep] cache_position={step_idx:5d}  " f"layer-forward wall-time={timings[step_idx]:.1f} us",
                flush=True,
            )
            ttnn.deallocate(output)
        else:
            output = layer.forward(hidden_states, past_key_value=paged_cache, cache_position=cache_position)[0]
            ttnn.deallocate(output)

        ttnn.deallocate(cache_position)

    ttnn.synchronize_device(mesh_device)

    # Summary: floor vs peak + linear-fit slope (approx SDPA cost per unit
    # cache_position growth). The per-step wall time also includes LN,
    # QKV, MLP -- those are constant in cache_position, so the *delta*
    # between samples is essentially the SDPA scaling cost.
    floor_us = timings[sample_positions[0]]
    peak_us = timings[sample_positions[-1]]
    delta_us = peak_us - floor_us
    print(
        f"\n[sdpa-sweep] floor (pos {sample_positions[0]}) = {floor_us:.1f} us, "
        f"peak (pos {sample_positions[-1]}) = {peak_us:.1f} us, "
        f"delta = {delta_us:.1f} us (~SDPA growth across {sample_positions[-1]} positions)",
        flush=True,
    )


@pytest.mark.parametrize(
    "device_params",
    [_dots_ocr_device_params()],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [_resolve_mesh_device_shape()],
    indirect=True,
)
def test_dots_ocr_prefill_one_layer(mesh_device):
    """Exercise one decoder layer in prefill mode with a paged KV cache."""
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    model_config = AutoConfig.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    model_config.num_hidden_layers = 1
    hf_model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True).to(dtype=torch.bfloat16).eval()
    model_config = hf_model.config
    layer = TTNNDotsOCRDecoderLayer.from_torch(hf_model.model.layers[0])
    layer._unique_name = "model.layers.0"
    layer.override_children_module_names()
    del hf_model

    set_device(layer, mesh_device, register_forward_hook=False, dump_visualization=False)
    layer.preprocess_weights()
    layer.move_weights_to_device()

    seq_len = 2816
    paged_cache = _create_paged_kv_cache(model_config, mesh_device, batch_size=1)
    hidden_states = ttnn.from_torch(
        torch.randn(1, seq_len, model_config.hidden_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cache_position = ttnn.from_torch(
        torch.arange(seq_len, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    seen = {"attn_prefill": False, "mlp": False}
    original_attn_prefill = layer.self_attn._forward_prefill
    original_attn_decode = layer.self_attn._forward_decode_paged
    original_mlp_forward = layer.mlp.forward

    def checked_attn_prefill(*args, **kwargs):
        output = original_attn_prefill(*args, **kwargs)
        _assert_ttnn_tensor(output[0], "prefill attention output")
        seen["attn_prefill"] = True
        return output

    def unexpected_attn_decode(*args, **kwargs):
        raise AssertionError("single-layer prefill test should not use decode attention")

    def checked_mlp_forward(hidden_states):
        _assert_ttnn_tensor(hidden_states, "prefill MLP input")
        output = original_mlp_forward(hidden_states)
        _assert_ttnn_tensor(output, "prefill MLP output")
        seen["mlp"] = True
        return output

    layer.self_attn._forward_prefill = checked_attn_prefill
    layer.self_attn._forward_decode_paged = unexpected_attn_decode
    layer.mlp.forward = checked_mlp_forward

    output = layer.forward(hidden_states, past_key_value=paged_cache, cache_position=cache_position)[0]
    ttnn.synchronize_device(mesh_device)

    _assert_ttnn_tensor(output, "prefill decoder layer output")
    num_devices = int(mesh_device.get_num_devices()) if hasattr(mesh_device, "get_num_devices") else 1
    valid_hidden_dims = {model_config.hidden_size}
    if num_devices > 1:
        valid_hidden_dims.add(model_config.hidden_size // num_devices)
    assert int(output.shape[0]) == 1
    assert int(output.shape[-2]) == seq_len
    assert int(output.shape[-1]) in valid_hidden_dims
    assert seen == {"attn_prefill": True, "mlp": True}

    layer.self_attn._forward_decode_paged = original_attn_decode


@pytest.mark.parametrize(
    "device_params",
    [_dots_ocr_device_params()],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [_resolve_mesh_device_shape()],
    indirect=True,
)
@pytest.mark.parametrize("num_vision_blocks", _dots_ocr_vision_one_layer_block_counts())
def test_dots_ocr_vision_one_layer(mesh_device, num_vision_blocks):
    """Exercise one or more identical dots.ocr vision blocks (norm, attn, MLP).

    Shapes and dtypes match the production vision-block perf slice at
    M=11264 (``11264 x 1536 x {4608,1536,4224}`` matmuls in perf_vis.txt).
    Set ``DOTS_OCR_VISION_NUM_BLOCKS=3`` or ``4`` to run only that count.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    model_config = AutoConfig.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    model_config.num_hidden_layers = 1
    vision_config = getattr(model_config, "vision_config", None)
    if vision_config is not None:
        for attr in ("num_hidden_layers", "num_layers", "depth"):
            if hasattr(vision_config, attr):
                setattr(vision_config, attr, num_vision_blocks)
    hf_model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True).to(dtype=torch.bfloat16).eval()
    blocks = getattr(hf_model.vision_tower, "blocks", getattr(hf_model.vision_tower, "layers", None))
    assert blocks is not None, "dots.ocr vision tower should expose blocks/layers"
    hf_block = blocks[0]
    hf_model.vision_tower.blocks = torch.nn.ModuleList([hf_block] * num_vision_blocks)
    vision_tower = TTNNDotsOCRVisionTower.from_torch(hf_model.vision_tower, hf_model.config)
    vision_tower._unique_name = "vision_tower"
    vision_tower.override_children_module_names()
    del hf_model

    assert len(vision_tower.blocks) == num_vision_blocks
    set_device(vision_tower, mesh_device, register_forward_hook=False, dump_visualization=False)
    vision_tower.preprocess_weights()
    vision_tower.move_weights_to_device()

    # M=11264 matches the traced vision-tower bucket in perf_vis.txt (not 12288).
    grid_thw = torch.tensor([[1, 88, 128]], dtype=torch.int32)
    seq_len = int(grid_thw[0, 0] * grid_thw[0, 1] * grid_thw[0, 2])
    assert seq_len == 11264
    hidden_size = int(vision_tower.hidden_size)
    num_heads = int(vision_tower.num_heads)
    head_dim = int(vision_tower.head_dim)
    token_shape = _vision_block_token_shape(seq_len, hidden_size)
    head_shape = _vision_head_shape(seq_len, num_heads, head_dim)

    hidden_states = ttnn.from_torch(
        torch.randn(1, 1, seq_len, hidden_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    seen = {"norm1": 0, "attn": 0, "norm2": 0, "mlp": 0}
    matmul_calls: list[tuple[int, int, int, ttnn.DataType | None]] = []
    typecast_calls: list[tuple[ttnn.DataType, ttnn.DataType]] = []
    mul_calls: list[tuple[ttnn.DataType | None, ttnn.DataType | None, ttnn.DataType | None]] = []

    def _wrap_block_submodules(tt_block):
        original_norm1_forward = tt_block.norm1.forward
        original_attn_forward = tt_block.attn.forward
        original_norm2_forward = tt_block.norm2.forward
        original_mlp_forward = tt_block.mlp.forward

        def checked_norm1_forward(hidden_states, *args, **kwargs):
            _assert_vision_tensor_spec(
                hidden_states,
                "vision norm1 input",
                shape=token_shape,
                dtype=ttnn.bfloat8_b,
                buffer_type=ttnn.BufferType.DRAM,
                sharded=False,
            )
            output = original_norm1_forward(hidden_states, *args, **kwargs)
            _assert_vision_tensor_spec(
                output,
                "vision norm1 output",
                shape=token_shape,
                dtype=ttnn.bfloat8_b,
                buffer_type=ttnn.BufferType.DRAM,
                sharded=False,
            )
            seen["norm1"] += 1
            return output

        def checked_attn_forward(hidden_states, *args, **kwargs):
            _assert_vision_tensor_spec(
                hidden_states,
                "vision attention input",
                shape=token_shape,
                dtype=ttnn.bfloat8_b,
                buffer_type=ttnn.BufferType.DRAM,
                sharded=False,
            )
            output = original_attn_forward(hidden_states, *args, **kwargs)
            _assert_vision_tensor_spec(
                output,
                "vision attention output",
                shape=token_shape,
                dtype=ttnn.bfloat8_b,
                buffer_type=ttnn.BufferType.L1,
                sharded=True,
            )
            seen["attn"] += 1
            return output

        def checked_norm2_forward(hidden_states, *args, **kwargs):
            _assert_vision_tensor_spec(
                hidden_states,
                "vision norm2 input",
                shape=token_shape,
                dtype=ttnn.bfloat8_b,
                buffer_type=ttnn.BufferType.DRAM,
                sharded=False,
            )
            output = original_norm2_forward(hidden_states, *args, **kwargs)
            _assert_vision_tensor_spec(
                output,
                "vision norm2 output",
                shape=token_shape,
                dtype=ttnn.bfloat8_b,
                buffer_type=ttnn.BufferType.L1,
                sharded=False,
            )
            seen["norm2"] += 1
            return output

        def checked_mlp_forward(hidden_states, *args, **kwargs):
            _assert_vision_tensor_spec(
                hidden_states,
                "vision MLP input",
                shape=token_shape,
                dtype=ttnn.bfloat8_b,
                buffer_type=ttnn.BufferType.L1,
                sharded=False,
            )
            output = original_mlp_forward(hidden_states, *args, **kwargs)
            _assert_vision_tensor_spec(
                output,
                "vision MLP output",
                shape=token_shape,
                dtype=ttnn.bfloat4_b,
                buffer_type=ttnn.BufferType.L1,
                sharded=False,
            )
            seen["mlp"] += 1
            return output

        tt_block.norm1.forward = checked_norm1_forward
        tt_block.attn.forward = checked_attn_forward
        tt_block.norm2.forward = checked_norm2_forward
        tt_block.mlp.forward = checked_mlp_forward

    for tt_block in vision_tower.blocks:
        _wrap_block_submodules(tt_block)
    original_linear = ttnn.linear
    original_typecast = ttnn.typecast
    original_mul = ttnn.mul
    original_create_heads = ttnn.experimental.nlp_create_qkv_heads
    original_rotary = ttnn.experimental.rotary_embedding
    original_concat_heads = ttnn.experimental.nlp_concat_heads
    original_sdpa = ttnn.transformer.scaled_dot_product_attention

    def _matmul_m_dim(activation: ttnn.Tensor) -> int:
        return int(activation.shape[0]) * int(activation.shape[1]) * int(activation.shape[2])

    def recording_linear(activation, weight, *args, **kwargs):
        out_dtype = kwargs.get("dtype")
        matmul_calls.append((_matmul_m_dim(activation), int(weight.shape[-2]), int(weight.shape[-1]), out_dtype))
        return original_linear(activation, weight, *args, **kwargs)

    def recording_typecast(tensor, dtype, *args, **kwargs):
        typecast_calls.append((tensor.dtype, dtype))
        return original_typecast(tensor, dtype, *args, **kwargs)

    def recording_mul(a, b, *args, **kwargs):
        mul_calls.append((a.dtype, b.dtype, kwargs.get("dtype")))
        return original_mul(a, b, *args, **kwargs)

    def recording_create_heads(qkv, *args, **kwargs):
        _assert_vision_tensor_spec(
            qkv,
            "nlp_create_qkv_heads input",
            shape=[1, 1, seq_len, hidden_size * 3],
            dtype=ttnn.bfloat8_b,
            buffer_type=ttnn.BufferType.DRAM,
            sharded=False,
        )
        q, k, v = original_create_heads(qkv, *args, **kwargs)
        for label, head in zip(("Q", "K", "V"), (q, k, v)):
            _assert_vision_tensor_spec(
                head,
                f"nlp_create_qkv_heads {label}",
                shape=head_shape,
                dtype=ttnn.bfloat8_b,
                buffer_type=ttnn.BufferType.L1,
                sharded=False,
            )
        return q, k, v

    def recording_rotary(tensor, cos, sin, *args, **kwargs):
        _assert_vision_tensor_spec(
            tensor,
            "rotary input",
            shape=head_shape,
            dtype=ttnn.bfloat8_b,
            buffer_type=ttnn.BufferType.L1,
            sharded=False,
        )
        out = original_rotary(tensor, cos, sin, *args, **kwargs)
        _assert_vision_tensor_spec(
            out,
            "rotary output",
            shape=head_shape,
            dtype=ttnn.bfloat8_b,
            buffer_type=ttnn.BufferType.DRAM,
            sharded=False,
        )
        return out

    def recording_concat_heads(ctx, *args, **kwargs):
        _assert_vision_tensor_spec(
            ctx,
            "nlp_concat_heads input",
            shape=head_shape,
            dtype=ttnn.bfloat8_b,
            buffer_type=ttnn.BufferType.DRAM,
            sharded=False,
        )
        out = original_concat_heads(ctx, *args, **kwargs)
        _assert_vision_tensor_spec(
            out,
            "nlp_concat_heads output",
            shape=token_shape,
            dtype=ttnn.bfloat8_b,
            buffer_type=ttnn.BufferType.L1,
            sharded=False,
        )
        return out

    def recording_sdpa(q, k, v, *args, **kwargs):
        _assert_vision_tensor_spec(
            q,
            "sdpa Q",
            shape=head_shape,
            dtype=ttnn.bfloat8_b,
            buffer_type=ttnn.BufferType.DRAM,
            sharded=False,
        )
        _assert_vision_tensor_spec(
            k,
            "sdpa K",
            shape=head_shape,
            dtype=ttnn.bfloat8_b,
            buffer_type=ttnn.BufferType.DRAM,
            sharded=False,
        )
        _assert_vision_tensor_spec(
            v,
            "sdpa V",
            shape=head_shape,
            dtype=ttnn.bfloat4_b,
            buffer_type=ttnn.BufferType.DRAM,
            sharded=False,
        )
        out = original_sdpa(q, k, v, *args, **kwargs)
        _assert_vision_tensor_spec(
            out,
            "sdpa output",
            shape=head_shape,
            dtype=ttnn.bfloat8_b,
            buffer_type=ttnn.BufferType.DRAM,
            sharded=False,
        )
        return out

    ttnn.linear = recording_linear
    ttnn.typecast = recording_typecast
    ttnn.mul = recording_mul
    ttnn.experimental.nlp_create_qkv_heads = recording_create_heads
    ttnn.experimental.rotary_embedding = recording_rotary
    ttnn.experimental.nlp_concat_heads = recording_concat_heads
    ttnn.transformer.scaled_dot_product_attention = recording_sdpa
    try:
        rot_mats, _ = vision_tower.rope.build_padded(grid_thw, seq_len, seq_len)
        output = hidden_states
        for tt_block in vision_tower.blocks:
            output = tt_block.forward(
                output,
                rot_mats=rot_mats,
                cu_seqlens=None,
                attention_logical_seq_len=seq_len,
            )
        ttnn.synchronize_device(mesh_device)
    finally:
        ttnn.linear = original_linear
        ttnn.typecast = original_typecast
        ttnn.mul = original_mul
        ttnn.experimental.nlp_create_qkv_heads = original_create_heads
        ttnn.experimental.rotary_embedding = original_rotary
        ttnn.experimental.nlp_concat_heads = original_concat_heads
        ttnn.transformer.scaled_dot_product_attention = original_sdpa

    _assert_vision_tensor_spec(
        output,
        "vision block stack output",
        shape=token_shape,
        dtype=ttnn.bfloat8_b,
        buffer_type=ttnn.BufferType.DRAM,
        sharded=False,
    )
    assert seen == {
        "norm1": num_vision_blocks,
        "attn": num_vision_blocks,
        "norm2": num_vision_blocks,
        "mlp": num_vision_blocks,
    }

    intermediate_size = int(vision_tower.blocks[0].mlp.tt_fc1_weight.shape[-1])
    per_block_matmuls = [
        (seq_len, hidden_size, hidden_size * 3, ttnn.bfloat8_b),  # qkv
        (seq_len, hidden_size, hidden_size, ttnn.bfloat8_b),  # o_proj
        (seq_len, hidden_size, intermediate_size, ttnn.bfloat8_b),  # gate
        (seq_len, hidden_size, intermediate_size, ttnn.bfloat4_b),  # up
        (seq_len, intermediate_size, hidden_size, ttnn.bfloat4_b),  # down
    ]
    expected_matmuls = per_block_matmuls * num_vision_blocks
    assert matmul_calls == expected_matmuls, f"matmul shapes/dtypes: got {matmul_calls}"
    assert typecast_calls == [(ttnn.bfloat8_b, ttnn.bfloat4_b)] * num_vision_blocks, f"V typecast: got {typecast_calls}"
    assert len(mul_calls) == num_vision_blocks, f"expected {num_vision_blocks} silu*mul, got {mul_calls}"
    for gate_dtype, up_dtype, out_dtype in mul_calls:
        assert gate_dtype == ttnn.bfloat8_b
        assert up_dtype == ttnn.bfloat4_b
        assert out_dtype == ttnn.bfloat8_b


@pytest.mark.parametrize(
    "device_params",
    [_dots_ocr_device_params()],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [_resolve_mesh_device_shape()],
    indirect=True,
)
@pytest.mark.parametrize("tp_decode_scheme", _dots_ocr_decode_full_decoder_tp_schemes())
def test_dots_ocr_decode_full_decoder_l1_boundaries(mesh_device, tp_decode_scheme):
    """Exercise all decoder layers in decode mode and require L1 attn/MLP boundaries.

    Also runs the same input through all 28 HF reference layers and compares
    the final output via PCC. This is the regression gate for accumulated
    multi-layer numerical drift (e.g. pushing BFP4 onto too many layers) that
    the single-layer PCC test cannot detect.

    On a TP mesh (e.g. ``MESH_DEVICE=P150x4`` -> ``(1, 4)``) the decoder TP
    scheme is exercised like ``test_dots_ocr_decode_one_layer_l1_boundaries``:
    both ``row`` and ``col_parallel`` shard the hidden dim across TP (sharded
    input/output, distributed TP4 RMSNorm); they differ only in whether the
    matmuls are K-dim (``row``) or N-dim column-parallel (``col_parallel``).
    Select schemes with ``DOTS_OCR_DECODE_FULL_DECODER_TP_SCHEMES`` (default
    ``row``).
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
        layer = TTNNDotsOCRDecoderLayer.from_torch(hf_layer, tp_decode_scheme=tp_decode_scheme)
        layer._unique_name = f"model.layers.{layer_idx}"
        layer.override_children_module_names()
        decoder_layers.append(layer)
    del hf_model

    decoder_stack = TTNNDotsOCRLayerStack(decoder_layers)
    decoder_stack._unique_name = "model.layer_stack"
    set_device(decoder_stack, mesh_device, register_forward_hook=False, dump_visualization=False)
    decoder_stack.preprocess_weights()
    decoder_stack.move_weights_to_device()

    num_devices = int(mesh_device.get_num_devices()) if hasattr(mesh_device, "get_num_devices") else 1
    is_tp_mesh = num_devices > 1 and hasattr(mesh_device, "shape") and list(mesh_device.shape)[-1] > 1
    # head_parallel stores only this device's LOCAL KV heads (= num_kv // TP);
    # row / col_parallel store the full replicated KV-head set.
    _tp_last = int(mesh_device.shape[-1]) if (hasattr(mesh_device, "shape") and list(mesh_device.shape)) else 1
    cache_kv_heads = (
        model_config.num_key_value_heads // _tp_last
        if tp_decode_scheme == "head_parallel"
        else model_config.num_key_value_heads
    )
    paged_cache = _create_paged_kv_cache(model_config, mesh_device, batch_size=1, num_kv_heads=cache_kv_heads)
    # ``row``, ``col_parallel`` and ``head_parallel`` all keep the decode
    # hidden/residual TP-sharded end to end, so all three shard the stack input
    # along hidden on a TP mesh and concat (dim=-1) the sharded output for PCC.
    uses_tp_shard = is_tp_mesh and tp_decode_scheme in ("row", "col_parallel", "head_parallel")
    if uses_tp_shard:
        input_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
    else:
        input_mapper = None
    hidden_state_kwargs = {"mesh_mapper": input_mapper} if input_mapper is not None else {}
    hidden_states = ttnn.from_torch(
        hidden_states_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        **hidden_state_kwargs,
    )
    if uses_tp_shard:
        assert int(hidden_states.shape[-1]) == model_config.hidden_size // int(mesh_device.shape[-1])
    elif is_tp_mesh:
        assert int(hidden_states.shape[-1]) == model_config.hidden_size
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
    if uses_tp_shard:
        assert int(output.shape[-1]) == model_config.hidden_size // int(mesh_device.shape[-1])
    elif is_tp_mesh:
        assert int(output.shape[-1]) == model_config.hidden_size
    assert all(boundaries == {"attn": True, "mlp": True} for boundaries in seen_boundaries.values())

    # ------------------------------------------------------------------
    # PCC verification across all 28 layers at cache_position=0.
    # ------------------------------------------------------------------
    # ``torch_output`` was computed at the top of the test, before any
    # TTNN setup, so the HF reference is unaffected by ``from_torch`` /
    # ``preprocess_weights`` in-place mutations on the HF layers.
    if uses_tp_shard:
        # Both ``row`` and ``col_parallel`` TP keep the hidden dim sharded across
        # TP all the way to the stack output; concat the per-device shards back
        # into full hidden.
        ttnn_output_torch = ttnn.to_torch(
            output,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1),
        )
    elif num_devices > 1:
        # Multi-device mesh (T3K DP (8,1) or TP (1,8)/``col_parallel``):
        # DP-with-batch=1 and TP-after-all-reduce both produce identical data on
        # every device. ``ConcatMeshToTensor(dim=0)`` stacks per-device slices
        # along batch; take the first as the canonical output.
        ttnn_output_torch = ttnn.to_torch(
            output,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )[:1]
    else:
        ttnn_output_torch = ttnn.to_torch(output)
    ttnn_output_torch = ttnn_output_torch.to(torch.bfloat16).reshape(torch_output.shape)

    # 28-layer accumulated bf16 + BFP4-weight + paged-SDPA drift is larger
    # than the single-layer case; 0.95 is the tight-but-survivable bar that
    # still catches the failures this test guards against (silent layout/
    # sharding bugs, wrong RoPE, accumulated BFP4 corruption like the OCR
    # demo's dropped 'K' in tabular data).
    assert_with_pcc(torch_output, ttnn_output_torch, pcc=0.95)


@pytest.mark.parametrize(
    "device_params",
    [_dots_ocr_device_params()],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [_resolve_mesh_device_shape()],
    indirect=True,
)
@pytest.mark.parametrize(
    "image_link",
    [
        "https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/demo/demo_image1.jpg",
    ],
)
def test_dots_ocr_vision(mesh_device, image_link):
    """Test standalone TTNN pipeline for dots.ocr with vision (image + text).

    Default mesh comes from ``MESH_DEVICE``. Plain T3K is the hybrid ``(2, 4)``
    shape and ``T3K_DP8`` is the working ``(8, 1)`` DP8/TP1 shape.
    ``DOTS_OCR_PARALLELISM=DP`` remains available for DP-only shapes. In DP mode
    the test sets one prompt row per DP stream.
    """
    pytest.importorskip("qwen_vl_utils")
    from qwen_vl_utils import process_vision_info
    from PIL import Image
    import requests

    pbatch = _dots_ocr_pipeline_batch_size()
    pipeline = TTNNDotsOCRPipeline.from_hf_model(
        model_path=DOTS_OCR_LOCAL_PATH,
        device=mesh_device,
        batch_size=pbatch,
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

    # Load and crop the image
    image = Image.open(requests.get(image_link, stream=True).raw)
    original_width, original_height = image.size

    # Crop to 57.5% of original height from the top
    new_height = int(original_height * 0.575)
    top = 0
    bottom = new_height

    # Crop box: (left, top, right, bottom)
    image = image.crop((0, top, original_width, bottom))

    print(f"Cropped image from {original_width}x{original_height} to {original_width}x{new_height}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": DOTS_OCR_LAYOUT_PROMPT},
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
        max_length=2800,
        return_tensors="pt",
    )

    input_ids = _dots_ocr_stack_input_ids_for_dp(inputs["input_ids"])
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    image_grid_thw = inputs["image_grid_thw"]

    pipeline.warmup(input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    DispatchManager.clear_timings()
    start_time = time.time()
    generated_ids = pipeline.generate(
        input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=180,
        stop_on_eos=False,
    )
    ttnn.synchronize_device(mesh_device)
    end_time = time.time()

    if isinstance(generated_ids[0], list):
        streams = [processor.decode(seq, skip_special_tokens=True) for seq in generated_ids]
        decoded = "\n--- stream ---\n".join(streams)
        num_tokens = sum(len(seq) for seq in generated_ids)
    else:
        decoded = processor.decode(generated_ids, skip_special_tokens=True)
        num_tokens = len(generated_ids)
    print(f"Pipeline VISION OUTPUT: {decoded}")

    total_time = end_time - start_time
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


@pytest.mark.parametrize(
    "device_params",
    [_dots_ocr_device_params()],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [_resolve_mesh_device_shape()],
    indirect=True,
)
def test_dots_ocr_vision_tower_dump(mesh_device):
    """Dump the full vision-tower output embeddings for the demo image.

    Used to validate vision-attention tensor parallelism: run once on a single
    chip (``MESH_DEVICE=P100``) and once on 4 chips (``MESH_DEVICE=P150x4``),
    then PCC-compare the two saved tensors. Real model weights + identical input
    mean the only difference is the attention head-sharding, so a high PCC proves
    the TP path matches the replicated single-chip path. Output is saved to
    ``/tmp/vis_tower_out_<MESH_DEVICE>.pt``.
    """
    pytest.importorskip("qwen_vl_utils")
    import json

    from qwen_vl_utils import process_vision_info
    from PIL import Image
    import requests
    from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoVideoProcessor, Qwen2_5_VLProcessor

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    image_link = "https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/demo/demo_image1.jpg"
    image_processor = AutoImageProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    _tokenizer = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    video_processor = AutoVideoProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    with open(os.path.join(DOTS_OCR_LOCAL_PATH, "chat_template.json")) as f:
        chat_template = json.load(f)["chat_template"]
    processor = Qwen2_5_VLProcessor(image_processor, _tokenizer, video_processor, chat_template=chat_template)
    processor.image_token = "<|imgpad|>"
    processor.image_token_id = 151665

    image = Image.open(requests.get(image_link, stream=True).raw)
    w, h = image.size
    image = image.crop((0, 0, w, int(h * 0.575)))
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe this image."}],
        }
    ]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    image_grid_thw = inputs["image_grid_thw"]

    hf_model = AutoModelForCausalLM.from_pretrained(
        DOTS_OCR_LOCAL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).eval()
    vision_tower = TTNNDotsOCRVisionTower.from_torch(hf_model.vision_tower, hf_model.config)
    vision_tower._unique_name = "vision_tower"
    vision_tower.override_children_module_names()
    del hf_model

    set_device(vision_tower, mesh_device, register_forward_hook=False, dump_visualization=False)
    vision_tower.preprocess_weights()
    vision_tower.move_weights_to_device()

    out = vision_tower.forward(pixel_values, image_grid_thw)
    ttnn.synchronize_device(mesh_device)

    num_devices = int(mesh_device.get_num_devices()) if hasattr(mesh_device, "get_num_devices") else 1
    if num_devices > 1:
        # The vision tower output is col-sharded along the channel/hidden dim
        # (H/num_devices per device). Concat the per-device shards along -1 to
        # reconstruct the full [1, 1, N, H] for a like-for-like PCC vs P100.
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    else:
        out_t = ttnn.to_torch(out)
    out_t = out_t.to(torch.float32)

    md = os.environ.get("MESH_DEVICE", "NONE")
    path = f"/tmp/vis_tower_out_{md}.pt"
    torch.save(out_t, path)
    print(f"[vis-dump] MESH_DEVICE={md} num_devices={num_devices} output shape={tuple(out_t.shape)} saved={path}")
