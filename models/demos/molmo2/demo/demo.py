# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Molmo2-8B demo: text / image / video generation on T3K.

Mirrors the structure of models/demos/qwen3_vl/demo/demo.py:
  - pytest.mark.parametrize over JSON prompt files
  - model loaded once per test; warm-up prefill on first batch
  - greedy (argmax) or temperature sampling
  - per-iteration throughput logging
  - profiler timers for prefill / compile / decode

Supported prompt JSON format (same as input_prompts argument):
  Text-only:    [[ {"role": "user", "content": "..."} ]]
  Image+text:   [[ {"role": "user", "content": [{"type":"image","image":"path_or_url"}, {"type":"text","text":"..."}]} ]]
  Video+text:   [[ {"role": "user", "content": [{"type":"video","video":"path_or_url"}, {"type":"text","text":"..."}]} ]]

  Each outer list entry is one user / prompt.

Run:
  export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
  MESH_DEVICE=T3K pytest models/demos/molmo2/demo/demo.py -v -k "text_only"
  MESH_DEVICE=T3K pytest models/demos/molmo2/demo/demo.py -v -k "image"
  MESH_DEVICE=T3K pytest models/demos/molmo2/demo/demo.py -v -k "video"

  # Custom prompt file:
  MESH_DEVICE=T3K pytest models/demos/molmo2/demo/demo.py --input_prompts my.json -v

CLI overrides (passed via pytest addopts or --flag):
  --input_prompts PATH    JSON file to load
  --max_new_tokens N      tokens to generate (default from parametrize)
  --batch_size N          users per batch
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.demos.molmo2.tt.vision_inputs import vision_inputs_from_processor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SAMPLE_DIR = Path(__file__).parent / "sample_inputs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_inputs(path: str, batch_size: int) -> list:
    with open(path) as f:
        data = json.load(f)
    # Each entry in the JSON is a list of messages for one user
    prompts = [entry if isinstance(entry, list) else [entry] for entry in data]
    if len(prompts) < batch_size:
        # Repeat to fill batch
        prompts = (prompts * ((batch_size // len(prompts)) + 1))[:batch_size]
    return prompts[:batch_size]


def build_inputs(processor, prompts: list):
    """Convert a list of message-lists into HF processor inputs.

    Handles text-only, image, and video prompts by inspecting content type.
    Images / videos are loaded from local paths or URLs using PIL / decord.
    """
    texts = processor.apply_chat_template(
        [p for p in prompts],  # list of message-lists
        tokenize=False,
        add_generation_prompt=True,
    )
    if isinstance(texts, str):
        texts = [texts]

    # Collect images and videos referenced in the prompts
    images, videos = [], []
    has_image, has_video = False, False
    for conversation in prompts:
        for msg in conversation:
            for item in msg.get("content") if isinstance(msg.get("content"), list) else []:
                if item.get("type") == "image":
                    src = item.get("image", "")
                    images.append(_load_image(src))
                    has_image = True
                elif item.get("type") == "video":
                    src = item.get("video", "")
                    videos.append(src)  # pass path to processor (uses decord)
                    has_video = True

    kwargs = {"text": texts, "return_tensors": "pt", "padding": True}
    if has_image:
        kwargs["images"] = images
    if has_video:
        kwargs["videos"] = videos[0] if len(videos) == 1 else videos

    return processor(**kwargs)


def _load_image(src: str) -> Image.Image:
    if src.startswith("http"):
        from io import BytesIO

        import requests

        r = requests.get(src, timeout=10)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    return Image.open(src).convert("RGB")


# ---------------------------------------------------------------------------
# Model / mesh fixtures (module-scoped — loaded once per test)
# ---------------------------------------------------------------------------

_MESH_SHAPE = {
    "N150": (1, 1),
    "N300": (1, 2),
    "T3K": (1, 8),
    "TG": (8, 4),
}.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))


@pytest.fixture(scope="module")
def mesh_device():
    rows, cols = _MESH_SHAPE if isinstance(_MESH_SHAPE, tuple) else (1, _MESH_SHAPE)
    is_single = rows * cols == 1
    if not is_single:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    logger.info(f"Opened {rows}×{cols} mesh: {device.get_num_devices()} devices")
    yield device
    ttnn.close_mesh_device(device)
    if not is_single:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def tt_model(mesh_device):
    """Load full Molmo2 TTNN model (weights + config)."""
    from models.demos.molmo2.tt.hf_loading import load_molmo2_state_dict
    from models.demos.molmo2.tt.model import TtMolmo2Model
    from models.demos.molmo2.tt.model_config import Molmo2Config
    from models.tt_transformers.tt.ccl import TT_CCL

    logger.info("Loading Molmo2-8B weights (bfloat16)...")
    state_dict = load_molmo2_state_dict(dtype=torch.bfloat16)

    cfg = Molmo2Config(mesh_device=mesh_device)
    ccl = TT_CCL(mesh_device)
    from pathlib import Path

    weight_cache = Path("/tmp/molmo2_weight_cache")
    weight_cache.mkdir(parents=True, exist_ok=True)
    model = TtMolmo2Model(
        mesh_device=mesh_device,
        tt_ccl=ccl,
        state_dict=state_dict,
        weight_cache_path=weight_cache,
        dtype=ttnn.bfloat16,
        configuration=cfg,
    )
    del state_dict
    # Same warm-up as the vLLM server (generator_vllm.initialize_vllm_model): compile every prefill bucket and the
    # vision kernels BEFORE the decode trace is captured. New compilations/allocations after trace capture can
    # corrupt or hang the traced decode (first-occurrence video shapes hang in vision pooling otherwise).
    from models.demos.molmo2.tt.model import PREFILL_BUCKETS

    model.warmup_all_buckets(bucket_sizes=PREFILL_BUCKETS, use_trace=False)
    model.warmup_vision_compile()
    logger.info("Model ready")
    return model, cfg


@pytest.fixture(scope="module")
def processor():
    from models.demos.molmo2.tt.hf_loading import load_molmo2_processor

    return load_molmo2_processor()


# ---------------------------------------------------------------------------
# Core demo function
# ---------------------------------------------------------------------------


def run_demo(
    mesh_device,
    tt_model,
    processor,
    request,
    input_prompts_path: str,
    max_new_tokens: int,
    batch_size: int,
    temperature: float = 0.0,
    stop_at_eos: bool = True,
):
    model, cfg = tt_model
    eos = processor.tokenizer.eos_token_id

    # CLI overrides
    input_prompts_path = request.config.getoption("--input_prompts") or input_prompts_path
    max_new_tokens = request.config.getoption("--max_new_tokens") or max_new_tokens
    batch_size = request.config.getoption("--batch_size") or batch_size

    logger.info(f"Loading prompts from {input_prompts_path} (batch={batch_size})")
    prompts = load_inputs(input_prompts_path, batch_size)

    # TtMolmo2Model decodes a single sequence (KV-cache slot 0), so users run back to back.
    responses = []
    total_tokens, decode_elapsed = 0, 0.0
    for u, prompt in enumerate(prompts):
        t0 = time.time()
        proc_out = build_inputs(processor, [prompt])
        input_ids = proc_out["input_ids"]  # [1, S]
        token_type_ids = proc_out.get("token_type_ids")
        pixel_values, pool_idx = vision_inputs_from_processor(proc_out)
        S = input_ids.shape[1]
        logger.info(f"User {u}: seq_len={S}  preprocess={time.time()-t0:.2f}s")
        # The KV cache / RoPE tables hold cfg.max_seq_len (36864 = model max_position_embeddings) positions.
        assert S < cfg.max_seq_len, f"prompt of {S} tokens does not fit the {cfg.max_seq_len}-token context"
        user_max_new = min(max_new_tokens, cfg.max_seq_len - S)
        if user_max_new < max_new_tokens:
            logger.warning(f"Clamping max_new_tokens {max_new_tokens} -> {user_max_new} (context {cfg.max_seq_len})")

        # ---- Warm-up (compile) pass — not timed ----
        # Run a minimal text-only prefill to trigger TTNN JIT compilation before measuring.
        if not getattr(model, "_demo_warmed_up", False):
            logger.info("Warm-up compile pass (not timed)...")
            t_warmup = time.time()
            warmup_ids = input_ids[:1, :32]  # first 32 tokens, text path only
            _ = model.forward_prefill(input_ids=warmup_ids, pixel_values=None, user_id=0)
            # One decode step to compile the decode kernels
            _ = model.forward_decode_step(int(warmup_ids[0, -1].item()), warmup_ids.shape[1])
            model._demo_warmed_up = True
            logger.info(f"  Warm-up done in {time.time()-t_warmup:.2f}s (not counted in timing)")

        # ---- Timed prefill ----
        t_prefill = time.time()
        logits = model.forward_prefill(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pooled_patches_idx=pool_idx,
            token_type_ids=token_type_ids,
            user_id=0,
        )
        prefill_ms = (time.time() - t_prefill) * 1000
        logger.info(f"  Prefill: {prefill_ms:.0f}ms  ({S} tokens, {S / (prefill_ms / 1000):.0f} tok/s)")

        # ---- Timed decode loop ----
        next_token = int(torch.argmax(logits[0, -1]).item())
        output_ids = []
        current_pos = S
        t_decode_start = time.time()
        for iteration in range(user_max_new):
            if stop_at_eos and next_token == eos:
                break
            output_ids.append(next_token)
            if iteration == user_max_new - 1:
                break
            t_iter = time.time()
            lg = model.forward_decode_step(next_token, current_pos)
            current_pos += 1
            if temperature == 0:
                next_token = int(lg[0].argmax().item())
            else:
                probs = torch.softmax(lg[0].float() / temperature, dim=-1)
                next_token = int(torch.multinomial(probs, 1).item())
            if iteration < 3 or iteration % 10 == 0:
                iter_ms = (time.time() - t_iter) * 1000
                logger.info(f"  iter {iteration+1}: {iter_ms:.0f}ms  {1000 / iter_ms:.1f} tok/s/user")
        user_decode_s = time.time() - t_decode_start
        decode_elapsed += user_decode_s
        total_tokens += len(output_ids)

        text = processor.tokenizer.decode(output_ids, skip_special_tokens=True)
        responses.append(text)
        prompt_text = (
            prompt[-1]["content"]
            if isinstance(prompt[-1]["content"], str)
            else next((c["text"] for c in prompt[-1]["content"] if c.get("type") == "text"), "")
        )
        logger.info(
            f"\n=== USER {u} ===\nPROMPT:   {prompt_text[:120]}\nRESPONSE: {text.strip()}\n"
            f"({len(output_ids)} tokens, {len(output_ids) / max(user_decode_s, 1e-9):.1f} tok/s)\n"
        )

    overall_tps = total_tokens / decode_elapsed if decode_elapsed > 0 else 0
    logger.info(f"Decode complete: {total_tokens} tokens in {decode_elapsed:.1f}s ({overall_tps:.1f} tok/s)")
    return responses


# ---------------------------------------------------------------------------
# Test parametrization (mirrors qwen3_vl/demo/demo.py structure)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_prompts, max_new_tokens, batch_size",
    [
        (  # text-only, batch=1
            str(SAMPLE_DIR / "text_only.json"),
            128,
            1,
        ),
        (  # text-only, batch=2
            str(SAMPLE_DIR / "text_only.json"),
            128,
            2,
        ),
        (  # image + text, batch=1
            str(SAMPLE_DIR / "image_demo.json"),
            200,
            1,
        ),
        (  # video + text, batch=1
            str(SAMPLE_DIR / "video_demo.json"),
            64,
            1,
        ),
        (  # 2 images + text, batch=1 — verifies multi-image via native image path
            str(SAMPLE_DIR / "multi_image_2_demo.json"),
            200,
            1,
        ),
        (  # 20 images + text, batch=1 — stress test near max=23 (36864//(8*196))
            str(SAMPLE_DIR / "multi_image_20_demo.json"),
            200,
            1,
        ),
    ],
    ids=[
        "text_only-batch1",
        "text_only-batch2",
        "image-batch1",
        "video-batch1",
        "multi-image-2-batch1",
        "multi-image-20-batch1",
    ],
)
def test_demo(
    mesh_device,
    tt_model,
    processor,
    request,
    input_prompts,
    max_new_tokens,
    batch_size,
):
    """Run Molmo2-8B end-to-end demo on T3K for text / image / video inputs."""
    responses = run_demo(
        mesh_device=mesh_device,
        tt_model=tt_model,
        processor=processor,
        request=request,
        input_prompts_path=input_prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        temperature=0.0,
        stop_at_eos=True,
    )

    assert responses, "No output generated"
    for u, r in enumerate(responses):
        assert len(r.strip()) > 0, f"User {u}: empty response"
        logger.info(f"User {u} response length: {len(r)} chars")
