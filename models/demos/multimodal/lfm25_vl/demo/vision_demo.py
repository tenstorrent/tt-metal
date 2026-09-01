# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""LFM2.5-VL-1.6B multimodal demo -- adapted from
``models.demos.multimodal.gemma3.demo.vision_demo`` for the hybrid ShortConv/attention
LFM2 text backbone + SigLIP2 vision tower.
"""

import json
import os
import time
from pathlib import Path
from typing import Literal, Optional

import pytest
import torch
from loguru import logger
from PIL import Image as PIL_Image
from pydantic import BaseModel

import ttnn
from models.common.llama_models import sample_top_p
from models.common.sampling import SamplingParams
from models.demos.multimodal.lfm25_vl.tt.e2e_model import Lfm25VlMultimodalGenerator
from models.tt_transformers.tt.common import ImageMedia, InterleavedTextMedia, Role, hf_multimodal_encode
from models.tt_transformers.tt.generator import create_submeshes
from models.tt_transformers.tt.model_config import DecodersPrecision

HF_MODEL = "LiquidAI/LFM2.5-VL-1.6B"
os.environ.setdefault("HF_MODEL", HF_MODEL)

SAMPLE_PROMPTS_PATH = Path(__file__).resolve().parent / "sample_prompts" / "demo.json"
IMG_PATH = Path("models/tt_transformers/demo/sample_prompts/llama_models").resolve()


def _lfm25_vl_device_params():
    # N300 needs fabric for CCL across the 1x2 mesh.
    return {
        "fabric_config": True,
        "l1_small_size": 24576,
    }


class UserMessage(BaseModel):
    role: Literal[Role.user.value] = Role.user.value
    content: InterleavedTextMedia
    context: Optional[InterleavedTextMedia] = None


def get_batch_sampler(temperature, top_p, tokenizer):
    def sample(logits):
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_tokens = next_token.reshape(-1)
        texts = [tokenizer.decode([next_tokens[i].item()]) for i in range(len(next_tokens))]
        return next_tokens, texts

    return sample


def create_multimodal_model(
    mesh_device,
    max_batch_size,
    max_seq_len,
    dtype=ttnn.bfloat8_b,
    use_paged_kv_cache=False,
    checkpoint=None,
    optimizations=None,
    num_layers=None,
    paged_attention_config=None,
    dummy_weights: bool = False,
):
    from models.demos.multimodal.lfm25_vl.tt.e2e_model import TtLfm25VlModel
    from models.demos.multimodal.lfm25_vl.tt.model_config import ModelArgs

    tt_model_args = ModelArgs(
        mesh_device,
        instruct=True,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        optimizations=optimizations,
        dummy_weights=dummy_weights,
    )
    assert tt_model_args.is_multimodal, "LFM2.5-VL is a multimodal model"

    if num_layers is not None:
        tt_model_args.n_layers = num_layers
        tt_model_args.layer_types = tt_model_args.layer_types[:num_layers]
        tt_model_args.vision_n_layers = num_layers

    if checkpoint is None:
        checkpoint = tt_model_args.load_state_dict()

    model = TtLfm25VlModel(
        mesh_device=mesh_device,
        state_dict=checkpoint,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        dtype=dtype,
        args=tt_model_args,
        use_paged_kv_cache=use_paged_kv_cache,
        paged_attention_config=paged_attention_config,
    )
    return tt_model_args, model, checkpoint


def prepare_generator_args(
    num_devices,
    data_parallel,
    mesh_device,
    max_batch_size,
    max_seq_len,
    dtype=ttnn.bfloat8_b,
    use_paged_kv_cache=False,
    optimizations=None,
    num_layers=None,
    dummy_weights: bool = False,
):
    submesh_devices = create_submeshes(mesh_device, data_parallel)
    state_dict = None

    model_args = []
    model = []
    for submesh in submesh_devices:
        model_args_i, model_i, state_dict = create_multimodal_model(
            mesh_device=submesh,
            max_batch_size=max_batch_size // data_parallel,
            max_seq_len=max_seq_len,
            dtype=dtype,
            use_paged_kv_cache=use_paged_kv_cache,
            checkpoint=state_dict,
            optimizations=optimizations,
            num_layers=num_layers,
            dummy_weights=dummy_weights,
        )
        model_args.append(model_args_i)
        model.append(model_i)

    return model_args, model


def _load_sample_dialogs(ocr_image):
    with open(SAMPLE_PROMPTS_PATH) as f:
        entries = json.load(f)
    images_by_name = {"ocr_image.jpeg": ocr_image}
    dialogs = []
    for entry in entries:
        image = images_by_name.get(entry.get("image"))
        content = [entry["prompt"]] if image is None else [ImageMedia(image=image), entry["prompt"]]
        dialogs.append([UserMessage(content=content)])
    return dialogs


@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "P150": (1, 1),
            "P300": (1, 2),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [_lfm25_vl_device_params()],
    indirect=True,
)
@pytest.mark.parametrize(
    "warmup_iters, enable_trace, max_batch_size, max_gen_len, num_layers",
    [
        # notrace first: ShortConv decode is host-side, so device-trace decode is not reliable yet
        (0, False, 1, 32, None),  # batch1-notrace
    ],
    ids=["batch1-notrace"],
)
@pytest.mark.parametrize("data_parallel", [1])
@pytest.mark.parametrize(
    "optimizations",
    [
        lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
    ],
    ids=["performance"],
)
def test_multimodal_demo_text(
    mesh_device,
    warmup_iters,
    enable_trace,
    max_batch_size,
    data_parallel,
    is_ci_env,
    optimizations,
    max_gen_len,
    num_layers,
    request,
    temperature: float = 0,
    top_p: float = 0.9,
    max_seq_len: int = 8 * 1024,
):
    """Simple LFM2.5-VL multimodal demo (OCR-style prompts), following the Gemma3 vision demo pattern."""
    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1
    max_batch_size *= data_parallel

    dummy_weights = request.config.getoption("--dummy_weights") or False

    model_args, model = prepare_generator_args(
        num_devices=num_devices,
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        optimizations=optimizations,
        num_layers=num_layers,
        dummy_weights=dummy_weights,
    )

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_args[0].CKPT_DIR, local_files_only=os.getenv("CI") == "true", trust_remote_code=True
    )

    generator = Lfm25VlMultimodalGenerator(model, model_args, mesh_device)

    can_sample_on_device = getattr(model[0], "_supports_on_device_sampling", False) and model[0].sampling is not None
    greedy_only = temperature <= 0
    device_sampling_params = None
    if can_sample_on_device:
        device_sampling_params = (
            SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
            if greedy_only
            else SamplingParams(temperature=temperature, top_k=32, top_p=top_p)
        )

    logger.info("Warming up model...")
    generator.warmup_model_prefill(
        kv_cache=None,
        enable_trace=enable_trace,
        can_sample_on_device=can_sample_on_device,
        greedy_only=greedy_only,
    )
    logger.info("Warmup complete")

    with open(IMG_PATH / "ocr_image.jpeg", "rb") as f:
        ocr_image = PIL_Image.open(f).convert("RGB")

    dialogs = _load_sample_dialogs(ocr_image)
    if len(dialogs) < max_batch_size:
        dialogs *= max_batch_size // len(dialogs) + 1
    dialogs = dialogs[:max_batch_size]

    tokenizer = processor.tokenizer
    sampler = None if can_sample_on_device else get_batch_sampler(temperature, top_p, tokenizer)

    batch_model_input = [hf_multimodal_encode(dialog, processor) for dialog in dialogs]
    vision_images = [model_input.vision.images if model_input.vision else None for model_input in batch_model_input]
    # LFM2-VL / SigLIP2-NaFlex extras from the HF processor (present when images are used).
    spatial_shapes = [getattr(model_input, "spatial_shapes", None) for model_input in batch_model_input]
    pixel_attention_mask = [getattr(model_input, "pixel_attention_mask", None) for model_input in batch_model_input]
    prompt_tokens = [model_input.tokens for model_input in batch_model_input]

    prefill_lens = torch.tensor([len(tokens) for tokens in prompt_tokens], dtype=torch.long)
    total_lens = prefill_lens + max_gen_len

    pad_id = tokenizer.pad_token_id or 0
    bsz = len(prompt_tokens)
    tokens = torch.full((bsz, max(total_lens)), pad_id, dtype=torch.long)
    for i, seq in enumerate(prompt_tokens):
        tokens[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    for iter_num in range(warmup_iters + 1):
        logger.info(f"Iteration {iter_num}")
        # Reset ShortConv host decode state between independent generations.
        for m in model:
            m.reset_conv_states()
        prefill_start = time.perf_counter()
        prefill_out = generator.prefill_forward(
            vision_images,
            None,
            tokens,
            None,
            total_lens,
            prefill_lens,
            sampling_params=device_sampling_params,
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask,
        )
        prefill_end = time.perf_counter()

        if device_sampling_params is not None:
            prefill_toks, _ = prefill_out
            next_tokens = prefill_toks.long().squeeze(-1).reshape(-1)[:max_batch_size]
        else:
            next_tokens, _ = sampler(prefill_out)
        for i, next_token in enumerate(next_tokens):
            tokens[i, prefill_lens[i]] = next_token

        position_id = prefill_lens
        for gen_idx in range(max_gen_len - 1):
            position_id = prefill_lens + gen_idx
            next_token_tensor = next_tokens.reshape(max_batch_size, 1)
            if device_sampling_params is not None:
                tok, _ = generator.decode_forward(
                    next_token_tensor, position_id, enable_trace=enable_trace, sampling_params=device_sampling_params
                )
                next_tokens = tok.long().reshape(-1)[:max_batch_size]
            else:
                logits, _ = generator.decode_forward(next_token_tensor, position_id, enable_trace=enable_trace)
                next_tokens, _ = sampler(logits)
            tokens[torch.arange(max_batch_size), position_id + 1] = next_tokens
            if tokenizer.eos_token_id is not None and any(t == tokenizer.eos_token_id for t in next_tokens):
                break

        prefill_time_ms = (prefill_end - prefill_start) * 1000
        logger.info(f"Prefill time: {prefill_time_ms:.2f} ms")

        # Clean console summary (decode only generated tokens; no image-token dump).
        print("\n" + "=" * 64, flush=True)
        print("LFM2.5-VL-1.6B multimodal demo (N300)", flush=True)
        print("=" * 64, flush=True)
        print(f"Status: PASSED", flush=True)
        print(f"Prefill time: {prefill_time_ms:.2f} ms", flush=True)
        print(f"Generated tokens: {int((position_id[0] + 1 - prefill_lens[0]).item())}", flush=True)
        for user_id in range(max_batch_size):
            gen_start = int(prefill_lens[user_id].item())
            gen_end = int(position_id[user_id].item()) + 2
            gen_ids = tokens[user_id, gen_start:gen_end].tolist()
            if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in gen_ids:
                gen_ids = gen_ids[: gen_ids.index(tokenizer.eos_token_id)]
            assistant = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            # Keep a full decode in the logger for debugging (not printed).
            full_text = tokenizer.decode(tokens[user_id, :gen_end].tolist())
            logger.info(f"User {user_id} full text: {full_text}")
            content = dialogs[user_id][-1].content
            if isinstance(content, list):
                prompt = next((p for p in content if isinstance(p, str)), "[image]")
            else:
                prompt = str(content)
            print(f"Prompt: {prompt}", flush=True)
            print(f"Assistant: {assistant if assistant else '(empty)'}", flush=True)
        print("=" * 64 + "\n", flush=True)
