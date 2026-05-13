# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test for Mistral-24B End-to-End Vision-Text Pipeline"""

import time

import torch
import pytest
from loguru import logger
import os
import ttnn
import torch.nn.functional as F
from models.tt_transformers.tt.ccl import TT_CCL
from models.common.sampling import SamplingParams
from models.tt_transformers.tt.common import PagedAttentionConfig

from models.tt_transformers.tt.model_config import DecodersPrecision
from models.experimental.mistral_24b.tt.model import MistralTransformer as Transformer

from models.experimental.mistral_24b.tt.generator import MistralGenerator

from models.experimental.mistral_24b.tt.pipeline.vision_model import TtMistralVisionTransformer
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole

from models.tt_transformers.tt.model_config import ModelArgs
from transformers import AutoProcessor, AutoModelForVision2Seq

import re


def run_reference_demo_pipeline(messages, model_id="mistralai/Mistral-Small-3.1-24B-Instruct-2503"):
    """
    Run Hugging Face reference demo model (Vision-Text pipeline) using given messages.
    """
    logger.info("Running reference HF vision-text model...")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model.eval()

    # Apply chat template
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, padding=True, padding_side="left"
    )

    # Extract images (already loaded)
    image_inputs = []
    for msg in messages:
        for item in msg["content"]:
            if item["type"] == "image":
                image_inputs.append(item["image"])

    # Tokenize and move to model device
    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.0,
            top_p=0.9,
            do_sample=False,
            pad_token_id=model.config.pad_token_id,
        )

    # Decode
    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    logger.info(f"HF reference model output: {output}")

    chat = parse_chat_output(output)
    display_chat(logger, chat)

    return output


def parse_chat_output(text):
    """Parse chat output format from generated text."""
    pattern = r"<\|(?P<role>user|assistant)\|>\s*(?P<message>.*?)(?=<\|(?:user|assistant|end)\|>|$)"
    matches = re.finditer(pattern, text, re.DOTALL)
    return [(match.group("role"), match.group("message").strip()) for match in matches]


def display_chat(logger, conversation):
    """Display chat conversation in formatted output."""
    for role, message in conversation:
        if role == "user":
            logger.info(f"👤 User: {message}")
        elif role == "assistant":
            logger.info(f"🤖 Assistant: {message}")


# Greedy on-device decode (Phase C harness): avoids full-logits readback + host argmax in the decode loop.
DECODE_GREEDY_SAMPLING = SamplingParams(
    temperature=0.0,
    top_k=1,
    top_p=1.0,
    enable_log_probs=False,
)


def next_token_ids_from_decode_output(decode_output):
    """Flatten token ids from ``decode_forward`` when ``sampling_params`` is set (on-device sampling)."""
    if isinstance(decode_output, tuple):
        tok, _ = decode_output
    else:
        tok = decode_output
    return tok.reshape(-1).long()


def read_first_token_id_from_device(tt_decode_output):
    """Per-step token read for greedy on-device sampling when decode_forward was
    called with read_from_device=False. Reads just slot 0 from one device's
    replica of the [1,1,1,padded_batch] token tensor; tokens are replicated
    across the mesh, so the value matches what process_decode_output_host would
    return, while skipping its per-device .cpu() + reshape/concat (the 30-70ms
    host-side gap between Tilize ops in the decoder perf report)."""
    item = tt_decode_output[0]
    tt_tok = item[0] if isinstance(item, tuple) else item
    return int(ttnn.to_torch(ttnn.get_device_tensors(tt_tok)[0]).flatten()[0].item())


def log_e2e_performance_measurements(
    *,
    batch_size,
    num_prefill_tokens,
    inference_prefill_time,
    inference_decode_time,
    decode_step_times,
    compile_prefill_time=None,
    compile_decode_time=None,
    vision_model_prefill_time=None,
    full_run_time=None,
):
    """
    Log throughput and latency metrics in the same shape as Qwen VL demos
    (see models/demos/qwen3_vl/demo/demo.py measurements + Performance metrics block).
    Vision/text prefill are not split here without a profiler; vision timing is optional.

    For decode, ``inference_decode_time`` is total wall time for the decode region and
    ``decode_step_times`` may be a synthetic list of identical ``wall_time / N`` entries
    (so ``num_decode_tokens`` matches ``N``) when per-call ``decode_forward`` timing is
    not meaningful (e.g. non-blocking trace with ``read_from_device=False``).
    """
    avg_time_to_first_token = inference_prefill_time / batch_size if batch_size else inference_prefill_time
    prefill_tok_s = (num_prefill_tokens / inference_prefill_time * batch_size) if inference_prefill_time > 0 else 0.0
    num_decode_tokens = len(decode_step_times)
    decode_tok_s_user = (num_decode_tokens / inference_decode_time) if inference_decode_time > 0 else 0.0
    decode_tok_s = decode_tok_s_user * batch_size

    measurements = {
        "prefill_tokens": num_prefill_tokens,
        "compile_prefill": compile_prefill_time,
        "compile_decode": compile_decode_time,
        "vision_model_prefill": vision_model_prefill_time,
        "inference_prefill": inference_prefill_time,
        "inference_decode": inference_decode_time,
        "prefill_time_to_token": avg_time_to_first_token,
        "prefill_t/s": prefill_tok_s,
        "decode_t/s/u": decode_tok_s_user,
        "decode_t/s": decode_tok_s,
        "num_decode_tokens": num_decode_tokens,
        "Total compile time": (
            (compile_prefill_time + compile_decode_time)
            if compile_prefill_time is not None and compile_decode_time is not None
            else None
        ),
        "Full demo runtime": full_run_time,
    }

    logger.info("")
    logger.info("=== Performance measurements (E2E test) ===")
    for key, value in measurements.items():
        if value is None:
            continue
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6g}")
        else:
            logger.info(f"  {key}: {value}")

    logger.info("")
    logger.info("=== Performance metrics ===")
    logger.info(
        f"Prefill tokens: {num_prefill_tokens} | TTFT (prefill to first token): "
        f"{avg_time_to_first_token * 1000:.2f}ms"
    )
    logger.info(
        f"Prefill throughput: {prefill_tok_s:.2f} tokens/s " f"(inference_prefill={inference_prefill_time:.4f}s)"
    )
    if decode_step_times:
        first_decode_s = decode_step_times[0]
        uniform_steps = len(decode_step_times) > 1 and all(s == first_decode_s for s in decode_step_times)
        step_label = "Mean decode step (wall/N)" if uniform_steps else "1st decode step"
        if first_decode_s > 0:
            logger.info(
                f"{step_label}: {first_decode_s * 1000:.2f}ms "
                f"[{1.0 / first_decode_s:.2f} t/s/u, {(1.0 / first_decode_s) * batch_size:.2f} t/s]"
            )
        else:
            logger.info(f"{step_label}: {first_decode_s * 1000:.2f}ms")
    logger.info(
        f"Decode: {num_decode_tokens} tokens in {inference_decode_time:.4f}s → "
        f"{decode_tok_s_user:.2f} tok/s/user, {decode_tok_s:.2f} tok/s aggregate"
    )
    if full_run_time is not None:
        logger.info(f"Full run (prefill + decode wall time): {full_run_time:.4f}s")
    logger.info("===")


def setup_vision_model_args(weights, max_seq_len, batch_size, mesh_device, optimizations):
    """Setup model arguments for vision-enabled model (Single Responsibility)."""
    instruct = True if weights == "instruct" else False

    model_args = ModelArgs(
        mesh_device=mesh_device,
        instruct=instruct,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
    )

    return model_args, instruct


SYSTEM_PROMPT = """You are mistralai/Mistral-Small-3.1-24B-Instruct-2503, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.
You power an AI assistant called Le Chat.
Your knowledge base was last updated on 2023-10-01.
The current date is 2026-05-07.

When you're not sure about some information, you say that you don't have the information and don't make up anything.
If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").
You are always very attentive to dates, in particular you try to resolve dates (e.g. "yesterday" is {yesterday}) and when asked about information at specific dates, you discard information that is at another date.
You follow these instructions in all languages, and always respond to the user in the language they use or request.
Next sections describe the capabilities that you have.

# WEB BROWSING INSTRUCTIONS

You cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to do so, you clarify the situation and ask the user to copy paste the text directly in the chat.

# MULTI-MODAL INSTRUCTIONS

You have the ability to read images, but you cannot generate images. You cannot read nor transcribe audio files or videos."""


def setup_vision_prompts_and_tokenizer(model_args, instruct):
    """Setup multimodal prompts and tokenizer for vision-enabled model."""

    image_url = "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/europe.png"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Which of the depicted countries has the best food? Which the second and third and fourth? "
                        "Name the country, its color on the map and one its city that is visible on the map, but is "
                        "not the capital. Make absolutely sure to only name a city that can be seen on the map."
                    ),
                },
                {"type": "image", "image": image_url},
            ],
        },
    ]

    tokenizer = model_args.tokenizer

    return messages, tokenizer


def process_vision_info(messages):
    """Extract images from messages.

    Supports content as a plain string (e.g. system message) or as a list of dicts
    where each item has a `type` field of "image" (with `image`) or "image_url"
    (with `image_url.url`).
    """

    image_inputs = []
    video_inputs = None  # Not used

    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "image" and "image" in item:
                image_inputs.append(item["image"])
            elif item_type == "image_url":
                url = item.get("image_url", {})
                if isinstance(url, dict) and "url" in url:
                    image_inputs.append(url["url"])
    return image_inputs, video_inputs


def process_real_vision_inputs(messages, model_args):
    """Process real image inputs using AutoProcessor (Interface Segregation)."""

    processor = AutoProcessor.from_pretrained(os.getenv("HF_MODEL"))

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, padding=True, padding_side="left"
    )

    image_inputs, video_inputs = process_vision_info(messages)

    encoded = processor(
        text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt", return_dict=True
    ).to("cpu", dtype=torch.bfloat16)

    input_ids = encoded["input_ids"]
    pixel_values = encoded["pixel_values"] if "pixel_values" in encoded else None
    attention_mask = encoded["attention_mask"] if "attention_mask" in encoded else None
    image_sizes = encoded["image_sizes"] if "image_sizes" in encoded else None

    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "attention_mask": attention_mask,
        "image_sizes": image_sizes,
        "processor": processor,
    }


def load_separate_models_like_test_end2end(model_args, mesh_device, dtype, paged_attention, page_params):
    """Load separate vision and text models following test_end2end.py pattern."""
    state_dict = model_args.load_state_dict()

    vision_prefix = "vision_tower."
    # Setup paged attention config (exactly like test_end2end.py)
    paged_attention_config = None
    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )

    tt_ccl = TT_CCL(mesh_device)
    # Load vision model (exactly like test_end2end.py)
    vision_model = TtMistralVisionTransformer(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        state_dict=state_dict,
        state_dict_prefix=vision_prefix,
        dtype=dtype,
        model_args=model_args,
    )

    # Load text model (exactly like test_end2end.py)
    text_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )
    return vision_model, text_model


def run_generation_exactly_like_test_end2end(
    vision_model,
    text_model,
    processed_inputs,
    model_args,
    page_table=None,
    paged_attention_config=None,
    max_gen_len=20,
    repetition_ngram_size=3,
):
    """Run generation following the EXACT pattern from test_end2end.py."""
    input_ids = processed_inputs["input_ids"]

    logger.info("Running generation exactly like test_end2end.py...")

    logger.info("Running Vision Model...")
    generator = MistralGenerator([text_model], [model_args], vision_model.mesh_device, tokenizer=model_args.tokenizer)
    tt_kv_cache = [[l.attention.layer_past for l in text_model.layers]] if paged_attention_config else None

    input_tokens_prefill_pt = input_ids
    batch_size = input_tokens_prefill_pt.shape[0]
    attention_mask = processed_inputs.get("attention_mask", None)
    if attention_mask is not None:
        decoding_pos = attention_mask.sum(dim=-1).tolist()
    else:
        decoding_pos = [input_tokens_prefill_pt.shape[1]] * batch_size
    prefill_lens = decoding_pos
    encoded_prompts = [input_ids[0].tolist()]

    logger.info("Running prefill...")
    run_t0 = time.perf_counter()
    prefill_t0 = time.perf_counter()
    logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        vision_model=vision_model,
        processed_inputs=processed_inputs,
    )
    prefill_t1 = time.perf_counter()
    inference_prefill_time = prefill_t1 - prefill_t0
    num_prefill_tokens = int(prefill_lens[0])

    prefilled_token = torch.argmax(logits, dim=-1)

    # logits: [1, 1, vocab_size]
    last_logits = logits[0, -1]  # shape: [vocab_size]
    probs = F.softmax(last_logits, dim=-1)

    top_k = 5
    topk_probs, topk_indices = torch.topk(probs, k=top_k)

    all_outputs = [encoded_prompts[0][: prefill_lens[0]]]
    all_outputs[0].append(int(prefilled_token[0].item()))

    current_pos = torch.tensor([decoding_pos[0]])
    out_tok = prefilled_token
    generation_length = max_gen_len

    results = []

    logger.info("Starting decode loop...")
    # Phase A: decode warmups before timed region (trace capture/replay steady-state, not measured).
    decode_warmup_iters = 1
    logger.info(f"Decode warmup: {decode_warmup_iters} steps (not timed)")
    for _ in range(decode_warmup_iters):
        decode_output = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=True,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=DECODE_GREEDY_SAMPLING,
        )
        out_tok = next_token_ids_from_decode_output(decode_output)
        current_pos = current_pos + 1

    # Cache eos id once — avoids tokenizer attribute lookup every step.
    eos_token_id = model_args.tokenizer.eos_token_id

    # Wall-clock the whole decode loop: with read_from_device=False, decode_forward
    # returns after non-blocking trace submit (~µs), so per-call timers are meaningless.
    loop_start = time.perf_counter()
    for _ in range(generation_length):
        # read_from_device=False: keep the sampled token on-device. The trace's
        # sampling op writes the next-step token directly into the trace input
        # buffer, so when reset_inputs=False (steady state after warmup) the host
        # `out_tok` is ignored by decode_forward — we don't need to feed it back.
        # This skips process_decode_output_host's per-device .cpu()+reshape+concat
        # which dominates the 30-70ms per-step gap in the report.
        decode_output = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=True,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=DECODE_GREEDY_SAMPLING,
            read_from_device=False,
        )

        # Lightweight per-step read for EOS / repetition checks.
        token_id = read_first_token_id_from_device(decode_output)
        # Mirror the device's token in `out_tok` so a future reset_inputs=True
        # path (e.g., page_table change) still sees the correct value.
        out_tok = torch.tensor([token_id], dtype=torch.long)

        # Stop if EOS detected
        if token_id == eos_token_id:
            logger.info("EOS token detected, stopping generation.")
            break

        all_outputs[0].append(token_id)
        current_pos = current_pos + 1

    loop_end = time.perf_counter()

    total_decode_time = loop_end - loop_start
    num_decoded = len(all_outputs[0]) - prefill_lens[0]
    _n = max(num_decoded, 1)
    decode_step_times = [total_decode_time / _n] * _n
    inference_decode_time = total_decode_time
    # Final response (exactly like test_end2end.py)
    response = model_args.tokenizer.decode(all_outputs[0], skip_special_tokens=True)
    logger.info(f"Final Generated Response:\n{response}")
    chat = parse_chat_output(response)
    display_chat(logger, chat)

    logger.info(f"Generated {len(results)} tokens successfully")

    run_t1 = time.perf_counter()
    full_run_time = run_t1 - run_t0

    log_e2e_performance_measurements(
        batch_size=batch_size,
        num_prefill_tokens=num_prefill_tokens,
        inference_prefill_time=inference_prefill_time,
        inference_decode_time=inference_decode_time,
        decode_step_times=decode_step_times,
        full_run_time=full_run_time,
    )

    return all_outputs[0]


def validate_e2e_outputs(results, expected_min_tokens=1):
    """Validate end-to-end pipeline outputs.

    Passes when at least ``expected_min_tokens`` decode steps recorded a new token
    in ``results`` (length check only; no per-entry schema checks).
    """
    if not results:
        logger.error("No results generated from E2E pipeline")
        return False

    if len(results) < expected_min_tokens:
        logger.warning(f"Generated only {len(results)} tokens, expected at least {expected_min_tokens}")
        return False
    return True


@pytest.mark.skip(reason="Disabled: see #45992")
@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.timeout(1800)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "weights, layers",
    [
        ("instruct", None),
    ],
    ids=["full"],
)
@pytest.mark.parametrize(
    "paged_attention",
    (
        True,
        # False,
    ),
    ids=(
        "paged_attention",
        # "default_attention",
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (1024 * 8,),  # Use smaller seq_len like test_end2end.py to avoid memory issues
)
@pytest.mark.parametrize(
    "optimizations",
    [
        lambda model_args: DecodersPrecision.accuracy(model_args.n_layers, model_args.model_name),
    ],
    ids=["accuracy"],
)
@pytest.mark.parametrize(
    "device_params",
    # Prefill/decode trace capture needs >30MiB on BH×4 (mesh_trace buffer limit vs trace_region_size).
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 35000000, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_e2e_vision_text_pipeline(
    weights,
    layers,
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    optimizations,
    mesh_device,
    reset_seeds,
    request,
    device_params,
):
    """Test end-to-end vision-text pipeline using proper Generator methods."""
    logger.info("Starting E2E vision-text pipeline test")

    # Use bfloat8_b like test_end2end.py for better memory efficiency
    dtype = ttnn.bfloat8_b

    # Setup vision-enabled model configuration
    model_args, instruct = setup_vision_model_args(weights, max_seq_len, batch_size, mesh_device, optimizations)

    if layers is not None:
        model_args.n_layers = layers

    # Setup vision prompts and tokenizer
    messages, tokenizer = setup_vision_prompts_and_tokenizer(model_args, instruct)

    # Process real vision inputs from images
    processed_inputs = process_real_vision_inputs(messages, model_args)

    # Load separate models following test_end2end.py pattern
    logger.info("Loading separate vision and text models like test_end2end.py...")
    vision_model, text_model = load_separate_models_like_test_end2end(
        model_args, mesh_device, dtype, paged_attention, page_params
    )

    # Setup page table for paged attention (exactly like test_end2end.py)
    page_table_tt = None
    paged_attention_config = None

    # Prepare page table for paged attention (exactly like test_end2end.py)
    page_table = None

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )

        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, -2) if batch_size > 1 else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    results = run_generation_exactly_like_test_end2end(
        vision_model, text_model, processed_inputs, model_args, page_table, paged_attention_config, max_gen_len=1024 * 4
    )

    # Validate results
    validation_passed = validate_e2e_outputs(results, expected_min_tokens=1)

    # Final validation
    if validation_passed and len(results) > 0:
        logger.info("E2E vision-text pipeline test PASSED!")
        logger.info(f"Successfully generated {len(results)} tokens")
    else:
        logger.error("E2E pipeline test failed")
        assert False, f"E2E pipeline failed - generated {len(results)} tokens, validation: {validation_passed}"
