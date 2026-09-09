# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 multimodal (vision + text) generation demo.

Mirrors ``text_demo.py`` prefill/decode + perf reporting, but routes images
through the TT vision tower and splices the projected soft tokens into the
text embeddings on device (``ttnn.scatter``, no host round-trip of vision
activations) inside ``Gemma4Model.ttnn_prefill_forward``.

Usage:
    pytest models/demos/gemma4/demo/vision_demo.py -v --timeout=1800

    MESH_DEVICE=P150x4 HF_MODEL=google/gemma-4-12B-it pytest \\
        models/demos/gemma4/demo/vision_demo.py -v -s --timeout=1800
"""

import gc
import json
import math
import os
import time
from io import BytesIO

import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.demos.gemma4.demo.text_demo import (
    _device_params,
    _mesh_shape_from_env,
    _right_size_page_max_num_blocks,
    _shorten_for_log,
    _snap_to_bucket,
)
from models.demos.gemma4.tt.common import create_tt_model
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import PagedAttentionConfig
from models.tt_transformers.tt.model_config import determine_device_name

SAMPLE_PROMPTS_DIR = "models/demos/gemma4/demo/sample_prompts"
_DEFAULT_IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"


def _load_conversation(prompt_file):
    with open(prompt_file) as f:
        data = json.load(f)
    assert len(data) >= 1, f"{prompt_file} has no conversations"
    return data[0]


def _load_image(uri):
    """Load an image from a URL or local path; fall back to a generated RGB image."""
    try:
        if str(uri).startswith("http"):
            import requests

            resp = requests.get(uri, timeout=60)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img.save("temp.jpeg")
            return img
        return Image.open(uri).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to load image from {uri} ({e}); using a generated fallback image")
        return Image.new("RGB", (1024, 768), color=(80, 120, 180))


def _collect_images(messages):
    """Collect image URIs/paths from chat-template content blocks."""
    images = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for el in content:
            if el.get("type") != "image":
                continue
            uri = el.get("image") or el.get("url") or el.get("path") or _DEFAULT_IMAGE_URL
            logger.info(uri)
            images.append(_load_image(uri))
    return images


def _process_prompt(processor, messages):
    """Apply the chat template + processor.

    Returns ``(input_ids [1, T], pixel_values, image_position_ids)``. Vision
    tensors are None for a text-only conversation.
    """
    images = _collect_images(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=images or None, return_tensors="pt")
    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    logger.info(f"pixel values: {pixel_values.shape}")
    image_position_ids = inputs.get("image_position_ids")
    if image_position_ids is None:
        image_position_ids = inputs.get("pixel_position_ids")
    return input_ids, pixel_values, image_position_ids


def run_vision_generation(
    mesh_device,
    model_path,
    prompt_file=None,
    max_new_tokens=32,
    num_layers=None,
    max_seq_len=4096,
    page_params=None,
    enable_decode_trace=True,
):
    """Run multimodal generation with Gemma4 (vision tower + text decoder).

    Perf reporting matches ``text_demo.run_generation``: compile vs measured
    prefill (TTFT), decode compile, tok/s/user, CI benchmark JSON.
    """
    from transformers import AutoProcessor, AutoTokenizer

    max_new_tokens = int(os.environ.get("GEMMA4_MAX_NEW_TOKENS", max_new_tokens))
    max_seq_len = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", max_seq_len))
    layers_env = os.environ.get("GEMMA4_NUM_LAYERS")
    if layers_env:
        num_layers = int(layers_env)

    is_ci_env = os.environ.get("CI") == "true"
    batch_size = 1

    if page_params is None:
        page_params = {"page_block_size": 64, "page_max_num_blocks": max_seq_len // 64}
    page_params = dict(page_params)
    page_params["page_max_num_blocks"] = _right_size_page_max_num_blocks(batch_size, max_seq_len, page_params)

    profiler = BenchmarkProfiler()
    profiler.start("run")

    profiler.start("loading_inputs")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    logger.info(f"Tokenizer/processor loaded from {model_path}")
    profiler.end("loading_inputs")

    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )
    page_table = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
        batch_size, paged_attention_config.max_num_blocks
    )

    logger.info(f"Creating model with {num_layers or 'all'} layers, max_seq_len={max_seq_len}...")
    t0 = time.time()
    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        model_path=model_path,
        create_kv_cache=True,
        paged_attention_config=paged_attention_config,
    )
    logger.info(f"Text model created in {time.time() - t0:.1f}s")

    t0 = time.time()
    model.init_vision_model(state_dict=state_dict)
    logger.info(f"Vision model attached in {time.time() - t0:.1f}s")

    is_mesh = hasattr(mesh_device, "shape")
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=replicate,
    )

    prompt_file = prompt_file or f"{SAMPLE_PROMPTS_DIR}/vision_demo.json"
    messages = _load_conversation(prompt_file)
    input_ids, pixel_values, image_position_ids = _process_prompt(processor, messages)
    input_ids = input_ids.squeeze(0)
    prompt_len = int(input_ids.shape[0])
    padded_len = _snap_to_bucket(prompt_len, max_seq_len)
    if prompt_len > padded_len:
        raise ValueError(
            f"Multimodal prompt ({prompt_len} tokens) exceeds prefill bucket {padded_len}; "
            f"raise max_seq_len (currently {max_seq_len})"
        )
    input_ids_padded = torch.nn.functional.pad(input_ids, (0, padded_len - prompt_len), value=0)
    n_image_tokens = int((input_ids == model.image_token_id).sum().item()) if model.image_token_id is not None else 0
    logger.info(
        f"Prompt tokens: {prompt_len} (padded to {padded_len}), "
        f"image_tokens={n_image_tokens}, "
        f"pixel_values={None if pixel_values is None else tuple(pixel_values.shape)}"
    )

    import torch.nn.functional as F

    embed_w = state_dict.get(
        "model.language_model.embed_tokens.weight",
        state_dict.get("model.embed_tokens.weight", torch.zeros(1)),
    )
    embeds_torch = (F.embedding(input_ids_padded.unsqueeze(0).long(), embed_w) * model.embed_scale).float()
    get_last_token = prompt_len - 1
    vision_kwargs = {"pixel_values": pixel_values, "image_position_ids": image_position_ids}

    def _build_prefill_embeds():
        tokens_tt = ttnn.from_torch(
            input_ids_padded.unsqueeze(0).to(torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        e = model.embed_tokens(tokens_tt)
        e = ttnn.reshape(e, (1, 1, padded_len, model_args.hidden_size))
        return ttnn.to_layout(e, ttnn.TILE_LAYOUT)

    logger.info("Prefill warmup (compiling, including vision tower)...")
    profiler.start("compile_prefill")
    warmup_embeds = _build_prefill_embeds()
    warmup_logits = model.ttnn_prefill_forward(
        warmup_embeds,
        page_table=page_table_tt,
        kv_cache=tt_kv_cache,
        get_last_token=get_last_token,
        input_ids_torch=input_ids_padded.unsqueeze(0),
        embeds_torch=embeds_torch,
        **vision_kwargs,
    )
    warmup_logits.deallocate(True)
    profiler.end("compile_prefill")
    logger.info(f"Prefill warmup done in {profiler.get_duration('compile_prefill'):.2f}s")

    logger.info("Prefilling (measured, vision + text)...")
    profiler.start("inference_prefill")
    embeds = _build_prefill_embeds()
    logits = model.ttnn_prefill_forward(
        embeds,
        page_table=page_table_tt,
        kv_cache=tt_kv_cache,
        get_last_token=get_last_token,
        input_ids_torch=input_ids_padded.unsqueeze(0),
        embeds_torch=embeds_torch,
        **vision_kwargs,
    )

    if is_mesh:
        logits_cpu = ttnn.to_torch(ttnn.get_device_tensors(logits)[0])
    else:
        logits_cpu = ttnn.to_torch(logits)
    logits.deallocate(True)
    pos_in_tile = (prompt_len - 1) % 32
    next_token = logits_cpu[0, 0, pos_in_tile, :].argmax().item()
    profiler.end("inference_prefill")
    logger.info(
        f"Prefill measured in {profiler.get_duration('inference_prefill'):.2f}s, "
        f"first token: {next_token} = '{tokenizer.decode([next_token])}'"
    )

    generated_tokens = [next_token]
    current_pos = prompt_len
    iteration = 0
    trace_id = None
    trace_output = None
    trace_device_inputs = None
    on_device_sampling = model.sampling is not None

    def _make_decode_inputs(tok, pos):
        pli_torch = model.compute_host_pli(tok)
        tokens_h = ttnn.from_torch(
            torch.tensor([[tok]], dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        pos_padded = torch.nn.functional.pad(
            torch.tensor([pos], dtype=torch.int32).reshape(1, 1), (0, 31), "constant", 0
        )
        inputs = {
            "tokens": tokens_h,
            "position": ttnn.from_torch(
                pos_padded,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
                mesh_mapper=replicate,
            ),
            "position_int32": ttnn.from_torch(
                torch.tensor([pos], dtype=torch.int32),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
                mesh_mapper=replicate,
            ),
        }
        if pli_torch is not None:
            inputs["pli"] = ttnn.from_torch(
                pli_torch.to(torch.bfloat16),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=replicate,
            )
        return inputs

    def _fwd(device_inputs):
        out = model.ttnn_decode_forward(
            x=device_inputs["tokens"],
            current_pos=device_inputs["position"],
            rot_mat_idxs=device_inputs["position_int32"],
            page_table=page_table_tt,
            kv_cache=tt_kv_cache,
            on_device_logits=on_device_sampling,
            pli_combined=device_inputs.get("pli"),
        )
        return out if isinstance(out, tuple) else (out, None)

    def _inputs_to_device(inputs):
        return {k: ttnn.to_device(v, device=mesh_device) for k, v in inputs.items() if v is not None}

    def _copy_inputs_to_trace(host_inputs):
        for k, v in host_inputs.items():
            if v is not None and k in trace_device_inputs:
                ttnn.copy_host_to_device_tensor(v, trace_device_inputs[k])

    def _extract_token(decode_output):
        if on_device_sampling:
            sampled = model.sampling.sample(decode_output, enable_trace=False)
            tt_tokens = sampled[0] if isinstance(sampled, tuple) else sampled
            sampled_cpu = ttnn.to_torch(ttnn.get_device_tensors(tt_tokens)[0]) if is_mesh else ttnn.to_torch(tt_tokens)
            return sampled_cpu.reshape(-1)[0].item()
        output_cpu = (
            ttnn.to_torch(ttnn.get_device_tensors(decode_output)[0]) if is_mesh else ttnn.to_torch(decode_output)
        )
        return output_cpu.squeeze().argmax().item()

    sample_mode = "device" if on_device_sampling else "host"
    logger.info(
        f"Decoding (trace={'ON' if enable_decode_trace else 'OFF'}, "
        f"embedding=device, sampling={sample_mode})..."
    )
    profiler.start("inference_decode")
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()

    try:
        for _step in range(max_new_tokens - 1):
            if iteration == 0:
                profiler.start("compile_decode")
            else:
                profiler.start(f"inference_decode_time_{iteration}")

            t_make_start = time.perf_counter()
            inputs_h = _make_decode_inputs(next_token, current_pos)
            t_make_end = time.perf_counter()

            if enable_decode_trace and trace_id is not None:
                _copy_inputs_to_trace(inputs_h)
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                decode_logits = trace_output
                t_enq_end = time.perf_counter()
            elif enable_decode_trace and iteration == 0:
                inputs_d = _inputs_to_device(inputs_h)
                decode_logits, _ = _fwd(inputs_d)
                next_token = _extract_token(decode_logits)
                generated_tokens.append(next_token)
                current_pos += 1
                profiler.end("compile_decode")
                decode_iteration_time = profiler.get_duration("compile_decode")
                logger.debug(
                    f"Iteration {iteration} (compile): {1000 * decode_iteration_time:.0f}ms @ "
                    f"{1 / decode_iteration_time:.1f} tok/s/user"
                )
                iteration += 1

                logger.info("Capturing decode trace...")
                inputs_h2 = _make_decode_inputs(next_token, current_pos)
                trace_device_inputs = _inputs_to_device(inputs_h2)
                trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                trace_output, _ = _fwd(trace_device_inputs)
                ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
                logger.info("Decode trace captured")

                profiler.start(f"inference_decode_time_{iteration}")
                _copy_inputs_to_trace(inputs_h2)
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                decode_logits = trace_output
                t_enq_end = time.perf_counter()
            else:
                inputs_d = _inputs_to_device(inputs_h)
                decode_logits, _ = _fwd(inputs_d)
                t_enq_end = time.perf_counter()

            next_token = _extract_token(decode_logits)
            t_sync_end = time.perf_counter()
            generated_tokens.append(next_token)
            current_pos += 1

            if iteration == 0:
                profiler.end("compile_decode")
                decode_iteration_time = profiler.get_duration("compile_decode")
            else:
                profiler.end(f"inference_decode_time_{iteration}")
                decode_iteration_time = profiler.get_duration(f"inference_decode_time_{iteration}")

            tokens_per_second_per_user = 1 / decode_iteration_time if decode_iteration_time > 0 else 0
            logger.debug(
                f"Iteration {iteration}: {1000 * decode_iteration_time:.0f}ms @ "
                f"{tokens_per_second_per_user:.1f} tok/s/user "
                f"| host_inputs={1000 * (t_make_end - t_make_start):.1f}ms "
                f"copy+enq={1000 * (t_enq_end - t_make_end):.1f}ms "
                f"exec+sync={1000 * (t_sync_end - t_enq_end):.1f}ms"
            )
            iteration += 1
            eos = tokenizer.eos_token_id
            if isinstance(eos, (list, tuple)):
                if next_token in eos:
                    break
            elif next_token == eos:
                break
    finally:
        if gc_was_enabled:
            gc.enable()
        if trace_id is not None:
            ttnn.release_trace(mesh_device, trace_id)

    profiler.end("inference_decode")

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    logger.info(f"\n==PROMPT\n{_shorten_for_log(json.dumps(messages))}\n==OUTPUT\n{generated_text.strip()}\n")

    num_tokens_generated_decode = iteration
    profiler.end("run")

    compile_prefill_time = profiler.get_duration("compile_prefill")
    compile_decode_time = profiler.get_duration("compile_decode")
    total_inference_prefill_time = profiler.get_duration("inference_prefill")

    total_inference_decode_time = 0
    for i in range(1, num_tokens_generated_decode):
        total_inference_decode_time += profiler.get_duration(f"inference_decode_time_{i}")

    avg_time_to_first_token = total_inference_prefill_time / batch_size
    avg_decode_iteration_time = (
        total_inference_decode_time / (num_tokens_generated_decode - 1) if num_tokens_generated_decode > 1 else 0
    )
    prefill_tok_s = prompt_len / total_inference_prefill_time * batch_size if total_inference_prefill_time > 0 else 0
    decode_tok_s_user = (
        (num_tokens_generated_decode - 1) / total_inference_decode_time
        if num_tokens_generated_decode > 1 and total_inference_decode_time > 0
        else 0
    )
    decode_tok_s = decode_tok_s_user * batch_size

    measurements = {
        "compile_prefill": compile_prefill_time,
        "compile_decode": compile_decode_time,
        "inference_prefill": total_inference_prefill_time,
        "inference_decode": total_inference_decode_time,
        "prefill_time_to_token": avg_time_to_first_token,
        "prefill_t/s": prefill_tok_s,
        "decode_t/s/u": decode_tok_s_user,
        "decode_t/s": decode_tok_s,
        "Total compile time": compile_prefill_time + compile_decode_time,
        "Full demo runtime": profiler.get_duration("run"),
    }

    tok_1_perf = profiler.get_duration("inference_decode_time_1") if 1 < num_tokens_generated_decode else 0
    tok_128_perf = profiler.get_duration("inference_decode_time_127") if 127 < num_tokens_generated_decode else 0

    logger.info("")
    logger.info("=== Performance metrics ===")
    if tok_1_perf > 0:
        logger.info(
            f"1st token decode time: {tok_1_perf * 1000:.2f}ms "
            f"[{round(1 / tok_1_perf, 2)} t/s/u, {round((1 / tok_1_perf) * batch_size, 2)} t/s]"
        )
    if tok_128_perf > 0:
        logger.info(
            f"128th token decode time: {tok_128_perf * 1000:.2f}ms "
            f"[{round(1 / tok_128_perf, 2)} t/s/u, {round((1 / tok_128_perf) * batch_size, 2)} t/s]"
        )
    logger.info("==")
    logger.info(f"Prefill compile time: {round(compile_prefill_time, 2)}s")
    logger.info(f"Decode compile time: {round(compile_decode_time, 2)}s")
    logger.info("")
    logger.info(f"Average Time to First Token (TTFT): {round(avg_time_to_first_token * 1000, 2)}ms")
    logger.info(
        f"Average speed: {round(avg_decode_iteration_time * 1000, 2)}ms @ "
        f"{round(decode_tok_s_user, 2)} tok/s/user ({round(decode_tok_s, 2)} tok/s throughput)"
    )
    logger.info(f"Generated {num_tokens_generated_decode} tokens")
    logger.info(f"Full demo runtime: {round(profiler.get_duration('run'), 2)}s")

    if is_ci_env:
        targets = {}
        bench_n_warmup_iter = {"inference_prefill": 0, "inference_decode": 1}
        benchmark_data = create_benchmark_data(profiler, measurements, bench_n_warmup_iter, targets)
        for i in range(1, num_tokens_generated_decode):
            benchmark_data.add_measurement(
                profiler,
                0,
                "inference_decode",
                f"time_to_token_{i}",
                profiler.get_duration(f"inference_decode_time_{i}") * 1000,
                step_warm_up_num_iterations=None,
                target=None,
            )
        num_iterations_for_avg = min(128, num_tokens_generated_decode)
        inference_decode_time_first_128 = sum(
            profiler.get_duration(f"inference_decode_time_{i}") for i in range(1, num_iterations_for_avg)
        )
        benchmark_data.add_measurement(
            profiler,
            0,
            "inference_decode",
            "avg_decode_time_first_128",
            inference_decode_time_first_128 * 1000 / max(1, num_iterations_for_avg - 1),
            step_warm_up_num_iterations=None,
            target=None,
        )
        benchmark_data.save_partial_run_json(
            profiler,
            run_type="demo_perf",
            ml_model_name="gemma4",
            ml_model_type="vlm",
            device_name=determine_device_name(mesh_device),
            num_layers=num_layers or model_args.num_hidden_layers,
            batch_size=batch_size,
            config_params={"multimodal": True},
            input_sequence_length=prompt_len,
            output_sequence_length=num_tokens_generated_decode,
        )

    return generated_text


@pytest.fixture
def model_path():
    return os.getenv("HF_MODEL") or os.getenv(
        "GEMMA4_MODEL_PATH", "/mnt/MLPerf/tt_dnn-models/google/gemma-4-26B-A4B-it"
    )


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [_mesh_shape_from_env()], indirect=True)
def test_demo_vision(mesh_device, model_path):
    """Full-model multimodal demo: vision tower → scatter fuse → prefill → decode."""
    max_new_tokens = int(os.environ.get("GEMMA4_MAX_NEW_TOKENS", 64))
    max_seq_len = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", 4096))
    page_block_size = 64
    page_params = {
        "page_block_size": page_block_size,
        "page_max_num_blocks": math.ceil(max_seq_len / page_block_size),
    }
    result = run_vision_generation(
        mesh_device=mesh_device,
        model_path=model_path,
        prompt_file=f"{SAMPLE_PROMPTS_DIR}/vision_demo.json",
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        page_params=page_params,
        enable_decode_trace=True,
    )
    assert result is not None and len(result) > 0
    logger.info(f"Vision demo output: {_shorten_for_log(result)}")
