# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Gemma3 text *profiling* demo with Tracy signposts.

A stripped-down single-user (batch 1, 1x1 mesh) version of ``text_demo.py``
whose only job is to make PREFILL and DECODE easy to cut out of a Tracy
capture:

    PREFILL_START ... PREFILL_END      <- the prefill device ops
    DECODE_START  ... DECODE_END       <- the steady-state decode ops

It builds the model with the same ``prepare_generator_args`` and the same
``performance`` decoder precision as ``text_demo.py``'s performance row, warms
up prefill and decode (so no compile falls inside a window), and runs
``--max_generated_tokens`` decode steps (default and maximum 2) with trace off, so the
device profiler sees every op.

Run (under tracy)
-----------------
    python -m tracy -r -p --op-support-count 40000 -o <out> -m pytest -- \\
        "models/demos/multimodal/gemma3/demo/text_demo_signpost.py::test_gemma3_signpost[blackhole-1x1-p150]" -x
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling import SamplingParams
from models.demos.multimodal.gemma3.demo.text_demo import load_inputs, prepare_generator_args
from models.demos.multimodal.gemma3.tt.gemma_e2e_model import TtGemmaModel
from models.demos.multimodal.gemma3.tt.gemma_multimodal_generator import GemmaMultimodalGenerator as Generator
from models.tt_transformers.tt.common import preprocess_inputs_prefill
from models.tt_transformers.tt.model_config import DecodersPrecision

try:
    # tracy.signpost logs a marker into the profiler timeline; outside a capture
    # it is a harmless log line, so it is always safe to call.
    from tracy import signpost
except Exception:  # pragma: no cover - tracy is part of this repo

    def signpost(header, message=None):
        logger.info(f"[signpost] {header}")


INPUT_PROMPTS_FILE = "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json"
MAX_SEQ_LEN = 1024
PAGE_PARAMS = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}
DEFAULT_DECODE_STEPS = 2
MAX_DECODE_STEPS = 2

DEVICE_PARAMS = {"fabric_config": True, "num_command_queues": 2, "l1_small_size": 24576}


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params, mesh_device",
    [pytest.param(DEVICE_PARAMS, (1, 1), id="1x1-p150")],
    indirect=True,
)
def test_gemma3_signpost(mesh_device, device_params, request, reset_seeds):
    """Single-user greedy Gemma3 text decode with PREFILL/DECODE tracy signposts."""
    assert tuple(mesh_device.shape) == (1, 1), f"this demo runs on a 1x1 mesh, got {tuple(mesh_device.shape)}"

    # Never more than MAX_DECODE_STEPS: a larger request is capped, not honoured.
    decode_steps = min(
        MAX_DECODE_STEPS,
        request.config.getoption("--max_generated_tokens")
        or int(os.getenv("TT_GEMMA3_SIGNPOST_STEPS", DEFAULT_DECODE_STEPS)),
    )
    global_batch_size = 1
    instruct = True

    # The performance decoder precision, as text_demo.py's "performance" row.
    optimizations = lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name)

    # One model per submesh; on a 1x1 mesh that is one entry in each list, and
    # the generator takes the per-model kv_cache list as it is.
    model_args, model, page_table, tt_kv_cache, tokenizer = prepare_generator_args(
        num_devices=1,
        data_parallel=1,
        mesh_device=mesh_device,
        instruct=instruct,
        global_batch_size=global_batch_size,
        optimizations=optimizations,
        max_seq_len=MAX_SEQ_LEN,
        page_params=PAGE_PARAMS,
        paged_attention=True,
        enable_program_trace=False,
    )
    generator = Generator(model, model_args, mesh_device, tokenizer=tokenizer)

    # Host sampling (argmax of the logits), as the token-matching gate does:
    # with trace off, reading device-sampled tokens back fails in decode
    # warmup (TT_FATAL tensor_impl.cpp:297, physical_data.size()), and the
    # host-logits path is the one text_demo.py runs untraced.
    can_sample_on_device = False
    device_sampling_params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0) if can_sample_on_device else None
    logger.info(f"Gemma3 signpost decode: device sampling={can_sample_on_device}, steps={decode_steps}")

    # Compile both paths before any window opens.
    num_blocks = PAGE_PARAMS["page_max_num_blocks_per_dp"] // global_batch_size
    generator.warmup_model_prefill(
        kv_cache=tt_kv_cache, enable_trace=False, can_sample_on_device=can_sample_on_device, greedy_only=True
    )
    generator.warmup_model_decode(
        kv_cache=tt_kv_cache,
        enable_trace=False,
        max_batch_size=global_batch_size,
        num_blocks=num_blocks,
        can_sample_on_device=can_sample_on_device,
        greedy_only=True,
    )

    prompts = load_inputs(INPUT_PROMPTS_FILE, global_batch_size, instruct)
    input_tokens_prefill_pt, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        prompts, tokenizer, model_args, instruct, decode_steps, max_prefill_len=MAX_SEQ_LEN
    )
    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)
    logger.info(f"Encoded prompt length: {prefill_lens[0]} tokens")

    # ========================= PREFILL =====================================
    ttnn.synchronize_device(mesh_device)
    signpost("PREFILL_START")
    prefill_out = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        warmup_prefill=False,
        sampling_params=device_sampling_params,
    )
    ttnn.synchronize_device(mesh_device)
    signpost("PREFILL_END")
    if device_sampling_params is not None:
        prefilled_token = prefill_out[0].long()
    else:
        prefilled_token = torch.argmax(prefill_out, dim=-1)
    out_tok = prefilled_token.view(global_batch_size, -1)[:, -1:].reshape(global_batch_size, 1)
    outputs = list(encoded_prompts[0][: prefill_lens[0]]) + [int(out_tok[0, 0].item())]
    current_pos = torch.tensor([decoding_pos[0]])

    # ========================= DECODE ======================================
    # ttnn is asynchronous: drain the device so late prefill ops do not land
    # after DECODE_START and get counted as decode.
    ttnn.synchronize_device(mesh_device)
    signpost("DECODE_START")
    for _step in range(decode_steps):
        logits, _ = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=False,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=device_sampling_params,
        )
        if device_sampling_params is not None:
            out_tok = logits.reshape(global_batch_size, -1)[:, -1:]
        else:
            _, out_tok = TtGemmaModel.sample_host(logits, temperature=0, top_p=0.08, on_host=True)
            out_tok = out_tok.reshape(global_batch_size, -1)[:, :1]
        current_pos += 1
        outputs.append(int(out_tok[0, 0].item()))
    ttnn.synchronize_device(mesh_device)
    signpost("DECODE_END")

    logger.info(f"Generated text: {tokenizer.decode(outputs[prefill_lens[0]:])}")
    logger.info("Gemma3 signpost demo completed")
