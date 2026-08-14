# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS *simplified* profiling demo with Tracy signposts.

Purpose
-------
A stripped-down single-user (batch=1, 1x1 mesh) version of ``text_demo.py`` whose
only job is to make the PREFILL and DECODE stages easy to separate in a Tracy
capture. It emits ``tracy.signpost(...)`` markers around each stage so the
op-perf CSV can be split cleanly:

    PREFILL_START ... PREFILL_END      <- all prefill device ops fall in here
    DECODE_START   ... DECODE_END      <- all steady-state decode ops fall in here

Use ``models/tt_transformers/scripts/op_perf_results.py --signpost DECODE`` (or
any signpost-aware parser) to filter to a single stage.

Key differences vs text_demo.py (intentionally simpler)
-------------------------------------------------------
* Only the 1x1 single-user greedy path. No data-parallel / row-sharded /
  long-context / seqlen-sweep / on-device-sampling branches.
* Trace is DISABLED BY DEFAULT (``--gpt-oss-decode-trace-off`` still accepted
  for compatibility but the default here is already off) so the device profiler
  captures per-op timings without trace pre-baking hiding them.
* Same input prompts file and the same preprocessing as text_demo.py.
* Same CLI options: ``--gpt-oss-max-tokens`` (and ``--gpt-oss-decode-trace-off``,
  accepted but a no-op since trace is already off).

Run (under tracy)
-----------------
    python -m tracy -r -p --op-support-count 40000 -o <out> \\
        -m pytest -- \\
        "models/demos/gpt_oss/demo/text_demo_signpost.py::test_gpt_oss_signpost[1x1]" \\
        --gpt-oss-max-tokens 8
"""


import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gpt_oss.demo.text_demo import prepare_gpt_oss_generator_args
from models.demos.gpt_oss.tests.test_factory import TestFactory, parametrize_mesh_with_fabric
from models.tt_transformers.demo.simple_text_demo import load_inputs
from models.tt_transformers.tt.common import preprocess_inputs_prefill
from models.tt_transformers.tt.generator import Generator

try:
    # tracy.signpost logs a marker into the profiler timeline. Outside a tracy
    # capture it degrades to a harmless log line, so it is always safe to call.
    from tracy import signpost
except Exception:  # pragma: no cover - tracy always present in this repo

    def signpost(header, message=None):
        logger.info(f"[signpost] {header}")


# Single input file, same as the canonical prefill_128 case in text_demo.py.
INPUT_PROMPTS_FILE = "models/demos/gpt_oss/demo/sample_prompts/input_data_questions_prefill_128.json"
MAX_SEQ_LEN = 4 * 1024
PAGE_PARAMS = {"page_block_size": 64, "page_max_num_blocks_per_dp": 4 * 1024 // 64}


@pytest.mark.timeout(3600)
@parametrize_mesh_with_fabric([(1, 1)])
def test_gpt_oss_signpost(
    mesh_device,
    device_params,
    request,
    state_dict,
):
    """Simplified single-user GPT-OSS decode with PREFILL/DECODE tracy signposts."""
    mesh_shape = tuple(mesh_device.shape)
    assert mesh_shape == (1, 1), f"This simplified demo only supports a 1x1 mesh, got {mesh_shape}"

    # --- CLI overrides (same options as text_demo.py) -----------------------
    # Trace is OFF BY DEFAULT here, because the whole point is per-op profiling and trace
    # pre-bakes the dispatch the profiler needs to attribute.
    #
    # But a wall clock taken without trace is not this model's wall clock: untraced, every
    # operation is dispatched from the host individually, so the host gap it reports is partly
    # an artifact of the profiling configuration rather than a cost the served model pays.
    # `--gpt-oss-decode-trace-on` exists for that run: same program, same token selection,
    # timed the way the model is actually served.
    _ = request.config.getoption("--gpt-oss-decode-trace-off")
    enable_decode_trace = bool(request.config.getoption("--gpt-oss-decode-trace-on"))
    enable_prefill_trace = False

    max_generated_tokens = 8  # small default; profiling only needs a few decode steps
    _max_tokens_override = request.config.getoption("--gpt-oss-max-tokens")
    if _max_tokens_override and _max_tokens_override > 0:
        max_generated_tokens = _max_tokens_override

    # --- Model / generator setup (mirrors text_demo.py) ---------------------
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    mesh_config = setup["mesh_config"]

    num_devices = mesh_device.get_num_devices()
    global_batch_size = 1

    (
        model_args,
        model,
        page_table,
        tt_kv_cache,
        tokenizer,
        processor,
        paged_attention_config,
    ) = prepare_gpt_oss_generator_args(
        num_devices=num_devices,
        data_parallel=1,
        mesh_device=mesh_device,
        global_batch_size=global_batch_size,
        optimizations=None,
        max_seq_len=MAX_SEQ_LEN,
        page_params=PAGE_PARAMS,
        paged_attention=True,
        mesh_config=mesh_config,
        state_dict=state_dict,
        users_row_sharded=False,
        long_context_mode=False,
    )

    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    # On-device sampling is unavailable at tp=1 (per-device padded vocab > 64K),
    # so this simplified demo always uses the host-launched / on-device-argmax
    # greedy path (same as text_demo.py's fallback).
    on_device_sampling_supported = all(getattr(m, "sampling", None) is not None for m in model)
    assert (
        not on_device_sampling_supported
    ), "This simplified demo targets the tp=1 greedy path; on-device sampling should be unavailable here."

    # --- Inputs (same prompts + preprocessing as text_demo.py) --------------
    real_prompts, _ = load_inputs(INPUT_PROMPTS_FILE, global_batch_size, instruct=False)
    (
        input_tokens_prefill_pt,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    ) = preprocess_inputs_prefill(
        real_prompts,
        tokenizer,
        model_args,
        instruct=False,
        max_generated_tokens=max_generated_tokens,
        max_prefill_len=MAX_SEQ_LEN,
    )
    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)
    logger.info(f"Input prompt: {real_prompts[0]}")
    logger.info(f"Encoded length: {prefill_lens[0]} tokens")

    # ========================= PREFILL STAGE ================================
    logger.info("Starting prefill...")
    signpost("PREFILL_START")
    prefill_result = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        enable_trace=enable_prefill_trace,
        warmup_prefill=False,
    )
    signpost("PREFILL_END")

    if isinstance(prefill_result, tuple):
        prefilled_token = prefill_result[0].squeeze(-1)
    else:
        prefilled_token = torch.argmax(prefill_result, dim=-1)
    logger.info(f"First generated token: '{tokenizer.decode(prefilled_token[0])}'")

    all_outputs = [encoded_prompts[0][: prefill_lens[0]]]
    all_outputs[0].append(int(prefilled_token[0].item()))

    current_pos = torch.tensor([decoding_pos[0]])
    out_tok = prefilled_token

    # ========================= DECODE STAGE =================================
    logger.info(f"Starting decode loop ({max_generated_tokens} tokens)...")
    # Drain the device before marking the decode region. ttnn is async: python
    # enqueues prefill work and returns, so without this barrier late prefill ops
    # execute AFTER the DECODE_START marker and tracy attributes them to decode.
    # That inflated the decode budget by a [128,32] FILL cluster (0.612 ms/tok)
    # which cannot be decode work at all -- decode has 1 row, never 128.
    ttnn.synchronize_device(mesh_device)
    signpost("DECODE_START")
    # How long each step takes end to end, host included. A device profile sums kernel durations,
    # so it cannot see a tensor read back to the host between two kernels -- and choosing a token
    # from a 16.78 MB logits tensor is exactly that. Reported in the same format text_demo.py uses,
    # because the tool reads one line for the whole model family rather than one per harness.
    step_times = []
    for iteration in range(max_generated_tokens):
        step_started = time.time()
        # Host-launched greedy: keep logits on device, argmax on device, read
        # back only the token id (identical result to torch.argmax). Same path
        # as text_demo.py's tp=1 fallback, kept minimal here.
        tt_logits = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=enable_decode_trace,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=None,
            read_from_device=False,
        )
        tl = tt_logits
        while isinstance(tl, (list, tuple)):
            tl = tl[0]
        try:
            if tl.dtype not in (ttnn.bfloat16, ttnn.float32):
                tl = ttnn.typecast(tl, ttnn.bfloat16)
            nb = out_tok.shape[0]
            V = tl.shape[-1]
            tl_rows = ttnn.slice(tl, (0, 0, 0, 0), (1, 1, nb, V))
            tl_rm = ttnn.to_layout(tl_rows, ttnn.ROW_MAJOR_LAYOUT)
            tt_am = ttnn.argmax(tl_rm, dim=-1)
            out_tok = ttnn.to_torch(tt_am).reshape(-1)[:nb].to(torch.int32).view(-1)
            ttnn.deallocate(tt_am)
        except Exception:
            logits = generator.process_decode_output_host(generator.read_decode_output(tt_logits), is_tokens=False)[0]
            out_tok = torch.argmax(logits, dim=-1).view(-1)

        current_pos += 1
        # Reading the token back is what makes the step complete on the host, so the clock stops
        # after it rather than after the enqueue: ttnn is async and timing the enqueue would
        # measure python.
        all_outputs[0].append(int(out_tok[0].item()))
        step_times.append(time.time() - step_started)
    signpost("DECODE_END")

    # The first step is excluded: it carries cache warming and, when trace is on, the capture. It
    # is the one step that is not the steady state, and with a short run it dominates the mean.
    settled = step_times[1:] or step_times
    if settled:
        average_s = sum(settled) / len(settled)
        logger.info(
            f"Average decode speed: {round(average_s * 1000, 2)}ms @ {round(1 / average_s, 2)} tok/s/user "
            f"({round(1 / average_s, 2)} tok/s throughput) over {len(settled)} settled steps"
        )

    logger.info(f"Generated text: {tokenizer.decode(all_outputs[0])}")
    logger.info("GPT-OSS signpost demo completed successfully!")
