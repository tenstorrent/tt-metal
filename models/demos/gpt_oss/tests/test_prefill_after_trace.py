# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Eager prefill of prompt lengths first seen AFTER the traces exist must reproduce the trace-free result.

A server warms a few padded prefill lengths, captures the prefill (128 tokens) and decode traces, then serves prompts
of arbitrary length. Any program that is compiled after the traces were captured lives next to the live traces and can
be overwritten by their replays (tenstorrent/tt-metal#55588), which shows up as intermittent garbage or a hang for
prompts whose shapes the warm-up never saw. This test pins the model-side invariant: with the padded lengths warmed,
a prompt of a NEW length (in particular a new last-token tile position within the padded length) must produce the
same greedy tokens after the traces were captured as it did eagerly before, run after run.

The corruption itself is probabilistic (it needs a replay to land on the late program's buffers), so run this with
TT_METAL_TRACE_ALLOC_TRACKING=1 (and TT_METAL_TRACE_ALLOC_TRACEBACKS=1 for Python tracebacks) to turn the hazard into a
hard failure: ttnn.execute_trace then raises at the first replay while any buffer allocated with a trace live is still
alive, naming the program-cache entries compiled late.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.common.sampling import SamplingParams
from models.demos.gpt_oss.demo.text_demo import prepare_gpt_oss_generator_args
from models.demos.gpt_oss.tests.test_factory import TestFactory, parametrize_mesh_with_fabric
from models.demos.gpt_oss.tests.test_multi_user_regression import (
    BLOCK_SIZE,
    PROMPT_FILES,
    _degenerate,
    _prefill,
    _prompts_for,
)
from models.tt_transformers.tt.common import get_padded_prefill_len, preprocess_inputs_prefill
from models.tt_transformers.tt.generator import Generator

# Prompt lengths in tokens whose 32-token tile (length mod padded length) the warm-up of the padded length does not
# compile: 300 and 700 pad to 1024 (warm-up sees the tile of 1024 tokens exactly), 1500 pads to 2048.
PROMPT_TOKENS = [int(t) for t in os.getenv("GPT_OSS_AFTER_TRACE_TOKENS", "300,700,1500").split(",")]
REPEATS = int(os.getenv("GPT_OSS_AFTER_TRACE_REPEATS", "3"))
GEN_TOKENS = 24
MAX_SEQ_LEN = 8192
REFERENCE_DIR = Path("generated/gpt_oss_prefill_after_trace")


def _prompt_with_tokens(tokenizer, num_tokens):
    """Natural text cut to (about) num_tokens tokens, from the harness's longest prompt."""
    text = _prompts_for(max(PROMPT_FILES), 1)[0]
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids) >= num_tokens, f"prompt source has only {len(ids)} tokens"
    return tokenizer.decode(ids[:num_tokens])


def _padded_len(tokenizer, model_args, prompt):
    """The prefill length the generator pads this prompt to (same preprocessing as _generate)."""
    _, _, decoding_pos, _ = preprocess_inputs_prefill(
        [prompt], tokenizer, model_args, instruct=False, max_generated_tokens=GEN_TOKENS, max_prefill_len=MAX_SEQ_LEN
    )
    return get_padded_prefill_len(int(decoding_pos[0]))


def _generate(generator, models, tt_kv_cache, page_table, tokenizer, model_args, prompt, sampling, enable_trace):
    input_tokens, _, decoding_pos, _ = preprocess_inputs_prefill(
        [prompt], tokenizer, model_args, instruct=False, max_generated_tokens=GEN_TOKENS, max_prefill_len=MAX_SEQ_LEN
    )
    input_tokens = torch.stack(input_tokens).view(1, -1)
    logits, _ = _prefill(
        generator, models, tt_kv_cache, page_table, input_tokens, decoding_pos, enable_trace=enable_trace
    )
    out_tok = torch.argmax(logits, dim=-1)
    tokens = [int(out_tok[0])]
    current_pos = torch.tensor(decoding_pos)
    for _ in range(GEN_TOKENS - 1):
        out_tok, _ = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=enable_trace,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=sampling,
        )
        current_pos += 1
        tokens.append(int(out_tok[0]))
    return int(decoding_pos[0]), tokens


def _reference_file(model_args):
    return REFERENCE_DIR / f"{model_args.base_model_name}_reference.json"


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("phase", ["reference", "after_trace"])
@parametrize_mesh_with_fabric([(1, 8)])
def test_prefill_after_trace_capture(mesh_device, device_params, state_dict, phase):
    """Two device sessions on purpose: the reference phase compiles the very programs the after-trace phase must
    find missing, so it runs first in its own session (fresh program cache) and hands the tokens over via a file."""
    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape[0] != 1 or mesh_shape[1] < 8:
        pytest.skip(f"targets 1x8 meshes, got {mesh_shape}")
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    torch.manual_seed(1234)
    model_args, models, page_table, tt_kv_cache, tokenizer, _processor, _cfg = prepare_gpt_oss_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=1,
        mesh_device=mesh_device,
        global_batch_size=1,
        optimizations=None,
        max_seq_len=MAX_SEQ_LEN,
        page_params={"page_block_size": BLOCK_SIZE, "page_max_num_blocks_per_dp": MAX_SEQ_LEN // BLOCK_SIZE},
        paged_attention=True,
        mesh_config=setup["mesh_config"],
        state_dict=state_dict,
        users_row_sharded=False,
    )
    generator = Generator(models, model_args, mesh_device, processor=None, tokenizer=tokenizer)
    sampling = SamplingParams(temperature=[0.0], top_k=[1], top_p=[1.0])
    run = lambda prompt, trace: _generate(
        generator, models, tt_kv_cache, page_table, tokenizer, model_args, prompt, sampling, trace
    )
    prompts = {n: _prompt_with_tokens(tokenizer, n) for n in PROMPT_TOKENS}
    ref_file = _reference_file(model_args[0])

    if phase == "reference":
        # Trace-free reference for every prompt (eager prefill, eager decode), handed to the next session.
        reference = {n: run(p, False)[1] for n, p in prompts.items()}
        for n, toks in reference.items():
            logger.info(f"{n} tokens: reference {tokenizer.decode(toks)!r}")
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        ref_file.write_text(json.dumps({str(n): toks for n, toks in reference.items()}))
        return

    assert ref_file.is_file(), f"run the 'reference' phase first ({ref_file} missing)"
    reference = {int(n): toks for n, toks in json.loads(ref_file.read_text()).items()}
    # 1. Warm the padded lengths the way a server does: one eager prefill per padded length, with a prompt of 3/4 of
    #    the bucket (prompt + generated tokens must stay inside it; a full-bucket prompt would spill into the next).
    padded = sorted({_padded_len(tokenizer, model_args, p) for p in prompts.values()})
    for length in padded:
        warm_prompt = _prompt_with_tokens(tokenizer, length * 3 // 4)
        assert _padded_len(tokenizer, model_args, warm_prompt) == length
        run(warm_prompt, False)
        logger.info(f"warmed padded length {length}")
    # 2. Capture the traces: a 128-token prompt takes the traced prefill and the first traced decode steps.
    run(_prompts_for(128, 1)[0], True)
    # 3. The prompts, whose last-token tiles no warm-up compiled, now run with the traces live, several times each.
    failures = []
    for rep in range(REPEATS):
        for n, p in prompts.items():
            pos, toks = run(p, True)
            same = toks == reference[n]
            logger.info(
                f"rep {rep} {n} tokens (prefill len {pos}): {'same' if same else 'DIFFERENT'} {tokenizer.decode(toks)!r}"
            )
            if not same or _degenerate(toks):
                failures.append((rep, n, tokenizer.decode(toks)))
    assert not failures, "outputs changed after trace capture (late-compiled programs?):\n" + "\n".join(
        f"rep {r} prompt {n} tokens -> {t!r}" for r, n, t in failures
    )
