# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import models.ttt_compat.demo.simple_text_demo as tt_demo
import models.ttt_compat.tt.common as tt_common
import models.ttt_compat.tt.model as tt_model
import ttnn
from models.demos.llama31_8b_p100_spec.tt.model import Transformer
from models.demos.llama31_8b_p100_spec.tt.spec_decode import SpecDecodeGenerator, draft_config
from models.ttt_compat.tt.common import PagedAttentionConfig, preprocess_inputs_prefill
from models.ttt_compat.tt.generator import Generator
from models.ttt_compat.tt.model_config import DecodersPrecision

PROMPTS = "models/ttt_compat/demo/sample_prompts/input_data_questions_prefill_128.json"
MAX_SEQ_LEN = 8192
BLOCKS = 1024


def create_tt_model(*args, **kwargs):
    saved = tt_model.Transformer
    tt_model.Transformer = Transformer
    try:
        return tt_common.create_tt_model(*args, **kwargs)
    finally:
        tt_model.Transformer = saved


def performance(model_args):
    return DecodersPrecision.performance(model_args.n_layers, model_args.model_name)


def build(mesh_device):
    from models.demos.llama31_8b_p100_spec.tt import eagle_draft, eagle_head

    head = eagle_head.head_dir()
    cfg = draft_config("eagle")
    model_args, model, kv_cache, state_dict = create_tt_model(
        mesh_device,
        instruct=True,
        max_batch_size=cfg["rows"],
        optimizations=performance,
        max_seq_len=MAX_SEQ_LEN,
        paged_attention_config=PagedAttentionConfig(block_size=32, max_num_blocks=BLOCKS),
        dtype=ttnn.bfloat8_b,
    )
    del state_dict
    page_table = torch.arange(BLOCKS, dtype=torch.int32).reshape(1, -1).repeat(cfg["rows"], 1)
    generator = Generator([model], [model_args], mesh_device, tokenizer=model_args.tokenizer)
    cache = os.path.join(os.environ.get("TT_CACHE_PATH", "/tmp"), "eagle_head")
    draft = eagle_draft.EagleDraft(
        mesh_device,
        model,
        model_args,
        head,
        model_args.CKPT_DIR,
        cache,
        performance,
        rows=cfg["rows"],
        max_seq_len=MAX_SEQ_LEN,
    )
    return model_args, SpecDecodeGenerator(generator, model_args, mesh_device, kv_cache, page_table, draft, cfg["k"])


def encode(model_args, num_prompts, max_generated_tokens, prompts_file=PROMPTS):
    _, prompts = tt_demo.load_inputs(prompts_file, num_prompts, True)
    encoded = []
    for p in prompts[:num_prompts]:
        toks, _, _, lens = preprocess_inputs_prefill(
            [p], model_args.tokenizer, [model_args], True, max_generated_tokens, max_prefill_len=MAX_SEQ_LEN
        )
        encoded.append((torch.stack(toks).view(1, -1), int(lens[0])))
    return encoded


@pytest.mark.parametrize(
    "num_prompts, max_generated_tokens, stop_at_eos",
    [(1, 4096, False)],
    ids=["performance-ci-1"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "num_command_queues": 1, "trace_region_size": 150000000}],
    indirect=True,
)
def test_demo_text(mesh_device, num_prompts, max_generated_tokens, stop_at_eos):
    model_args, generator = build(mesh_device)
    stops = set(model_args.tokenizer.stop_tokens) if stop_at_eos else set()
    encoded = encode(model_args, num_prompts, max_generated_tokens)
    generator.generate(encoded[0][0], encoded[0][1], 8, stops)
    n = s = p = 0.0
    for toks, length in encoded:
        out, stats = generator.generate(toks, length, max_generated_tokens, stops)
        n += len(out) - 1
        s += stats["decode_s"]
        p += stats["prefill_s"]
    logger.info(f"Prefill time to first token: {1000 * p / len(encoded):.2f} ms")
    logger.info(f"Decode: {n / s:.2f} tok/s/user")
