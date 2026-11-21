# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import models.ttt_compat.tt.common as tt_common
import ttnn
from models.common.device_utils import is_blackhole
from models.demos.llama31_8b_p100_v3.demo.demo import BLOCKS, MAX_SEQ_LEN, build, encode, performance
from models.demos.llama31_8b_p100_v3.tt.spec_decode import sampling_params
from models.ttt_compat.tt.common import PagedAttentionConfig
from models.ttt_compat.tt.generator import Generator

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="Blackhole P100 only")


def main_tokens(mesh_device, encoded, new_tokens):
    args, model, kv, sd = tt_common.create_tt_model(
        mesh_device,
        instruct=True,
        max_batch_size=1,
        optimizations=performance,
        max_seq_len=MAX_SEQ_LEN,
        paged_attention_config=PagedAttentionConfig(block_size=32, max_num_blocks=BLOCKS),
        dtype=ttnn.bfloat8_b,
    )
    del sd
    pt = torch.arange(BLOCKS, dtype=torch.int32).reshape(1, -1)
    gen = Generator([model], [args], mesh_device, tokenizer=args.tokenizer)
    sp = sampling_params()
    result = []
    for toks, length in encoded:
        out = gen.prefill_forward_text(
            toks,
            page_table=pt,
            kv_cache=[kv],
            prompt_lens=[length],
            sampling_params=sp,
            warmup_prefill=True,
            enable_trace=True,
        )
        t = out[0] if isinstance(out, tuple) else out
        tok = int(t.reshape(-1)[0])
        seq, pos = [tok], length
        for i in range(new_tokens):
            out = gen.decode_forward(
                torch.tensor([[tok]], dtype=torch.int64),
                torch.tensor([pos], dtype=torch.int64),
                page_table=pt,
                kv_cache=[kv],
                enable_trace=True,
                sampling_params=sp,
                reload_inputs=True,
                reload_page_table=False,
                reload_sampling_params=True,
                reset_sampling_state=i == 0,
            )
            t = out[0] if isinstance(out, tuple) else out
            tok = int(t.reshape(-1)[0])
            seq.append(tok)
            pos += 1
        result.append(seq)
    for layer in kv:
        for t in layer:
            ttnn.deallocate(t)
    return args, result


@pytest.mark.parametrize("num_prompts, new_tokens", [(4, 128)])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "num_command_queues": 1, "trace_region_size": 150000000}],
    indirect=True,
)
def test_output_matches_main(mesh_device, num_prompts, new_tokens):
    args, generator = build(mesh_device)
    encoded = encode(args, num_prompts, new_tokens)
    ours = [generator.generate(t, length, new_tokens, set())[0] for t, length in encoded]
    _, ref = main_tokens(mesh_device, encoded, new_tokens)
    assert ours == ref


@pytest.mark.parametrize("new_tokens", [1024])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "num_command_queues": 1, "trace_region_size": 150000000}],
    indirect=True,
)
def test_long_context_output_matches_main(mesh_device, new_tokens):
    args, generator = build(mesh_device)
    encoded = [
        encode(args, 1, new_tokens, f"models/ttt_compat/demo/sample_prompts/input_data_long_{length}.json")[0]
        for length in ("1k", "2k", "4k")
    ]
    ours = [generator.generate(t, length, new_tokens, set())[0] for t, length in encoded]
    _, ref = main_tokens(mesh_device, encoded, new_tokens)
    assert ours == ref
