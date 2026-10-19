# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A prefill of a new length after the decode trace capture must not be corrupted by later decode replays.

A device tensor allocated after a trace was captured can sit in that trace's scratch region and is overwritten by every
replay of it. Model.prepare_inputs_prefill used to cache the RoPE slice of each prefill length on its first use, so the
slice of a length first seen after the decode trace capture was overwritten by the decode replays and every later
prefill of that length returned garbage (served-style request mix: every 128-token prompt after a 4096-token one).

A 2-layer model with random weights prefills a long prompt, captures a decode trace, prefills a short prompt of a new
padded length, replays the decode trace, prefills the short prompt again and requires bit-identical logits. It also
requires that an eager prefill keeps no device tensor alive in the model.

The short prefill compiles new programs, and those can own long-lived device buffers (e.g. global semaphores) placed in
the trace's scratch region. The generator therefore re-captures the decode trace before the next replay when programs
were compiled since its capture (Model._tt_recapture_decode_trace_on_new_programs); the test does the same, so it checks
the serving protocol rather than the memory layout of the program cache.

Run on a QuietBox 2 (Blackhole 1x4):
    HF_MODEL=models/demos/gpt_oss/configs/gpt-oss-20b pytest models/demos/gpt_oss/tests/unit/test_prefill_after_decode_trace.py
"""

import os

import pytest
import torch

import ttnn
from models.tt_transformers.tt.common import PagedAttentionConfig, get_padded_prefill_len
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format

from ..test_factory import TestFactory, parametrize_mesh_with_fabric

NUM_LAYERS = 2
PAGE_BLOCK = 64
PAGE_BLOCKS = 64  # 4096-token KV cache
LONG_PROMPT = 2000  # padded to 2048
SHORT_PROMPT = 100  # padded to 128: a prefill length first seen after the decode trace capture
DECODE_POS = 3000  # the decode step writes the KV cache past both prompts
REPLAYS = 16


def _prefill(model, ids, page_table):
    """Eager prefill of ids [1, n] (padded like the generator) -> the last position's logits on the host."""
    n = ids.shape[-1]
    padded = torch.zeros(1, get_padded_prefill_len(n), dtype=torch.long)
    padded[:, :n] = ids
    x, rot_global, rot_local, tt_page_table, _, _ = model.prepare_inputs_prefill(padded, page_table=page_table)
    last = n - 1
    tt_out = model.ttnn_prefill_forward(
        x,
        rot_mats_global=rot_global,
        rot_mats_local=rot_local,
        page_table=tt_page_table,
        get_last_token=(last // 32) * 32,
    )
    return model.process_output_prefill(tt_out, last % 32).float()


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Needs a QuietBox 2 for about a minute; the CI gpt-oss-20b unit job has a 5-minute budget.",
)
@parametrize_mesh_with_fabric([(1, 4)])
def test_prefill_after_decode_trace_capture(mesh_device, device_params, reset_seeds):
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.ccl import CCLManager
    from models.demos.gpt_oss.tt.model import Model
    from models.demos.gpt_oss.utils.general_utils import get_default_num_links

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]
    config.num_hidden_layers = NUM_LAYERS
    state_dict = convert_hf_qkv_to_meta_format(GptOssForCausalLM(config).eval().state_dict(), config.head_dim)

    mesh_shape = tuple(mesh_device.shape)
    page_table = torch.arange(PAGE_BLOCKS, dtype=torch.int32).reshape(1, PAGE_BLOCKS)
    model = Model(
        mesh_device=mesh_device,
        hf_config=config,
        state_dict=state_dict,
        ccl_manager=CCLManager(mesh_device, num_links=get_default_num_links(mesh_device)),
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        paged_attention_config=PagedAttentionConfig(block_size=PAGE_BLOCK, max_num_blocks=PAGE_BLOCKS),
        mesh_config=MeshConfig(mesh_shape, decode=ModeConfig(tp=mesh_shape[1], ep=mesh_shape[0])),
        create_kv_cache=True,
        max_local_batch_size=1,
    )

    long_ids = torch.randint(0, config.vocab_size, (1, LONG_PROMPT))
    short_ids = torch.randint(0, config.vocab_size, (1, SHORT_PROMPT))
    _prefill(model, long_ids, page_table)

    # Decode trace over persistent inputs (compile once eagerly, then capture), as the generator does.
    decode_inputs = model.prepare_inputs_decode(torch.tensor([7]), torch.tensor([DECODE_POS]), page_table=page_table)
    tokens, current_pos, rot_idxs, tt_page_table = decode_inputs[:4]

    def decode():
        return model.ttnn_decode_forward(
            tokens=tokens, current_pos=current_pos, rot_mat_idxs=rot_idxs, page_table=tt_page_table
        )

    def capture():
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        decode()
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        return trace

    decode()
    trace_id = capture()
    captured_programs = mesh_device.num_program_cache_entries()

    try:
        live_slices = set(model._prefill_rope_slices)
        first = _prefill(model, short_ids, page_table)
        kept = set(model._prefill_rope_slices) - live_slices
        if mesh_device.num_program_cache_entries() != captured_programs:
            # As the generator does at the start of the next request.
            ttnn.release_trace(mesh_device, trace_id)
            trace_id = None
            trace_id = capture()
        for _ in range(REPLAYS):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        again = _prefill(model, short_ids, page_table)
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh_device, trace_id)

    assert torch.isfinite(first).all()
    assert torch.equal(
        first, again
    ), f"prefill logits changed after decode replays (max abs diff {(first - again).abs().max().item()})"
    assert not kept, f"an eager prefill kept device tensors alive (rope slices for lengths {sorted(kept)})"
