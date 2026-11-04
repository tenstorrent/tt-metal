# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A prefill of a new length after the decode trace capture must not be corrupted by later decode replays.

A device tensor allocated after a trace was captured can sit in that trace's scratch region and is overwritten by every
replay of it. Model.prepare_inputs_prefill used to cache the RoPE slice of each prefill length on its first use, so the
slice of a length first seen after the decode trace capture was overwritten by the decode replays and every later
prefill of that length returned garbage (served-style request mix: every 128-token prompt after a 4096-token one).
The new length's programs can also own long-lived device buffers (e.g. the global semaphores of the prefill's
ttnn.all_reduce) placed in that region, so the generator re-captures the decode trace when programs were compiled
since its capture (Model._tt_recapture_decode_trace_on_new_programs).

Through the generator, as the demo and vLLM drive it: a 2-layer model with random weights prefills a long prompt,
decodes one traced step (the generator captures the decode trace), prefills a short prompt of a new padded length (the
generator re-captures the decode trace at the end of that prefill), replays decode steps that reload no inputs (as
async decode does), prefills the short prompt again and requires bit-identical logits. It also requires that an eager
prefill keeps no device tensor alive in the model, and that a prompt of another length within the same padded length
compiles no program (its last token is in another tile: the prefill tail slices it with runtime bounds).

Bit-identical logits need a reproducible prefill. The model is built in the fused decode configuration (gpt-oss-20b on
a whole 1x4 Blackhole mesh, TP 4, batch 1), where every prefill all-reduce is one ttnn.all_reduce (attention/prefill.py
passes use_all_reduce=fused_decode, the MoE always does); the default reduce_scatter + all_gather pair of the attention
output is not bit-reproducible on Blackhole.

Run on a QuietBox 2 (Blackhole 1x4):
    HF_MODEL=models/demos/gpt_oss/configs/gpt-oss-20b pytest models/demos/gpt_oss/tests/unit/test_prefill_after_decode_trace.py
"""

import os

import pytest
import torch

import ttnn
from models.ttt_compat.tt.common import PagedAttentionConfig
from models.ttt_compat.tt.generator import Generator
from models.ttt_compat.tt.load_checkpoints import convert_hf_qkv_to_meta_format

from ..test_factory import TestFactory, parametrize_mesh_with_fabric

NUM_LAYERS = 2
PAGE_BLOCK = 64
PAGE_BLOCKS = 64  # 4096-token KV cache
LONG_PROMPT = 2000  # padded to 2048
SHORT_PROMPT = 100  # padded to 128: a prefill length first seen after the decode trace capture; last token in tile 96
OTHER_SHORT_PROMPT = 40  # also padded to 128, last token in tile 32
DECODE_POS = 3000  # the decode steps write the KV cache past every prompt
DECODE_STEPS = 16


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Needs a QuietBox 2 for about a minute; the CI gpt-oss-20b unit job has a 5-minute budget.",
)
@parametrize_mesh_with_fabric([(1, 4)])
def test_prefill_after_decode_trace_capture(mesh_device, device_params, reset_seeds):
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.model import Model

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]
    config.num_hidden_layers = NUM_LAYERS
    state_dict = convert_hf_qkv_to_meta_format(GptOssForCausalLM(config).eval().state_dict(), config.head_dim)

    mesh_shape = tuple(mesh_device.shape)
    model = Model(
        mesh_device=mesh_device,
        hf_config=config,
        state_dict=state_dict,
        ccl_manager=setup["ccl_manager"],
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        paged_attention_config=PagedAttentionConfig(block_size=PAGE_BLOCK, max_num_blocks=PAGE_BLOCKS),
        mesh_config=MeshConfig(mesh_shape, decode=ModeConfig(tp=mesh_shape[1], ep=mesh_shape[0])),
        create_kv_cache=True,
        max_local_batch_size=1,
    )
    assert all(layer.self_attn.fused_decode for layer in model.layers), (
        "the prefill all-reduce is bit-reproducible only in the fused decode configuration "
        "(gpt-oss-20b, whole 1x4 Blackhole mesh, batch 1)"
    )
    model_args = setup["model_args"]  # dummy-weight ModelArgs (max batch 1) skip the HF config: give the logits size
    model_args.vocab_size = config.vocab_size
    generator = Generator([model], [model_args], mesh_device)
    kv_cache = [[layer.self_attn.layer_past for layer in model.layers]]
    page_table = torch.arange(PAGE_BLOCKS, dtype=torch.int32).reshape(1, PAGE_BLOCKS)

    def prefill(ids):
        return generator.prefill_forward_text(
            ids,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=[ids.shape[-1]],
            enable_trace=False,
            warmup_prefill=False,
        ).float()

    def decode(reload_inputs):
        return generator.decode_forward(
            torch.tensor([[7]]),
            torch.tensor([DECODE_POS]),
            page_table=page_table,
            kv_cache=kv_cache,
            enable_trace=True,
            read_from_device=False,
            sampling_params=None,
            reload_inputs=reload_inputs,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )

    long_ids = torch.randint(0, config.vocab_size, (1, LONG_PROMPT))
    short_ids = torch.randint(0, config.vocab_size, (1, SHORT_PROMPT))
    other_short_ids = torch.randint(0, config.vocab_size, (1, OTHER_SHORT_PROMPT))

    try:
        prefill(long_ids)
        decode(reload_inputs=True)  # the generator compiles and captures the decode trace
        captured = dict(generator.trace_ids_decode[False])
        assert generator._decode_trace_recaptures == 0

        live_slices = set(model._prefill_rope_slices)
        first = prefill(short_ids)  # compiles the 128-token programs after the capture
        kept = set(model._prefill_rope_slices) - live_slices
        assert generator._decode_trace_recaptures == 1, "the new prefill length did not re-capture the decode trace"
        assert generator.trace_ids_decode[False] != captured

        for _ in range(DECODE_STEPS):
            decode(reload_inputs=False)
        ttnn.synchronize_device(mesh_device)
        again = prefill(short_ids)

        programs = mesh_device.num_program_cache_entries()
        prefill(other_short_ids)
        new_programs = mesh_device.num_program_cache_entries() - programs
        recaptures = generator._decode_trace_recaptures
    finally:
        generator.release_persistent_capture()

    assert torch.isfinite(first).all()
    assert torch.equal(
        first, again
    ), f"prefill logits changed after decode replays (max abs diff {(first - again).abs().max().item()})"
    assert not kept, f"an eager prefill kept device tensors alive (rope slices for lengths {sorted(kept)})"
    assert new_programs == 0, f"a new prompt length in a known padded length compiled {new_programs} programs"
    assert recaptures == 1, f"{recaptures} decode trace re-captures, expected 1 (for the new padded length)"
