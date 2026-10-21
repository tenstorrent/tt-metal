# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight decoder parity and traced cached-continuation regressions."""

import copy

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4_26b_a4b_qb2.tt.generator import Gemma4Generator
from models.demos.gemma4_26b_a4b_qb2.tt.model import Checkpoint


def host(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).float()


def assert_decoder_close(reference, actual):
    # Quantized BFP8 weights/KV and routed MoE arithmetic use the bring-up's
    # 0.995 layer PCC gate, not the 0.999 gate for a single BF16 primitive.
    assert torch.isfinite(actual).all()
    passed, pcc = comp_pcc(reference, actual, 0.995)
    assert passed, f"Checkpoint decoder PCC {pcc} < 0.995"


def check_decoder(generator, layer_number):
    model = generator.model
    index = model.layer_indices[layer_number]
    config = copy.deepcopy(model.config)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        reference = Gemma4TextDecoderLayer(config, index)
    reference.load_state_dict(
        {key: value.float() for key, value in Checkpoint().load(f"model.language_model.layers.{index}.").items()},
        assign=True,
    )
    reference.eval()
    x = torch.randn(1, 65, config.hidden_size).bfloat16().float()
    kind = config.layer_types[index]
    cos, sin = Gemma4TextRotaryEmbedding(config)(x, torch.arange(128)[None], layer_type=kind)
    mask = torch.zeros(64, 64).masked_fill(torch.triu(torch.ones(64, 64, dtype=torch.bool), 1), -torch.inf)
    hf_cache = DynamicCache()
    with torch.no_grad():
        expected = reference(
            x[:, :64],
            position_embeddings=(cos[:, :64], sin[:, :64]),
            attention_mask=mask[None, None],
            past_key_values=hf_cache,
        )
    caches, table = model.allocate_cache(slots=1, context=128)
    layer, cache = model.layers[layer_number], caches[layer_number]
    device_table = model.upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    actual = layer.prefill_forward(
        model.upload(x[:, :64][None]),
        rope_mats=model.rope_prefill[kind],
        page_table=device_table,
        kv_cache=cache,
    )
    assert_decoder_close(expected, host(actual).squeeze(0))
    with torch.no_grad():
        expected = reference(
            x[:, 64:],
            position_embeddings=(cos[:, 64:65], sin[:, 64:65]),
            attention_mask=torch.zeros(1, 1, 1, 65),
            past_key_values=hf_cache,
        )
    hidden = model.upload(x[:, 64:][None])
    position = model.upload(torch.tensor([[64]], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    cache_position = model.upload(torch.tensor([64], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    def forward():
        return layer.decode_forward(
            hidden,
            rope_mats=model.rope_decode[kind],
            current_pos=position,
            cache_pos=cache_position,
            page_table=device_table,
            kv_cache=cache,
        )

    forward()  # Compile the exact decode signature before capture.
    ttnn.synchronize_device(model.mesh)
    trace = ttnn.begin_trace_capture(model.mesh, cq_id=0)
    actual = forward()
    ttnn.end_trace_capture(model.mesh, trace, cq_id=0)
    try:
        ttnn.execute_trace(model.mesh, trace, cq_id=0, blocking=True)
        assert_decoder_close(expected, host(actual).squeeze(0))
    finally:
        ttnn.release_trace(model.mesh, trace)


def snapshot(caches):
    return [
        [ttnn.to_torch(device).float().clone() for tensor in pair for device in ttnn.get_device_tensors(tensor)]
        for pair in caches
    ]


def check_continuation(generator, start):
    model = generator.model
    end = start + 9
    tokens = torch.randint(10, 1000, (1, end))
    caches, table = model.allocate_cache(slots=2, context=end)
    active_table = table[:1]
    generator.prefill_forward(tokens[:, :start], page_table=active_table, kv_cache=caches, prompt_lens=[start])
    # Compare with the same teacher-forced decode arithmetic, not cold prefill.
    expected = None
    for position in range(start, end):
        logits = generator.decode_forward(
            tokens[:, position : position + 1],
            torch.tensor([position]),
            page_table=active_table,
            kv_cache=caches,
            return_logits=True,
        )
        expected = generator._read_logits(logits)[..., :1, :].clone()
    generator._release_trace()
    caches, table = model.allocate_cache(slots=2, context=end)
    active_table = table[:1]
    generator.prefill_forward(tokens[:, :start], page_table=active_table, kv_cache=caches, prompt_lens=[start])
    before = snapshot(caches)
    actual = generator.prefill_forward(
        tokens,
        page_table=active_table,
        kv_cache=caches,
        prompt_lens=[end],
        start_pos=[start],
    )
    actual = generator._read_logits(actual).reshape(1, 1, 1, -1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    after = snapshot(caches)
    for old_layer, new_layer in zip(before, after):
        for old, new in zip(old_layer, new_layer):
            assert torch.equal(old[table[1].long()], new[table[1].long()]), "Neighbor request pages changed"
            heads, width = old.shape[1], old.shape[-1]
            old_prefix = old[table[0].long()].permute(1, 0, 2, 3).reshape(heads, -1, width)[:, :start]
            new_prefix = new[table[0].long()].permute(1, 0, 2, 3).reshape(heads, -1, width)[:, :start]
            assert torch.equal(old_prefix, new_prefix), "Cached prefix changed"
    generator._release_trace()


def test_checkpoint_decoder_and_cached_continuation(qb2_mesh, monkeypatch):
    torch.manual_seed(42)
    torch.set_num_threads(8)
    monkeypatch.setenv("GEMMA4_PREFIX_TRACE_SUFFIX", "1")
    # Exercise the submitted TP4 decoder and precision policy for both attention kinds.
    generator = Gemma4Generator(qb2_mesh, max_seq_len=2048, layer_indices=(0, 5))
    try:
        for layer_number in range(2):
            check_decoder(generator, layer_number)
        for start in (31, generator.model.config.sliding_window - 1):
            check_continuation(generator, start)
    finally:
        generator._release_trace()
