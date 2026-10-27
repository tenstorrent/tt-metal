# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight decoder parity and traced cached-continuation regressions."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4_26b_a4b_qb2.tt.generator import Gemma4Generator
from models.demos.gemma4_26b_a4b_qb2.tt.model import MODEL_ID, REVISION, Checkpoint

PREFILL_LENGTH = 65


def host(tensor):
    replicas = [ttnn.to_torch(value).float() for value in ttnn.get_device_tensors(tensor)]
    assert all(torch.equal(replicas[0], value) for value in replicas[1:]), "Decoder output replicas differ"
    return replicas[0]


def assert_decoder_close(reference, actual, label):
    # Preserve the original text-activation/FP32-HF accuracy gate. Arbitrary
    # Gaussian hidden states were diagnostic stress inputs, not this contract.
    assert torch.isfinite(reference).all() and torch.isfinite(actual).all(), label
    passed, pcc = comp_pcc(reference, actual, 0.995)
    print(f"DECODER_NUMERICS {label} pcc={pcc}", flush=True)
    assert passed, f"{label}: checkpoint decoder PCC {pcc} < 0.995"


def reference_layer(config, index):
    with torch.device("meta"):
        layer = Gemma4TextDecoderLayer(config, index)
    layer.load_state_dict(
        {key: value.float() for key, value in Checkpoint().load(f"model.language_model.layers.{index}.").items()},
        assign=True,
    )
    return layer.eval()


@pytest.fixture(scope="module")
def text_layer_inputs():
    """Regenerate the original FP32 HF input recipe from small, pinned token IDs."""
    fixture = json.loads(Path(__file__).with_name("decoder_input_tokens.json").read_text())
    assert fixture["model"] == MODEL_ID
    assert fixture["revision"] == fixture["tokenizer_revision"] == REVISION
    tokens = torch.tensor(fixture["token_ids"])
    length = PREFILL_LENGTH + 1
    assert tokens.numel() == length
    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION).text_config
    config._attn_implementation = "eager"
    with torch.inference_mode():
        embedding = Checkpoint().load("model.language_model.embed_tokens.")["weight"]
        hidden = embedding[tokens].float()[None] * torch.tensor(config.hidden_size**0.5, dtype=torch.float32)
        del embedding
        inputs = {0: hidden.bfloat16().float()}
        mask = torch.zeros(length, length).masked_fill(
            torch.triu(torch.ones(length, length, dtype=torch.bool), 1), -torch.inf
        )[None, None]
        rotary = Gemma4TextRotaryEmbedding(config)
        # Layer 5 must receive the real HF 0..4 boundary, not layer-0 or random input.
        for index in range(5):
            layer = reference_layer(config, index)
            rope = rotary(hidden, torch.arange(length)[None], layer_type=config.layer_types[index])
            hidden = layer(hidden, position_embeddings=rope, attention_mask=mask)
            del layer
        inputs[5] = hidden.bfloat16().float()
    return inputs


@torch.inference_mode()
def check_decoder(generator, layer_number, x):
    model = generator.model
    index = model.layer_indices[layer_number]
    config = copy.deepcopy(model.config)
    config._attn_implementation = "eager"
    reference = reference_layer(config, index)
    length = PREFILL_LENGTH
    assert x.shape == (1, length + 1, config.hidden_size)
    label = (
        f"layer={index} length={length} precision={model.precision_config['config_id']} "
        f"input_sha256={hashlib.sha256(x.numpy().tobytes()).hexdigest()}"
    )
    kind = config.layer_types[index]
    cos, sin = Gemma4TextRotaryEmbedding(config)(x, torch.arange(128)[None], layer_type=kind)
    mask = torch.zeros(length, length).masked_fill(
        torch.triu(torch.ones(length, length, dtype=torch.bool), 1), -torch.inf
    )
    hf_cache = DynamicCache()
    with torch.no_grad():
        expected = reference(
            x[:, :length],
            position_embeddings=(cos[:, :length], sin[:, :length]),
            attention_mask=mask[None, None],
            past_key_values=hf_cache,
        )
    caches, table = model.allocate_cache(slots=1, context=128)
    layer, cache = model.layers[layer_number], caches[layer_number]
    device_table = model.upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    actual = layer.prefill_forward(
        model.upload(x[:, :length][None]),
        rope_mats=model.rope_prefill[kind],
        page_table=device_table,
        kv_cache=cache,
    )
    assert_decoder_close(expected, host(actual).squeeze(0), label + " phase=prefill")
    with torch.no_grad():
        expected = reference(
            x[:, length:],
            position_embeddings=(cos[:, length : length + 1], sin[:, length : length + 1]),
            attention_mask=torch.zeros(1, 1, 1, length + 1),
            past_key_values=hf_cache,
        )
    hidden = model.upload(x[:, length:][None])
    position = model.upload(torch.tensor([[length]], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    cache_position = model.upload(torch.tensor([length], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

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
        assert_decoder_close(expected, host(actual).squeeze(0), label + " phase=traced_decode")
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


@pytest.fixture
def generator(qb2_mesh):
    torch.manual_seed(42)
    torch.set_num_threads(8)
    generator = Gemma4Generator(qb2_mesh, max_seq_len=2048, layer_indices=(0, 5))
    try:
        yield generator
    finally:
        generator._release_trace()


@pytest.mark.parametrize("layer_number", [0, 1], ids=["sliding", "full"])
def test_checkpoint_decoder_numerics(generator, text_layer_inputs, layer_number):
    check_decoder(generator, layer_number, text_layer_inputs[generator.model.layer_indices[layer_number]])


@pytest.mark.parametrize("boundary", ["partial_page", "sliding_window"])
def test_cached_continuation(generator, monkeypatch, boundary):
    monkeypatch.setenv("GEMMA4_PREFIX_TRACE_SUFFIX", "1")
    start = 31 if boundary == "partial_page" else generator.model.config.sliding_window - 1
    check_continuation(generator, start)
