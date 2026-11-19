# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TT language-model prefill against the torch reference on one Blackhole P100."""

from dataclasses import replace

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.alpamayo2_edge.reference import golden_cache
from models.experimental.alpamayo2_edge.reference.model import ReferenceModel
from models.experimental.alpamayo2_edge.reference.text import (
    MLP,
    DecoderLayer,
    KVCache,
    attention_bias,
    mrope_cos_sin,
)
from models.experimental.alpamayo2_edge.reference.weights import load_weights
from models.experimental.alpamayo2_edge.tests._util import PCC, pcc
from models.experimental.alpamayo2_edge.tests.conftest import P100_MESH
from models.experimental.alpamayo2_edge.tt.common import round_up
from models.experimental.alpamayo2_edge.tt.text import AlpamayoCache, ReluSquaredFeedForward, TtLanguageModel
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality


@pytest.fixture(scope="module")
def lm_weights(model_dir):
    return load_weights(model_dir, "vlm.model.language_model.")


@pytest.fixture(scope="module")
def head_weights(model_dir):
    return load_weights(model_dir, "vlm.lm_head.")


@pytest.mark.parametrize("prompt", [3236, 3200, 3256, 3329])
@pytest.mark.parametrize("max_new_tokens, action_rows", [(256, 64), (1, 64), (1000, 32)])
def test_cache_size_covers_prefill_padding(prompt, max_new_tokens, action_rows):
    size = AlpamayoCache.size_for(prompt, max_new_tokens=max_new_tokens, action_rows=action_rows)
    assert size % 64 == 0
    # The prefill writes every padded row; the expert then needs its action rows after the 32-aligned prefix
    # of up to `max_new_tokens` generated tokens.
    assert size >= round_up(prompt, 128)
    assert size >= round_up(prompt + max_new_tokens, 32) + action_rows
    assert size < max(round_up(prompt, 128), round_up(prompt + max_new_tokens, 32) + action_rows) + 64


@P100_MESH
def test_lm_head_dtype(mesh_device, cfg):
    config = TtLanguageModel.config_from(replace(cfg.text, layers=1))
    default = TtLanguageModel(config, device=mesh_device)
    bf16 = TtLanguageModel(config, device=mesh_device, lm_head_dtype=ttnn.bfloat16)
    assert default.final_linear.weight.dtype == ttnn.bfloat8_b
    assert bf16.final_linear.weight.dtype == ttnn.bfloat16
    assert bf16.final_linear.weight.total_shape == default.final_linear.weight.total_shape


@P100_MESH
def test_relu2_ffn(mesh_device, cfg, lm_weights):
    reference = MLP(lm_weights, "layers.0.mlp.", torch.float32)
    torch.manual_seed(0)
    x = torch.randn(1, 128, cfg.text.hidden)
    with torch.no_grad():
        golden = reference(x)

    ffn = ReluSquaredFeedForward(cfg.text.hidden, cfg.text.mlp, mesh_device=mesh_device)
    ffn.load_torch_state_dict(
        {
            "linear_in.weight": lm_weights["layers.0.mlp.fc1.weight"],
            "linear_out.weight": lm_weights["layers.0.mlp.fc2.weight"],
        }
    )
    out = tensor.to_torch(ffn.forward(tensor.from_torch(x, device=mesh_device)))
    assert out.shape == golden.shape
    assert_quality(golden, out, pcc=PCC)


@P100_MESH
@pytest.mark.timeout(1500)
def test_text_layer(mesh_device, cfg, lm_weights, head_weights, sample_inputs):
    """Layer 0 on random hidden states, fed through `forward` as all-vision rows so the real prefill path runs."""
    tcfg = replace(cfg.text, layers=1)
    position_ids = sample_inputs.position_ids
    seq = position_ids.shape[-1]
    torch.manual_seed(0)
    hidden = torch.randn(1, seq, tcfg.hidden)

    reference = DecoderLayer(tcfg, lm_weights, 0, torch.float32)
    cos, sin = mrope_cos_sin(position_ids, tcfg.head_dim, tcfg.rope_theta, tcfg.mrope_section)
    with torch.no_grad():
        golden = reference(hidden, cos, sin, attention_bias(seq, 0, None, True, torch.float32), None)

    config = replace(TtLanguageModel.config_from(tcfg), final_norm=False, final_linear=False)
    model = TtLanguageModel(config, device=mesh_device)
    model.load_torch_state_dict(TtLanguageModel.convert_state(lm_weights, head_weights))
    out = model.forward(
        tensor.from_torch(sample_inputs.input_ids, device=mesh_device, dtype=ttnn.uint32),
        positions=tensor.from_torch(position_ids.float(), device=mesh_device, dtype=ttnn.float32),
        vision_embeds=tensor.from_torch(hidden[0], device=mesh_device),
        vision_mask=tensor.from_torch(torch.ones(1, seq), device=mesh_device),
        skip_final_linear=True,
    )
    out = tensor.to_torch(out)
    assert out.shape == golden.shape
    logger.info(f"layer 0 PCC {pcc(golden, out) * 100:.4f} %")
    assert_quality(golden, out, pcc=PCC)


def _reference_prefill(cfg, inputs) -> dict:
    """Last-token logits, the vision embeddings the LM consumed, and K/V of the first and last layers."""
    model = ReferenceModel.from_pretrained(cfg)
    captured = {}
    model.vision.register_forward_hook(lambda _m, _a, out: captured.update(vision=out))

    def progress(index):
        return lambda *_: print(f"reference LM layer {index + 1}/{cfg.text.layers}", flush=True)

    for i, layer in enumerate(model.lm.text.layers):
        layer.register_forward_hook(progress(i))
    logits, kv = model.prefill(inputs)
    last = cfg.text.layers - 1
    return {
        "logits": logits,
        "vision": captured["vision"],
        "k": {0: kv.k[0], last: kv.k[last]},
        "v": {0: kv.v[0], last: kv.v[last]},
    }


@P100_MESH
@pytest.mark.timeout(1500)
def test_prefill(mesh_device, cfg, lm_weights, head_weights, sample_inputs):
    inputs = sample_inputs
    golden = golden_cache.cached(
        "lm_prefill",
        {
            "input_ids": inputs.input_ids,
            "position_ids": inputs.position_ids,
            "patches": inputs.patches,
            "grid": inputs.image_grid_thw,
        },
        lambda: _reference_prefill(cfg, inputs),
        cfg=cfg,
    )
    seq = inputs.input_ids.shape[1]
    image_rows = inputs.input_ids == cfg.tokens.image
    assert int(image_rows.sum()) == golden["vision"].shape[0]
    logger.info(f"S={seq}, {golden['vision'].shape[0]} vision rows, {cfg.text.layers} layers")

    model = TtLanguageModel(TtLanguageModel.config_from(cfg.text), device=mesh_device)
    model.load_torch_state_dict(TtLanguageModel.convert_state(lm_weights, head_weights))
    cache = AlpamayoCache(
        device=mesh_device, size=AlpamayoCache.size_for(seq, max_new_tokens=256, action_rows=64), batch_size=1
    )

    logits = model.prefill(
        tensor.from_torch(inputs.input_ids, device=mesh_device, dtype=ttnn.uint32),
        positions=tensor.from_torch(inputs.position_ids.float(), device=mesh_device, dtype=ttnn.float32),
        vision_embeds=tensor.from_torch(golden["vision"], device=mesh_device),
        vision_mask=tensor.from_torch(image_rows, device=mesh_device),
        cache=cache,
    )
    logits = tensor.to_torch(logits).float()
    assert logits.shape == (1, cfg.text.vocab)
    logger.info(
        f"last-token logits PCC {pcc(golden['logits'], logits) * 100:.4f} %, "
        f"argmax tt {int(logits.argmax())} ref {int(golden['logits'].argmax())}"
    )

    assert int(tensor.to_torch(cache.position)[0]) == seq
    results = {}
    for layer in golden["k"]:
        k, v = (tensor.to_torch(t) for t in cache.layer_kv(layer))
        assert k.shape == v.shape == (1, cfg.text.kv_heads, cache.size, cfg.text.head_dim)
        for name, actual, expected in (("K", k, golden["k"][layer]), ("V", v, golden["v"][layer])):
            results[f"{name}{layer}"] = (expected, actual[:, :, :seq])
            logger.info(f"layer {layer} {name} PCC {pcc(expected, actual[:, :, :seq]) * 100:.4f} %")

    # A cache rebuilt from the reference must look exactly like the one the prefill filled.
    kv = KVCache(cfg.text.layers)
    for layer in golden["k"]:
        kv.k[layer], kv.v[layer] = golden["k"][layer], golden["v"][layer]
    rebuilt = AlpamayoCache.from_reference(kv, size=cache.size, mesh_device=mesh_device)
    for layer in golden["k"]:
        for ours, theirs in zip(rebuilt.layer_kv(layer), cache.layer_kv(layer)):
            assert ours.shape == theirs.shape
            assert ours.dtype == theirs.dtype
            assert ours.layout == theirs.layout

    assert_quality(golden["logits"], logits, pcc=PCC)
    for expected, actual in results.values():
        assert_quality(expected, actual, pcc=PCC)


@P100_MESH
def test_cache_from_reference(mesh_device, cfg):
    tcfg = cfg.text
    rows = 100
    torch.manual_seed(0)
    kv = KVCache(2)
    for layer in range(2):
        kv.append(
            layer,
            torch.randn(1, tcfg.kv_heads, rows, tcfg.head_dim),
            torch.randn(1, tcfg.kv_heads, rows, tcfg.head_dim),
        )
    size = AlpamayoCache.size_for(rows, max_new_tokens=256, action_rows=64)
    cache = AlpamayoCache.from_reference(kv, size=size, mesh_device=mesh_device)
    assert cache.size == size and cache.batch_size == 1
    assert int(tensor.to_torch(cache.position)[0]) == rows

    for layer in range(2):
        for actual, expected in zip(cache.layer_kv(layer), (kv.k[layer], kv.v[layer])):
            assert actual.dtype == ttnn.bfloat16 and actual.layout == ttnn.TILE_LAYOUT
            actual = tensor.to_torch(actual)
            assert actual.shape == (1, tcfg.kv_heads, size, tcfg.head_dim)
            assert torch.equal(actual[:, :, :rows], expected.bfloat16())
            assert not actual[:, :, rows:].any()

    repeated = cache.repeat_to(3)
    assert repeated.batch_size == 3 and repeated.size == size
    assert tensor.to_torch(repeated.position).tolist() == [rows] * 3
    for layer in range(2):
        for actual, source in zip(repeated.layer_kv(layer), cache.layer_kv(layer)):
            assert torch.equal(tensor.to_torch(actual), tensor.to_torch(source).expand(3, -1, -1, -1))


@P100_MESH
def test_cache_rejects_bad_sizes(mesh_device, cfg, expect_error):
    kv = KVCache(1)
    shape = (1, cfg.text.kv_heads, 200, cfg.text.head_dim)
    kv.append(0, torch.zeros(shape), torch.zeros(shape))
    with expect_error(ValueError, "multiple of 64"):
        AlpamayoCache(device=mesh_device, size=100, batch_size=1)
    with expect_error(ValueError, "cannot hold"):
        AlpamayoCache.from_reference(kv, size=192, mesh_device=mesh_device)
