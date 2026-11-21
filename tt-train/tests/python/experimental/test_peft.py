# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LoRA adapters between ttml's fused layers and HF PEFT's per-projection ones.

Device-free. The layouts are the ones ``hf_layout`` gives a Llama, read off the loader's rules, so q's and k's rows
are permuted for RoPE as on a real model.
"""

import numpy as np
import pytest

from ttml.experimental.errors import UnsupportedTrainingConfig
from ttml.experimental.models.utils.peft import (
    PEFT_PREFIX,
    check_peft_keys,
    lora_to_peft,
    peft_adapter_config,
    peft_keys,
    peft_to_lora,
    with_peft_prefix,
)
from ttml.models.llama import HfTensor, HfTensors, LlamaConfig
from ttml.models.llama.safetensors_loader import _hf_name, _rules

HEADS, KV_HEADS, HIDDEN, INTERMEDIATE, RANK = 4, 2, 32, 64, 4
HEAD_DIM = HIDDEN // HEADS
Q_ROWS, KV_ROWS = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM

CONFIG = LlamaConfig(
    hidden_size=HIDDEN,
    num_attention_heads=HEADS,
    num_key_value_heads=KV_HEADS,
    num_hidden_layers=2,
    intermediate_size=INTERMEDIATE,
    vocab_size=64,
    max_position_embeddings=32,
)

QKV, OUT, GATE_UP, DOWN = "attention/qkv_linear", "attention/out_linear", "mlp/w_gate_up", "mlp/w2"
# The rows of each HF tensor a layer holds, and their columns.
SHAPES = {
    QKV: ((Q_ROWS, KV_ROWS, KV_ROWS), HIDDEN),
    OUT: ((HIDDEN,), HIDDEN),
    GATE_UP: ((INTERMEDIATE, INTERMEDIATE), HIDDEN),
    DOWN: ((HIDDEN,), INTERMEDIATE),
}
Q_AND_V = (True, False, True)


def layout(linear: str, layer: int = 0) -> HfTensors:
    """What ``hf_layout`` says the weight of ``blocks.<layer>.<linear>`` holds."""
    param = f"Llama/blocks/{layer}/{linear}/weight"
    rule = next(rule for rule in _rules(CONFIG, set()) if rule.param == param)
    rows, cols = SHAPES[linear]
    tensors = tuple(HfTensor(_hf_name(source), (n, cols)) for source, n in zip(rule.sources, rows))
    return HfTensors(tensors, transform=rule.transform, inverse=rule.inverse)


def block_rows(held: HfTensors) -> list[slice]:
    ends = np.cumsum([tensor.shape[0] for tensor in held.tensors])
    return [slice(end - tensor.shape[0], end) for end, tensor in zip(ends, held.tensors)]


def ttml_adapter(rng, held: HfTensors, adapted) -> tuple[np.ndarray, np.ndarray]:
    """A trained ttml layer's ``(a, b)``: random adapters, and zero rows in ``b`` for the blocks that aren't adapted."""
    a = rng.standard_normal((sum(adapted) * RANK, held.tensors[0].shape[1])).astype(np.float32)
    b = rng.standard_normal((sum(tensor.shape[0] for tensor in held.tensors), RANK)).astype(np.float32)
    for rows, has_adapter in zip(block_rows(held), adapted):
        if not has_adapter:
            b[rows] = 0.0
    return a, b


def peft_adapter(rng, held: HfTensors, adapted) -> dict[str, np.ndarray]:
    """Random PEFT tensors for the projections ``adapted`` flags."""
    tensors = {}
    for tensor, has_adapter in zip(held.tensors, adapted):
        if has_adapter:
            key_a, key_b = peft_keys(tensor)
            tensors[key_a] = rng.standard_normal((RANK, tensor.shape[1])).astype(np.float32)
            tensors[key_b] = rng.standard_normal((tensor.shape[0], RANK)).astype(np.float32)
    return tensors


# Every subset of the projections a layer holds that LoRA can target.
ADAPTED = [
    (QKV, (True, True, True)),
    (QKV, Q_AND_V),
    (QKV, (False, True, False)),
    (QKV, (False, False, True)),
    (GATE_UP, (True, True)),
    (GATE_UP, (False, True)),
    (OUT, (True,)),
    (DOWN, (True,)),
]
ADAPTED_IDS = ["qkv", "q,v", "k", "v", "gate,up", "up", "o", "down"]


def test_peft_names():
    (o_proj,) = layout(OUT).tensors
    assert peft_keys(o_proj) == (
        "base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight",
    )


def test_q_and_v_round_trip_through_peft():
    """The qkv layer of a run that targets q and v: PEFT gets adapters for q and v only, and they come back as the
    layer's, with k's rows of lora_B still zero."""
    held = layout(QKV)
    a, b = ttml_adapter(np.random.default_rng(0), held, Q_AND_V)
    q, k, v = held.tensors

    tensors = lora_to_peft(a, b, held, Q_AND_V)

    assert sorted(tensors) == sorted([*peft_keys(q), *peft_keys(v)])
    assert tensors[peft_keys(q)[1]].shape == (Q_ROWS, RANK) and tensors[peft_keys(v)[1]].shape == (KV_ROWS, RANK)
    a_back, b_back = peft_to_lora(tensors, held, Q_AND_V, RANK)
    np.testing.assert_array_equal(a_back, a)
    np.testing.assert_array_equal(b_back, b)
    assert not b_back[block_rows(held)[1]].any(), "k's rows of lora_B are not zero"


@pytest.mark.parametrize("linear, adapted", ADAPTED, ids=ADAPTED_IDS)
def test_a_ttml_adapter_round_trips_through_peft(linear, adapted):
    held = layout(linear)
    a, b = ttml_adapter(np.random.default_rng(1), held, adapted)
    a_back, b_back = peft_to_lora(lora_to_peft(a, b, held, adapted), held, adapted, RANK)
    np.testing.assert_array_equal(a_back, a)
    np.testing.assert_array_equal(b_back, b)


@pytest.mark.parametrize("linear, adapted", ADAPTED, ids=ADAPTED_IDS)
def test_a_peft_adapter_round_trips_through_ttml(linear, adapted):
    """As a run started from a PEFT adapter saves it again; the projections without an adapter get zero rows."""
    held = layout(linear)
    tensors = peft_adapter(np.random.default_rng(2), held, adapted)

    a, b = peft_to_lora(tensors, held, adapted, RANK)

    assert a.dtype == b.dtype == np.float32
    for rows, has_adapter in zip(block_rows(held), adapted):
        assert has_adapter or not b[rows].any()
    back = lora_to_peft(a, b, held, adapted)
    assert sorted(back) == sorted(tensors)
    for name, value in tensors.items():
        np.testing.assert_array_equal(back[name], value, err_msg=name)


@pytest.mark.parametrize("adapted", [(True, True, True), Q_AND_V], ids=["qkv", "q,v"])
def test_the_peft_adapters_make_the_layers_weight_update(adapted):
    """ttml's update to the fused weight, ``B_i @ A_i`` in each adapted block's rows, is PEFT's update to each
    projection with its rows mapped as the loader maps the weight's."""
    held = layout(QKV)
    a, b = ttml_adapter(np.random.default_rng(3), held, adapted)
    tensors = lora_to_peft(a, b, held, adapted)

    ttml_update = np.zeros((b.shape[0], a.shape[1]), np.float32)
    slot = 0
    for rows, has_adapter in zip(block_rows(held), adapted):
        if has_adapter:
            ttml_update[rows] = b[rows] @ a[slot * RANK : (slot + 1) * RANK]
            slot += 1
    hf_updates = []
    for tensor, has_adapter in zip(held.tensors, adapted):
        key_a, key_b = peft_keys(tensor)
        hf_updates.append(tensors[key_b] @ tensors[key_a] if has_adapter else np.zeros(tensor.shape, np.float32))
    np.testing.assert_allclose(np.concatenate(held.transform(*hf_updates)), ttml_update, rtol=1e-6, atol=1e-6)
    # Otherwise the check above would pass without the permutation.
    q_rows = block_rows(held)[0]
    assert not np.array_equal(tensors[peft_keys(held.tensors[0])[1]], b[q_rows]), "q's rows were not permuted"


def test_lora_b_must_have_the_rows_of_the_layer(expect_error):
    held = layout(QKV)
    a, b = ttml_adapter(np.random.default_rng(4), held, Q_AND_V)
    with expect_error(ValueError, f"lora_B has {Q_ROWS} rows"):
        lora_to_peft(a, b[:Q_ROWS], held, Q_AND_V)


def test_lora_a_must_have_a_block_per_adapted_projection(expect_error):
    held = layout(QKV)
    a, b = ttml_adapter(np.random.default_rng(5), held, Q_AND_V)
    with expect_error(ValueError, f"lora_A has {RANK} rows for 2 adapted blocks of rank {RANK}"):
        lora_to_peft(a[:RANK], b, held, Q_AND_V)


@pytest.mark.parametrize("ab", ["A", "B"])
def test_a_peft_tensor_of_another_shape_is_rejected(ab, expect_error):
    """An adapter of another rank, or for another model."""
    held = layout(QKV)
    tensors = peft_adapter(np.random.default_rng(6), held, Q_AND_V)
    key_a, key_b = peft_keys(held.tensors[2])
    key = key_a if ab == "A" else key_b
    tensors[key] = tensors[key][:, :-1] if ab == "B" else tensors[key][:-1]
    with expect_error(UnsupportedTrainingConfig, f"{key} has shape"):
        peft_to_lora(tensors, held, Q_AND_V, RANK)


def test_the_adapter_must_have_exactly_the_tensors_of_the_models_layers(expect_error):
    rng = np.random.default_rng(7)
    layers = [(layout(QKV, layer), Q_AND_V) for layer in range(2)] + [(layout(OUT), (True,))]
    tensors = {}
    for held, adapted in layers:
        tensors.update(peft_adapter(rng, held, adapted))
    check_peft_keys(tensors, layers)

    missing = {key: value for key, value in tensors.items() if "layers.1.self_attn.v_proj" not in key}
    with expect_error(UnsupportedTrainingConfig, r"missing \[.*layers\.1\.self_attn\.v_proj"):
        check_peft_keys(missing, layers)
    k_proj = peft_keys(layout(QKV).tensors[1])[0]
    with expect_error(UnsupportedTrainingConfig, r"unexpected \[.*layers\.0\.self_attn\.k_proj"):
        check_peft_keys({**tensors, k_proj: np.zeros((RANK, HIDDEN), np.float32)}, layers)


def test_bare_peft_names_get_the_prefix():
    named = with_peft_prefix({"model.layers.0.self_attn.o_proj.lora_A.weight": 1, f"{PEFT_PREFIX}x": 2})
    assert named == {f"{PEFT_PREFIX}model.layers.0.self_attn.o_proj.lora_A.weight": 1, f"{PEFT_PREFIX}x": 2}


def test_adapter_config():
    config = peft_adapter_config("meta-llama/Llama-3.2-1B", 8, 16, 0.05, ["v_proj", "q_proj"], "CAUSAL_LM")
    assert config["peft_type"] == "LORA" and config["task_type"] == "CAUSAL_LM"
    assert (config["r"], config["lora_alpha"], config["lora_dropout"]) == (8, 16, 0.05)
    assert config["target_modules"] == ["q_proj", "v_proj"]
    assert config["base_model_name_or_path"] == "meta-llama/Llama-3.2-1B"
    assert not config["use_rslora"] and not config["use_dora"] and config["bias"] == "none"
