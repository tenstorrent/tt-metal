# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""CPU control-flow coverage only; no learned/numerical or native qualification."""
import numpy as np
import pytest

from .test_generate_validation import controlled_backend, finite_logits, request, unchanged
from .backend import Backend


@pytest.fixture
def make_model():
    def make(plan=(), positions=1024):
        seed, observed = controlled_backend(plan)
        model = object.__new__(Backend)
        model.__dict__.update(seed.__dict__)
        model.config["max_position_embeddings"] = positions
        return model, observed

    return make


@pytest.mark.parametrize(
    "cap,positions",
    [(1, 2), (64, 128), (65, 128), (127, 128), (128, 129), (129, 130), (255, 256), (256, 257), (256, 1024)],
)
def test_explicit_cap_prefix_and_accounting(make_model, cap, positions):
    model, seen = make_model([finite_logits(7)] * (cap - 1), positions)
    inputs = request()
    if cap == 1:
        inputs = tuple(x[:, :2].copy() for x in inputs)
    originals = tuple(x.copy() for x in inputs)
    out = model.generate(*inputs, np.int64(4), np.int64(cap))
    assert out.tolist() == [[2, 4] + [7] * (cap - 1)]
    assert out.dtype == np.int64
    assert [p.shape[1] for p in seen["prefixes"]] == list(range(2, cap + 1))
    assert all(c == {} for c in seen["caches"])
    unchanged(inputs, originals)


@pytest.mark.parametrize(
    "positions,cap",
    [
        (128, 128),
        (65, 65),
        (129, 129),
        (256, 256),
        (257, 257),
        (1024, 257),
        (1024, 0),
        (1024, -1),
        (1024, True),
        (1024, np.bool_(False)),
        (1024, 65.0),
        (1024, "65"),
        (1024, None),
    ],
)
def test_invalid_cap_before_encode(make_model, positions, cap):
    model, seen = make_model(positions=positions)
    with pytest.raises(ValueError, match="max_new_tokens"):
        model.generate(*request(), 4, cap)
    assert seen["encodes"] == [] and seen["prefixes"] == []


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_cap_prefix_consistency_and_aba(make_model, batch):
    model, seen = make_model()
    ids, mask = (np.repeat(x, batch, axis=0) for x in request())

    def run(cap):
        seen["pending"].extend([finite_logits(7)] * ((cap - 1) * batch))
        return model.generate(ids, mask, 4, cap)

    a = run(65)
    b = run(128)
    again = run(65)
    np.testing.assert_array_equal(a, b[:, :66])
    np.testing.assert_array_equal(a, again)
    assert all(c == {} for c in seen["caches"])


@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize("mask_dtype", [np.int64, np.bool_])
def test_mixed_eos_pad_masks_and_single_reverse_rightpadding(make_model, batch, mask_dtype):
    rows = [[5 + i, 1, 2, 1] for i in range(batch)]
    ids = np.array(rows, dtype=np.int64)
    mask = np.array([[1, 0, 1, 0]] * batch, dtype=mask_dtype)
    lengths = [1, 66, 3, 127][:batch]
    plans = [[finite_logits(7)] * (n - 1) + [finite_logits(2)] for n in lengths]

    def run(order, padding=0):
        model, seen = make_model([x for i in order for x in plans[i]])
        source = np.pad(ids[order], ((0, 0), (0, padding)), constant_values=1)
        masks = np.pad(mask[order], ((0, 0), (0, padding)), constant_values=0)
        originals = source.copy(), masks.copy()
        out = model.generate(source, masks, 4, 128)
        unchanged((source, masks), originals)
        for j, i in enumerate(order):
            wanted = [2, 4] + [7] * (lengths[i] - 1) + [2]
            assert out[j, : len(wanted)].tolist() == wanted
            assert np.all(out[j, len(wanted) :] == 1)
            np.testing.assert_array_equal(seen["encodes"][j][1], masks[j : j + 1])
        assert len(seen["prefixes"]) == sum(lengths[i] for i in order)
        assert all(c == {} for c in seen["caches"])
        return out

    order = list(range(batch))
    out = run(order)
    np.testing.assert_array_equal(run(order, 3), out)
    np.testing.assert_array_equal(run(order[::-1]), out[::-1])
    for i in order:
        single = run([i])
        np.testing.assert_array_equal(single[0], out[i, : single.shape[1]])


@pytest.mark.parametrize("fault", ["exception", "nan", "inf"])
def test_failure_after64_clears_cache_and_recovers(make_model, fault):
    bad = finite_logits()
    bad[0, 0, 0] = np.nan if fault == "nan" else np.inf
    failure = RuntimeError("long decode failure") if fault == "exception" else bad
    model, seen = make_model([finite_logits(7)] * 63 + [failure, finite_logits(2)])
    inputs = request()
    originals = tuple(x.copy() for x in inputs)
    with pytest.raises(RuntimeError if fault == "exception" else FloatingPointError):
        model.generate(*inputs, 4, 128)
    assert seen["prefixes"][-1].shape[1] == 65
    assert all(c == {} for c in seen["caches"])
    np.testing.assert_array_equal(model.generate(*inputs, 4, 128), [[2, 4, 2]])
    assert seen["caches"][-1] is not seen["caches"][0]
    unchanged(inputs, originals)


@pytest.mark.parametrize("kind", ["source257", "batch5", "mask_shape", "mask_value", "all_masked"])
def test_input_guards_still_precede_encode(make_model, kind):
    model, seen = make_model()
    ids, mask = request()
    if kind == "source257":
        ids, mask = np.full((1, 257), 5), np.ones((1, 257), dtype=np.int64)
    elif kind == "batch5":
        ids, mask = np.repeat(ids, 5, axis=0), np.repeat(mask, 5, axis=0)
    elif kind == "mask_shape":
        mask = mask[:, :-1]
    elif kind == "mask_value":
        mask[0, 0] = 2
    else:
        mask[:] = 0
    with pytest.raises(ValueError):
        model.generate(ids, mask, 4, 128)
    assert not seen["encodes"] and not seen["prefixes"]


def test_public_forward_remains64(make_model):
    model, seen = make_model()
    with pytest.raises(ValueError):
        model.forward(*request(), np.full((1, 65), 4, dtype=np.int64))
    assert not seen["encodes"]


@pytest.mark.parametrize("positions", [128, 257, 1024])
@pytest.mark.parametrize("entry", ["cli", "api"])
def test_text_cap_admission_before_tokenization(monkeypatch, tmp_path, positions, entry):
    import importlib
    import json
    from pathlib import Path

    text_api = importlib.import_module(".translate", __package__)
    config = json.loads((Path(__file__).parent / "benchmark_cases/600m/config.json").read_text())
    config["max_position_embeddings"] = positions
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    seen = []

    class TokenizationReached(Exception):
        pass

    def stop(*args, **kwargs):
        seen.append(True)
        raise TokenizationReached()

    monkeypatch.setattr(text_api, "load_text_inputs", stop)

    def invoke(cap):
        if entry == "api":
            return text_api.translate(tmp_path, config, None, "eng_Latn", "fra_Latn", ["Hello"], max_new_tokens=cap)
        return text_api.main(
            [
                "--checkpoint",
                str(tmp_path),
                "--config",
                str(config_path),
                "--source-language",
                "eng_Latn",
                "--target-language",
                "fra_Latn",
                "--device",
                "0",
                "--max-new-tokens",
                str(cap),
                "--text",
                "Hello",
            ]
        )

    boundary = min(positions - 1, 256)
    for cap in (65, boundary):
        with pytest.raises(TokenizationReached):
            invoke(cap)
    assert len(seen) == 2
    with pytest.raises(ValueError, match="max_new_tokens"):
        invoke(boundary + 1)
    assert len(seen) == 2
