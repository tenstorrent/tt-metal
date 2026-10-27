# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""CPU control-flow regressions for the real generate API, not TT numerics.

Place beside backend.py in the package. Only this model instance's encode/decode
are replaced; imports and process-global TT state are left untouched.
"""

import numpy as np
import pytest

from models.experimental.nllb.tt.backend import Backend


def finite_logits(token=7):
    values = np.full((1, 1, 16), -3.0, dtype=np.float32)
    values[0, 0, token] = 3.0
    return values


def controlled_backend(plan=()):
    model = object.__new__(Backend)
    model.vocab, model.pad = 16, 1
    model.config = dict(
        vocab_size=16,
        max_position_embeddings=128,
        pad_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=2,
        bos_token_id=0,
        unk_token_id=3,
    )
    observed = dict(encodes=[], prefixes=[], caches=[], pending=list(plan))

    def encode(ids, mask):
        observed["encodes"].append((ids.copy(), mask.copy()))
        return object(), mask.copy()

    def decode(prefix, encoder, valid, *, final_token_only=False, cross_kv=None):
        assert final_token_only is True
        assert isinstance(cross_kv, dict)
        observed["prefixes"].append(prefix.copy())
        observed["caches"].append(cross_kv)
        cross_kv["test-owned-reference"] = object()
        if not observed["pending"]:
            raise AssertionError("unexpected decoder step")
        result = observed["pending"].pop(0)
        if isinstance(result, Exception):
            raise result
        return result.copy()

    model.encode, model.decode = encode, decode
    return model, observed


def request():
    return np.array([[5, 2, 1]], dtype=np.int64), np.array([[1, 1, 0]], dtype=np.int64)


def unchanged(inputs, originals):
    for actual, original in zip(inputs, originals):
        assert actual.dtype == original.dtype and actual.shape == original.shape
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("cap", [1, 2])
@pytest.mark.parametrize(
    "field", ["pad_token_id", "eos_token_id", "decoder_start_token_id", "bos_token_id", "unk_token_id"]
)
def test_reserved_target_rejected_before_encode(field, cap):
    model, observed = controlled_backend([finite_logits()])
    ids, mask = request()
    originals = ids.copy(), mask.copy()
    with pytest.raises(ValueError, match="reserved"):  # allow-pytest.raises: CPU-only check.
        model.generate(ids, mask, model.config[field], cap)
    assert not observed["encodes"] and not observed["prefixes"]
    unchanged((ids, mask), originals)


@pytest.mark.parametrize("target", [-1, 16, True, np.bool_(False), 4.0, "4", None])
def test_target_type_and_range_rejected_before_encode(target):
    model, observed = controlled_backend()
    with pytest.raises(ValueError, match="target_id"):  # allow-pytest.raises: CPU-only check.
        model.generate(*request(), target, 2)
    assert not observed["encodes"] and not observed["prefixes"]


@pytest.mark.parametrize("cap", [0, 128, True, 2.0])
def test_cap_type_and_range_rejected_before_encode(cap):
    model, observed = controlled_backend()
    with pytest.raises(ValueError, match="max_new_tokens"):  # allow-pytest.raises: CPU-only check.
        model.generate(*request(), 4, cap)
    assert not observed["encodes"] and not observed["prefixes"]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf], ids=["nan", "positive_inf", "negative_inf"])
@pytest.mark.parametrize("location", ["nonwinning", "winning", "entire_vector"])
def test_nonfinite_last_logits_fail_and_release_cache(value, location):
    logits = finite_logits()
    if location == "entire_vector":
        logits.fill(value)
    else:
        logits[0, -1, 0 if location == "nonwinning" else 7] = value
    model, observed = controlled_backend([logits, finite_logits(7), finite_logits(2)])
    ids, mask = request()
    originals = ids.copy(), mask.copy()
    with pytest.raises(FloatingPointError, match="Nonfinite last-token logits"):  # allow-pytest.raises: CPU-only check.
        model.generate(ids, mask, 4, 2)
    assert len(observed["caches"]) == 1 and observed["caches"][0] == {}
    unchanged((ids, mask), originals)
    recovered = model.generate(ids, mask, 4, 4)
    np.testing.assert_array_equal(recovered, [[2, 4, 7, 2]])
    assert observed["caches"][0] is not observed["caches"][1]
    assert observed["caches"][1] is observed["caches"][2]
    assert all(cache == {} for cache in observed["caches"])
    unchanged((ids, mask), originals)


def test_decoder_exception_releases_cache_and_next_call_recovers():
    model, observed = controlled_backend([RuntimeError("injected decode failure"), finite_logits(2)])
    ids, mask = request()
    originals = ids.copy(), mask.copy()
    with pytest.raises(RuntimeError, match="injected decode failure"):  # allow-pytest.raises: CPU-only check.
        model.generate(ids, mask, 4, 2)
    assert len(observed["caches"]) == 1 and observed["caches"][0] == {}
    unchanged((ids, mask), originals)
    np.testing.assert_array_equal(model.generate(ids, mask, 4, 3), [[2, 4, 2]])
    assert observed["caches"][0] is not observed["caches"][1]
    assert all(cache == {} for cache in observed["caches"])
    unchanged((ids, mask), originals)


@pytest.mark.parametrize("mask_type", [np.int64, np.bool_])
def test_finite_cap1_returns_language_without_decoding(mask_type):
    model, observed = controlled_backend()
    ids, mask = request()
    mask = mask.astype(mask_type)
    originals = ids.copy(), mask.copy()
    output = model.generate(ids, mask, np.int64(4), np.int64(1))
    np.testing.assert_array_equal(output, [[2, 4]])
    assert output.dtype == np.int64
    assert len(observed["encodes"]) == 1 and not observed["prefixes"]
    unchanged((ids, mask), originals)


def test_finite_batch_eos_padding_cap_and_request_local_cache():
    model, observed = controlled_backend([finite_logits(2), finite_logits(7), finite_logits(8), finite_logits(9)])
    ids = np.array([[5, 2, 1], [6, 5, 2]], dtype=np.int64)
    mask = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.int64)
    originals = ids.copy(), mask.copy()
    output = model.generate(ids, mask, 4, 4)
    np.testing.assert_array_equal(output, [[2, 4, 2, 1, 1], [2, 4, 7, 8, 9]])
    assert output.dtype == np.int64
    assert len(observed["encodes"]) == 2
    assert [p.shape[1] for p in observed["prefixes"]] == [2, 2, 3, 4]
    assert observed["caches"][0] is not observed["caches"][1]
    assert all(cache is observed["caches"][1] for cache in observed["caches"][1:])
    assert all(cache == {} for cache in observed["caches"])
    unchanged((ids, mask), originals)


def test_undeclared_optional_special_ids_are_not_guessed():
    model, observed = controlled_backend()
    del model.config["bos_token_id"]
    del model.config["unk_token_id"]
    # The raw API has config, not a tokenizer language mapping. Optional IDs
    # cannot be invented; the text API validates actual language identifiers.
    np.testing.assert_array_equal(model.generate(*request(), 3, 1), [[2, 3]])
    assert not observed["prefixes"]
