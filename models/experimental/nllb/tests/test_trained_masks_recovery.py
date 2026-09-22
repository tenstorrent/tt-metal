# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Opt-in trained same-TT regressions; no FP32 quality or performance assertion."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from models.experimental.nllb.tests.process_runner import run_task


def mask_cases(ids, mask, pad):
    """Retain token positions; change only visibility and appended masked keys."""
    length = int(mask[0].sum())
    assert ids.shape[0] == 1 and 10 <= length <= 32
    assert np.all(mask[0, :length] == 1) and not mask[0, length:].any()
    base = np.full((2, 33), pad, dtype=np.int64)
    base[:, :length] = ids[0, :length]
    visible = np.zeros_like(base)
    visible[:, :length] = 1
    visible[0, :2] = 0  # Leading holes: tokens/positions are deliberately retained.
    visible[1, [2, 5]] = 0  # Interior holes, with visible keys on both sides.
    padded = np.pad(base, ((0, 0), (0, 32)), constant_values=pad)
    padded_mask = np.pad(visible, ((0, 0), (0, 32)))
    trailing = padded.copy()
    assert base[0, 3] != pad
    trailing[:, 33:] = base[0, 3]  # All new keys remain masked; earlier positions stay unchanged.
    return (base, visible), (padded, padded_mask), (trailing, padded_mask.copy())


def same_visible_positions(valid, expected):
    valid = np.asarray(valid)
    return (
        valid.ndim == 1
        and np.all((valid == 0) | (valid == 1))
        and np.array_equal(np.flatnonzero(valid), np.flatnonzero(expected))
    )


def checked_request(model, inputs, target, sync, calls, tensor_check, *, sentinel=None):
    """Observe actual decode before fault injection, and retain only the cache dict."""
    original = model.decode
    supplied = tuple(value.copy() for value in inputs)
    held, observed = [], []
    start = calls[0]

    def decode(*args, **kwargs):
        cache = kwargs.get("cross_kv")
        assert isinstance(cache, dict) and kwargs.get("final_token_only") is True
        valid = np.asarray(args[2])
        # Dropping/adding only a zero suffix is legal; never compact leading/interior holes.
        assert any(same_visible_positions(valid, row) for row in supplied[1]), "source mask positions changed"
        before = calls[0]
        result = original(*args, **kwargs)
        sync()  # A queued native failure must not masquerade as our Python sentinel.
        assert calls[0] > before, "decode executed no observed TT learned operations"
        assert len(cache) == model.config["decoder_layers"], "cross-KV not populated for every decoder layer"
        for key, pair in cache.items():
            assert key.endswith(".encoder_attn") and isinstance(pair, tuple) and len(pair) == 2
            assert all(tensor_check(value) for value in pair), "cache lacks actual TT tensors"
        held.append(cache)
        observed.append(True)
        if sentinel is not None:
            raise sentinel
        return result

    model.decode = decode
    try:
        try:
            result = model.generate(*supplied, target, 4)
            sync()
        except RuntimeError as error:
            if sentinel is None or error is not sentinel:
                raise
            assert observed, "sentinel raised before trained cache population"
            return None
        assert sentinel is None, "injected sentinel was swallowed"
        assert observed and calls[0] > start, "generation never executed the trained decoder"
        return np.asarray(result).copy()
    finally:
        model.decode = original
        for expected, actual in zip(inputs, supplied):
            np.testing.assert_array_equal(actual, expected, err_msg="input mutated")
        assert all(not cache for cache in held), "cross-KV remains populated on return/unwind"


def exercise(model, inputs, target, sync, calls, tensor_check, mode):
    def request(values, **kw):
        return checked_request(model, values, target, sync, calls, tensor_check, **kw)

    if mode == "recovery":
        baseline = request(inputs)
        sentinel = RuntimeError("trained-cache-injected-sentinel")
        assert request(inputs, sentinel=sentinel) is None
        np.testing.assert_array_equal(request(inputs), baseline)
    else:
        assert mode == "masks"
        base, padded, trailing = mask_cases(*inputs, model.config["pad_token_id"])
        baseline = request(base)
        np.testing.assert_array_equal(request(padded), baseline)
        np.testing.assert_array_equal(request(trailing), baseline)
        bad = (base[0].copy(), base[1].copy())
        bad[1][1] = 0
        originals = tuple(value.copy() for value in bad)
        before = calls[0]
        with pytest.raises(ValueError, match="unmasked"):  # allow-pytest.raises: CPU-only check.
            model.generate(*bad, target, 4)
        assert calls[0] == before, "invalid mixed batch performed learned TT operations"
        for expected, actual in zip(originals, bad):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(request(base), baseline)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("masks", "recovery"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--tokenizer-directory")
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--precision", choices=("bf16", "bfp8_b"), default="bf16")
    args = parser.parse_args(argv)
    import torch
    import ttnn

    from models.experimental.nllb.tt import backend
    from models.experimental.nllb.reference.envelope_regression import learned_compute_guard
    from models.experimental.nllb.tt.nllb_validation import validate_config
    from models.experimental.nllb.demo.translate import load_text_inputs

    root = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else (root if root.is_dir() else root.parent) / "config.json"
    config = validate_config(json.loads(config_path.read_text()))
    _, ids, mask, target = load_text_inputs(
        root,
        config,
        "eng_Latn",
        "fra_Latn",
        [
            "The students visited the library yesterday and borrowed several interesting books about the history of France."
        ],
        tokenizer_directory=args.tokenizer_directory,
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = ttnn.open_device(device_id=args.device)
    try:
        model = backend.create_backend(str(root), config, device, precision=args.precision)
        assert model.precision_policy["mode"] == args.precision
        with learned_compute_guard(torch, ttnn) as calls:
            exercise(
                model,
                (ids, mask),
                target,
                lambda: ttnn.synchronize_device(device),
                calls,
                lambda value: isinstance(value, ttnn.Tensor) and len(value.shape) == 4 and all(value.shape),
                args.mode,
            )
        assert calls[0] > 0
        del model
    finally:
        ttnn.close_device(device)
    print(
        json.dumps(
            dict(
                passed=True,
                mode=args.mode,
                precision=args.precision,
                tt_calls=calls[0],
                device_closed=True,
                scope="same-TT behavior; FP32 unassessed",
            )
        ),
        flush=True,
    )


@pytest.mark.parametrize("mode", ("masks", "recovery"))
def test_trained_masks_and_recovery(mode, nllb_device_id, tmp_path):
    checkpoint = os.environ.get("NLLB_TEST_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set NLLB_TEST_CHECKPOINT for trained model tests")
    precision = os.environ.get("NLLB_TEST_PRECISION", "bf16")
    assert precision in ("bf16", "bfp8_b")
    options = dict(mode=mode, checkpoint=checkpoint, device=nllb_device_id, precision=precision)
    for variable, option in (("NLLB_TEST_CONFIG", "config"), ("NLLB_TEST_TOKENIZER", "tokenizer_directory")):
        if os.environ.get(variable):
            options[option] = os.environ[variable]
    timeout = int(os.environ.get("NLLB_TEST_TIMEOUT", "280"))
    assert 30 <= timeout <= 1200, "NLLB_TEST_TIMEOUT must be between 30 and 1200 seconds"
    child = run_task("trained", options, timeout=timeout)
    assert child.returncode == 0, child.stdout + child.stderr
    print(child.stdout)


# These CPU tests exercise observation/failure semantics, not hardware execution.
class _FakeModel:
    def __init__(self, calls, defect):
        self.config = dict(decoder_layers=1)
        self.calls, self.defect = calls, defect

    def decode(self, *args, cross_kv, **kwargs):
        if self.defect != "no-ops":
            self.calls[0] += 1
        if self.defect == "early-exception":
            raise RuntimeError("not our sentinel")
        if self.defect != "empty-cache":
            cross_kv["decoder.layers.0.encoder_attn"] = (
                SimpleNamespace(tt=self.defect != "non-tensor"),
                SimpleNamespace(tt=True),
            )
        return np.zeros((1, 1, 4))

    def generate(self, ids, mask, target, cap):
        cache = {}
        try:
            try:
                valid = mask[0].copy()
                if self.defect == "mask-lost":
                    valid[:] = 0
                self.decode(None, None, valid, cross_kv=cache, final_token_only=True)
            except RuntimeError:
                if self.defect != "swallow":
                    raise
            if self.defect == "mutation":
                ids[0, 0] = 99
            return np.array([[2, target, 2]])
        finally:
            if self.defect != "leak":
                cache.clear()


@pytest.mark.parametrize(
    "defect", ("no-ops", "empty-cache", "early-exception", "swallow", "mutation", "leak", "non-tensor", "mask-lost")
)
def test_observer_cpu_rejects_false_recovery(defect):
    calls = [0]
    model = _FakeModel(calls, defect)
    original = model.decode
    sentinel = None if defect == "mutation" else RuntimeError("sentinel")
    with pytest.raises((AssertionError, RuntimeError)):  # allow-pytest.raises: CPU-only check.
        checked_request(
            model,
            (np.array([[5]]), np.ones((1, 1), dtype=np.int64)),
            4,
            lambda: None,
            calls,
            lambda value: getattr(value, "tt", False),
            sentinel=sentinel,
        )
    assert model.decode == original


def test_observer_cpu_accepts_exact_unwind_and_recovery():
    calls = [0]
    model = _FakeModel(calls, None)
    inputs = (np.array([[5]]), np.ones((1, 1), dtype=np.int64))
    exercise(model, inputs, 4, lambda: None, calls, lambda value: getattr(value, "tt", False), "recovery")
    assert calls[0] == 3


def test_mask_cases_cpu_preserves_holes_and_valid_positions():
    ids = np.arange(10, 26, dtype=np.int64)[None]
    mask = np.ones_like(ids)
    base, padded, trailing = mask_cases(ids, mask, 1)
    np.testing.assert_array_equal(base[0][:, :16], np.repeat(ids, 2, axis=0))
    assert np.flatnonzero(base[1][0]).tolist() == list(range(2, 16))
    assert np.flatnonzero(base[1][1]).tolist() == [i for i in range(16) if i not in (2, 5)]
    for changed in (padded, trailing):
        np.testing.assert_array_equal(changed[0][:, :33], base[0])
        np.testing.assert_array_equal(changed[1][:, :33], base[1])
        assert not changed[1][:, 33:].any()
    assert np.all(padded[0][:, 33:] == 1) and np.all(trailing[0][:, 33:] != 1)


def test_visible_positions_cpu_allows_only_zero_suffix_changes():
    expected = np.zeros(65, dtype=np.int64)
    expected[[2, 7, 31]] = 1
    assert same_visible_positions(expected[:32], expected)
    assert same_visible_positions(np.pad(expected, (0, 31)), expected)
    assert not same_visible_positions(expected[:31], expected)
    assert not same_visible_positions(np.ones(3), expected)
    changed = expected.copy()
    changed[64] = 1
    assert not same_visible_positions(changed, expected)


if __name__ == "__main__":
    main()
