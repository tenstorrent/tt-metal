# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""CPU-only defect models validate the checks, not trained model acceptance."""

import json
import time

import numpy as np
import pytest

if __package__:
    from . import envelope_regression as helper
else:
    import envelope_regression as helper

CONFIG = dict(vocab_size=512, decoder_start_token_id=2, eos_token_id=2, pad_token_id=1)


class Tokenizer:
    def convert_tokens_to_ids(self, language):
        return {"eng_Latn": 10, "fra_Latn": 11}[language]

    def encode(self, text, add_special_tokens):
        index = helper.PARAGRAPHS.index(text)
        return [10] + [20 + index] * ([300, 310, 6, 0][index]) + [2]


class Model:
    config = CONFIG

    def __init__(self, defect=None):
        self.defect = defect

    def decode(self, prefix, encoded, valid, final_token_only=False, cross_kv=None):
        cross_kv["real-cache-placeholder"] = 1
        if self.defect == "prefix_mutation":
            prefix[0, 0] = 4
        token = 30 + encoded
        if encoded >= 22 or self.defect == "early_eos":
            token = 2
        logits = np.zeros((1, 1, CONFIG["vocab_size"]), dtype=np.float32)
        logits[0, 0, token] = 1
        return logits

    def generate(self, ids, mask, target, cap):
        if self.defect == "source_mutation":
            ids[0, 0] = 4
        rows = []
        for index in range(len(ids)):
            length = int(mask[index].sum())
            encoded = int(ids[index, 1]) if length > 2 else 23
            if self.defect == "padding_dependent" and len(ids) == 1 and length == 255 and ids.shape[1] == 256:
                encoded = 24
            tokens, cache = [2, target], {}
            try:
                for _ in range(1, cap):
                    logits = self.decode(
                        np.array([tokens], dtype=np.int64), encoded, length, final_token_only=True, cross_kv=cache
                    )
                    tokens.append(int(logits[0, 0].argmax()))
                    if tokens[-1] == 2:
                        break
            finally:
                if self.defect != "cache_leak":
                    cache.clear()
            rows.append(tokens)
        if self.defect == "reverse_rows" and len(rows) == 4:
            rows.reverse()
        out = np.full((len(rows), max(map(len, rows))), 1, dtype=np.int64)
        for index, row in enumerate(rows):
            out[index, : len(row)] = row
        return out


def exercise(defect=None):
    inputs, meta = helper.build_inputs(Tokenizer(), CONFIG)
    return helper.exercise(Model(defect), inputs, meta, lambda: None, time.monotonic() + 10)


def test_real_source_lengths_and_special_tokens():
    inputs, meta = helper.build_inputs(Tokenizer(), CONFIG)
    assert meta["source_lengths"] == [255, 256, 8, 2]
    assert inputs["attention_mask"].sum(1).tolist() == [255, 256, 8, 2]
    assert inputs["input_ids"][0, 254] == inputs["input_ids"][1, 255] == 2

    class ShortTokenizer(Tokenizer):
        def encode(self, text, add_special_tokens):
            return [10, 20, 2]

    with pytest.raises(ValueError, match="too short"):
        helper.build_inputs(ShortTokenizer(), CONFIG)


def test_complete_observed_envelope_and_no_oracle_claim():
    outputs, result = exercise()
    assert len(outputs) == 8 and result["same_tt_passed"]
    assert result["observations"]["base__cap64"]["observed_prefixes"][0][-2:] == [63, 64]
    assert "fp32" not in result


@pytest.mark.parametrize(
    "defect,message",
    [
        ("source_mutation", "source inputs"),
        ("prefix_mutation", "prefix mutated"),
        ("cache_leak", "cache cleanup"),
        ("reverse_rows", "decode events"),
    ],
)
def test_defective_execution_fails_and_restores_decode(defect, message):
    model = Model(defect)
    inputs, meta = helper.build_inputs(Tokenizer(), CONFIG)
    with pytest.raises(AssertionError, match=message):
        helper.observe_request(
            model, inputs["input_ids"], inputs["attention_mask"], 11, 64, lambda: None, time.monotonic() + 10
        )
    assert "decode" not in model.__dict__


def test_early_eos_cannot_claim_observed_boundary():
    _, result = exercise("early_eos")
    assert result["checks"]["mixed_natural_eos"]
    assert not result["checks"]["actual_boundary_63_64"] and not result["same_tt_passed"]


def test_padding_dependent_defect_is_detected_independently():
    _, result = exercise("padding_dependent")
    assert result["checks"]["actual_boundary_63_64"]
    assert not result["checks"]["right_padding"] and not result["same_tt_passed"]


@pytest.mark.parametrize("row", [[2, 11, 50], [2, 11, 1, 2], [2, 11, 2, 50], [2, 12, 2]])
def test_cap_eos_padding_and_language_semantics(row):
    with pytest.raises(AssertionError):
        helper.canonical(np.array([row], dtype=np.int64), 1, 11, 64, CONFIG)


def test_missing_decode_event_cannot_claim_long_output():
    model = Model()

    def fabricate(ids, mask, target, cap):
        return np.array([[2, target] + [50] * (cap - 1)] * len(ids), dtype=np.int64)

    model.generate = fabricate
    inputs, meta = helper.build_inputs(Tokenizer(), CONFIG)
    with pytest.raises(AssertionError, match="decode events"):
        helper.observe_request(
            model, inputs["input_ids"], inputs["attention_mask"], 11, 64, lambda: None, time.monotonic() + 10
        )


def test_pinned_oracle_separates_behavior_and_fp32_disagreement(tmp_path):
    outputs, result = exercise()
    identity = {
        "config_sha256": "config",
        "weight_sha256": {"pytorch_model.bin": "weights"},
        "input_hashes": {"input_ids": "inputs"},
    }
    fixture = dict(
        schema="nllb-portable-envelope-fp32-v1",
        identity=identity,
        precision="fp32",
        tf32=False,
        outputs={k: v.tolist() for k, v in outputs.items()},
    )
    path = tmp_path / "fp32.json"
    path.write_text(json.dumps(fixture))
    result_fp32 = helper.compare_oracle(path, helper.file_hash(path), identity, outputs, 11, CONFIG)
    assert result_fp32["assessed"] and result_fp32["exact"]
    fixture["outputs"]["base__cap64"][0][10] = 51
    path.write_text(json.dumps(fixture))
    assert not helper.compare_oracle(path, helper.file_hash(path), identity, outputs, 11, CONFIG)["exact"]
    assert result["same_tt_passed"]
    with pytest.raises(ValueError, match="identity"):
        helper.compare_oracle(path, helper.file_hash(path), dict(identity, config_sha256="other"), outputs, 11, CONFIG)
    with pytest.raises(ValueError, match="SHA256"):
        helper.compare_oracle(path, "0" * 64, identity, outputs, 11, CONFIG)


@pytest.mark.parametrize("defect", ["language", "eos", "pad", "vocabulary"])
def test_bad_tokenizer_rejected(defect):
    class BrokenTokenizer(Tokenizer):
        def encode(self, text, add_special_tokens):
            result = super().encode(text, add_special_tokens)
            if defect == "language":
                result[0] = 12
            elif defect == "eos":
                result[-1] = 20
            elif defect == "pad":
                result[1] = 1
            elif defect == "vocabulary":
                result[1] = 512
            return result

    with pytest.raises(ValueError):
        helper.build_inputs(BrokenTokenizer(), CONFIG)


def test_deadline_fails_without_losing_wrapper_cleanup():
    inputs, meta = helper.build_inputs(Tokenizer(), CONFIG)
    model = Model()
    with pytest.raises(TimeoutError):
        helper.observe_request(model, inputs["input_ids"], inputs["attention_mask"], 11, 64, lambda: None, 0)
    assert "decode" not in model.__dict__


def test_checkpoint_payload_hashes_include_all_indexed_shards(tmp_path):
    import hashlib

    (tmp_path / "one.bin").write_bytes(b"first")
    (tmp_path / "two.bin").write_bytes(b"second")
    index = tmp_path / "pytorch_model.bin.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "one.bin", "b": "two.bin"}}))
    assert helper.weight_hashes(tmp_path) == {
        "one.bin": hashlib.sha256(b"first").hexdigest(),
        "two.bin": hashlib.sha256(b"second").hexdigest(),
    }
    index.write_text(json.dumps({"weight_map": {"a": "../outside.bin"}}))
    with pytest.raises(ValueError, match="filenames"):
        helper.weight_hashes(tmp_path)


@pytest.mark.parametrize("precision", ["bf16", "bfp8_b"])
def test_precision_parser_and_same_fp32_oracle(precision, tmp_path):
    arguments = helper.argument_parser().parse_args(
        ["--checkpoint", "weights", "--device", "0", "--output", "report", "--precision", precision]
    )
    assert arguments.precision == precision and arguments.timeout == 280
    outputs, _ = exercise()
    fixture = dict(
        schema="nllb-portable-envelope-fp32-v1",
        identity={},
        precision="fp32",
        tf32=False,
        outputs={k: v.tolist() for k, v in outputs.items()},
    )
    path = tmp_path / "fp32.json"
    path.write_text(json.dumps(fixture))
    compared = helper.compare_oracle(path, helper.file_hash(path), {}, outputs, 11, CONFIG, precision)
    assert (
        compared["exact"] and compared["candidate_precision"] == precision and compared["reference_precision"] == "fp32"
    )


def test_unsupported_precision_is_explicitly_rejected():
    with pytest.raises(SystemExit) as error:
        helper.argument_parser().parse_args(
            ["--checkpoint", "weights", "--device", "0", "--output", "report", "--precision", "fp8"]
        )
    assert error.value.code == 2


def test_finalize_closes_after_first_write_error_and_never_marks_passed(tmp_path):
    report, writes, closed = {}, [], []

    def writer(path, value):
        writes.append(dict(value))
        if len(writes) == 1:
            raise OSError("disk error")
        helper.write_report(path, value)

    path = tmp_path / "report.json"
    helper.finalize(report, path, "device", closed.append, True, writer)
    assert closed == ["device"] and all(not value["passed"] for value in writes)
    assert report["device_closed"] and not report["passed"] and "write_error" in report
    assert not json.loads(path.read_text())["passed"]


def test_finalize_close_failure_leaves_failed_receipt(tmp_path):
    path, report, states = tmp_path / "report.json", {}, []

    def close(device):
        states.append(json.loads(path.read_text()))
        raise RuntimeError("close failed")

    helper.finalize(report, path, "device", close, True)
    assert not states[0]["passed"] and not states[0]["device_closed"]
    assert not report["passed"] and report["close_error"] == "close failed"
    assert not json.loads(path.read_text())["passed"]


def test_final_atomic_write_failure_preserves_failed_receipt(tmp_path, monkeypatch):
    from pathlib import Path

    path, report, closed = tmp_path / "report.json", {}, []
    original = Path.replace
    calls = []

    def replace(source, target):
        calls.append(target)
        if len(calls) == 2:
            raise OSError("final replace failed")
        return original(source, target)

    monkeypatch.setattr(Path, "replace", replace)
    helper.finalize(report, path, "device", closed.append, True)
    assert closed == ["device"] and not report["passed"]
    assert not json.loads(path.read_text())["passed"]


def test_success_receipt_only_after_close(tmp_path):
    path, report = tmp_path / "report.json", {}

    def close(device):
        assert not json.loads(path.read_text())["passed"]

    helper.finalize(report, path, "device", close, True)
    assert report["passed"] and json.loads(path.read_text())["device_closed"]


def test_shard_symlinks_cannot_escape_checkpoint(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    payload = tmp_path / "external.bin"
    payload.write_bytes(b"weights")
    (root / "shard.bin").symlink_to(payload)
    (root / "pytorch_model.bin.index.json").write_text(json.dumps({"weight_map": {"w": "shard.bin"}}))
    with pytest.raises(ValueError, match="outside"):
        helper.weight_hashes(root)
    # A caller-explicit single payload is allowed, including a symlink to it.
    assert helper.weight_hashes(root / "shard.bin") == {"external.bin": helper.file_hash(payload)}
    (root / "shard.bin").unlink()
    (root / "inside.bin").write_bytes(b"weights")
    (root / "shard.bin").symlink_to(root / "inside.bin")
    assert helper.weight_hashes(root) == {"shard.bin": helper.file_hash(root / "inside.bin")}


def test_private_scaled_attention_dispatch_is_blocked_and_wrapper_restored():
    import torch
    from types import SimpleNamespace

    original = lambda *args: None
    ttnn = SimpleNamespace(matmul=original)
    q = torch.zeros((1, 1, 2, 4))
    with pytest.raises(AssertionError, match="aten._scaled_dot_product"):
        with helper.learned_compute_guard(torch, ttnn):
            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(q, q, q)
    assert ttnn.matmul is original


@pytest.mark.parametrize("invocation", ["module", "file"])
@pytest.mark.parametrize("dependency", ["backend", "nllb_validation"])
def test_entrypoint_resolves_sibling_backend_in_package_and_file_modes(tmp_path, invocation, dependency):
    import os
    from pathlib import Path
    import subprocess
    import sys

    package = tmp_path / "portable_nllb"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "envelope_regression.py").write_bytes(Path(helper.__file__).read_bytes())
    (package / "backend.py").write_text("")
    (package / (dependency + ".py")).write_text("raise RuntimeError('EXPECTED_SIBLING_IMPORT')\n")
    # Dependency stubs prove import routing without loading/opening TT hardware.
    (tmp_path / "torch.py").write_text("")
    (tmp_path / "ttnn.py").write_text("")
    (tmp_path / "transformers.py").write_text("AutoTokenizer = None\n")
    mode = (
        ["-m", "portable_nllb.envelope_regression"]
        if invocation == "module"
        else [str(package / "envelope_regression.py")]
    )
    child = subprocess.run(
        [sys.executable, *mode, "--checkpoint", "unused", "--device", "0", "--output", "unused"],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, (str(tmp_path), os.environ.get("PYTHONPATH"))))),
        capture_output=True,
        text=True,
    )
    assert child.returncode != 0 and "EXPECTED_SIBLING_IMPORT" in child.stderr
