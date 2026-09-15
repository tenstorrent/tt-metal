# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Synthetic host tests only; never imports TTNN or opens a device."""

import ast
import copy
import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import captured_inputs as C
import captured_fullchip as D


def synthetic(heads=2, length=512):
    generator = torch.Generator().manual_seed(20260915)
    tensors = [torch.randn((1, heads, length, 128), generator=generator).bfloat16() for _ in range(3)]
    return dict(
        schema=C.SCHEMA,
        **dict(zip(("q", "k", "v"), tensors)),
        metadata=dict(
            causal=False,
            mask=None,
            scale=C.SCALE,
            provenance=dict(
                source_kind="synthetic",
                model_id="no-model",
                layer_id="test-layer",
                capture_stage="synthetic operator-boundary QKV",
                notes=[None, True, 1, 0.5],
            ),
        ),
    )


class ForbiddenObject:
    pass


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.artifact = synthetic()

    def test_valid_roundtrip_and_stable_hashes(self):
        artifact = self.artifact
        expected = C.validate_artifact(artifact)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.pt"
            torch.save(artifact, path)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            prior = torch.serialization.get_safe_globals()
            values, info = C.load_capture(path)
            self.assertEqual(prior, torch.serialization.get_safe_globals())
            self.assertEqual(info["artifact_sha256"], before)
            self.assertEqual(info["input_sha256"], expected["input_sha256"])
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
            for name, tensor in zip(("q", "k", "v"), values):
                self.assertTrue(torch.equal(tensor, artifact[name]))
            with self.assertRaises(ValueError):
                C.load_capture(path, max_file_bytes=1)

    def test_weights_only_no_fallback_and_restore_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forbidden.pt"
            value = dict(self.artifact, extra=ForbiddenObject())
            torch.save(value, path)
            previous = torch.serialization.get_safe_globals()
            with torch.serialization.safe_globals([ForbiddenObject]):
                with self.assertRaises(Exception):
                    C.load_capture(path)
                self.assertIn(ForbiddenObject, torch.serialization.get_safe_globals())
            self.assertEqual(previous, torch.serialization.get_safe_globals())
            with mock.patch.object(C.torch, "load", side_effect=RuntimeError("unsupported")) as loader:
                with self.assertRaisesRegex(RuntimeError, "unsupported"):
                    C.load_capture(path)
                self.assertEqual(loader.call_count, 1)
                self.assertEqual(loader.call_args.kwargs, dict(weights_only=True, map_location="cpu"))

    def test_atomic_path_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.pt"
            replacement = Path(directory) / "replacement.pt"
            torch.save(self.artifact, path)
            torch.save(self.artifact, replacement)
            original_load = torch.load

            def replace_after_load(*args, **kwargs):
                value = original_load(*args, **kwargs)
                replacement.replace(path)
                return value

            with mock.patch.object(C.torch, "load", side_effect=replace_after_load):
                with self.assertRaisesRegex(ValueError, "Capture path changed while loading"):
                    C.load_capture(path)

    def test_tensor_rejections(self):
        bad = [
            self.artifact["q"].float(),
            self.artifact["q"].transpose(1, 2),
            self.artifact["q"][:, :, :256, :].contiguous(),
            self.artifact["q"].repeat(2, 1, 1, 1),
            self.artifact["q"][:, :, :, :64].contiguous(),
            self.artifact["q"][:, :1].contiguous(),
            self.artifact["q"].clone().requires_grad_(),
            torch.nn.Parameter(self.artifact["q"]),
        ]
        for tensor in bad:
            with self.subTest(shape=tensor.shape, dtype=tensor.dtype), self.assertRaises(ValueError):
                C.validate_artifact(dict(self.artifact, q=tensor))
        for value in (float("inf"), float("nan"), -float("inf")):
            tensor = self.artifact["q"].clone()
            tensor[0, 0, 0, 0] = value
            with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                C.validate_artifact(dict(self.artifact, q=tensor))

    def test_no_implicit_noncontiguous_copy(self):
        tensor = torch.empty((1, 2, 512, 256), dtype=torch.bfloat16)[..., ::2]
        tensor.copy_(self.artifact["q"])
        self.assertEqual(tensor.shape, self.artifact["q"].shape)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            C.validate_artifact(dict(self.artifact, q=tensor))

    def test_metadata_and_semantic_rejections(self):
        mutations = [
            ("causal", True),
            ("causal", 0),
            ("mask", "none"),
            ("mask", []),
            ("scale", 0.125),
            ("scale", float("nan")),
            ("provenance", {}),
            ("unknown_bias", None),
        ]
        for key, value in mutations:
            meta = copy.deepcopy(self.artifact["metadata"])
            meta[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                C.validate_artifact(dict(self.artifact, metadata=meta))
        for value in (torch.tensor(0), (1, 2), {1: "value"}, ForbiddenObject(), "x" * 65536):
            meta = copy.deepcopy(self.artifact["metadata"])
            meta["provenance"]["extra"] = value
            with self.assertRaises(ValueError):
                C.validate_artifact(dict(self.artifact, metadata=meta))
        with self.assertRaises(ValueError):
            C.validate_artifact(dict(self.artifact, mask=None))

    def test_bit_hash_not_float_equality(self):
        plus = torch.zeros((1, 1, 512, 128), dtype=torch.bfloat16)
        minus = plus.clone()
        minus[0, 0, 0, 0] = -0.0
        self.assertTrue(torch.equal(plus, minus))
        self.assertNotEqual(C.tensor_sha256(plus), C.tensor_sha256(minus))
        known = torch.tensor([0.0, 1.0, -2.0], dtype=torch.bfloat16)
        self.assertEqual(C.tensor_sha256(known), hashlib.sha256(bytes.fromhex("0000803f00c0")).hexdigest())

    def test_row_selection(self):
        self.assertEqual(C.select_rows(512, 3), [0, 255, 511])
        self.assertEqual(C.select_rows(512, 1), [0])
        self.assertEqual(C.select_rows(512, 1000), list(range(512)))
        for length, count in ((0, 2), (512, 0), (512, -1), (True, 2)):
            with self.assertRaises(ValueError):
                C.select_rows(length, count)

    def test_online_reference_matches_dense(self):
        values = [self.artifact[k] for k in ("q", "k", "v")]
        rows = C.select_rows(512, 7)
        actual = C.reference(values, rows, query_batch=3, key_block=97)
        q, k, v = [t.double() for t in values]
        expected = ((q[..., rows, :] @ k.transpose(-1, -2)) / math.sqrt(128)).softmax(-1) @ v
        torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-15)

    def test_uniform_and_zero_reference_metrics(self):
        q = torch.zeros_like(self.artifact["q"])
        k = self.artifact["k"]
        v = torch.ones_like(q)
        ref = C.reference([q, k, v], [0, 511], key_block=97)
        torch.testing.assert_close(ref, torch.ones_like(ref), rtol=0, atol=0)
        metrics = C.metrics(ref, ref, v)
        self.assertEqual(metrics["l2_pct"], 0)
        self.assertIsNone(metrics["pcc"])
        self.assertIsNone(metrics["residual_l2_pct"])
        zeros = torch.zeros_like(ref)
        zmetrics = C.metrics(zeros, zeros, torch.zeros_like(v))
        self.assertIsNone(zmetrics["l2_pct"])
        self.assertEqual(zmetrics["max_relative_error_pct"], 0)
        mismatch = C.metrics(torch.ones_like(zeros), zeros, torch.zeros_like(v))
        self.assertTrue(mismatch["max_relative_error_unbounded"])
        self.assertIsNone(mismatch["max_relative_error_pct"])

    def test_metric_l2_pcc_gain(self):
        ref = self.artifact["v"][:, :, :2].double()
        values = C.metrics(2 * ref, ref, self.artifact["v"])
        self.assertAlmostEqual(values["l2_pct"], 100)
        self.assertAlmostEqual(values["pcc"], 1)
        self.assertAlmostEqual(values["gain"], 2)
        self.assertAlmostEqual(values["max_relative_error_pct"], 100)

    def test_six_configurations_and_source_pins(self):
        info = C.validate_artifact(self.artifact)
        for variant in D.VARIANTS:
            c = D.configuration(variant, info, 22, True)
            self.assertEqual(c.cores, 4)
            self.assertEqual(c.q_chunk, 256)
            self.assertEqual(c.fix_correction, variant == "fast")
            self.assertEqual(c.native_exp, variant.startswith("lofi_"))
        with self.assertRaises(ValueError):
            D.configuration("main", info, 3, True)
        D.verify_input_hashes([self.artifact[k] for k in ("q", "k", "v")], info)
        pins = D.sources()
        self.assertTrue(any(p.endswith("chain_link.hpp") for p in pins))
        self.assertTrue(any(p.endswith("captured_inputs.py") for p in pins))
        self.assertTrue(all(len(digest) == 64 for digest in pins.values()))
        for directory in ("single-core-resident-v1/main", "bf16-denom-pair-v3/candidate"):
            prefix = "experiments/sdpa-l2/" + directory + "/"
            for filename in ("compute_common.hpp", "compute_streaming.hpp", "ckernel_sfpu_sdpa.h"):
                self.assertTrue(any(p.startswith(prefix) and p.endswith("/" + filename) for p in pins))

    def test_no_eager_device_import(self):
        for path in (Path(C.__file__), Path(D.__file__)):
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, ast.Import):
                    self.assertFalse(any(a.name in ("ttnn", "fullchip") for a in node.names))


if __name__ == "__main__":
    unittest.main()
