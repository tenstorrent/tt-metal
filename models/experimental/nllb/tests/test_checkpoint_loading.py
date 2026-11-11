# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Imported-code CPU contract tests. Optional NLLB_TEST_CHECKPOINT/NLLB_TEST_CONFIG."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from models.experimental.nllb.tt import backend
from models.experimental.nllb.tt.nllb_validation import validate_config, validate_checkpoint, checkpoint_shapes


def fixture_config():
    return dict(
        d_model=32,
        vocab_size=64,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_position_embeddings=1024,
        pad_token_id=1,
    )


class CheckpointTests(unittest.TestCase):
    def test_file_directory_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = torch.arange(3), torch.arange(4)
            torch.save({"a": a}, root / "pytorch_model.bin")
            self.assertTrue(torch.equal(backend.load_checkpoint(root)["a"], a))
            self.assertTrue(torch.equal(backend.load_checkpoint(root / "pytorch_model.bin")["a"], a))
            torch.save({"a": a, "c": a + 1}, root / "part1.bin")
            torch.save({"b": b}, root / "part2.bin")
            index = root / "pytorch_model.bin.index.json"
            index.write_text(json.dumps({"weight_map": {"a": "part1.bin", "c": "part1.bin", "b": "part2.bin"}}))
            real_load = torch.load
            with patch.object(torch, "load", wraps=real_load) as calls:
                weights = backend.load_checkpoint(root)
            self.assertEqual(calls.call_count, 2)
            for call in calls.call_args_list:
                self.assertEqual(call.kwargs, {"map_location": "cpu", "weights_only": True})
            self.assertTrue(torch.equal(weights["b"], b))
            for mapping in ({}, {"missing": "part1.bin"}, {"a": "../outside.bin"}):
                index.write_text(json.dumps({"weight_map": mapping}))
                with self.assertRaises(ValueError):
                    backend.load_checkpoint(root)

    def test_checkpoint_shapes_and_aliases(self):
        config = fixture_config()
        weights = {key: torch.zeros(shape) for key, shape in checkpoint_shapes(config).items()}
        validate_checkpoint(weights, config)
        weights["lm_head.weight"] = weights["model.shared.weight"].clone()
        weights["lm_head.weight"][0, 0] = 1
        with self.assertRaisesRegex(ValueError, "Tied"):
            validate_checkpoint(weights, config)
        del weights["lm_head.weight"]
        weights["model.encoder.layers.0.fc1.weight"] = torch.zeros(32, 64)
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_checkpoint(weights, config)

    def test_config_rejects_incompatible(self):
        for change in (
            dict(d_model=31),
            dict(encoder_layers=0),
            dict(pad_token_id=0),
            dict(activation_function="gelu"),
            dict(tie_word_embeddings=False),
            dict(decoder_start_token_id=0),
            dict(d_model=True),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config(dict(fixture_config(), **change))

    def test_adjacent_config_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = fixture_config()
            Path(tmp, "config.json").write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError, "config mismatch"):
                backend.create_backend(tmp, dict(c, scale_embedding=False), object())

    @unittest.skipUnless(os.environ.get("NLLB_TEST_CHECKPOINT"), "optional checkpoint fixture")
    def test_configurable_real_checkpoint(self):
        path = Path(os.environ["NLLB_TEST_CHECKPOINT"])
        config_path = Path(
            os.environ.get("NLLB_TEST_CONFIG", str((path if path.is_dir() else path.parent) / "config.json"))
        )
        config = validate_config(json.loads(config_path.read_text()))
        validate_checkpoint(backend.load_checkpoint(path), config)


class RequestTests(unittest.TestCase):
    def setUp(self):
        # Import the real public methods; invalid requests must fail before any TT operation.
        self.model = backend.Backend.__new__(backend.Backend)
        self.model.config = validate_config(fixture_config())
        self.model.vocab = self.model.config["vocab_size"]
        self.ids = np.array([[4, 2]], dtype=np.int64)
        self.mask = np.ones_like(self.ids)

    def test_rejects_bad_requests(self):
        for ids, mask in (
            (self.ids.astype(float), self.mask),
            (np.array([[64, 2]]), self.mask),
            (self.ids, np.array([[1, 2]])),
            (self.ids, np.zeros_like(self.mask)),
            (np.zeros((5, 2), dtype=int), np.ones((5, 2), dtype=int)),
            (np.zeros((1, 257), dtype=int), np.ones((1, 257), dtype=int)),
        ):
            with self.subTest(shape=ids.shape), self.assertRaises(ValueError):
                self.model.generate(ids, mask, 3, 4)
        for target, cap in ((64, 4), (3, 0), (3, 257), (3, 1.5), (True, 4)):
            with self.subTest(target=target, cap=cap), self.assertRaises(ValueError):
                self.model.generate(self.ids, self.mask, target, cap)
        with self.assertRaisesRegex(ValueError, "batch"):
            self.model.forward(self.ids, self.mask, np.array([[2], [2]]))

    def test_unsupported_precision(self):
        for mode in ("fp16", "fp32", "fp8"):
            with self.assertRaisesRegex(ValueError, "Supported precisions"):
                backend.create_backend("unused", fixture_config(), None, precision=mode)


if __name__ == "__main__":
    unittest.main()
