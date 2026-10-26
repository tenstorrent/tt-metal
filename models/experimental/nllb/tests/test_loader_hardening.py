# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Small ordinary CPU regressions; no checkpoint downloads or TT device needed."""

import io
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import numpy as np
import torch

from models.experimental.nllb.tt.nllb_validation import load_checkpoint, validate_checkpoint, validate_config
from models.experimental.nllb.demo import translate


def fixture():
    c = dict(
        d_model=8,
        vocab_size=20,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=12,
        decoder_ffn_dim=16,
        max_position_embeddings=1024,
    )
    w = {"model.shared.weight": torch.zeros(20, 8)}
    # Independent architecture fixture: do not call checkpoint_shapes.
    for side, f in (("encoder", 12), ("decoder", 16)):
        for suffix in ("weight", "bias"):
            w[f"model.{side}.layer_norm.{suffix}"] = torch.zeros(8)
        p = f"model.{side}.layers.0"
        norms = ["self_attn_layer_norm", "final_layer_norm"]
        attns = ["self_attn"]
        if side == "decoder":
            norms.append("encoder_attn_layer_norm")
            attns.append("encoder_attn")
        for n in norms:
            for suffix in ("weight", "bias"):
                w[f"{p}.{n}.{suffix}"] = torch.zeros(8)
        for a in attns:
            for q in ("q_proj", "k_proj", "v_proj", "out_proj"):
                w[f"{p}.{a}.{q}.weight"] = torch.zeros(8, 8)
                w[f"{p}.{a}.{q}.bias"] = torch.zeros(8)
        for name, shape in (("fc1.weight", (f, 8)), ("fc1.bias", (f,)), ("fc2.weight", (8, f)), ("fc2.bias", (8,))):
            w[f"{p}.{name}"] = torch.zeros(shape)
    return c, w


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.index = self.root / "pytorch_model.bin.index.json"

    def test_duplicate_objects_rejected_before_payload(self):
        for raw in (
            '{"weight_map":{"a":"a.bin","a":"b.bin"}}',
            '{"weight_map":{},"weight_map":{"a":"a.bin"}}',
            '{"metadata":{"n":1,"n":2},"weight_map":{"a":"a.bin"}}',
            "[]",
            '{"weight_map":{}}',
            '{"weight_map":{"a":12}}',
        ):
            with self.subTest(raw=raw):
                self.index.write_text(raw)
                with patch.object(torch, "load", side_effect=AssertionError("payload reached")):
                    with self.assertRaises(ValueError):
                        load_checkpoint(self.root)

    def test_unindexed_and_nontensor_entries(self):
        for extra in (torch.ones(1), "metadata", None, 17):
            torch.save({"a": torch.zeros(1), "extra": extra}, self.root / "part.bin")
            self.index.write_text('{"weight_map":{"a":"part.bin"}}')
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                load_checkpoint(self.root)
        self.index.unlink()
        for value in ([torch.zeros(1)], {"a": None}, {3: torch.zeros(1)}):
            torch.save(value, self.root / "pytorch_model.bin")
            for path in (self.root, self.root / "pytorch_model.bin"):
                with self.subTest(value=value, path=path), self.assertRaises(ValueError):
                    load_checkpoint(path)
        torch.save({"a": torch.zeros(1)}, self.root / "part.bin")
        self.index.write_text('{"weight_map":{"a":"part.bin","missing":"part.bin"}}')
        with self.assertRaisesRegex(ValueError, "missing"):
            load_checkpoint(self.root)

    def test_resolved_alias_load_once_and_containment(self):
        torch.save({"a": torch.zeros(1), "b": torch.ones(1)}, self.root / "part.bin")
        (self.root / "alias.bin").symlink_to("part.bin")
        self.index.write_text('{"weight_map":{"a":"part.bin","b":"alias.bin"}}')
        with patch.object(torch, "load", wraps=torch.load) as calls:
            self.assertEqual(set(load_checkpoint(self.root)), {"a", "b"})
        self.assertEqual(calls.call_count, 1)
        self.assertEqual(calls.call_args.kwargs, {"map_location": "cpu", "weights_only": True})
        with tempfile.TemporaryDirectory() as external:
            outside = Path(external) / "outside.bin"
            torch.save({"a": torch.zeros(1)}, outside)
            (self.root / "escape.bin").symlink_to(outside)
            for name in ("escape.bin", str(outside), "../outside.bin"):
                self.index.write_text(json.dumps({"weight_map": {"a": name}}))
                with patch.object(torch, "load", side_effect=AssertionError("payload reached")):
                    with self.assertRaisesRegex(ValueError, "inside"):
                        load_checkpoint(self.root)

    def test_independent_architecture_aliases_markers_and_extras(self):
        c, w = fixture()
        torch.save(w, self.root / "pytorch_model.bin")
        for path in (self.root, self.root / "pytorch_model.bin"):
            validate_checkpoint(load_checkpoint(path), c)
        for name in ("lm_head.weight", "model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight"):
            validate_checkpoint(dict(w, **{name: w["model.shared.weight"]}), c)
            for bad in (torch.ones(20, 8), "metadata"):
                with self.assertRaises(ValueError):
                    validate_checkpoint(dict(w, **{name: bad}), c)
        for side in ("encoder", "decoder"):
            validate_checkpoint(dict(w, **{f"model.{side}.embed_positions._float_tensor": torch.zeros(1)}), c)
        for extra in (torch.ones(1), "metadata"):
            with self.assertRaises(ValueError):
                validate_checkpoint(dict(w, unexpected=extra), c)
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_checkpoint(w, dict(c, decoder_ffn_dim=17))
        for change in (dict(encoder_attention_heads=3), dict(d_model=True), dict(tie_word_embeddings=False)):
            with self.assertRaises(ValueError):
                validate_config(dict(c, **change))

    def test_cpu_import_and_help_without_ttnn(self):
        probe = """import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('ttnn','_ttnn') or fullname.split('.')[-1] == 'backend':
   raise AssertionError('unexpected hardware import: '+fullname)
sys.meta_path.insert(0,Block())
import importlib
importlib.import_module('models.experimental.nllb.tt.nllb_validation')
translate = importlib.import_module('models.experimental.nllb.demo.translate')
translate.main(['--help'])
"""
        root = Path(__file__).resolve().parents[4]
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=20, cwd=root)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--tokenizer-directory", r.stdout)


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tokens = self.root / "tokenizer"
        self.tokens.mkdir()
        self.c, _ = fixture()
        self.tokenizer = MagicMock(pad_token_id=1, eos_token_id=2, unk_token_id=3)
        self.vocab = {"eng_Latn": 4, "fra_Latn": 5, "<pad>": 1, "</s>": 2, "<unk>": 3}
        self.tokenizer.get_vocab.return_value = self.vocab
        self.tokenizer.return_value = {"input_ids": np.array([[4, 2]]), "attention_mask": np.array([[1, 1]])}
        self.loader = patch("transformers.AutoTokenizer.from_pretrained", return_value=self.tokenizer).start()
        self.asset_view = patch.object(
            translate, "official_tokenizer_view", side_effect=lambda directory: nullcontext(directory)
        ).start()
        self.addCleanup(patch.stopall)

    def inputs(self, **kw):
        return translate.load_text_inputs(self.root / "weights.bin", self.c, "eng_Latn", "fra_Latn", ["hello"], **kw)

    def test_explicit_directory_and_default(self):
        _, ids, mask, target = self.inputs(tokenizer_directory=self.tokens)
        self.assertEqual(self.loader.call_args.args, (str(self.tokens),))
        self.assertEqual(
            self.loader.call_args.kwargs,
            dict(use_fast=True, local_files_only=True, trust_remote_code=False, src_lang="eng_Latn"),
        )
        self.assertEqual(target, 5)
        self.assertEqual(ids.tolist(), [[4, 2]])
        self.inputs()
        self.assertEqual(self.loader.call_args.args, (str(self.root),))
        translate.load_text_inputs(self.root, self.c, "eng_Latn", "fra_Latn", ["hello"])
        self.assertEqual(self.loader.call_args.args, (str(self.root),))
        with self.assertRaisesRegex(ValueError, "directory"):
            self.inputs(tokenizer_directory=self.root / "absent")

    def test_vocab_and_language_errors(self):
        for vocab in (
            {},
            dict(self.vocab, bad=-1),
            dict(self.vocab, bad=20),
            dict(self.vocab, fra_Latn=2),
            dict(self.vocab, fra_Latn=3),
        ):
            self.tokenizer.get_vocab.return_value = vocab
            with self.subTest(vocab=vocab), self.assertRaises(ValueError):
                self.inputs()
        for lang in (None, 3, "invalid"):
            with self.assertRaises(ValueError):
                translate.load_text_inputs(self.root, self.c, lang, "fra_Latn", ["hello"])

    def test_api_and_cli_directory_routing(self):
        import ttnn

        from models.experimental.nllb.tt import backend

        model = SimpleNamespace(
            generate=MagicMock(return_value=np.array([[2, 5, 2]])),
            precision_policy={"mode": "bf16"},
            _trace_failures=[],
            _decode_trace=None,
        )
        self.tokenizer.batch_decode.return_value = ["bonjour"]
        with (
            patch.object(backend, "create_backend", return_value=model),
            patch.object(ttnn, "synchronize_device"),
            patch.object(ttnn, "is_trace_capture_active", return_value=False),
        ):
            result = translate.translate(
                self.root / "weights.bin",
                self.c,
                object(),
                "eng_Latn",
                "fra_Latn",
                ["hello"],
                max_new_tokens=3,
                tokenizer_directory=self.tokens,
            )
        self.assertEqual(result["token_ids"], [[2, 5, 2]])
        self.assertEqual(self.loader.call_args.args, (str(self.tokens),))
        config = self.root / "config.json"
        config.write_text(json.dumps(self.c))
        args = [
            "--checkpoint",
            str(self.root / "weights.bin"),
            "--config",
            str(config),
            "--tokenizer-directory",
            str(self.tokens),
            "--source-language",
            "eng_Latn",
            "--target-language",
            "fra_Latn",
            "--device",
            "0",
            "--max-new-tokens",
            "3",
            "--text",
            "hello",
        ]
        out = io.StringIO()
        with (
            patch.object(ttnn, "open_device", return_value=MagicMock()),
            patch.object(ttnn, "is_trace_capture_active", return_value=False),
            patch.object(ttnn, "close_device") as close,
            patch.object(torch, "set_num_threads"),
            patch.object(torch, "set_num_interop_threads"),
            patch.object(translate, "translate", return_value=result) as api,
        ):
            self.assertEqual(translate.main(args, result_stream=out), 0)
        self.assertEqual(api.call_args.kwargs["tokenizer_directory"], str(self.tokens))
        self.assertEqual(json.loads(out.getvalue()), result)
        close.assert_called_once()

    @unittest.skipUnless(os.environ.get("NLLB_TEST_TOKENIZER"), "optional real tokenizer directory")
    def test_real_tokenizer_directory(self):
        patch.stopall()
        directory = Path(os.environ["NLLB_TEST_TOKENIZER"])
        config = validate_config(json.loads((directory / "config.json").read_text()))
        a = translate.load_text_inputs(directory, config, "eng_Latn", "fra_Latn", ["Hello world."])
        b = translate.load_text_inputs(
            self.root / "weights.bin", config, "eng_Latn", "fra_Latn", ["Hello world."], tokenizer_directory=directory
        )
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[2], b[2])
        self.assertEqual(a[3], b[3])


if __name__ == "__main__":
    unittest.main()
