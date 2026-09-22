# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Pinned local tokenizer tests; no TT, weights, downloads or network."""

import hashlib
import subprocess
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from transformers import AutoTokenizer

if __package__:
    from . import translate
else:
    import translate

ASSETS = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model")
CACHE = Path(
    os.environ.get(
        "NLLB_TEST_TOKENIZER",
        Path.home()
        / ".cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d",
    )
)


class AssetViewUnitTests(unittest.TestCase):
    """Small synthetic manifest tests always run; they do not certify official data."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = self.root / "assets"
        self.assets.mkdir()
        files = {}
        for name in ASSETS:
            data = ("fixture-" + name).encode()
            (self.assets / name).write_bytes(data)
            files[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        self.manifest = {"models": {key: {"files": files} for key in ("600m", "1.3b-distilled", "3.3b")}}
        (self.root / "official-assets.json").write_text(json.dumps(self.manifest))
        self.patch = patch.object(translate, "__file__", str(self.root / "translate.py"))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_valid_then_missing_or_same_size_tampered_asset(self):
        with translate.official_tokenizer_view(self.assets) as view:
            self.assertEqual({p.name for p in view.iterdir()}, set(ASSETS))
        self.assertFalse(view.exists())
        path = self.assets / "tokenizer.json"
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            with translate.official_tokenizer_view(self.assets):
                self.fail("missing accepted")
        path.write_bytes(b"X" + original[1:])
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            with translate.official_tokenizer_view(self.assets):
                self.fail("changed accepted")

    def test_exception_cleanup_and_no_unreviewed_files(self):
        (self.assets / "config.json").write_text('{"tokenizer_class":"unreviewed"}')
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with translate.official_tokenizer_view(self.assets) as view:
                self.assertEqual({p.name for p in view.iterdir()}, set(ASSETS))
                raise RuntimeError("sentinel")
        self.assertFalse(view.exists())

    def test_extra_added_tokens_rejected_before_view(self):
        (self.assets / "added_tokens.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "added_tokens"):
            with translate.official_tokenizer_view(self.assets):
                self.fail("extra tokens accepted")

    def test_disagreeing_model_assets_rejected(self):
        value = json.loads((self.root / "official-assets.json").read_text())
        value["models"]["3.3b"]["files"]["tokenizer.json"]["sha256"] = "0" * 64
        (self.root / "official-assets.json").write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "share"):
            with translate.official_tokenizer_view(self.assets):
                self.fail("model disagreement accepted")


class TokenizerAssetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all((CACHE / n).is_file() for n in (*ASSETS, "config.json")):
            raise unittest.SkipTest(
                "set NLLB_TEST_TOKENIZER to four official tokenizer assets plus matching config.json"
            )
        cls.config = json.loads((CACHE / "config.json").read_text())

    def copied(self, root):
        for name in ASSETS:
            shutil.copyfile(CACHE / name, root / name)

    def inputs(self, root, texts=("Hello world.", "Bonjour à tous.")):
        return translate.load_text_inputs(root, self.config, "eng_Latn", "fra_Latn", list(texts))

    def test_actual_official_tokenization_and_decode_after_view_cleanup(self):
        direct = AutoTokenizer.from_pretrained(
            str(CACHE), use_fast=True, local_files_only=True, trust_remote_code=False, src_lang="eng_Latn"
        )
        texts = ["Hello world.", "Bonjour à tous.", "你好。", "", "مرحبا بالعالم"]
        for batch in (texts[:4], texts[4:]):
            tokenizer, ids, mask, target = self.inputs(CACHE, batch)
            expected = direct(batch, padding=True, truncation=False, return_tensors="np")
            np.testing.assert_array_equal(ids, expected["input_ids"])
            np.testing.assert_array_equal(mask, expected["attention_mask"])
            self.assertEqual(target, direct.convert_tokens_to_ids("fra_Latn"))
            self.assertEqual(
                tokenizer.batch_decode(ids, skip_special_tokens=True),
                direct.batch_decode(ids, skip_special_tokens=True),
            )
            self.assertFalse(Path(tokenizer.name_or_path).exists())

    def test_shared_three_model_pins_and_hf_symlinks_no_weight_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ASSETS:
                (root / name).symlink_to((CACHE / name).resolve())
            original = Path.open

            def guarded(path, *args, **kwargs):
                self.assertFalse(path.name.startswith("pytorch_model"))
                return original(path, *args, **kwargs)

            (root / "pytorch_model.bin").write_bytes(b"NEVER READ")
            with patch.object(Path, "open", guarded), translate.official_tokenizer_view(root) as view:
                self.assertEqual({p.name for p in view.iterdir()}, set(ASSETS))
            self.assertFalse(view.exists())

    def test_modified_token_mapping_rejected_before_constructor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.copied(root)
            path = root / "tokenizer.json"
            document = json.loads(path.read_text())
            vocab = document["model"]["vocab"]
            tokens = [token for token, index in vocab.items() if index in (94124, 15697)]
            self.assertEqual(len(tokens), 2)
            vocab[tokens[0]], vocab[tokens[1]] = vocab[tokens[1]], vocab[tokens[0]]
            path.write_text(json.dumps(document))
            with patch.object(AutoTokenizer, "from_pretrained") as loader:
                with self.assertRaisesRegex(ValueError, "tokenizer asset"):
                    self.inputs(root)
                loader.assert_not_called()

    def test_each_modified_official_asset_rejected_before_constructor(self):
        for name in ASSETS:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.copied(root)
                path = root / name
                data = path.read_bytes()
                path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
                with patch.object(AutoTokenizer, "from_pretrained") as loader:
                    with self.assertRaisesRegex(ValueError, "tokenizer asset"):
                        self.inputs(root)
                    loader.assert_not_called()

    def test_extra_added_tokens_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.copied(root)
            (root / "added_tokens.json").write_text('{"newtoken":256047}')
            with patch.object(AutoTokenizer, "from_pretrained") as loader:
                with self.assertRaisesRegex(ValueError, "added_tokens"):
                    self.inputs(root)
                loader.assert_not_called()

    def test_unreviewed_model_config_and_sidecars_not_loaded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.copied(root)
            (root / "config.json").write_text('{"tokenizer_class":"BertTokenizer","model_type":"bert"}')
            (root / "tokenizer.extra.json").write_text('{"added_tokens":[{"id":256057,"content":"wrong"}]}')
            tokenizer, ids, mask, target = self.inputs(root)
            expected = self.inputs(CACHE)
            np.testing.assert_array_equal(ids, expected[1])
            np.testing.assert_array_equal(mask, expected[2])
            self.assertEqual(target, expected[3])

    def test_loader_failure_cleans_view(self):
        paths = []

        def fail(directory, **kwargs):
            paths.append(Path(directory))
            self.assertEqual({p.name for p in paths[-1].iterdir()}, set(ASSETS))
            raise RuntimeError("constructor sentinel")

        with patch.object(AutoTokenizer, "from_pretrained", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "constructor sentinel"):
                self.inputs(CACHE)
        self.assertEqual(len(paths), 1)
        self.assertFalse(paths[0].exists())

    def test_invalid_public_inputs_still_fail_before_constructor(self):
        for src, texts in [("invalid", ["hello"]), ("eng_Latn", []), ("eng_Latn", ["a"] * 5), ("eng_Latn", [7])]:
            with patch.object(AutoTokenizer, "from_pretrained") as loader:
                with self.assertRaises(ValueError):
                    translate.load_text_inputs(CACHE, self.config, src, "fra_Latn", texts)
                loader.assert_not_called()

    def test_relocated_namespace_package_import_and_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root / "models/experimental/nllb"
            package.mkdir(parents=True)
            # Own the top-level fixture package even if inherited paths contain models/__init__.py.
            (root / "models/__init__.py").write_text("")
            for name in ("translate.py", "nllb_validation.py", "official-assets.json"):
                shutil.copyfile(Path(translate.__file__).with_name(name), package / name)
            code = (
                "import json,sys;from pathlib import Path;"
                "from models.experimental.nllb import translate,nllb_validation;"
                "expected=Path(sys.argv[2]).resolve();"
                "assert Path(translate.__file__).resolve()==expected/'translate.py';"
                "assert Path(nllb_validation.__file__).resolve()==expected/'nllb_validation.py';"
                "p=Path(sys.argv[1]);"
                "t,i,m,l=translate.load_text_inputs(p,json.loads((p/'config.json').read_text()),"
                "'eng_Latn','fra_Latn',['Hello world.']);"
                "assert i.shape==m.shape and l==256057;print('PACKAGE_OK')"
            )
            result = subprocess.run(
                [sys.executable, "-c", code, str(CACHE), str(package)],
                cwd=root,
                env=dict(
                    os.environ,
                    PYTHONPATH=os.pathsep.join(filter(None, (str(root), os.environ.get("PYTHONPATH", "")))),
                    PYTHONDONTWRITEBYTECODE="1",
                    HF_HUB_OFFLINE="1",
                ),
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("PACKAGE_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
