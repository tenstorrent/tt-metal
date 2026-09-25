# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU contract for the standard LTX encoder prompt cache; no TTNN import.

Run directly with Python to avoid device-aware repository conftest imports.
Execute the production methods with filesystem fixtures and unloaded/loaded shells.
This checks cache decisions, not encoder numerics or arbitrary custom model overrides.
"""

from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
import tempfile
import unittest
from collections import namedtuple
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
ENCODER = ROOT / "encoders" / "gemma" / "encoder_pair.py"
PIPELINE = ROOT / "pipelines" / "ltx" / "pipeline_ltx.py"


def _load(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    namespace["__file__"] = str(path)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class EmbeddingCacheIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.sources = self.directory / "gemma"
        self.sources.mkdir()
        self.checkpoint = self.directory / "ltx.safetensors"
        self.binary = self.directory / "_ttnn.so"
        for path in (
            self.checkpoint,
            self.binary,
            self.sources / "model-00001.safetensors",
            self.sources / "tokenizer.json",
        ):
            path.write_bytes(b"original fixture")
        env = patch.dict(os.environ, {"TT_DIT_CACHE_DIR": str(self.directory / "cache")})
        env.start()
        self.addCleanup(env.stop)
        cache_namespace = _load(ROOT / "utils" / "cache.py", {"source_id"}, {"Path": Path})
        self.cache = SimpleNamespace(source_id=cache_namespace["source_id"], CACHE_VERSION=1)
        namespace = {
            "Path": Path,
            "glob": glob,
            "hashlib": hashlib,
            "os": os,
            "cache_module": self.cache,
            "ttnn": SimpleNamespace(_ttnn=SimpleNamespace(__file__=str(self.binary))),
        }
        self.identity = _load(ENCODER, {"_gemma_shards", "embedding_cache_identity"}, namespace)[
            "embedding_cache_identity"
        ]
        self.path_method = _load(PIPELINE, {"_device_embed_cache_path"}, {"os": os, "json": json, "hashlib": hashlib})[
            "_device_embed_cache_path"
        ]

    def pair(self, tp=4):
        factor = namedtuple("ParallelFactor", "factor mesh_axis")
        config = namedtuple("EncoderParallelConfig", "tensor_parallel sequence_parallel")
        pair = SimpleNamespace(
            checkpoint_name=str(self.checkpoint),
            gemma_path=str(self.sources),
            mode="av",
            _num_layers=2,
            _hidden_layer_index=-1,
            _sequence_length=1024,
            _video_dim=4096,
            _audio_dim=2048,
            gemma_encoder=None,
            video_connector=None,
            audio_connector=None,
            mesh_device=SimpleNamespace(shape=(2, tp), arch=lambda: "wormhole_b0"),
            parallel_config=config(factor(tp, 1), None),
            ccl_manager=SimpleNamespace(topology="Linear", num_links=2),
        )
        pair.embedding_cache_identity = MethodType(self.identity, pair)
        return pair

    def load_shells(self, pair):
        native = os.environ["LTX_GEMMA_NATIVE_GQA"] == "1"
        stats = os.environ["LTX_CONNECTOR_QK_STATS"] == "1" and pair.parallel_config.tensor_parallel.factor > 1
        pair.gemma_encoder = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=SimpleNamespace(_native_gqa=native)) for _ in range(pair._num_layers)]
        )
        for axis in ("video", "audio"):
            setattr(
                pair,
                f"{axis}_connector",
                SimpleNamespace(
                    transformer_1d_blocks=[SimpleNamespace(_qk_stats=stats) for _ in range(8)],
                    num_learnable_registers=128,
                ),
            )

    def test_standard_cold_and_loaded_identity_match_for_mesh_and_policy(self):
        for tp in (1, 4, 8):
            for native in ("0", "1"):
                for stats in ("0", "1"):
                    with self.subTest(tp=tp, native=native, stats=stats), patch.dict(
                        os.environ, {"LTX_GEMMA_NATIVE_GQA": native, "LTX_CONNECTOR_QK_STATS": stats}
                    ):
                        pair = self.pair(tp)
                        cold = pair.embedding_cache_identity()
                        self.load_shells(pair)
                        self.assertEqual(cold, pair.embedding_cache_identity())
                        # Module constructors consume flags once. An environment
                        # change must not relabel existing resident/evicted shells.
                        os.environ["LTX_GEMMA_NATIVE_GQA"] = str(1 - int(native))
                        os.environ["LTX_CONNECTOR_QK_STATS"] = str(1 - int(stats))
                        self.assertEqual(cold, pair.embedding_cache_identity())
                        self.assertNotEqual(cold, self.pair(tp).embedding_cache_identity())

    def test_independent_native_and_connector_policies_get_fresh_keys(self):
        identities = []
        for native, stats in (("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")):
            with patch.dict(os.environ, {"LTX_GEMMA_NATIVE_GQA": native, "LTX_CONNECTOR_QK_STATS": stats}):
                identities.append(json.dumps(self.pair().embedding_cache_identity(), sort_keys=True))
        self.assertEqual(len(set(identities)), 4)

    def test_checkpoint_tokenizer_and_binding_replacements_invalidate(self):
        pair = self.pair()
        for path in (
            self.checkpoint,
            self.sources / "model-00001.safetensors",
            self.sources / "tokenizer.json",
            self.binary,
        ):
            with self.subTest(path=path):
                before = pair.embedding_cache_identity()
                path.write_bytes(b"changed fixture with different size")
                self.assertNotEqual(before, pair.embedding_cache_identity())

    def test_prompt_list_boundaries_and_legacy_namespace_are_preserved(self):
        legacy = self.directory / "cache" / "ltx-embeddings" / "old.device.pt"
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"legacy entry remains untouched")
        pipe = SimpleNamespace(gemma_encoder_pair=self.pair())
        first = self.path_method(pipe, ["a||b", "c"])
        second = self.path_method(pipe, ["a", "b||c"])
        self.assertNotEqual(first, second)
        self.assertEqual(first, self.path_method(pipe, ["a||b", "c"]))
        self.assertEqual(Path(first).parent.name, "ltx-embeddings-v2")
        self.assertFalse(Path(first).exists())
        self.assertEqual(legacy.read_bytes(), b"legacy entry remains untouched")

    def test_forced_encode_does_not_require_or_write_a_cache_key(self):
        calls = []
        result = object()
        pipe = SimpleNamespace(
            _device_embed_cache_path=lambda prompts: self.fail("forced encode consulted cache"),
            gemma_encoder_pair=SimpleNamespace(encode=lambda prompts: calls.append(prompts) or result),
        )
        method = _load(PIPELINE, {"encode_prompts"}, {"Watchdog": lambda _: nullcontext(), "os": os})["encode_prompts"]
        self.assertIs(method(pipe, ["prompt"], use_cache=False), result)
        self.assertEqual(calls, [["prompt"]])


if __name__ == "__main__":
    unittest.main()
