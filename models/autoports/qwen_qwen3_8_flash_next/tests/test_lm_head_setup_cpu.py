# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Production model constructor/logits AST tests, without TTNN or devices."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


class HeadSetupTest(unittest.TestCase):
    def build(self, tied):
        source = Path(__file__).resolve().parents[1] / "tt/model.py"
        owner = next(
            n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "QwenModel"
        )
        methods = [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in ("__init__", "logits")]
        self.reads, self.uploads, self.groups = [], [], []
        embedding = torch.arange(64 * 32).reshape(64, 32).float()
        head = -embedding
        config = SimpleNamespace(num_hidden_layers=48, ple_layer_ids=[], tie_word_embeddings=tied)
        compute = object()
        test = self

        class Source:
            def __init__(self, checkpoint):
                pass

            def tensor(self, key):
                test.reads.append(key)
                return embedding if key.endswith("embed_tokens.weight") else head

            def read(self, prefix, strip):
                return {}

        class Loader:
            def __init__(self, weights, mesh):
                self.compute = compute

            def shards(self, parts, **kw):
                test.uploads.append((parts, kw))
                return object()

        class Group:
            def __init__(self, loader, parts, policies):
                self.parts, self.policies = parts, policies
                # Mirror production deduplication for identical phase policies.
                self.decode = self.rows = SimpleNamespace(compute=object())
                test.groups.append(self)

        class Projection:
            def __init__(self, group, native_forward):
                test.assertIsNone(native_forward)
                self.group = group

            def __call__(self, x):
                return ("prepared_head", x)

        ns = {
            "Checkpoint": Source,
            "AutoConfig": SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace(text_config=config)),
            "MeshWorkspace": lambda mesh: object(),
            "Collectives": lambda mesh: object(),
            "MeshLoader": Loader,
            "FinalMixer": lambda *a: lambda x: ("final_mixer", x),
            "DramProjectionConfig": lambda **kw: SimpleNamespace(**kw),
            "PhaseDramProjectionGroup": Group,
            "PhaseDramProjection": Projection,
            "ttnn": SimpleNamespace(ROW_MAJOR_LAYOUT="ROW_MAJOR"),
        }
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), ns)
        cls = type("ActualModelHeadSetup", (), {name: ns[name] for name in ("__init__", "logits")})
        model = cls("checkpoint", SimpleNamespace(shape=(1, 4)), packed_cache="unused", layer_indices=())
        return model, embedding if tied else head, compute

    def check_head(self, tied):
        model, expected, compute = self.build(tied)
        self.assertEqual(len(self.uploads), 1, "Only token embedding uses the native shard uploader")
        self.assertEqual(self.uploads[0][1], {"dim": 1, "layout": "ROW_MAJOR"})
        self.assertEqual(len(self.groups), 1)
        group = self.groups[0]
        self.assertTrue(torch.equal(torch.cat([parts[0] for parts in group.parts]), expected))
        self.assertIs(group.policies["decode"], group.policies["rows"])
        self.assertEqual(
            vars(group.policies["decode"]),
            dict(
                cores=8,
                readers=3,
                block=1,
                dtype="bfloat16",
                fidelity="HiFi4",
                packed=False,
                activation_dtype="bfloat16",
            ),
        )
        self.assertIs(group.decode.compute, compute)
        self.assertEqual(model.logits("residual"), ("prepared_head", ("final_mixer", "residual")))
        self.assertEqual("lm_head.weight" in self.reads, not tied)

    def test_untied_real_head_is_prepared_once_without_native_tt_duplicate(self):
        self.check_head(False)

    def test_tied_head_reuses_hf_embedding_partition_without_native_tt_duplicate(self):
        self.check_head(True)


if __name__ == "__main__":
    unittest.main()
