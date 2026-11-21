# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the H3 vision tower loads through the content-keyed converted-weight cache."""

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
LOADER = "encoders/qwen3vl/loader_minimax_h3.py"
NAMES = {"_vision_shards", "minimax_h3_vision_sources", "minimax_h3_vision_config", "build_minimax_h3_vision_tower"}


class Recorder:
    def __init__(self):
        self.calls = []

    def load_model(self, module, **kwargs):
        self.calls.append(kwargs)


class Tower:
    def __init__(self, **kwargs):
        self.direct = None

    def load_torch_state_dict(self, state):
        self.direct = state


def loader(recorder):
    tree = ast.parse((ROOT / LOADER).read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    ns = {
        "json": json,
        "os": os,
        "Path": Path,
        "cache": recorder,
        "Qwen3VlVisionModel": Tower,
        "_VISION_PREFIX": "model.visual.",
        "load_minimax_h3_vision_state_dict": lambda d: {"state_of": str(d)},
    }
    exec(compile("from __future__ import annotations\n" + ast.unparse(ast.Module(selected, [])), LOADER, "exec"), ns)
    return ns


def checkpoint(d):
    vision = {"depth": 1, "hidden_size": 8, "num_heads": 1, "intermediate_size": 8, "patch_size": 16}
    vision |= {"spatial_merge_size": 2, "num_position_embeddings": 4, "out_hidden_size": 8}
    vision |= {"deepstack_visual_indexes": []}
    (d / "config.json").write_text(json.dumps({"vision_config": vision}))
    weight_map = {
        "model.visual.blocks.0.attn.qkv.weight": "model-00009-of-00014.safetensors",
        "model.visual.pos_embed.weight": "model-00009-of-00014.safetensors",
        "model.language_model.layers.0.mlp.up_proj.weight": "model-00001-of-00014.safetensors",
    }
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


class H3VisionTowerCache(unittest.TestCase):
    def test_sharded_tower_loads_through_the_cache_keyed_on_its_shards(self):
        rec = Recorder()
        ns = loader(rec)
        with tempfile.TemporaryDirectory() as d:
            checkpoint(Path(d))
            tower, _ = ns["build_minimax_h3_vision_tower"](
                d, mesh_device=SimpleNamespace(shape=(4, 8)), parallel_config="tp4_sp8"
            )
            self.assertIsNone(tower.direct)
            (call,) = rec.calls
            self.assertEqual(call["model_name"], "minimax-h3")
            self.assertEqual(call["subfolder"], "vision_tower")
            self.assertEqual(call["parallel_config"], "tp4_sp8")
            self.assertEqual(call["mesh_shape"], (4, 8))
            self.assertEqual(call["sources"], [Path(d) / "model-00009-of-00014.safetensors"])
            self.assertEqual(call["get_torch_state_dict"](), {"state_of": d})

    def test_replicated_tower_loads_directly(self):
        rec = Recorder()
        ns = loader(rec)
        with tempfile.TemporaryDirectory() as d:
            checkpoint(Path(d))
            tower, _ = ns["build_minimax_h3_vision_tower"](d, mesh_device=SimpleNamespace(shape=(1, 1)))
            self.assertEqual(tower.direct, {"state_of": d})
            self.assertEqual(rec.calls, [])

    def test_unloaded_build_touches_no_weights(self):
        rec = Recorder()
        ns = loader(rec)
        with tempfile.TemporaryDirectory() as d:
            checkpoint(Path(d))
            tower, _ = ns["build_minimax_h3_vision_tower"](
                d, mesh_device=SimpleNamespace(shape=(4, 8)), parallel_config="p", load_weights=False
            )
            self.assertIsNone(tower.direct)
            self.assertEqual(rec.calls, [])


if __name__ == "__main__":
    unittest.main()
