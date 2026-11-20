# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the H3 VAE cache is keyed by the checkpoint's content and the decoder loads at model load."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from test_h3_audio_ccl_preparation import methods

PIPE = "pipelines/minimax_h3/pipeline_minimax_h3.py"
VAE = "models/vae/minimax_h3/vae_minimax_h3.py"


class Recorder:
    def __init__(self):
        self.calls = []

    def load_model(self, module, **kwargs):
        self.calls.append(kwargs)


def pipeline(weights_dir, recorder):
    ns = {"json": json, "Path": Path, "cache": recorder, "MODEL_NAME": "m", "conv3d_blocking_hash": lambda m: ""}
    cls = type("P", (), methods(PIPE, "MiniMaxH3Pipeline", {"_safetensors_files", "_cache_submodel"}, ns))
    obj = cls()
    obj.weights_dir = Path(weights_dir)
    obj.vae_parallel_config = "dp"
    obj.mesh_device = SimpleNamespace(shape=(4, 8))
    return obj


class H3VaeCacheAtLoad(unittest.TestCase):
    def test_sources_are_the_sharded_vae_files(self):
        with tempfile.TemporaryDirectory() as d:
            vae = Path(d) / "vae"
            vae.mkdir()
            (vae / "diffusion_pytorch_model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "s2.safetensors", "b": "s1.safetensors", "c": "s1.safetensors"}})
            )
            rec = Recorder()
            p = pipeline(d, rec)
            self.assertEqual(p._safetensors_files("vae"), [vae / "s1.safetensors", vae / "s2.safetensors"])
            p._cache_submodel(object(), "vae_decoder_t1_h2_w3", {})
            self.assertEqual(rec.calls[0]["sources"], [vae / "s1.safetensors", vae / "s2.safetensors"])

    def test_single_file_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            vae = Path(d) / "vae"
            vae.mkdir()
            p = pipeline(d, Recorder())
            with self.assertRaises(FileNotFoundError):
                p._safetensors_files("vae")
            (vae / "diffusion_pytorch_model.safetensors").write_bytes(b"")
            self.assertEqual(p._safetensors_files("vae"), [vae / "diffusion_pytorch_model.safetensors"])

    def test_load_decoder_loads_the_decoder_subfolder(self):
        cls = type("V", (), methods(VAE, "MiniMaxH3Vae", {"load_decoder"}, {}))
        v = cls()
        v.decoder = object()
        v._decoder_subfolder = lambda: "vae_decoder_x"
        seen = []
        v._ensure_loaded = lambda module, sub: seen.append((module, sub))
        v.load_decoder()
        self.assertEqual(seen, [(v.decoder, "vae_decoder_x")])

    def test_init_loads_decoder_for_coresident_only(self):
        import ast

        src = (Path(__file__).resolve().parents[2] / PIPE).read_text()
        init = next(
            n for c in ast.parse(src).body if isinstance(c, ast.ClassDef) and c.name == "MiniMaxH3Pipeline"
            for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        text = ast.unparse(init)
        self.assertRegex(text, r"if self\.coresident:\n\s+self\._vae\.load_decoder\(\)")
        self.assertLess(text.index("self._vae.load_decoder()"), text.index("if warmup:"))


if __name__ == "__main__":
    unittest.main()
