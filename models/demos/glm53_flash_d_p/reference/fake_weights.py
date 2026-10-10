# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Random GLM-5.3-Flash weights with the checkpoint's names and shapes, for device perf runs without the checkpoint.

FakeLoader stands in for reference/weights.py:WeightLoader (get / has / layer / is_fp8 / weight): every tensor a block
reads comes out random (deterministic per name) in the checkpoint's shape and dtype, from fake_manifest.json (the
config's text_config plus one layer of each block type, read from the checkpoint headers by `python -m
models.demos.glm53_flash_d_p.reference.fake_weights --hf <checkpoint dir>`). Routed experts are never materialised:
loader.fake tells build_experts_ag to give the flat expert uninitialised weights (FlatRoutedExpert(weights="fake")).

Values only need to be finite and roughly unit-scale: linear weights N(0, 1 / fan_in), norm weights 1, biases 0,
A_log log U(1, 16). Routing therefore comes out near-uniform over the 288 experts (random router, random x);
GLM_FAKE_HOT=n gives experts 0 .. n-1 a +100 correction bias, so every token picks them (the hot-expert case).
"""

from __future__ import annotations

import json
import os
import re
import zlib
from pathlib import Path

import torch

MANIFEST = Path(__file__).with_name("fake_manifest.json")
PREFIX = "model.language_model."
# one layer of each block type: 0 KDA + dense MLP, 3 DSA + MoE, 4 KDA + MoE
REPS = {"kda_dense": 0, "dsa_moe": 3, "kda_moe": 4}
_LAYER = re.compile(r"^" + re.escape(PREFIX) + r"layers\.(\d+)\.(.*)$")


def fake_config():
    """GlmConfig from the manifest's text_config (the checkpoint's config.json, no checkpoint needed)."""
    import tempfile

    from models.demos.glm53_flash_d_p.reference.glm_ref import GlmConfig

    raw = json.loads(MANIFEST.read_text())["config"]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"text_config": raw}, f)
    try:
        return GlmConfig.from_json(f.name)
    finally:
        os.unlink(f.name)


class FakeLoader:
    fake = True

    def __init__(self, cfg=None):
        m = json.loads(MANIFEST.read_text())
        self.cfg = cfg or fake_config()
        self.kinds = m["layers"]  # block type -> {name under layers.L.: [shape, dtype]}
        self.top = m["top"]  # name -> [shape, dtype] (embedding, final norm)
        self.hot = int(os.environ.get("GLM_FAKE_HOT", "0"))

    def _kind(self, i):
        if not self.cfg.is_moe(i):
            return "kda_dense"
        return "kda_moe" if self.cfg.is_kda(i) else "dsa_moe"

    def _spec(self, name):
        m = _LAYER.match(name)
        if m is None:
            return self.top.get(name)
        sub = re.sub(r"^mlp\.experts\.\d+\.", "mlp.experts.0.", m.group(2))
        return self.kinds[self._kind(int(m.group(1)))].get(sub)

    def has(self, name: str) -> bool:
        return self._spec(name) is not None

    def is_fp8(self, name: str) -> bool:
        return False  # weight() returns the dequantized form directly

    def get(self, name: str) -> torch.Tensor:
        spec = self._spec(name)
        assert spec is not None, f"fake manifest has no {name}"
        shape, dtype = spec
        g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
        leaf = name.rsplit(".", 1)[-1] if not name.endswith(".weight") else name.rsplit(".", 2)[-2]
        if "norm" in name and name.endswith(".weight"):
            t = torch.ones(shape)
        elif name.endswith("bias") or name.endswith("_base"):
            t = torch.zeros(shape)
            if name.endswith("e_score_correction_bias") and self.hot:
                t[: self.hot] = 100.0
        elif leaf == "A_log":
            t = torch.log(torch.rand(shape, generator=g) * 15 + 1)
        elif name.endswith("_scale"):
            t = torch.ones(shape)
        else:
            fan_in = shape[-1] if len(shape) > 1 else 1
            t = torch.randn(shape, generator=g) / max(fan_in, 1) ** 0.5
        return t.to(getattr(torch, dtype))

    def layer(self, i: int, name: str) -> torch.Tensor:
        return self.get(f"{PREFIX}layers.{i}.{name}")

    def weight(self, name: str, dtype=torch.float32) -> torch.Tensor:
        return self.get(name).to(dtype)


def write_manifest(hf: str) -> None:
    """Read config.json and the safetensors headers of one layer per block type (no tensor data)."""
    from safetensors import safe_open

    from models.demos.glm53_flash_d_p.reference.weights import WeightLoader

    cfg = json.loads(Path(hf, "config.json").read_text())["text_config"]
    wl = WeightLoader(hf)
    layers = {k: {} for k in REPS}
    top = {}
    handles = {}
    for name, fname in wl.weight_map.items():
        if name.endswith("_scale_inv") or not name.startswith(PREFIX):
            continue
        m = _LAYER.match(name)
        if m is None:
            dst, sub = top, name
        else:
            kind = next((k for k, i in REPS.items() if i == int(m.group(1))), None)
            if kind is None or (re.match(r"mlp\.experts\.(\d+)\.", m.group(2)) and ".experts.0." not in name):
                continue
            dst, sub = layers[kind], m.group(2)
        if fname not in handles:
            handles[fname] = safe_open(os.path.join(hf, fname), framework="pt")
        sl = handles[fname].get_slice(name)
        dt = str(sl.get_dtype()).lower()
        dtype = {"bf16": "bfloat16", "f32": "float32", "f8_e4m3": "bfloat16", "f16": "float16"}.get(dt, "float32")
        dst[sub] = [list(sl.get_shape()), dtype]
    MANIFEST.write_text(json.dumps({"config": cfg, "layers": layers, "top": top}, indent=0, sort_keys=True) + "\n")
    print(f"wrote {MANIFEST}: " + ", ".join(f"{k} {len(v)} tensors" for k, v in layers.items()) + f", top {len(top)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="GLM-5.3-Flash checkpoint dir (config.json + safetensors)")
    write_manifest(ap.parse_args().hf)
