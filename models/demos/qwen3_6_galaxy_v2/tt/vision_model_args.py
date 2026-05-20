# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V2-VISION-V2 bridge: lets qwen3_vl's TTNN vision blocks consume qwen3.6 weights.

qwen3.6-27B reports `model_type=qwen3_5` / `architectures=Qwen3_5ForConditionalGeneration`,
neither of which transformers 4.57.1 recognizes. We register tiny subclasses of
`Qwen3VLConfig` / `Qwen3VLTextConfig` against AutoConfig so the existing
`VisionModelArgs(ModelArgs)` machinery loads qwen3.6's config unchanged.

Then `Qwen36VisionModelArgs` overrides `reference_vision_model()` (and the
helpers that depend on it) to load the qwen3.6 `model.visual.*` safetensors
weights directly into `Qwen3VLVisionModel` — bypassing the
`AutoModelForCausalLM.from_pretrained()` call which would fail.

Architecturally the qwen3.6 vision encoder is a `Qwen3VLVisionModel` with
`out_hidden_size=5120` and `deepstack_visual_indexes=[]` (verified in
V1: tests/test_vision_reference_v1.py — 333 keys load strict, forward
produces (seq//4, 5120) with no NaN/Inf).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from models.demos.qwen3_vl.tt.model_config import VisionModelArgs

# -------------------------------------------------------------------------
# Step 1 — register qwen3.6's model_type strings with AutoConfig.
# Required because transformers 4.57.1 doesn't ship `Qwen3_5Config`.
# Registration is idempotent (ValueError raised on re-register is swallowed).
# -------------------------------------------------------------------------


class _Qwen35TextConfig(Qwen3VLTextConfig):
    model_type = "qwen3_5_text"


class _Qwen35Config(Qwen3VLConfig):
    model_type = "qwen3_5"
    sub_configs = {"text_config": _Qwen35TextConfig, "vision_config": Qwen3VLVisionConfig}


def _register_qwen35_config_aliases() -> None:
    for model_type, cls in [("qwen3_5_text", _Qwen35TextConfig), ("qwen3_5", _Qwen35Config)]:
        try:
            AutoConfig.register(model_type, cls)
        except ValueError:
            pass  # already registered


_register_qwen35_config_aliases()


# -------------------------------------------------------------------------
# Step 2 — direct safetensors loader (proven in V1 test).
# -------------------------------------------------------------------------


def load_qwen36_visual_state_dict(checkpoint_dir: str | Path) -> dict[str, torch.Tensor]:
    """Read every `model.visual.*` tensor across all safetensors shards.

    Strips the `model.visual.` prefix so keys line up with
    `Qwen3VLVisionModel.state_dict()` exactly (verified by V1 strict-load).
    """
    ckpt = Path(checkpoint_dir)
    if ckpt.is_dir() and (ckpt / "model.safetensors.index.json").exists():
        snapshot = ckpt
    else:
        # Treat as a HF model id — find the cached snapshot dir.
        cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{str(ckpt).replace('/', '--')}"
        snap_root = cache_root / "snapshots"
        if not snap_root.exists():
            raise FileNotFoundError(f"No HF snapshot under {snap_root}; HF_MODEL path not local")
        snapshots = sorted(snap_root.iterdir())
        if not snapshots:
            raise FileNotFoundError(f"No snapshots in {snap_root}")
        snapshot = snapshots[-1]

    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    visual_keys = [k for k in weight_map if k.startswith("model.visual.")]
    by_shard: dict[str, list[str]] = {}
    for k in visual_keys:
        by_shard.setdefault(weight_map[k], []).append(k)
    out: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(snapshot / shard, framework="pt", device="cpu") as f:
            for k in keys:
                out[k[len("model.visual.") :]] = f.get_tensor(k)
    return out


# -------------------------------------------------------------------------
# Step 3 — the bridge args class.
# -------------------------------------------------------------------------


class Qwen36VisionModelArgs(VisionModelArgs):
    """qwen3_vl's VisionModelArgs, retargeted at qwen3.6's vision encoder.

    Overrides `reference_vision_model` to load weights directly from
    safetensors instead of going through `AutoModelForCausalLM.from_pretrained`
    (which fails because transformers 4.57.1 doesn't ship `Qwen3_5ForConditionalGeneration`).
    """

    _CACHED_REFERENCE_VISION: Qwen3VLVisionModel | None = None

    def reference_vision_model(self, depth: int | None = None) -> Qwen3VLVisionModel:
        """Construct a HF `Qwen3VLVisionModel` and strict-load qwen3.6 visual.* weights."""
        # Cache across calls so the patch_merger / block / attention / mlp accessors
        # don't reload weights repeatedly.
        if self._CACHED_REFERENCE_VISION is not None and depth is None:
            return self._CACHED_REFERENCE_VISION

        hf_vision_config = self.hf_config.vision_config
        if depth is not None and depth != hf_vision_config.depth:
            # Build a depth-truncated copy of the vision config for sub-tests.
            from dataclasses import replace as _replace

            try:
                hf_vision_config = _replace(hf_vision_config, depth=depth)
            except TypeError:
                hf_vision_config = type(hf_vision_config)(**{**hf_vision_config.to_dict(), "depth": depth})

        model = Qwen3VLVisionModel(hf_vision_config).eval()
        sd = load_qwen36_visual_state_dict(self.CKPT_DIR)
        if depth is None:
            model.load_state_dict(sd, strict=True)
        else:
            missing, _ = model.load_state_dict(sd, strict=False)
            assert not missing, f"depth-truncated load still missing keys: {missing[:4]}"

        if depth is None:
            type(self)._CACHED_REFERENCE_VISION = model
        return model
