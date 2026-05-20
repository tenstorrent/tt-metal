# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V2-VISION-V1 sanity test.

Confirms that the HF transformers `Qwen3VLVisionModel` (which qwen3.6 v2's
TTNN bring-up will use as the reference) can be:
  (1) constructed from qwen3.6's `vision_config` JSON block,
  (2) loaded strictly with qwen3.6's `model.visual.*` weights, and
  (3) run forward on synthetic patch input without NaN/Inf and produces
      the expected output shape `(seq // spatial_merge_unit, out_hidden_size)`.

HF transformers 4.57.1 does NOT yet ship a `Qwen3_5ForConditionalGeneration`
class. qwen3.6's vision_config matches the qwen3_vl arch exactly except for
`out_hidden_size` (5120 vs 3584) and `deepstack_visual_indexes` ([] vs
[8,16,24]) — both passed through the `Qwen3VLVisionConfig` constructor.

Pure-CPU, runs in seconds. No mesh device required.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

_SNAPSHOT = Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/"
    "snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)


def _load_qwen36_vision_state_dict() -> dict[str, torch.Tensor]:
    """Read every `model.visual.*` tensor across the 15 shards into a dict.

    Strips the leading `model.visual.` prefix so the keys match HF's
    `Qwen3VLVisionModel.state_dict()` layout.
    """
    index = json.loads((_SNAPSHOT / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    visual_keys = [k for k in weight_map if k.startswith("model.visual.")]
    by_shard: dict[str, list[str]] = {}
    for k in visual_keys:
        by_shard.setdefault(weight_map[k], []).append(k)
    out: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(_SNAPSHOT / shard, framework="pt", device="cpu") as f:
            for k in keys:
                out[k[len("model.visual.") :]] = f.get_tensor(k)
    return out


def test_qwen36_vision_config_translates_to_hf_qwen3vl():
    """qwen3.6 vision_config -> HF Qwen3VLVisionConfig with no field drift."""
    cfg = json.loads((_SNAPSHOT / "config.json").read_text())
    vc = {k: v for k, v in cfg["vision_config"].items() if k != "model_type"}
    hf_cfg = Qwen3VLVisionConfig(**vc)

    assert hf_cfg.depth == 27
    assert hf_cfg.hidden_size == 1152
    assert hf_cfg.intermediate_size == 4304
    assert hf_cfg.num_heads == 16
    assert hf_cfg.patch_size == 16
    assert hf_cfg.spatial_merge_size == 2
    assert hf_cfg.temporal_patch_size == 2
    assert hf_cfg.num_position_embeddings == 2304
    assert hf_cfg.out_hidden_size == 5120
    assert list(hf_cfg.deepstack_visual_indexes) == []
    assert hf_cfg.hidden_act == "gelu_pytorch_tanh"


def test_qwen36_vision_weights_strict_load_into_hf_qwen3vl():
    """qwen3.6 `model.visual.*` weights load with strict=True (no missing / unexpected)."""
    cfg = json.loads((_SNAPSHOT / "config.json").read_text())
    vc = {k: v for k, v in cfg["vision_config"].items() if k != "model_type"}
    hf_cfg = Qwen3VLVisionConfig(**vc)
    model = Qwen3VLVisionModel(hf_cfg)

    sd = _load_qwen36_vision_state_dict()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"missing keys ({len(missing)}):", missing[:8])
    print(f"unexpected keys ({len(unexpected)}):", unexpected[:8])
    assert not missing, f"HF expected keys not in qwen3.6 visual.*: {missing[:8]}"
    assert not unexpected, f"qwen3.6 visual.* keys not used by HF: {unexpected[:8]}"


@pytest.mark.parametrize("grid_h,grid_w", [(14, 14)])
def test_qwen36_vision_forward_runs_and_shape_correct(grid_h: int, grid_w: int):
    """Forward on a small synthetic patch tensor — sanity check shape + no NaN/Inf.

    Synthetic input shape mirrors what the HF preprocessor would emit:
        (seq_len, in_channels * temporal_patch_size * patch_size**2)
        = (1 * grid_h * grid_w, 3 * 2 * 16 * 16) = (seq_len, 1536)

    Output:
        (seq_len // spatial_merge_unit, out_hidden_size) = (seq//4, 5120).
    """
    cfg = json.loads((_SNAPSHOT / "config.json").read_text())
    vc = {k: v for k, v in cfg["vision_config"].items() if k != "model_type"}
    hf_cfg = Qwen3VLVisionConfig(**vc)
    model = Qwen3VLVisionModel(hf_cfg).eval()
    model.load_state_dict(_load_qwen36_vision_state_dict(), strict=True)

    seq_len = grid_h * grid_w
    patch_feat_dim = hf_cfg.in_channels * hf_cfg.temporal_patch_size * hf_cfg.patch_size**2
    assert patch_feat_dim == 1536

    torch.manual_seed(0)
    hidden_states = torch.randn(seq_len, patch_feat_dim, dtype=torch.float32)
    grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)

    with torch.no_grad():
        out, deepstack = model(hidden_states, grid_thw=grid_thw)

    expected_seq = seq_len // (hf_cfg.spatial_merge_size**2)
    assert out.shape == (expected_seq, hf_cfg.out_hidden_size), out.shape
    assert torch.isfinite(out).all(), "NaN / Inf in vision encoder output"
    assert list(deepstack) == [], "qwen3.6 has empty deepstack_visual_indexes"
    print(
        f"output shape {tuple(out.shape)}, "
        f"mean={out.mean().item():.4f}, std={out.std().item():.4f}, "
        f"min={out.min().item():.4f}, max={out.max().item():.4f}"
    )
