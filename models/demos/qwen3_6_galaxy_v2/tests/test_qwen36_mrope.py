# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V3-MROPE-1 CPU sanity: our `build_mrope_cos_sin` matches HF's
`Qwen3VLTextRotaryEmbedding.forward` exactly.

Pure CPU. Validates both:
  (a) Text-only case (3 axes identical) collapses to 1D RoPE so existing
      text-only tests / decode trace stay correct.
  (b) Multimodal case (3 axes diverge) matches HF's interleaved M-RoPE.
"""

import json
from pathlib import Path

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mrope import build_mrope_cos_sin, get_rope_index

_SNAPSHOT = Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/"
    "snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)


def _load_qwen36_text_config() -> Qwen3VLTextConfig:
    cfg = json.loads((_SNAPSHOT / "config.json").read_text())
    tc = {k: v for k, v in cfg["text_config"].items() if k != "model_type"}
    text_cfg = Qwen3VLTextConfig(**tc)
    # qwen3.6 stores rope settings under `rope_parameters`; HF reads from
    # `rope_scaling`. Mirror the dict over so the HF rotary class can find it.
    text_cfg.rope_scaling = text_cfg.rope_parameters
    # HF's `compute_default_rope_parameters` reads `config.rope_theta` from the
    # top of the config, but qwen3.6 ships it inside `rope_parameters`. Override
    # the default 5e6 with qwen3.6's actual 1e7 so the reference matches.
    text_cfg.rope_theta = text_cfg.rope_parameters["rope_theta"]
    return text_cfg


def _qwen36_rope_params() -> dict:
    tc = _load_qwen36_text_config()
    rp = tc.rope_parameters
    head_dim = tc.head_dim
    partial_rotary_dim = int(head_dim * rp["partial_rotary_factor"])
    return {
        "rope_theta": rp["rope_theta"],
        "head_dim": head_dim,
        "partial_rotary_dim": partial_rotary_dim,
        "mrope_section": rp["mrope_section"],
    }


@torch.no_grad()
def test_qwen36_rope_params_are_what_we_expect():
    """Sanity: qwen3.6's text_config has the rope params we built around."""
    params = _qwen36_rope_params()
    assert params["rope_theta"] == 10_000_000
    assert params["partial_rotary_dim"] == 64  # head_dim 256 × partial_rotary_factor 0.25
    assert params["mrope_section"] == [11, 11, 10]
    assert sum(params["mrope_section"]) == params["partial_rotary_dim"] // 2 == 32


@torch.no_grad()
def test_mrope_text_only_collapses_to_1d_rope():
    """For text-only inputs (3 axes identical), our M-RoPE == HF's 1D RoPE."""
    params = _qwen36_rope_params()

    B, S = 1, 64
    # Text-only: a 1D ramp broadcast to 3 axes
    pos_1d = torch.arange(S, dtype=torch.long).unsqueeze(0)  # [B=1, S]
    pos_3d = pos_1d[None, ...].expand(3, B, S).contiguous()  # [3, B, S]

    cos_ours, sin_ours = build_mrope_cos_sin(
        pos_3d,
        rope_theta=params["rope_theta"],
        partial_rotary_dim=params["partial_rotary_dim"],
        mrope_section=params["mrope_section"],
    )

    # HF reference path — produces full-head_dim cos/sin
    tc = _load_qwen36_text_config()
    rot = Qwen3VLTextRotaryEmbedding(tc)
    dummy_x = torch.zeros(B, S, 1, dtype=torch.float32)
    cos_ref, sin_ref = rot(dummy_x, pos_3d)

    print(f"shapes: ours cos {tuple(cos_ours.shape)}, ref cos {tuple(cos_ref.shape)}")
    assert cos_ours.shape == cos_ref.shape, f"cos shape mismatch: {cos_ours.shape} vs {cos_ref.shape}"
    torch.testing.assert_close(cos_ours, cos_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sin_ours, sin_ref, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_mrope_multimodal_3axis_matches_hf():
    """Multimodal: 3 axes diverge. Ours matches HF's interleaved M-RoPE."""
    params = _qwen36_rope_params()
    B = 1
    # Simulated multimodal positions: 10 text tokens, then 4 image tokens with (t, h, w) coords
    # (text part: all 3 axes equal; image part: divergent)
    text_ids = torch.arange(10, dtype=torch.long)
    image_t = torch.tensor([10, 10, 10, 10], dtype=torch.long)
    image_h = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    image_w = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    pos_t = torch.cat([text_ids, image_t]).unsqueeze(0)
    pos_h = torch.cat([text_ids, image_h]).unsqueeze(0)
    pos_w = torch.cat([text_ids, image_w]).unsqueeze(0)
    pos_3d = torch.stack([pos_t, pos_h, pos_w], dim=0)  # [3, 1, 14]

    cos_ours, sin_ours = build_mrope_cos_sin(
        pos_3d,
        rope_theta=params["rope_theta"],
        partial_rotary_dim=params["partial_rotary_dim"],
        mrope_section=params["mrope_section"],
    )

    tc = _load_qwen36_text_config()
    rot = Qwen3VLTextRotaryEmbedding(tc)
    dummy_x = torch.zeros(B, pos_3d.shape[2], 1, dtype=torch.float32)
    cos_ref, sin_ref = rot(dummy_x, pos_3d)

    torch.testing.assert_close(cos_ours, cos_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sin_ours, sin_ref, rtol=1e-5, atol=1e-5)
    print(
        f"divergent-axis M-RoPE matches HF: cos sample {cos_ours[0, 12, :4].tolist()}, "
        f"sin sample {sin_ours[0, 12, :4].tolist()}"
    )


@torch.no_grad()
def test_get_rope_index_text_only():
    """Pure-text sequence: all 3 axes equal a 1D ramp."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    pos_ids, deltas = get_rope_index(input_ids)
    assert pos_ids.shape == (3, 1, 5)
    ramp = torch.arange(5, dtype=torch.long)
    torch.testing.assert_close(pos_ids[0, 0], ramp)
    torch.testing.assert_close(pos_ids[1, 0], ramp)
    torch.testing.assert_close(pos_ids[2, 0], ramp)
    assert deltas.tolist() == [[5 - 5]] == [[0]]  # max+1 = 5, len = 5


@torch.no_grad()
def test_get_rope_index_text_image_text():
    """Text + image + text: image_pad positions get 3D grid coords.

    Mirrors real HF processor output: a `<|vision_start|>` token (248053)
    immediately precedes the `<|image_pad|>` run — `get_rope_index` counts
    vision segments off vision_start (matching HF `Qwen3VLModel.get_rope_index`).
    """
    IMG = 248056
    VSTART = 248053
    # 1 text token, vision_start, image (grid_thw=[[1,4,4]] → 2x2=4 patch tokens), 2 text tokens
    n_img = (4 // 2) * (4 // 2)  # 2*2 = 4
    input_ids = torch.tensor([[100, VSTART] + [IMG] * n_img + [200, 201]], dtype=torch.long)
    image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    pos_ids, deltas = get_rope_index(input_ids, image_grid_thw=image_grid_thw)
    S = input_ids.shape[1]
    assert pos_ids.shape == (3, 1, S)

    # First 2 tokens (text 100 + vision_start): [0, 1] on all axes
    torch.testing.assert_close(pos_ids[0, 0, :2], torch.tensor([0, 1]))
    torch.testing.assert_close(pos_ids[1, 0, :2], torch.tensor([0, 1]))
    torch.testing.assert_close(pos_ids[2, 0, :2], torch.tensor([0, 1]))

    # Image: st_idx starts at 2. grid_t=1, grid_h=2, grid_w=2.
    # t_idx = [0,0,0,0], h_idx = [0,0,1,1], w_idx = [0,1,0,1], each +2
    torch.testing.assert_close(pos_ids[0, 0, 2:6], torch.tensor([2, 2, 2, 2]))
    torch.testing.assert_close(pos_ids[1, 0, 2:6], torch.tensor([2, 2, 3, 3]))
    torch.testing.assert_close(pos_ids[2, 0, 2:6], torch.tensor([2, 3, 2, 3]))

    # After image, running advances to max(grid coords) + 1 = 3 + 1 = 4.
    # Next 2 text tokens: [4, 5] all axes.
    torch.testing.assert_close(pos_ids[0, 0, 6:8], torch.tensor([4, 5]))
    torch.testing.assert_close(pos_ids[1, 0, 6:8], torch.tensor([4, 5]))
    torch.testing.assert_close(pos_ids[2, 0, 6:8], torch.tensor([4, 5]))
    print(f"pos_ids[0,0]={pos_ids[0, 0].tolist()}")
    print(f"pos_ids[1,0]={pos_ids[1, 0].tolist()}")
    print(f"pos_ids[2,0]={pos_ids[2, 0].tolist()}")
