# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""3D mRoPE for image requests (stage 12B) against the bf16 HF image-decision golden.

CPU (no device):
- ``rope.get_rope_index`` (host port of HF ``Qwen3_5Model.get_rope_index``) on the processor output
  of each of the 8 golden image rows == the golden ``position_ids``, bit for bit;
- the per-request cos/sin built from those ids (``mrope_freqs`` + ``host_tables``, BF16) == HF
  ``Qwen3_5TextRotaryEmbedding(x_bf16, position_ids)``, bit for bit, after the pair permutation;
- text-only ids (``arange`` on 3 streams) give exactly the setup-time ``PplxRotary`` angles.

Device: full-attention layer 3 fed the HF layer-2 output of an image row (golden spliced
``inputs_embeds`` -> HF bf16 layers 0-2), with ``PplxRequestRotary`` tables, vs the HF layer-3
output: PCC >= 0.995 over the real rows. Control (recorded, not gated): the same input with the
1D text tables.

Run::

    pytest models/demos/pplx_decider_v1_27b/tests/pcc/test_mrope.py -q -s
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from models.common.utility_functions import comp_pcc
from models.demos.pplx_decider_v1_27b.tests.test_utils import (
    DEVICE_PARAMS,
    build_rotary,
    build_tt_layer,
    check_pcc,
    model_args,
    reader,
    to_device,
    to_host,
)
from models.demos.pplx_decider_v1_27b.tt.rope import (
    PplxRequestRotary,
    get_rope_index,
    host_tables,
    mrope_freqs,
    rope_head_permutation,
    text_inv_freq,
)

E2E_GOLDEN = Path(
    os.environ.get("PPLX_DECIDER_IMAGE_GOLDEN", "/local/ttuser/gtobar/artifacts/pplx_decider/goldens/vision/e2e")
)
ROW_IDS = [
    "v01_dominant_color",
    "v02_count_circles",
    "v03_receipt_total",
    "v04_tallest_bar",
    "v05_red_circle_yes",
    "v06_red_circle_no",
    "v07_brightness_dark",
    "v08_progress_fill",
]
LAYER = 3  # first full-attention layer
LAYER_ROW = "v02_count_circles"  # 234 image tokens, rope positions jump back by 216 after the image


@lru_cache(maxsize=1)
def golden_rows() -> dict:
    return {r["id"]: r for r in map(json.loads, (E2E_GOLDEN / "prompts.jsonl").read_text().splitlines())}


def golden_tensor(row_id: str, name: str) -> torch.Tensor:
    with safe_open(str(E2E_GOLDEN / "golden_bf16.safetensors"), framework="pt") as f:
        return f.get_tensor(f"{row_id}.{name}")


@lru_cache(maxsize=1)
def tokenizer():
    from models.demos.pplx_decider_v1_27b.reference.image_decision_prompts import image_tokenizer

    return image_tokenizer()


@lru_cache(maxsize=8)
def encoded(row_id: str) -> dict:
    """The app's processor output for a golden row (re-encoded from its PNG)."""
    from models.demos.pplx_decider_v1_27b.reference.image_decision_prompts import encode

    enc = encode(tokenizer(), golden_rows()[row_id]["row"])
    assert enc["input_ids"][0].tolist() == golden_tensor(row_id, "input_ids").tolist(), "re-encoding drifted"
    return enc


def hf_cos_sin(position_ids: torch.Tensor):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    rope = Qwen3_5TextRotaryEmbedding(reader().text_config)
    seq = position_ids.shape[-1]
    return rope(torch.empty(1, seq, 1, dtype=torch.bfloat16), position_ids.reshape(3, 1, seq))


def tt_cos_sin_bf16(position_ids: torch.Tensor):
    a = model_args()
    cos, sin = host_tables(mrope_freqs(position_ids, text_inv_freq(a.rotary_dim, a.rope_theta)), a.head_dim)
    return cos.to(torch.bfloat16), sin.to(torch.bfloat16)


@pytest.mark.parametrize("row_id", ROW_IDS)
def test_rope_index_matches_golden(row_id):
    enc = encoded(row_id)
    pos = get_rope_index(enc["input_ids"], enc["mm_token_type_ids"], enc["image_grid_thw"], spatial_merge_size=2)
    golden = golden_tensor(row_id, "position_ids")
    assert pos.shape == golden.shape and pos.dtype == golden.dtype
    assert torch.equal(pos, golden)
    # Rope positions after the image are not token indices (the reason start_pos stays a cache index).
    assert int(pos[0, -1]) < pos.shape[1] - 1


@pytest.mark.parametrize("row_id", ROW_IDS)
def test_mrope_cos_sin_matches_hf(row_id):
    a = model_args()
    pos = golden_tensor(row_id, "position_ids")
    hf_cos, hf_sin = hf_cos_sin(pos)  # [1, S, 64] bf16, neox halves
    ones = torch.ones(hf_cos.shape[1], a.head_dim - a.rotary_dim, dtype=torch.bfloat16)
    perm = rope_head_permutation(a.head_dim, a.rotary_dim)
    want_cos = torch.cat([hf_cos[0], ones], dim=-1)[:, perm]
    want_sin = torch.cat([hf_sin[0], 0 * ones], dim=-1)[:, perm]
    cos, sin = tt_cos_sin_bf16(pos)
    assert torch.equal(cos, want_cos), float((cos.float() - want_cos.float()).abs().max())
    assert torch.equal(sin, want_sin), float((sin.float() - want_sin.float()).abs().max())


def test_text_only_ids_equal_1d_table():
    a = model_args()
    seq = 1024
    inv = text_inv_freq(a.rotary_dim, a.rope_theta)
    ids = torch.arange(seq).view(1, -1).expand(3, -1)
    one_d = torch.arange(seq, dtype=torch.float32)[:, None] * inv[None, :]  # PplxRotary._build_tables
    assert torch.equal(mrope_freqs(ids, inv), one_d)


# ----------------------------------------------------------------------------------------------
# Device: one full-attention layer on an image row
# ----------------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def hf_layer_io(row_id: str, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(input, output) of HF layer ``layer`` for the golden spliced embeddings, bf16 as the golden."""
    from models.demos.pplx_decider_v1_27b.reference.hf_decision_golden import stream_layers
    from models.demos.pplx_decider_v1_27b.reference.hf_image_decision_golden import mrope_position_embeddings
    from models.demos.pplx_decider_v1_27b.reference.hf_reference import build_decoder_layer

    pos = golden_tensor(row_id, "position_ids")
    pe = mrope_position_embeddings(reader().text_config, pos.reshape(3, 1, -1), torch.bfloat16)
    hs = [golden_tensor(row_id, "inputs_embeds")[None].to(torch.bfloat16)]
    build = lambda i: build_decoder_layer(reader(), i, torch.bfloat16)
    stream_layers(hs, [pe], build, range(layer))
    x = hs[0].clone()
    stream_layers(hs, [pe], build, [layer])
    # Same code and dtype as the e2e golden; the last token should match the stored trace
    # (CPU thread count can change bf16 reduction order, so this is reported, not asserted).
    stored = golden_tensor(row_id, "layer_last_hidden")[layer]
    print(
        f"HF layer {layer} last token vs stored golden trace: bit_exact={torch.equal(hs[0][0, -1], stored)} "
        f"max_abs={float((hs[0][0, -1].float() - stored.float()).abs().max())}"
    )
    return x[0], hs[0][0]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_full_attention_layer_image_row(device):
    a = model_args()
    assert a.layer_kind(LAYER) == "full_attention"
    x, want = hf_layer_io(LAYER_ROW, LAYER)
    seq, bucket = x.shape[0], 1024
    layer = build_tt_layer(device, LAYER)
    pos = golden_tensor(LAYER_ROW, "position_ids")
    padded = torch.nn.functional.pad(x.float(), (0, 0, 0, bucket - seq))[None]
    results = {}
    for name, rotary in (
        (
            "mrope",
            PplxRequestRotary.from_position_ids(
                pos, bucket, rotary_dim=a.rotary_dim, theta=a.rope_theta, head_dim=a.head_dim, mesh_device=device
            ),
        ),
        ("1d_control", build_rotary(device)),
    ):
        out = to_host(layer(to_device(padded, device), rotary), (1, bucket, a.hidden_size))[0, :seq]
        # The residual dominates the layer output; the layer's own update (out - x) shows the rope effect.
        results[f"{name}_update_pcc"] = float(comp_pcc((want - x).float(), out - x.float(), 0.0)[1])
        if name == "mrope":
            results[name] = check_pcc(
                want.float(), out, module="decoder_layer_full_attention_mrope_image", layer=LAYER, seq_len=seq
            )
        else:
            results[name] = float(comp_pcc(want.float(), out, 0.0)[1])
    print(f"layer {LAYER} on {LAYER_ROW} (S={seq}, bucket {bucket}): PCC {json.dumps(results)}")
    assert results["mrope_update_pcc"] > results["1d_control_update_pcc"]
