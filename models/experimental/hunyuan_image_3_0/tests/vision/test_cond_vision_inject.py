# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Device PCC test for the Instruct cond-vision scatter (tt/vision/inject.py).
# Injects projected SigLIP2 features into a contiguous <img> span and checks it
# matches the reference masked scatter (instantiate_vit_image_tokens) for the
# single contiguous-span case. Uses the session `device` fixture from conftest.py.
#
# Run:
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/vision/test_cond_vision_inject.py -v -s

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.experimental.hunyuan_image_3_0.ref.vision.inject import (
    contiguous_image_mask,
    scatter_vit_image_tokens,
)
from models.experimental.hunyuan_image_3_0.tt.vision.inject import scatter_cond_vision_embeddings

H = 4096
PCC_THR = 0.999  # scatter is pure data movement -> near-exact (bf16 round-trip only)


def _up(device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


@pytest.mark.parametrize(
    "B,S,start,stop",
    [
        (1, 128, 32, 96),  # text on both sides
        (2, 256, 64, 192),  # batched
        (1, 96, 0, 64),  # span at sequence start (no text_pre)
        (1, 96, 32, 96),  # span at sequence end (no text_post)
    ],
)
def test_scatter_cond_vision_pcc(device, B, S, start, stop):
    torch.manual_seed(0)
    n_img = stop - start
    hidden = torch.randn(B, S, H)
    image_embeds = torch.randn(B, n_img, H)

    # Golden: the ref/ scatter (verbatim upstream masked_select + scatter_) over a
    # contiguous <img> mask — the case the device op handles.
    image_masks = contiguous_image_mask(B, S, slice(start, stop))
    ref = scatter_vit_image_tokens(hidden, image_embeds, image_masks)

    h_tt = _up(device, hidden)
    e_tt = _up(device, image_embeds)
    out_tt = scatter_cond_vision_embeddings(h_tt, e_tt, slice(start, stop))
    out = ttnn.to_torch(out_tt).float()
    ttnn.deallocate(out_tt)
    ttnn.deallocate(h_tt)
    ttnn.deallocate(e_tt)

    assert tuple(out.shape) == tuple(ref.shape), f"shape {tuple(out.shape)} != {tuple(ref.shape)}"
    passing, pcc = comp_pcc(ref, out, PCC_THR)
    logger.info(f"cond-vision scatter B={B} S={S} span=[{start}:{stop}] PCC={pcc} (>= {PCC_THR}, passing={passing})")
    assert passing, f"PCC {pcc} < {PCC_THR}"
