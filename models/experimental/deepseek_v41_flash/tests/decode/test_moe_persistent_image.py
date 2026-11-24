# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The persistent TP12 MoE's image twins (``PersistentMoE(image_bias=True)``): an image-span token's request names its
layer's twin row of the layer table, which ranks the experts with the router's ``bias_vl`` instead of its bias.

Same setup as ``test_moe_persistent.py``: a random layer at V4.1's shapes on the 3x4 MoE submesh at row 5, here with a
random ``gate.bias_vl`` far enough from its bias that most rows pick other experts with it, and one TP4 requester on
row 4. The server is launched as the instances' server is (one-row requests on a 4-row block launch). Every row is
sent as a text request and as an image request, then 4-row image blocks; each output is checked against the torch
reference ranked with the matching bias, and each row that picks other experts as an image token must be closer to
the image reference than to the text one.

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_moe_persistent_image.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tests.decode.test_moe_persistent import (
    PARENT_MESH,
    PCC,
    REQUEST_ROWS,
    TP_SIZE,
    _pcc,
    _random_layer,
    _reference,
    _setup,
    _single_senders,
    _snapshot,
)
from models.experimental.deepseek_v41_flash.tt.config import CACHE_DIR, DEFAULT_MODEL_DIR
from models.experimental.deepseek_v41_flash.tt.decode.moe import DeepSeekV41PreloadedExperts, DeepSeekV41SparseMoeBlock
from models.experimental.deepseek_v41_flash.tt.decode.persistent_moe import PersistentMoE, untile_rows
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache

ROWS = 4
NUM_ROWS = 8  # one-row requests, each as text and as image
NUM_BLOCKS = 2  # 4-row image blocks
# bias_vl's spread: the random layer's scores spread ~0.3 and its text bias 0.1, so with this most rows' top 6 change.
BIAS_VL_SCALE = 0.5


def _image_layer(device, config):
    """``(provider, weights, image_weights, moe)``: test_moe_persistent's random layer 0 (its cached experts) with a
    random ``gate.bias_vl``; ``image_weights`` is ``weights`` ranked with it, for the torch reference."""
    provider, weights = _random_layer(config, seed=0)
    g = torch.Generator().manual_seed(1000)
    weights["gate.bias_vl"] = torch.randn(config.num_local_experts, generator=g) * BIAS_VL_SCALE
    cache = WeightCache(CACHE_DIR).sub("persistent_test.random0.ffn")
    experts = DeepSeekV41PreloadedExperts(config, provider, device, dtype=ttnn.bfloat4_b, cache=cache, tp_size=TP_SIZE)
    # The bias table is cached apart from the router's other tensors: a stale one from other values would pass silently.
    table = cache.file("gate.bias_table")
    moe = DeepSeekV41SparseMoeBlock(
        config, {k: (lambda v=v: v) for k, v in weights.items()}, device, experts, cache=cache, tp_size=TP_SIZE
    )
    assert moe.gate.bias_table is not None, f"the router built no image bias table (cache {table})"
    image_weights = dict(weights, **{"gate.e_score_correction_bias": weights["gate.bias_vl"]})
    return provider, weights, image_weights, moe


def _topk(x, weights, config, k):
    gate_w = weights["gate.weight"] / getattr(config, "gate_temp", 1.0)
    biased = torch.sqrt(torch.nn.functional.softplus(gate_w @ x)) + weights["gate.e_score_correction_bias"]
    return set(biased.topk(k).indices.tolist())


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash config not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@torch.no_grad()
def test_moe_persistent_image_twins(mesh_device):
    moe_mesh, request_meshes, config, socket_core, request_core = _setup(mesh_device)
    provider, weights, image_weights, moe = _image_layer(moe_mesh, config)
    k = moe.experts.top_k
    hidden = config.hidden_size

    torch.manual_seed(0)
    rows = torch.randn(NUM_ROWS, hidden).to(torch.bfloat16).float()
    blocks = torch.randn(NUM_BLOCKS, ROWS, hidden).to(torch.bfloat16).float()

    moe_op = PersistentMoE(
        moe_mesh,
        request_meshes[-1:],
        {0: moe},
        socket_core,
        request_core,
        _single_senders(REQUEST_ROWS[-1:]),
        rows=1,
        block_rows=ROWS,
        image_bias=True,
    )
    assert moe_op.image_table_rows == {0: 1}, moe_op.image_table_rows
    text_out, image_out, block_out = [], [], []
    try:
        for x in rows:
            for image, outs in ((False, text_out), (True, image_out)):
                moe_op.upload(0, x)
                out = moe_op.request(0, 0, image=image)
                outs.append(PersistentMoE.row_of(out).float())
                ttnn.deallocate(out)
        moe_op.use_rows(ROWS)
        for block in blocks:
            moe_op.upload(0, block)
            out = moe_op.request(0, 0, image=True)
            block_out.append(untile_rows(PersistentMoE.row_of(out).float(), ROWS))
            ttnn.deallocate(out)
    finally:
        moe_op.stop()

    def best(out, x, w):
        return max(_pcc(c, out) for c in _reference(x, provider, w, config, moe.experts)[0])

    report, failed, moved = [], False, 0
    for i, x in enumerate(rows):
        text_pcc = best(text_out[i], x, weights)
        image_pcc = best(image_out[i], x, image_weights)
        image_vs_text = best(image_out[i], x, weights)
        changed = _topk(x, weights, config, k) != _topk(x, image_weights, config, k)
        moved += changed
        failed |= text_pcc < PCC or image_pcc < PCC or (changed and image_vs_text >= image_pcc)
        report.append(
            f"row {i}: text vs text ref {text_pcc:.5f}, image vs image ref {image_pcc:.5f}, image vs text ref "
            f"{image_vs_text:.5f}{' (top-k changes as an image token)' if changed else ' (same top-k either way)'}"
        )
    for b, block in enumerate(blocks):
        for r, x in enumerate(block):
            pcc = best(block_out[b][r], x, image_weights)
            failed |= pcc < PCC
            report.append(f"image block {b} row {r}: vs image ref {pcc:.5f}")
    logger.info("\n" + "\n".join(report))
    assert moved, "no row picks other experts with bias_vl: the test cannot tell the twin rows from the text rows"
    assert not failed, "\n".join(report)
