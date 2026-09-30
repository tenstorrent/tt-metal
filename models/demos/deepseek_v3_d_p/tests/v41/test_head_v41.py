# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 embedding and output head (bead F1) vs the reference ``ParallelEmbedding``, final ``norm``
and ``ParallelHead`` at V4.1 dims, synthetic and checkpoint weights (embed: shard 02, norm/head: shard 43).

Inputs come from the §6 oracle of V4.1 layer 2 on one 5120-token chunk (the same run as
``test_moe_v41``): its prompt tokens for the embedding, and the collapsed final hidden
``hc_pre(x_out, pre_out)`` for the head. The head is evaluated on the last row (the reference's own
logits) and on rows spread over SP chips and tile offsets.

Bars: embedding exact (``torch.equal``); head logits PCC >= 0.999 (new linear op bar) and identical top-1
on every row; bit-identical repeats.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as O
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tt.v41.head import TtV41Embedding, TtV41Head
from tests.ttnn.utils_for_testing import comp_pcc

LAYER = 2
SEQ = 5120
HEAD_PCC = 0.999
ROWS = (SEQ - 1, 0, 31, 32, 1234, 1279, 1280, 2559, 2560, 3999)

MESHES = [
    pytest.param(
        shape,
        fabric2d_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=shape, topology=f"mesh-{shape[0]}x{shape[1]}"),
        id=f"fabric2d-mesh-{shape[0]}x{shape[1]}",
    )
    for shape in ((2, 4), (4, 2))
]


def _reference(source: str):
    if source == "real" and not (O.HF_SNAPSHOT / "model.safetensors.index.json").is_file():
        pytest.skip("V4.1 checkpoint shards not downloaded")
    spec = O.real_spec((LAYER,), SEQ, checkpoint=O.HF_SNAPSHOT if source == "real" else None)
    model = O.build_reference(spec)
    result = O.oracle(spec, O.random_tokens(spec), model)
    return model, result


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("weights_source", ["synthetic", "real"])
@pytest.mark.parametrize("mesh_device, device_params", MESHES + galaxy_meshes(), indirect=True)
def test_v41_embedding_head(mesh_device, device_params, weights_source):
    model, result = _reference(weights_source)
    shape = tuple(mesh_device.shape)
    concat = ttnn.ConcatMesh2dToTensor(mesh_device, shape, dims=(2, 3))

    # --- embedding: exact lookup
    tokens = result["tokens"]
    with torch.no_grad():
        expected_embed = model.embed(tokens[None].clone())[0]
    embed = TtV41Embedding(mesh_device, C, model.embed.weight.data)
    tt_ids = ttnn.from_torch(
        tokens.to(torch.int32).reshape(1, 1, SEQ),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(2, None)),
    )
    embeds = [ttnn.to_torch(embed(tt_ids), mesh_composer=concat).reshape(SEQ, C.EMB_SIZE) for _ in range(2)]
    assert torch.equal(embeds[0], embeds[1]), "embedding is not bit-identical across repeats"
    assert torch.equal(embeds[0], expected_embed), "embedding differs from the reference lookup"
    del embed

    # --- final norm + head on single rows
    block = result["blocks"][LAYER]
    with torch.no_grad(), v41.set_dtype(torch.bfloat16):
        hidden = model.layers[0].hc_pre(block["x_out"][None], block["pre_out"][None])  # [1, S, D] bf16
        rows = list(ROWS)
        # one row at a time, shaped exactly as the reference head sees its last position
        expected = torch.stack([model.head(model.norm(hidden[:, r : r + 1]))[0] for r in rows])  # fp32
    assert torch.equal(expected[0], result["logits"]), "reference head differs from the oracle's last-position logits"

    head = TtV41Head(mesh_device, C, model.norm.weight.data, model.head.weight.data.to(torch.bfloat16))
    tt_hidden = ttnn.from_torch(
        hidden[None],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(2, 3)),
    )
    report = {}
    for i, row in enumerate(rows):
        runs = [head.logits_to_host(*head(tt_hidden, row)) for _ in range(2 if i == 0 else 1)]
        if i == 0:
            assert torch.equal(runs[0], runs[1]), "head logits are not bit-identical across repeats"
        got, want = runs[0], expected[i]
        assert torch.isfinite(got).all(), f"row {row}: non-finite logits"
        report[row] = {
            "pcc": comp_pcc(want, got, 0.0)[1],
            "top1": (int(want.argmax()), int(got.argmax())),
            "max_abs_err": (want - got).abs().max().item(),
        }
    logger.info(f"V41_HEAD_RESULT mesh={list(shape)} weights={weights_source} {report}")
    for row, r in report.items():
        assert r["pcc"] >= HEAD_PCC, (row, r)
        assert r["top1"][0] == r["top1"][1], (row, r)
