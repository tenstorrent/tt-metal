# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TT vision tower and projector against the torch reference on one Blackhole P100."""

from dataclasses import replace

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.alpamayo2_edge.reference import golden_cache
from models.experimental.alpamayo2_edge.reference.inputs import prepare_sample_inputs, synthetic_sample
from models.experimental.alpamayo2_edge.reference.vision import VisionTower, vision_cu_seqlens
from models.experimental.alpamayo2_edge.reference.weights import load_weights
from models.experimental.alpamayo2_edge.tests.conftest import P100_MESH
from models.experimental.alpamayo2_edge.tt.vision import PosEmbedCache, TtProjector, TtVisionTower, identity_rope
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality

PCC = 0.99
# Two images whose windows are not tile-aligned: a 20x36 grid and a 14x26 grid.
BLOCK_ROWS = 1084
BLOCK_CU_SEQLENS = [0, 720, BLOCK_ROWS]
ASPECTS = {"16x9": (1080, 1920), "4x3": (1080, 1440)}


@pytest.fixture(scope="module")
def visual_weights(model_dir):
    return load_weights(model_dir, "vlm.model.visual.")


@pytest.fixture(scope="module")
def projector_weights(model_dir):
    return load_weights(model_dir, "vlm.model.projector.")


def _pcc(golden: torch.Tensor, actual: torch.Tensor) -> float:
    stacked = torch.stack([golden.flatten().double(), actual.flatten().double()])
    return torch.corrcoef(stacked)[0, 1].item()


def _tt_tower(vcfg, pcfg, visual_weights, projector_weights, mesh_device) -> TtVisionTower:
    tower = TtVisionTower(vcfg, pcfg, mesh_device=mesh_device)
    tower.load_torch_state_dict(TtVisionTower.convert_state(visual_weights, projector_weights))
    return tower


@P100_MESH
def test_vision_block(mesh_device, cfg, visual_weights, projector_weights):
    vcfg = replace(cfg.vision, layers=1)
    reference = VisionTower(vcfg, cfg.projector, visual_weights, projector_weights)
    torch.manual_seed(0)
    x = torch.randn(BLOCK_ROWS, vcfg.hidden)
    with torch.no_grad():
        golden = reference.blocks[0](x, BLOCK_CU_SEQLENS)

    tower = _tt_tower(vcfg, cfg.projector, visual_weights, projector_weights, mesh_device)
    out = tower.blocks[0].forward(
        tensor.bf16_tensor(x, device=mesh_device),
        pos_embeds=identity_rope(BLOCK_ROWS, mesh_device=mesh_device),
        cu_seqlens=BLOCK_CU_SEQLENS,
    )
    assert_quality(golden, tensor.to_torch(out), pcc=PCC)


@P100_MESH
def test_projector(mesh_device, cfg, visual_weights, projector_weights):
    reference = VisionTower(replace(cfg.vision, layers=0), cfg.projector, visual_weights, projector_weights)
    torch.manual_seed(0)
    x = torch.randn(BLOCK_ROWS, cfg.vision.hidden)
    with torch.no_grad():
        golden = reference.project(x)

    projector = TtProjector(cfg.vision, cfg.projector, mesh_device=mesh_device)
    projector.load_torch_state_dict(projector_weights)
    out = tensor.to_torch(projector.forward(tensor.bf16_tensor(x, device=mesh_device)))
    assert out.shape == (BLOCK_ROWS // cfg.vision.merge**2, cfg.projector.out)
    assert_quality(golden, out, pcc=PCC)


def _reference_trace(cfg, visual_weights, projector_weights, patches, grid) -> dict:
    """The reference output, plus every block's output in bf16 for the per-block PCC log."""
    reference = VisionTower(cfg.vision, cfg.projector, visual_weights, projector_weights)
    blocks = []

    def capture(index):
        def hook(_module, _args, output):
            blocks.append(output.bfloat16())
            print(f"reference block {index + 1}/{len(reference.blocks)}", flush=True)

        return hook

    for i, block in enumerate(reference.blocks):
        block.register_forward_hook(capture(i))
    with torch.no_grad():
        out = reference(patches, grid)
    return {"blocks": blocks, "out": out}


@P100_MESH
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("aspect", list(ASPECTS))
def test_vision_tower(mesh_device, cfg, visual_weights, projector_weights, aspect):
    inputs = prepare_sample_inputs(cfg, synthetic_sample(0, src_hw=ASPECTS[aspect]))
    patches, grid = inputs.patches, inputs.image_grid_thw
    golden = golden_cache.cached(
        f"vision_tower_{aspect}",
        {"patches": patches, "grid": grid},
        lambda: _reference_trace(cfg, visual_weights, projector_weights, patches, grid),
        cfg=cfg,
    )
    n = patches.shape[0]
    cu_seqlens = vision_cu_seqlens(grid)
    logger.info(f"{aspect}: grids {sorted(set(map(tuple, grid.tolist())))}, {n} patches, {len(cu_seqlens) - 1} images")

    tower = _tt_tower(cfg.vision, cfg.projector, visual_weights, projector_weights, mesh_device)
    pos_cache = PosEmbedCache(
        visual_weights["embeddings.position_embedding.weight"], merge=cfg.vision.merge, mesh_device=mesh_device
    )
    tt_patches = tensor.bf16_tensor(patches, device=mesh_device)
    tt_pos = pos_cache.get(grid)
    assert pos_cache.get(grid) is tt_pos, "position embeddings must be uploaded once per grid"

    # Block by block first, for the per-block PCC log; the assertion is on `forward` below.
    x = tower.embed(tt_patches, tt_pos)
    rope = identity_rope(n, mesh_device=mesh_device)
    for i, block in enumerate(tower.blocks):
        x = block.forward(x, pos_embeds=rope, cu_seqlens=cu_seqlens)
        logger.info(f"{aspect} block {i:2d} PCC {_pcc(golden['blocks'][i], tensor.to_torch(x)) * 100:.4f} %")
    ttnn.deallocate(x)

    out = tensor.to_torch(tower.forward(tt_patches, tt_pos, cu_seqlens))
    assert out.shape == golden["out"].shape
    logger.info(f"{aspect} tower+projector PCC {_pcc(golden['out'], out) * 100:.4f} %")
    assert_quality(golden["out"], out, pcc=PCC)
