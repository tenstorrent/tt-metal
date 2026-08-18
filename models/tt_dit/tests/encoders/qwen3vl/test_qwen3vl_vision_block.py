# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Qwen3-VL vision transformer block on device against the HF reference; the
# aspect sweep, capacity grids and perf mode live in test_qwen3vl_vision_tower.py.

import pytest
import torch
import transformers

import ttnn

from ....encoders.qwen3vl.vision_qwen3vl import (
    Qwen3VlVisionBlock,
    resolve_vision_parallel,
    vision_cu_seqlens,
    vision_rope_tensors,
)
from ....utils import tensor
from ....utils.check import assert_quality
from .common import (
    HEAD_DIM,
    HIDDEN_ACT,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    NORM_EPS,
    NUM_HEADS,
    SPATIAL_MERGE_SIZE,
    VISION_PARAMS,
    resolve_parallel,
    skip_if_sp_misaligned,
    sp_shard,
    vision_config,
)

PADDED_HEAD_DIM = 96  # head_dim 72 is not tile-aligned; q/k/v/o zero-pad to 96, scale must stay 72**-0.5

GRIDS = {
    "canvas_1to1": [[1, 48, 48]],  # 2304 rows, 1 block -- cheapest real shape
    "image_and_video": [[1, 128, 128], [3, 48, 48]],  # 23296, 4 blocks, both sizing rules
}


@pytest.fixture(scope="module")
def reference():
    torch.manual_seed(0)
    return transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionModel._from_config(vision_config(depth=1)).eval()


def _golden_pos_embeds(grid):
    from transformers.vision_utils import get_vision_position_ids

    position_ids = get_vision_position_ids(grid, SPATIAL_MERGE_SIZE)
    rot = transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionRotaryEmbedding(HEAD_DIM // 2)
    freqs = rot(position_ids)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _parallel(submesh, tp_axis, sp_axis, num_links):
    cfg, ccl = resolve_parallel(submesh, tp_axis, sp_axis, num_links)
    if cfg is None:
        return None
    return resolve_vision_parallel(submesh, cfg, ccl)


@VISION_PARAMS
@pytest.mark.parametrize("name", list(GRIDS))
def test_block_on_device(reference, mesh_device, submesh_shape, tp_axis, sp_axis, num_links, name):
    """The whole pre-norm block; multi-block grids exercise the per-frame SDPA loop and the padded-head-dim scale."""
    import math

    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    grid = torch.tensor(GRIDS[name], dtype=torch.long)
    total = sum(t * h * w for t, h, w in GRIDS[name])
    skip_if_sp_misaligned(total, submesh, sp_axis)

    assert HEAD_DIM == 72 and HEAD_DIM % 32 != 0, "the whole padding question presumes a misaligned 72"
    assert math.ceil(HEAD_DIM / 32) * 32 == PADDED_HEAD_DIM
    assert HEAD_DIM**-0.5 != pytest.approx(PADDED_HEAD_DIM**-0.5), "the two temperatures must differ"

    cu_seqlens = vision_cu_seqlens(grid)
    assert len(cu_seqlens) - 1 == int(grid[:, 0].sum()), f"expected one block per frame, got {cu_seqlens}"
    assert cu_seqlens[0] == 0 and cu_seqlens[-1] == total, f"cu_seqlens must span [0, {total}]: {cu_seqlens}"

    torch.manual_seed(0)
    x = torch.randn(total, HIDDEN_SIZE)
    cos, sin = _golden_pos_embeds(grid)
    with torch.no_grad():
        golden = reference.blocks[0](
            x,
            cu_seqlens=torch.tensor(list(cu_seqlens), dtype=torch.int32),
            position_embeddings=(cos, sin),
        ).float()

    block = Qwen3VlVisionBlock(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        hidden_act=HIDDEN_ACT,
        norm_eps=NORM_EPS,
        mesh_device=submesh,
        parallel=_parallel(submesh, tp_axis, sp_axis, num_links),
    )
    block.load_torch_state_dict(reference.blocks[0].state_dict())
    tt_cos, tt_sin = vision_rope_tensors(grid, head_dim=HEAD_DIM, spatial_merge_size=SPATIAL_MERGE_SIZE)
    out = block.forward(
        sp_shard(x, submesh, sp_axis),
        pos_embeds=(sp_shard(tt_cos, submesh, sp_axis), sp_shard(tt_sin, submesh, sp_axis)),
        cu_seqlens=cu_seqlens,
    )
    actual = tensor.to_torch(out, mesh_axes=[sp_axis, None])

    assert_quality(golden, actual, pcc=0.99)
