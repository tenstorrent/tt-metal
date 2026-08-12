# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Qwen3-VL vision transformer block, on device, against the HF reference.
#
# One test, parametrized over two grids and three parallel configurations. It
# runs the whole pre-norm block -- LayerNorm, attention, residual, LayerNorm, MLP,
# residual -- so the submodules are covered through it rather than separately.
# This file exists for block-level debugging granularity; the shape/aspect sweep,
# the capacity grids and the perf mode all live in test_qwen3vl_vision_tower.py.
#
# The tower's rotary is spatial only: `(row, col)` within one image, no temporal
# axis, and so a different grid from the decoder's 3-axis M-RoPE.
#
# `head_dim` is 1152 // 16 == 72, which is not tile-aligned. ttnn SDPA rejects it
# (`TT_FATAL logical_shape[3] == legacy_shape[3]`), so q/k/v/o are zero-padded to
# 96 at load time and `scale` is passed explicitly as 72 ** -0.5. Letting SDPA
# default it would use 96 and silently change the softmax temperature -- wrong
# output rather than a crash -- so that contract is asserted inside the test.
#
# `cu_seqlens` IS a block-level concern: the block hands it to the attention, which
# loops one SDPA per image and per video frame. The multi-block grids below cover
# that here, where the loop lives, rather than only through the tower. The aspect
# sweep is the tower's job -- see test_qwen3vl_vision_tower.py -- because the
# position table and the merge reshape are what actually consume `h` and `w`.
# =============================================================================

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

PADDED_HEAD_DIM = 96

# Real shapes, measured end to end through the checkpoint's own image processor. Two sizing rules feed
# this block, and they differ by 4x on the short edge:
#   - keyframes and reference VIDEOS go through `resolve_canvas_size` -- 768 px short edge, area capped
#     at 768x1344 -- so 48 patches on the short edge;
#   - reference IMAGES go through `resolve_reference_image_size` -- 2048 px short edge, NO area cap --
#     so 128 patches on the short edge, and up to 512 on the long one at 4:1.
#
# The block sees a flat row sequence plus per-row cos/sin, so the aspect ratio changes the rope *values*
# but not the arithmetic path; the aspect sweep therefore lives in the tower test, where the
# position table and the merge reshape actually consume `h` and `w`. What does matter here is sequence
# LENGTH and BLOCK STRUCTURE: `cu_seqlens` is passed straight through to the attention, which loops one
# SDPA per block, so the number and sizes of blocks change control flow.
#
# Two grids cover the block's distinct code paths -- the padded-head-dim scale runs on every grid, so
# what differentiates them is the cu_seqlens loop:
#   - `canvas_1to1`: the cheapest real shape, one block, the single-SDPA baseline;
#   - `image_and_video`: multiple blocks of unequal length from BOTH sizing rules (one reference image
#     plus three video frames), so a block that ignored the boundaries fails here.
# The other real shapes -- the aspect pairs, the 65536-row single blocks and the 18-block `max_load`
# ceiling -- run through test_qwen3vl_vision_tower.py, where capacity and perf belong.
GRIDS = {
    "canvas_1to1": [[1, 48, 48]],  # 2304 rows, 1 block -- cheapest real shape
    "image_and_video": [[1, 128, 128], [3, 48, 48]],  # 23296, 4 blocks, both sizing rules
}


@pytest.fixture(scope="module")
def reference():
    torch.manual_seed(0)
    return transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionModel._from_config(vision_config(depth=1)).eval()


def _golden_pos_embeds(grid):
    """`(cos, sin)` exactly as the reference tower builds them."""
    from transformers.vision_utils import get_vision_position_ids

    position_ids = get_vision_position_ids(grid, SPATIAL_MERGE_SIZE)
    rot = transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionRotaryEmbedding(HEAD_DIM // 2)
    freqs = rot(position_ids)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


# ------------------------------------------------------------------------- device

# Mesh/parallel configs are the shared `VISION_PARAMS` from common.py.
#
# Every case here checks PCC against the CPU golden; both grids are small enough that the quadratic
# CPU attention is affordable. The `perf` mode (no golden, shape/finiteness only, for the capacity
# grids and `python -m tracy`) lives in test_qwen3vl_vision_tower.py.


def _parallel(submesh, tp_axis, sp_axis, num_links):
    """The resolved `VisionParallel` these submodules take, or `None` when fully replicated."""
    cfg, ccl = resolve_parallel(submesh, tp_axis, sp_axis, num_links)
    if cfg is None:
        return None
    return resolve_vision_parallel(submesh, cfg, ccl)


@VISION_PARAMS
@pytest.mark.parametrize("name", list(GRIDS))
def test_block_on_device(reference, mesh_device, submesh_shape, tp_axis, sp_axis, num_links, name):
    """The whole pre-norm block: LayerNorm + attention + LayerNorm + MLP, both residuals.

    `cu_seqlens` is passed on both sides, so the multi-block grids exercise the per-block SDPA loop in
    `Qwen3VlVisionAttention.forward` at the level it lives rather than only through the tower. Attention
    must not cross from one image or video frame into the next; a block ignoring the boundaries would
    still match on the single-block grids.

    Also pins the padded-head-dim contract, which has no separate test: `head_dim` 72 is not
    tile-aligned, q/k/v/o are zero-padded to 96 at load time, and `scale` must stay `72 ** -0.5`.
    Letting SDPA default it would use 96 and silently change the softmax temperature -- wrong output
    rather than a crash, so it is checked here where the padded attention actually runs.
    """
    import math

    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    grid = torch.tensor(GRIDS[name], dtype=torch.long)
    total = sum(t * h * w for t, h, w in GRIDS[name])
    skip_if_sp_misaligned(total, submesh, sp_axis)

    assert HEAD_DIM == 72 and HEAD_DIM % 32 != 0, "the whole padding question presumes a misaligned 72"
    assert math.ceil(HEAD_DIM / 32) * 32 == PADDED_HEAD_DIM
    assert HEAD_DIM**-0.5 != pytest.approx(PADDED_HEAD_DIM**-0.5), "the two temperatures must differ"

    # One block per FRAME, so the count is sum(t), not the number of grid rows.
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
