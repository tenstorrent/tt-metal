# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Qwen3-VL vision transformer block, on device, against the HF reference.
#
# One test, parametrized over seven grids and three parallel configurations. It
# runs the whole pre-norm block -- LayerNorm, attention, residual, LayerNorm, MLP,
# residual -- so the submodules are covered through it rather than separately.
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
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor

HIDDEN_SIZE = 1152
NUM_HEADS = 16
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 72 -- deliberately not tile-aligned
PADDED_HEAD_DIM = 96
INTERMEDIATE_SIZE = 4304
SPATIAL_MERGE_SIZE = 2
NORM_EPS = 1e-6
HIDDEN_ACT = "gelu_pytorch_tanh"

# Real shapes, measured end to end through the checkpoint's own image processor. Two sizing rules feed
# this block, and they differ by 4x on the short edge:
#   - keyframes and reference VIDEOS go through `resolve_canvas_size` -- 768 px short edge, area capped
#     at 768x1344 -- so 48 patches on the short edge;
#   - reference IMAGES go through `resolve_reference_image_size` -- 2048 px short edge, NO area cap --
#     so 128 patches on the short edge, and up to 512 on the long one at 4:1.
#
# The block sees a flat row sequence plus per-row cos/sin, so the aspect ratio changes the rope *values*
# but not the arithmetic path; the 14-way aspect sweep therefore lives in the tower test, where the
# position table and the merge reshape actually consume `h` and `w`. What does matter here is sequence
# LENGTH and BLOCK STRUCTURE: `cu_seqlens` is passed straight through to the attention, which loops one
# SDPA per block, so the number and sizes of blocks change control flow.
GRIDS = {
    "canvas_1to1": [[1, 48, 48]],  # 2304 rows, 1 block -- cheapest real shape
    "ref_1to1": [[1, 128, 128]],  # 16384, 1 block -- reference-image sizing
    "ref_4to1": [[1, 128, 512]],  # 65536, 1 block -- longest single block
    "video_3_frames": [[3, 48, 48]],  # 6912, 3 blocks from ONE grid row
    "two_refs": [[1, 128, 128], [1, 128, 170]],  # 38144, 2 blocks of unequal length
    "image_and_video": [[1, 128, 128], [3, 48, 48]],  # 23296, 4 blocks, both sizing rules
    # The documented ceiling: nine reference images and three reference videos. 18 blocks, and the
    # longest sequence the block will ever see. Under `single` nothing is sharded, so the MLP
    # intermediate alone is 168192 x 4304 x 2 B ~= 1.45 GiB in one tensor -- this is the case that
    # tests capacity and dispatch rather than arithmetic.
    "max_load": [[1, 128, 128]] * 9 + [[3, 48, 48]] * 3,  # 168192, 18 blocks
}


def _config(depth=1):
    return transformers.Qwen3VLVisionConfig(
        depth=depth,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        in_channels=3,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        num_position_embeddings=2304,
        out_hidden_size=5120,
        hidden_act=HIDDEN_ACT,
        deepstack_visual_indexes=[0],
        initializer_range=0.02,
    )


@pytest.fixture(scope="module")
def reference():
    torch.manual_seed(0)
    return transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionModel._from_config(_config()).eval()


def _golden_pos_embeds(grid):
    """`(cos, sin)` exactly as the reference tower builds them."""
    from transformers.vision_utils import get_vision_position_ids

    position_ids = get_vision_position_ids(grid, SPATIAL_MERGE_SIZE)
    rot = transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionRotaryEmbedding(HEAD_DIM // 2)
    freqs = rot(position_ids)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


# ------------------------------------------------------------------------- device

# `single` is the replicated reference. The parallel configs shard the block itself: TP fractures the
# 16 heads (2/device at TP=8) and the MLP's intermediate; SP splits the patch rows and runs ring SDPA.
# TP and SP must occupy different mesh axes, so the 8x4 system covers TP=8 x SP=4.
#
# Only TP=8 is deployed, with SP either off or 4, so the configs are named for both factors. SP alone
# (TP=1) is not a configuration this model will ever run in and is not covered.
#
# `device_params` is per-config: the CCL paths need FABRIC_1D, but requesting it on a 1x1 mesh has no
# remote ethernet partner and times out in router init.
_L1_SMALL = 32768
_FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": _L1_SMALL}
_NO_FABRIC = {"l1_small_size": _L1_SMALL}

_MESH = [
    pytest.param((1, 1), (1, 1), None, None, 1, _NO_FABRIC, id="single"),
    pytest.param((8, 4), (8, 4), 0, None, 2, _FABRIC, id="tp8_sp1"),
    pytest.param((8, 4), (8, 4), 0, 1, 2, _FABRIC, id="tp8_sp4"),
]
_PARAMS = pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis", "sp_axis", "num_links", "device_params"),
    _MESH,
    indirect=["mesh_device", "device_params"],
)


def _parallel(submesh, tp_axis, sp_axis, num_links):
    """The resolved `VisionParallel` these submodules take, or `None` when fully replicated."""
    if tp_axis is None and sp_axis is None:
        return None
    shape = tuple(submesh.shape)
    cfg = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=shape[tp_axis] if tp_axis is not None else 1, mesh_axis=tp_axis),
        sequence_parallel=(ParallelFactor(factor=shape[sp_axis], mesh_axis=sp_axis) if sp_axis is not None else None),
    )
    ccl = CCLManager(submesh, num_links=num_links, topology=ttnn.Topology.Linear)
    return resolve_vision_parallel(submesh, cfg, ccl)


def _skip_if_sp_misaligned(total, submesh, sp_axis):
    """SP needs a tile-aligned per-device row count.

    Ring SDPA requires `N_local_q % 32 == 0`, which at SP=4 means a multiple of 128 patches. Every grid
    here satisfies that, so nothing currently skips -- but the area-capped canvases do not (4032 is
    31.5 x 128), so this fires as soon as one of those is added. A misaligned grid is a skip, not a
    failure: it says the shape cannot be sequence-parallel at this factor, which is a property of the
    shape rather than of the port.
    """
    if sp_axis is None:
        return
    sp = tuple(submesh.shape)[sp_axis]
    if total % (sp * 32) != 0:
        pytest.skip(f"{total} patches do not divide into {sp} tile-aligned shards (needs a multiple of {sp * 32})")


def _sp_shard(x, submesh, sp_axis):
    """Upload row-sharded on the SP axis (replicated when SP is off)."""
    if sp_axis is None:
        return bf16_tensor(x, device=submesh)
    return bf16_tensor(x, device=submesh, mesh_axis=sp_axis, shard_dim=0)


@_PARAMS
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
    _skip_if_sp_misaligned(total, submesh, sp_axis)

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
        _sp_shard(x, submesh, sp_axis),
        pos_embeds=(_sp_shard(tt_cos, submesh, sp_axis), _sp_shard(tt_sin, submesh, sp_axis)),
        cu_seqlens=cu_seqlens,
    )
    assert_quality(golden, tensor.to_torch(out, mesh_axes=[sp_axis, None]), pcc=0.99)
