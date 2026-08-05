# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Qwen3-VL vision tower: 2-D rotary, attention, MLP and block.
#
# The tower's rotary is spatial only -- `(row, col)` within one image, no
# temporal axis -- and so is a different grid from the decoder's 3-axis M-RoPE.
#
# `head_dim` is 1152 // 16 == 72, which is not tile-aligned. ttnn SDPA rejects it
# (`TT_FATAL logical_shape[3] == legacy_shape[3]`), so q/k/v/o are padded to 96 at
# load time and `scale` is passed explicitly as 72 ** -0.5. Leaving `scale` to
# SDPA would make it 96 ** -0.5 and silently change the softmax temperature --
# wrong output rather than a crash, which is why there is a test for it below.
#
# Scope: one attention block. `cu_seqlens = repeat_interleave(h*w, t).cumsum()`
# makes a single image exactly one block, which is all `fl2va` needs; multiple
# images and video frames need block-diagonal masking and are not covered yet.
# =============================================================================

import pytest
import torch
import transformers

import ttnn

from ....encoders.qwen3vl.vision_qwen3vl import (
    Qwen3VlVisionAttention,
    Qwen3VlVisionBlock,
    Qwen3VlVisionMLP,
    resolve_vision_parallel,
    vision_rope_position_ids,
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

# One image is one attention block. 48x48 is the 1:1 canvas; 4x6 keeps the CPU reference cheap.
GRIDS = {"small": [[1, 4, 6]], "square": [[1, 48, 48]]}


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


# --------------------------------------------------------------------------- host


@pytest.mark.parametrize("name", list(GRIDS))
def test_rope_position_ids_match_reference(name):
    """The `(row, col)` grid matches `transformers.vision_utils.get_vision_position_ids`."""
    from transformers.vision_utils import get_vision_position_ids

    grid = torch.tensor(GRIDS[name], dtype=torch.long)
    expected = get_vision_position_ids(grid, SPATIAL_MERGE_SIZE)
    actual = vision_rope_position_ids(grid, spatial_merge_size=SPATIAL_MERGE_SIZE)
    assert torch.equal(actual, expected), f"{name}: vision position ids differ"


@pytest.mark.parametrize("name", list(GRIDS))
def test_rope_tensors_match_reference(name):
    """`(cos, sin)` match the reference's rotary module."""
    grid = torch.tensor(GRIDS[name], dtype=torch.long)
    expected_cos, expected_sin = _golden_pos_embeds(grid)
    cos, sin = vision_rope_tensors(grid, head_dim=HEAD_DIM, spatial_merge_size=SPATIAL_MERGE_SIZE)

    total = sum(t * h * w for t, h, w in GRIDS[name])
    assert cos.shape == (total, HEAD_DIM), f"{tuple(cos.shape)} != {(total, HEAD_DIM)}"
    assert torch.allclose(cos, expected_cos, atol=1e-6), f"cos max diff {(cos - expected_cos).abs().max():.2e}"
    assert torch.allclose(sin, expected_sin, atol=1e-6), f"sin max diff {(sin - expected_sin).abs().max():.2e}"


def test_video_frames_repeat_the_spatial_grid():
    """A video's frames each carry the same `(row, col)` grid -- the tower has no temporal axis.

    Each frame is its own attention block, so position only has to be unique within a frame.
    """
    grid = torch.tensor([[3, 4, 6]], dtype=torch.long)
    ids = vision_rope_position_ids(grid, spatial_merge_size=SPATIAL_MERGE_SIZE)
    per_frame = 4 * 6
    assert ids.shape == (3 * per_frame, 2)
    assert torch.equal(ids[:per_frame], ids[per_frame : 2 * per_frame])
    assert torch.equal(ids[:per_frame], ids[2 * per_frame :])


def test_the_padded_head_dim_needs_an_explicit_scale():
    """72 pads to 96, and the two softmax temperatures differ -- hence the explicit `scale`.

    The padding channels are zero and contribute nothing to the dot products, so the temperature must
    stay that of the real 72. Leaving it to SDPA's default would use 96 and yield plausible but wrong
    attention rather than an error, so the arithmetic motivating the override is pinned here and the
    module's own `scale` is checked in `test_attention_on_device`.
    """
    import math

    assert HEAD_DIM == 72 and HEAD_DIM % 32 != 0, "the whole padding question presumes a misaligned 72"
    assert math.ceil(HEAD_DIM / 32) * 32 == PADDED_HEAD_DIM
    assert HEAD_DIM**-0.5 != pytest.approx(PADDED_HEAD_DIM**-0.5)


@pytest.mark.parametrize("axis", [0, 1], ids=["rows_qkv", "cols_proj"])
def test_head_dim_padding_is_a_zero_extension(axis):
    """Padding 72 -> 96 must zero-extend each head independently, not reshape across heads.

    A reshape that got the head stride wrong would still produce the right *shape*, so this checks the
    real channels survive in place and the tail is exactly zero.
    """
    from ....encoders.qwen3vl.vision_qwen3vl import _pad_head_dim

    torch.manual_seed(0)
    kw = dict(num_heads=NUM_HEADS, head_dim=HEAD_DIM, padded=PADDED_HEAD_DIM)
    if axis == 0:
        w = torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE)
        got = _pad_head_dim(w, axis=0, **kw)
        assert got.shape == (NUM_HEADS * PADDED_HEAD_DIM, HIDDEN_SIZE)
        got = got.reshape(NUM_HEADS, PADDED_HEAD_DIM, HIDDEN_SIZE)
        assert torch.equal(got[:, :HEAD_DIM], w.reshape(NUM_HEADS, HEAD_DIM, HIDDEN_SIZE))
        assert torch.all(got[:, HEAD_DIM:] == 0)
    else:
        w = torch.randn(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM)
        got = _pad_head_dim(w, axis=1, **kw)
        assert got.shape == (HIDDEN_SIZE, NUM_HEADS * PADDED_HEAD_DIM)
        got = got.reshape(HIDDEN_SIZE, NUM_HEADS, PADDED_HEAD_DIM)
        assert torch.equal(got[:, :, :HEAD_DIM], w.reshape(HIDDEN_SIZE, NUM_HEADS, HEAD_DIM))
        assert torch.all(got[:, :, HEAD_DIM:] == 0)


def test_head_dim_padding_is_a_noop_when_aligned():
    """An already-aligned head_dim must pass through untouched, so aligned models are unaffected."""
    from ....encoders.qwen3vl.vision_qwen3vl import _pad_head_dim

    w = torch.randn(NUM_HEADS * 128, HIDDEN_SIZE)
    got = _pad_head_dim(w, num_heads=NUM_HEADS, head_dim=128, padded=128, axis=0)
    assert got is w


# ------------------------------------------------------------------------- device

# `single` is the replicated reference. The parallel configs shard the tower itself: TP fractures the
# 16 heads (2/device at TP=8) and the MLP's intermediate; SP splits the patch rows and runs ring SDPA.
# TP and SP must occupy different mesh axes, so the 8x4 system covers TP=8 x SP=4.
#
# `device_params` is per-config: the CCL paths need FABRIC_1D, but requesting it on a 1x1 mesh has no
# remote ethernet partner and times out in router init.
_L1_SMALL = 32768
_FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": _L1_SMALL}
_NO_FABRIC = {"l1_small_size": _L1_SMALL}

_MESH = [
    pytest.param((1, 1), (1, 1), None, None, 1, _NO_FABRIC, id="single"),
    pytest.param((8, 4), (8, 4), 0, None, 2, _FABRIC, id="tp8"),
    pytest.param((8, 4), (8, 4), None, 1, 2, _FABRIC, id="sp4"),
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

    Ring SDPA requires `N_local_q % 32 == 0`, which is stricter than the merger's 4-row merge group.
    The `small` grid is 24 patches and cannot divide into 4 tile-aligned shards, so it is a skip rather
    than a failure -- it exists to keep the CPU reference cheap, not to exercise SP.
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
def test_mlp_on_device(reference, mesh_device, submesh_shape, tp_axis, sp_axis, num_links, name):
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    parallel = _parallel(submesh, tp_axis, sp_axis, num_links)
    total = sum(t * h * w for t, h, w in GRIDS[name])
    _skip_if_sp_misaligned(total, submesh, sp_axis)
    torch.manual_seed(0)
    x = torch.randn(total, HIDDEN_SIZE)
    with torch.no_grad():
        golden = reference.blocks[0].mlp(x).float()

    mlp = Qwen3VlVisionMLP(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        hidden_act=HIDDEN_ACT,
        mesh_device=submesh,
        parallel=parallel,
    )
    mlp.load_torch_state_dict(reference.blocks[0].mlp.state_dict())
    # SP leaves the result row-sharded (only the tower gathers), so rows are composed back here.
    out = mlp.forward(_sp_shard(x, submesh, sp_axis))
    assert_quality(golden, tensor.to_torch(out, mesh_axes=[sp_axis, None]), pcc=0.998)


@_PARAMS
@pytest.mark.parametrize("name", list(GRIDS))
def test_attention_on_device(reference, mesh_device, submesh_shape, tp_axis, sp_axis, num_links, name):
    """One image is one block, so the reference's `cu_seqlens = [0, h*w]` path is plain attention."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    grid = torch.tensor(GRIDS[name], dtype=torch.long)
    total = sum(t * h * w for t, h, w in GRIDS[name])
    _skip_if_sp_misaligned(total, submesh, sp_axis)

    torch.manual_seed(0)
    x = torch.randn(total, HIDDEN_SIZE)
    cos, sin = _golden_pos_embeds(grid)
    with torch.no_grad():
        golden = (
            reference.blocks[0]
            .attn(x, cu_seqlens=torch.tensor([0, total], dtype=torch.int32), position_embeddings=(cos, sin))
            .float()
        )

    attn = Qwen3VlVisionAttention(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        mesh_device=submesh,
        parallel=_parallel(submesh, tp_axis, sp_axis, num_links),
    )
    # the override that keeps the padding zeros from changing the softmax temperature
    assert attn.head_dim == HEAD_DIM and attn.padded_head_dim == PADDED_HEAD_DIM
    assert attn.scale == pytest.approx(HEAD_DIM**-0.5)
    attn.load_torch_state_dict(reference.blocks[0].attn.state_dict())
    tt_cos, tt_sin = vision_rope_tensors(grid, head_dim=HEAD_DIM, spatial_merge_size=SPATIAL_MERGE_SIZE)
    # cos/sin are per-row, so they shard alongside the hidden states under SP.
    out = attn.forward(
        _sp_shard(x, submesh, sp_axis),
        pos_embeds=(_sp_shard(tt_cos, submesh, sp_axis), _sp_shard(tt_sin, submesh, sp_axis)),
    )
    assert_quality(golden, tensor.to_torch(out, mesh_axes=[sp_axis, None]), pcc=0.99)


@_PARAMS
@pytest.mark.parametrize("name", list(GRIDS))
def test_block_on_device(reference, mesh_device, submesh_shape, tp_axis, sp_axis, num_links, name):
    """The whole pre-norm block: LayerNorm + attention + LayerNorm + MLP, both residuals."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    grid = torch.tensor(GRIDS[name], dtype=torch.long)
    total = sum(t * h * w for t, h, w in GRIDS[name])
    _skip_if_sp_misaligned(total, submesh, sp_axis)

    torch.manual_seed(0)
    x = torch.randn(total, HIDDEN_SIZE)
    cos, sin = _golden_pos_embeds(grid)
    with torch.no_grad():
        golden = reference.blocks[0](
            x, cu_seqlens=torch.tensor([0, total], dtype=torch.int32), position_embeddings=(cos, sin)
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
    )
    assert_quality(golden, tensor.to_torch(out, mesh_axes=[sp_axis, None]), pcc=0.99)
