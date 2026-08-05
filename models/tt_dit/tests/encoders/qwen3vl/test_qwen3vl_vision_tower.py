# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Qwen3-VL vision tower: patch mergers and the assembled tower.
#
# The tower returns two things the decoder needs, and both are checked here:
#   - the merged output tokens (the reference's `pooler_output`), which the
#     decoder scatters into the text sequence at `<|image_pad|>` positions;
#   - one deepstack feature per entry of `deepstack_visual_indexes`, which the
#     decoder adds into its first few layers.
#
# `use_postshuffle_norm` is the only structural difference between the output
# merger and the deepstack mergers, and it is not just where the reshape sits:
# pre-shuffle normalizes each patch over `hidden_size`, post-shuffle normalizes
# the concatenated group of four over `hidden_size * merge ** 2`. Different
# statistics, so a test pins that they disagree.
#
# Depth is reduced for the device tests -- 27 layers of a 4032-patch keyframe is
# a long CPU reference -- but `deepstack_visual_indexes` is exercised at more than
# one layer so the routing is real.
# =============================================================================

import pytest
import torch
import transformers

import ttnn

from ....encoders.qwen3vl.vision_qwen3vl import Qwen3VlVisionModel, Qwen3VlVisionPatchMerger, resolve_vision_parallel
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor

HIDDEN_SIZE = 1152
NUM_HEADS = 16
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
INTERMEDIATE_SIZE = 4304
SPATIAL_MERGE_SIZE = 2
MERGED_SIZE = HIDDEN_SIZE * SPATIAL_MERGE_SIZE**2  # 4608
OUT_HIDDEN_SIZE = 5120
NUM_POSITION_EMBEDDINGS = 2304
NORM_EPS = 1e-6
HIDDEN_ACT = "gelu_pytorch_tanh"

# Shallow enough for a CPU reference, deep enough that deepstack routing is non-trivial.
DEPTH = 6
DEEPSTACK_INDEXES = [1, 3]

GRIDS = {"small": [[1, 4, 6]], "square": [[1, 48, 48]]}


def _config():
    return transformers.Qwen3VLVisionConfig(
        depth=DEPTH,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        in_channels=3,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        num_position_embeddings=NUM_POSITION_EMBEDDINGS,
        out_hidden_size=OUT_HIDDEN_SIZE,
        hidden_act=HIDDEN_ACT,
        deepstack_visual_indexes=DEEPSTACK_INDEXES,
        initializer_range=0.02,
    )


@pytest.fixture(scope="module")
def reference():
    torch.manual_seed(0)
    return transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionModel._from_config(_config()).eval()


def _tower(reference, submesh, parallel_config=None, ccl_manager=None):
    tower = Qwen3VlVisionModel(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        depth=DEPTH,
        intermediate_size=INTERMEDIATE_SIZE,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        num_position_embeddings=NUM_POSITION_EMBEDDINGS,
        out_hidden_size=OUT_HIDDEN_SIZE,
        hidden_act=HIDDEN_ACT,
        norm_eps=NORM_EPS,
        deepstack_visual_indexes=DEEPSTACK_INDEXES,
        mesh_device=submesh,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    tower.load_torch_state_dict(reference.state_dict())
    return tower


# --------------------------------------------------------------------------- host


def test_postshuffle_norm_normalizes_over_a_different_width(reference):
    """The two merger variants differ *only* in the norm, and that norm has a different width.

    Pre-shuffle normalizes each patch over 1152; post-shuffle normalizes the concatenated group of four
    over 4608. The two therefore cannot share weights at all -- which is a stronger statement than
    "their outputs differ" and is why `use_postshuffle_norm` has to be plumbed rather than inferred.
    """
    pre, post = reference.merger, reference.deepstack_merger_list[0]
    assert pre.norm.normalized_shape == (HIDDEN_SIZE,)
    assert post.norm.normalized_shape == (MERGED_SIZE,)
    # everything else is identical, so the norm is the whole difference
    for name in ("linear_fc1", "linear_fc2"):
        assert getattr(pre, name).weight.shape == getattr(post, name).weight.shape


def test_norm_placement_changes_the_statistics():
    """Normalizing before vs after the merge reshape gives different values, not just a different order.

    Isolated from the learned weights: affine is off, so only the statistic being computed can differ.
    A port that put the reshape on the wrong side of the norm would still produce correct shapes.
    """
    torch.manual_seed(0)
    x = torch.randn(4 * 8, HIDDEN_SIZE)
    pre_then_reshape = torch.nn.functional.layer_norm(x, (HIDDEN_SIZE,), eps=NORM_EPS).reshape(-1, MERGED_SIZE)
    reshape_then_post = torch.nn.functional.layer_norm(x.reshape(-1, MERGED_SIZE), (MERGED_SIZE,), eps=NORM_EPS)
    assert pre_then_reshape.shape == reshape_then_post.shape
    assert not torch.allclose(pre_then_reshape, reshape_then_post, atol=1e-3)


def test_pos_embed_weight_is_kept_on_the_host(reference):
    """The position table is popped from the state dict rather than pushed to the device.

    It is only read by host arithmetic, so a strict load must not report it as unexpected.
    """
    tower = Qwen3VlVisionModel.__new__(Qwen3VlVisionModel)
    tower._pos_embed_weight = None
    state = {"pos_embed.weight": reference.pos_embed.weight.detach().clone(), "other.weight": torch.zeros(2, 2)}
    Qwen3VlVisionModel._prepare_torch_state(tower, state)
    assert "pos_embed.weight" not in state, "pos_embed must not reach the device"
    assert tower._pos_embed_weight.shape == (NUM_POSITION_EMBEDDINGS, HIDDEN_SIZE)


def test_prepare_pos_embeds_needs_loaded_weights(expect_error):
    """Asking for position embeddings before loading is an error, not silent zeros."""
    tower = Qwen3VlVisionModel.__new__(Qwen3VlVisionModel)
    tower._pos_embed_weight = None
    with expect_error(ValueError, "call load_torch_state_dict first"):
        Qwen3VlVisionModel.prepare_pos_embeds(tower, torch.tensor([[1, 4, 6]]))


# ------------------------------------------------------------------------- device

# TP fractures heads/intermediate/merger; SP splits patch rows and ring-attends. Different axes, so
# the 8x4 system gives TP=8 x SP=4. `device_params` is per-config: FABRIC_1D has no ethernet partner
# on a 1x1 mesh and times out in router init.
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


def _parallel_args(submesh, tp_axis, sp_axis, num_links):
    """`(parallel_config, ccl_manager)` for `Qwen3VlVisionModel`, or `(None, None)` when replicated."""
    if tp_axis is None and sp_axis is None:
        return None, None
    shape = tuple(submesh.shape)
    cfg = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=shape[tp_axis] if tp_axis is not None else 1, mesh_axis=tp_axis),
        sequence_parallel=(ParallelFactor(factor=shape[sp_axis], mesh_axis=sp_axis) if sp_axis is not None else None),
    )
    return cfg, CCLManager(submesh, num_links=num_links, topology=ttnn.Topology.Linear)


def _skip_if_sp_misaligned(total, submesh, sp_axis):
    """Ring SDPA needs `N_local_q % 32 == 0`, stricter than the merger's 4-row merge group.

    The `small` grid (24 patches) cannot divide into 4 tile-aligned shards; it exists to keep the CPU
    reference cheap, so it is a skip rather than a failure.
    """
    if sp_axis is None:
        return
    sp = tuple(submesh.shape)[sp_axis]
    if total % (sp * 32) != 0:
        pytest.skip(f"{total} patches do not divide into {sp} tile-aligned shards (needs a multiple of {sp * 32})")


def _shard(x, submesh, sp_axis):
    """Upload row-sharded on the SP axis; the tower gathers its own output, so only inputs shard."""
    if sp_axis is None:
        return bf16_tensor(x, device=submesh)
    return bf16_tensor(x, device=submesh, mesh_axis=sp_axis, shard_dim=0)


@_PARAMS
@pytest.mark.parametrize("postshuffle", [False, True], ids=["output_merger", "deepstack_merger"])
def test_merger_on_device(reference, mesh_device, submesh_shape, tp_axis, sp_axis, num_links, postshuffle):
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    parallel = (
        None
        if (tp_axis is None and sp_axis is None)
        else resolve_vision_parallel(submesh, *_parallel_args(submesh, tp_axis, sp_axis, num_links))
    )
    ref = reference.deepstack_merger_list[0] if postshuffle else reference.merger

    torch.manual_seed(0)
    x = torch.randn(4 * 16, HIDDEN_SIZE)
    with torch.no_grad():
        golden = ref(x).float()

    merger = Qwen3VlVisionPatchMerger(
        hidden_size=HIDDEN_SIZE,
        out_hidden_size=OUT_HIDDEN_SIZE,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        norm_eps=NORM_EPS,
        use_postshuffle_norm=postshuffle,
        mesh_device=submesh,
        parallel=parallel,
    )
    merger.load_torch_state_dict(ref.state_dict())
    # The merger does not gather across SP (only the tower does), so rows compose back here.
    out = merger.forward(_shard(x, submesh, sp_axis))
    actual = tensor.to_torch(out, mesh_axes=[sp_axis, None])

    assert actual.shape[-2:] == (16, OUT_HIDDEN_SIZE), f"{tuple(actual.shape)}"
    assert_quality(golden, actual, pcc=0.99)


@_PARAMS
@pytest.mark.parametrize("name", list(GRIDS))
def test_tower_on_device(reference, mesh_device, submesh_shape, tp_axis, sp_axis, num_links, name):
    """The full tower: merged output tokens and every deepstack feature."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    grid = torch.tensor(GRIDS[name], dtype=torch.long)
    total = sum(t * h * w for t, h, w in GRIDS[name])
    _skip_if_sp_misaligned(total, submesh, sp_axis)
    patch_dim = 3 * 2 * 16 * 16

    torch.manual_seed(0)
    patches = torch.randn(total, patch_dim)
    with torch.no_grad():
        ref_out = reference(patches, grid_thw=grid, return_dict=True)
    golden_tokens = ref_out.pooler_output.float()
    golden_deepstack = [f.float() for f in ref_out.deepstack_features]
    assert len(golden_deepstack) == len(DEEPSTACK_INDEXES), "reference did not emit one feature per index"

    tower = _tower(reference, submesh, *_parallel_args(submesh, tp_axis, sp_axis, num_links))
    cos, sin = tower.prepare_rope(grid)
    pos = tower.prepare_pos_embeds(grid)
    # Inputs shard on rows under SP; the tower's outputs come back gathered and replicated.
    tokens, deepstack = tower.forward(
        _shard(patches, submesh, sp_axis),
        pos_embeds=_shard(pos, submesh, sp_axis),
        rope=(_shard(cos, submesh, sp_axis), _shard(sin, submesh, sp_axis)),
    )

    merged = total // SPATIAL_MERGE_SIZE**2
    actual_tokens = tensor.to_torch(tokens, mesh_axes=[None, None])
    assert actual_tokens.shape[-2:] == (merged, OUT_HIDDEN_SIZE), f"{tuple(actual_tokens.shape)}"
    assert_quality(golden_tokens, actual_tokens, pcc=0.99)

    assert len(deepstack) == len(DEEPSTACK_INDEXES), f"expected {len(DEEPSTACK_INDEXES)} features"
    for i, (feature, golden_feature) in enumerate(zip(deepstack, golden_deepstack)):
        actual = tensor.to_torch(feature, mesh_axes=[None, None])
        assert actual.shape[-2:] == (merged, OUT_HIDDEN_SIZE), f"deepstack {i}: {tuple(actual.shape)}"
        assert_quality(golden_feature, actual, pcc=0.99)


@_PARAMS
def test_deepstack_features_are_taken_from_distinct_layers(
    reference, mesh_device, submesh_shape, tp_axis, sp_axis, num_links
):
    """Each deepstack feature must come from its own layer, not the same one twice.

    Routing by list index is easy to get wrong in a way that still yields the right count and shapes,
    so this checks the features actually differ.
    """
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    grid = torch.tensor(GRIDS["small"], dtype=torch.long)
    total = sum(t * h * w for t, h, w in GRIDS["small"])
    _skip_if_sp_misaligned(total, submesh, sp_axis)

    torch.manual_seed(0)
    tower = _tower(reference, submesh, *_parallel_args(submesh, tp_axis, sp_axis, num_links))
    cos, sin = tower.prepare_rope(grid)
    pos = tower.prepare_pos_embeds(grid)
    _, deepstack = tower.forward(
        _shard(torch.randn(total, 3 * 2 * 16 * 16), submesh, sp_axis),
        pos_embeds=_shard(pos, submesh, sp_axis),
        rope=(_shard(cos, submesh, sp_axis), _shard(sin, submesh, sp_axis)),
    )
    a, b = (tensor.to_torch(f, mesh_axes=[None, None]) for f in deepstack)
    assert not torch.allclose(a, b, atol=1e-2), "deepstack features are identical; layer routing is wrong"
