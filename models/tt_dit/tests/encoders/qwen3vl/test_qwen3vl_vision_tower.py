# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Qwen3-VL vision tower end to end, on device, against the HF reference.
#
# One test, parametrized over eighteen grids and three parallel configurations. It runs
# the assembled tower -- patch embed, interpolated position embeddings, every
# block, the output merger and the deepstack mergers -- and checks both things the
# decoder consumes:
#   - the merged output tokens (the reference's `pooler_output`), which the decoder
#     scatters into the text sequence at `<|image_pad|>` positions;
#   - one deepstack feature per entry of `deepstack_visual_indexes`, which the
#     decoder adds into its first few layers.
#
# Submodules are covered through the tower rather than individually. That includes
# both merger variants: `use_postshuffle_norm` is their only structural
# difference, and it is not merely where the reshape sits -- pre-shuffle
# normalizes each patch over `hidden_size`, post-shuffle normalizes the
# concatenated group of four over `hidden_size * merge ** 2`, so the two compute
# different statistics and cannot share weights.
#
# Every shape here is one the model actually runs, at the released depth of 27 with
# deepstack taps at 8/16/24. Nothing is scaled down: the tower is only 595M
# parameters, so a dummy-weight copy at full depth is cheap, and a reduced one would
# exercise geometry that never occurs. See the grid tables below for the two
# distinct sizing rules and why both orientations of each aspect are present.
#
# The same paths on the released weights live in
# tests/models/minimax_h3/test_vision_conditioner_minimax_h3.py.
# =============================================================================

import pytest
import torch
import transformers

import ttnn

from ....encoders.qwen3vl.vision_qwen3vl import Qwen3VlVisionModel, vision_cu_seqlens
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
OUT_HIDDEN_SIZE = 5120
NUM_POSITION_EMBEDDINGS = 2304
NORM_EPS = 1e-6
HIDDEN_ACT = "gelu_pytorch_tanh"

# The released tower: 27 blocks with deepstack taps at 8, 16 and 24. The whole 595M model, so a
# dummy-weight copy is affordable and there is no reason to run a shallower one.
DEPTH = 27
DEEPSTACK_INDEXES = [8, 16, 24]

# Every grid the model can actually present, measured end to end through the checkpoint's own image
# processor rather than derived by hand. TWO sizing rules feed the tower and they differ 4x on the short
# edge, which is the thing most easily got wrong:
#
#   - keyframes (`fl2va`) and reference VIDEOS go through `resolve_canvas_size`: 768 px short edge, area
#     capped at 768x1344, each axis rounded to a multiple of 32 px. At a 16-pixel patch that is 48
#     patches on the short edge -- except at the aspect extremes, where the AREA CAP binds first and
#     pushes the short edge down to 32.
#   - reference IMAGES (`ref2va`) go through `resolve_reference_image_size`: 2048 px short edge, NO area
#     cap, upscaling intended. That is 128 patches on the short edge and up to 512 on the long one.
#
# Both orientations of every aspect are present. They are not redundant despite equal patch counts:
# `canvas_16to9` is 48x84 while `canvas_4to1` is 32x126, both 4032, and only the pairing catches an
# `h`-versus-`w` error in the bilinear position table or the 2x2 merge reshape. Every grid has even `h`
# and `w`, so the merge is always valid.
#
# All four area-capped canvases are 4032 patches, which is 31.5 x 128, so NONE of them can be
# sequence-parallel at SP=4. That is a property of the model's geometry: at the largest canvas, SP=4 is
# simply unavailable.
_CANVAS = {  # patches  blocks  SP=4
    "canvas_1to1": [[1, 48, 48]],  # 2304    1     ok
    "canvas_4to3": [[1, 48, 64]],  # 3072    1     ok
    "canvas_3to4": [[1, 64, 48]],  # 3072    1     ok
    "canvas_16to9": [[1, 48, 84]],  # 4032    1     skip -- area cap
    "canvas_9to16": [[1, 84, 48]],  # 4032    1     skip -- area cap
    "canvas_4to1": [[1, 32, 126]],  # 4032    1     skip -- area cap, short edge forced to 32
    "canvas_1to4": [[1, 126, 32]],  # 4032    1     skip -- area cap
}
_REFERENCE = {
    "ref_1to1": [[1, 128, 128]],  # 16384    1     ok
    "ref_4to3": [[1, 128, 170]],  # 21760    1     ok
    "ref_3to4": [[1, 170, 128]],  # 21760    1     ok
    "ref_16to9": [[1, 128, 228]],  # 29184    1     ok
    "ref_9to16": [[1, 228, 128]],  # 29184    1     ok
    "ref_4to1": [[1, 128, 512]],  # 65536    1     ok
    "ref_1to4": [[1, 512, 128]],  # 65536    1     ok
}
# `cu_seqlens = repeat_interleave(h*w, t).cumsum()` makes an image one block and a video one block PER
# FRAME, and attention must never cross a boundary. `two_refs` gets its blocks from separate grid rows
# and `video_3_frames` from a single row with `t > 1`; both must agree, and a tower that ignored blocking
# would still pass every single-block grid above. `image_and_video` is the only case that mixes the two
# sizing rules inside one sequence.
_MULTI = {
    "two_refs": [[1, 128, 128], [1, 128, 170]],  # 38144    2     ok, unequal lengths
    "video_3_frames": [[3, 48, 48]],  # 6912    3     ok, one grid row
    "image_and_video": [[1, 128, 128], [3, 48, 48]],  # 23296    4     ok, both rules
    # The documented ceiling: nine reference images and three reference videos.
    "max_load": [[1, 128, 128]] * 9 + [[3, 48, 48]] * 3,  # 168192   18     ok
}

GRIDS = {**_CANVAS, **_REFERENCE, **_MULTI}


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


# ------------------------------------------------------------------------- device

# TP fractures heads/intermediate/merger; SP splits patch rows and ring-attends. Different axes, so
# the 8x4 system gives TP=8 x SP=4.
#
# Only TP=8 is deployed, with SP either off or 4, so the configs are named for both factors. SP alone
# (TP=1) is not a configuration this model will ever run in and is not covered.
#
# `device_params` is per-config: FABRIC_1D has no ethernet partner on a 1x1 mesh and times out in
# router init.
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

    At SP=4 that means a multiple of 128 patches. Every grid here satisfies it except the four
    area-capped canvases, which are all 4032 = 31.5 x 128 -- so at the largest output canvas, in any
    aspect ratio, SP=4 is unavailable. That is a property of the model's geometry, not of the test.
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
@pytest.mark.parametrize("name", list(GRIDS))
def test_tower_on_device(reference, mesh_device, submesh_shape, tp_axis, sp_axis, num_links, name):
    """The full tower: patch embed, position embeddings, every block, all four mergers.

    Covers the whole stack through its outputs -- the merged tokens and one deepstack feature per
    `deepstack_visual_indexes` entry -- so the submodules are exercised here rather than separately. The
    two merger variants both run, since the output merger is pre-shuffle (norm over 1152) and the
    deepstack mergers are post-shuffle (norm over 4608).

    The multi-reference grids make `cu_seqlens` load-bearing: attention must not cross from one image or
    video frame into the next. The reference derives its own boundaries from `grid_thw`, so a tower that
    ignored blocking would disagree here while still passing on the single-block grids.
    """
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

    # One block per frame, so the count is sum(t) rather than the number of grid rows.
    cu_seqlens = vision_cu_seqlens(grid)
    assert len(cu_seqlens) - 1 == int(grid[:, 0].sum()), f"expected one block per frame, got {cu_seqlens}"
    assert cu_seqlens[0] == 0 and cu_seqlens[-1] == total, f"cu_seqlens must span [0, {total}]: {cu_seqlens}"

    tower = _tower(reference, submesh, *_parallel_args(submesh, tp_axis, sp_axis, num_links))
    cos, sin = tower.prepare_rope(grid)
    pos = tower.prepare_pos_embeds(grid)
    # Inputs shard on rows under SP; the tower's outputs come back gathered and replicated.
    tokens, deepstack = tower.forward(
        _shard(patches, submesh, sp_axis),
        pos_embeds=_shard(pos, submesh, sp_axis),
        rope=(_shard(cos, submesh, sp_axis), _shard(sin, submesh, sp_axis)),
        cu_seqlens=cu_seqlens,
    )

    merged = total // SPATIAL_MERGE_SIZE**2
    actual_tokens = tensor.to_torch(tokens, mesh_axes=[None, None])
    assert actual_tokens.shape[-2:] == (merged, OUT_HIDDEN_SIZE), f"{tuple(actual_tokens.shape)}"
    assert_quality(golden_tokens, actual_tokens, pcc=0.99)

    assert len(deepstack) == len(DEEPSTACK_INDEXES), f"expected {len(DEEPSTACK_INDEXES)} features"
    features = []
    for i, (feature, golden_feature) in enumerate(zip(deepstack, golden_deepstack)):
        actual = tensor.to_torch(feature, mesh_axes=[None, None])
        assert actual.shape[-2:] == (merged, OUT_HIDDEN_SIZE), f"deepstack {i}: {tuple(actual.shape)}"
        assert_quality(golden_feature, actual, pcc=0.99)
        features.append(actual)

    # Routing by list index can yield the right count and shapes while tapping one layer twice.
    for i in range(len(features) - 1):
        assert not torch.allclose(
            features[i], features[i + 1], atol=1e-2
        ), f"deepstack features {i} and {i + 1} are identical; layer routing is wrong"
