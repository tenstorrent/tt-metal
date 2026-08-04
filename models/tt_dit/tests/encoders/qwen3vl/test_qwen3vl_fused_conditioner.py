# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Qwen3-VL conditioner end to end with an image: the `fl2va` path.
#
# A real image goes through the real image processor, MiniMax-H3's presentation
# is assembled the way `encoders.py::encode_prompt` assembles it, and the
# mid-stack tap is compared against `Qwen3VLModel`. Everything the earlier
# increments built has to agree at once:
#
#   vision tower  ->  merged tokens   ->  scattered at <|image_pad|>
#                 ->  deepstack       ->  added at the leading decoder layers
#   mrope_position_ids  ->  create_rope_tensors(interleaved=True)
#
# This is the first test where the interleaved rotary layout is not a no-op: the
# prompt contains a vision run, so the three M-RoPE axes diverge and the chunked
# layout would be wrong. It is also the first end-to-end check that the golden's
# internally-computed position ids match ours -- `Qwen3VLModel.forward` builds
# its own, so a mismatch shows up as a PCC failure rather than an error.
#
# Geometry is reduced to keep the CPU reference affordable, but the properties
# that caused trouble are preserved: the vision head_dim is misaligned (48, pads
# to 64), the position table is smaller than the patch grid so interpolation is
# live, and there is more than one deepstack feature.
# =============================================================================

import pytest
import torch
import transformers
from PIL import Image

import ttnn

from ....encoders.qwen3vl.model_qwen3vl import (
    Qwen3VlTextEncoder,
    create_rope_tensors,
    mrope_position_ids,
    vision_token_runs,
)
from ....encoders.qwen3vl.vision_qwen3vl import Qwen3VlVisionModel
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor

# text stack
TEXT_HIDDEN = 128
TEXT_HEADS = 4
TEXT_HEAD_DIM = 32
TEXT_LAYERS = 4
MROPE_SECTION = [6, 5, 5]  # sums to TEXT_HEAD_DIM // 2
ROPE_THETA = 10000.0
TAP = 3  # mid-stack, and after every deepstack injection

# vision tower -- 192 / 4 heads = head_dim 48, deliberately not tile-aligned
VIS_HIDDEN = 192
VIS_HEADS = 4
VIS_DEPTH = 4
VIS_INTERMEDIATE = 256
DEEPSTACK_INDEXES = [1, 3]
NUM_POSITION_EMBEDDINGS = 64  # 8x8 -- smaller than the patch grid, so interpolation is live
SPATIAL_MERGE_SIZE = 2
PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2

IMAGE_TOKEN_ID = 151655
VISION_START_ID = 151652
VISION_END_ID = 151653
VOCAB = 152000


@pytest.fixture(scope="module")
def image_processor():
    """Small pixel limits so a 96x64 image is grid [1, 4, 6] and the CPU reference stays cheap."""
    return transformers.Qwen2VLImageProcessor(
        patch_size=PATCH_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        merge_size=SPATIAL_MERGE_SIZE,
        size={"shortest_edge": PATCH_SIZE * PATCH_SIZE * 4, "longest_edge": PATCH_SIZE * PATCH_SIZE * 36},
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )


@pytest.fixture(scope="module")
def reference():
    config = transformers.Qwen3VLConfig(
        text_config={
            "vocab_size": VOCAB,
            "hidden_size": TEXT_HIDDEN,
            "intermediate_size": 256,
            "num_hidden_layers": TEXT_LAYERS,
            "num_attention_heads": TEXT_HEADS,
            "num_key_value_heads": 2,
            "head_dim": TEXT_HEAD_DIM,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": ROPE_THETA,
                "mrope_section": MROPE_SECTION,
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": VIS_DEPTH,
            "hidden_size": VIS_HIDDEN,
            "num_heads": VIS_HEADS,
            "intermediate_size": VIS_INTERMEDIATE,
            "in_channels": 3,
            "patch_size": PATCH_SIZE,
            "temporal_patch_size": TEMPORAL_PATCH_SIZE,
            "spatial_merge_size": SPATIAL_MERGE_SIZE,
            "num_position_embeddings": NUM_POSITION_EMBEDDINGS,
            "out_hidden_size": TEXT_HIDDEN,  # merged tokens must fit the decoder's width
            "hidden_act": "gelu_pytorch_tanh",
            "deepstack_visual_indexes": DEEPSTACK_INDEXES,
        },
        image_token_id=IMAGE_TOKEN_ID,
        vision_start_token_id=VISION_START_ID,
        vision_end_token_id=VISION_END_ID,
    )
    torch.manual_seed(0)
    return transformers.AutoModel.from_config(config).eval()


def _presentation(num_image_tokens, *, label_len=4, prompt_len=7):
    """MiniMax-H3's `fl2va` presentation, in the order `encode_prompt` builds it.

    A `"<Picture 1>: "` label, then `<|vision_start|>`, one run of `<|image_pad|>`, `<|vision_end|>`,
    then the prompt. Token ids for the text are arbitrary; only the vision ids have to be real.
    """
    torch.manual_seed(1)
    label = torch.randint(1000, 2000, (label_len,))
    prompt = torch.randint(2000, 3000, (prompt_len,))
    ids = torch.cat(
        [
            label,
            torch.tensor([VISION_START_ID]),
            torch.full((num_image_tokens,), IMAGE_TOKEN_ID),
            torch.tensor([VISION_END_ID]),
            prompt,
        ]
    ).unsqueeze(0)
    # 0 text, 1 image -- what `Qwen3VLProcessor.create_mm_token_type_ids` derives from the pad ids
    type_ids = (ids == IMAGE_TOKEN_ID).long()
    return ids, type_ids


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape"), [pytest.param((1, 1), (1, 1), id="single")], indirect=["mesh_device"]
)
def test_fused_conditioner_with_an_image(reference, image_processor, mesh_device, submesh_shape):
    """The mid-stack tap with an image present, against `Qwen3VLModel`."""
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    vision = image_processor(images=[Image.new("RGB", (96, 64), color=(90, 140, 200))], return_tensors="pt")
    pixel_values, grid = vision["pixel_values"], vision["image_grid_thw"]
    num_patches = int(grid[0][1] * grid[0][2])
    num_image_tokens = num_patches // SPATIAL_MERGE_SIZE**2
    ids, type_ids = _presentation(num_image_tokens)
    seq_len = ids.shape[1]

    # --- golden: the reference builds its own position ids and runs its own tower ---
    captured: dict[int, torch.Tensor] = {}
    handle = reference.language_model.layers[TAP].register_forward_hook(
        lambda m, i_, o: captured.__setitem__(TAP, (o[0] if isinstance(o, tuple) else o).detach())
    )
    with torch.no_grad():
        reference(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            mm_token_type_ids=type_ids,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            use_cache=False,
        )
    handle.remove()
    golden = captured[TAP].float()
    assert golden.shape == (1, seq_len, TEXT_HIDDEN)

    # --- port: tower, then decoder with the vision arguments ---
    tower = Qwen3VlVisionModel(
        hidden_size=VIS_HIDDEN,
        num_heads=VIS_HEADS,
        depth=VIS_DEPTH,
        intermediate_size=VIS_INTERMEDIATE,
        in_channels=3,
        patch_size=PATCH_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        num_position_embeddings=NUM_POSITION_EMBEDDINGS,
        out_hidden_size=TEXT_HIDDEN,
        hidden_act="gelu_pytorch_tanh",
        norm_eps=1e-6,
        deepstack_visual_indexes=DEEPSTACK_INDEXES,
        mesh_device=submesh,
    )
    tower.load_torch_state_dict(reference.visual.state_dict())
    vis_cos, vis_sin = tower.prepare_rope(grid)
    merged, deepstack = tower.forward(
        bf16_tensor(pixel_values, device=submesh),
        pos_embeds=bf16_tensor(tower.prepare_pos_embeds(grid), device=submesh),
        rope=(bf16_tensor(vis_cos, device=submesh), bf16_tensor(vis_sin, device=submesh)),
    )
    assert len(deepstack) == len(DEEPSTACK_INDEXES)

    encoder = Qwen3VlTextEncoder(
        vocab_size=VOCAB,
        hidden_size=TEXT_HIDDEN,
        intermediate_size=256,
        hidden_act="silu",
        num_hidden_layers=TEXT_LAYERS,
        num_attention_heads=TEXT_HEADS,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        rope_theta=ROPE_THETA,
        mrope_section=MROPE_SECTION,
        head_dim=TEXT_HEAD_DIM,
        activation_layers=(TAP,),
        device=submesh,
        parallel_config=EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=0)),
        ccl_manager=CCLManager(submesh, num_links=1, topology=ttnn.Topology.Linear),
    )
    encoder.load_torch_state_dict(reference.language_model.state_dict())

    # The vision run makes the three M-RoPE axes diverge, so the interleaved layout is load-bearing
    # here -- unlike the text-only conditioner path, where it is provably a no-op.
    position_ids = mrope_position_ids(type_ids, image_grid_thw=grid, spatial_merge_size=SPATIAL_MERGE_SIZE)
    cos, sin = create_rope_tensors(
        1,
        seq_len,
        None,
        TEXT_HEAD_DIM,
        ROPE_THETA,
        MROPE_SECTION,
        position_ids=position_ids,
        interleaved=True,
    )
    runs = vision_token_runs(ids, IMAGE_TOKEN_ID)
    assert runs == [(5, num_image_tokens)], f"unexpected vision layout: {runs}"

    out = encoder.forward(
        ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=submesh),
        attention_mask=None,
        pos_embeds=(bf16_tensor(cos, device=submesh), bf16_tensor(sin, device=submesh)),
        vision_embeds=merged,
        vision_runs=runs,
        deepstack_embeds=deepstack,
    )[0]
    actual = tensor.to_torch(out, mesh_axes=[None, None, None])

    assert actual.shape[-2:] == (seq_len, TEXT_HIDDEN), f"{tuple(actual.shape)}"
    assert_quality(golden, actual, pcc=0.99)


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape"), [pytest.param((1, 1), (1, 1), id="single")], indirect=["mesh_device"]
)
def test_the_chunked_rotary_layout_is_now_wrong(reference, image_processor, mesh_device, submesh_shape):
    """With a vision run the chunked layout no longer reproduces the reference.

    The complement of the conditioner test's tripwire, which asserts the two layouts coincide for
    text-only prompts. Here they must not: if this ever passed, `interleaved` would be cosmetic and the
    `t2va` justification for using the chunked path would have been load-bearing for the wrong reason.
    """
    vision = image_processor(images=[Image.new("RGB", (96, 64))], return_tensors="pt")
    grid = vision["image_grid_thw"]
    num_image_tokens = int(grid[0][1] * grid[0][2]) // SPATIAL_MERGE_SIZE**2
    _, type_ids = _presentation(num_image_tokens)
    seq_len = type_ids.shape[1]

    position_ids = mrope_position_ids(type_ids, image_grid_thw=grid, spatial_merge_size=SPATIAL_MERGE_SIZE)
    assert not torch.equal(position_ids[0], position_ids[1]), "axes agree; the prompt has no real vision run"

    args = (1, seq_len, None, TEXT_HEAD_DIM, ROPE_THETA, MROPE_SECTION)
    chunked = create_rope_tensors(*args, position_ids=position_ids, interleaved=False)
    interleaved = create_rope_tensors(*args, position_ids=position_ids, interleaved=True)
    assert not torch.equal(chunked[0], interleaved[0]), "cos identical despite a vision run"
    assert not torch.equal(chunked[1], interleaved[1]), "sin identical despite a vision run"
