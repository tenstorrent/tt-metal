# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# MiniMax-H3 conditioner with an image, on the RELEASED weights: the fl2va path.
#
# WIP: test_fused_conditioner_real_weights FAILS at PCC 98.4764% against the
# 0.99 threshold, and the cause is NOT established. The two vision-tower cases
# pass. Do not read a green run of this file as fl2va being verified end to end.
#
# What is known about the failure:
#   - 98.4764% / RMSE 18.7% is bit-reproducible: measured four times, in-suite
#     and in isolation (-k fused), before and after a device reboot. It is not
#     flaky, not cross-test interference, and not hardware.
#   - A standalone script feeding the SAME decoder the reference's own vision
#     output scored 99.8789%, and our tower's 99.8006% -- which would say the
#     injection is correct and the tower contributes ~0.08 points. But that
#     script disagrees with this test by 1.3 points on what should be identical
#     work, and it had two defects of its own, so it is the less trustworthy of
#     the two and its numbers are recorded here only as a lead.
#   - The reduced-geometry equivalent in
#     tests/encoders/qwen3vl/test_qwen3vl_fused_conditioner.py passes at
#     99.9917%, so whatever this is needs the real depth, width or tap index to
#     show up -- 64 layers and a tap at 50, versus 4 layers and a tap at 3.
#
# The companion to test_text_encoder_minimax_h3.py, which covers t2va at PCC
# 99.9993%. Everything above that point was verified with reduced geometry and
# random weights -- structure, not fidelity. This closes that gap for the tower:
#
#   - the vision tower at its real geometry (depth 27, hidden 1152, head_dim 72
#     padded to 96, deepstack [8, 16, 24], a 48x48 position table) on the
#     released 595M-parameter weights -- PASSES at 99.5580% (tokens) and
#     99.9010 / 99.8785 / 99.6816 (deepstack), on both real canvases;
#   - the fused conditioner, with MiniMax-H3's exact presentation built by the
#     real tokenizer and the real image processor, tapped at hidden_states[50]
#     -- currently FAILING, see above.
#
# The conditioner is ~32B params, so the decoder runs TP=8 on a Galaxy while the
# 595M tower rides along replicated. Large-host test: it needs the ~62 GiB of
# shards resolvable and about that much RAM, and skips when they are not.
# =============================================================================

import os

import pytest
import torch
import transformers
from huggingface_hub import snapshot_download
from loguru import logger
from PIL import Image

import ttnn

from ....encoders.qwen3vl.model_qwen3vl import (
    Qwen3VlTextEncoder,
    create_rope_tensors,
    mrope_position_ids,
    vision_token_runs,
)
from ....encoders.qwen3vl.vision_qwen3vl import Qwen3VlVisionModel, vision_cu_seqlens
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor

_LOCAL_MIRROR = "/data/cglagovich/MiniMax-H3-diffusers"
_HF_REPO = "MiniMaxAI/MiniMax-H3"
_SUBFOLDER = "text_encoder"
_PATTERNS = [f"{_SUBFOLDER}/*"]

# A 448x448 image is grid [1, 28, 28] -- 784 patches, 196 tokens. Real geometry throughout, sized so
# the 64-layer CPU reference stays affordable; the full 768x1344 canvas is 4032 patches / 1008 tokens
# and is exercised by the tower-only test.
FUSED_IMAGE = (448, 448)
KEYFRAME_IMAGE = (1344, 768)


def _test_image(size):
    """A deterministically textured image.

    Content matters here, not just geometry. A flat colour gives near-identical patches, so the merged
    tokens are near-identical rows and PCC across them is dominated by the small inter-row differences
    -- which is precisely where bf16 noise lives. Measured on the released weights: a solid colour
    scores 96.2% where texture scores 99.6%, on identical code and the same mesh. The low number is a
    property of the metric on a degenerate input, not of the port.
    """
    generator = torch.Generator().manual_seed(0)
    pixels = (torch.rand(size[1], size[0], 3, generator=generator) * 255).to(torch.uint8)
    return Image.fromarray(pixels.numpy())


def _conditioner_dir() -> str:
    """`MINIMAX_H3_REPO`, then the local mirror, then a scoped Hub snapshot. Missing is a skip."""
    try:
        ref = os.environ.get("MINIMAX_H3_REPO", "").strip()
        if ref and os.path.isdir(ref):
            root = ref
        elif not ref and os.path.isdir(_LOCAL_MIRROR):
            root = _LOCAL_MIRROR
        else:
            repo_id = ref or _HF_REPO
            logger.info(f"MiniMax-H3 conditioner not local; fetching {_PATTERNS} from {repo_id}")
            root = snapshot_download(repo_id=repo_id, allow_patterns=_PATTERNS)
        return os.path.join(root, _SUBFOLDER)
    except Exception as exc:  # noqa: BLE001 - transport/auth/gating failures are a skip, not a failure
        pytest.skip(f"MiniMax-H3 conditioner unavailable ({_LOCAL_MIRROR}, then {_HF_REPO}): {exc}")


@pytest.fixture(scope="module")
def conditioner():
    """The released conditioner, loaded once. `dtype` matches the checkpoint's own bf16."""
    path = _conditioner_dir()
    hf, info = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, output_loading_info=True
    )
    bad = {k: sorted(info[k])[:5] for k in ("missing_keys", "unexpected_keys", "mismatched_keys") if info[k]}
    assert not bad, f"conditioner load key mismatch: {bad}"
    return path, hf.model.eval()


def _tower(reference_visual, submesh):
    vc = reference_visual.config
    tower = Qwen3VlVisionModel(
        hidden_size=vc.hidden_size,
        num_heads=vc.num_heads,
        depth=vc.depth,
        intermediate_size=vc.intermediate_size,
        in_channels=vc.in_channels,
        patch_size=vc.patch_size,
        temporal_patch_size=vc.temporal_patch_size,
        spatial_merge_size=vc.spatial_merge_size,
        num_position_embeddings=vc.num_position_embeddings,
        out_hidden_size=vc.out_hidden_size,
        hidden_act=vc.hidden_act,
        norm_eps=1e-6,
        deepstack_visual_indexes=vc.deepstack_visual_indexes,
        mesh_device=submesh,
    )
    tower.load_torch_state_dict(reference_visual.state_dict())
    return tower


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis", "num_links"),
    [pytest.param((4, 8), (4, 8), 1, 2, id="tp8_axis1")],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768}], indirect=True
)
@pytest.mark.parametrize("size", [FUSED_IMAGE, KEYFRAME_IMAGE], ids=["448sq", "keyframe_768x1344"])
def test_vision_tower_real_weights(conditioner, mesh_device, submesh_shape, tp_axis, num_links, size):
    """The released vision tower: merged tokens and all three deepstack features.

    `head_dim` is 72 here, the misalignment the padding exists for, and the 48x48 position table is
    smaller than either grid, so the bilinear interpolation is live in both cases.
    """
    path, reference = conditioner
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    processor = transformers.AutoImageProcessor.from_pretrained(path)

    vision = processor(images=[_test_image(size)], return_tensors="pt")
    pixel_values, grid = vision["pixel_values"], vision["image_grid_thw"]
    vc = reference.visual.config
    assert vc.hidden_size // vc.num_heads == 72, "the padding path is not being exercised"

    with torch.no_grad():
        ref_out = reference.visual(pixel_values, grid_thw=grid, return_dict=True)
    assert len(ref_out.deepstack_features) == len(vc.deepstack_visual_indexes)

    tower = _tower(reference.visual, submesh)
    cos, sin = tower.prepare_rope(grid)
    tokens, deepstack = tower.forward(
        bf16_tensor(pixel_values.float(), device=submesh),
        pos_embeds=bf16_tensor(tower.prepare_pos_embeds(grid), device=submesh),
        rope=(bf16_tensor(cos, device=submesh), bf16_tensor(sin, device=submesh)),
        cu_seqlens=vision_cu_seqlens(grid),
    )

    logger.info(f"minimax-h3 vision tower [real] {size[0]}x{size[1]} grid={grid[0].tolist()}:")
    assert_quality(ref_out.pooler_output.float(), tensor.to_torch(tokens, mesh_axes=[None, None]), pcc=0.99)
    for i, (feature, golden) in enumerate(zip(deepstack, ref_out.deepstack_features)):
        logger.info(f"  deepstack {i} (vision layer {vc.deepstack_visual_indexes[i]}):")
        assert_quality(golden.float(), tensor.to_torch(feature, mesh_axes=[None, None]), pcc=0.99)


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis", "num_links"),
    [pytest.param((4, 8), (4, 8), 1, 2, id="tp8_axis1")],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768}], indirect=True
)
def test_fused_conditioner_real_weights(conditioner, mesh_device, submesh_shape, tp_axis, num_links):
    """WIP -- FAILING at PCC 98.4764% against the 0.99 threshold, cause not established.

    Left failing rather than relaxed or skipped: the number is bit-reproducible across four runs and
    the reduced-geometry equivalent passes at 99.9917%, so there is something real here that only the
    released depth and tap index expose. Relaxing the threshold would bury it; skipping would stop it
    being measured. See the file header for what has been ruled out.

    MiniMax-H3's presentation is built here the way `encoders.py::encode_prompt` builds it -- a
    `"<Picture 1>: "` label, then a vision block, then the prompt verbatim, with no chat template and
    no special tokens -- using the checkpoint's own tokenizer and image processor.
    """
    from diffusers.modular_pipelines.minimax_h3.packing import MINIMAX_H3_TEXT_ENCODER_LAYER as TAP

    path, reference = conditioner
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    tp_factor = tuple(submesh.shape)[tp_axis]

    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    processor = transformers.AutoImageProcessor.from_pretrained(path)
    vision = processor(images=[_test_image(FUSED_IMAGE)], return_tensors="pt")
    pixel_values, grid = vision["pixel_values"], vision["image_grid_thw"]
    merge = reference.visual.config.spatial_merge_size**2
    num_image_tokens = int(grid[0].prod()) // merge

    # exactly the order encode_prompt assembles
    label = tokenizer("<Picture 1>: ", add_special_tokens=False)["input_ids"]
    image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    block = (
        [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
        + [image_pad] * num_image_tokens
        + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
    )
    prompt = tokenizer("a robot dancing", add_special_tokens=False)["input_ids"]
    ids = torch.tensor([label + block + prompt], dtype=torch.long)
    # 0 text, 1 image -- what `Qwen3VLProcessor.create_mm_token_type_ids` derives from the pad ids
    type_ids = (ids == image_pad).long()
    seq_len = ids.shape[1]

    # --- golden: the reference runs its own tower and builds its own position ids ---
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
    cfg = reference.language_model.config
    assert golden.shape == (1, seq_len, cfg.hidden_size)

    # --- port ---
    tower = _tower(reference.visual, submesh)
    vis_cos, vis_sin = tower.prepare_rope(grid)
    merged, deepstack = tower.forward(
        bf16_tensor(pixel_values.float(), device=submesh),
        pos_embeds=bf16_tensor(tower.prepare_pos_embeds(grid), device=submesh),
        rope=(bf16_tensor(vis_cos, device=submesh), bf16_tensor(vis_sin, device=submesh)),
        cu_seqlens=vision_cu_seqlens(grid),
    )

    rope_params = getattr(cfg, "rope_parameters", None) or cfg.rope_scaling
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    encoder = Qwen3VlTextEncoder(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        hidden_act=cfg.hidden_act,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=rope_params["rope_theta"],
        mrope_section=rope_params["mrope_section"],
        head_dim=head_dim,
        activation_layers=(TAP,),
        device=submesh,
        parallel_config=EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis)),
        ccl_manager=CCLManager(submesh, num_links=num_links, topology=ttnn.Topology.Linear),
    )
    encoder.load_torch_state_dict(reference.language_model.state_dict())

    # A vision run makes the three M-RoPE axes diverge, so the interleaved layout is load-bearing here.
    assert rope_params.get("mrope_interleaved") is True, "this checkpoint is expected to be interleaved"
    position_ids = mrope_position_ids(
        type_ids, image_grid_thw=grid, spatial_merge_size=reference.visual.config.spatial_merge_size
    )
    cos, sin = create_rope_tensors(
        1,
        seq_len,
        None,
        head_dim,
        rope_params["rope_theta"],
        rope_params["mrope_section"],
        position_ids=position_ids,
        interleaved=True,
    )
    runs = vision_token_runs(ids, image_pad)
    assert runs == [(len(label) + 1, num_image_tokens)], f"unexpected vision layout: {runs}"

    out = encoder.forward(
        ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=submesh),
        attention_mask=None,
        pos_embeds=(bf16_tensor(cos, device=submesh), bf16_tensor(sin, device=submesh)),
        vision_embeds=merged,
        vision_runs=runs,
        deepstack_embeds=deepstack,
    )[0]
    actual = tensor.to_torch(out, mesh_axes=[None, None, None])

    logger.info(
        f"minimax-h3 fused conditioner [real] TP={tp_factor} layer {TAP} of {cfg.num_hidden_layers}, "
        f"seq={seq_len} ({num_image_tokens} image tokens):"
    )
    assert actual.shape[-2:] == (seq_len, cfg.hidden_size)
    assert_quality(golden, actual, pcc=0.99)
