# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# MiniMax-H3 conditioner with an image (fl2va path), on the RELEASED weights. The fused
# conditioner is a strict xfail: massive-activation rows disagree; see the xfail reason and
# git history. Large-host test: ~62 GiB of shards and RAM; skips when unavailable.

import re

import numpy as np
import pytest
import torch
import transformers
from loguru import logger
from PIL import Image

import ttnn

from ....encoders.qwen3vl.loader_minimax_h3 import MINIMAX_H3_TEXT_ENCODER_LAYER as TAP
from ....encoders.qwen3vl.loader_minimax_h3 import build_minimax_h3_text_encoder
from ....encoders.qwen3vl.model_qwen3vl import create_rope_tensors, mrope_position_ids, vision_token_runs
from ....encoders.qwen3vl.vision_qwen3vl import Qwen3VlVisionModel, vision_cu_seqlens
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor
from .common import CONDITIONER_SUBFOLDER, conditioner_checkpoint_dir, load_reference_conditioner

_PATTERNS = [f"{CONDITIONER_SUBFOLDER}/*"]

# the production keyframe canvas; 448x448 is not a canvas `resolve_canvas_size` yields
KEYFRAME_IMAGE = (1344, 768)

# per-row bars by row class: whole-tensor PCC is dominated by a few massive rows and only logged
FUSED_MAX_TEXT_ROW_ERROR = 0.05  # measured max 0.0247 (HiFi2 calibration; HiFi4 is lower)
FUSED_MAX_MEDIAN_ROW_ERROR = 0.15  # measured median 0.0901 (HiFi2 calibration; HiFi4 is lower)
MASSIVE_ROW_MULTIPLE = 10.0  # rows above this multiple of the median norm are "massive activations"


def _test_image(size):
    """Frame 0 of the calibrated t2va generation: the metric is content-sensitive, and this is the fl2va gate's exact keyframe."""
    from pathlib import Path

    import imageio.v3 as iio

    from ....pipelines.minimax_h3.packing import prepare_keyframe_image

    source = Path.home() / "h3_t2va_artifacts" / "t2va.mp4"
    if not source.is_file():
        pytest.skip(
            f"no calibrated t2va artifact at {source}; run test_pipeline_minimax_h3.py first. These are "
            "released-weights production-shape gates, so they condition on real content rather than "
            "inventing some"
        )
    frame = Image.fromarray(np.asarray(iio.imread(source, index=0, plugin="pyav"))).convert("RGB")
    width, height = size
    return prepare_keyframe_image(frame, height, width, True)


@pytest.fixture(scope="module")
def conditioner():
    path = conditioner_checkpoint_dir(_PATTERNS)
    hf = load_reference_conditioner(path)
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
@pytest.mark.parametrize("size", [KEYFRAME_IMAGE], ids=["keyframe_768x1344"])
def test_vision_tower_real_weights(conditioner, mesh_device, submesh_shape, tp_axis, num_links, size):
    """head_dim 72 exercises the padding path; the 48x48 position table makes bilinear interpolation live."""
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
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Massive-activation rows disagree: the reference produces 7 rows whose norm exceeds 10x the "
        "median (up to 79x). With the HiFi4 decoder linears the pipeline now always runs (whole-tensor "
        "PCC 85.82%, up from 70.89%), we reproduce 6 of them -- rows 102 and 128 recovered, row 63 "
        "still missing -- and invent 1 (row 156). Text rows and the median vision row both pass; the "
        "shape, content and tap are all production. strict=True so improving the conditioner's "
        "precision further forces a return here."
    ),
)
@pytest.mark.parametrize("size", [KEYFRAME_IMAGE], ids=["keyframe_768x1344"])
def test_fused_conditioner_real_weights(conditioner, mesh_device, submesh_shape, tp_axis, num_links, size):
    """Production shape, content and tap; the loose per-row bars still gate regressions (rotary, tags, scatter, deepstack) hard."""
    path, reference = conditioner
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    tp_factor = tuple(submesh.shape)[tp_axis]

    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    processor = transformers.AutoImageProcessor.from_pretrained(path)
    vision = processor(images=[_test_image(size)], return_tensors="pt")
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
    type_ids = (ids == image_pad).long()
    seq_len = ids.shape[1]

    # golden: hidden_states[TAP], as production reads it; a hook on layers[TAP] captures TAP + 1
    with torch.no_grad():
        outputs = reference(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            mm_token_type_ids=type_ids,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            use_cache=False,
            output_hidden_states=True,
        )
    golden = outputs.hidden_states[TAP].float()
    cfg = reference.language_model.config
    assert golden.shape == (1, seq_len, cfg.hidden_size)
    assert len(outputs.hidden_states) == cfg.num_hidden_layers + 1

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

    # the production builder; state dict truncated because `load_torch_state_dict` is strict
    encoder, _ = build_minimax_h3_text_encoder(
        path,
        mesh_device=submesh,
        parallel_config=EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis)),
        ccl_manager=CCLManager(submesh, num_links=num_links, topology=ttnn.Topology.Linear),
        is_fsdp=False,
        num_layers=TAP,
        load_weights=False,
    )
    layer_re = re.compile(r"^layers\.(\d+)\.")
    truncated = {
        key: value
        for key, value in reference.language_model.state_dict().items()
        if not (m := layer_re.match(key)) or int(m.group(1)) < TAP
    }
    encoder.load_torch_state_dict(truncated)

    # a vision run makes the three M-RoPE axes diverge, so the interleaved layout is load-bearing
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
        f"minimax-h3 fused conditioner [real] TP={tp_factor} hidden_states[{TAP}] "
        f"(= layer {TAP - 1} of a {TAP}-layer stack), {size[0]}x{size[1]} grid={grid[0].tolist()}, "
        f"seq={seq_len} ({num_image_tokens} image tokens):"
    )
    assert actual.shape[-2:] == (seq_len, cfg.hidden_size)

    # per-row relative L2 error: what the DiT's `context_embedder` actually consumes
    g = golden[0].double()
    p = actual.reshape(golden.shape)[0].double()
    row_error = (p - g).norm(dim=1) / g.norm(dim=1)
    is_text = ~type_ids[0].bool()
    norms = g.norm(dim=1)
    median_norm = float(norms.median())
    golden_massive = norms > MASSIVE_ROW_MULTIPLE * median_norm
    ours_massive = p.norm(dim=1) > MASSIVE_ROW_MULTIPLE * median_norm
    ordinary = ~golden_massive & ~ours_massive

    assert_quality(golden, actual)  # logs whole-tensor PCC / CCC / RMSE without gating on them
    logger.info(
        f"  row norms: median {median_norm:.1f}, max {float(norms.max()):.1f} "
        f"({float(norms.max()) / median_norm:.0f}x median)"
    )
    for name, mask in (
        ("text", is_text),
        ("ordinary vision", ordinary & ~is_text),
        ("massive (either side)", golden_massive | ours_massive),
    ):
        if mask.any():
            e = row_error[mask]
            logger.info(
                f"  {name:22s} n={int(mask.sum()):4d}  median {float(e.median()) * 100:7.2f} %  "
                f"max {float(e.max()) * 100:8.2f} %"
            )
    logger.info(
        f"  massive-activation rows: golden {int(golden_massive.sum())} at "
        f"{golden_massive.nonzero().flatten().tolist()}, ours {int(ours_massive.sum())} at "
        f"{ours_massive.nonzero().flatten().tolist()}"
    )

    assert float(row_error[is_text].max()) < FUSED_MAX_TEXT_ROW_ERROR, (
        f"text rows are {float(row_error[is_text].max()) * 100:.2f} % off; the decoder path itself is wrong, "
        "not just the vision fidelity"
    )
    assert float(row_error.median()) < FUSED_MAX_MEDIAN_ROW_ERROR, (
        f"median per-row error {float(row_error.median()) * 100:.2f} % exceeds "
        f"{FUSED_MAX_TEXT_ROW_ERROR * 100:.0f} %; the typical row has regressed"
    )

    # the strict-xfail check; see the xfail reason
    missing = int((golden_massive & ~ours_massive).sum())
    spurious = int((ours_massive & ~golden_massive).sum())
    assert missing == 0 and spurious == 0, (
        f"massive-activation rows disagree: {missing} present in the reference and absent from ours, "
        f"{spurious} present in ours and absent from the reference "
        f"(golden {int(golden_massive.sum())} such rows, ours {int(ours_massive.sum())}). These rows carry "
        f"norms up to {float(norms.max()) / median_norm:.0f}x the median, so missing one dominates every "
        "whole-tensor metric."
    )
