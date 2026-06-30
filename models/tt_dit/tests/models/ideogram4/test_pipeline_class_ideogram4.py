# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Drive the Ideogram4Pipeline class end-to-end on device (TP=4, all models
# resident: encoder + cond/uncond transformers + VAE), real weights.
#
# Resolution status (task #24):
#   * 512px  -> FAITHFUL (image std ~46).
#   * 1024px / 2048px -> the denoiser under-converges the latent at 4096/16384 tokens
#     (latent std ~0.59 vs ~0.98 at 512px); the image decodes flat (std ~10) on BOTH
#     the device AND the reference host VAE, so the VAE is ruled out. Preset-independent
#     (TURBO_12 and QUALITY_48 both flat). The per-call denoiser is verified at 4096
#     tokens (PCC 0.998) and the schedule matches the reference exactly, so the leading
#     suspect is real-weight bf16 fidelity compounding over the 4x longer attention
#     (consistent with the earlier outlier finding). OPEN: needs a reference-pipeline
#     trajectory comparison and/or higher attention fidelity at long sequence.
# =============================================================================

import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn

from ....pipelines.ideogram4.pipeline import Ideogram4Pipeline


# =============================================================================
# Trace-correctness sanity (random weights, tiny model, 1x1 mesh -> no fabric):
# capture the SAME transformer-forward closure the pipeline traces, then PCC the
# traced replay against the eager (untraced) forward. This isolates "is tracing the
# transformer forward lossless?" from the full-pipeline orchestration, per the
# allocation-after-capture investigation. PCC must be ~1.0 (tracing is lossless).
# =============================================================================
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 23887872}], indirect=True)
@pytest.mark.parametrize("num_layers", [2], ids=["layers2"])
def test_traced_forward_pcc(*, mesh_device, num_layers) -> None:
    from ....models.transformers.transformer_ideogram4 import Ideogram4Transformer
    from ....parallel.config import DiTParallelConfig, ParallelFactor
    from ....parallel.manager import CCLManager
    from ....reference.ideogram4 import modeling_ideogram4
    from ....reference.ideogram4.constants import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR
    from ....utils import tensor
    from ....utils.check import assert_quality
    from ....utils.tensor import bf16_tensor
    from ....utils.tracing import Tracer
    from .test_transformer_ideogram4_model import _build_model_inputs

    torch.manual_seed(0)
    llm_len, image_len = 64, 256
    seq_len = llm_len + image_len
    config = modeling_ideogram4.Ideogram4Config(num_layers=num_layers)
    torch_model = modeling_ideogram4.Ideogram4Transformer(config).to(dtype=torch.bfloat16).eval()
    llm_features, x, t, position_ids, segment_ids, indicator = _build_model_inputs(config, 1, llm_len, image_len)
    llm_features, x = llm_features.to(torch.bfloat16), x.to(torch.bfloat16)

    pc = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=1, mesh_axis=1),
        sequence_parallel=ParallelFactor(factor=1, mesh_axis=0),
    )
    ccl = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    tt_model = Ideogram4Transformer(
        emb_dim=config.emb_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        intermediate_size=config.intermediate_size,
        adaln_dim=config.adanln_dim,
        in_channels=config.in_channels,
        llm_features_dim=config.llm_features_dim,
        norm_eps=config.norm_eps,
        mesh_device=mesh_device,
        ccl_manager=ccl,
        parallel_config=pc,
        padding_config=None,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    cos, sin = torch_model.rotary_emb(position_ids)
    t_sin = Ideogram4Transformer.sinusoidal_embedding(t, config.emb_dim)
    llm_mask = (indicator == LLM_TOKEN_INDICATOR).to(torch.float32).unsqueeze(-1)
    img_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.float32).unsqueeze(-1)
    image_idx = (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.int32)

    fixed = dict(
        llm=bf16_tensor(llm_features, device=mesh_device),
        cos=bf16_tensor(cos.unsqueeze(1), device=mesh_device),
        sin=bf16_tensor(sin.unsqueeze(1), device=mesh_device),
        idx=tensor.from_torch(image_idx, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        llm_mask=bf16_tensor(llm_mask, device=mesh_device),
        img_mask=bf16_tensor(img_mask, device=mesh_device),
    )

    def fwd(x_in, t_in):
        return tt_model(
            x=x_in,
            llm_features=fixed["llm"],
            t_sin=t_in,
            cos=fixed["cos"],
            sin=fixed["sin"],
            image_indicator_index=fixed["idx"],
            llm_token_mask=fixed["llm_mask"],
            output_image_mask=fixed["img_mask"],
            spatial_sequence_length=seq_len,
        )

    x_dev = bf16_tensor(x, device=mesh_device)
    t_dev = bf16_tensor(t_sin.unsqueeze(1), device=mesh_device)
    eager = tensor.to_torch(fwd(x_dev, t_dev), mesh_axes=[None, None, None])

    tracer = Tracer(fwd, device=mesh_device, clone_prep_inputs=False)
    traced = tensor.to_torch(tracer(x_dev, t_dev), mesh_axes=[None, None, None])
    # replay once more with the same inputs -> must still match (no buffer corruption)
    traced2 = tensor.to_torch(tracer(x_dev, t_dev), mesh_axes=[None, None, None])
    tracer.release_trace()

    image_mask = (indicator == OUTPUT_IMAGE_INDICATOR)[0]
    logger.info("traced-forward PCC: eager-vs-traced and eager-vs-replay2")
    assert_quality(eager[:, image_mask], traced[:, image_mask], pcc=0.999)
    assert_quality(eager[:, image_mask], traced2[:, image_mask], pcc=0.999)


def _typography_prompt_json() -> str:
    """Typography-heavy in-distribution JSON caption that showcases Ideogram 4's
    best-in-class text rendering (crisp legible lettering + bbox layout + palette).

    Ideogram 4 was trained exclusively on structured-JSON captions; quoting the exact
    text strings and placing them with bbox is the reliable way to get them rendered.
    """
    import json

    caption = {
        "aspect_ratio": "1:1",
        "high_level_description": (
            "A bold retro travel poster with large, crisp typography. A condensed sans-serif "
            'headline reading "EXPLORE THE WILD" arcs across the top, a clean letter-spaced '
            'subtitle "NATIONAL PARKS · EST. 1872" sits beneath it, and a small banner at the '
            'bottom reads "ADVENTURE AWAITS". Set over a stylized sunrise mountain landscape '
            "with pine silhouettes; flat vintage screen-print look, sharp legible lettering."
        ),
        "colour_palette": ["#F4A259", "#2E4600", "#1B3A4B", "#F2E8CF", "#BC4749"],
        "compositional_deconstruction": {
            "background": (
                "Stylized sunrise sky in warm cream and orange bands, layered mountain ridges in "
                "deep green and teal, a row of pine-tree silhouettes along the lower third, flat "
                "screen-print texture with subtle grain."
            ),
            "elements": [
                {
                    "type": "text",
                    "bbox": [70, 70, 290, 930],
                    "text": "EXPLORE THE WILD",
                    "desc": "Large bold condensed sans-serif headline in cream, all caps, gentle "
                    "upward arc, high contrast against the sky.",
                },
                {
                    "type": "text",
                    "bbox": [300, 180, 400, 820],
                    "text": "NATIONAL PARKS · EST. 1872",
                    "desc": "Smaller clean uppercase subtitle in warm orange, generously "
                    "letter-spaced, centered beneath the headline.",
                },
                {
                    "type": "text",
                    "bbox": [840, 250, 960, 750],
                    "text": "ADVENTURE AWAITS",
                    "desc": "Small bold uppercase banner text in cream on a deep-red rounded ribbon.",
                },
                {
                    "type": "obj",
                    "bbox": [410, 60, 880, 940],
                    "desc": "Layered mountain range at sunrise with a bold circular sun disc, deep "
                    "green and teal ridges, vintage poster shading.",
                },
            ],
        },
    }
    return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))


# Typography showcase prompt (JSON is the known-good format; plain prose washes out
# at >=1024px per the in-weights-moderation finding).
PROMPT = _typography_prompt_json()


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis"),
    [
        pytest.param((2, 4), (1, 4), 1, id="tp4"),
        pytest.param((4, 2), (4, 2), 1, id="sp4tp2"),  # full-mesh SP4xTP2 denoiser default
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "l1_small_size": 65536,
            # Trace region for the per-branch transformer-forward traces (cond + uncond).
            # 60MB headroom; bump if "trace region" OOM at higher resolution.
            "trace_region_size": 60000000,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    ("height", "width"), [(512, 512), (1024, 1024), (2048, 2048)], ids=["512px", "1024px", "2048px"]
)
@pytest.mark.parametrize("preset", ["V4_TURBO_12", "V4_QUALITY_48"])
def test_pipeline_class(*, mesh_device, submesh_shape, tp_axis, height, width, preset) -> None:
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    pipe = Ideogram4Pipeline.from_pretrained(submesh, tp_axis=tp_axis)

    # Warmup run: triggers JIT compile + program-cache fill (NOT timed).
    pipe(PROMPT, height=height, width=width, preset=preset, seed=1234)
    # Timed run: program cache is warm, so the timings exclude JIT compile.
    img = pipe(PROMPT, height=height, width=width, preset=preset, seed=1234)
    t = pipe.timings
    logger.info(
        f"WARM E2E LATENCY {height}px {preset}: total={t['total']:.2f}s | encode={t['encode']:.2f}s "
        f"| denoise={t['denoise']:.2f}s ({t['denoise_per_step']*1000:.0f}ms/step) | decode={t['decode']:.2f}s"
    )

    out = f"/data/cglagovich/ideogram4_pipeline_class_{height}_{preset}.png"
    Image.fromarray(img).save(out)
    logger.info(f"pipeline class saved {out} shape={img.shape} std={img.std():.1f} preset={preset}")
    assert img.shape == (height, width, 3)
    assert img.std() > 1.0
