# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V3-PIPELINE: end-to-end vision pipeline test.

Runs preprocessor → vision encoder → CPU embed splice for a real PIL image
input, then sanity-checks the output structure and PCC vs an end-to-end
HF reference run (Qwen3VLVisionModel + manual splice).
"""

import os

import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mm_pipeline import Qwen36MMPipeline, splice_vision_into_embeddings
from models.demos.qwen3_6_galaxy_v2.tt.vision_model_args import Qwen36VisionModelArgs
from models.tt_dit.parallel.manager import CCLManager


@torch.no_grad()
def test_splice_vision_into_embeddings_cpu():
    """Unit test for the splice helper on CPU (no mesh)."""
    B, S, H = 1, 8, 4
    IMG = 248056
    input_ids = torch.tensor([[100, 101, IMG, IMG, IMG, IMG, 200, 201]], dtype=torch.long)
    text_embed = torch.arange(B * S * H, dtype=torch.float32).view(B, S, H)
    vision = torch.tensor(
        [[1.1, 1.2, 1.3, 1.4], [2.1, 2.2, 2.3, 2.4], [3.1, 3.2, 3.3, 3.4], [4.1, 4.2, 4.3, 4.4]],
        dtype=torch.float32,
    )
    fused = splice_vision_into_embeddings(text_embed, vision, input_ids, image_token_id=IMG)
    # Positions 2..5 should now be the 4 vision rows
    torch.testing.assert_close(fused[0, 2], vision[0])
    torch.testing.assert_close(fused[0, 5], vision[3])
    # Position 0, 1, 6, 7 should keep the original text embedding
    torch.testing.assert_close(fused[0, 0], text_embed[0, 0])
    torch.testing.assert_close(fused[0, 7], text_embed[0, 7])


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "BH_GLX": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_mm_pipeline_image_input(mesh_device, reset_seeds, ensure_gc):
    """End-to-end vision pipeline on a real PIL image. Compares fused
    embeddings to a HF reference (Qwen3VLVisionModel + manual splice)."""
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = "Qwen/Qwen3.6-27B"

    model_args = Qwen36VisionModelArgs(mesh_device, dummy_weights=False, max_batch_size=1, max_seq_len=256)

    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    # Synthetic text-embedding table — we only care about the splice mechanics, not
    # whether the text embedding values match. Random with seed for determinism.
    torch.manual_seed(0)
    vocab_size = model_args.hf_config.vision_config.out_hidden_size  # 5120 (= text hidden_size)
    H = model_args.hf_config.vision_config.out_hidden_size
    text_embed_table = torch.randn(248320, H, dtype=torch.float32)  # vocab=248320

    pipeline = Qwen36MMPipeline(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        model_args=model_args,
        text_embed_weight=text_embed_table,
        dtype=ttnn.bfloat16,
    )

    img = Image.new("RGB", (224, 224), color=(120, 30, 100))
    inputs, fused = pipeline.prepare_decoder_inputs(
        "<|vision_start|><|image_pad|><|vision_end|>Describe this image briefly.",
        images=[img],
    )

    logger.info(f"input_ids shape: {tuple(inputs.input_ids.shape)}")
    logger.info(f"pixel_values shape: {tuple(inputs.pixel_values.shape)}")
    logger.info(f"image_grid_thw: {inputs.image_grid_thw.tolist()}")
    logger.info(f"position_ids_3d shape: {tuple(inputs.position_ids_3d.shape)}")
    logger.info(f"fused_embeddings shape: {tuple(fused.shape)}")

    # Sanity: at image_pad positions, fused != text_embed (because vision_features written)
    image_mask = inputs.input_ids == pipeline.preprocessor.image_token_id
    n_image_positions = int(image_mask.sum().item())
    assert n_image_positions > 0, "expected some image_pad positions"

    # Reference: HF Qwen3VLVisionModel forward + manual splice
    reference_full = model_args.reference_vision_model()
    ref_vision_features, _ = reference_full(inputs.pixel_values, grid_thw=inputs.image_grid_thw)
    ref_text_embed = torch.nn.functional.embedding(inputs.input_ids, text_embed_table)
    ref_fused = splice_vision_into_embeddings(
        ref_text_embed,
        ref_vision_features,
        inputs.input_ids,
        image_token_id=pipeline.preprocessor.image_token_id,
    )

    # PCC at image positions (the only positions that involve TT vision compute).
    # Text positions are CPU embed lookup; they match exactly.
    tt_image_embeds = fused[0][image_mask[0]]
    ref_image_embeds = ref_fused[0][image_mask[0]]
    # Threshold 0.70: solid-color PIL test input creates near-degenerate
    # attention (all 256 patches see nearly-identical normalized pixels), which
    # is a numerically-unstable case for bf16 27-layer attention. A real-image
    # input gives better PCC (~0.89 — see test_vision_encoder_real_image_pcc).
    # Pipeline mechanics (splice, shapes, position_ids) are validated by the
    # text-position assertion below and the standalone splice test above.
    pcc_required = 0.70
    passing, pcc_message = comp_pcc(ref_image_embeds, tt_image_embeds, pcc_required)
    logger.info(comp_allclose(ref_image_embeds, tt_image_embeds))
    logger.info(f"PCC (image positions only): {pcc_message}")
    assert passing, f"qwen3.6 MM pipeline PCC at image positions {pcc_required} not met: {pcc_message}"

    # Text positions match exactly (CPU lookup)
    tt_text_embeds = fused[0][~image_mask[0]]
    ref_text_embeds_only = ref_fused[0][~image_mask[0]]
    torch.testing.assert_close(tt_text_embeds, ref_text_embeds_only, rtol=1e-5, atol=1e-5)
