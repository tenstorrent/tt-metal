import torch
import pytest
from loguru import logger
import os
import ttnn
import numpy as np
from models.utility_functions import skip_for_grayskull
from models.experimental.mochi.common import compute_metrics
from genmo.mochi_preview.pipelines import sample_model as reference_sample_model
from models.experimental.mochi.pipelines_tt import sample_model_tt
from genmo.mochi_preview.pipelines import (
    linear_quadratic_schedule,
    get_conditioning,
    t5_tokenizer,
    T5_MODEL,
)
from transformers import T5EncoderModel
from models.experimental.mochi.tests.test_tt_asymm_dit_joint import create_models


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("n_layers", [1, 2, 4, 48], ids=["L1", "L2", "L4", "L48"])
def test_sample_model(mesh_device, use_program_cache, reset_seeds, n_layers):
    """Test TensorTorch sample_model against reference implementation."""
    mesh_device.enable_async(True)
    device = torch.device("cpu")

    # Create models
    ref_dit, tt_dit, _ = create_models(mesh_device, n_layers)

    # Test parameters
    width = 848
    height = 480
    num_frames = 163
    num_steps = 1
    cfg_scale = 6.0
    prompt = """A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl
    filled with lemons and sprigs of mint against a peach-colored background."""
    negative_prompt = ""

    # Create schedules
    # sigma_schedule = linear_quadratic_schedule(num_steps, 0.025)
    sigma_schedule = linear_quadratic_schedule(num_steps + 1, 0.025)[: num_steps + 1]
    cfg_schedule = [cfg_scale] * num_steps

    # Get text conditioning
    tokenizer = t5_tokenizer(T5_MODEL)
    text_encoder = T5EncoderModel.from_pretrained(T5_MODEL)
    text_encoder.eval()
    conditioning = get_conditioning(
        tokenizer=tokenizer,
        encoder=text_encoder,
        device=torch.device("cpu"),  # Keep on CPU as in pipeline
        batch_inputs=False,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )

    # Common arguments for both sample_model functions
    sample_args = {
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "sigma_schedule": sigma_schedule,
        "cfg_schedule": cfg_schedule,
        "num_inference_steps": num_steps,
        "seed": 1234,
    }

    # Run both implementations
    logger.info("Running reference sample_model")
    reference_output = reference_sample_model(device=device, dit=ref_dit, conditioning=conditioning, **sample_args)

    logger.info("Running TT sample_model")
    tt_output = sample_model_tt(device=device, dit=tt_dit, conditioning=conditioning, **sample_args)

    # Compute metrics
    pcc_required = 0.985
    pcc, mse, mae = compute_metrics(reference_output, tt_output)

    logger.info(f"Sample Model Output Metrics:")
    logger.info(f"PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    passing = pcc >= pcc_required

    if passing:
        logger.info("Sample model test Passed!")
    else:
        logger.warning("Sample model test Failed!")
        logger.warning(f"PCC {pcc} below required {pcc_required}")

    assert passing, f"Sample model output does not meet PCC requirement {pcc_required}"
