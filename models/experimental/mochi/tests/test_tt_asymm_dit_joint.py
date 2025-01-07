import torch
import pytest
from loguru import logger
import os
import ttnn
from models.experimental.mochi.asymm_dit_joint import TtAsymmDiTJoint
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from genmo.mochi_preview.dit.joint_model.asymm_models_joint import AsymmDiTJoint as RefAsymmDiTJoint
from models.experimental.mochi.common import get_mochi_dir, get_cache_path, compute_metrics
from genmo.mochi_preview.pipelines import (
    get_conditioning,
    compute_packed_indices,
    t5_tokenizer,
    T5_MODEL,
)
from transformers import T5EncoderModel


def create_models(mesh_device, n_layers: int = 48):
    """Create and initialize both reference and TT models.

    Args:
        mesh_device: The mesh device to use for the TT model
        n_layers: Number of transformer layers to create

    Returns:
        tuple: (reference_model, tt_model, state_dict)
    """
    # Model configuration
    config = dict(
        depth=n_layers,
        patch_size=2,
        num_heads=24,
        hidden_size_x=3072,
        hidden_size_y=1536,
        mlp_ratio_x=4.0,
        mlp_ratio_y=4.0,
        in_channels=12,
        qk_norm=True,
        qkv_bias=False,
        out_bias=True,
        patch_embed_bias=True,
        timestep_mlp_bias=True,
        timestep_scale=1000.0,
        t5_feat_dim=4096,
        t5_token_length=256,
        rope_theta=10000.0,
        attention_mode="sdpa",
    )

    from safetensors.torch import load_file

    weights_path = os.path.join(get_mochi_dir(), "dit.safetensors")
    state_dict = load_file(weights_path)

    if n_layers < 48:
        # Filter state dict to only include the first n_layers
        first_block_names = [f"blocks.{i}." for i in range(n_layers)]
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if "blocks" not in k or any(k.startswith(lbn) for lbn in first_block_names)
        }

    # Create reference model
    reference_model = RefAsymmDiTJoint(**config)
    reference_model.load_state_dict(state_dict)

    # Create TT model
    weight_cache_path = get_cache_path(os.environ.get("FAKE_DEVICE"))
    tt_model = TtAsymmDiTJoint(
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        **config,
    )

    return reference_model, tt_model, state_dict


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize("n_layers", [1, 2, 48], ids=["L1", "L2", "L48"])
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_tt_asymm_dit_joint_inference(mesh_device, n_layers, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat16
    mesh_device.enable_async(True)

    # Create models using common function
    reference_model, tt_model, _ = create_models(mesh_device, n_layers)

    # Create input tensors
    batch_size = 1
    time_steps = 28
    height = 60
    width = 106
    PATCH_SIZE = 2

    num_visual_tokens = time_steps * height * width // (PATCH_SIZE**2)
    max_seqlen_in_batch_kv = num_visual_tokens + tt_model.t5_token_length

    x = torch.randn(batch_size, tt_model.in_channels, time_steps, height, width)
    sigma = torch.ones(batch_size)
    t5_feat = torch.randn(batch_size, tt_model.t5_token_length, tt_model.t5_feat_dim)
    t5_mask = torch.ones(batch_size, tt_model.t5_token_length).bool()

    # Create packed indices
    packed_indices = {
        "max_seqlen_in_batch_kv": max_seqlen_in_batch_kv,
        "valid_token_indices_kv": None,
        "cu_seqlens_kv": None,
    }

    logger.info("Run TtAsymmDiTJoint")
    tt_output = tt_model(
        x=x,
        sigma=sigma,
        y_feat=[t5_feat],
        y_mask=[t5_mask],
        packed_indices=packed_indices,
    )

    # Get reference output
    reference_output = reference_model(
        x=x,
        sigma=sigma,
        y_feat=[t5_feat],
        y_mask=[t5_mask],
        packed_indices=packed_indices,
    )

    # Compute metrics
    pcc, mse, mae = compute_metrics(reference_output, tt_output)

    # Check if model meets requirements
    pcc_required = 0.99
    passing = pcc >= pcc_required

    logger.info(comp_allclose(reference_output, tt_output))
    logger.info(f"PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    if passing:
        logger.info("TtAsymmDiTJoint Passed!")
    else:
        logger.warning("TtAsymmDiTJoint Failed!")

    assert (
        passing
    ), f"TtAsymmDiTJoint output does not meet PCC requirement {pcc_required}: {pcc}, MSE: {mse}, MAE: {mae}."


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
def test_tt_asymm_dit_joint_prepare(mesh_device, use_program_cache, reset_seeds):
    """Test that prepare() function produces identical outputs to reference implementation."""
    mesh_device.enable_async(True)

    # Create models using common function with 0 layers since we only need prepare()
    reference_model, tt_model, _ = create_models(mesh_device, n_layers=0)

    # Create input tensors
    batch_size = 1
    time_steps = 28
    height = 60
    width = 106

    x = torch.randn(batch_size, tt_model.in_channels, time_steps, height, width)
    sigma = torch.ones(batch_size)
    t5_feat = torch.randn(batch_size, tt_model.t5_token_length, tt_model.t5_feat_dim)
    t5_mask = torch.ones(batch_size, tt_model.t5_token_length).bool()

    # Run prepare() for both models
    tt_outputs = tt_model.prepare(x, sigma, t5_feat, t5_mask)
    ref_outputs = reference_model.prepare(x, sigma, t5_feat, t5_mask)

    # Check each output tensor
    for tt_out, ref_out, name in zip(tt_outputs, ref_outputs, ["x", "c", "y_feat", "rope_cos", "rope_sin"]):
        pcc = torch.corrcoef(torch.stack([tt_out.flatten(), ref_out.flatten()]))[0, 1].item()
        logger.info(f"{name} PCC: {pcc}")
        assert pcc > 0.99, f"prepare() {name} output does not match reference implementation"

    logger.info("prepare() outputs match reference implementation")


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
def test_tt_asymm_dit_joint_real_inputs(mesh_device, use_program_cache, reset_seeds, n_layers):
    """Test the model with real inputs processed just like in the pipeline."""
    dtype = ttnn.bfloat16
    cfg_scale = 6.0
    mesh_device.enable_async(True)

    # Create models using common function
    reference_model, tt_model, state_dict = create_models(mesh_device, n_layers)

    def model_fn(model, x, sigma, cond, is_tt_model):
        """Function to run model with both conditional and unconditional inputs.

        Args:
            model: Either reference_model or tt_model
            x: Input tensor
            sigma: Timestep tensor
            cond: Conditioning dict with y_feat, y_mask, and packed_indices

        Returns:
            Combined output using cfg_scale=1.0
        """
        # Run unconditional
        if is_tt_model:
            uncond_out = model(
                x=x,
                sigma=sigma,
                y_feat=cond["null"]["y_feat"],
                y_mask=cond["null"]["y_mask"],
                packed_indices=cond["null"]["packed_indices"],
                uncond=True,
            )
        else:
            uncond_out = model(
                x=x,
                sigma=sigma,
                y_feat=cond["null"]["y_feat"],
                y_mask=cond["null"]["y_mask"],
                packed_indices=cond["null"]["packed_indices"],
            )

        # Run conditional
        cond_out = model(
            x=x,
            sigma=sigma,
            y_feat=cond["cond"]["y_feat"],
            y_mask=cond["cond"]["y_mask"],
            packed_indices=cond["cond"]["packed_indices"],
        )

        # Combine outputs with cfg_scale=1.0 as in pipeline
        return uncond_out + cfg_scale * (cond_out - uncond_out), cond_out, uncond_out

    # Default arguments from cli.py
    width = 848
    height = 480
    num_frames = 163
    prompt = """A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl
    filled with lemons and sprigs of mint against a peach-colored background."""
    negative_prompt = ""

    # Constants from pipelines.py
    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 6
    IN_CHANNELS = 12
    batch_size = 1

    # Calculate latent dimensions
    latent_t = ((num_frames - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = width // SPATIAL_DOWNSAMPLE, height // SPATIAL_DOWNSAMPLE

    # Create input latents
    z = torch.randn(
        (batch_size, IN_CHANNELS, latent_t, latent_h, latent_w),
        dtype=torch.float32,
    )

    # Create text embeddings using T5
    device = torch.device("cpu")  # Keep on CPU as in pipeline
    tokenizer = t5_tokenizer(T5_MODEL)
    text_encoder = T5EncoderModel.from_pretrained(T5_MODEL)

    # Get conditioning using pipeline function
    conditioning = get_conditioning(
        tokenizer=tokenizer,
        encoder=text_encoder,
        device=device,
        batch_inputs=False,  # Single input mode
        prompt=prompt,
        negative_prompt=negative_prompt,
    )

    # Compute packed indices
    num_latents = latent_t * latent_h * latent_w
    for cond_type in ["cond", "null"]:
        conditioning[cond_type]["packed_indices"] = compute_packed_indices(
            device, conditioning[cond_type]["y_mask"][0], num_latents
        )

    # Create sigma (timestep)
    sigma = torch.ones(batch_size)

    logger.info("Run TtAsymmDiTJoint")
    tt_output, tt_cond_out, tt_uncond_out = model_fn(tt_model, z, sigma, conditioning, is_tt_model=True)

    logger.info("Run reference model")
    reference_output, reference_cond_out, reference_uncond_out = model_fn(
        reference_model, z, sigma, conditioning, is_tt_model=False
    )

    # Compute metrics for all outputs
    outputs = [
        (reference_output, tt_output, "combined"),
        (reference_cond_out, tt_cond_out, "conditional"),
        (reference_uncond_out, tt_uncond_out, "unconditional"),
    ]

    pcc_required = 0.99
    passing = True

    for ref, tt, name in outputs:
        pcc, mse, mae = compute_metrics(ref, tt)
        logger.info(f"\n{name.title()} Output Metrics:")
        logger.info(comp_allclose(ref, tt))
        logger.info(f"PCC: {pcc}, MSE: {mse}, MAE: {mae}")

        output_passing = pcc >= pcc_required
        passing = passing and output_passing

        if output_passing:
            logger.info(f"{name.title()} output Passed!")
        else:
            logger.warning(f"{name.title()} output Failed!")
            logger.warning(f"PCC {pcc} below required {pcc_required}")

    if passing:
        logger.info("\nAll TtAsymmDiTJoint outputs with real inputs Passed!")
    else:
        logger.warning("\nSome TtAsymmDiTJoint outputs with real inputs Failed!")

    assert passing, "One or more TtAsymmDiTJoint outputs did not meet PCC requirements"
