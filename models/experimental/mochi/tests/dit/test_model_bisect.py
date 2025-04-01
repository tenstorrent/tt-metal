import torch
import pytest
from loguru import logger
import os
import json
from pathlib import Path
import ttnn
from models.experimental.mochi.asymm_dit_joint import TtAsymmDiTJoint as TtAsymmDiTJoint

# from models.experimental.mochi.tt.dit.model import AsymmDiTJoint as TtAsymmDiTJoint
from models.utility_functions import (
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from genmo.mochi_preview.dit.joint_model.asymm_models_joint import AsymmDiTJoint as RefAsymmDiTJoint
from models.experimental.mochi.common import get_mochi_dir, get_cache_path, compute_metrics

# from models.experimental.mochi.tt.common import get_mochi_dir, get_cache_path, compute_metrics
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
@pytest.mark.parametrize("n_layers", [1, 2], ids=["L1", "L2"])
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

    # Create input tensors
    batch_size = 1
    time_steps = 28
    height = 60
    width = 106
    PATCH_SIZE = 2

    num_visual_tokens = time_steps * height * width // (PATCH_SIZE**2)

    # Create models using common function
    reference_model, tt_model, _ = create_models(mesh_device, n_layers)

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
    tt_output = tt_model.forward(
        x=x,
        sigma=sigma,
        y_feat=[t5_feat],
        y_mask=[t5_mask],
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

    # Create input tensors
    batch_size = 1
    time_steps = 28
    height = 60
    width = 106
    PATCH_SIZE = 2

    num_visual_tokens = time_steps * height * width // (PATCH_SIZE**2)

    # Create models using common function with 0 layers since we only need prepare()
    reference_model, tt_model, _ = create_models(mesh_device, n_layers=0)

    x = torch.randn(batch_size, tt_model.in_channels, time_steps, height, width)
    sigma = torch.ones(batch_size)
    t5_feat = torch.randn(batch_size, tt_model.t5_token_length, tt_model.t5_feat_dim)
    t5_mask = torch.ones(batch_size, tt_model.t5_token_length).bool()

    # Run prepare() for both models
    tt_outputs = tt_model.prepare(x, sigma, t5_feat, t5_mask)
    ref_outputs = reference_model.prepare(x, sigma, t5_feat, t5_mask)

    x, c, y_feat, rope_cos, rope_sin, _, _ = tt_outputs
    x = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-2))
    x = x[:, :, :num_visual_tokens]
    c = ttnn.to_torch(c, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]
    y_feat = ttnn.to_torch(y_feat, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]
    rope_cos = ttnn.to_torch(rope_cos, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-3))
    rope_sin = ttnn.to_torch(rope_sin, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-3))

    # Undo cos/sin permutation and stacking
    def unstack(x):
        x = x.permute(0, 2, 1, 3).squeeze(0)  # N, num_heads, head_dim
        x = x.reshape(*x.shape[:-1], -1, 2)
        return x[..., 0]

    rope_cos = unstack(rope_cos)
    rope_sin = unstack(rope_sin)
    tt_outputs = [x, c, y_feat, rope_cos, rope_sin]

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
@pytest.mark.parametrize("n_layers", [1, 48], ids=["L1", "L48"])
def test_tt_asymm_dit_joint_model_fn(mesh_device, use_program_cache, reset_seeds, n_layers):
    """Test the model with real inputs processed just like in the pipeline."""
    dtype = ttnn.bfloat16
    cfg_scale = 6.0
    sigma_schedule = [0.2, 0.1]
    dsigma = sigma_schedule[0] - sigma_schedule[1]
    mesh_device.enable_async(True)

    def model_fn(model, x, sigma, cond):
        """Function to run model with both conditional and unconditional inputs.

        Args:
            model: Either reference_model or tt_model
            x: Input tensor
            sigma: Timestep tensor
            cond: Conditioning dict with y_feat, y_mask, and packed_indices

        Returns:
            Combined output using cfg_scale=1.0
        """
        uncond_out = model(
            x=x,
            sigma=sigma,
            y_feat=cond["null"]["y_feat"],
            y_mask=cond["null"]["y_mask"],
            packed_indices=cond["null"]["packed_indices"],
        )
        cond_out = model(
            x=x,
            sigma=sigma,
            y_feat=cond["cond"]["y_feat"],
            y_mask=cond["cond"]["y_mask"],
            packed_indices=cond["cond"]["packed_indices"],
        )

        # Combine outputs with cfg_scale=1.0 as in pipeline
        pred = uncond_out + cfg_scale * (cond_out - uncond_out)
        z = x + dsigma * pred
        return z, pred, cond_out, uncond_out

    def model_fn_tt(model, z_BCTHW, sigma, cond):
        T, H, W = z_BCTHW.shape[-3:]
        cond_text = cond["cond"]
        cond_null = cond["null"]
        tt_cfg_scale = ttnn.from_torch(
            torch.tensor(cfg_scale),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.float32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        tt_dsigma = ttnn.from_torch(
            torch.tensor(dsigma),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.float32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        rope_cos_1HND, rope_sin_1HND, trans_mat = model.prepare_rope_features(T=latent_t, H=latent_h, W=latent_w)
        # Note that conditioning contains list of len 1 to index into
        cond_y_feat_1BLY, cond_y_pool_11BX = model.prepare_text_features(
            t5_feat=cond_text["y_feat"][0], t5_mask=cond_text["y_mask"][0]
        )
        uncond_y_feat_1BLY, uncond_y_pool_11BX = model.prepare_text_features(
            t5_feat=cond_null["y_feat"][0], t5_mask=cond_null["y_mask"][0]
        )
        # z_1BNI, N = model.preprocess_input(z_BCTHW)
        z_1BNI = model.preprocess_input(z_BCTHW)

        cond_z_1BNI = model.forward_inner(
            x_1BNI=z_1BNI,
            sigma=sigma,
            y_feat_1BLY=cond_y_feat_1BLY,
            y_pool_11BX=cond_y_pool_11BX,
            rope_cos_1HND=rope_cos_1HND,
            rope_sin_1HND=rope_sin_1HND,
            trans_mat=trans_mat,
            # N=N,
            uncond=False,
        )

        uncond_z_1BNI = model.forward_inner(
            x_1BNI=z_1BNI,
            sigma=sigma,
            y_feat_1BLY=uncond_y_feat_1BLY,
            y_pool_11BX=uncond_y_pool_11BX,
            rope_cos_1HND=rope_cos_1HND,
            rope_sin_1HND=rope_sin_1HND,
            trans_mat=trans_mat,
            # N=num_visual_tokens,
            uncond=True,
        )

        assert cond_z_1BNI.shape == uncond_z_1BNI.shape

        # TODO: need to update pred in fp32 for correctness?
        # Push to float32 for higher precision
        cond_z_1BNI = ttnn.typecast(cond_z_1BNI, dtype=ttnn.float32)
        uncond_z_1BNI = ttnn.typecast(uncond_z_1BNI, dtype=ttnn.float32)
        pred = uncond_z_1BNI + tt_cfg_scale * (cond_z_1BNI - uncond_z_1BNI)

        print(pred.shape)
        print(z_1BNI.shape)
        z_1BNI = ttnn.typecast(z_1BNI, dtype=ttnn.float32)

        z = z_1BNI + tt_dsigma * pred

        torch_output = model.reverse_preprocess(z, T, H, W)

        torch_pred = model.reverse_preprocess(pred, T, H, W)
        torch_cond = model.reverse_preprocess(cond_z_1BNI, T, H, W)
        torch_uncond = model.reverse_preprocess(uncond_z_1BNI, T, H, W)
        # torch_output = model.reverse_preprocess(z, T, H, W, N)

        # torch_pred = model.reverse_preprocess(pred, T, H, W, N)
        # torch_cond = model.reverse_preprocess(cond_z_1BNI, T, H, W, N)
        # torch_uncond = model.reverse_preprocess(uncond_z_1BNI, T, H, W, N)
        return torch_output, torch_pred, torch_cond, torch_uncond

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

    PATCH_SIZE = 2
    num_visual_tokens = latent_t * latent_h * latent_w // (PATCH_SIZE**2)

    # Create models using common function
    reference_model, tt_model, _ = create_models(mesh_device, n_layers)

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
    sigma = torch.full((batch_size,), sigma_schedule[0])

    logger.info("Run TtAsymmDiTJoint")
    tt_output, tt_pred, tt_cond_out, tt_uncond_out = model_fn_tt(tt_model, z, sigma, conditioning)

    torch_tt_output = z + dsigma * tt_pred

    if os.environ.get("SAVE_PT_TENSORS", "0") == "1":
        logger.info("Run reference model")
        reference_output, reference_pred, reference_cond_out, reference_uncond_out = model_fn(
            reference_model, z, sigma, conditioning
        )
    else:
        logger.info("Load reference model outputs")
        reference_output = torch.load("combined_ref.pt")
        reference_pred = torch.load("pred_ref.pt")
        reference_cond_out = torch.load("conditional_ref.pt")
        reference_uncond_out = torch.load("unconditional_ref.pt")

    # Compute metrics for all outputs
    outputs = [
        (reference_output, torch_tt_output, "combined_tt_on_host"),
        (reference_output, tt_output, "combined"),
        (reference_pred, tt_pred, "pred"),
        (reference_cond_out, tt_cond_out, "conditional"),
        (reference_uncond_out, tt_uncond_out, "unconditional"),
    ]

    if os.environ.get("SAVE_PT_TENSORS", "0") == "1":
        for ref, tt, name in outputs[1:]:
            torch.save(ref, f"{name}_ref.pt")

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
@pytest.mark.parametrize("n_layers", [1, 48], ids=["L1", "L48"])
def test_model_fn_no_e2e_optim(mesh_device, use_program_cache, reset_seeds, n_layers):
    """Test the model with real inputs processed just like in the pipeline."""
    dtype = ttnn.bfloat16
    cfg_scale = 6.0
    sigma_schedule = [0.2, 0.1]
    dsigma = sigma_schedule[0] - sigma_schedule[1]
    mesh_device.enable_async(True)

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
            cond_out = model(
                x=x,
                sigma=sigma,
                y_feat=cond["cond"]["y_feat"],
                y_mask=cond["cond"]["y_mask"],
                packed_indices=cond["cond"]["packed_indices"],
            )
        else:
            uncond_out = model(
                x=x,
                sigma=sigma,
                y_feat=cond["null"]["y_feat"],
                y_mask=cond["null"]["y_mask"],
                packed_indices=cond["null"]["packed_indices"],
            )
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

    PATCH_SIZE = 2
    num_visual_tokens = latent_t * latent_h * latent_w // (PATCH_SIZE**2)

    # Create models using common function
    reference_model, tt_model, _ = create_models(mesh_device, n_layers)

    # Create input latents
    # z = torch.randn(
    #     (batch_size, IN_CHANNELS, latent_t, latent_h, latent_w),
    #     dtype=torch.float32,
    # )
    z = torch.load("z_input.pt")
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
    sigma = torch.full((batch_size,), sigma_schedule[0])

    logger.info("Run TtAsymmDiTJoint")
    tt_pred, tt_cond_out, tt_uncond_out = model_fn(tt_model, z, sigma, conditioning, is_tt_model=True)

    # torch_tt_output = z + dsigma * tt_pred

    # if os.environ.get("SAVE_PT_TENSORS", "0") == "1":
    #     logger.info("Run reference model")
    #     reference_pred, reference_cond_out, reference_uncond_out = model_fn(
    #         reference_model, z, sigma, conditioning, is_tt_model=False
    #     )
    # else:
    logger.info("Load reference model outputs")
    reference_pred = torch.load("pred_ref.pt")
    reference_cond_out = torch.load("conditional_ref.pt")
    reference_uncond_out = torch.load("unconditional_ref.pt")

    # Compute metrics for all outputs
    outputs = [
        (reference_pred, tt_pred, "pred"),
        (reference_cond_out, tt_cond_out, "conditional"),
        (reference_uncond_out, tt_uncond_out, "unconditional"),
    ]

    # if os.environ.get("SAVE_PT_TENSORS", "0") == "1":
    #     for ref, tt, name in outputs:
    #         torch.save(ref, f"{name}_ref.pt")

    # logger.info("Run reference model")
    # reference_pred, reference_cond_out, reference_uncond_out = model_fn(
    #     reference_model, z, sigma, conditioning, is_tt_model=False
    # )
    # # Compute metrics for all outputs
    # outputs = [
    #     (reference_pred, tt_pred, "pred"),
    #     (reference_cond_out, tt_cond_out, "conditional"),
    #     (reference_uncond_out, tt_uncond_out, "unconditional"),
    # ]

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
@pytest.mark.parametrize("n_layers", [1, 5, 48], ids=["L1", "L5", "L48"])
def test_each_step(mesh_device, use_program_cache, reset_seeds, n_layers):
    """Test the model with real inputs processed just like in the pipeline."""
    mesh_device.enable_async(True)

    def model_fn(model, x, sigma, cond_text, cond_null, cfg_scale, is_tt_model):
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
                y_feat=cond_null["y_feat"],
                y_mask=cond_null["y_mask"],
                # packed_indices=cond_null["packed_indices"],
                uncond=True,
            )
            cond_out = model(
                x=x,
                sigma=sigma,
                y_feat=cond_text["y_feat"],
                y_mask=cond_text["y_mask"],
                # packed_indices=cond_text["packed_indices"],
            )
        else:
            assert False
            # uncond_out = model(
            #     x=x,
            #     sigma=sigma,
            #     y_feat=cond["null"]["y_feat"],
            #     y_mask=cond["null"]["y_mask"],
            #     packed_indices=cond["null"]["packed_indices"],
            # )
            # cond_out = model(
            #     x=x,
            #     sigma=sigma,
            #     y_feat=cond["cond"]["y_feat"],
            #     y_mask=cond["cond"]["y_mask"],
            #     packed_indices=cond["cond"]["packed_indices"],
            # )

        # Combine outputs with cfg_scale=1.0 as in pipeline
        pred = uncond_out + cfg_scale * (cond_out - uncond_out)

        return pred, cond_out, uncond_out

    # Create models using common function
    reference_model, tt_model, _ = create_models(mesh_device, n_layers)

    batch_size = 1

    saved_outputs_dir = Path("../saved_outputs")

    # z = torch.load(saved_outputs_dir / "z_input.pt")
    cond_text = torch.load(saved_outputs_dir / "cond_text.pt", map_location="cpu")
    cond_null = torch.load(saved_outputs_dir / "cond_null.pt", map_location="cpu")

    args = json.load(open(saved_outputs_dir / "generation_config.json"))
    height = args["height"]
    width = args["width"]
    num_frames = args["num_frames"]
    sigma_schedule = args["sigma_schedule"]
    cfg_schedule = args["cfg_schedule"]
    num_inference_steps = args["num_inference_steps"]
    seed = args["seed"]
    # Create sigma (timestep)
    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 6
    IN_CHANNELS = 12
    batch_size = 1
    PATCH_SIZE = 2
    # Calculate latent dimensions
    latent_t = ((num_frames - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = width // SPATIAL_DOWNSAMPLE, height // SPATIAL_DOWNSAMPLE
    num_visual_tokens = latent_t * latent_h * latent_w // (PATCH_SIZE**2)
    # Input text conditioning is padded to 256. Must remove padding to make JointAttention
    # aware of true sequence length!
    num_text_tokens = cond_text["packed_indices"]["max_seqlen_in_batch_kv"] - num_visual_tokens
    cond_text["y_feat"][0] = cond_text["y_feat"][0][:, :num_text_tokens, :]
    cond_text["y_mask"][0] = cond_text["y_mask"][0][:, :num_text_tokens]
    z = torch.load(saved_outputs_dir / "in_z_0.pt", map_location="cpu")

    logger.info("Run TtAsymmDiTJoint")
    for i in range(num_inference_steps):
        print(f"Step {i} of {num_inference_steps}")
        sigma = sigma_schedule[i]
        cfg_scale = cfg_schedule[i]
        dsigma = sigma_schedule[i] - sigma_schedule[i + 1]
        print(f"cfg: {cfg_scale}")
        print(f'sigma: {sigma} vs reference: {torch.load(saved_outputs_dir / f"sigma_{i}.pt")}')
        print(f'dsigma: {dsigma} vs reference: {torch.load(saved_outputs_dir / f"dsigma_{i}.pt")}')
        compare_to_reference(z, saved_outputs_dir / f"in_z_{i}.pt")
        sigma = torch.full((batch_size,), sigma)

        pred, cond_out, uncond_out = model_fn(tt_model, z, sigma, cond_text, cond_null, cfg_scale, is_tt_model=True)
        z = z + dsigma * pred

        compare_to_reference(cond_out, saved_outputs_dir / f"out_cond_{i}.pt")
        compare_to_reference(uncond_out, saved_outputs_dir / f"out_uncond_{i}.pt")
        compare_to_reference(pred, saved_outputs_dir / f"out_pred_{i}.pt")
        compare_to_reference(z, saved_outputs_dir / f"out_z_{i}.pt")


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
@pytest.mark.parametrize("n_layers", [1, 48], ids=["L1", "L48"])
@pytest.mark.parametrize("is_teacher_forcing", [True, False], ids=["teacher_forcing", "no_forcing"])
def test_each_step_post_e2e_optim(mesh_device, use_program_cache, reset_seeds, n_layers, is_teacher_forcing):
    """Test the model with real inputs processed just like in the pipeline."""
    mesh_device.enable_async(True)

    def model_fn_tt(model, z_BCTHW, sigma, cond_text, cond_null, cfg_scale, dsigma):
        T, H, W = z_BCTHW.shape[-3:]
        # tt_cfg_scale = ttnn.from_torch(
        #     torch.tensor(cfg_scale),
        #     device=mesh_device,
        #     layout=ttnn.TILE_LAYOUT,
        #     dtype=ttnn.float32,
        #     mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        # )
        # tt_dsigma = ttnn.from_torch(
        #     torch.tensor(dsigma),
        #     device=mesh_device,
        #     layout=ttnn.TILE_LAYOUT,
        #     dtype=ttnn.float32,
        #     mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        # )
        rope_cos_1HND, rope_sin_1HND, trans_mat = model.prepare_rope_features(T=latent_t, H=latent_h, W=latent_w)
        # Note that conditioning contains list of len 1 to index into
        cond_y_feat_1BLY, cond_y_pool_11BX = model.prepare_text_features(
            t5_feat=cond_text["y_feat"][0], t5_mask=cond_text["y_mask"][0]
        )
        uncond_y_feat_1BLY, uncond_y_pool_11BX = model.prepare_text_features(
            t5_feat=cond_null["y_feat"][0], t5_mask=cond_null["y_mask"][0]
        )
        # z_1BNI, N = model.preprocess_input(z_BCTHW)
        z_1BNI = model.preprocess_input(z_BCTHW)

        cond_z_1BNI = model.forward_inner(
            x_1BNI=z_1BNI,
            sigma=sigma,
            y_feat_1BLY=cond_y_feat_1BLY,
            y_pool_11BX=cond_y_pool_11BX,
            rope_cos_1HND=rope_cos_1HND,
            rope_sin_1HND=rope_sin_1HND,
            trans_mat=trans_mat,
            # N=num_visual_tokens,
            uncond=False,
        )

        uncond_z_1BNI = model.forward_inner(
            x_1BNI=z_1BNI,
            sigma=sigma,
            y_feat_1BLY=uncond_y_feat_1BLY,
            y_pool_11BX=uncond_y_pool_11BX,
            rope_cos_1HND=rope_cos_1HND,
            rope_sin_1HND=rope_sin_1HND,
            trans_mat=trans_mat,
            # N=num_visual_tokens,
            uncond=True,
        )

        assert cond_z_1BNI.shape == uncond_z_1BNI.shape

        # TODO: need to update pred in fp32 for correctness?
        # Push to float32 for higher precision
        # cond_z_1BNI = ttnn.typecast(cond_z_1BNI, dtype=ttnn.float32)
        # uncond_z_1BNI = ttnn.typecast(uncond_z_1BNI, dtype=ttnn.float32)
        # pred = uncond_z_1BNI + tt_cfg_scale * (cond_z_1BNI - uncond_z_1BNI)
        # pred = uncond_z_1BNI + cfg_scale * (cond_z_1BNI - uncond_z_1BNI)

        # # z_1BNI = ttnn.typecast(z_1BNI, dtype=ttnn.float32)

        # # z = z_1BNI + tt_dsigma * pred
        # z = z_1BNI + dsigma * pred

        # torch_output = model.reverse_preprocess(z, T, H, W)
        # torch_pred = model.reverse_preprocess(pred, T, H, W)
        torch_cond = model.reverse_preprocess(cond_z_1BNI, T, H, W)
        torch_uncond = model.reverse_preprocess(uncond_z_1BNI, T, H, W)

        torch_pred = torch_uncond + cfg_scale * (torch_cond - torch_uncond)
        torch_output = z_BCTHW + dsigma * torch_pred

        # torch_output = model.reverse_preprocess(z, T, H, W, N)
        # torch_pred = model.reverse_preprocess(pred, T, H, W, N)
        # torch_cond = model.reverse_preprocess(cond_z_1BNI, T, H, W, N)
        # torch_uncond = model.reverse_preprocess(uncond_z_1BNI, T, H, W, N)
        return torch_output, torch_pred, torch_cond, torch_uncond

    # Create models using common function
    reference_model, tt_model, _ = create_models(mesh_device, n_layers)

    batch_size = 1
    # Constants from pipelines.py
    saved_outputs_dir = Path("../saved_outputs")

    # z = torch.load(saved_outputs_dir / "z_input.pt")
    cond_text = torch.load(saved_outputs_dir / "cond_text.pt", map_location="cpu")
    cond_null = torch.load(saved_outputs_dir / "cond_null.pt", map_location="cpu")

    args = json.load(open(saved_outputs_dir / "generation_config.json"))
    height = args["height"]
    width = args["width"]
    num_frames = args["num_frames"]
    sigma_schedule = args["sigma_schedule"]
    cfg_schedule = args["cfg_schedule"]
    num_inference_steps = args["num_inference_steps"]

    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 6
    IN_CHANNELS = 12
    batch_size = 1
    PATCH_SIZE = 2
    # Calculate latent dimensions
    latent_t = ((num_frames - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = width // SPATIAL_DOWNSAMPLE, height // SPATIAL_DOWNSAMPLE
    num_visual_tokens = latent_t * latent_h * latent_w // (PATCH_SIZE**2)

    num_text_tokens = cond_text["packed_indices"]["max_seqlen_in_batch_kv"] - num_visual_tokens
    cond_text["y_feat"][0] = cond_text["y_feat"][0][:, :num_text_tokens, :]
    cond_text["y_mask"][0] = cond_text["y_mask"][0][:, :num_text_tokens]

    z = torch.load(saved_outputs_dir / "in_z_0.pt", map_location="cpu")

    logger.info("Run TtAsymmDiTJoint")
    for i in range(num_inference_steps):
        print(f"Step {i} of {num_inference_steps}")
        sigma = sigma_schedule[i]
        cfg_scale = cfg_schedule[i]
        dsigma = sigma_schedule[i] - sigma_schedule[i + 1]
        print(f"cfg: {cfg_scale}")
        print(f'sigma: {sigma} vs reference: {torch.load(saved_outputs_dir / f"sigma_{i}.pt")}')
        print(f'dsigma: {dsigma} vs reference: {torch.load(saved_outputs_dir / f"dsigma_{i}.pt")}')
        if is_teacher_forcing:
            z = torch.load(saved_outputs_dir / f"in_z_{i}.pt", map_location="cpu")
        else:
            compare_to_reference(z, saved_outputs_dir / f"in_z_{i}.pt")
        sigma = torch.full((batch_size,), sigma)

        z, pred, cond_out, uncond_out = model_fn_tt(tt_model, z, sigma, cond_text, cond_null, cfg_scale, dsigma)

        compare_to_reference(cond_out, saved_outputs_dir / f"out_cond_{i}.pt")
        compare_to_reference(uncond_out, saved_outputs_dir / f"out_uncond_{i}.pt")
        compare_to_reference(pred, saved_outputs_dir / f"out_pred_{i}.pt")
        compare_to_reference(z, saved_outputs_dir / f"out_z_{i}.pt")


def compare_to_reference(tt, ref_path):
    ref = torch.load(ref_path, map_location="cpu")
    pcc, mse, mae = compute_metrics(ref, tt)
    logger.info(f"\n{ref_path.stem} Output Metrics:")
    logger.info(comp_allclose(ref, tt))
    logger.info(f"PCC: {pcc}, MSE: {mse}, MAE: {mae}")


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
def test_tt_asymm_dit_joint_preprocess(mesh_device, use_program_cache, reset_seeds):
    """Test that we can reverse the preprocessing of input"""
    mesh_device.enable_async(True)
    B, C, T, H, W = 1, 12, 28, 60, 106
    PATCH_SIZE = 2
    num_visual_tokens = T * H * W // (PATCH_SIZE**2)
    _, tt_model, _ = create_models(mesh_device, n_layers=0)
    x = torch.randn(B, C, T, H, W)
    x_1BNI, N = tt_model.preprocess_input(x)
    x_back = tt_model.reverse_preprocess(x_1BNI, T, H, W, N)
    pcc, mse, mae = compute_metrics(x, x_back)
    logger.info(f"PCC: {pcc}, MSE: {mse}, MAE: {mae}")
    assert pcc >= 0.99
