# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from safetensors.torch import load_file as safetensors_load_file
from huggingface_hub import hf_hub_download
from models.common.utility_functions import comp_pcc
from models.experimental.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.experimental.tt_dit.models.transformers.sd35_med.transformer_sd35_medium import SD35MediumMMDiTX
from models.experimental.tt_dit.tests.models.sd35_med.test_patch_embed_sd35_medium import PatchEmbedRef
from models.experimental.tt_dit.tests.models.sd35_med.test_timestep_embed_sd35_medium import TimestepEmbedderRef
from models.experimental.tt_dit.tests.models.sd35_med.test_vector_embed_sd35_medium import VectorEmbedderRef
from models.experimental.tt_dit.tests.models.sd35_med.test_joint_block_sd35_medium import JointBlock
from models.experimental.tt_dit.tests.models.sd35_med.test_final_layer_sd35_medium import FinalLayer


class ReferenceMMDiTX(torch.nn.Module):
    """Reference PyTorch implementation matching MM-DiT for SD3.5 Medium"""

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 16,
        depth: int = 28,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = False,
        out_channels: int = 16,
        pos_embed_max_size: int = 32,
        num_patches: int = None,
        qkv_bias: bool = True,
        dtype=torch.bfloat16,
        device=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.pos_embed_max_size = pos_embed_max_size

        # SD3.5 Medium uses fixed architecture values
        hidden_size = 1536  # SD3.5 Medium fixed hidden size
        num_heads = 24  # SD3.5 Medium fixed number of heads
        self.num_heads = num_heads

        # Patch embedding: Conv2d(16, 1536, kernel_size=(2, 2), stride=(2, 2))
        self.x_embedder = PatchEmbedRef(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size,
            bias=True,
            flatten=True,
        )

        # Timestep embedding: MLP(256 -> 1536 -> 1536)
        self.t_embedder = TimestepEmbedderRef(hidden_size, frequency_embedding_size=256, dtype=dtype)

        # Class embedding: MLP(2048 -> 1536 -> 1536) - always present
        self.y_embedder = VectorEmbedderRef(input_dim=2048, hidden_size=hidden_size, dtype=dtype)

        # Context embedding: Linear(4096, 1536)
        self.context_embedder = torch.nn.Linear(
            in_features=4096,
            out_features=hidden_size,
            bias=True,
            dtype=dtype,
            device=device,
        )

        # Position embedding
        if num_patches is not None:
            self.register_buffer(
                "pos_embed",
                torch.zeros(1, num_patches, hidden_size, dtype=dtype, device=device),
            )
        else:
            self.pos_embed = None

        # Joint blocks: 28 blocks total
        # Blocks 0-12: 13 blocks with dual attention (x_block_self_attn=True)
        # Blocks 13-22: 10 blocks with single attention (x_block_self_attn=False)
        # Block 23: Last block with pre_only context_block (pre_only=True, x_block_self_attn=False)
        self.joint_blocks = torch.nn.ModuleList()
        for i in range(depth):
            if i < 13:
                # Blocks 0-12: dual attention
                x_block_self_attn = True
                pre_only = False
            elif i < 23:
                # Blocks 13-22: single attention
                x_block_self_attn = False
                pre_only = False
            else:
                # Block 23: last block, pre_only context_block
                x_block_self_attn = False
                pre_only = True

            self.joint_blocks.append(
                JointBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    pre_only=pre_only,
                    rmsnorm=False,  # LayerNorm, not RMSNorm
                    scale_mod_only=False,
                    swiglu=False,  # GELU, not SwiGLU
                    qk_norm="rms",  # RMSNorm for QK
                    x_block_self_attn=x_block_self_attn,
                )
            )

        # Final layer: norm_final, linear(1536 -> 64), adaLN_modulation(1536 -> 3072)
        # 64 = patch_size * patch_size * out_channels = 2 * 2 * 16
        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=out_channels,
            dtype=dtype,
            device=device,
        )

    def cropped_pos_embed(self, hw):
        assert self.pos_embed_max_size is not None
        p = self.patch_size
        h, w = hw
        h = h // p
        w = w // p

        # Calculate actual spatial dimensions from pos_embed shape
        # pos_embed shape is [1, num_patches, hidden_size]
        num_patches = self.pos_embed.shape[1]
        spatial_size = int(num_patches**0.5)  # Assuming square grid
        assert spatial_size * spatial_size == num_patches, f"num_patches {num_patches} is not a perfect square"

        assert h <= spatial_size, (h, spatial_size)
        assert w <= spatial_size, (w, spatial_size)
        top = (spatial_size - h) // 2
        left = (spatial_size - w) // 2

        from einops import rearrange

        spatial_pos_embed = rearrange(
            self.pos_embed,
            "1 (h w) c -> 1 h w c",
            h=spatial_size,
            w=spatial_size,
        )
        spatial_pos_embed = spatial_pos_embed[:, top : top + h, left : left + w, :]
        spatial_pos_embed = rearrange(spatial_pos_embed, "1 h w c -> 1 (h w) c")
        return spatial_pos_embed

    def unpatchify(self, x, hw=None):
        c = self.out_channels
        p = self.patch_size
        if hw is None:
            h = w = int(x.shape[1] ** 0.5)
        else:
            h, w = hw
            h = h // p
            w = w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_core_with_concat(self, x, c_mod, context=None, skip_layers=[]):
        for i, block in enumerate(self.joint_blocks):
            if i in skip_layers:
                continue
            context, x = block(context, x, c=c_mod)

        x = self.final_layer(x, c_mod)
        return x

    def forward(self, x, t, y=None, context=None, skip_layers=[]):
        hw = x.shape[-2:]
        x = self.x_embedder(x)
        if self.pos_embed is not None:
            x = x + self.cropped_pos_embed(hw)

        c = self.t_embedder(t)
        if y is not None:
            y_emb = self.y_embedder(y)
            c = c + y_emb

        if context is not None:
            context = self.context_embedder(context)
        else:
            context = None

        x = self.forward_core_with_concat(x, c, context, skip_layers)
        x = self.unpatchify(x, hw=hw)
        return x


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "input_size, patch_size, in_channels, depth, seq_len, batch_size",
    [
        (32, 2, 16, 28, 256, 1),  # SD3.5 Medium config
    ],
    ids=["sd35_medium"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_sd35_medium_mmdit(device, dtype, input_size, patch_size, in_channels, depth, seq_len, batch_size, reset_seeds):
    """Test SD3.5 Medium MMDiT forward pass"""
    torch.manual_seed(1234)

    # SD3.5 Medium uses fixed architecture values
    hidden_size = 1536  # SD3.5 Medium fixed hidden size
    num_heads = 24  # SD3.5 Medium fixed number of heads
    num_patches = (input_size // patch_size) ** 2

    # Load MMDiT weights directly from safetensors file
    model_checkpoint_path = "stabilityai/stable-diffusion-3.5-medium"
    safetensors_filename = "sd3.5_medium.safetensors"
    logger.info(f"Loading MMDiT weights from safetensors file: {model_checkpoint_path}/{safetensors_filename}")

    # Download the safetensors file from Hugging Face
    safetensors_path = hf_hub_download(
        repo_id=model_checkpoint_path,
        filename=safetensors_filename,
        repo_type="model",
    )
    logger.info(f"Downloaded safetensors file to: {safetensors_path}")

    # Load the safetensors file
    all_weights = safetensors_load_file(safetensors_path)
    logger.info(f"Loaded {len(all_weights)} total keys from safetensors file")

    # Filter only MMDiT-related weights
    # The safetensors file contains weights for the entire SD3.5 model
    # MMDiT weights are prefixed with "model.diffusion_model."
    mmdit_weights = {}
    mmdit_prefix = "model.diffusion_model."

    # Expected MMDiT component prefixes in the safetensors file
    mmdit_component_prefixes = [
        "model.diffusion_model.x_embedder",
        "model.diffusion_model.pos_embed",
        "model.diffusion_model.t_embedder",
        "model.diffusion_model.y_embedder",
        "model.diffusion_model.context_embedder",
        "model.diffusion_model.joint_blocks",
        "model.diffusion_model.final_layer",
    ]

    logger.info(f"Filtering MMDiT weights from safetensors (looking for '{mmdit_prefix}' prefix)...")
    for key, value in all_weights.items():
        # Check if this key belongs to MMDiT (diffusion_model component)
        if any(key.startswith(prefix) for prefix in mmdit_component_prefixes):
            # Remove "model.diffusion_model." prefix to match the state dict format
            if key.startswith(mmdit_prefix):
                mmdit_key = key[len(mmdit_prefix) :]
            else:
                mmdit_key = key
            mmdit_weights[mmdit_key] = value

    logger.info(f"Extracted {len(mmdit_weights)} MMDiT-related keys from safetensors")

    # If no keys found, try alternative patterns
    if len(mmdit_weights) == 0:
        logger.warning("No keys found with 'model.diffusion_model.' prefix. Trying alternative key patterns...")
        # Try looking for keys that match MMDiT patterns with different prefixes
        alternative_prefixes = [
            "transformer.",
            "diffusion_model.",
            "model.transformer.",
        ]
        for alt_prefix in alternative_prefixes:
            logger.info(f"  Trying prefix: '{alt_prefix}'")
            for key, value in all_weights.items():
                if key.startswith(alt_prefix):
                    # Check if it's a transformer-related key
                    if any(
                        comp in key
                        for comp in [
                            "joint_blocks",
                            "transformer_blocks",
                            "x_embedder",
                            "pos_embed",
                            "t_embedder",
                            "y_embedder",
                            "context_embedder",
                            "final_layer",
                            "norm_out",
                            "proj_out",
                            "time_text_embed",
                        ]
                    ):
                        mmdit_key = key[len(alt_prefix) :] if key.startswith(alt_prefix) else key
                        mmdit_weights[mmdit_key] = value
            if len(mmdit_weights) > 0:
                logger.info(f"  Found {len(mmdit_weights)} keys with prefix '{alt_prefix}'")
                break

        # Last resort: try without any prefix (flat structure)
        if len(mmdit_weights) == 0:
            logger.warning("  Trying flat structure (no prefix)...")
            flat_components = [
                "x_embedder",
                "pos_embed",
                "t_embedder",
                "y_embedder",
                "context_embedder",
                "joint_blocks",
                "transformer_blocks",
                "final_layer",
                "norm_out",
                "proj_out",
            ]
            for key, value in all_weights.items():
                if any(key.startswith(comp) for comp in flat_components):
                    mmdit_weights[key] = value
            if len(mmdit_weights) > 0:
                logger.info(f"  Found {len(mmdit_weights)} keys with flat structure")

    if len(mmdit_weights) == 0:
        raise ValueError(
            f"No MMDiT weights found in safetensors file. "
            f"Total keys in file: {len(all_weights)}. "
            f"Sample keys: {list(all_weights.keys())[:20]}"
        )

    logger.info(f"Sample MMDiT keys (first 10): {list(mmdit_weights.keys())[:10]}")
    logger.info(f"Sample MMDiT keys (last 10): {list(mmdit_weights.keys())[-10:]}")

    # Check pos_embed size in safetensors to determine pos_embed_max_size
    pos_embed_max_size_from_weights = None
    if "pos_embed" in mmdit_weights:
        pos_embed_shape = mmdit_weights["pos_embed"].shape
        if len(pos_embed_shape) == 3:  # [1, num_patches, hidden_size]
            num_patches_in_weights = pos_embed_shape[1]
            # Calculate spatial size: num_patches = spatial_size^2
            pos_embed_max_size_from_weights = int(num_patches_in_weights**0.5)
            logger.info(f"Found pos_embed in safetensors with shape {pos_embed_shape}")
            logger.info(f"Calculated pos_embed_max_size from weights: {pos_embed_max_size_from_weights}")

    # Detect actual depth (number of joint_blocks) from safetensors
    actual_depth = depth  # Default to test parameter
    joint_block_indices = set()
    for key in mmdit_weights.keys():
        if key.startswith("joint_blocks."):
            # Extract block index: "joint_blocks.23.context_block..." -> 23
            parts = key.split(".")
            if len(parts) >= 2:
                try:
                    block_idx = int(parts[1])
                    joint_block_indices.add(block_idx)
                except ValueError:
                    pass

    if joint_block_indices:
        actual_depth = max(joint_block_indices) + 1  # +1 because indices are 0-based
        logger.info(f"Detected {actual_depth} joint blocks in safetensors (indices: {sorted(joint_block_indices)})")
        if actual_depth != depth:
            logger.warning(f"Depth mismatch: test parameter depth={depth}, but safetensors has {actual_depth} blocks")
    else:
        logger.warning(f"Could not detect joint_blocks from safetensors, using test parameter depth={depth}")

    # Use pos_embed_max_size from weights if available, otherwise use default
    actual_pos_embed_max_size = pos_embed_max_size_from_weights if pos_embed_max_size_from_weights else 32
    actual_num_patches = (
        actual_pos_embed_max_size * actual_pos_embed_max_size if pos_embed_max_size_from_weights else num_patches
    )

    # Map safetensors keys to TTNN model format
    # Safetensors may use: joint_blocks.* (already correct), or transformer_blocks.* (needs mapping)
    # Also handles: pos_embed.* -> x_embedder.*, time_text_embed.* -> t_embedder.*, etc.
    def map_hf_to_ttnn_keys(hf_state_dict):
        """Map safetensors/Hugging Face state dict keys to TTNN model keys"""
        ttnn_state_dict = {}

        for key, value in hf_state_dict.items():
            # Map pos_embed.* to x_embedder.* (patch embedding)
            if key.startswith("pos_embed."):
                new_key = key.replace("pos_embed.", "x_embedder.")
                ttnn_state_dict[new_key] = value
            # Keep pos_embed buffer as is (positional embedding)
            elif key == "pos_embed":
                ttnn_state_dict[key] = value
            # Map time_text_embed.timestep_embed.* to t_embedder.*
            # HF has time_text_embed.timestep_embed.mlp.0.* and mlp.2.*
            # TTNN expects t_embedder.mlp.0.* and t_embedder.mlp.2.*
            elif key.startswith("time_text_embed.timestep_embed."):
                new_key = key.replace("time_text_embed.timestep_embed.", "t_embedder.")
                ttnn_state_dict[new_key] = value
            # Map time_text_embed.* directly to t_embedder.* (if no timestep_embed subfolder)
            elif key.startswith("time_text_embed."):
                new_key = key.replace("time_text_embed.", "t_embedder.")
                ttnn_state_dict[new_key] = value
            # Map context_embedder (should be the same)
            elif key.startswith("context_embedder."):
                ttnn_state_dict[key] = value
            # Map y_embedder (class embedding) - Sequential MLP structure
            elif key.startswith("y_embedder."):
                ttnn_state_dict[key] = value
            # Map x_embedder (should be the same, already correct)
            elif key.startswith("x_embedder."):
                ttnn_state_dict[key] = value
            # joint_blocks.* is already correct (safetensors format)
            elif key.startswith("joint_blocks."):
                ttnn_state_dict[key] = value
            # Map transformer_blocks.* to joint_blocks.* (if using old naming)
            elif key.startswith("transformer_blocks."):
                new_key = key.replace("transformer_blocks.", "joint_blocks.")
                ttnn_state_dict[new_key] = value
            # Map norm_out.* and proj_out.* to final_layer.*
            # HF norm_out.norm.* -> final_layer.norm_final.*
            elif key.startswith("norm_out.norm."):
                new_key = key.replace("norm_out.norm.", "final_layer.norm_final.")
                ttnn_state_dict[new_key] = value
            # HF norm_out.linear.* -> final_layer.adaLN_modulation.* (modulation linear layer)
            elif key.startswith("norm_out.linear."):
                new_key = key.replace("norm_out.linear.", "final_layer.adaLN_modulation.")
                ttnn_state_dict[new_key] = value
            # HF proj_out.* -> final_layer.linear.* (output projection)
            elif key.startswith("proj_out."):
                new_key = key.replace("proj_out.", "final_layer.linear.")
                ttnn_state_dict[new_key] = value
            # final_layer.* is already correct
            elif key.startswith("final_layer."):
                ttnn_state_dict[key] = value
            # Keep any other keys as-is (might be valid)
            else:
                ttnn_state_dict[key] = value

        return ttnn_state_dict

    # Map HF keys to TTNN keys
    logger.info("")
    logger.info("Mapping safetensors keys to TTNN model format...")
    ttnn_state_dict = map_hf_to_ttnn_keys(mmdit_weights)
    logger.info(f"Mapped {len(mmdit_weights)} MMDiT keys to {len(ttnn_state_dict)} TTNN keys")

    # Show some example mappings
    logger.info("Sample key mappings:")
    sample_keys = list(mmdit_weights.keys())[:5]
    for key in sample_keys:
        mapped_key = None
        for ttnn_key in ttnn_state_dict.keys():
            if key in ttnn_key or ttnn_key in key:
                mapped_key = ttnn_key
                break
        if mapped_key and mapped_key != key:
            logger.info(f"  {key[:60]:60s} -> {mapped_key[:60]}")
        elif mapped_key:
            logger.info(f"  {key[:60]:60s} -> (unchanged)")

    # Create reference model from ReferenceMMDiTX class
    logger.info("Creating reference model from ReferenceMMDiTX class...")
    logger.info(
        f"Using depth={actual_depth}, pos_embed_max_size={actual_pos_embed_max_size}, num_patches={actual_num_patches}"
    )
    ref_model = ReferenceMMDiTX(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        depth=actual_depth,
        mlp_ratio=4.0,
        learn_sigma=False,
        out_channels=in_channels,
        pos_embed_max_size=actual_pos_embed_max_size,
        num_patches=actual_num_patches,
        qkv_bias=True,
        dtype=torch.bfloat16,
        device=None,
    )
    ref_model.eval()
    print(ref_model)

    # Load weights into reference model
    try:
        ref_model.load_state_dict(ttnn_state_dict, strict=True)
        logger.info("Loaded weights into reference model (strict mode)")
    except Exception as e:
        logger.warning(f"Strict loading failed, trying non-strict mode: {e}")
        # Try without strict mode
        missing_keys, unexpected_keys = ref_model.load_state_dict(ttnn_state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys in reference model: {missing_keys[:10]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in reference model: {unexpected_keys[:10]}...")
        logger.info("Loaded weights into reference model (non-strict mode)")

    # Create parallel config
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=1, mesh_axis=None),
        sequence_parallel=ParallelFactor(factor=1, mesh_axis=None),
    )

    # Create TTNN model
    # Use the same depth and pos_embed_max_size as reference model for consistency
    tt_model = SD35MediumMMDiTX(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        depth=actual_depth,
        mlp_ratio=4.0,
        learn_sigma=False,
        out_channels=in_channels,
        pos_embed_max_size=actual_pos_embed_max_size,
        num_patches=actual_num_patches,
        qkv_bias=True,
        mesh_device=device,
        ccl_manager=None,
        parallel_config=parallel_config,
    )

    # Load weights into TTNN model (using mapped keys)
    tt_model.load_state_dict(ttnn_state_dict)
    logger.info("Loaded weights into TTNN model")

    # Create inputs
    # Reference model expects: x (spatial), t (timestep), y (optional), context (prompt_embed)
    spatial_input = torch.randn(batch_size, in_channels, input_size, input_size, dtype=torch.bfloat16)
    # Context embedder expects input_features=4096, output_features=1536
    # So prompt_embed should have shape [batch_size, seq_len, 4096]
    context_input_dim = 4096  # context_embedder in_features
    prompt_embed = torch.randn(batch_size, 77, context_input_dim, dtype=torch.bfloat16)  # Text embeddings (4096 dim)
    # y_embedder expects input_dim=2048
    y_input_dim = 2048  # y_embedder input_dim
    y_input = torch.randn(batch_size, y_input_dim, dtype=torch.bfloat16)
    timestep = torch.randint(0, 1000, (batch_size,), dtype=torch.long)

    # Reference model forward
    with torch.no_grad():
        ref_output = ref_model(
            x=spatial_input,
            t=timestep,
            y=y_input,
            context=prompt_embed,
        )

    # TTNN forward
    # Convert inputs to TTNN format
    # TTNN model expects: x (spatial), t (timestep), y (optional), context (prompt_embed)
    tt_spatial_input = ttnn.from_torch(spatial_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_t_input = ttnn.from_torch(timestep.unsqueeze(-1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_y_input = ttnn.from_torch(y_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_context_input = ttnn.from_torch(prompt_embed, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = tt_model(tt_spatial_input, tt_t_input, y=tt_y_input, context=tt_context_input)

    # Convert back and compare
    tt_output_torch = ttnn.to_torch(tt_output)
    # Handle shape differences - TTNN output might have extra batch dimension
    # Reference output shape: (batch_size, channels, height, width)
    # TTNN output might be: (1, batch_size, channels, height, width) or (batch_size, channels, height, width)
    if tt_output_torch.ndim == 5:
        tt_output_torch = tt_output_torch[0]  # Remove leading batch dimension if present
    elif tt_output_torch.ndim == 4 and tt_output_torch.shape[0] == 1 and ref_output.shape[0] == batch_size:
        tt_output_torch = tt_output_torch[0]  # Remove batch dimension if it's 1

    # Ensure shapes match
    if tt_output_torch.shape != ref_output.shape:
        logger.warning(f"Shape mismatch: Reference output {ref_output.shape} vs TTNN output {tt_output_torch.shape}")
        # Try to reshape if dimensions are compatible
        if tt_output_torch.numel() == ref_output.numel():
            tt_output_torch = tt_output_torch.reshape(ref_output.shape)

    passing, pcc = comp_pcc(ref_output, tt_output_torch, 0.99)
    logger.info(f"MMDiT PCC: {pcc}")

    assert passing, f"PCC check failed: {pcc}"

    logger.info("SD3.5 Medium MMDiT test passed!")
