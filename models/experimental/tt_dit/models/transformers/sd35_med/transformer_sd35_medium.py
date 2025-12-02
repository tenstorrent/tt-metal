# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SD3.5 Medium Multi-Modal Diffusion Transformer (MMDiTX) Implementation
Complete model integrating all components for joint context-x processing.
"""

import ttnn
import torch
from loguru import logger
from models.experimental.tt_dit.layers.linear import Linear
from models.experimental.tt_dit.models.transformers.sd35_med.patch_embed_sd35_medium import SD35MediumPatchEmbed
from models.experimental.tt_dit.models.transformers.sd35_med.timestep_embed_sd35_medium import TimestepEmbedder
from models.experimental.tt_dit.models.transformers.sd35_med.vector_embed_sd35_medium import VectorEmbedder
from models.experimental.tt_dit.models.transformers.sd35_med.joint_block_sd35_medium import SD35MediumJointBlock
from models.experimental.tt_dit.models.transformers.sd35_med.final_layer_sd35_medium import SD35MediumFinalLayer
from models.experimental.tt_dit.utils.substate import substate


class SD35MediumMMDiTX:
    """Complete MMDiTX model for SD3.5 Medium"""

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
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
    ):
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.depth = depth
        self.out_channels = out_channels
        self.pos_embed_max_size = pos_embed_max_size
        self.mesh_device = mesh_device

        # SD3.5 Medium uses fixed architecture values
        self.hidden_size = 1536  # SD3.5 Medium fixed hidden size
        self.num_heads = 24  # SD3.5 Medium fixed number of heads
        self.head_dim = self.hidden_size // self.num_heads

        # Patch embedding: Conv2d(16, 1536, kernel_size=(2, 2), stride=(2, 2))
        self.x_embedder = SD35MediumPatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.hidden_size,
            mesh_device=mesh_device,
        )

        # Positional embedding
        self.pos_embed = self._create_pos_embed()

        # Time embedding: MLP(256 -> 1536 -> 1536)
        self.t_embedder = TimestepEmbedder(
            hidden_size=self.hidden_size,
            frequency_embedding_size=256,
            dtype=torch.bfloat16,
            mesh_device=mesh_device,
        )

        # Class embedding: MLP(2048 -> 1536 -> 1536)
        self.y_embedder = VectorEmbedder(
            input_dim=2048,
            hidden_size=self.hidden_size,
            mesh_device=mesh_device,
            dtype=torch.bfloat16,
        )

        # Context embedding: Linear(4096, 1536)
        self.context_embedder = Linear(
            in_features=4096,
            out_features=self.hidden_size,
            bias=True,
            mesh_device=mesh_device,
        )

        # Joint blocks: 28 blocks total
        # Blocks 0-12: 13 blocks with dual attention (x_block_self_attn=True)
        # Blocks 13-22: 10 blocks with single attention (x_block_self_attn=False)
        # Block 23: Last block with pre_only context_block (pre_only=True, x_block_self_attn=False)
        self.joint_blocks = []
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
                SD35MediumJointBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    pre_only=pre_only,
                    scale_mod_only=False,
                    swiglu=False,  # GELU, not SwiGLU
                    qk_norm="rms",  # RMSNorm for QK
                    x_block_self_attn=x_block_self_attn,
                    mesh_device=mesh_device,
                    ccl_manager=ccl_manager,
                    parallel_config=parallel_config,
                )
            )

        # Final layer: norm_final, linear(1536 -> 64), adaLN_modulation(1536 -> 3072)
        self.final_layer = SD35MediumFinalLayer(
            hidden_size=self.hidden_size,
            patch_size=patch_size,
            out_channels=self.out_channels,
            mesh_device=mesh_device,
        )

        # Compute config
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _create_pos_embed(self):
        """Create positional embedding parameters"""
        pos_embed_shape = (1, self.pos_embed_max_size * self.pos_embed_max_size, self.hidden_size)
        pos_embed = torch.randn(pos_embed_shape, dtype=torch.bfloat16)
        return ttnn.from_torch(pos_embed, device=self.mesh_device)

    def _cropped_pos_embed(self, hw, num_patches):
        """Extract cropped positional embeddings

        Args:
            hw: Tuple of (height, width) in pixels
            num_patches: Actual number of patches from patch embedder output
        Returns:
            Positional embeddings with shape [1, num_patches, hidden_size]
        """
        h, w = hw
        p = self.patch_size
        h_patches, w_patches = h // p, w // p

        # Calculate start indices for center crop
        h_start = (self.pos_embed_max_size - h_patches) // 2
        w_start = (self.pos_embed_max_size - w_patches) // 2

        # Reshape pos_embed to spatial format: [1, pos_embed_max_size, pos_embed_max_size, hidden_size]
        pos_embed_flat = ttnn.reshape(self.pos_embed, [1, -1, self.hidden_size])
        pos_embed_spatial = ttnn.reshape(
            pos_embed_flat, [1, self.pos_embed_max_size, self.pos_embed_max_size, self.hidden_size]
        )

        # Crop the spatial positional embedding
        # Slice: [1, h_start:h_start+h_patches, w_start:w_start+w_patches, hidden_size]
        cropped_pos_embed = ttnn.slice(
            pos_embed_spatial, [0, h_start, w_start, 0], [1, h_start + h_patches, w_start + w_patches, self.hidden_size]
        )

        # Reshape back to [1, num_patches, hidden_size] to match patch embedder output
        # Use the actual num_patches from patch embedder output to ensure shape match
        cropped_pos_embed = ttnn.reshape(cropped_pos_embed, [1, num_patches, self.hidden_size])

        return cropped_pos_embed

    def _unpatchify(self, x, hw):
        """Convert patch embeddings back to image format

        Args:
            x: Tensor of shape [B, num_patches, patch_size^2 * out_channels]
            hw: Tuple of (height, width) in pixels
        Returns:
            Tensor of shape [B, out_channels, H, W]
        """
        h, w = hw
        p = self.patch_size
        c = self.out_channels

        # Calculate number of patches in each dimension
        h_patches = h // p
        w_patches = w // p

        # x shape: [B, num_patches, patch_size^2 * out_channels]
        # Reshape to: [B, h_patches, w_patches, p, p, c]
        x = ttnn.reshape(x, [x.shape[0], h_patches, w_patches, p, p, c])

        # Permute: nhwpqc -> nchpwq
        # This is equivalent to: [B, h_patches, w_patches, p, p, c] -> [B, c, h_patches, p, w_patches, p]
        x = ttnn.permute(x, (0, 5, 1, 3, 2, 4))

        # Reshape to: [B, c, h_patches * p, w_patches * p] = [B, c, H, W]
        x = ttnn.reshape(x, [x.shape[0], c, h_patches * p, w_patches * p])

        return x

    def forward_core_with_concat(self, x, c, context, skip_layers=None, controlnet_hidden_states=None):
        """Forward pass through joint blocks"""
        if skip_layers is None:
            skip_layers = []

        # context is B, L', D
        # x is B, L, D
        for i, block in enumerate(self.joint_blocks):
            if i in skip_layers:
                continue
            context, x = block(context, x, c)

            if controlnet_hidden_states is not None:
                controlnet_block_interval = len(self.joint_blocks) // len(controlnet_hidden_states)
                x = x + controlnet_hidden_states[i // controlnet_block_interval]

        return x

    def __call__(
        self,
        x=None,
        t=None,
        y=None,
        context=None,
        controlnet_hidden_states=None,
        skip_layers=None,
        spatial=None,
        prompt_embed=None,
        pooled_projections=None,
        timestep=None,
        N=None,
        L=None,
    ):
        """
        Forward pass of MMDiTX with support for both old and new API.

        New API:
            x: (N, C, H, W) tensor of spatial inputs in NCHW format
            t: (N,) tensor of diffusion timesteps
            y: (N, D) tensor of class embeddings (optional)
            context: (N, L', D') tensor of context embeddings

        Old API (for pipeline compatibility):
            spatial: ttnn.Tensor [1, B, N, D] or [B, N, D] (spatial patches - already patched)
            prompt_embed: ttnn.Tensor [1, B, L, D'] (context embeddings)
            pooled_projections: ttnn.Tensor [1, B, D''] (class embeddings)
            timestep: ttnn.Tensor [1, B] (timesteps)
            N: int (spatial sequence length - number of patches)
            L: int (prompt sequence length)
        """
        # Support old API for pipeline compatibility
        if spatial is not None:
            # Old API: spatial is already in patch format [1, B, N, D] or [B, N, D]
            # We need to extract batch dimension and use it directly
            if len(spatial.shape) == 4:  # [1, B, N, D]
                spatial = ttnn.reshape(spatial, [spatial.shape[1], spatial.shape[2], spatial.shape[3]])
            # spatial is now [B, N, D] where N is num_patches

            # Extract batch dimension from other inputs
            if len(prompt_embed.shape) == 4:  # [1, B, L, D']
                prompt_embed = ttnn.reshape(
                    prompt_embed, [prompt_embed.shape[1], prompt_embed.shape[2], prompt_embed.shape[3]]
                )
            # prompt_embed is now [B, L, D']

            if len(pooled_projections.shape) == 3:  # [1, B, D'']
                pooled_projections = ttnn.reshape(
                    pooled_projections, [pooled_projections.shape[1], pooled_projections.shape[2]]
                )
            # pooled_projections is now [B, D'']

            if len(timestep.shape) == 2:  # [1, B]
                timestep = ttnn.reshape(timestep, [timestep.shape[1]])
            # timestep is now [B]

            # Use the old API path
            # Note: spatial is already patched, so we skip patch embedding
            # We need to add positional embeddings if not already added
            # For now, we'll assume positional embeddings need to be added
            # But we don't know H and W from just N (num_patches)
            # We'll need to infer or store them

            # Actually, the pipeline should handle unpatchifying before calling this
            # But for backward compatibility, let's try to work with patches directly
            # This is complex, so we'll raise an error for now and require the new API
            raise NotImplementedError(
                "Old API (spatial, prompt_embed, pooled_projections, timestep, N, L) requires spatial to be in image format, not patch format. "
                "Please update the pipeline to use the new API: (x, t, y, context) where x is in NCHW format."
            )

        # New API
        if x is None or t is None:
            raise ValueError(
                "Either provide (x, t, y, context) for new API or (spatial, prompt_embed, pooled_projections, timestep, N, L) for old API"
            )

        # Convert from NCHW to NHWC format for patch embedder
        # Input: [B, C, H, W] -> [B, H, W, C]
        x = ttnn.permute(x, (0, 2, 3, 1))

        # Get input dimensions (H, W) from TTNN Shape object
        # After permute, x.shape is Shape([1, 32, 32, 16]), so shape[1] is H and shape[2] is W
        h = x.shape[1]
        w = x.shape[2]
        hw = (h, w)

        # Patch embedding (expects NHWC format: [B, H, W, C])
        x = self.x_embedder(x)

        # Get actual number of patches from patch embedder output
        # x shape is [B, num_patches, hidden_size] after patch embedding
        actual_num_patches = x.shape[1]

        # Add positional embeddings
        pos_embed = self._cropped_pos_embed(hw, actual_num_patches)
        x = x + pos_embed

        # Time embedding: MLP(256 -> 1536 -> 1536)
        c = self.t_embedder(t)

        # Class embedding (if available): MLP(2048 -> 1536 -> 1536)
        if y is not None:
            y_emb = self.y_embedder(y)
            c = c + y_emb

        # Context embedding: Linear(4096, 1536)
        if context is not None:
            context = self.context_embedder(context)
        else:
            context = None

        # Forward through joint blocks
        x = self.forward_core_with_concat(x, c, context, skip_layers, controlnet_hidden_states)

        # Final layer
        x = self.final_layer(x, c)

        # Unpatchify: convert from [B, num_patches, patch_size^2 * out_channels] to [B, out_channels, H, W]
        x = self._unpatchify(x, hw)

        return x

    def load_state_dict(self, state_dict):
        """Load weights from PyTorch state dict"""
        logger.info("=" * 80)
        logger.info("Loading MMDiT transformer weights - checking component availability")
        logger.info("=" * 80)

        loaded_components = {}
        missing_components = []

        # Expected components
        expected_components = [
            "x_embedder",
            "pos_embed",
            "t_embedder",
            "y_embedder",
            "context_embedder",
            "final_layer",
        ]

        # Check for expected components
        logger.info(f"State dict contains {len(state_dict)} total keys")
        logger.info(f"Sample keys: {list(state_dict.keys())[:10]}")
        logger.info("")

        # Extract MMDiT weights from state dict, handling various prefixes
        # Try different prefixes that might be used in the state dict
        prefixes = [
            "model.diffusion_model.",
            "transformer.",
            "diffusion_model.",
            "model.transformer.",
            "",  # No prefix (flat structure)
        ]

        mmdit_state_dict = {}
        found_prefix = None

        # First, try to detect the prefix by looking for MMDiT component keys
        for prefix in prefixes:
            # Try to find keys with this prefix that match MMDiT components
            test_keys = [
                k
                for k in state_dict.keys()
                if k.startswith(prefix)
                and (
                    "x_embedder" in k
                    or "pos_embed" in k
                    or "t_embedder" in k
                    or "y_embedder" in k
                    or "context_embedder" in k
                    or "joint_blocks" in k
                    or "final_layer" in k
                    or "norm_final" in k
                    or "proj_out" in k
                )
            ]
            if test_keys:
                found_prefix = prefix
                logger.info(f"Found MMDiT weights with prefix: '{prefix}'")
                logger.info(f"  Sample keys: {test_keys[:3]}")
                # Extract all MMDiT-related keys
                for key, value in state_dict.items():
                    if key.startswith(prefix):
                        # Remove prefix to get the component name
                        new_key = key[len(prefix) :]
                        mmdit_state_dict[new_key] = value
                break

        if not mmdit_state_dict:
            # If no prefix found, check if keys are already in the expected format
            test_keys = [
                k
                for k in state_dict.keys()
                if (
                    "x_embedder" in k
                    or "pos_embed" in k
                    or "t_embedder" in k
                    or "y_embedder" in k
                    or "context_embedder" in k
                    or "joint_blocks" in k
                    or "final_layer" in k
                    or "norm_final" in k
                    or "proj_out" in k
                )
            ]
            if test_keys:
                logger.info("Found MMDiT weights without prefix (flat structure)")
                mmdit_state_dict = state_dict
            else:
                logger.warning("Could not find MMDiT weights with any known prefix or format.")
                logger.warning("  Available key prefixes:")
                all_prefixes = set()
                for key in list(state_dict.keys())[:50]:  # Check first 50 keys
                    parts = key.split(".")
                    if len(parts) > 1:
                        all_prefixes.add(".".join(parts[:2]) + ".")
                logger.warning(f"  Sample prefixes: {list(all_prefixes)[:5]}")
                mmdit_state_dict = state_dict  # Use as-is and let individual loaders handle it
        else:
            logger.info(f"Extracted {len(mmdit_state_dict)} MMDiT weight keys")

        # Use the extracted state dict for loading
        state_dict = mmdit_state_dict

        # Load patch embedding
        logger.info("Checking x_embedder (patch embedding)...")
        x_embedder_state = substate(state_dict, "x_embedder")
        if x_embedder_state:
            if "pos_embed" in x_embedder_state:
                x_embedder_state = {k: v for k, v in x_embedder_state.items() if k != "pos_embed"}
            logger.info(f"  ✓ Found x_embedder with keys: {list(x_embedder_state.keys())}")
            self.x_embedder.load_state_dict(x_embedder_state)
            loaded_components["x_embedder"] = True
        else:
            logger.warning("  ✗ Missing x_embedder weights")
            missing_components.append("x_embedder")
            loaded_components["x_embedder"] = False

        # Load positional embedding
        logger.info("Checking pos_embed (positional embedding)...")
        if "pos_embed" in state_dict:
            logger.info(f"  ✓ Found pos_embed with shape: {state_dict['pos_embed'].shape}")
            self.pos_embed = ttnn.from_torch(state_dict["pos_embed"], device=self.mesh_device)
            loaded_components["pos_embed"] = True
        else:
            logger.warning("  ✗ Missing pos_embed")
            missing_components.append("pos_embed")
            loaded_components["pos_embed"] = False

        # Load time embedding
        logger.info("Checking t_embedder (timestep embedding)...")
        t_embedder_state = substate(state_dict, "t_embedder")
        if t_embedder_state:
            logger.info(f"  ✓ Found t_embedder with keys: {list(t_embedder_state.keys())}")
            # Map from reference model structure to TTNN structure
            if "mlp.0.weight" in t_embedder_state:
                logger.info("  Using reference model format (mlp.0, mlp.2)")
                self.t_embedder.linear1.load_state_dict(
                    {
                        "weight": t_embedder_state["mlp.0.weight"],
                        "bias": t_embedder_state["mlp.0.bias"],
                    }
                )
                self.t_embedder.linear2.load_state_dict(
                    {
                        "weight": t_embedder_state["mlp.2.weight"],
                        "bias": t_embedder_state["mlp.2.bias"],
                    }
                )
            else:
                # Direct structure: t_embedder.linear1 and t_embedder.linear2
                if "linear1.weight" in t_embedder_state:
                    logger.info("  Using direct format (linear1, linear2)")
                    self.t_embedder.linear1.load_state_dict(substate(t_embedder_state, "linear1"))
                    self.t_embedder.linear2.load_state_dict(substate(t_embedder_state, "linear2"))
                else:
                    # Fallback: try 0 and 1 indices
                    if "0.weight" in t_embedder_state:
                        logger.info("  Using index format (0, 1)")
                        self.t_embedder.linear1.load_state_dict(substate(t_embedder_state, "0"))
                        self.t_embedder.linear2.load_state_dict(substate(t_embedder_state, "1"))
                    else:
                        logger.warning("  ✗ Could not find t_embedder weights in any expected format")
                        missing_components.append("t_embedder")
                        loaded_components["t_embedder"] = False
                        t_embedder_state = None
            if t_embedder_state:
                loaded_components["t_embedder"] = True
        else:
            logger.warning("  ✗ Missing t_embedder")
            missing_components.append("t_embedder")
            loaded_components["t_embedder"] = False

        # Load class embedding - Sequential MLP structure
        logger.info("Checking y_embedder (class embedding)...")
        y_embedder_state = substate(state_dict, "y_embedder")
        if y_embedder_state:
            logger.info(f"  ✓ Found y_embedder with keys: {list(y_embedder_state.keys())}")
            # Handle Sequential MLP structure: mlp.0 and mlp.2 (with SiLU at index 1)
            if "mlp.0.weight" in y_embedder_state:
                logger.info("  Using reference model format (mlp.0, mlp.2)")
                self.y_embedder.linear1.load_state_dict(
                    {
                        "weight": y_embedder_state["mlp.0.weight"],
                        "bias": y_embedder_state["mlp.0.bias"],
                    }
                )
                self.y_embedder.linear2.load_state_dict(
                    {
                        "weight": y_embedder_state["mlp.2.weight"],
                        "bias": y_embedder_state["mlp.2.bias"],
                    }
                )
            else:
                # Direct structure: y_embedder.linear1 and y_embedder.linear2
                if "linear1.weight" in y_embedder_state:
                    logger.info("  Using direct format (linear1, linear2)")
                    self.y_embedder.linear1.load_state_dict(substate(y_embedder_state, "linear1"))
                    self.y_embedder.linear2.load_state_dict(substate(y_embedder_state, "linear2"))
                else:
                    # Fallback: try 0 and 1 indices
                    if "0.weight" in y_embedder_state:
                        logger.info("  Using index format (0, 1)")
                        self.y_embedder.linear1.load_state_dict(substate(y_embedder_state, "0"))
                        self.y_embedder.linear2.load_state_dict(substate(y_embedder_state, "1"))
                    else:
                        logger.warning("  ✗ Could not find y_embedder weights in any expected format")
                        missing_components.append("y_embedder")
                        loaded_components["y_embedder"] = False
                        y_embedder_state = None
            if y_embedder_state:
                loaded_components["y_embedder"] = True
        else:
            logger.warning("  ✗ Missing y_embedder")
            missing_components.append("y_embedder")
            loaded_components["y_embedder"] = False

        # Load context embedder
        logger.info("Checking context_embedder...")
        context_embedder_state = substate(state_dict, "context_embedder")
        if context_embedder_state and self.context_embedder is not None:
            logger.info(f"  ✓ Found context_embedder with keys: {list(context_embedder_state.keys())}")
            self.context_embedder.load_state_dict(context_embedder_state)
            loaded_components["context_embedder"] = True
        else:
            logger.warning("  ✗ Missing context_embedder")
            missing_components.append("context_embedder")
            loaded_components["context_embedder"] = False

        # Load joint blocks
        logger.info(f"Checking joint_blocks (expecting {len(self.joint_blocks)} blocks)...")
        joint_blocks_loaded = 0
        joint_blocks_missing = []
        for i, block in enumerate(self.joint_blocks):
            joint_block_state = substate(state_dict, f"joint_blocks.{i}")
            if joint_block_state:
                block.load_state_dict(joint_block_state)
                joint_blocks_loaded += 1
            else:
                joint_blocks_missing.append(i)
                logger.warning(f"  ✗ Missing joint_blocks.{i}")

        if joint_blocks_loaded == len(self.joint_blocks):
            logger.info(f"  ✓ All {joint_blocks_loaded} joint blocks loaded successfully")
            loaded_components["joint_blocks"] = True
        else:
            logger.warning(f"  ✗ Only {joint_blocks_loaded}/{len(self.joint_blocks)} joint blocks loaded")
            logger.warning(f"    Missing blocks: {joint_blocks_missing}")
            missing_components.append(f"joint_blocks (missing: {joint_blocks_missing})")
            loaded_components["joint_blocks"] = False

        # Load final layer
        logger.info("Checking final_layer...")
        final_layer_state = substate(state_dict, "final_layer")
        if final_layer_state:
            logger.info(f"  ✓ Found final_layer with keys: {list(final_layer_state.keys())}")
            self.final_layer.load_state_dict(final_layer_state)
            loaded_components["final_layer"] = True
        else:
            logger.warning("  ✗ Missing final_layer")
            missing_components.append("final_layer")
            loaded_components["final_layer"] = False

        # Load register parameters (optional)
        if hasattr(self, "register") and "register" in state_dict:
            logger.info("Checking register (optional)...")
            logger.info(f"  ✓ Found register with shape: {state_dict['register'].shape}")
            self.register = ttnn.from_torch(state_dict["register"], device=self.mesh_device)
            loaded_components["register"] = True

        # Summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("Weight Loading Summary")
        logger.info("=" * 80)
        for component, loaded in loaded_components.items():
            status = "✓ LOADED" if loaded else "✗ MISSING"
            logger.info(f"  {component:20s}: {status}")

        if missing_components:
            logger.warning("")
            logger.warning(f"⚠️  WARNING: {len(missing_components)} component(s) missing:")
            for comp in missing_components:
                logger.warning(f"    - {comp}")
        else:
            logger.info("")
            logger.info("✓ All transformer weights loaded successfully!")

        logger.info("=" * 80)

    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        """Convert latents to patch format for compatibility with pipeline

        Args:
            latents: Tensor of shape (N, H, W, C) in NHWC format
        Returns:
            Tensor of shape (1, N, (H/P) * (W/P), P * P * C)
        """
        batch_size, height, width, channels = latents.shape
        patch = self.patch_size

        if height % patch != 0 or width % patch != 0:
            msg = f"height ({height}) and width ({width}) must be divisible by patch_size ({patch})"
            raise ValueError(msg)

        latents = latents.reshape([batch_size, height // patch, patch, width // patch, patch, channels])
        return latents.transpose(2, 3).flatten(3, 5).flatten(1, 2).unsqueeze(0)

    def unpatchify(self, spatial: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        """Convert patch format back to latents for compatibility with pipeline

        Args:
            spatial: Tensor of shape (1, N, (H/P) * (W/P), P * P * C)
            height: Target height in pixels
            width: Target width in pixels
        Returns:
            Tensor of shape (N, H, W, C) in NHWC format
        """
        one, batch_size, _, _ = spatial.shape
        assert one == 1
        patch = self.patch_size

        if height % patch != 0 or width % patch != 0:
            msg = f"height ({height}) and width ({width}) must be divisible by patch_size ({patch})"
            raise ValueError(msg)

        spatial = spatial.reshape([batch_size, height // patch, width // patch, patch, patch, -1])
        return spatial.transpose(2, 3).flatten(3, 4).flatten(1, 2)


# Alias for backward compatibility with pipeline
SD35MediumTransformer2DModel = SD35MediumMMDiTX
