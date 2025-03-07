import ttnn
import torch
from typing import List, Optional
from models.common.lightweightmodule import LightweightModule
from .conv1x1 import Conv1x1
from .resblock import ResBlock
from .upsample import CausalUpsampleBlock
from .common import load_decoder_weights
from loguru import logger


class Decoder(LightweightModule):
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        state_dict_prefix: str = "",
        out_channels=3,
        base_channels=128,
        channel_multipliers=[1, 2, 4, 6],
        temporal_expansions=[1, 2, 3],
        spatial_expansions=[2, 2, 2],
        num_res_blocks=[3, 3, 4, 6, 3],
        latent_dim=12,
        has_attention=[False, False, False, False, False],
        output_norm=False,
        nonlinearity="silu",
        output_nonlinearity="silu",
        causal=True,
    ):
        """
        TTNN implementation of the VAE Decoder.
        """
        self.input_channels = latent_dim
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.output_nonlinearity = output_nonlinearity
        assert nonlinearity == "silu"
        assert causal
        assert not any(has_attention), "Attention is not supported in the decoder"
        attn_block = None
        # Calculate channels for each level
        ch = [mult * base_channels for mult in channel_multipliers]
        self.num_up_blocks = len(ch) - 1
        assert len(num_res_blocks) == self.num_up_blocks + 2

        assert len(temporal_expansions) == len(spatial_expansions) == self.num_up_blocks
        assert len(num_res_blocks) == len(has_attention) == self.num_up_blocks + 2

        # Create the initial projection from latent space
        self.input_proj = Conv1x1(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}blocks.0.0.",
            in_channels=latent_dim,
            out_channels=ch[-1],
        )

        # First set of residual blocks
        self.first_blocks = []
        for i in range(num_res_blocks[-1]):
            self.first_blocks.append(
                ResBlock(
                    mesh_device=mesh_device,
                    state_dict=state_dict,
                    state_dict_prefix=f"{state_dict_prefix}blocks.0.{i+1}.",
                    channels=ch[-1],
                    attn_block=attn_block,
                    causal=causal,
                    padding_mode="replicate",
                )
            )

        # Create upsampling blocks
        self.up_blocks = []
        for i in range(self.num_up_blocks):
            self.up_blocks.append(
                CausalUpsampleBlock(
                    mesh_device=mesh_device,
                    state_dict=state_dict,
                    state_dict_prefix=f"{state_dict_prefix}blocks.{i+1}.",
                    in_channels=ch[-i - 1],
                    out_channels=ch[-i - 2],
                    num_res_blocks=num_res_blocks[-i - 2],
                    attn_block=attn_block,
                    temporal_expansion=temporal_expansions[-i - 1],
                    spatial_expansion=spatial_expansions[-i - 1],
                    causal=causal,
                    padding_mode="replicate",
                )
            )

        # Last set of residual blocks
        self.last_blocks = []
        for i in range(num_res_blocks[0]):
            self.last_blocks.append(
                ResBlock(
                    mesh_device=mesh_device,
                    state_dict=state_dict,
                    state_dict_prefix=f"{state_dict_prefix}blocks.{self.num_up_blocks+1}.{i}.",
                    channels=ch[0],
                    attn_block=attn_block,
                    causal=causal,
                    padding_mode="replicate",
                )
            )

        # Final output projection
        self.output_proj = Conv1x1(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}output_proj.",
            in_channels=ch[0],
            out_channels=out_channels,
            bias=True,
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def forward(self, x_NTHWC):
        """
        Forward pass for the decoder.

        Args:
            x_NTHWC: Input tensor in NTHWC layout

        Returns:
            Output tensor in NTHWC layout
        """
        # Initial projection
        x_NTHWC = self.input_proj(x_NTHWC)

        # First set of residual blocks
        for block in self.first_blocks:
            x_NTHWC = block(x_NTHWC)

        # Upsampling blocks
        for block in self.up_blocks:
            x_NTHWC = block(x_NTHWC)

        # Last set of residual blocks
        for block in self.last_blocks:
            x_NTHWC = block(x_NTHWC)

        # Apply output nonlinearity if needed
        if self.output_nonlinearity == "silu":
            x_tile_NTHWC = ttnn.to_layout(x_NTHWC, ttnn.TILE_LAYOUT)
            ttnn.deallocate(x_NTHWC)
            x_tile_NTHWC = ttnn.silu(x_tile_NTHWC, output_tensor=x_tile_NTHWC)  # in-place
            x_NTHWC = ttnn.to_layout(x_tile_NTHWC, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(x_tile_NTHWC)
        else:
            assert not self.output_nonlinearity  # StyleGAN3 omits the to-RGB nonlinearity.

        # Final projection
        x_NTHWC = self.output_proj(x_NTHWC)

        return x_NTHWC

    @classmethod
    def from_pretrained(cls, mesh_device, **kwargs):
        """
        Create a TtDecoder from pretrained weights.

        Args:
            mesh_device: TTNN mesh device
            **kwargs: Additional arguments to pass to the constructor

        Returns:
            TtDecoder: Initialized decoder
        """
        state_dict = load_decoder_weights()
        if state_dict is None:
            logger.error("Failed to load decoder weights")
            return None

        return cls(mesh_device=mesh_device, state_dict=state_dict, **kwargs)
