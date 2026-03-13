import ttnn
import torch
from models.tt_dit.layers.module import Module
from models.tt_dit.models.vae.vae_wan2_1 import (
    WanCausalConv3d,
    WanResidualBlock,
    WanAttentionBlock,
    WanMidBlock,
    WanResample,
)
from models.experimental.lingbot_va.tt.residual_down_block import WanResidualDownBlock
from models.tt_dit.layers.module import ModuleList
from models.tt_dit.layers.normalization import RMSNorm

CACHE_T = 2
NUM_FEAT_CACHE_SLOTS = 32
_ALIGNMENT = 32
_MAX_C_IN_BLOCK = 128


def _cap_conv3d_blocking(module):
    """Walk all WanCausalConv3d children and cap C_in_block to avoid L1 overflow."""
    for _, child in module.named_children():
        if isinstance(child, WanCausalConv3d):
            cfg = child.conv_config
            if cfg.C_in_block > _MAX_C_IN_BLOCK:
                new_c_in = _ALIGNMENT
                for candidate in range(_MAX_C_IN_BLOCK, _ALIGNMENT - 1, -_ALIGNMENT):
                    if child.in_channels % candidate == 0:
                        new_c_in = candidate
                        break
                child.conv_config = ttnn.Conv3dConfig(
                    weights_dtype=cfg.weights_dtype,
                    output_layout=cfg.output_layout,
                    T_out_block=cfg.T_out_block,
                    W_out_block=cfg.W_out_block,
                    H_out_block=cfg.H_out_block,
                    C_out_block=cfg.C_out_block,
                    C_in_block=new_c_in,
                    compute_with_storage_grid_size=cfg.compute_with_storage_grid_size,
                )
        else:
            _cap_conv3d_blocking(child)


class VaeWanEncoder(Module):
    def __init__(
        self,
        in_channels: int = 3,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        is_residual: bool = False,  # wan 2.2 vae use a residual downblock
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
    ) -> None:
        super().__init__()

        # assert not is_residual, "is_residual is not supported"
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv_in = WanCausalConv3d(
            in_channels,
            dims[0],
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

        # downsample blocks.
        self.down_blocks = ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if is_residual:
                self.down_blocks.append(
                    WanResidualDownBlock(
                        in_dim=in_dim,
                        out_dim=out_dim,
                        num_res_blocks=num_res_blocks,
                        temperal_downsample=temperal_downsample[i] if i != len(dim_mult) - 1 else False,
                        down_flag=i != len(dim_mult) - 1,
                        mesh_device=mesh_device,
                        ccl_manager=ccl_manager,
                        parallel_config=parallel_config,
                    )
                )
            else:
                for _ in range(num_res_blocks):
                    self.down_blocks.append(
                        WanResidualBlock(
                            in_dim=in_dim,
                            out_dim=out_dim,
                            mesh_device=mesh_device,
                            ccl_manager=ccl_manager,
                            parallel_config=parallel_config,
                        )
                    )
                    if scale in attn_scales:
                        self.down_blocks.append(
                            WanAttentionBlock(
                                dim=out_dim,
                                mesh_device=mesh_device,
                                ccl_manager=ccl_manager,
                                parallel_config=parallel_config,
                            )
                        )
                    in_dim = out_dim

                # downsample block
                if i != len(dim_mult) - 1:
                    mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                    self.down_blocks.append(
                        WanResample(
                            dim=out_dim,
                            mode=mode,
                            mesh_device=mesh_device,
                            ccl_manager=ccl_manager,
                            parallel_config=parallel_config,
                        )
                    )
                    scale /= 2.0

        # middle blocks
        self.mid_block = WanMidBlock(
            dim=out_dim, num_layers=1, mesh_device=mesh_device, ccl_manager=ccl_manager, parallel_config=parallel_config
        )
        if out_dim > 384:
            for attn in self.mid_block.attentions:
                attn.sdpa_program_config = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
                    q_chunk_size=32,
                    k_chunk_size=32,
                    exp_approx_mode=False,
                )

        # output blocks
        self.norm_out = RMSNorm(
            embedding_dim=out_dim, norm_eps=1e-6, norm_elementwise_affine=True, bias=False, mesh_device=mesh_device
        )
        self.conv_out = WanCausalConv3d(
            out_dim,
            z_dim,
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

        _cap_conv3d_blocking(self)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "norm_out.gamma" in state:
            state["norm_out.weight"] = state.pop("norm_out.gamma").squeeze()

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        logical_h: int,
        feat_cache: list[ttnn.Tensor] | None = None,
        feat_idx: list[int] = [0],
    ) -> tuple[ttnn.Tensor, int]:
        if feat_cache is None:
            feat_cache = [None] * NUM_FEAT_CACHE_SLOTS
            feat_idx = [0]

        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
            if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None:
                # Current activation is too short, so append the cached activation as well
                cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)
            x_BTHWC = self.conv_in(x_BTHWC, logical_h, feat_cache[idx])
            feat_cache[idx] = cache_x_BTHWC
            feat_idx[0] += 1
        else:
            x_BTHWC = self.conv_in(x_BTHWC, logical_h)

        ## downsamples
        for down_block in self.down_blocks:
            if isinstance(down_block, WanResample):
                x_BTHWC, logical_h = down_block(x_BTHWC, logical_h, feat_cache, feat_idx)
            elif isinstance(down_block, WanResidualBlock):
                x_BTHWC = down_block(x_BTHWC, logical_h, feat_cache, feat_idx)
            elif isinstance(down_block, WanAttentionBlock):
                x_BTHWC = down_block(x_BTHWC, logical_h)
            elif isinstance(down_block, WanResidualDownBlock):
                x_BTHWC = down_block(x_BTHWC, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                raise ValueError(f"Unsupported downblock type: {type(down_block)}")

        ## middle
        x_BTHWC = self.mid_block(x_BTHWC, logical_h, feat_cache, feat_idx)

        ## head
        x_tile_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)
        x_norm_tile_BTHWC = self.norm_out(x_tile_BTHWC)
        x_silu_tile_BTHWC = ttnn.silu(x_norm_tile_BTHWC)
        x_BTHWC = ttnn.to_layout(x_silu_tile_BTHWC, ttnn.ROW_MAJOR_LAYOUT)

        if feat_cache is not None:
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
            if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None:
                # Current activation is too short, so append the cached activation as well
                cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)
            x_BTHWC = self.conv_out(x_BTHWC, logical_h, feat_cache[idx])
            feat_cache[idx] = cache_x_BTHWC
            feat_idx[0] += 1
        else:
            x_BTHWC = self.conv_out(x_BTHWC, logical_h)
        return x_BTHWC, logical_h
