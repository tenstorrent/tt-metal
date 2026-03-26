# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import ttnn
import torch
from models.common.utility_functions import is_blackhole
from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.models.vae.vae_wan2_1 import (
    WanCausalConv3d,
    WanAttentionBlock,
    WanMidBlock,
    WanResample,
)
from models.experimental.lingbot_va.tt.residual_block import WanResidualBlock
from models.experimental.lingbot_va.tt.residual_down_block import WanResidualDownBlock
from models.tt_dit.layers.normalization import RMSNorm
from models.tt_dit.utils.conv3d import count_convs

CACHE_T = 2
_ALIGNMENT = 32
_DEFAULT_MAX_C_IN_BLOCK = 320


def _iter_tt_module_submodules(module: Module):
    """Depth-first walk of tt_dit Module hierarchy."""
    yield module
    for _, child in module.named_children():
        yield from _iter_tt_module_submodules(child)


def patch_wan_causal_conv_wormhole_bf16_parity(mm: WanCausalConv3d, mesh_device) -> None:
    """Use a Wormhole bf16 conv kernel config closer to torch reference numerics."""
    if mesh_device is None or is_blackhole():
        return
    arch = mesh_device.arch()
    mm.compute_kernel_config = ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _cap_conv3d_blocking(module, max_c_in_block: int):
    """Walk all WanCausalConv3d children and cap C_in_block to avoid L1 overflow."""
    for _, child in module.named_children():
        if isinstance(child, WanCausalConv3d):
            cfg = child.conv_config
            if cfg.C_in_block > max_c_in_block:
                new_c_in = _ALIGNMENT
                for candidate in range(max_c_in_block, _ALIGNMENT - 1, -_ALIGNMENT):
                    if child.in_channels % candidate == 0:
                        new_c_in = candidate
                        break
                # Preserve all existing Conv3dConfig fields from get_conv3d_config and only override C_in_block.
                # Rebuilding Conv3dConfig from a subset of fields can silently drop newer defaults after rebases.
                cfg.C_in_block = new_c_in
        else:
            _cap_conv3d_blocking(child, max_c_in_block)


class WanVAEEncoder(Module):
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
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.dtype = dtype
        self._norm_out_compute_kernel_config = None

        if mesh_device is not None:
            self._norm_out_compute_kernel_config = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=is_blackhole(),
                packer_l1_acc=False,
            )

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
            dtype=dtype,
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
                        dtype=dtype,
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
                            dtype=dtype,
                        )
                    )
                    if scale in attn_scales:
                        self.down_blocks.append(
                            WanAttentionBlock(
                                dim=out_dim,
                                mesh_device=mesh_device,
                                ccl_manager=ccl_manager,
                                parallel_config=parallel_config,
                                dtype=dtype,
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
                            dtype=dtype,
                        )
                    )
                    scale /= 2.0

        # middle blocks
        self.mid_block = WanMidBlock(
            dim=out_dim,
            num_layers=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
        )
        if out_dim > 384:
            for attn in self.mid_block.attentions:
                attn.sdpa_program_config = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
                    q_chunk_size=32,
                    k_chunk_size=32,
                    exp_approx_mode=False,
                )

        # Match diffusers WanEncoder3d head order: norm_out -> silu -> conv_out.
        self.norm_out = RMSNorm(
            embedding_dim=out_dim,
            norm_eps=1e-12,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
            dtype=dtype,
            fused_activation=None,
        )
        self.conv_out = WanCausalConv3d(
            out_dim,
            z_dim,
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
        )

        # Cap by default to avoid static buffer/L1 pressure issues; allow explicit opt-out for PCC debugging.
        _disable_cap = os.environ.get("LINGBOT_VA_ENCODER_SKIP_CAP_CONV3D", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        _max_c_in_block_env = os.environ.get("LINGBOT_VA_ENCODER_MAX_C_IN_BLOCK", "").strip()
        max_c_in_block = (
            int(_max_c_in_block_env)
            if _max_c_in_block_env.isdigit() and int(_max_c_in_block_env) >= _ALIGNMENT
            else _DEFAULT_MAX_C_IN_BLOCK
        )
        max_c_in_block = max(_ALIGNMENT, (max_c_in_block // _ALIGNMENT) * _ALIGNMENT)
        if not _disable_cap:
            _cap_conv3d_blocking(self, max_c_in_block=max_c_in_block)
        self._patch_wormhole_math_for_encoder_parity()
        # One feat_cache slot per WanCausalConv3d call order in forward (must match streaming wrapper list size).
        self.num_causal_conv_slots = count_convs(self)

    def _patch_wormhole_math_for_encoder_parity(self) -> None:
        if self.mesh_device is None or is_blackhole():
            return
        arch = self.mesh_device.arch()
        for m in _iter_tt_module_submodules(self):
            if isinstance(m, WanCausalConv3d):
                patch_wan_causal_conv_wormhole_bf16_parity(m, self.mesh_device)
            elif isinstance(m, WanResidualBlock):
                m.matmul_compute_kernel_config = ttnn.init_device_compute_kernel_config(
                    arch,
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=False,
                )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "norm_out.gamma" in state:
            state["norm_out.weight"] = state.pop("norm_out.gamma").squeeze()

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        logical_h: int,
        feat_cache: list[ttnn.Tensor] | None = None,
        feat_idx: list[int] | None = None,
    ) -> tuple[ttnn.Tensor, int]:
        if feat_cache is None:
            feat_cache = [None] * self.num_causal_conv_slots
        if feat_idx is None:
            feat_idx = [0]

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

        # Match tt_dit WanEncoder3d: conv3d outputs ROW_MAJOR; residual / attention expect TILE.
        # WanResample expects ROW_MAJOR input, then we convert back to TILE for the next block.
        x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)

        for down_block in self.down_blocks:
            if isinstance(down_block, WanResample):
                x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.ROW_MAJOR_LAYOUT)
                x_BTHWC, logical_h = down_block(x_BTHWC, logical_h, feat_cache, feat_idx)
                x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)
            elif isinstance(down_block, WanResidualBlock):
                x_BTHWC = down_block(x_BTHWC, logical_h, feat_cache, feat_idx)
            elif isinstance(down_block, WanAttentionBlock):
                x_BTHWC = down_block(x_BTHWC, logical_h)
            elif isinstance(down_block, WanResidualDownBlock):
                x_BTHWC, logical_h = down_block(x_BTHWC, logical_h, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                raise ValueError(f"Unsupported downblock type: {type(down_block)}")

        x_BTHWC = self.mid_block(x_BTHWC, logical_h, feat_cache, feat_idx)

        if self._norm_out_compute_kernel_config is not None:
            x_silu_tile_BTHWC = self.norm_out(x_BTHWC, compute_kernel_config=self._norm_out_compute_kernel_config)
        else:
            x_silu_tile_BTHWC = self.norm_out(x_BTHWC)
        x_silu_tile_BTHWC = ttnn.silu(x_silu_tile_BTHWC)
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
