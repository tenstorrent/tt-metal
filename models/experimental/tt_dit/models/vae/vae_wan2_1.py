import torch
import ttnn
from loguru import logger
from ...layers.normalization import RMSNorm
from ...layers.linear import Linear
from ...utils.conv3d import _ntuple, get_conv3d_config, prepare_conv3d_weights
from ...utils.substate import substate, indexed_substates

CACHE_T = 2


class WanAttentionBlock:
    def __init__(
        self,
        dim,
        mesh_device,
    ):
        self.dim = dim
        self.mesh_device = mesh_device

        self.norm = RMSNorm(
            embedding_dim=dim,
            norm_eps=1e-6,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
        )
        self.to_qkv = Linear(
            in_features=dim,
            out_features=dim * 3,
            mesh_device=mesh_device,
        )
        self.proj = Linear(
            in_features=dim,
            out_features=dim,
            mesh_device=mesh_device,
        )

        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=32,
            k_chunk_size=256,
            exp_approx_mode=False,  # NOTE: False is more correct
        )
        self.hifi4_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def load_state_dict(self, state_dict):
        def permute_conv2d_weights(weight):
            out_c, in_c, kh, kw = weight.shape
            assert kh == kw == 1
            weight = weight.permute(0, 2, 3, 1).reshape(out_c, in_c)
            return weight

        self.to_qkv.load_state_dict(
            {
                "weight": permute_conv2d_weights(state_dict["to_qkv.weight"]),
                "bias": state_dict["to_qkv.bias"],
            }
        )
        self.proj.load_state_dict(
            {
                "weight": permute_conv2d_weights(state_dict["proj.weight"]),
                "bias": state_dict["proj.bias"],
            }
        )

        self.norm.load_state_dict(
            {
                "weight": state_dict["norm.gamma"].squeeze(),
            }
        )

    def __call__(self, x_BTHWC):
        assert len(x_BTHWC.shape) == 5
        assert x_BTHWC.layout == ttnn.ROW_MAJOR_LAYOUT
        residual_BTHWC = x_BTHWC
        B, T, H, W, C = x_BTHWC.shape

        x_TNC = ttnn.reshape(x_BTHWC, (B * T, H * W, C))
        x_TNC = ttnn.to_layout(x_TNC, ttnn.TILE_LAYOUT)
        x_TNC = self.norm(x_TNC, compute_kernel_config=self.hifi4_compute_kernel_config)
        x_TND = self.to_qkv(x_TNC, compute_kernel_config=self.mm_compute_kernel_config, core_grid=self.core_grid)
        q_THNC, k_THNC, v_THNC = ttnn.transformer.split_query_key_value_and_split_heads(
            x_TND, num_heads=1, transpose_key=False
        )
        out_THNC = ttnn.transformer.scaled_dot_product_attention(
            q_THNC,
            k_THNC,
            v_THNC,
            is_causal=False,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        out_TNC = ttnn.transformer.concatenate_heads(out_THNC)
        out_TND = self.proj(out_TNC, compute_kernel_config=self.mm_compute_kernel_config, core_grid=self.core_grid)
        out_TND = ttnn.to_layout(out_TND, ttnn.ROW_MAJOR_LAYOUT)
        out_BTHWC = ttnn.reshape(out_TND, (B, T, H, W, C))
        return out_BTHWC + residual_BTHWC


class WanCausalConv3d:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        mesh_device,
        stride=1,
        padding=0,
    ):
        self.in_channels = in_channels
        self.unpadded_out_channels = out_channels
        self.TILE_WIDTH = 32
        self.out_channels = self.TILE_WIDTH if out_channels < self.TILE_WIDTH else out_channels
        if self.out_channels != self.unpadded_out_channels:
            logger.warning(f"Padding out_channels from {self.unpadded_out_channels} to {self.out_channels}")

        self.kernel_size = _ntuple(kernel_size, 3)
        self.stride = _ntuple(stride, 3)
        self.mesh_device = mesh_device

        padding = _ntuple(padding, 3)
        # t padding is handled explicitly and depends on the cache.
        # conv3d can handle HW padding internally.
        self.t_front_padding = 2 * padding[0]
        self.padding = (0, padding[1], padding[2])

        self.conv_config = get_conv3d_config(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            padding_mode="zeros",
            grid_size=self.mesh_device.compute_with_storage_grid_size(),
        )

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def load_state_dict(self, state_dict):
        def maybe_pad_out_channels(weight, bias):
            if self.out_channels != self.unpadded_out_channels:
                weight = torch.nn.functional.pad(
                    weight, (0, 0, 0, 0, 0, 0, 0, 0, 0, self.out_channels - self.unpadded_out_channels)
                )
                bias = torch.nn.functional.pad(bias, (0, self.out_channels - self.unpadded_out_channels))
            return weight, bias

        padded_weight, padded_bias = maybe_pad_out_channels(state_dict["weight"], state_dict["bias"])

        self.conv_weight, self.conv_bias = prepare_conv3d_weights(
            self.mesh_device,
            padded_weight,
            padded_bias,
            self.conv_config,
        )

    def __call__(self, x_BTHWC, cache_x_BTHWC=None):
        # NOTE: T padding is handled explicitly and depends on the cache
        t_front_padding = self.t_front_padding
        if cache_x_BTHWC is not None and t_front_padding > 0:
            # concat on T
            x_BTHWC = ttnn.concat([cache_x_BTHWC, x_BTHWC], dim=1)
            t_front_padding -= cache_x_BTHWC.shape[1]
        if t_front_padding > 0:
            # Padding only works on the lowest 3 dims. reshape input.
            B, T, H, W, C = x_BTHWC.shape
            x_BTNC = ttnn.reshape(x_BTHWC, (B, T, H * W, C))
            x_BTNC = ttnn.pad(x_BTNC, [(0, 0), (t_front_padding, 0), (0, 0), (0, 0)], value=0.0)
            x_BTHWC = ttnn.reshape(x_BTNC, (B, T + t_front_padding, H, W, C))

        return ttnn.experimental.conv3d(
            input_tensor=x_BTHWC,
            weight_tensor=self.conv_weight,
            bias_tensor=self.conv_bias,
            config=self.conv_config,
            compute_kernel_config=self.compute_kernel_config,
        )


class WanResidualBlock:
    def __init__(
        self,
        in_dim,
        out_dim,
        mesh_device,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mesh_device = mesh_device

        self.norm1 = RMSNorm(
            embedding_dim=in_dim,
            norm_eps=1e-6,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
        )
        self.conv1 = WanCausalConv3d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
        )
        self.norm2 = RMSNorm(
            embedding_dim=out_dim,
            norm_eps=1e-6,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
        )
        self.conv2 = WanCausalConv3d(
            in_channels=out_dim,
            out_channels=out_dim,
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
        )

        if in_dim != out_dim:
            self.conv_shortcut = Linear(
                in_features=in_dim,
                out_features=out_dim,
                mesh_device=mesh_device,
            )
        else:
            self.conv_shortcut = None

        self.hifi4_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def load_state_dict(self, state_dict):
        def rename_norm_state(state):
            return {"weight": state["gamma"].squeeze()}

        self.norm1.load_state_dict(rename_norm_state(substate(state_dict, "norm1")))
        self.norm2.load_state_dict(rename_norm_state(substate(state_dict, "norm2")))
        self.conv1.load_state_dict(substate(state_dict, "conv1"))
        self.conv2.load_state_dict(substate(state_dict, "conv2"))

        def conv_1d_to_matmul_weight(weight):
            out_c, in_c, kt, kh, kw = weight.shape
            assert kt == kh == kw == 1
            weight = weight.reshape(out_c, in_c)
            return weight

        if self.conv_shortcut is not None:
            self.conv_shortcut.load_state_dict(
                {
                    "weight": conv_1d_to_matmul_weight(state_dict["conv_shortcut.weight"]),
                    "bias": state_dict["conv_shortcut.bias"],
                }
            )

    def __call__(self, x_BTHWC, feat_cache=None, feat_idx=[0]):
        x_tile_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)
        # ttnn.deallocate(x_BTHWC)
        h_tile_BTHWC = (
            self.conv_shortcut(
                x_tile_BTHWC, compute_kernel_config=self.hifi4_compute_kernel_config, core_grid=self.core_grid
            )
            if self.conv_shortcut is not None
            else x_tile_BTHWC
        )
        x_norm_tile_BTHWC = self.norm1(x_tile_BTHWC, compute_kernel_config=self.hifi4_compute_kernel_config)
        x_silu_tile_BTHWC = ttnn.silu(x_norm_tile_BTHWC)  # NOTE: potential correctness issue
        # ttnn.deallocate(x_norm_tile_BTHWC)
        x_BTHWC = ttnn.to_layout(x_silu_tile_BTHWC, ttnn.ROW_MAJOR_LAYOUT)
        # ttnn.deallocate(x_silu_tile_BTHWC)

        # Cached conv
        if feat_cache is not None:
            # Prepare to cache the current activation for future use
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
            if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None:
                # Current activation is too short, so append the cached activation as well
                cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)

            x_conv_BTHWC = self.conv1(x_BTHWC, feat_cache[idx])
            # ttnn.deallocate(x_BTHWC)
            # NOTE: Should deallocate feat_cache[idx] after it's reassigned
            feat_cache[idx] = cache_x_BTHWC
            feat_idx[0] += 1
        else:
            x_conv_BTHWC = self.conv1(x_BTHWC)
            # ttnn.deallocate(x_BTHWC)

        x_tile_BTHWC = ttnn.to_layout(x_conv_BTHWC, ttnn.TILE_LAYOUT)
        # ttnn.deallocate(x_conv_BTHWC)
        x_norm_tile_BTHWC = self.norm2(x_tile_BTHWC, compute_kernel_config=self.hifi4_compute_kernel_config)
        # ttnn.deallocate(x_tile_BTHWC)
        x_silu_tile_BTHWC = ttnn.silu(x_norm_tile_BTHWC)  # NOTE: potential correctness issue
        # ttnn.deallocate(x_norm_tile_BTHWC)
        x_BTHWC = ttnn.to_layout(x_silu_tile_BTHWC, ttnn.ROW_MAJOR_LAYOUT)
        # ttnn.deallocate(x_silu_tile_BTHWC)

        # Cached conv
        if feat_cache is not None:
            # Prepare to cache the current activation for future use
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
            if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None:
                # Current activation is too short, so append the cached activation as well
                cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)

            x_conv_BTHWC = self.conv2(x_BTHWC, feat_cache[idx])
            # ttnn.deallocate(x_BTHWC)
            # NOTE: Should deallocate feat_cache[idx] after it's reassigned
            feat_cache[idx] = cache_x_BTHWC
            feat_idx[0] += 1
        else:
            x_conv_BTHWC = self.conv2(x_BTHWC)
            # ttnn.deallocate(x_BTHWC)

        # Add residual
        x_tile_BTHWC = ttnn.to_layout(x_conv_BTHWC, ttnn.TILE_LAYOUT)
        # ttnn.deallocate(x_conv_BTHWC)
        x_tile_BTHWC = ttnn.add(h_tile_BTHWC, x_tile_BTHWC)
        # ttnn.deallocate(h_tile_BTHWC)
        x_BTHWC = ttnn.to_layout(x_tile_BTHWC, ttnn.ROW_MAJOR_LAYOUT)
        # ttnn.deallocate(x_tile_BTHWC)
        return x_BTHWC


class WanMidBlock:
    def __init__(
        self,
        dim,
        mesh_device,
        num_layers=1,
    ):
        self.dim = dim
        self.mesh_device = mesh_device
        resnets = []
        attentions = []

        resnets.append(
            WanResidualBlock(
                in_dim=dim,
                out_dim=dim,
                mesh_device=mesh_device,
            )
        )

        for _ in range(num_layers):
            attentions.append(
                WanAttentionBlock(
                    dim=dim,
                    mesh_device=mesh_device,
                )
            )
            resnets.append(
                WanResidualBlock(
                    in_dim=dim,
                    out_dim=dim,
                    mesh_device=mesh_device,
                )
            )

        self.resnets = resnets
        self.attentions = attentions

    def load_state_dict(self, state_dict):
        for i in range(len(self.resnets)):
            self.resnets[i].load_state_dict(substate(state_dict, f"resnets.{i}"))
        for i in range(len(self.attentions)):
            self.attentions[i].load_state_dict(substate(state_dict, f"attentions.{i}"))

    def __call__(self, x_BTHWC, feat_cache=None, feat_idx=[0]):
        x_res_BTHWC = self.resnets[0](x_BTHWC, feat_cache, feat_idx)
        # ttnn.deallocate(x_BTHWC)
        x_BTHWC = x_res_BTHWC
        for i in range(len(self.attentions)):
            x_attn_BTHWC = self.attentions[i](x_BTHWC)
            # ttnn.deallocate(x_BTHWC)
            x_BTHWC = self.resnets[i + 1](x_attn_BTHWC, feat_cache, feat_idx)
            # ttnn.deallocate(x_attn_BTHWC)
        return x_BTHWC


class WanConv2d:
    """
    A conv2d implemented with conv3d.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        mesh_device,
        stride=1,
        padding=0,
    ):
        self.in_channels = in_channels
        self.unpadded_out_channels = out_channels
        self.TILE_WIDTH = 32
        self.out_channels = self.TILE_WIDTH if out_channels < self.TILE_WIDTH else out_channels
        if self.out_channels != self.unpadded_out_channels:
            logger.warning(f"Padding out_channels from {self.unpadded_out_channels} to {self.out_channels}")

        self.kernel_size = _ntuple(kernel_size, 3)
        self.stride = _ntuple(stride, 3)
        self.mesh_device = mesh_device

        self.padding = _ntuple(padding, 3)

        self.conv_config = get_conv3d_config(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            padding_mode="zeros",
            grid_size=self.mesh_device.compute_with_storage_grid_size(),
        )
        print(f"Loaded conv_config: {self.conv_config}")

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def load_state_dict(self, state_dict):
        def conv2d_to_conv3d_weight(weight):
            weight = weight.unsqueeze(2)
            return weight

        reshaped_weight = conv2d_to_conv3d_weight(state_dict["weight"])

        self.conv_weight, self.conv_bias = prepare_conv3d_weights(
            self.mesh_device,
            reshaped_weight,
            state_dict["bias"],
            self.conv_config,
        )
        print(f"Loaded conv_weight: {self.conv_weight.shape}, conv_bias: {self.conv_bias.shape}")

    def __call__(self, x_BTHWC):
        return ttnn.experimental.conv3d(
            input_tensor=x_BTHWC,
            weight_tensor=self.conv_weight,
            bias_tensor=self.conv_bias,
            config=self.conv_config,
            compute_kernel_config=self.compute_kernel_config,
        )


class WanResample:
    def __init__(
        self,
        dim,
        mode,
        mesh_device,
        upsample_out_dim=None,
    ):
        self.dim = dim
        self.mode = mode
        self.mesh_device = mesh_device
        upsample_out_dim = upsample_out_dim or dim // 2

        assert mode in ["upsample2d", "upsample3d"]

        self.conv = WanConv2d(
            in_channels=dim,
            out_channels=upsample_out_dim,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            mesh_device=mesh_device,
        )

        if mode == "upsample3d":
            self.time_conv = WanCausalConv3d(
                in_channels=dim,
                out_channels=dim * 2,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                mesh_device=mesh_device,
            )

    def load_state_dict(self, state_dict):
        self.conv.load_state_dict(substate(state_dict, "resample.1"))

        if self.mode == "upsample3d":
            self.time_conv.load_state_dict(substate(state_dict, "time_conv"))

    def __call__(self, x_BTHWC, feat_cache=None, feat_idx=[0]):
        B, T, H, W, C = x_BTHWC.shape
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    t_start = x_BTHWC.shape[1] - CACHE_T
                    cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
                    is_rep = isinstance(feat_cache[idx], str) and feat_cache[idx] == "Rep"
                    assert not (
                        isinstance(feat_cache[idx], str) and not is_rep
                    ), "If feat_cache[idx] is a string, it must be 'Rep'"
                    if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None and not is_rep:
                        cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)

                    if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None and is_rep:
                        # When feat_cache[idx] is "Rep", we need to pad the cache_x_BTHWC with zeros
                        # Padding only works on the lowest 3 dims
                        cache_x_B1NC = ttnn.reshape(cache_x_BTHWC, (B, 1, H * W, C))
                        cache_x_BTNC = ttnn.pad(cache_x_B1NC, [(0, 0), (1, 0), (0, 0), (0, 0)], value=0.0)
                        cache_x_BTHWC = ttnn.reshape(cache_x_BTNC, (B, 2, H, W, C))

                    if is_rep:
                        x_time_BTHWU = self.time_conv(x_BTHWC)
                    else:
                        x_time_BTHWU = self.time_conv(x_BTHWC, feat_cache[idx])
                    # ttnn.deallocate(x_BTHWC)
                    x_BTHWU = x_time_BTHWU
                    feat_cache[idx] = cache_x_BTHWC
                    feat_idx[0] += 1

                    T1 = x_BTHWU.shape[1]
                    x_BTHW2C = ttnn.reshape(x_BTHWU, (B, T1, H, W, 2, C))
                    # ttnn.deallocate(x_BTHWU)
                    x_BT2HWC = ttnn.permute(x_BTHW2C, (0, 1, 4, 2, 3, 5))
                    # ttnn.deallocate(x_BTHW2C)
                    x_BTHWC = ttnn.reshape(x_BT2HWC, (B, T1 * 2, H, W, C))
            else:
                assert False, "feat_cache is None"

        T2 = x_BTHWC.shape[1]
        x_NHWC = ttnn.reshape(x_BTHWC, (B * T2, H, W, C))
        x_upsamped_NHWC = ttnn.upsample(x_NHWC, scale_factor=2)
        # ttnn.deallocate(x_NHWC)
        H2, W2 = x_upsamped_NHWC.shape[1], x_upsamped_NHWC.shape[2]
        x_BTHWC = ttnn.reshape(x_upsamped_NHWC, (B, T2, H2, W2, C))
        x_conv_BTHWC = self.conv(x_BTHWC)
        # ttnn.deallocate(x_BTHWC)
        return x_conv_BTHWC


class WanUpBlock:
    def __init__(
        self,
        in_dim,
        out_dim,
        num_res_blocks,
        mesh_device,
        upsample_mode=None,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_res_blocks = num_res_blocks
        self.upsample_mode = upsample_mode
        self.mesh_device = mesh_device

        assert upsample_mode in ["upsample2d", "upsample3d"] or upsample_mode is None

        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                WanResidualBlock(
                    in_dim=current_dim,
                    out_dim=out_dim,
                    mesh_device=mesh_device,
                )
            )
            current_dim = out_dim
        self.resnets = resnets

        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = WanResample(
                dim=out_dim,
                mode=upsample_mode,
                mesh_device=mesh_device,
            )

    def load_state_dict(self, state_dict):
        for i in range(len(self.resnets)):
            self.resnets[i].load_state_dict(substate(state_dict, f"resnets.{i}"))
        if self.upsamplers is not None:
            self.upsamplers.load_state_dict(substate(state_dict, "upsamplers.0"))

    def __call__(self, x_BTHWC, feat_cache=None, feat_idx=[0]):
        for resnet in self.resnets:
            x_res_BTHWC = resnet(x_BTHWC, feat_cache, feat_idx)
            # ttnn.deallocate(x_BTHWC)
            x_BTHWC = x_res_BTHWC
        if self.upsamplers is not None:
            x_upsampled_BTHWC = self.upsamplers(x_BTHWC, feat_cache, feat_idx)
            # ttnn.deallocate(x_BTHWC)
            x_BTHWC = x_upsampled_BTHWC
        return x_BTHWC


class WanDecoder3d:
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        out_channels: int = 3,
        is_residual: bool = False,
        mesh_device=None,
    ):
        assert not is_residual, "is_residual is not supported"
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample
        self.mesh_device = mesh_device

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        # init block
        self.conv_in = WanCausalConv3d(z_dim, dims[0], kernel_size=3, padding=1, mesh_device=mesh_device)

        # middle blocks
        self.mid_block = WanMidBlock(dims[0], num_layers=1, mesh_device=mesh_device)

        # upsample blocks
        self.up_blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i > 0 and not is_residual:
                # wan vae 2.1
                in_dim = in_dim // 2

            # determine if we need upsampling
            up_flag = i != len(dim_mult) - 1
            # determine upsampling mode, if not upsampling, set to None
            upsample_mode = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"
            # Create and add the upsampling block
            # NOTE: Different codepath if is_residual. Not implemented yet.
            up_block = WanUpBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                upsample_mode=upsample_mode,
                mesh_device=mesh_device,
            )
            self.up_blocks.append(up_block)

        # output blocks
        self.norm_out = RMSNorm(
            embedding_dim=out_dim, norm_eps=1e-6, norm_elementwise_affine=True, bias=False, mesh_device=mesh_device
        )
        self.conv_out = WanCausalConv3d(out_dim, out_channels, kernel_size=3, padding=1, mesh_device=mesh_device)

    def load_state_dict(self, state_dict):
        self.conv_in.load_state_dict(substate(state_dict, "conv_in"))
        self.mid_block.load_state_dict(substate(state_dict, "mid_block"))
        for i, state in enumerate(indexed_substates(state_dict, "up_blocks")):
            self.up_blocks[i].load_state_dict(state)

        def rename_norm_state(state):
            return {"weight": state["gamma"].squeeze()}

        self.norm_out.load_state_dict(rename_norm_state(substate(state_dict, "norm_out")))
        self.conv_out.load_state_dict(substate(state_dict, "conv_out"))

    def __call__(self, x_BTHWC, feat_cache=None, feat_idx=[0], first_chunk=False):
        # NOTE: first_chunk is not used. It would be needed for WanResidualUpBlock.
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
            if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None:
                # Current activation is too short, so append the cached activation as well
                cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)
            x_BTHWC = self.conv_in(x_BTHWC, feat_cache[idx])
            feat_cache[idx] = cache_x_BTHWC
            feat_idx[0] += 1
        else:
            x_BTHWC = self.conv_in(x_BTHWC)

        ## middle
        x_BTHWC = self.mid_block(x_BTHWC, feat_cache, feat_idx)
        # DEBUG
        # return x_BTHWC

        ## upsamples
        for up_block in self.up_blocks:
            x_BTHWC = up_block(x_BTHWC, feat_cache, feat_idx)

        ## head
        x_tile_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)
        # # ttnn.deallocate(x_BTHWC)
        x_norm_tile_BTHWC = self.norm_out(x_tile_BTHWC)
        # # ttnn.deallocate(x_tile_BTHWC)
        x_silu_tile_BTHWC = ttnn.silu(x_norm_tile_BTHWC)
        # # ttnn.deallocate(x_norm_tile_BTHWC)
        x_BTHWC = ttnn.to_layout(x_silu_tile_BTHWC, ttnn.ROW_MAJOR_LAYOUT)
        # # ttnn.deallocate(x_silu_tile_BTHWC)

        if feat_cache is not None:
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
            if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None:
                # Current activation is too short, so append the cached activation as well
                cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)
            x_BTHWC = self.conv_out(x_BTHWC, feat_cache[idx])
            feat_cache[idx] = cache_x_BTHWC
            feat_idx[0] += 1
        else:
            x_BTHWC = self.conv_out(x_BTHWC)
        return x_BTHWC
