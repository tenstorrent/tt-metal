import ttnn
from .resblock import TtResBlock
from .conv1x1 import TtConv1x1
from ..common import as_replicated_tensor
from models.common.lightweightmodule import LightweightModule


class TtCausalUpsampleBlock(LightweightModule):
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        state_dict_prefix: str,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        temporal_expansion: int = 2,
        spatial_expansion: int = 2,
        has_attention: bool = False,
        affine: bool = True,
        attn_block=None,
        causal: bool = True,
        prune_bottleneck: bool = False,
        padding_mode: str = "replicate",
        bias: bool = True,
    ):
        assert causal
        assert not prune_bottleneck
        assert not has_attention

        self.blocks = []
        for i in range(num_res_blocks):
            self.blocks.append(
                TtResBlock(
                    mesh_device=mesh_device,
                    state_dict=state_dict,
                    state_dict_prefix=f"{state_dict_prefix}blocks.{i}.",
                    channels=in_channels,
                    affine=affine,
                    attn_block=None,
                    causal=causal,
                    prune_bottleneck=prune_bottleneck,
                    padding_mode=padding_mode,
                    bias=bias,
                )
            )

        self.temporal_expansion = temporal_expansion
        self.spatial_expansion = spatial_expansion
        self.out_channels = out_channels

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # Swizzle conv1x1 weights
        def swizzle_weight(w):
            # X (C texp sexp sexp) -> X (texp sexp sexp C)
            w = w.reshape(-1, out_channels, temporal_expansion, spatial_expansion, spatial_expansion)
            w = w.permute(0, 2, 3, 4, 1)
            w = w.reshape(-1, temporal_expansion * spatial_expansion * spatial_expansion * out_channels)
            return w.squeeze()

        self.proj = TtConv1x1(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}proj.",
            in_channels=in_channels,
            out_channels=out_channels * temporal_expansion * (spatial_expansion**2),
            bias=bias,
            swizzle_weight=swizzle_weight,
        )

    def depth_to_spacetime(self, x_NTHWC):
        texp, sexp = self.temporal_expansion, self.spatial_expansion
        B, T, H, W, C = x_NTHWC.shape[0], x_NTHWC.shape[1], x_NTHWC.shape[2], x_NTHWC.shape[3], x_NTHWC.shape[4]
        x = ttnn.reshape(x_NTHWC, [B, T, H, W, texp, sexp, sexp, self.out_channels])
        x = ttnn.permute(x, [0, 1, 4, 2, 5, 3, 6, 7])  # (B T texp H sexp W sexp C_out)

        x = ttnn.reshape(x, [B, T * texp, H * sexp, W * sexp, self.out_channels])
        if texp > 1:
            # Drop the first texp - 1 frames.
            x = ttnn.slice(x, [0, texp - 1, 0, 0, 0], [B, T * texp, H * sexp, W * sexp, self.out_channels])
        return x

    def forward(self, x_NTHWC):
        for block in self.blocks:
            x_NTHWC = block(x_NTHWC)

        x_NTHWO = self.proj(x_NTHWC)

        x_NTHWC = self.depth_to_spacetime(x_NTHWO)
        ttnn.deallocate(x_NTHWO)
        return x_NTHWC
