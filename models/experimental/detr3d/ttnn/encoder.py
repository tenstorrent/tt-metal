import ttnn
import torch
import copy
from typing import Optional
from models.common.lightweightmodule import LightweightModule
from dataclasses import dataclass, asdict


@dataclass
class EncoderArgs:
    d_model: int = None
    nhead: int = None
    dim_feedforward: int = None
    dropout: float = None
    normalize_before: bool = True
    use_ffn: bool = True
    model_config: dict = None


def get_clones(module, N):
    return [copy.deepcopy(module) for i in range(N)]


class TtTransformerEncoder(LightweightModule):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x: ttnn.Tensor, mode="decode") -> ttnn.Tensor:
        for layer in self.layers:
            x = layer(x, mode=mode)

        if self.norm is not None:
            x = self.norm(x, mode=mode)

        return x


class TtMaskedTransformerEncoder(LightweightModule):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        masking_radius,
        interim_downsampling,
        norm=None,
        device=None,
        encoder_args=EncoderArgs(),
        parameters=None,
    ):
        self.layers = []

        for i in range(num_layers):
            self.layers.append(
                encoder_layer(
                    device,
                    **asdict(encoder_args),
                    parameters=parameters.layers[i],
                )
            )

        self.num_layers = num_layers
        self.masking_radius = masking_radius
        self.interim_downsampling = interim_downsampling
        self.norm = norm
        self.device = device

        assert len(masking_radius) == num_layers

    # def compute_mask(self, xyz, radius, dist=None):
    #     if dist is None or dist.shape[1] != xyz.shape[1]:
    #         # Convert to ttnn tensor if needed
    #         if not isinstance(xyz, ttnn.Tensor):
    #             xyz = ttnn.from_torch(xyz, device=self.device)

    #         # Compute pairwise distances using ttnn operations
    #         # ttnn doesn't have cdist, so we implement it manually
    #         # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    #         xyz_norm_sq = ttnn.sum(ttnn.mul(xyz, xyz), dim=-1, keepdim=True)
    #         xyz_norm_sq_t = ttnn.transpose(xyz_norm_sq, -2, -1)
    #         xyz = ttnn.to_layout(xyz, ttnn.TILE_LAYOUT)
    #         dot_product = ttnn.matmul(xyz, ttnn.transpose(xyz, -2, -1))
    #         dist = xyz_norm_sq + xyz_norm_sq_t - 2 * dot_product
    #         dist = ttnn.sqrt(dist)

    #     # Create mask where distances >= radius
    #     mask = ttnn.ge(dist, radius)
    #     return mask, dist

    def compute_mask(self, xyz, radius, dist=None):
        xyz_torch = ttnn.to_torch(xyz)
        if dist is None or dist.shape[1] != xyz_torch.shape[1]:
            dist = torch.cdist(xyz_torch, xyz_torch, p=2)
        # entries that are True in the mask do not contribute to self-attention
        # so points outside the radius are not considered
        mask = dist >= radius
        mask = torch.zeros_like(mask, dtype=torch.float).masked_fill_(mask, float("-inf"))
        mask_ttnn = ttnn.from_torch(mask, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT)
        return mask_ttnn, dist

    def forward(
        self,
        src,
        mask: Optional[ttnn.Tensor] = None,
        src_key_padding_mask: Optional[ttnn.Tensor] = None,
        pos: Optional[ttnn.Tensor] = None,
        xyz: Optional[ttnn.Tensor] = None,
        transpose_swap: Optional[bool] = False,
    ):
        # Convert inputs to ttnn tensors if needed
        if not isinstance(src, ttnn.Tensor):
            src = ttnn.from_torch(src, device=self.device)

        if transpose_swap:
            bs, c, h, w = src.shape
            # Flatten and permute: (bs, c, h, w) -> (h*w, bs, c)
            src = ttnn.reshape(src, (bs, c, h * w))
            src = ttnn.transpose(src, 1, 2)  # (bs, h*w, c)
            src = ttnn.transpose(src, 0, 1)  # (h*w, bs, c)

            if pos is not None:
                if not isinstance(pos, ttnn.Tensor):
                    pos = ttnn.from_torch(pos, device=self.device)
                pos = ttnn.reshape(pos, (bs, c, h * w))
                pos = ttnn.transpose(pos, 1, 2)
                pos = ttnn.transpose(pos, 0, 1)

        output = src
        xyz_dist = None
        xyz_inds = None
        # print(f"//////////////////////////////////Starting the encoder//////////////////////////////////////")

        for idx, layer in enumerate(self.layers):
            attn_mask = None
            # print(
            #     f"//////////////////////////////////Starting the encoder layer {idx}//////////////////////////////////////"
            # )
            if self.masking_radius[idx] > 0:
                attn_mask, xyz_dist = self.compute_mask(xyz, self.masking_radius[idx], xyz_dist)

                # Expand mask for multi-head attention
                # bsz, n, n = attn_mask.shape
                # nhead = layer.nhead
                attn_mask = ttnn.unsqueeze(attn_mask, 1)  # (bsz, 1, n, n)
                # attn_mask = ttnn.repeat(attn_mask, (1, nhead, 1, 1))  # (bsz, nhead, n, n)
                # attn_mask = ttnn.reshape(attn_mask, (bsz * nhead, n, n))
                # attn_mask = ttnn.reshape(attn_mask, (bsz, n, n))

            # print(f"{output.shape=}")
            # print(f"{attn_mask.shape=}")
            # print(f"{src_key_padding_mask.shape=}")

            output = ttnn.permute(output, (1, 0, 2))

            # print(f"{output.shape=}")
            # print(f"{attn_mask.shape=}")
            # Call transformer layer with ttnn-compatible attention
            output = layer(output, src_mask=attn_mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

            # print(f"{output.shape=}")
            # print(f"{attn_mask.shape=}")
            output = ttnn.permute(output, (1, 0, 2))

            # print(f"{output.shape=}")
            # # print(f"{attn_mask.shape=}")
            # print(
            #     f"//////////////////////////////////Finished the encoder layer {idx}//////////////////////////////////////"
            # )

            if idx == 0 and self.interim_downsampling:
                # Reshape for downsampling: (npoints, batch, channel) -> (batch, channel, npoints)
                output = ttnn.permute(output, (1, 0, 2))  # 2048, 1, 256 -> 1, 2048, 256
                output = ttnn.permute(output, (0, 2, 1))  # 1, 2048, 256 -> 1, 256, 2048
                print(f"{output.shape=}")

                if not isinstance(xyz, torch.Tensor):
                    xyz_torch = ttnn.to_torch(xyz, dtype=torch.float)
                if not isinstance(output, torch.Tensor):
                    output_torch = ttnn.to_torch(output, dtype=torch.float)

                # Apply downsampling (this would need to be implemented in ttnn)
                xyz_torch, output_torch, xyz_inds = self.interim_downsampling(xyz_torch, output_torch)

                xyz = ttnn.from_torch(xyz_torch, device=self.device, dtype=ttnn.bfloat16)
                output = ttnn.from_torch(output_torch, device=self.device, dtype=ttnn.bfloat16)
                print(f"{output.shape=}")

                # output = ttnn.permute(output, (1, 0, 2)) # 2048, 1, 256 -> 1, 2048, 256
                # output = ttnn.permute(output, (0, 2, 1)) # 1, 2048, 256 -> 1, 256, 2048
                output = ttnn.permute(output, (2, 0, 1))
                print(f"{output.shape=}")

        if self.norm is not None:
            output = self.norm(output)

        if transpose_swap:
            # Reshape back to original format
            output = ttnn.transpose(output, 0, 1)  # (bs, h*w, c)
            output = ttnn.transpose(output, 1, 2)  # (bs, c, h*w)
            output = ttnn.reshape(output, (bs, c, h, w))
        print(f"//////////////////////////////////Finished the encoder//////////////////////////////////////")
        print(f"{output.shape=}")

        return xyz, output, xyz_inds
