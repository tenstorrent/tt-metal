# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import List, Optional
from models.experimental.detr3d.reference.model_utils import (
    QueryAndGroup,
    GatherOperation,
    FurthestPointSampling,
    shift_scale_points,
    BoxProcessor,
    Conv2d,
)
import copy
import math
import numpy as np
from functools import partial

import torch
import torch.nn.functional as F


class Conv1d(torch.nn.Conv1d):
    """
    A wrapper around :class:`torch.nn.Conv1d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv1d`:
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def get_nested_list_shape(lst):
    shape = []
    while isinstance(lst, (list, tuple)):
        shape.append(len(lst))
        if len(lst) == 0:
            break
        lst = lst[0]
    return tuple(shape)


def truncate_nested_list(lst, max_items=3):
    if isinstance(lst, (list, tuple)):
        if len(lst) > max_items:
            return lst[:max_items] + ["..."]
        return [truncate_nested_list(item) for item in lst]
    return lst


class SharedMLP(nn.Sequential):
    def __init__(
        self,
        args: List[int],
        *,
        bn: bool = False,
        activation=nn.ReLU(inplace=True),
        preact: bool = False,
        first: bool = False,
        name: str = "",
    ):
        super().__init__()

        for i in range(len(args) - 1):
            # print(
            #     "i and args are  ",
            #     i,
            #     ((not first or not preact or (i != 0)) and bn),
            # )
            self.add_module(
                name + "layer{}".format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation if (not first or not preact or (i != 0)) else None,
                    preact=preact,
                ),
            )


# 1: PointnetSAModuleVotes
class PointnetSAModuleVotes(nn.Module):
    """Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes"""

    def __init__(
        self,
        *,
        mlp: List[int],
        npoint: int = None,
        radius: float = None,
        nsample: int = None,
        bn: bool = True,
        use_xyz: bool = True,
        pooling: str = "max",
        sigma: float = None,  # for RBF pooling
        normalize_xyz: bool = False,  # noramlize local XYZ with radius
        sample_uniformly: bool = False,
        ret_unique_cnt: bool = False,
    ):
        super().__init__()
        params = {k: v for k, v in locals().items() if k != "self"}
        print("ref PointnetSAModuleVotes init is called with params:")
        for k, v in params.items():
            print(f"  {k}: {v}")

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius / 2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt

        if npoint is not None:
            self.grouper = QueryAndGroup(
                radius,
                nsample,
                use_xyz=use_xyz,
                ret_grouped_xyz=True,
                normalize_xyz=normalize_xyz,
                sample_uniformly=sample_uniformly,
                ret_unique_cnt=ret_unique_cnt,
            )
        else:
            self.grouper = pointnet2_utils.GroupAll(use_xyz, ret_grouped_xyz=True)

        mlp_spec = mlp
        if use_xyz and len(mlp_spec) > 0:
            mlp_spec[0] += 3
        # print("inputs to SharedMLP", mlp_spec, bn)
        self.mlp_module = SharedMLP(mlp_spec, bn=bn)
        self.gather_operation = GatherOperation()
        self.furthest_point_sample = FurthestPointSampling()

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor = None, inds: torch.Tensor = None
    ) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        inds : torch.Tensor
            (B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        """
        params = {k: v for k, v in locals().items() if k != "self"}
        print(f"ref PointnetSAModuleVotes forward called with params:")
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: Tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
            elif isinstance(v, (int, float, bool)):
                print(f"  {k}: {v}")
            elif v is None:
                print(f"  {k}: None")
            else:
                print(f"  {k}: {type(v).__name__}")

        # xyz: torch.Tensor,features: torch.Tensor ,inds: torch.Tensor",xyz.shape,features.shape,inds.shape
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if inds is None:
            inds = self.furthest_point_sample(xyz, self.npoint)
        else:
            assert inds.shape[1] == self.npoint
        new_xyz = (
            self.gather_operation(xyz_flipped, inds).transpose(1, 2).contiguous() if self.npoint is not None else None
        )

        if not self.ret_unique_cnt:
            grouped_features, grouped_xyz = self.grouper(xyz, new_xyz, features)  # (B, C, npoint, nsample)
        else:
            grouped_features, grouped_xyz, unique_cnt = self.grouper(
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample), (B,3,npoint,nsample), (B,npoint)
        # print("input to shared mlpforward ", grouped_features.shape)
        new_features = self.mlp_module(grouped_features)  # (B, mlp[-1], npoint, nsample)
        if self.pooling == "max":
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
        elif self.pooling == "avg":
            new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
        elif self.pooling == "rbf":
            # Use radial basis function kernel for weighted sum of features (normalized by nsample and sigma)
            # Ref: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
            rbf = torch.exp(
                -1 * grouped_xyz.pow(2).sum(1, keepdim=False) / (self.sigma**2) / 2
            )  # (B, npoint, nsample)
            new_features = torch.sum(new_features * rbf.unsqueeze(1), -1, keepdim=True) / float(
                self.nsample
            )  # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

        if not self.ret_unique_cnt:
            return new_xyz, new_features, inds
        else:
            return new_xyz, new_features, inds, unique_cnt


# 2: Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead=4,
        dim_feedforward=128,
        dropout=0.0,
        dropout_attn=None,
        activation="relu",
        normalize_before=True,
        norm_name="ln",
        use_ffn=True,
        ffn_use_bias=True,
    ):
        super().__init__()
        # params = {k: v for k, v in locals().items() if k != 'self'}
        # print("TransformerEncoderLayer init is called")
        # for k, v in params.items():
        #     print(f"  {k}: {v}")
        if dropout_attn is None:
            dropout_attn = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
        self.use_ffn = use_ffn
        if self.use_ffn:
            # Implementation of Feedforward model
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=ffn_use_bias)
            # self.dropout = nn.Dropout(dropout, inplace=True)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=ffn_use_bias)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            # self.dropout2 = nn.Dropout(dropout, inplace=True)

        self.norm1 = nn.LayerNorm(d_model)
        # self.dropout1 = nn.Dropout(dropout, inplace=True)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        # print("TransformerEncoderLayer forwardpost is called")
        params = {k: v for k, v in locals().items() if k != "self"}
        # for k, v in params.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"  {k}: Tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
        #     elif isinstance(v, (int, float, bool)):
        #         print(f"  {k}: {v}")
        #     elif v is None:
        #         print(f"  {k}: None")
        #     else:
        #         print(f"  {k}: {type(v).__name__}")
        q = k = self.with_pos_embed(src, pos)
        value = src
        src2 = self.self_attn(q, k, value=value, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + src2  # self.dropout1(src2)
        # if self.use_norm_fn_on_input: # NOTE: is this required?
        #     src = self.norm1(src)
        src = self.norm1(src)  # Removing if self.use_norm_fn_on_input
        if self.use_ffn:
            src2 = self.linear2(self.activation(self.linear1(src)))
            src = src + src2  # self.dropout2(src2)
            src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        return_attn_weights: Optional[Tensor] = False,
    ):
        # print("TransformerEncoderLayer forwardpre is called")
        # params = {k: v for k, v in locals().items() if k != 'self'}
        # for k, v in params.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"  {k}: Tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
        #     elif isinstance(v, (int, float, bool)):
        #         print(f"  {k}: {v}")
        #     elif v is None:
        #         print(f"  {k}: None")
        #     else:
        #         print(f"  {k}: {type(v).__name__}")
        src2 = self.norm1(src)
        value = src2
        q = k = self.with_pos_embed(src2, pos)
        src2, attn_weights = self.self_attn(
            q, k, value=value, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + src2  # self.dropout1(src2)
        if self.use_ffn:
            src2 = self.norm2(src)
            src2 = self.linear2(self.activation(self.linear1(src2)))
            src = src + src2  # self.dropout2(src2)
        if return_attn_weights:
            return src, attn_weights
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        return_attn_weights: Optional[Tensor] = False,
    ):
        # print("TransformerEncoderLayer forwardddd is called")
        # params = {k: v for k, v in locals().items() if k != 'self'}
        # for k, v in params.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"  {k}: Tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
        #     elif isinstance(v, (int, float, bool)):
        #         print(f"  {k}: {v}")
        #     elif v is None:
        #         print(f"  {k}: None")
        #     else:
        #         print(f"  {k}: {type(v).__name__}")
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, return_attn_weights)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

    def extra_repr(self):
        st = ""
        if hasattr(self.self_attn, "dropout"):
            st += f"attn_dr={self.self_attn.dropout}"
        return st


# 3: Transformer Encoder init
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, weight_init_name="xavier_uniform"):
        super().__init__()
        params = {k: v for k, v in locals().items() if k != "self"}
        # print("TransformerEncoder init is called")
        # for k, v in params.items():
        #     print(f"  {k}: {v}")
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name):
        func = nn.init.xavier_uniform_
        for p in self.parameters():
            if p.dim() > 1:
                func(p)


# 4:MaskedTransformerEncoder


class MaskedTransformerEncoder(TransformerEncoder):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        masking_radius,
        interim_downsampling,
        norm=None,
        weight_init_name="xavier_uniform",
    ):
        super().__init__(encoder_layer, num_layers, norm=norm, weight_init_name=weight_init_name)
        params = {k: v for k, v in locals().items() if k != "self"}
        # print("MaskedTransformerEncoder init is called")
        # for k, v in params.items():
        #     print(f"  {k}: {v}")
        assert len(masking_radius) == num_layers
        self.masking_radius = masking_radius
        self.interim_downsampling = interim_downsampling

    def compute_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        xyz: Optional[Tensor] = None,
        transpose_swap: Optional[bool] = False,
    ):
        # print("MaskedTransformerEncoder forward is called")
        params = {k: v for k, v in locals().items() if k != "self"}
        # for k, v in params.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"  {k}: Tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
        #     elif isinstance(v, (int, float, bool)):
        #         print(f"  {k}: {v}")
        #     elif v is None:
        #         print(f"  {k}: None")
        #     else:
        #         print(f"  {k}: {type(v).__name__}")
        if transpose_swap:
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)

        output = src
        xyz_dist = None
        xyz_inds = None

        for idx, layer in enumerate(self.layers):
            mask = None
            if self.masking_radius[idx] > 0:
                mask, xyz_dist = self.compute_mask(xyz, self.masking_radius[idx], xyz_dist)
                # mask must be tiled to num_heads of the transformer
                bsz, n, n = mask.shape
                nhead = layer.nhead
                mask = mask.unsqueeze(1)
                mask = mask.repeat(1, nhead, 1, 1)
                mask = mask.view(bsz * nhead, n, n)

            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

            if idx == 0 and self.interim_downsampling:
                # output is npoints x batch x channel. make batch x channel x npoints
                output = output.permute(1, 2, 0)
                xyz, output, xyz_inds = self.interim_downsampling(xyz, output)
                # swap back
                output = output.permute(2, 0, 1)

        if self.norm is not None:
            output = self.norm(output)

        if transpose_swap:
            output = output.permute(1, 2, 0).view(bs, c, h, w).contiguous()

        return xyz, output, xyz_inds


# 5: Generic MLP (ENCODER TO DECODER PROJ)


class GenericMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        norm_fn_name=None,
        activation="relu",
        use_conv=False,
        dropout=None,
        hidden_use_bias=False,
        output_use_bias=True,
        output_use_activation=False,
        output_use_norm=False,
        weight_init_name=None,
    ):
        super().__init__()
        params = {k: v for k, v in locals().items() if k != "self"}
        # print("GenericMLP init is called")
        # for k, v in params.items():
        #     print(f"  {k}: {v}")
        activation = nn.ReLU
        norm = None
        if norm_fn_name is not None:
            norm = nn.BatchNorm1d
        if norm_fn_name == "ln" and use_conv:
            norm = lambda x: nn.GroupNorm(1, x)  # easier way to use LayerNorm

        if dropout is not None:
            if not isinstance(dropout, list):
                dropout = [dropout for _ in range(len(hidden_dims))]

        layers = []
        prev_dim = input_dim
        for idx, x in enumerate(hidden_dims):
            # print("hiii")
            if use_conv:
                # layer = Conv1d(
                #     prev_dim, x, 1, bias=hidden_use_bias, norm=norm(x) if norm else None, activation=activation()
                # )
                layer = nn.Conv1d(prev_dim, x, 1, bias=hidden_use_bias)
                layers.append(layer)
                if norm:
                    layers.append(norm(x))
                layers.append(activation())
            else:
                layer = nn.Linear(prev_dim, x, bias=hidden_use_bias)
                layers.append(layer)
                if norm:
                    layers.append(norm(x))
                layers.append(activation())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout[idx]))
            prev_dim = x
        if use_conv:
            # layer = Conv1d(
            #     prev_dim,
            #     output_dim,
            #     1,
            #     bias=output_use_bias,
            #     norm=norm(output_dim) if output_use_norm else None,
            #     activation=activation() if output_use_activation else None,
            # )
            layer = nn.Conv1d(prev_dim, output_dim, 1, bias=output_use_bias)
            layers.append(layer)
            if output_use_norm:
                # print("iwbefiuwbefw")
                layers.append(norm(output_dim))

            if output_use_activation:
                layers.append(activation())
        else:
            layer = nn.Linear(prev_dim, output_dim, bias=output_use_bias)
            layers.append(layer)
            if output_use_norm:
                # print("iwbefiuwbefw")
                layers.append(norm(output_dim))

            if output_use_activation:
                layers.append(activation())

        self.layers = nn.Sequential(*layers)

        if weight_init_name is not None:
            self.do_weight_init(weight_init_name)

        # print("wifne", self.layers)

    def do_weight_init(self, weight_init_name):
        func = None
        for _, param in self.named_parameters():
            if param.dim() > 1:  # skips batchnorm/layernorm
                func(param)

    def forward(self, x):
        # print("GenericMLP forward is called")
        params = {k: v for k, v in locals().items() if k != "self"}
        # for k, v in params.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"  {k}: Tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
        #     elif isinstance(v, (int, float, bool)):
        #         print(f"  {k}: {v}")
        #     elif v is None:
        #         print(f"  {k}: None")
        #     else:
        #         print(f"  {k}: {type(v).__name__}")
        output = self.layers(x)
        # for i, layer in enumerate(self.layers):
        #     output = layer(x)
        #     print(f"torch layer no{i} out is",output.shape)
        return output


# 6: PositionEmbeddingCoordsSine


class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(
        self,
        temperature=10000,
        normalize=False,
        scale=None,
        pos_type="fourier",
        d_pos=None,
        d_in=3,
        gauss_scale=1.0,
    ):
        super().__init__()
        # print("PositionEmbeddingCoordsSine init is called")
        params = {k: v for k, v in locals().items() if k != "self"}
        # for k, v in params.items():
        #     print(f"  {k}: {v}")
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ["sine", "fourier"]
        self.pos_type = pos_type
        self.scale = scale
        if pos_type == "fourier":
            assert d_pos is not None
            assert d_pos % 2 == 0
            # define a gaussian matrix input_ch -> output_ch
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer("gauss_B", B)
            self.d_pos = d_pos

    def get_sine_embeddings(self, xyz, num_channels, input_range):
        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        ndim = num_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        # automatically handle remainder by assiging it to the first dim
        rems = num_channels - (ndim * xyz.shape[2])

        assert (
            ndim % 2 == 0
        ), f"Cannot handle odd sized ndim={ndim} where num_channels={num_channels} and xyz={xyz.shape}"

        final_embeds = []
        prev_dim = 0

        for d in range(xyz.shape[2]):
            cdim = ndim
            if rems > 0:
                # add remainder in increments of two to maintain even size
                cdim += 2
                rems -= 2

            if cdim != prev_dim:
                dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

            # create batch x cdim x nccords embedding
            raw_pos = xyz[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            pos = raw_pos[:, :, None] / dim_t
            pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim

        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2
        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)

        xyz *= 2 * np.pi
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(bsize, npoints, d_out)
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def forward(self, xyz, num_channels=None, input_range=None):
        # print("PositionEmbeddingCoordsSine forward is called",input_range)
        # params = {k: v for k, v in locals().items() if k != 'self'}
        # for k, v in params.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"  {k}: Tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
        #     elif isinstance(v, (int, float, bool)):
        #         print(f"  {k}: {v}")
        #     elif v is None:
        #         print(f"  {k}: None")
        #     elif isinstance(v, (list, tuple)):
        #         try:
        #             nested_shape = get_nested_list_shape(v)
        #             print(f"  {k}: {type(v).__name__}(shape={nested_shape}, values={truncate_nested_list(v)})")
        #         except Exception as e:
        #             print(f"  {k}: {type(v).__name__}, could not get shape or values ({e})")
        #     else:
        # print(f"  {k}: {type(v).__name__}")
        assert isinstance(xyz, torch.Tensor)
        assert xyz.ndim == 3
        # xyz is batch x npoints x 3
        if self.pos_type == "sine":
            with torch.no_grad():
                return self.get_sine_embeddings(xyz, num_channels, input_range)
        elif self.pos_type == "fourier":
            with torch.no_grad():
                return self.get_fourier_embeddings(xyz, num_channels, input_range)
        else:
            raise ValueError(f"Unknown {self.pos_type}")

    def extra_repr(self):
        st = f"type={self.pos_type}, scale={self.scale}, normalize={self.normalize}"
        if hasattr(self, "gauss_B"):
            st += f", gaussB={self.gauss_B.shape}, gaussBsum={self.gauss_B.sum().item()}"
        return st


# 7 : Decoder Layer


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead=4,
        dim_feedforward=256,
        dropout=0.1,
        dropout_attn=None,
        activation="relu",
        normalize_before=True,
        norm_fn_name="ln",
    ):
        super().__init__()
        # print("TransformerDecoderLayer init is called")
        # params = {k: v for k, v in locals().items() if k != 'self'}
        # for k, v in params.items():
        #     print(f"  {k}: {v}")
        if dropout_attn is None:
            dropout_attn = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.dropout3 = nn.Dropout(dropout, inplace=True)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        return_attn_weights: Optional[bool] = False,
    ):
        # print("TransformerDecoderLayer forward post is called")
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        return_attn_weights: Optional[bool] = False,
    ):
        # print("TransformerDecoderLayer forward pre is called")
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        # q=k=v=tgt2=tgt
        tgt2, _ = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        # return tgt2, None
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, attn = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        return_attn_weights: Optional[bool] = False,
    ):
        # print("TransformerDecoderLayer forwarddd is called")
        # params = {k: v for k, v in locals().items() if k != 'self'}
        # for k, v in params.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"  {k}: Tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
        #     elif isinstance(v, (int, float, bool)):
        #         print(f"  {k}: {v}")
        #     elif v is None:
        #         print(f"  {k}: None")
        #     else:
        #         print(f"  {k}: {type(v).__name__}")
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
                return_attn_weights,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
            return_attn_weights,
        )


# 8: Transformer Decoder


class TransformerDecoder(nn.Module):
    def __init__(
        self, decoder_layer, num_layers, norm_fn_name="ln", return_intermediate=False, weight_init_name="xavier_uniform"
    ):
        super().__init__()
        # print("TransformerDecoder init is called")
        params = {k: v for k, v in locals().items() if k != "self"}
        # for k, v in params.items():
        #     print(f"  {k}: {v}")
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = None
        if norm_fn_name is not None:
            self.norm = nn.LayerNorm(self.layers[0].linear2.out_features)
        self.return_intermediate = return_intermediate
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name):
        func = nn.init.xavier_uniform_
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        transpose_swap: Optional[bool] = False,
        return_attn_weights: Optional[bool] = False,
    ):
        # print("TransformerDecoder forward is called")
        params = {k: v for k, v in locals().items() if k != "self"}
        # for k, v in params.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"  {k}: Tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
        #     elif isinstance(v, (int, float, bool)):
        #         print(f"  {k}: {v}")
        #     elif v is None:
        #         print(f"  {k}: None")
        #     else:
        #         print(f"  {k}: {type(v).__name__}")
        if transpose_swap:
            bs, c, h, w = memory.shape
            memory = memory.flatten(2).permute(2, 0, 1)  # memory: bs, c, t -> t, b, c
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)
        output = tgt

        intermediate = []
        attns = []

        for layer in self.layers:
            output, attn = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                return_attn_weights=return_attn_weights,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
            if return_attn_weights:
                attns.append(attn)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if return_attn_weights:
            attns = torch.stack(attns)

        if self.return_intermediate:
            return torch.stack(intermediate), attns

        return output, attns


def build_preencoder(args):
    mlp_dims = [3 * int(args.use_color), 64, 128, args.enc_dim]
    # print("pre encoder is PointnetSAModuleVotes")
    preencoder = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=args.preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    return preencoder


def build_encoder(args):
    if args.enc_type == "vanilla":
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
        )
        encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=args.enc_nlayers)
    elif args.enc_type in ["masked"]:
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
        )
        interim_downsampling = PointnetSAModuleVotes(
            radius=0.4,
            nsample=32,
            npoint=args.preenc_npoints // 2,
            mlp=[args.enc_dim, 256, 256, args.enc_dim],
            normalize_xyz=True,
        )

        masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
        encoder = MaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,
            interim_downsampling=interim_downsampling,
            masking_radius=masking_radius,
        )
    else:
        raise ValueError(f"Unknown encoder type {args.enc_type}")
    return encoder


def build_decoder(args):
    decoder_layer = TransformerDecoderLayer(
        d_model=args.dec_dim,
        nhead=args.dec_nhead,
        dim_feedforward=args.dec_ffn_dim,
        dropout=args.dec_dropout,
    )
    decoder = TransformerDecoder(decoder_layer, num_layers=args.dec_nlayers, return_intermediate=True)
    return decoder


def build_3detr(args, dataset_config):
    pre_encoder = build_preencoder(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    model = Model3DETR(
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        mlp_dropout=args.mlp_dropout,
        num_queries=args.nqueries,
    )
    output_processor = BoxProcessor(dataset_config)
    return model, output_processor


# 9: Detr3d model


class Model3DETR(nn.Module):
    """
    Main 3DETR model. Consists of the following learnable sub-models
    - pre_encoder: takes raw point cloud, subsamples it and projects into "D" dimensions
                Input is a Nx3 matrix of N point coordinates
                Output is a N'xD matrix of N' point features
    - encoder: series of self-attention blocks to extract point features
                Input is a N'xD matrix of N' point features
                Output is a N''xD matrix of N'' point features.
                N'' = N' for regular encoder; N'' = N'//2 for masked encoder
    - query computation: samples a set of B coordinates from the N'' points
                and outputs a BxD matrix of query features.
    - decoder: series of self-attention and cross-attention blocks to produce BxD box features
                Takes N''xD features from the encoder and BxD query features.
    - mlp_heads: Predicts bounding box parameters and classes from the BxD box features
    """

    def __init__(
        self,
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=256,
        decoder_dim=256,
        position_embedding="fourier",
        mlp_dropout=0.3,
        num_queries=256,
    ):
        super().__init__()
        # print("Model3DETR init is called")
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [encoder_dim]
        else:
            hidden_dims = [encoder_dim, encoder_dim]
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        # print(
        #     "args are",
        #     encoder_dim,
        #     hidden_dims,
        #     decoder_dim,
        # )
        self.pos_embedding = PositionEmbeddingCoordsSine(d_pos=decoder_dim, pos_type=position_embedding, normalize=True)
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        self.decoder = decoder
        self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)

        self.num_queries = num_queries
        self.box_processor = BoxProcessor(dataset_config)
        self.furthest_point_sample = FurthestPointSampling()

    def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the box
        # add 1 for background/not-an-object class
        semcls_head = mlp_func(output_dim=dataset_config.num_semcls + 1)

        # geometry of the box
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)
        angle_cls_head = mlp_func(output_dim=dataset_config.num_angle_bin)
        angle_reg_head = mlp_func(output_dim=dataset_config.num_angle_bin)

        mlp_heads = [
            ("sem_cls_head", semcls_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("angle_cls_head", angle_cls_head),
            ("angle_residual_head", angle_reg_head),
        ]
        self.mlp_heads = nn.ModuleDict(mlp_heads)

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_inds = self.furthest_point_sample(encoder_xyz, self.num_queries)
        query_inds = query_inds.long()
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)

        # Gater op above can be replaced by the three lines below from the pointnet2 codebase
        # xyz_flipped = encoder_xyz.transpose(1, 2).contiguous()
        # query_xyz = gather_operation(xyz_flipped, query_inds.int())
        # query_xyz = query_xyz.transpose(1, 2)
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed

    def _break_up_pc(self, pc):
        # pc may contain color/normals.

        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, point_clouds):
        xyz, features = self._break_up_pc(point_clouds)
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz, features)
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # inds: batch x npoints

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        pre_enc_features = pre_enc_features.permute(2, 0, 1)

        # xyz points are in batch x npointx channel order
        enc_xyz, enc_features, enc_inds = self.encoder(pre_enc_features, xyz=pre_enc_xyz)
        if enc_inds is None:
            # encoder does not perform any downsampling
            enc_inds = pre_enc_inds
        else:
            # use gather here to ensure that it works for both FPS and random sampling
            enc_inds = torch.gather(pre_enc_inds, 1, enc_inds.type(torch.int64))
        return enc_xyz, enc_features, enc_inds

    def get_box_predictions(self, query_xyz, point_cloud_dims, box_features):
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x channel x num_queries
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3],
        )
        box_features = box_features.reshape(num_layers * batch, channel, num_queries)

        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        cls_logits = self.mlp_heads["sem_cls_head"](box_features).transpose(1, 2)
        center_offset = self.mlp_heads["center_head"](box_features).sigmoid().transpose(1, 2) - 0.5
        size_normalized = self.mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
        angle_logits = self.mlp_heads["angle_cls_head"](box_features).transpose(1, 2)
        angle_residual_normalized = self.mlp_heads["angle_residual_head"](box_features).transpose(1, 2)

        # reshape outputs to num_layers x batch x nqueries x noutput
        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
        center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
        size_normalized = size_normalized.reshape(num_layers, batch, num_queries, -1)
        angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
        angle_residual_normalized = angle_residual_normalized.reshape(num_layers, batch, num_queries, -1)
        angle_residual = angle_residual_normalized * (np.pi / angle_residual_normalized.shape[-1])

        outputs = []
        for l in range(num_layers):
            # box processor converts outputs so we can get a 3D bounding box
            (
                center_normalized,
                center_unnormalized,
            ) = self.box_processor.compute_predicted_center(center_offset[l], query_xyz, point_cloud_dims)
            angle_continuous = self.box_processor.compute_predicted_angle(angle_logits[l], angle_residual[l])
            size_unnormalized = self.box_processor.compute_predicted_size(size_normalized[l], point_cloud_dims)
            box_corners = self.box_processor.box_parametrization_to_corners(
                center_unnormalized, size_unnormalized, angle_continuous
            )

            # below are not used in computing loss (only for matching/mAP eval)
            # we compute them with no_grad() so that distributed training does not complain about unused variables
            with torch.no_grad():
                (
                    semcls_prob,
                    objectness_prob,
                ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits[l])

            box_prediction = {
                "sem_cls_logits": cls_logits[l],
                "center_normalized": center_normalized.contiguous(),
                "center_unnormalized": center_unnormalized,
                "size_normalized": size_normalized[l],
                "size_unnormalized": size_unnormalized,
                "angle_logits": angle_logits[l],
                "angle_residual": angle_residual[l],
                "angle_residual_normalized": angle_residual_normalized[l],
                "angle_continuous": angle_continuous,
                "objectness_prob": objectness_prob,
                "sem_cls_prob": semcls_prob,
                "box_corners": box_corners,
            }
            outputs.append(box_prediction)

        # intermediate decoder layer outputs are only used during training
        aux_outputs = outputs[:-1]
        outputs = outputs[-1]

        return {
            "outputs": outputs,  # output from last layer of decoder
            "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        }

    def forward(self, inputs, encoder_only=False):
        # print("Model3DETR forward is called")
        point_clouds = inputs["point_clouds"]

        enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds)
        # print("input tn gen", enc_features.shape)
        enc_features = self.encoder_to_decoder_projection(enc_features.permute(1, 2, 0)).permute(2, 0, 1)
        # encoder features: npoints x batch x channel
        # encoder xyz: npoints x batch x 3

        if encoder_only:
            # return: batch x npoints x channels
            return enc_xyz, enc_features.transpose(0, 1)

        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        query_xyz, query_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims)
        # query_embed: batch x channel x npoint
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        # decoder expects: npoints x batch x channel
        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)
        box_features = self.decoder(tgt, enc_features, query_pos=query_embed, pos=enc_pos)[0]
        box_predictions = box_features
        box_predictions = self.get_box_predictions(query_xyz, point_cloud_dims, box_features)
        return box_predictions


def build_3detr(args, dataset_config):
    pre_encoder = build_preencoder(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    model = Model3DETR(
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        mlp_dropout=args.mlp_dropout,
        num_queries=args.nqueries,
    )
    output_processor = BoxProcessor(dataset_config)
    return model, output_processor
