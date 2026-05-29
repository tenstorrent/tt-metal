# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.experimental.audiox.tt.common import linear_weight, to_tt
from models.experimental.audiox.tt.rotary import precompute_rotary_cos_sin
from models.experimental.audiox.tt.transformer_block import TtTransformerBlock


def _block_state_dict(sd: dict, prefix: str) -> dict:
    plen = len(prefix)
    return {k[plen:]: v for k, v in sd.items() if k.startswith(prefix)}


class TtContinuousTransformer:
    """TTNN port of the AudioX continuous-transformer stack: optional input
    projection, prepend conditioning, N transformer blocks sharing a single
    rotary table, optional output projection."""

    def __init__(
        self,
        mesh_device,
        state_dict: dict,
        depth: int,
        dim_heads: int = 64,
        cross_attend: bool = False,
        dim_in: int = None,
        dim_out: int = None,
        lazy_layers: bool = False,
    ):
        sd = state_dict
        self.mesh_device = mesh_device
        self.dim_heads = dim_heads
        self.rot_dim = max(dim_heads // 2, 32)
        self.cross_attend = cross_attend
        self.lazy_layers = lazy_layers

        self.project_in_w = to_tt(linear_weight(sd["project_in.weight"]), mesh_device) if dim_in is not None else None
        self.project_out_w = (
            to_tt(linear_weight(sd["project_out.weight"]), mesh_device) if dim_out is not None else None
        )

        if lazy_layers:
            self.layer_state_dicts = [_block_state_dict(sd, f"layers.{i}.") for i in range(depth)]
            self.layers = None
        else:
            self.layer_state_dicts = None
            self.layers = [
                TtTransformerBlock(
                    mesh_device=mesh_device,
                    state_dict=_block_state_dict(sd, f"layers.{i}."),
                    dim_heads=dim_heads,
                    cross_attend=cross_attend,
                )
                for i in range(depth)
            ]

    def deallocate(self) -> None:
        if self.project_in_w is not None:
            ttnn.deallocate(self.project_in_w, force=True)
        if self.project_out_w is not None:
            ttnn.deallocate(self.project_out_w, force=True)

        if self.layers is None:
            return

        for layer in self.layers:
            layer.deallocate()

    def __call__(
        self,
        x: ttnn.Tensor,
        prepend_embeds: ttnn.Tensor = None,
        context: ttnn.Tensor = None,
    ) -> ttnn.Tensor:
        if self.project_in_w is not None:
            x = ttnn.linear(x, self.project_in_w)

        if prepend_embeds is not None:
            x = ttnn.concat([prepend_embeds, x], dim=1)

        seq_len = x.shape[1]
        cos, sin = precompute_rotary_cos_sin(seq_len, self.rot_dim, mesh_device=self.mesh_device)

        if self.lazy_layers:
            for layer_sd in self.layer_state_dicts:
                layer = TtTransformerBlock(
                    mesh_device=self.mesh_device,
                    state_dict=layer_sd,
                    dim_heads=self.dim_heads,
                    cross_attend=self.cross_attend,
                )
                try:
                    x = layer(x, context=context, rotary_cos=cos, rotary_sin=sin)
                finally:
                    layer.deallocate()
        else:
            for layer in self.layers:
                x = layer(x, context=context, rotary_cos=cos, rotary_sin=sin)

        if self.project_out_w is not None:
            x = ttnn.linear(x, self.project_out_w)

        return x
