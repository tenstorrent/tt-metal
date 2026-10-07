# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of BEVFormer's temporal self-attention (``reference/temporal_self_attention.py``).

The value stacks two BEV maps per sample, the previous BEV and the current queries. The offsets
and attention weights read both concatenated and give one set of points per map; the two sampled
results are averaged. ``model_preprocessing.create_temporal_self_attention_parameters`` orders the
Linears' output channels queue-major, so each map's channels are a contiguous half of the row and
splitting the maps onto the batch is a reshape and one permute. The grid scale is folded into the
offset Linear, and the deformable attention core (``multi_scale_deformable_attn_ttnn``) is shared
with the other attentions. Forward runs on device only.

The sampling grid is float32 by default: in bfloat16 a point in (0.5, 1) moves in steps of 2^-8,
0.8 px on the 200x200 BEV map.
"""

import torch
import ttnn
from models.experimental.bevformer.model_config import EMBED_DIMS, GRID_DTYPE, NUM_HEADS, TSA_NUM_POINTS

from models.experimental.bevformer.tt.tt_ms_deformable_attention import multi_scale_deformable_attn_ttnn


def tsa_grid_bias(reference_points, num_heads, num_points, dtype):
    """``2 * ref - 1`` for the folded offset Linear, one copy per (head, point):
    ``(bs * 2, num_query, 1, 2)`` reference points in [0, 1] -> ``(bs * 2, num_query, num_heads * num_points * 2)``.
    Shared by every layer of a frame."""
    rows, num_query = reference_points.shape[0], reference_points.shape[1]
    ref = ttnn.to_layout(reference_points, ttnn.ROW_MAJOR_LAYOUT)
    if ref.dtype != dtype:
        ref = ttnn.typecast(ref, dtype)
    ref = ttnn.reshape(ref, (rows, num_query, 1, 2))
    ref = ttnn.sub(ttnn.mul(ref, 2.0), 1.0)
    ref = ttnn.repeat(ref, ttnn.Shape((1, 1, num_heads * num_points, 1)))
    return ttnn.reshape(ref, (rows, num_query, num_heads * num_points * 2))


class TTTemporalSelfAttention:
    """Temporal self-attention over one ``(bev_h, bev_w)`` BEV map (one level), sampled twice per
    sample: once in the previous BEV, once in the current queries.

    The parameters must come from ``create_temporal_self_attention_parameters``, whose Linears
    emit queue-major channels; the reference module's head-major ones would split wrongly.
    """

    def __init__(
        self,
        params,
        device,
        *,
        bev_shape,
        embed_dims=EMBED_DIMS,
        num_heads=NUM_HEADS,
        num_points=TSA_NUM_POINTS,
        num_bev_queue=2,
        grid_dtype=GRID_DTYPE,
        grid_sample_compute_config=None,
    ):
        """``params`` from ``create_temporal_self_attention_parameters``; ``bev_shape`` the
        ``(bev_h, bev_w)`` map both queue entries are sampled from."""
        assert num_bev_queue == 2, "the value stacks the previous BEV and the current query"
        self.device = device
        self.params = params
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.bev_h, self.bev_w = bev_shape
        self.spatial_shapes = torch.tensor([[self.bev_h, self.bev_w]])
        self.grid_dtype = grid_dtype
        self.grid_sample_compute_config = grid_sample_compute_config
        self.sampling_offsets_weight, self.sampling_offsets_bias = self._fold_grid_scale(params.sampling_offsets)

    def _fold_grid_scale(self, sampling_offsets):
        """Scale the offset Linear by ``2 / [bev_w, bev_h]``: dividing by the map size and the
        ``[0, 1] -> [-1, 1]`` rescale are one constant per channel, (x, y) alternating."""
        out_features = sampling_offsets.weight.shape[-1]
        assert out_features == self.num_bev_queue * self.num_heads * self.num_points * 2, (
            f"sampling_offsets width {out_features}: expected one BEV level, "
            f"{self.num_bev_queue} maps x {self.num_heads} heads x {self.num_points} points x (x, y)"
        )
        scale = torch.tensor([2.0 / self.bev_w, 2.0 / self.bev_h]).repeat(out_features // 2).reshape(1, out_features)
        scale = ttnn.from_torch(scale, device=self.device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

        # Kept in the grid's dtype: the Linear emits the grid, so a float32 grid needs float32 weights.
        def fold(tensor):
            folded = ttnn.mul(ttnn.typecast(tensor, ttnn.float32), scale)
            return folded if self.grid_dtype == ttnn.float32 else ttnn.typecast(folded, tensor.dtype)

        return fold(sampling_offsets.weight), fold(sampling_offsets.bias)

    def __call__(self, query, value, query_pos, grid_bias):
        """``query`` and ``query_pos`` ``(bs, num_query, C)``; ``value`` ``(bs * 2, num_query, C)``,
        the previous BEV and the encoder's input query stacked per sample, or None to stack the
        query with itself; ``grid_bias`` from :func:`tsa_grid_bias`. Returns ``(bs, num_query, C)``."""
        bs, num_query, embed_dims = query.shape
        queue = self.num_bev_queue
        identity = query
        if value is None:
            value = ttnn.reshape(
                ttnn.concat([ttnn.unsqueeze(query, 1), ttnn.unsqueeze(query, 1)], dim=1),
                (bs * queue, num_query, embed_dims),
            )
        # value[:bs], as upstream: the first bs stacked maps, which is each sample's previous BEV
        # only at bs=1 (BEVFormer's inference batch).
        query = ttnn.concat([value[:bs], ttnn.add(query, query_pos)], dim=-1)

        value = ttnn.linear(value, self.params.value_proj.weight, bias=self.params.value_proj.bias)
        value = ttnn.reshape(value, (bs * queue, num_query, self.num_heads, embed_dims // self.num_heads))

        # Queue-major channels: (bs, nq, queue * rest) -> (bs * queue, nq, rest).
        def split_queue(tensor):
            width = tensor.shape[-1] // queue
            tensor = ttnn.reshape(tensor, (bs, num_query, queue, width))
            return ttnn.reshape(ttnn.permute(tensor, (0, 2, 1, 3)), (bs * queue, num_query, width))

        offsets = ttnn.linear(
            query, self.sampling_offsets_weight, bias=self.sampling_offsets_bias, dtype=self.grid_dtype
        )
        grids = ttnn.add(ttnn.to_layout(split_queue(offsets), ttnn.ROW_MAJOR_LAYOUT), grid_bias)
        grids = ttnn.reshape(grids, (bs * queue, num_query, self.num_heads, 1, self.num_points, 2))

        weights = ttnn.linear(query, self.params.attention_weights.weight, bias=self.params.attention_weights.bias)
        weights = ttnn.softmax(ttnn.reshape(weights, (bs, num_query, queue * self.num_heads, self.num_points)), dim=-1)
        weights = ttnn.reshape(
            split_queue(ttnn.reshape(weights, (bs, num_query, -1))),
            (bs * queue, num_query, self.num_heads, 1, self.num_points),
        )

        output = multi_scale_deformable_attn_ttnn(
            value, self.spatial_shapes, grids, weights, self.device, self.grid_sample_compute_config
        )
        output = ttnn.mean(ttnn.reshape(output, (bs, queue, num_query, embed_dims)), dim=1)
        output = ttnn.linear(
            ttnn.to_layout(output, ttnn.TILE_LAYOUT), self.params.output_proj.weight, bias=self.params.output_proj.bias
        )
        return ttnn.add(output, identity)
