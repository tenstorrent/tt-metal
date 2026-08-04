# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.uniad.tt.ttnn_utils import multi_scale_deformable_attn_pytorch
from models.experimental.uniad.tt.ttnn_ffn import TtFFN


class TtMultiScaleDeformableAttention:
    def __init__(
        self,
        params,
        device,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        im2col_step=64,
        batch_first=False,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {embed_dims} and {num_heads}")
        dim_per_head = embed_dims // num_heads
        self.params = params
        self.device = device
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.fp16_enabled = False

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        # Layer-invariant tensors — same caching pattern as in TtTemporalSelfAttention
        # and TtMSDeformableAttention3D. spatial_shapes/reference_points are
        # constant across the encoder's 6 layers.
        self._offset_normalizer_xy_cache = None
        self._reference_xy_cache = None
        self._reference_xy_cache_ref = None
        # Cached Python ints for spatial shapes — `.item()` on a device
        # tensor is a host read and breaks trace capture; populate once
        # via `.item()` during warm-up and pass through to MSDA via
        # `value_spatial_shapes_list`.
        self._spatial_shapes_list_cache = None

    def __call__(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        params = self.params
        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = ttnn.permute(query, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        # The old per-layer `assert ttnn.sum(spatial_shapes[:,0]*spatial_shapes[:,1])
        # == num_value` forced a slice+mul+sum and a host sync on every call.
        # The equivalent check now runs once, off the hot/trace path, when
        # `_spatial_shapes_list_cache` is first populated (see below) — it
        # reuses the `.item()` reads that cache already pays for.
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = ttnn.linear(value, params.value_proj.weight, bias=params.value_proj.bias)
        if key_padding_mask is not None:
            mask = ttnn.unsqueeze(key_padding_mask, dim=-1)
            mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
            # Replace `where(mask, zeros, value)` with `value * (1 - mask)`:
            # same numerical effect (zero out where mask is 1, keep value
            # elsewhere) without the per-forward ttnn.zeros_like allocation
            # that blocks ttnn.trace_capture.
            value = ttnn.multiply(value, ttnn.rsub(mask, 1.0))
        value = ttnn.reshape(value, (bs, num_value, self.num_heads, -1))
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        sampling_offsets = ttnn.linear(query, params.sampling_offsets.weight, bias=params.sampling_offsets.bias)
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        )
        attention_weights = ttnn.linear(query, params.attention_weights.weight, bias=params.attention_weights.bias)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels * self.num_points)
        )

        attention_weights = ttnn.softmax(attention_weights, dim=-1)

        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels, self.num_points)
        )

        if reference_points.shape[-1] == 2:
            # `ttnn.div(sampling_offsets, [w, h])` on bf16 loses precision
            # against the host-fp32 reference. Precompute reciprocal
            # `[1/w, 1/h]` in fp32 and multiply.
            if self._offset_normalizer_xy_cache is None:
                _shapes_torch = ttnn.to_torch(spatial_shapes).to(torch.float32)
                _recip = torch.stack([1.0 / _shapes_torch[..., 1], 1.0 / _shapes_torch[..., 0]], dim=-1)
                _recip = _recip.reshape(1, 1, 1, _recip.shape[0], 1, 2)
                self._offset_normalizer_xy_cache = ttnn.from_torch(
                    _recip, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
                )
            offset_normalizer_xy_recip = self._offset_normalizer_xy_cache
            if self._reference_xy_cache_ref is not reference_points:
                self._reference_xy_cache = ttnn.reshape(
                    reference_points,
                    (reference_points.shape[0], reference_points.shape[1], 1, reference_points.shape[2], 1, 2),
                )
                self._reference_xy_cache_ref = reference_points
            reference_xy = self._reference_xy_cache
            sampling_offsets = ttnn.to_layout(sampling_offsets, ttnn.TILE_LAYOUT)
            sampling_offsets = ttnn.multiply(sampling_offsets, offset_normalizer_xy_recip)

            sampling_locations = ttnn.add(reference_xy, sampling_offsets)
        elif reference_points.shape[-1] == 4:
            reference_points = ttnn.unsqueeze(reference_points, dim=2)
            reference_points = ttnn.unsqueeze(reference_points, dim=2)
            reference_points_reshape = reference_points[:, :, :, :, :, :2]
            sampling_locations = (
                reference_points_reshape + ttnn.div(sampling_offsets, self.num_points) * reference_points_reshape * 0.5
            )

        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, " f"but got {reference_points.shape[-1]} instead."
            )
        if self._spatial_shapes_list_cache is None:
            num_levels = spatial_shapes.shape[0]
            self._spatial_shapes_list_cache = [
                (int(spatial_shapes[lvl][0].item()), int(spatial_shapes[lvl][1].item())) for lvl in range(num_levels)
            ]
            # One-time replacement for the removed per-layer assert: the
            # spatial shapes must account for exactly `num_value` tokens.
            # Runs only on the first (cache-populating) call, so it never
            # touches the trace-replay hot path.
            total_tokens = sum(h * w for h, w in self._spatial_shapes_list_cache)
            assert total_tokens == num_value, (
                f"spatial_shapes describe {total_tokens} value tokens but value "
                f"has {num_value} (shapes={self._spatial_shapes_list_cache})"
            )
        output = multi_scale_deformable_attn_pytorch(
            value,
            spatial_shapes,
            None,
            sampling_locations,
            attention_weights,
            None,
            self.device,
            value_spatial_shapes_list=self._spatial_shapes_list_cache,
        )

        output = ttnn.linear(output, params.output_proj.weight, bias=params.output_proj.bias)
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        output = ttnn.add(output, identity)

        return output


class TtDetrTransformerEncoder:
    def __init__(
        self,
        params,
        device,
        embed_dims=256,
        num_layers=6,
        num_heads=8,
        num_levels=4,
        num_points=4,
        im2col_step=64,
        batch_first=False,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        self.device = device
        self.params = params
        self.num_layers = num_layers
        self.layers = []
        for i in range(num_layers):
            attn = TtMultiScaleDeformableAttention(
                params=params.layers[i].attentions[0],
                device=device,
            )
            ffn = TtFFN(
                params=params.layers[i].ffns[0].ffn.ffn0,
                device=device,
            )
            self.layers.append([attn, ffn])
        # Trace-replay state — same pattern as TtBEVFormerEncoder.
        self._trace_cache = {}
        self._trace_seen_shapes = set()

    def __call__(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        query_key_padding_mask=None,
        **kwargs,
    ):
        # Only the inputs that flow into the recorded device ops need to be
        # tracked for trace replay; key/value/identity/key_padding_mask/
        # level_start_index and **kwargs are not read by the layer loop.
        args_dict = dict(
            query=query,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            query_key_padding_mask=query_key_padding_mask,
        )
        return self._run_layers_traced(args_dict)

    def _run_layers(self, *, query, query_pos, reference_points, spatial_shapes, query_key_padding_mask):
        for i, layer in enumerate(self.layers):
            temp_key = temp_value = query
            query = layer[0](
                query,
                temp_key,
                temp_value,
                identity=None,
                query_pos=query_pos,
                key_pos=query_pos,
                attn_mask=None,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points,
                key_padding_mask=query_key_padding_mask,
            )

            query = ttnn.layer_norm(
                query,
                weight=self.params.layers[i].norms[0].weight,
                bias=self.params.layers[i].norms[0].bias,
            )
            query = layer[1](query)
            query = ttnn.layer_norm(
                query,
                weight=self.params.layers[i].norms[1].weight,
                bias=self.params.layers[i].norms[1].bias,
            )

        return query

    @staticmethod
    def _trace_shape_key(args_dict):
        def _t(x):
            return tuple(int(d) for d in x.shape) if isinstance(x, ttnn.Tensor) else None

        return tuple(_t(args_dict[k]) for k in sorted(args_dict.keys()))

    def _clone_static(self, args_dict):
        static = {}
        for k, v in args_dict.items():
            if isinstance(v, ttnn.Tensor):
                static[k] = ttnn.clone(v, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                static[k] = v
        return static

    @staticmethod
    def _refresh_static(args_dict, static):
        for k, v in args_dict.items():
            if isinstance(v, ttnn.Tensor):
                ttnn.copy(v, static[k])

    def _run_layers_traced(self, args_dict):
        shape_key = self._trace_shape_key(args_dict)
        cached = self._trace_cache.get(shape_key)
        if cached is not None:
            self._refresh_static(args_dict, cached["static_inputs"])
            ttnn.execute_trace(self.device, cached["trace_id"], cq_id=0, blocking=False)
            return cached["static_output"]

        if shape_key not in self._trace_seen_shapes:
            self._trace_seen_shapes.add(shape_key)
            return self._run_layers(**args_dict)

        static_inputs = self._clone_static(args_dict)
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        static_output = self._run_layers(**static_inputs)
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        ttnn.execute_trace(self.device, trace_id, cq_id=0, blocking=False)
        self._trace_cache[shape_key] = {
            "trace_id": trace_id,
            "static_inputs": static_inputs,
            "static_output": static_output,
        }
        return static_output
