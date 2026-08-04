# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.experimental.uniad.tt.ttnn_utils import multi_scale_deformable_attn_pytorch
from models.experimental.uniad.tt.ttnn_enc_timing import record as _enc_record, sync_now as _enc_sync_now


class TtSpatialCrossAttention:
    def __init__(
        self,
        device,
        params,
        embed_dims=256,
        num_cams=6,
        pc_range=None,
        init_cfg=None,
        batch_first=False,
        deformable_attention=dict(type="MSDeformableAttention3D", embed_dims=256, num_levels=4),
        **kwargs,
    ):
        super(TtSpatialCrossAttention, self).__init__()

        self.device = device
        self.params = params
        self.init_cfg = init_cfg
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = TtMSDeformableAttention3D(device=self.device, params=params, num_levels=4)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.batch_first = batch_first

    def __call__(
        self,
        query,
        key,
        value,
        residual=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        reference_points_cam=None,
        bev_mask=None,
        level_start_index=None,
        flag="encoder",
        **kwargs,
    ):
        # Pop encoder-side per-op timing dict so it isn't forwarded into
        # ttnn ops as a kwarg. SCA-internal sub-phase timing was used during
        # one-time profiling; the dict is still passed through into MSDA
        # below so the *MSDA* sub-phase breakdown remains available when
        # TT_UNIAD_TIMING=1.
        _enc_stats = kwargs.pop("_enc_stats", None)
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query  # ttnn.from_torch(query, dtype = ttnn.bfloat16, layout = ttnn.ROW_MAJOR_LAYOUT, device = self.device)
            # slots is allocated lazily below (either device-side via scatter_add
            # path or host-side torch.zeros in the fallback path).
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape

        D = reference_points_cam.size(3)
        # The encoder pre-computes per-camera valid-query indexes once and
        # threads them through kwargs to avoid recomputing on every layer.
        # Fall back to computing them inline for callers that don't provide.
        precomputed = kwargs.get("_precomputed_indexes", None)
        if precomputed is not None:
            indexes = precomputed
        else:
            indexes = []
            for i, mask_per_img in enumerate(bev_mask):
                index_query_per_img = ttnn.sum(mask_per_img[0], -1)
                index_query_per_img = ttnn.to_torch(index_query_per_img)
                index_query_per_img = index_query_per_img.nonzero().squeeze(-1).to(torch.long)
                indexes.append(index_query_per_img)

        # `count` is bev_mask-derived only, so it's identical across all
        # encoder layers. The encoder caches it on first call and threads
        # via kwargs to avoid 5 redundant computations per inference.
        precomputed_count = kwargs.get("_precomputed_count", None)
        # Device-side rebatching path (encoder threads gather indexes +
        # prebuilt reference_points_rebatch via kwargs). Replaces the host
        # to_torch + nested Python loop + from_torch round-trip on every
        # layer with a single ttnn.gather + reshape. The gather index is
        # (bs, num_cams * max_len, embed_dims), built once per forward.
        precomputed_gather_index_concat = kwargs.get("_precomputed_gather_index_concat", None)
        precomputed_ref_rebatch = kwargs.get("_precomputed_ref_rebatch", None)
        precomputed_max_len = kwargs.get("_precomputed_max_len", None)
        if (
            precomputed_gather_index_concat is not None
            and precomputed_ref_rebatch is not None
            and precomputed_max_len is not None
        ):
            max_len = precomputed_max_len
            # One gather call rather than six per-camera. Padded positions
            # (> len_i within each camera band) reuse idx[0] and their
            # outputs are silently dropped by scatter_add below (which only
            # writes the first len_i positions per camera).
            gathered = ttnn.gather(query, dim=1, index=precomputed_gather_index_concat)
            queries_rebatch = ttnn.reshape(gathered, (bs, self.num_cams, max_len, self.embed_dims))
            reference_points_rebatch = precomputed_ref_rebatch
        else:
            # Fallback host-loop rebatching path. The full UniAD warm path
            # always provides `_precomputed_*` from the encoder precompute
            # block (built once per forward, reused across all 6 SCA
            # layers), so this branch should only fire from standalone
            # PCC tests that call SCA without the encoder wrapper. Under
            # `ttnn.trace_capture`, this branch would silently break the
            # trace via `ttnn.to_torch` and a Python-driven shape — guard
            # with a warning so any accidental fall-through is loud.
            import warnings as _warnings

            _warnings.warn(
                "TtSpatialCrossAttention: rebatch fallback path taken — "
                "`_precomputed_gather_index_concat / _precomputed_ref_rebatch / "
                "_precomputed_max_len` were not provided. This path uses host "
                "loops and is incompatible with ttnn.trace_capture.",
                stacklevel=2,
            )
            max_len = max([each.shape[0] for each in indexes])
            query = ttnn.to_torch(query)
            queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
            reference_points_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, D, 2])
            for j in range(bs):
                for i, reference_points_per_img in enumerate(reference_points_cam):
                    index_query_per_img = indexes[i]  # already torch.long
                    queries_rebatch[j, i, : len(index_query_per_img)] = query[j, index_query_per_img]
                    reference_points_rebatch[j, i, : len(index_query_per_img)] = reference_points_per_img[
                        j, index_query_per_img
                    ]
            queries_rebatch = ttnn.from_torch(queries_rebatch, dtype=ttnn.bfloat16, device=self.device)
            reference_points_rebatch = ttnn.from_torch(
                reference_points_rebatch, dtype=ttnn.bfloat16, device=self.device
            )
        num_cams, l, bs, embed_dims = key.shape
        num_cams, l, bs, embed_dims = key.shape

        key = ttnn.permute(key, (2, 0, 1, 3))
        key = ttnn.reshape(key, (bs * self.num_cams, l, self.embed_dims))

        value = ttnn.permute(value, (2, 0, 1, 3))
        value = ttnn.reshape(value, (bs * self.num_cams, l, self.embed_dims))
        queries = self.deformable_attention(
            query=ttnn.reshape(queries_rebatch, (bs * self.num_cams, max_len, self.embed_dims)),
            key=key,
            value=value,
            reference_points=ttnn.reshape(reference_points_rebatch, (bs * self.num_cams, max_len, D, 2)),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            _enc_stats=_enc_stats,
        )
        queries = ttnn.reshape(queries, (bs, self.num_cams, max_len, self.embed_dims))

        # Device-side scatter-accumulate: replaces the ttnn.to_torch(queries)
        # + Python loop + ttnn.from_torch(slots) round-trip with per-camera
        # ttnn.scatter_add. The per-camera expanded index tensors are
        # precomputed once in the encoder (one-time host upload).
        precomputed_scatter_indexes = kwargs.get("_precomputed_scatter_indexes", None)
        if precomputed_scatter_indexes is not None:
            # `ttnn.zeros(shape, ...)` uploads the fill constant via
            # host->device write, which is rejected inside
            # `ttnn.begin_trace_capture`. `x - x` produces an equivalent
            # zero tensor (same shape/dtype/layout) without a host transfer.
            slots = ttnn.subtract(inp_residual, inp_residual)
            for i, scatter_idx in enumerate(precomputed_scatter_indexes):
                len_i = scatter_idx.shape[1]
                src = queries[:, i, :len_i, :]
                slots = ttnn.scatter_add(slots, 1, scatter_idx, src)
        else:
            # Fallback host path (for standalone PCC tests without the encoder's
            # precompute step).
            queries = ttnn.to_torch(queries)
            slots_host = torch.zeros(list(inp_residual.shape), dtype=torch.float32)
            for j in range(bs):
                for i, index_query_per_img in enumerate(indexes):
                    slots_host[j, index_query_per_img] += queries[j, i, : len(index_query_per_img)]
            slots = ttnn.from_torch(slots_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        if precomputed_count is not None:
            count = precomputed_count
        else:
            count = ttnn.sum(bev_mask, -1) > 0
            count = ttnn.permute(count, (1, 2, 0))
            count = ttnn.sum(count, -1)
            count = ttnn.clamp(count, min=1.0)
            count = ttnn.unsqueeze(count, -1)
        slots = ttnn.div(slots, count)
        slots = ttnn.linear(slots, self.params.output_proj.weight, bias=self.params.output_proj.bias)

        return slots + inp_residual


class TtMSDeformableAttention3D:
    def __init__(
        self,
        device,
        params,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=8,
        im2col_step=64,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {embed_dims} and {num_heads}")
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False
        self.device = device
        self.params = params
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        # Cached Python ints for spatial shapes — see TtCustomMSDeformableAttention.
        self._spatial_shapes_list_cache = None
        # offset_normalizer_xy depends on spatial_shapes only (constant per
        # encoder forward); caching avoids rebuilding stack+reshape on every
        # layer. reference_xy depends on reference_points which the encoder
        # rebuilds per forward but reuses across all 6 layers — cache it
        # keyed by Python object identity.
        self._offset_normalizer_xy_cache = None
        self._reference_xy_cache = None
        self._reference_xy_cache_ref = None

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
        _enc_stats = kwargs.pop("_enc_stats", None)
        _t = _enc_sync_now(self.device) if _enc_stats is not None else 0.0
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = ttnn.permute(query, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        # `assert (ttnn.sum(...) == num_value)` removed: comparing a ttnn tensor
        # with a Python int forces a device->host read, which breaks trace.
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = ttnn.linear(value, params.value_proj.weight, bias=params.value_proj.bias)
        if key_padding_mask is not None:
            mask = key_padding_mask[..., None]
            value = ttnn.where(mask, ttnn.zeros_like(value), value)
        value = ttnn.reshape(value, (bs, num_value, self.num_heads, -1))
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        if _enc_stats is not None:
            _enc_record(_enc_stats, "sca3d_value_proj", _t, self.device)
            _t = _enc_sync_now(self.device)

        sampling_offsets = ttnn.linear(query, params.sampling_offsets.weight, bias=params.sampling_offsets.bias)
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        )
        if _enc_stats is not None:
            _enc_record(_enc_stats, "sca3d_offsets_lin", _t, self.device)
            _t = _enc_sync_now(self.device)

        attention_weights = ttnn.linear(query, params.attention_weights.weight, bias=params.attention_weights.bias)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels * self.num_points)
        )
        attention_weights = ttnn.softmax(attention_weights, -1)

        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels, self.num_points)
        )
        if _enc_stats is not None:
            _enc_record(_enc_stats, "sca3d_attn_lin", _t, self.device)
            _t = _enc_sync_now(self.device)

        if reference_points.shape[-1] == 2:
            # `ttnn.divide(sampling_offsets, [w, h])` on bf16 loses precision
            # against the host-fp32 reference (per-module BEV PCC 0.99→0.68).
            # Precompute reciprocal `[1/w, 1/h]` in fp32 and multiply.
            if self._offset_normalizer_xy_cache is None:
                _shapes_torch = ttnn.to_torch(spatial_shapes).to(torch.float32)
                _recip = torch.stack([1.0 / _shapes_torch[..., 1], 1.0 / _shapes_torch[..., 0]], dim=-1)
                _recip = _recip.reshape(1, 1, 1, _recip.shape[0], 1, _recip.shape[1])
                self._offset_normalizer_xy_cache = ttnn.from_torch(
                    _recip, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
                )
            offset_normalizer_xy_recip = self._offset_normalizer_xy_cache
            bs_r, num_query, num_Z_anchors, _ = reference_points.shape
            if self._reference_xy_cache_ref is not reference_points:
                self._reference_xy_cache = ttnn.reshape(
                    reference_points,
                    (bs_r, num_query, 1, 1, 1, reference_points.shape[-2], reference_points.shape[-1]),
                )
                self._reference_xy_cache_ref = reference_points
            reference_xy = self._reference_xy_cache

            sampling_locations = ttnn.multiply(sampling_offsets, offset_normalizer_xy_recip)
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_locations.shape
            sampling_locations = ttnn.reshape(
                sampling_locations,
                (bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy),
            )
            sampling_locations = ttnn.add(reference_xy, sampling_locations)
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors
            sampling_locations = ttnn.reshape(
                sampling_locations, (bs, num_query, num_heads, num_levels, num_all_points, xy)
            )

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f"Last dim of reference_points must be" f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        # Cache spatial_shapes as Python ints once (host read happens during
        # warm-up, not under trace capture).
        if self._spatial_shapes_list_cache is None:
            num_levels = spatial_shapes.shape[0]
            self._spatial_shapes_list_cache = [
                (int(spatial_shapes[lvl][0].item()), int(spatial_shapes[lvl][1].item())) for lvl in range(num_levels)
            ]
        if _enc_stats is not None:
            _enc_record(_enc_stats, "sca3d_locations", _t, self.device)
            _t = _enc_sync_now(self.device)

        output = multi_scale_deformable_attn_pytorch(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
            self.device,
            value_spatial_shapes_list=self._spatial_shapes_list_cache,
            _enc_stats=_enc_stats,
        )
        if _enc_stats is not None:
            _enc_record(_enc_stats, "sca3d_msda", _t, self.device)

        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        return output
