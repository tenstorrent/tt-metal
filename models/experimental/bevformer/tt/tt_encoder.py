# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN BEVFormer Encoder implementation.

This module implements the main BEVFormer encoder using TTNN operations that
combines spatial cross-attention and temporal self-attention to learn Bird's-Eye-View
representations from multi-camera images. The encoder processes BEV queries through
multiple transformer layers that enable both spatial feature extraction from camera
views and temporal modeling.
"""

import ttnn
import torch
from typing import Optional, List, Dict, Any, NamedTuple

from .tt_spatial_cross_attention import SCARebatchPlan, SCARebatchPlanner, SCAVisibility, TTSpatialCrossAttention
from .tt_common import grid_linear_kwargs, l1_if_fits, layer_norm, signpost
from .tt_temporal_self_attention import TTTemporalSelfAttention, shifted_reference_points
from .tt_point_sampling_3d_2d import camera_rows, image_size, point_sampling_3d_to_2d_ttnn, world_points_ttnn
from ..reference.point_sampling_3d_2d import generate_reference_points

from loguru import logger

# Enable/disable logging output
ENABLE_LOGGING = False


class CameraProjection(NamedTuple):
    """One frame's reference grid projected to the cameras, with the visibility the spatial
    cross-attention plan is built from.

    Attributes:
        reference_points_cam: Bfloat16 ``[num_cams, B, num_queries, D, 2]``.
        bev_mask: Bfloat16 ``[num_cams, B, num_queries, D]``.
        visibility: The encoder planner's :meth:`SCARebatchPlanner.count_visible` of ``bev_mask``,
            with its counts' readback queued once :meth:`SCARebatchPlanner.read_counts` has run.
        rebatch_plan: The plan built from ``visibility``; None until a caller builds it, and the
            encoder then builds it itself.
    """

    reference_points_cam: ttnn.Tensor
    bev_mask: ttnn.Tensor
    visibility: SCAVisibility
    rebatch_plan: Optional[SCARebatchPlan] = None


class TTBEVFormerLayer:
    """
    TTNN implementation of a single BEVFormer transformer layer.

    This layer combines spatial cross-attention and temporal self-attention
    to process BEV queries using TTNN operations. Each layer performs:
    1. Temporal self-attention (optional) - models temporal dependencies
    2. Spatial cross-attention - extracts features from camera views
    3. Feed-forward network - additional processing
    4. Layer normalization after each component

    Args:
        device: TTNN device for computation
        params: Parameter dict containing weights and biases
        embed_dims (int): Feature embedding dimensions
        num_heads (int): Number of attention heads
        num_levels (int): Number of feature pyramid levels
        num_points (int): Number of sampling points in deformable attention
        num_cams (int): Number of cameras
        use_temporal_self_attention (bool): Whether to use temporal self-attention
        use_spatial_cross_attention (bool): Whether to use spatial cross-attention
        feedforward_channels (int): FFN intermediate channel size
        batch_first (bool): Whether batch dimension is first
        spatial_shapes: Multi-scale feature shapes [num_levels, 2]
        bev_shape: BEV grid shape [1, 2] containing [bev_h, bev_w]
        **kwargs: Additional arguments

    L1 placement. Both attentions' fused ops stage their value and output in L1 only when the
    op's circular buffers still fit below the lowest live L1 buffer (L1 is allocated top-down;
    ``ttnn.experimental.can_use_fused_msda_from_offsets_l1_value``), and fall back to DRAM, much
    slower, with only a warning otherwise. So each attention stages its value first, into an
    empty L1, keeps its other intermediates in L1 only until the fused op runs, and frees them;
    after its fused op, the cross-attention allocates L1 only once the op's output is freed. The
    layer's input and norm1's output, which live across a fused op, stay in DRAM; norm2's output
    and the FFN, between the cross-attention and norm3, run in L1. Every group of intermediates
    goes to DRAM instead when it does not fit L1 (``tt_common.l1_if_fits``).
    """

    def __init__(
        self,
        device,
        params,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        num_cams: int = 6,
        use_temporal_self_attention: bool = True,
        use_spatial_cross_attention: bool = True,
        feedforward_channels: int = 1024,
        batch_first: bool = True,
        *,
        spatial_shapes,
        bev_shape,
        **kwargs,
    ):
        self.device = device
        self.params = params
        self.embed_dims = embed_dims
        self.use_temporal_self_attention = use_temporal_self_attention
        self.use_spatial_cross_attention = use_spatial_cross_attention
        self.batch_first = batch_first
        self.feedforward_channels = feedforward_channels
        self.linear_kwargs = grid_linear_kwargs(device)

        # Temporal Self-Attention
        if use_temporal_self_attention and hasattr(params, "temporal_self_attention"):
            self.temporal_self_attention = TTTemporalSelfAttention(
                device=device,
                params=params.temporal_self_attention,
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=1,  # Single level for temporal attention
                num_points=num_points,
                batch_first=batch_first,
                spatial_shapes=bev_shape,
                residual=False,
                **kwargs,
            )

        # Spatial Cross-Attention
        if use_spatial_cross_attention and hasattr(params, "spatial_cross_attention"):
            deform_config = {
                "embed_dims": embed_dims,
                "num_heads": num_heads,
                "num_levels": num_levels,
                "num_points": num_points,
            }
            self.spatial_cross_attention = TTSpatialCrossAttention(
                device=device,
                params=params.spatial_cross_attention,
                embed_dims=embed_dims,
                num_cams=num_cams,
                batch_first=batch_first,
                deformable_attention=deform_config,
                spatial_shapes=spatial_shapes,
                residual=False,
                **kwargs,
            )

    def forward(
        self,
        bev_query,
        value=None,
        bev_pos=None,
        prev_bev=None,
        shift=None,
        reference_points_cam=None,
        bev_mask=None,
        rebatch_plan=None,
        bev_reference_points=None,
        **kwargs,
    ):
        """
        Forward pass for a single TTBEVFormerLayer.

        Args:
            bev_query: Current BEV query features [B, num_queries, embed_dims]
            value: Batch-first multi-camera features [B, num_cams, H*W, embed_dims], shared by
                every layer.
            bev_pos: BEV positional encoding [B, num_queries, embed_dims]
            prev_bev: The previous BEV, rotated to this frame, stacked per sample with the encoder's
                input queries, [B*2, num_queries, embed_dims], as upstream's encoder builds it; None
                on the first frame.
            shift: Ego translation in BEV fractions, float32 ROW_MAJOR [B, 1, 1, 2]; applied only
                with ``prev_bev``.
            reference_points_cam: Camera reference points [num_cams, B, num_queries, D, 2]
            bev_mask: Validity mask for camera projections [num_cams, B, num_queries, D]
            rebatch_plan: Shared SCA rebatch plan for this frame
            bev_reference_points: 2D BEV reference points on device, float32 ROW_MAJOR
                [B, num_queries, 1, 2], or with a previous BEV every BEV map's
                [B, num_queries, 2, 2] from ``shifted_reference_points``, with ``shift`` None.
                The encoder owns the grid, widens its leading dimension to the runtime batch
                and applies the shift once before the layer loop.

        Returns:
            Updated BEV features [B, num_queries, embed_dims]
        """
        signpost(header="TTNN BEVFormerLayer Forward Start")

        signpost(header="BEVLayer TSA Start")
        # Combine current BEV with history for temporal modeling
        temp_query = self.temporal_self_attention(
            query=bev_query,
            value=prev_bev,  # Use previous BEV as value for temporal context
            query_pos=bev_pos,
            reference_points=bev_reference_points,
            shift=shift,
            **kwargs,
        )

        signpost(header="BEVLayer TSA End")

        # Each attention's and the FFN's residual is added inside the LayerNorm that follows it.
        # norm1 writes DRAM and frees the self-attention's L1 output: the cross-attention's fused
        # op needs that L1.
        bev_query = layer_norm(temp_query, self.params.norm1, residual=bev_query, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(temp_query)

        signpost(header="BEVLayer SCA Start")

        # Spatial Cross-Attention
        spatial_query = self.spatial_cross_attention(
            query=bev_query,
            value=value,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            rebatch_plan=rebatch_plan,
            **kwargs,
        )

        signpost(header="BEVLayer SCA End")

        # The FFN's input, intermediate and output live in L1 when they fit: no fused op holds L1
        # between norm2 and norm3, which writes the layer's output back to DRAM.
        bs, num_queries, embed_dims = bev_query.shape
        ffn_memory = l1_if_fits(
            self.device, bs * num_queries * (2 * embed_dims + self.feedforward_channels) * bev_query.element_size()
        )
        bev_query = layer_norm(spatial_query, self.params.norm2, residual=bev_query, memory_config=ffn_memory)
        ttnn.deallocate(spatial_query)

        signpost(header="BEVLayer FFN Start")

        # Feed-Forward Network
        ffn_output = self._forward_ffn(bev_query, ffn_memory)

        signpost(header="BEVLayer FFN End")

        bev_query = layer_norm(ffn_output, self.params.norm3, residual=bev_query, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        signpost(header="TTNN BEVFormerLayer Forward End")

        return bev_query

    def _forward_ffn(self, x, memory_config):
        """
        Forward pass through the feed-forward network using TTNN operations.

        Args:
            x: Input tensor [B, num_queries, embed_dims]
            memory_config: Where the intermediate and the output go.

        Returns:
            Output tensor [B, num_queries, embed_dims]
        """
        # Check if FFN parameters exist - mixed object/dict structure
        if not (
            hasattr(self.params, "ffn") and hasattr(self.params.ffn, "linear1") and hasattr(self.params.ffn, "linear2")
        ):
            if ENABLE_LOGGING:
                logger.warning("FFN parameters not found, skipping FFN forward pass")
            return ttnn.zeros_like(x)

        # First linear layer and its ReLU: [embed_dims] -> [feedforward_channels]
        # FFN has mixed structure: params.ffn.linear1 is a dict, not an object
        x = ttnn.linear(
            x,
            self.params.ffn.linear1["weight"],
            bias=self.params.ffn.linear1["bias"],
            activation="relu",
            memory_config=memory_config,
            **self.linear_kwargs,
        )

        # Second linear layer: [feedforward_channels] -> [embed_dims]
        x = ttnn.linear(
            x,
            self.params.ffn.linear2["weight"],
            bias=self.params.ffn.linear2["bias"],
            memory_config=memory_config,
            **self.linear_kwargs,
        )

        return x

    def __call__(self, *args, **kwargs):
        """Make the class callable"""
        return self.forward(*args, **kwargs)

    def extra_repr(self) -> str:
        """String representation for debugging"""
        return (
            f"embed_dims={self.embed_dims}, "
            f"use_temporal_self_attention={self.use_temporal_self_attention}, "
            f"use_spatial_cross_attention={self.use_spatial_cross_attention}, "
            f"feedforward_channels={self.feedforward_channels}, "
            f"batch_first={self.batch_first}"
        )


class TTBEVFormerEncoder:
    """
    TTNN implementation of BEVFormer Encoder.

    The encoder transforms multi-camera features into unified BEV representations
    using spatiotemporal transformers implemented with TTNN operations. It consists
    of multiple TTBEVFormerLayer modules that perform spatial cross-attention and
    temporal self-attention.

    Args:
        device: TTNN device for computation
        params: Parameter dict containing all layer weights and biases
        num_layers (int): Number of transformer layers
        embed_dims (int): Feature embedding dimensions
        num_heads (int): Number of attention heads
        num_levels (int): Number of feature pyramid levels
        num_points (int): Number of sampling points in deformable attention
        num_cams (int): Number of cameras
        pc_range (List[float]): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        num_points_in_pillar (int): Number of points sampled in each pillar
        return_intermediate (bool): Whether to return intermediate layer outputs
        dataset (str): Dataset type for specific configurations
        feedforward_channels (int): FFN intermediate channel size
        batch_first (bool): Whether batch dimension is first
        z_cfg (Dict[str, Any]): Z-axis configuration for point sampling
        bev_h (int): BEV grid height. The reference-point grid and bev_shape are built
            from it at construction, so the grid cannot change between forwards; a
            different grid needs a new encoder instance.
        bev_w (int): BEV grid width. See bev_h.
        spatial_shapes: Multi-scale feature shapes [num_levels, 2]. Fixed for the lifetime
            of the encoder and its attention modules, which fold it into their sampling-offset
            Linear at construction. Feeding features at a different resolution requires a
            new encoder instance; it cannot be changed between forwards.
        **kwargs: Additional arguments

    Batch size is not a constructor argument: it is read from bev_query on every forward,
    so one instance serves a varying batch size without reconstruction.
    """

    def __init__(
        self,
        device,
        params,
        num_layers: int = 6,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        num_cams: int = 6,
        pc_range: List[float] = None,
        num_points_in_pillar: int = 4,
        return_intermediate: bool = False,
        dataset: str = "nuScenes",
        feedforward_channels: int = 1024,
        batch_first: bool = True,
        z_cfg: Dict[str, Any] = None,
        *,
        bev_h: int,
        bev_w: int,
        spatial_shapes,
        **kwargs,
    ):
        self.device = device
        self.params = params

        if pc_range is None:
            pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

        if z_cfg is None:
            z_cfg = {
                "num_points": num_points_in_pillar,
                "start": pc_range[2],  # z_min
                "end": pc_range[5],  # z_max
            }

        if spatial_shapes is None:
            raise ValueError("spatial_shapes is required")
        if not isinstance(spatial_shapes, torch.Tensor):
            spatial_shapes = torch.as_tensor(spatial_shapes)
        if spatial_shapes.ndim != 2 or spatial_shapes.shape != (num_levels, 2):
            raise ValueError(f"spatial_shapes must have shape [{num_levels}, 2], got {tuple(spatial_shapes.shape)}")
        if spatial_shapes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError(f"spatial_shapes must contain integers, got {spatial_shapes.dtype}")
        if torch.any(spatial_shapes <= 0):
            raise ValueError(f"spatial_shapes dimensions must be positive, got {spatial_shapes.tolist()}")
        spatial_shapes = spatial_shapes.to(dtype=torch.long).clone()

        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_cams = num_cams
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.return_intermediate = return_intermediate
        self.batch_first = batch_first
        self.z_cfg = z_cfg
        self.feedforward_channels = feedforward_channels
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.spatial_shapes = spatial_shapes
        self.bev_shape = torch.tensor([[bev_h, bev_w]], dtype=torch.long)

        # The grid is fixed by bev_h, bev_w, and z_cfg; changing any of them requires
        # a new encoder instance. Batch size only broadcasts the leading dimension.
        self._reference_points_3d = generate_reference_points(
            bev_h=bev_h,
            bev_w=bev_w,
            z_cfg=z_cfg,
            batch_size=1,
            dtype=torch.float32,
        )
        # float32: in bfloat16 a 100-cell grid's centres round by up to 0.4 cells. ROW_MAJOR, where
        # the trailing (1, 2) is not padded to a tile.
        self._bev_reference_points = ttnn.from_torch(
            self._reference_points_3d[:, :, 0, :2].unsqueeze(2),
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        # The grid in world coordinates, which every frame projects to the cameras.
        self.world_points = world_points_ttnn(self._reference_points_3d, pc_range, device)
        # One per encoder, as its capacity is the camera rig's high-water mark.
        self.rebatch_planner = SCARebatchPlanner(device)

        # Build transformer layers
        self.layers = []
        for layer_idx in range(num_layers):
            layer_params = getattr(params, f"layer_{layer_idx}", None)
            if layer_params is None:
                if ENABLE_LOGGING:
                    logger.warning(f"Parameters for layer {layer_idx} not found")
                continue

            layer = TTBEVFormerLayer(
                device=device,
                params=layer_params,
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                num_cams=num_cams,
                feedforward_channels=feedforward_channels,
                batch_first=batch_first,
                spatial_shapes=spatial_shapes,
                bev_shape=self.bev_shape,
                **kwargs,
            )
            self.layers.append(layer)

    def project_cameras(self, img_metas, projection=None) -> CameraProjection:
        """Project the reference grid to the frame's cameras and queue their visible counts' readback.

        Nothing here waits for the device. Called ahead of the backbone, the counts are back long
        before the encoder needs them to size its rebatch plan.

        Args:
            img_metas: Camera metadata with each sample's ``lidar2img`` and the images' size.
            projection: The cameras' projection rows (``tt_point_sampling_3d_2d.camera_rows``) on
                device, float32 TILE; None builds them from ``img_metas`` and uploads them.
        """
        if projection is None:
            projection = ttnn.from_torch(
                self.projection_rows(img_metas), device=self.device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
            )
        cameras = self.project_cameras_on_device(projection)
        return cameras._replace(visibility=self.rebatch_planner.read_counts(cameras.visibility))

    def project_cameras_on_device(self, projection) -> CameraProjection:
        """:meth:`project_cameras` without the readback: device work only, as a trace records it."""
        reference_points_cam, bev_mask = point_sampling_3d_to_2d_ttnn(
            reference_points=None,
            pc_range=self.pc_range,
            lidar2img=None,
            device=self.device,
            # The grid broadcasts over the batch.
            world_points=self.world_points,
            projection=projection,
            # Projected in float32; bfloat16 is the only dtype the rebatch plan's gather takes.
            output_dtype=ttnn.bfloat16,
        )
        return CameraProjection(reference_points_cam, bev_mask, self.rebatch_planner.count_visible(bev_mask))

    def projection_rows(self, img_metas) -> torch.Tensor:
        """Host projection rows (``tt_point_sampling_3d_2d.camera_rows``) of the frame's cameras over
        this encoder's reference grid."""
        grid = self.world_points
        return camera_rows(self._host_lidar2img(img_metas), image_size(img_metas), grid.num_queries, grid.num_points)

    @staticmethod
    def _host_lidar2img(img_metas):
        """Each sample's ``lidar2img``, the transform from lidar to image pixels, as torch
        [B, num_cams, 4, 4] float32."""
        return torch.stack([torch.as_tensor(meta["lidar2img"], dtype=torch.float32) for meta in img_metas])

    def forward(
        self,
        bev_query,
        key=None,
        value=None,
        bev_pos=None,
        prev_bev=None,
        shift=None,
        img_metas: Optional[List[Dict[str, Any]]] = None,
        cameras: Optional[CameraProjection] = None,
        **kwargs,
    ):
        """
        Forward pass of TTNN BEVFormer encoder.

        Args:
            bev_query: Initial BEV query features [B, num_queries, embed_dims]
            key: Batch-first multi-camera features [B, num_cams, H*W, embed_dims], as the spatial
                cross-attention reads them
            value: Same as key (optional); falls back to ``key`` when omitted
            bev_pos: BEV positional encoding [B, num_queries, embed_dims]
            prev_bev: The previous BEV, rotated to this frame, stacked per sample with the encoder's
                input queries, [B*2, num_queries, embed_dims], as upstream's encoder builds it; None
                on the first frame.
            shift: Ego translation in BEV fractions, float32 ROW_MAJOR [B, 1, 1, 2]; applied only
                with ``prev_bev``.
            img_metas: Camera metadata, projected here with :meth:`project_cameras`; not with ``cameras``.
            cameras: The frame's :meth:`project_cameras`, made ahead of the forward. Its plan is built
                here, waiting for the counts, unless it carries one.

        Returns:
            Final BEV features [B, num_queries, embed_dims]
            If return_intermediate=True, returns list of intermediate features.
        """
        signpost(header="TTNN BEVFormerEncoder Forward Start")

        output = bev_query
        intermediate = []

        # Validate runtime tensor dimensions against the configured model geometry.
        bs, num_queries, _ = bev_query.shape
        assert (
            num_queries == self.bev_h * self.bev_w
        ), f"num_queries {num_queries} != bev_h*bev_w {self.bev_h * self.bev_w}"

        # Batch is a pure broadcast of the stored grid, widened on device and freed once used.
        bev_reference_points = (
            self._bev_reference_points
            if bs == 1
            else ttnn.repeat(self._bev_reference_points, ttnn.Shape((bs, 1, 1, 1)))
        )
        # The previous BEV's shifted reference points are the same for every layer.
        layer_reference_points, layer_shift = bev_reference_points, shift
        if prev_bev is not None and shift is not None:
            layer_reference_points, layer_shift = shifted_reference_points(bev_reference_points, shift), None

        shapes = self.spatial_shapes
        camera_features = value if value is not None else key
        if camera_features is not None:
            expected_L = shapes.prod(dim=1).sum().item()
            L = camera_features.shape[2]
            assert expected_L == L, (
                f"Spatial shapes mismatch: spatial_shapes total ({expected_L}) != camera spatial dimension ({L}). "
                f"spatial_shapes: {shapes.tolist()}, camera features shape: {camera_features.shape}"
            )

        signpost(header="BEVEncoder Reference Points Start")

        assert cameras is None or img_metas is None, "pass img_metas or cameras, not both"
        if img_metas is not None:
            cameras = self.project_cameras(img_metas)
        reference_points_cam, bev_mask, rebatch_plan = None, None, None
        if cameras is not None:
            reference_points_cam, bev_mask, visibility, rebatch_plan = cameras
            # Camera projection is per-frame, so the SCA rebatch plan is too. Layers only gather query.
            if rebatch_plan is None:
                rebatch_plan = self.rebatch_planner(reference_points_cam, visibility)

        signpost(header="BEVEncoder Reference Points End")

        # Process through transformer layers
        for lid, layer in enumerate(self.layers):
            signpost(header=f"BEVEncoder Layer {lid} Start")

            output = layer(
                bev_query=output,
                value=camera_features,
                bev_pos=bev_pos,
                prev_bev=prev_bev,
                shift=layer_shift,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                rebatch_plan=rebatch_plan,
                bev_reference_points=layer_reference_points,
                **kwargs,
            )
            signpost(header=f"BEVEncoder Layer {lid} End")

            if self.return_intermediate:
                intermediate.append(output)

        if layer_reference_points is not bev_reference_points:
            ttnn.deallocate(layer_reference_points)
        if bev_reference_points is not self._bev_reference_points:
            ttnn.deallocate(bev_reference_points)

        signpost(header="TTNN BEVFormerEncoder Forward End")

        if self.return_intermediate:
            return intermediate

        return output

    def __call__(self, *args, **kwargs):
        """Make the class callable"""
        return self.forward(*args, **kwargs)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"num_layers={self.num_layers}, embed_dims={self.embed_dims}, "
            f"num_heads={self.num_heads}, num_levels={self.num_levels}, "
            f"num_points={self.num_points}, num_cams={self.num_cams}, "
            f"pc_range={self.pc_range}, num_points_in_pillar={self.num_points_in_pillar}, "
            f"bev_h={self.bev_h}, bev_w={self.bev_w}, "
            f"feedforward_channels={self.feedforward_channels}, z_cfg={self.z_cfg}"
        )
