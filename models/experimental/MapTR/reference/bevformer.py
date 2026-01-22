# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Consolidated BEVFormer module - imports from flat structure

# Import all BEVFormer components from root-level files
from models.experimental.MapTR.reference.bevformer_encoder import (
    BEVFormerEncoder,
    BEVFormerLayer,
)
from models.experimental.MapTR.reference.bevformer_decoder import (
    DetectionTransformerDecoder,
    CustomMSDeformableAttention,
)
from models.experimental.MapTR.reference.bevformer_temporal_attention import (
    TemporalSelfAttention,
)
from models.experimental.MapTR.reference.bevformer_spatial_attention import (
    SpatialCrossAttention,
    MSDeformableAttention3D,
    MSIPM3D,
)
from models.experimental.MapTR.reference.bevformer_base_layer import (
    MyCustomBaseTransformerLayer,
    MyCustomBaseTransformerLayerWithoutSelfAttn,
)
from models.experimental.MapTR.reference.bevformer_deformable_attn import (
    MultiScaleDeformableAttnFunction_fp32,
    MultiScaleDeformableAttnFunction_fp16,
)

__all__ = [
    "BEVFormerEncoder",
    "BEVFormerLayer",
    "DetectionTransformerDecoder",
    "CustomMSDeformableAttention",
    "TemporalSelfAttention",
    "SpatialCrossAttention",
    "MSDeformableAttention3D",
    "MSIPM3D",
    "MyCustomBaseTransformerLayer",
    "MyCustomBaseTransformerLayerWithoutSelfAttn",
    "MultiScaleDeformableAttnFunction_fp32",
    "MultiScaleDeformableAttnFunction_fp16",
]
