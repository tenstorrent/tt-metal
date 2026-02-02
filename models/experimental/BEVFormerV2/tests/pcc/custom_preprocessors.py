# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

"""
Custom preprocessors for PCC tests.
"""

from models.experimental.BEVFormerV2.reference.resnet import ResNet
from models.experimental.BEVFormerV2.reference.fpn import FPN
from models.experimental.BEVFormerV2.reference.encoder import BEVFormerEncoder
from models.experimental.BEVFormerV2.reference.decoder import DetectionTransformerDecoder
from models.experimental.BEVFormerV2.reference.temporal_self_attention import TemporalSelfAttention
from models.experimental.BEVFormerV2.reference.spatial_cross_attention import SpatialCrossAttention
from models.experimental.BEVFormerV2.reference.multihead_attention import (
    MultiheadAttention,
    CustomMSDeformableAttention,
)

from models.experimental.BEVFormerV2.tt.model_preprocessing import (
    extract_transformer_layers_parameters,
    extract_resnet_parameters,
    extract_fpn_parameters,
    extract_temporal_self_attention_parameters,
    extract_spatial_cross_attention_parameters,
    extract_multihead_attention_parameters,
    extract_custom_ms_deformable_attention_parameters,
)


def custom_preprocessor_resnet(model, name):
    """Preprocessor for ResNet models."""
    parameters = {}
    if isinstance(model, ResNet):
        parameters.update(extract_resnet_parameters(model))
    return parameters


def custom_preprocessor_fpn(model, name):
    """Preprocessor for FPN models."""
    parameters = {}
    if isinstance(model, FPN):
        parameters.update(extract_fpn_parameters(model))
    return parameters


def custom_preprocessor_encoder(model, name):
    """Preprocessor for BEVFormerEncoder models."""
    parameters = {}
    if isinstance(model, BEVFormerEncoder):
        parameters.update(extract_transformer_layers_parameters(model))
    return parameters


def custom_preprocessor_decoder(model, name):
    """Preprocessor for DetectionTransformerDecoder models."""
    parameters = {}
    if isinstance(model, DetectionTransformerDecoder):
        parameters.update(extract_transformer_layers_parameters(model))
    return parameters


def custom_preprocessor_temporal_self_attention(model, name):
    """Preprocessor for TemporalSelfAttention models."""
    parameters = {}
    if isinstance(model, TemporalSelfAttention):
        parameters["temporal_self_attention"] = extract_temporal_self_attention_parameters(model)
    return parameters


def custom_preprocessor_spatial_cross_attention(model, name):
    """Preprocessor for SpatialCrossAttention models."""
    parameters = {}
    if isinstance(model, SpatialCrossAttention):
        parameters["spatial_cross_attention"] = extract_spatial_cross_attention_parameters(model)
    return parameters


def custom_preprocessor_multihead_attention(model, name):
    """Preprocessor for MultiheadAttention models."""
    parameters = {}
    if isinstance(model, MultiheadAttention):
        parameters["multihead_attention"] = extract_multihead_attention_parameters(model)
    return parameters


def custom_preprocessor_custom_ms_deformable_attention(model, name):
    """Preprocessor for CustomMSDeformableAttention models."""
    parameters = {}
    if isinstance(model, CustomMSDeformableAttention):
        parameters["custom_ms_deformable_attention"] = extract_custom_ms_deformable_attention_parameters(model)
    return parameters
