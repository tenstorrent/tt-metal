# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Fused decode path of gpt-oss (one token per device, TP over one mesh row; Blackhole 1x4).

The model, decoder layers, attention, experts and router switch to these ops when
config.fused_decode_supported() is true; every other mesh / batch keeps the original decode path.

Usage:
    from models.demos.gpt_oss.tt.fused_decode import fused_decode_supported

    if fused_decode_supported(mesh_device, mesh_config, hf_config, use_throughput_experts, tokens_per_device):
        ...  # Model / DecoderLayer build the fused decode ops below
"""

from .boundary import DecodeBoundary, PendingBoundary
from .config import fused_decode_supported
from .inputs import DecodeInputs
from .stream import ExpertDownStream, ExpertGateUpStream, LinearStream
from .terminal import DecodeTerminal, FusedSamplingGenerator

__all__ = [
    "DecodeBoundary",
    "DecodeInputs",
    "DecodeTerminal",
    "ExpertDownStream",
    "ExpertGateUpStream",
    "FusedSamplingGenerator",
    "LinearStream",
    "PendingBoundary",
    "fused_decode_supported",
]
