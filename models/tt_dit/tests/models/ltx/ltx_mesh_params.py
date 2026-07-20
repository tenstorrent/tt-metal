# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest

import ttnn
from models.tt_dit.utils.test import line_params_req_exact_devices, ring_params_req_exact_devices

_line = line_params_req_exact_devices
_ring = ring_params_req_exact_devices

_1x1sp0tp1nl1_line_is_fsdp0 = pytest.param(
    (1, 1),
    0,
    1,
    1,
    {},
    ttnn.Topology.Linear,
    False,
    id="1x1sp0tp1nl1_line_is_fsdp0",
)
_2x2sp0tp1nl2_line_is_fsdp1 = pytest.param(
    (2, 2),
    0,
    1,
    2,
    _line,
    ttnn.Topology.Linear,
    True,
    id="2x2sp0tp1nl2_line_is_fsdp1",
)
_2x4sp0tp1nl1_line_is_fsdp1 = pytest.param(
    (2, 4),
    0,
    1,
    1,
    _line,
    ttnn.Topology.Linear,
    True,
    id="2x4sp0tp1nl1_line_is_fsdp1",
)
_2x4sp1tp0nl1_line_is_fsdp1 = pytest.param(
    (2, 4),
    1,
    0,
    1,
    _line,
    ttnn.Topology.Linear,
    True,
    id="2x4sp1tp0nl1_line_is_fsdp1",
)
_2x4sp1tp0nl2_line_is_fsdp0 = pytest.param(
    (2, 4),
    1,
    0,
    2,
    _line,
    ttnn.Topology.Linear,
    False,
    id="2x4sp1tp0nl2_line_is_fsdp0",
)  # is_fsdp=False mirrors production BH 2x4 (dynamic_load path, not per-layer FSDP gathers)

# WH has 4 links.
_4x8sp1tp0nl4_ring_is_fsdp1 = pytest.param(
    (4, 8),
    1,
    0,
    4,
    _ring,
    ttnn.Topology.Ring,
    True,
    id="4x8sp1tp0nl4_ring_is_fsdp1",
)
_4x8sp1tp0nl2_line_is_fsdp0 = pytest.param(
    (4, 8),
    1,
    0,
    2,
    _line,
    ttnn.Topology.Linear,
    False,
    id="4x8sp1tp0nl2_line_is_fsdp0",
)
_4x8sp1tp0nl2_ring_is_fsdp0 = pytest.param(
    (4, 8),
    1,
    0,
    2,
    _ring,
    ttnn.Topology.Ring,
    False,
    id="4x8sp1tp0nl2_ring_is_fsdp0",
)
_4x32sp1tp0nl2_ring_is_fsdp0 = pytest.param(
    (4, 32),
    1,
    0,
    2,
    _ring,
    ttnn.Topology.Ring,
    False,
    id="4x32sp1tp0nl2_ring_is_fsdp0",
)

LTX_PIPELINE_MESH_PARAMS = [
    _2x2sp0tp1nl2_line_is_fsdp1,
    _2x4sp0tp1nl1_line_is_fsdp1,
    _2x4sp1tp0nl2_line_is_fsdp0,
    _4x8sp1tp0nl4_ring_is_fsdp1,
    _4x8sp1tp0nl2_line_is_fsdp0,
    _4x8sp1tp0nl2_ring_is_fsdp0,
    _4x32sp1tp0nl2_ring_is_fsdp0,
]

# No 1x1 config: real-grid shapes require SP padding, and video self-attention only masks
# padded keys via ring SDPA's logical_n (sp>1). Production never runs sp=1.
LTX_TRANSFORMER_MESH_PARAMS = [
    _2x4sp0tp1nl1_line_is_fsdp1,
    _2x4sp1tp0nl2_line_is_fsdp0,
    _4x8sp1tp0nl4_ring_is_fsdp1,
    _4x8sp1tp0nl2_ring_is_fsdp0,
    _4x8sp1tp0nl2_line_is_fsdp0,
]

LTX_ATTENTION_MESH_PARAMS = [
    _1x1sp0tp1nl1_line_is_fsdp0,
    _2x4sp0tp1nl1_line_is_fsdp1,
    _2x4sp1tp0nl1_line_is_fsdp1,
    _4x8sp1tp0nl4_ring_is_fsdp1,
    _4x8sp1tp0nl2_line_is_fsdp0,
]

# Upsampler uses VaeHWParallelConfig (h_axis, w_axis) instead of DiT sp/tp axes.
_2x4h0w1nl1_line = pytest.param(
    (2, 4),
    0,
    1,
    1,
    {**_line, "trace_region_size": 23887872},
    ttnn.Topology.Linear,
    id="2x4h0w1nl1_line",
)

_4x8h0w1nl2_line = pytest.param(
    (4, 8),
    0,
    1,
    2,
    {**_line, "trace_region_size": 23887872},
    ttnn.Topology.Linear,
    id="4x8h0w1nl2_line",
)

LTX_UPSAMPLER_MESH_PARAMS = [
    _2x4h0w1nl1_line,
    _4x8h0w1nl2_line,
]

# VAE decoder mesh params — multi-device configs for full LTXVideoDecoder parity.
# Format: (mesh_shape, h_axis, w_axis, num_links, device_params, topology)
_LTX_SKIP_SMALL_MESH_FABRIC = os.environ.get("LTX_VAE_FORCE_SMALL_MESH_FABRIC", "0") != "1"

_line_fabric = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True}

_1x1h0w1nl1_line_vae = pytest.param(
    (1, 1),
    0,
    1,
    1,
    _line_fabric,
    ttnn.Topology.Linear,
    marks=pytest.mark.skipif(
        _LTX_SKIP_SMALL_MESH_FABRIC,
        reason="Known flaky fabric handshake on 1x1 mesh; set LTX_VAE_FORCE_SMALL_MESH_FABRIC=1 to force-run.",
    ),
    id="1x1h0w1nl1_line",
)

_2x4h0w1nl1_line_vae = pytest.param(
    (2, 4),
    0,
    1,
    1,
    _line_fabric,
    ttnn.Topology.Linear,
    id="2x4h0w1nl1_line",
)

_2x4h1w0nl1_line_vae = pytest.param(
    (2, 4),
    1,
    0,
    1,
    _line_fabric,
    ttnn.Topology.Linear,
    id="2x4h1w0nl1_line",
)

_1x8h0w1nl1_line_vae = pytest.param(
    (1, 8),
    0,
    1,
    1,
    _line_fabric,
    ttnn.Topology.Linear,
    id="1x8h0w1nl1_line",
)

_1x4h1w0nl1_line_vae = pytest.param(
    (1, 4),
    1,
    0,
    1,
    _line_fabric,
    ttnn.Topology.Linear,
    marks=pytest.mark.skipif(
        _LTX_SKIP_SMALL_MESH_FABRIC,
        reason="Known flaky fabric handshake on 1x4 mesh; set LTX_VAE_FORCE_SMALL_MESH_FABRIC=1 to force-run.",
    ),
    id="1x4h1w0nl1_line",
)

_4x8h0w1nl2_line_vae = pytest.param(
    (4, 8),
    0,
    1,
    2,
    _line_fabric,
    ttnn.Topology.Linear,
    id="4x8h0w1nl2_line",
)

LTX_VAE_DECODER_MESH_PARAMS = [
    _1x1h0w1nl1_line_vae,
    _2x4h0w1nl1_line_vae,
    _2x4h1w0nl1_line_vae,
    _1x8h0w1nl1_line_vae,
    _1x4h1w0nl1_line_vae,
    _4x8h0w1nl2_line_vae,
]

# Multi-device only (2K decode doesn't fit on a single device).
LTX_VAE_DECODER_MULTI_ONLY_MESH_PARAMS = [
    _2x4h0w1nl1_line_vae,
    _2x4h1w0nl1_line_vae,
    _4x8h0w1nl2_line_vae,
]

# Encoder production mesh (mirrors the I2V pipeline layout).
LTX_VAE_ENCODER_PROD_MESH_PARAMS = [
    _4x8h0w1nl2_line_vae,
]
