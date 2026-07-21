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


def _with_dynamic_load(param, dynamic_load):
    """Append dynamic_node arg to a 7-field mesh param, yielding the pipeline signature
    (mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp, dynamic_load).
    Some tests use 7-field mesh_device params, some need an 8th dynamic_load. This function
    allows us concatenate them together"""
    v = param.values  # (mesh_shape, sp, tp, num_links, device_params, topology, is_fsdp)
    assert len(v) == 7, f"expected a 7-field mesh param, got {len(v)}: {param.id}"
    return pytest.param(*v, dynamic_load, marks=param.marks, id=param.id)


# dynamic_load=True only for the 2x4 (BH-like) configs — mirrors production, where 2x4 pages
# weights in/out (dynamic_load path) to avoid init-time DRAM OOM rather than per-layer FSDP gathers.
_PIPELINE_DL_TRUE = {"2x4sp0tp1nl1_line_is_fsdp1", "2x4sp1tp0nl2_line_is_fsdp0"}

LTX_PIPELINE_MESH_PARAMS_DL = [_with_dynamic_load(p, p.id in _PIPELINE_DL_TRUE) for p in LTX_PIPELINE_MESH_PARAMS]


def _override_base_device_params(base, device_params, *, id=None):
    """Swap the device_params (element 4) of a shared building block, keeping its geometry.
    Optional id override for variants that need a distinct test id (e.g. the i2v prefix)."""
    v = base.values  # (mesh_shape, sp, tp, num_links, device_params, topology, is_fsdp)
    return pytest.param(*v[:4], device_params, *v[5:], marks=base.marks, id=id or base.id)


# ---------------------------------------------------------------------------
# Distilled AV pipeline mesh params. Same geometry as the pipeline configs, but the audio
# decode chain needs bigger device pools on a few configs, and the WH 4x8 ring config runs
# dynamic_load. Reuse the shared building blocks for geometry (single-sourced); override
# device_params only where audio decode diverges, then append the distilled dynamic_load.
# ---------------------------------------------------------------------------
# Override device_params for the audio decode chain. Each dict augments the base line/ring
# params only with the pool(s) that config actually needs:
#   l1_small_size: the audio vocoder's native conv1d/conv2d taps (depthwise audio filters) run an
#     UntilizeWithHalo gather whose sharding/config tensors allocate from the dedicated L1_SMALL
#     pool. It defaults to 0, which OOMs the vocoder in decode; 32 KB matches the audio component
#     tests.
#   worker_l1_size: the WH 4x8 ring config's RingAttention otherwise hits a kernel code-size error;
#     the larger worker L1 gives its command stream room.
#   trace_region_size: under LTX_TRACED=1 both stage traces' command streams (stage-1 + the
#     larger-sequence stage-2) live here; measured need is ~236 MB at 1080p (get_trace_buffers_size),
#     so 500 MB leaves headroom.
_line_l1small = {**_line, "l1_small_size": 32768}
_ring_worker_l1 = {"worker_l1_size": 1344544, **_ring}
_line_trace = {**_line, "trace_region_size": 500_000_000, "l1_small_size": 32768}
_ring_trace = {**_ring, "trace_region_size": 500_000_000, "l1_small_size": 32768}

LTX_DISTILLED_MESH_PARAMS_DL = [
    _with_dynamic_load(_2x2sp0tp1nl2_line_is_fsdp1, False),
    _with_dynamic_load(_2x4sp0tp1nl1_line_is_fsdp1, True),
    # BH on 2x4: L1_SMALL scratch for the vocoder conv taps.
    _with_dynamic_load(_override_base_device_params(_2x4sp1tp0nl2_line_is_fsdp0, _line_l1small), True),
    # WH (ring) on 4x8: bigger worker L1 for RingAttention.
    _with_dynamic_load(_override_base_device_params(_4x8sp1tp0nl4_ring_is_fsdp1, _ring_worker_l1), True),
    # BH (linear) on 4x8.
    _with_dynamic_load(_4x8sp1tp0nl2_line_is_fsdp0, False),
    # BH (ring) on 4x8: trace region + L1_SMALL for the traced decode.
    _with_dynamic_load(_override_base_device_params(_4x8sp1tp0nl2_ring_is_fsdp0, _ring_trace), False),
    _with_dynamic_load(_4x32sp1tp0nl2_ring_is_fsdp0, False),
]

# I2V chained t2v->i2v; both configs are traced (trace region + L1_SMALL). 4x8 Galaxy (ring):
# full-res 1088x1920 latent shards unevenly on the 4x8 mesh (s1 cond latent 17x30, full 34x60),
# so the VAE encoder fold + even-shard padding must handle non-mesh-aligned dims here — the 2x4
# loudbox shards evenly and never hits this, so it's kept as a distinct id.
LTX_DISTILLED_I2V_MESH_PARAMS_DL = [
    _with_dynamic_load(_override_base_device_params(_2x4sp1tp0nl2_line_is_fsdp0, _line_trace), True),
    _with_dynamic_load(
        _override_base_device_params(_4x8sp1tp0nl2_ring_is_fsdp0, _ring_trace, id="i2v_4x8sp1tp0nl2_ring_is_fsdp0"),
        False,
    ),
]

# Audio-decode-only profiling: both configs use the traced line params (trace region + L1_SMALL)
# so LTX_TRACED=1 can capture the vocoder.
LTX_DISTILLED_AUDIO_MESH_PARAMS_DL = [
    _with_dynamic_load(_override_base_device_params(_2x4sp1tp0nl2_line_is_fsdp0, _line_trace), True),
    _with_dynamic_load(_override_base_device_params(_4x8sp1tp0nl2_line_is_fsdp0, _line_trace), False),
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

# ---------------------------------------------------------------------------
# Audio mesh params — (mesh_device, mesh_shape, sp_axis, tp_axis, num_links,
#                       dynamic_load, device_params, topology, is_fsdp)
# Audio tests create a submesh from mesh_device using mesh_shape, hence the
# duplicate shape field.
# ---------------------------------------------------------------------------
_line_audio = {**_line, "l1_small_size": 32768}
_ring_audio = {**_ring, "l1_small_size": 32768}
_ring_trace_audio = {**_ring_audio, "trace_region_size": 300_000_000}

LTX_AUDIO_MESH_PARAMS_FULL = [
    pytest.param(
        (2, 2),
        (2, 2),
        0,
        1,
        2,
        False,
        _line_audio,
        ttnn.Topology.Linear,
        True,
        id="2x2sp0tp1nl2_line_is_fsdp1",
    ),
    pytest.param(
        (2, 4),
        (2, 4),
        0,
        1,
        1,
        True,
        _line_audio,
        ttnn.Topology.Linear,
        True,
        id="2x4sp0tp1nl1_line_is_fsdp1",
    ),
    pytest.param(
        (2, 4),
        (2, 4),
        1,
        0,
        2,
        True,
        _line_audio,
        ttnn.Topology.Linear,
        False,
        id="2x4sp1tp0nl2_line_is_fsdp0",
    ),
    pytest.param(
        (4, 8),
        (4, 8),
        1,
        0,
        4,
        False,
        _ring_audio,
        ttnn.Topology.Ring,
        True,
        id="4x8sp1tp0nl4_ring_is_fsdp1",
    ),
    pytest.param(
        (4, 8),
        (4, 8),
        1,
        0,
        2,
        False,
        _line_audio,
        ttnn.Topology.Linear,
        False,
        id="4x8sp1tp0nl2_line_is_fsdp0",
    ),
    pytest.param(
        (4, 8),
        (4, 8),
        1,
        0,
        2,
        False,
        _ring_trace_audio,
        ttnn.Topology.Ring,
        False,
        id="4x8sp1tp0nl2_ring_is_fsdp0",
    ),
    pytest.param(
        (4, 32),
        (4, 32),
        1,
        0,
        2,
        False,
        _ring_audio,
        ttnn.Topology.Ring,
        False,
        id="4x32sp1tp0nl2_ring_is_fsdp0",
    ),
]
