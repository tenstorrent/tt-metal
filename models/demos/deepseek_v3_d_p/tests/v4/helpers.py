# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Stream packing, mesh rows, device<->host transfers and PCC grading shared by the V4 tests."""

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor

# How far below its model's floor a CSA layer is graded: with production top-k, bf16 can swap
# near-tied candidates at the selection boundary, which no other attention kind is exposed to, and
# the swaps compound with the error the layer inherits.
CSA_LAYER_MARGIN = 0.01


def pack_streams(x, tp_factor):
    """[1, S, n, D] -> [1, 1, S, n*D], ordered for a TP-sharded mesh.

    The device packs the streams on the last dim, and TP shards that dim contiguously -- so a plain
    reshape would hand chip 0 the whole of stream 0 instead of its own D-slice of every stream. The
    columns are reordered chip-major-then-stream so the contiguous split lands right; inside a chip
    the order is stream-major again, which is what the hyper-connection slices.
    """
    b, seq, n, d = x.shape
    return x.reshape(b, seq, n, tp_factor, d // tp_factor).permute(0, 1, 3, 2, 4).reshape(b, 1, seq, n * d)


def unpack_streams(t, n, tp_factor):
    """The inverse of pack_streams: [1, 1, S, n*D] -> [1, S, n, D]."""
    b, _, seq, width = t.shape
    d = width // n
    return t.reshape(b, seq, tp_factor, n, d // tp_factor).permute(0, 1, 3, 2, 4).reshape(b, seq, n, d)


def mesh_params(payload: int):
    """The 4x2 (LoudBox) and 8x4 (Galaxy) rows, with the variant's own fabric payload."""
    return [
        pytest.param(
            (4, 2),
            fabric2d_device_params(fabric_payload_size=payload),
            1,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="fabric2d-mesh-4x2",
        ),
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=payload),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ]


def upload_tokens(mesh_device, input_ids):
    """``[1, S]`` host ids -> the SP-sharded id tensor the first rank takes."""
    ms = tuple(mesh_device.shape)
    return prepare_prefill_input_tensor(input_ids[0].tolist(), mesh_device, ms[0], False, ms, 0)


def streams_to_host(mesh_device, t, n):
    """Packed device streams ``[1, 1, S/sp, n*D/tp]`` -> ``[1, S, n, D]``."""
    ms = tuple(mesh_device.shape)
    full = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=ms, dims=(2, 3)))
    return unpack_streams(full, n, ms[1]).float()


def hidden_to_host(mesh_device, t):
    """Device hidden ``[1, 1, S/sp, D/tp]`` -> ``[1, S, D]``."""
    ms = tuple(mesh_device.shape)
    full = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=ms, dims=(2, 3)))
    return full[0].float()


def layer_floor(config, layer_idx: int, floor: float) -> float:
    """``floor``, lowered by ``CSA_LAYER_MARGIN`` on a CSA layer."""
    if config.layer_types[layer_idx] == "compressed_sparse_attention":
        return round(floor - CSA_LAYER_MARGIN, 6)
    return floor


def assert_layers_and_output(config, per_layer, ref_layers, out, ref_out, floor, label):
    """Every layer's streams against the reference at its ``layer_floor``, and the final output at ``floor``."""
    failures = []
    for i, (dev, ref) in enumerate(zip(per_layer, ref_layers)):
        _, pcc = comp_pcc(ref.float(), dev)
        logger.info(f"[{label}] layer {i} PCC: {pcc:.6f}")
        if pcc < layer_floor(config, i, floor):
            failures.append(f"layer {i} {pcc:.6f} < {layer_floor(config, i, floor)}")
    _, pcc = comp_pcc(ref_out.float(), out)
    logger.info(f"[{label}] output PCC: {pcc:.6f}")
    if pcc < floor:
        failures.append(f"output {pcc:.6f} < {floor}")
    assert not failures, f"[{label}] {', '.join(failures)}"
