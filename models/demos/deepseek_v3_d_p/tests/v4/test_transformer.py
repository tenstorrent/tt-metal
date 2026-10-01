# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for TtV4Transformer: embed -> hyper-connection expand -> V4 blocks -> hc_head -> norm.

Random weights on both sides, against the CPU stack composed from the same reference modules the block
test uses (reference.deepseek_v4.model). What this adds over the block test is the model level: the
device embedding and stream expand feeding layer 0, layers chaining their packed streams, the TP-sharded
head collapse and the final norm. Each layer's output is scored, so a failure names the layer.

Depth stops at 2: layer 2 is CSA in both models, which has no device implementation.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.hf_config import v4_hf_config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.model import (
    build_v4_model_reference,
    v4_model_forward,
    v4_model_state_dict,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.v4.test_block import _unpack_streams
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v4 import TtV4Transformer

# Pro's floor is the lower one, as in the block test: 128 heads and a 7168-wide hidden make every bf16
# reduction longer.
_CASES = [
    pytest.param(DeepSeekV4ProConfig, 0.98, 1, id="pro-L1"),
    pytest.param(DeepSeekV4ProConfig, 0.98, 2, id="pro-L2"),
    pytest.param(DeepSeekV4FlashConfig, 0.99, 1, id="flash-L1"),
    pytest.param(DeepSeekV4FlashConfig, 0.99, 2, id="flash-L2"),
]
_SEED = 42


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
    return _unpack_streams(full, n, ms[1]).float()


def hidden_to_host(mesh_device, t):
    """Device hidden ``[1, 1, S/sp, D/tp]`` -> ``[1, S, D]``."""
    ms = tuple(mesh_device.shape)
    full = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=ms, dims=(2, 3)))
    return full[0].float()


def assert_layers_and_output(per_layer, ref_layers, out, ref_out, floor, label):
    """Every layer's streams and the final output against the reference, each at ``floor``."""
    failures = []
    for i, (dev, ref) in enumerate(zip(per_layer, ref_layers)):
        _, pcc = comp_pcc(ref.float(), dev)
        logger.info(f"[{label}] layer {i} PCC: {pcc:.6f}")
        if pcc < floor:
            failures.append(f"layer {i} {pcc:.6f}")
    _, pcc = comp_pcc(ref_out.float(), out)
    logger.info(f"[{label}] output PCC: {pcc:.6f}")
    if pcc < floor:
        failures.append(f"output {pcc:.6f}")
    assert not failures, f"[{label}] below {floor}: {', '.join(failures)}"


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    mesh_params(DeepSeekV4ProConfig.FABRIC_PAYLOAD_SIZE),
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("seq_len", [5120], ids=["seq5120"])
@pytest.mark.parametrize("model_config, floor, num_layers", _CASES)
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_transformer(mesh_device, device_params, num_links, seq_len, model_config, floor, num_layers):
    """The first ``num_layers`` layers of a V4 model and its tail vs the composed CPU reference, unchunked."""
    config = v4_hf_config(model_config, num_layers, max_seq_len=seq_len)
    logger.info(
        f"[v4 transformer] {model_config.__name__} L{num_layers}: {config.layer_types} / {config.mlp_layer_types}"
    )

    ref = build_v4_model_reference(config, seed=_SEED)
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
    ref_out, ref_layers = v4_model_forward(ref, config, input_ids)

    # The fp32 reference is the bulk of host memory (~100 GB per Pro layer); only the bf16 weights
    # and the attention modules have to outlive it.
    state_dict = v4_model_state_dict(ref, config)
    del ref
    model = TtV4Transformer(
        mesh_device,
        config,
        model_config,
        state_dict,
        num_layers,
        seq_len,
        max_seq_len=seq_len,
        num_links=num_links,
        topology=per_axis_topology(device_params["fabric_config"]),
    )
    del state_dict

    per_layer = []
    out = model(
        upload_tokens(mesh_device, input_ids),
        actual_isl=seq_len,
        layer_tap=lambda _i, h: per_layer.append(streams_to_host(mesh_device, h, config.hc_mult)),
    )
    assert_layers_and_output(
        per_layer,
        ref_layers,
        hidden_to_host(mesh_device, out),
        ref_out,
        floor,
        f"{model_config.__name__} L{num_layers}",
    )
