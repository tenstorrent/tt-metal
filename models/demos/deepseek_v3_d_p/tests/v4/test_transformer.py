# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for TtV4Transformer: embed -> hyper-connection expand -> V4 blocks -> hc_head -> norm.

Random weights on both sides, against the CPU stack composed from the same reference modules the block
test uses (reference.deepseek_v4.model). What this adds over the block test is the model level: the
device embedding and stream expand feeding layer 0, layers chaining their packed streams, the TP-sharded
head collapse and the final norm. Each layer's output is scored, so a failure names the layer.

Depth 3 is the shallowest that reaches CSA: layer 2 is CSA in both, after two HCA layers in Pro and
two SWA layers in Flash. Flash's first HCA layer is 3, so at this depth Flash HCA runs only in the
block tests and the full-depth job.
"""

import pytest
import torch
from loguru import logger

from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.hf_config import v4_hf_config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.model import (
    build_v4_model_reference,
    v4_model_forward,
    v4_model_state_dict,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.tests.v4.helpers import (
    assert_layers_and_output,
    hidden_to_host,
    mesh_params,
    streams_to_host,
    upload_tokens,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v4 import TtV4Transformer

# Pro's floor is the lower one, as in the block test: 128 heads and a 7168-wide hidden make every bf16
# reduction longer.
_PRO_FLOOR, _FLASH_FLOOR = 0.98, 0.99
_NUM_LAYERS = [pytest.param(1, id="L1"), pytest.param(3, id="L3")]
_SEED = 42


def run_transformer(mesh_device, device_params, num_links, seq_len, model_config, floor, num_layers):
    """The first ``num_layers`` layers of a V4 model and its tail vs the composed CPU reference, unchunked."""
    config = v4_hf_config(model_config, num_layers, max_seq_len=seq_len)
    logger.info(
        f"[v4 transformer] {model_config.__name__} L{num_layers}: {config.layer_types} / {config.mlp_layer_types}"
    )

    ref = build_v4_model_reference(config, seed=_SEED)
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
    ref_out, ref_layers = v4_model_forward(ref, config, input_ids)

    # The reference's bf16 experts (~50 GB per Pro layer) are shared with the state dict, not copied;
    # dropping the reference frees the rest of it.
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
        config,
        per_layer,
        ref_layers,
        hidden_to_host(mesh_device, out),
        ref_out,
        floor,
        f"{model_config.__name__} L{num_layers}",
    )


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    mesh_params(DeepSeekV4ProConfig.FABRIC_PAYLOAD_SIZE),
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("seq_len", [5120], ids=["seq5120"])
@pytest.mark.parametrize("num_layers", _NUM_LAYERS)
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_pro_transformer(mesh_device, device_params, num_links, seq_len, num_layers):
    run_transformer(mesh_device, device_params, num_links, seq_len, DeepSeekV4ProConfig, _PRO_FLOOR, num_layers)


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    mesh_params(DeepSeekV4FlashConfig.FABRIC_PAYLOAD_SIZE),
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("seq_len", [5120], ids=["seq5120"])
@pytest.mark.parametrize("num_layers", _NUM_LAYERS)
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_flash_transformer(mesh_device, device_params, num_links, seq_len, num_layers):
    run_transformer(mesh_device, device_params, num_links, seq_len, DeepSeekV4FlashConfig, _FLASH_FLOOR, num_layers)
