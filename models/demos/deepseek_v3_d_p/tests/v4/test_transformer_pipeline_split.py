# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pipeline-split PCC for TtV4Transformer: two ranks on one mesh, layer 0 on the first and layers 1-2 on
the second, against the unsplit CPU reference.

Random weights. The first rank's output is the packed residual streams, handed straight to the second
rank, so this grades the handoff shape and the per-rank build: embedding only on the first, head and
norm only on the last, global layer indices on both. Layers 1-2 are hash-routed, so the second rank is
given the host ids the first rank would read off its token tensor.
"""

import pytest
import torch

from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.hf_config import v4_hf_config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.model import (
    build_v4_model_reference,
    v4_model_forward,
    v4_model_state_dict,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.tests.v4.test_transformer import (
    assert_layers_and_output,
    hidden_to_host,
    mesh_params,
    streams_to_host,
    upload_tokens,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v4 import TtV4Transformer

_CASES = [
    pytest.param(DeepSeekV4ProConfig, 0.98, id="pro"),
    pytest.param(DeepSeekV4FlashConfig, 0.99, id="flash"),
]
_NUM_LAYERS = 3
_BOUNDARY = 1
_SEED = 11


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    mesh_params(DeepSeekV4ProConfig.FABRIC_PAYLOAD_SIZE),
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("seq_len", [5120], ids=["seq5120"])
@pytest.mark.parametrize("model_config, floor", _CASES)
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_transformer_pipeline_split(mesh_device, device_params, num_links, seq_len, model_config, floor):
    config = v4_hf_config(model_config, _NUM_LAYERS, max_seq_len=seq_len)
    ref = build_v4_model_reference(config, seed=_SEED)
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
    ref_out, ref_layers = v4_model_forward(ref, config, input_ids)

    topology = per_axis_topology(device_params["fabric_config"])
    slices = ((0, _BOUNDARY, True, False), (_BOUNDARY, _NUM_LAYERS - _BOUNDARY, False, True))
    # The reference's bf16 experts (~50 GB per Pro layer) are shared with the state dict, not copied;
    # dropping the reference frees the rest of it.
    state_dicts = [
        v4_model_state_dict(ref, config, first_layer_idx=first, num_layers=count) for first, count, _, _ in slices
    ]
    del ref
    ranks = []
    for (first, count, is_first, is_last), state_dict in zip(slices, state_dicts):
        ranks.append(
            TtV4Transformer(
                mesh_device,
                config,
                model_config,
                state_dict,
                count,
                seq_len,
                first_layer_idx=first,
                is_first_rank=is_first,
                is_last_rank=is_last,
                max_seq_len=seq_len,
                num_links=num_links,
                topology=topology,
            )
        )
    del state_dicts, state_dict
    assert ranks[0].hc_head is None and ranks[1].embed is None

    per_layer = []
    tap = lambda _i, h: per_layer.append(streams_to_host(mesh_device, h, config.hc_mult))
    handoff = ranks[0](upload_tokens(mesh_device, input_ids), actual_isl=seq_len, layer_tap=tap)
    out = ranks[1](handoff, actual_isl=seq_len, input_ids=input_ids, layer_tap=tap)

    assert_layers_and_output(
        config,
        per_layer,
        ref_layers,
        hidden_to_host(mesh_device, out),
        ref_out,
        floor,
        f"{model_config.__name__} split@{_BOUNDARY}",
    )
