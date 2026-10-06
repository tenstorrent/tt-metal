# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Chunked-prefill PCC for TtV4Transformer: the prompt in 5120-token chunks against one unchunked CPU pass.

Random weights. Chunk c+1 of every layer attends over the state chunk c left, so the per-layer and
output PCC over the whole prompt grade the attention state carried across chunk boundaries, at model
level. Every chunk drives the same device shapes, so the program cache must not grow after chunk 0.
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
from models.demos.deepseek_v3_d_p.tests.v4.test_transformer import (
    assert_layers_and_output,
    hidden_to_host,
    mesh_params,
    streams_to_host,
    upload_tokens,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v4 import TtV4Transformer
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS

CHUNK = PREFILL_CHUNK_TOKENS
_CASES = [
    pytest.param(DeepSeekV4ProConfig, 0.98, id="pro-L3"),
    pytest.param(DeepSeekV4FlashConfig, 0.99, id="flash-L3"),
]
_NUM_LAYERS = 3
_SEED = 7


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    mesh_params(DeepSeekV4ProConfig.FABRIC_PAYLOAD_SIZE),
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("n_chunks", [2], ids=["chunks2"])
@pytest.mark.parametrize("model_config, floor", _CASES)
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_transformer_chunked(mesh_device, device_params, num_links, n_chunks, model_config, floor):
    total = n_chunks * CHUNK
    config = v4_hf_config(model_config, _NUM_LAYERS, max_seq_len=total)
    ref = build_v4_model_reference(config, seed=_SEED)
    input_ids = torch.randint(0, config.vocab_size, (1, total))
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
        _NUM_LAYERS,
        CHUNK,
        max_seq_len=total,
        num_links=num_links,
        topology=per_axis_topology(device_params["fabric_config"]),
    )
    del state_dict

    per_layer = [[] for _ in range(_NUM_LAYERS)]
    outs = []
    mesh_device.enable_program_cache()
    programs = mesh_device.num_program_cache_entries()
    for chunk in range(n_chunks):
        start = chunk * CHUNK
        out = model(
            upload_tokens(mesh_device, input_ids[:, start : start + CHUNK]),
            actual_isl=CHUNK,
            actual_start=start,
            layer_tap=lambda i, h: per_layer[i].append(streams_to_host(mesh_device, h, config.hc_mult)),
        )
        outs.append(hidden_to_host(mesh_device, out))
        now = mesh_device.num_program_cache_entries()
        if chunk:
            assert now == programs, f"chunk {chunk} compiled {now - programs} new program(s) ({programs} -> {now})"
        programs = now
        logger.info(f"  chunk {chunk} done (start={start})")

    assert_layers_and_output(
        config,
        [torch.cat(chunks, dim=1) for chunks in per_layer],
        ref_layers,
        torch.cat(outs, dim=1),
        ref_out,
        floor,
        f"{model_config.__name__} L{_NUM_LAYERS} x{n_chunks} chunks",
    )
