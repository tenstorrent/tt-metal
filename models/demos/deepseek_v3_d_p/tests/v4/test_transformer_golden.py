# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Chunked-prefill PCC for TtV4Transformer against a vLLM golden, real weights, free-running.

The first two layers of the checkpoint, fed the golden's own prompt a chunk at a time, every layer's
output compared against ``decoder_output_layer_{i}`` over the whole prompt. Unlike
``test_block_chunked.py`` nothing is teacher-forced: layer 1 consumes layer 0's device output, so the
error at layer 1 includes everything the embedding, the expand and layer 0 introduced.

The trace has no final-norm stream, so the tail is not graded here.
"""

import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_pcc, is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.hf_config import v4_hf_config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.tests.v4 import golden
from models.demos.deepseek_v3_d_p.tests.v4.test_transformer import mesh_params, streams_to_host, upload_tokens
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v4 import TtV4Transformer
from models.demos.deepseek_v3_d_p.tt.v4.weights import V4CheckpointLayers, v4_model_from_checkpoint
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS

CHUNK = PREFILL_CHUNK_TOKENS
SEQ_CACHE = 55 * 1024  # 56320, the length the golden was captured at
_NUM_LAYERS = 2  # layer 2 is CSA in both models
_LAYER_PCC = 0.99

_GOLDEN = {DeepSeekV4ProConfig: golden.V4_PRO, DeepSeekV4FlashConfig: golden.V4_FLASH}


def run_transformer_golden(mesh_device, device_params, num_links, model_config, n_chunks):
    # Asserts rather than skips: this row is meant for CI, and a skipped row reads as a green one.
    gold = _GOLDEN[model_config]
    trace = golden.resolve_trace(gold)
    assert trace is not None, f"no golden trace at {gold.trace}; ${gold.trace_env} overrides the path"
    checkpoint = golden.resolve_checkpoint(gold)
    assert checkpoint is not None, f"no checkpoint at {gold.checkpoint}; ${' / $'.join(gold.ckpt_envs)} override it"
    missing = [i for i in range(_NUM_LAYERS) if i not in trace.kept_layers]
    assert not missing, f"{trace.path.name} did not keep decoder_output for layers {missing}"

    total = n_chunks * CHUNK
    assert total <= SEQ_CACHE, f"{n_chunks} chunks ({total}) exceed the golden's {SEQ_CACHE}"
    config = v4_hf_config(model_config, _NUM_LAYERS, max_seq_len=SEQ_CACHE)
    state_dict = v4_model_from_checkpoint(config, checkpoint, load_embed=True, load_tail=True)
    state_dict["layers"] = V4CheckpointLayers(config, checkpoint, 0, _NUM_LAYERS)
    model = TtV4Transformer(
        mesh_device,
        config,
        model_config,
        state_dict,
        _NUM_LAYERS,
        CHUNK,
        max_seq_len=SEQ_CACHE,
        num_links=num_links,
        topology=per_axis_topology(device_params["fabric_config"]),
    )

    n = config.hc_mult
    per_layer = [torch.zeros(1, total, n, config.hidden_size) for _ in range(_NUM_LAYERS)]
    for chunk in range(n_chunks):
        start = chunk * CHUNK

        def tap(i, h, start=start):
            per_layer[i][:, start : start + CHUNK] = streams_to_host(mesh_device, h, n)

        model(
            upload_tokens(mesh_device, trace.token_ids(CHUNK, start)),
            actual_isl=CHUNK,
            actual_start=start,
            layer_tap=tap,
        )
        logger.info(f"  chunk {chunk} done (start={start})")

    failures = []
    for i in range(_NUM_LAYERS):
        truth = trace.decoder_output(i, 0, total).reshape(1, total, n, config.hidden_size)
        _, pcc = comp_pcc(truth.float(), per_layer[i])
        logger.info(f"[v4 transformer golden {model_config.__name__}] layer {i} PCC: {pcc:.6f}")
        if pcc < _LAYER_PCC:
            failures.append(f"layer {i} {pcc:.6f}")
    assert not failures, f"below {_LAYER_PCC}: {', '.join(failures)}"


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    mesh_params(DeepSeekV4ProConfig.FABRIC_PAYLOAD_SIZE),
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("n_chunks", [11], ids=["chunks11"])
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_pro_transformer_golden(mesh_device, device_params, num_links, n_chunks):
    run_transformer_golden(mesh_device, device_params, num_links, DeepSeekV4ProConfig, n_chunks)


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    mesh_params(DeepSeekV4FlashConfig.FABRIC_PAYLOAD_SIZE),
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("n_chunks", [11], ids=["chunks11"])
@pytest.mark.skipif(not is_blackhole(), reason="V4 attention is Blackhole-only")
@pytest.mark.timeout(0)
def test_v4_flash_transformer_golden(mesh_device, device_params, num_links, n_chunks):
    run_transformer_golden(mesh_device, device_params, num_links, DeepSeekV4FlashConfig, n_chunks)
