# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill PCC of ``DeepSeekV41PrefillDecoderLayer`` against the checkpoint's own ``Block``, on real weights.

A case runs a group of layers over a prompt, each on its own random streams and ``pre_mix`` and in layer
order -- the order their shared compressed KV / selection needs. The reference (``tests/reference.py``) runs the
same blocks over the whole prompt in one pass at ``start_pos = 0``, an Engram layer's ``model.Engram`` ahead of
its block as ``Transformer.forward`` does. The device sees the prompt as chunks of ``chunk`` tokens.

Engram layers get random bf16 rows, handed to the device and (through ``GivenRows``) to the reference, so no
n-gram table is read.

The streams carry the input through the hyper-connection mix almost unchanged, so the output alone says little
about the sublayers: the check is PCC of what the layer adds to the streams (``out - x``) and, by absolute error,
the ``pre_mix`` it hands on.

Expected to fail until ``fused_experts_prefill`` is built for V4.1's shapes (see :mod:`..tt.prefill.moe`): the
device half runs first so that failure is quick, before the reference is built.

The converted ttnn weights are cached under ``CACHE_DIR`` (``$DEEPSEEK_V4_CACHE_DIR``, default ``../cache``, per checkpoint),
shared with the decode tests; ``tools/fill_expert_cache.py`` converts the routed experts ahead of time.

    pytest models/experimental/deepseek_v41_flash/tests/prefill/test_decoder_layer.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v41_flash.tests.reference import (
    GivenRows,
    load_reference,
    reference_blocks,
    reference_engram_with_embed,
)
from models.experimental.deepseek_v41_flash.tt.config import (
    CACHE_DIR,
    DEFAULT_MODEL_DIR,
    decoder_layer_weights,
    expert_provider,
    load_config,
)
from models.experimental.deepseek_v41_flash.tt.decode.moe import DeepSeekV41PreloadedExperts
from models.experimental.deepseek_v41_flash.tt.prefill.attention import PrefillShared
from models.experimental.deepseek_v41_flash.tt.prefill.decoder_layer import DeepSeekV41PrefillDecoderLayer
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir

PCC = 0.95
# The carried pre is ~sigmoid(bias) plus a small input-dependent part; bf16 alone costs ~1e-3.
PRE_MIX_ATOL = 0.01


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


def _to_device(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)
@pytest.mark.parametrize(
    "layers, seq_len, chunk",
    (
        ((0,), 128, 128),  # window only
        ((1,), 128, 128),  # window only, Engram
        ((2, 3), 128, 128),  # ratio 2: kv + index source, consumer
        ((14,), 256, 128),  # ratio 2 kv + index source, Engram, chunked
        ((20, 21, 24), 128, 128),  # ratio 1: source, consumer, index-only source
    ),
    ids=("window", "window_engram", "ratio2", "ratio2_engram_chunked", "ratio1"),
)
@torch.no_grad()
def test_decoder_layer_prefill(device, layers, seq_len, chunk):
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    hc = config.hc_mult
    model = load_reference(snapshot)

    cache = WeightCache(CACHE_DIR)
    layer_caches = {i: cache.sub(f"layers.{i}") for i in layers}
    experts = {
        i: DeepSeekV41PreloadedExperts(
            config, expert_provider(loader, i), device, dtype=ttnn.bfloat4_b, cache=layer_caches[i].sub("ffn")
        )
        for i in layers
    }
    # Built before the device layers for the rope tables they take; the reference runs only after the device has.
    reference = reference_blocks(model, snapshot, loader, layers, seq_len)
    decoder_layers = {
        i: DeepSeekV41PrefillDecoderLayer(
            config,
            i,
            decoder_layer_weights(loader, i, config),
            device,
            {
                ("main" if config.compress_ratios[i] == 0 else "compress"): (
                    reference[i].attn.freqs_cis.real,
                    reference[i].attn.freqs_cis.imag,
                )
            },
            experts[i],
            cache=layer_caches[i],
        )
        for i in layers
    }

    torch.manual_seed(0)
    streams = torch.randn(len(layers), seq_len, hc, config.hidden_size).bfloat16().float()
    pre_mixes = torch.rand(len(layers), seq_len, hc).bfloat16().float()
    rows = {
        i: torch.randn(seq_len, layer.engram.kin).bfloat16().float()
        for i, layer in decoder_layers.items()
        if layer.engram is not None
    }

    # Device: every layer over every chunk.
    states = {i: layer.new_state() for i, layer in decoder_layers.items()}
    actual = {i: ([], []) for i in layers}  # (streams, pre_mix) per chunk
    for start in range(0, seq_len, chunk):
        shared = PrefillShared()
        for j, i in enumerate(layers):
            layer = decoder_layers[i]
            out, out_pre = layer(
                _to_device(streams[j, start : start + chunk].unsqueeze(0), device),
                _to_device(pre_mixes[j, start : start + chunk].reshape(1, chunk, 1, hc), device),
                states[i],
                shared,
                engram_rows=(
                    _to_device(rows[i][start : start + chunk].reshape(1, 1, chunk, -1), device)
                    if layer.engram is not None
                    else None
                ),
            )
            actual[i][0].append(ttnn.to_torch(out).reshape(chunk, hc, -1).float())
            actual[i][1].append(ttnn.to_torch(out_pre).reshape(chunk, hc).float())
        if shared.entry_sel is not None:
            ttnn.deallocate(shared.entry_sel)

    # Reference: every layer over the whole prompt.
    for j, i in enumerate(layers):
        x = streams[j : j + 1]
        h = x
        if decoder_layers[i].engram is not None:
            engram = reference_engram_with_embed(
                model, snapshot, loader, i, GivenRows(rows[i].unsqueeze(0), config.engram_head_dim)
            )
            h = engram(x, torch.zeros(1, seq_len, 1, dtype=torch.long))  # the hash ids are ignored
        expected, expected_pre = reference[i](h, 0, pre_mixes[j : j + 1], None)

        out = torch.cat(actual[i][0]).reshape(expected.shape)
        out_pre = torch.cat(actual[i][1]).reshape(expected_pre.shape)
        pre_error = (out_pre - expected_pre).abs().max().item()
        passing, pcc = comp_pcc((expected - x).reshape(-1), (out - x).reshape(-1), PCC)
        logger.info(
            f"layer {i} (ratio {config.compress_ratios[i]}): streams (out - x) PCC {pcc}, pre_mix error {pre_error}"
        )
        assert passing, f"layer {i}: streams (out - x) PCC {pcc} < {PCC}"
        assert pre_error <= PRE_MIX_ATOL, f"layer {i}: pre_mix max abs error {pre_error} > {PRE_MIX_ATOL}"
