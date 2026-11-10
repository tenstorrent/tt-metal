# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Device-op profile of one ``DeepSeekV41PrefillDecoderLayer`` over one prefill chunk, on real weights.

The device half of :mod:`.test_decoder_layer` only: no reference block is built (just its attention, for the
rope tables) and nothing is compared. A warm-up chunk fills the program cache; the profiled chunk, run on a fresh
state, sits between ``PREFILL_LAYER_START`` / ``PREFILL_LAYER_END`` signposts.

    python -m tracy -p -r -v -m pytest models/experimental/deepseek_v41_flash/tests/prefill/test_decoder_layer_profile.py
"""

import pytest
import torch
from tracy import signpost

import ttnn
from models.experimental.deepseek_v41_flash.tests.reference import load_reference, reference_attentions
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


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


def _to_device(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)
@pytest.mark.parametrize("chunk", (1024,))
@pytest.mark.parametrize(
    "layer",
    (0, 1, 2, 20),
    ids=("window", "window_engram", "ratio2_source", "ratio1_source"),
)
@torch.no_grad()
def test_decoder_layer_prefill_profile(device, layer, chunk):
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    hc = config.hc_mult
    model = load_reference(snapshot)

    layer_cache = WeightCache(CACHE_DIR).sub(f"layers.{layer}")
    experts = DeepSeekV41PreloadedExperts(
        config, expert_provider(loader, layer), device, dtype=ttnn.bfloat4_b, cache=layer_cache.sub("ffn")
    )
    freqs_cis = reference_attentions(model, snapshot, loader, (layer,), chunk)[layer].freqs_cis
    decoder_layer = DeepSeekV41PrefillDecoderLayer(
        config,
        layer,
        decoder_layer_weights(loader, layer, config),
        device,
        {("main" if config.compress_ratios[layer] == 0 else "compress"): (freqs_cis.real, freqs_cis.imag)},
        experts,
        cache=layer_cache,
    )

    torch.manual_seed(0)
    streams = _to_device(torch.randn(1, chunk, hc, config.hidden_size), device)
    pre_mix = _to_device(torch.rand(1, chunk, 1, hc), device)
    engram_rows = None
    if decoder_layer.engram is not None:
        engram_rows = _to_device(torch.randn(1, 1, chunk, decoder_layer.engram.kin), device)

    for profiled in (False, True):
        state, shared = decoder_layer.new_state(), PrefillShared()
        ttnn.synchronize_device(device)
        if profiled:
            signpost("PREFILL_LAYER_START")
        out, out_pre = decoder_layer(streams, pre_mix, state, shared, engram_rows=engram_rows)
        ttnn.synchronize_device(device)
        if profiled:
            signpost("PREFILL_LAYER_END")
        for tensor in (out, out_pre, shared.entry_idx, shared.candidates):
            if tensor is not None:
                ttnn.deallocate(tensor)
