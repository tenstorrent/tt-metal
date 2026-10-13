# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill PCC of V4's ``DeepSeekV4PrefillHyperConnection`` with V4.1's ``pre_mix``, on real weights.

V4.1 collapses the streams with the ``pre`` the previous sublayer computed (``pre_mix``) and hands on the
``pre`` this hyper-connection computes, which the prefill block takes through the fused op's ``pre_mix``
argument. The reference is the checkpoint's ``Block.hc_mixes`` / ``hc_pre`` (``tests/reference.py``) run
over the same ``fn`` / ``base`` / ``scale``.

Each case checks ``post``, ``comb``, the own ``pre`` and ``collapsed``: with a ``pre_mix`` (V4.1) and without
one (V4: ``collapsed`` is weighted by the ``pre`` the call computes).

    pytest models/experimental/deepseek_v41_flash/tests/prefill/test_hyperconnection.py
"""

import types

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v41_flash.tests.reference import load_reference, model_args
from models.experimental.deepseek_v41_flash.tt.config import (
    CACHE_DIR,
    DEFAULT_MODEL_DIR,
    decoder_layer_weights,
    load_config,
)
from models.experimental.deepseek_v4_flash.tt.decode.decoder_layer import _strip_prefix
from models.experimental.deepseek_v4_flash.tt.prefill.hyperconnection import DeepSeekV4PrefillHyperConnection
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir

PCC = 0.999
# Token counts: a partial tile row of ``fused_w`` (40), a whole number of them (128), and enough to make
# every core of the op loop over several tokens (1024).
SEQ_LENS = (40, 128, 1024)


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


def _pcc(expected: torch.Tensor, actual: torch.Tensor, what: str) -> None:
    passing, pcc = comp_pcc(expected.reshape(-1), actual.reshape(-1).float(), PCC)
    logger.info(f"{what}: PCC {pcc}")
    assert passing, f"{what}: PCC {pcc} < {PCC}"


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize(
    "layer, which, cache_name",
    ((1, "attn_hc", "hc_attn"), (39, "ffn_hc", "hc_ffn")),
    ids=("layer1_attn", "layer39_ffn"),
)
@torch.no_grad()
def test_hyperconnection_prefill(device, layer, which, cache_name, seq_len):
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    weights = _strip_prefix(decoder_layer_weights(loader, layer, config), which)
    hc = DeepSeekV4PrefillHyperConnection(
        config, weights, device, cache=WeightCache(CACHE_DIR).sub(f"layers.{layer}.{cache_name}")
    )

    # ``Block.hc_mixes`` / ``hc_pre`` read only these attributes of their block.
    model = load_reference(snapshot)
    args = model_args(model, snapshot, max_seq=1)
    block = types.SimpleNamespace(
        norm_eps=args.norm_eps, hc_mult=args.hc_mult, hc_sinkhorn_iters=args.hc_sinkhorn_iters, hc_eps=args.hc_eps
    )
    fn, scale, base = (weights[k]().float() for k in ("fn", "scale", "base"))

    torch.manual_seed(seq_len)
    streams = torch.randn(1, seq_len, config.hc_mult, config.hidden_size).bfloat16().float()
    pre_mix = torch.rand(1, seq_len, config.hc_mult).bfloat16().float()
    expected_pre, expected_post, expected_comb = model.Block.hc_mixes(block, streams, fn, scale, base)

    def to_device(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # V4.1: collapse with the carried pre_mix, return this call's pre.
    post, comb, collapsed, pre = hc(to_device(streams), pre_mix=to_device(pre_mix.unsqueeze(2)))
    _pcc(expected_post, ttnn.to_torch(post), "post")
    _pcc(expected_comb, ttnn.to_torch(comb), "comb")
    _pcc(expected_pre, ttnn.to_torch(pre), "pre")
    _pcc(model.Block.hc_pre(block, streams, pre_mix), ttnn.to_torch(collapsed), "collapsed (pre_mix)")

    # V4: collapse with this call's own pre.
    post, comb, collapsed = hc(to_device(streams))
    _pcc(expected_post, ttnn.to_torch(post), "post (no pre_mix)")
    _pcc(model.Block.hc_pre(block, streams, expected_pre), ttnn.to_torch(collapsed), "collapsed (own pre)")
