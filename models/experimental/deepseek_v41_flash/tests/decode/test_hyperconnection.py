# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Decode PCC of V4's ``DeepSeekV4HyperConnection(row_major_streams=True)`` on ``S``-row blocks, on real weights.

DSpark's parallel verify steps ``S`` consecutive positions of one session at once, so the decoder layer's
streams arrive as ``[1, S, hc, D]`` ROW_MAJOR, width-sharded ``[S * hc, D / 8]`` over the 8 collapse cores.
``test_hyperconnection_decode_s5`` checks ``post``, ``comb``, the own ``pre`` and the ``pre_mix`` collapse
against the checkpoint's ``Block.hc_mixes`` / ``hc_pre`` (``tests/reference.py``), and against the same rows
stepped one at a time through the single-row path.

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_hyperconnection.py
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
from models.experimental.deepseek_v4_flash.tt.common import width_sharded_l1_config
from models.experimental.deepseek_v4_flash.tt.decode.decoder_layer import _strip_prefix
from models.experimental.deepseek_v4_flash.tt.decode.hyperconnection import (
    _HC_COLLAPSE_CORES,
    DeepSeekV4HyperConnection,
)
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir

PCC = 0.999
# DSpark's block: the token plus four drafts.
S = 5


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
@pytest.mark.parametrize(
    "layer, which, cache_name",
    ((1, "attn_hc", "hc_attn"), (39, "ffn_hc", "hc_ffn")),
    ids=("layer1_attn", "layer39_ffn"),
)
@torch.no_grad()
def test_hyperconnection_decode_s5(device, layer, which, cache_name):
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    weights = _strip_prefix(decoder_layer_weights(loader, layer, config), which)
    hc_module = DeepSeekV4HyperConnection(
        config,
        weights,
        device,
        cache=WeightCache(CACHE_DIR).sub(f"layers.{layer}.{cache_name}"),
        row_major_streams=True,
    )

    # ``Block.hc_mixes`` / ``hc_pre`` read only these attributes of their block.
    model = load_reference(snapshot)
    args = model_args(model, snapshot, max_seq=1)
    block = types.SimpleNamespace(
        norm_eps=args.norm_eps, hc_mult=args.hc_mult, hc_sinkhorn_iters=args.hc_sinkhorn_iters, hc_eps=args.hc_eps
    )
    fn, scale, base = (weights[k]().float() for k in ("fn", "scale", "base"))

    hc, d = config.hc_mult, config.hidden_size
    torch.manual_seed(layer)
    streams = torch.randn(1, S, hc, d).bfloat16().float()
    pre_mix = torch.rand(1, S, 1, hc).bfloat16().float()
    expected_pre, expected_post, expected_comb = model.Block.hc_mixes(block, streams, fn, scale, base)
    expected_collapsed = model.Block.hc_pre(block, streams, pre_mix.reshape(1, S, hc))

    def streams_to_device(x):
        rows = x.shape[1]
        return ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=width_sharded_l1_config(rows * hc, d, device, num_cores=_HC_COLLAPSE_CORES, tile_height=1),
        )

    def pre_mix_to_device(x):
        return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    names = ("post", "comb", "collapsed", "pre")
    outputs = hc_module(streams_to_device(streams), pre_mix=pre_mix_to_device(pre_mix))
    shapes = ([1, S, hc, 1], [1, S, hc, hc], [1, S, 1, d], [1, S, 1, hc])
    for name, x, shape in zip(names, outputs, shapes):
        assert list(x.shape) == shape, f"{name}: shape {list(x.shape)}, expected {shape}"
    got = {name: ttnn.to_torch(x).float() for name, x in zip(names, outputs)}
    _pcc(expected_post, got["post"], "post")
    _pcc(expected_comb, got["comb"], "comb")
    _pcc(expected_pre, got["pre"], "pre")
    _pcc(expected_collapsed, got["collapsed"], "collapsed (pre_mix)")

    # The same rows stepped one at a time through the single-row path.
    steps = [
        hc_module(streams_to_device(streams[:, j : j + 1]), pre_mix=pre_mix_to_device(pre_mix[:, j : j + 1]))
        for j in range(S)
    ]
    for k, name in enumerate(names):
        single = torch.cat([ttnn.to_torch(step[k]).float().reshape(1, 1, -1) for step in steps], dim=1)
        _pcc(single, got[name], f"{name} vs {S} single-row steps")
