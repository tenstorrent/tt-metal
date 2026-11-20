# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host arithmetic only (run it under the mock Galaxy): what the prepared two-term streams of fused/moe/prepare.py
hold, against "pre-rounded" terms (the stream given q1 = BFP4(w) and BFP4(w - q1) instead of w and w - q1), and
against what a tensor created directly with the sharded memory config would hold (BF16 first, then BFP4:
layout.to_device_stream without host_round).

prepare.py packs its streams on the host (prepare.host_stream: no memory config, one rounding) and moves the result
to the device, so its two terms are already BFP4(w) and BFP4(w - BFP4(w)); and the checkpoint's weights are BF16, so
w and w - BFP4(w) are BF16 values and the BF16 hop of a sharded tensor would not change them either. This test
shows both on the shared expert of layer 1 and the dense FFN of layer 0 (chip $HY4_FID_CHIP's share), and that the
stream file on disk holds these values. Needs $HY4_HF_MODEL. Log lines start with HY4_MK9."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.moe import layout as L
from models.experimental.hy4_preview.fused.moe import prepare

CHIP = int(os.environ.get("HY4_FID_CHIP", "13"))
STREAMS = os.environ.get("HY4_MOE_STREAMS", prepare.DEFAULT_OUT)


def round_tiles(w):
    return L.bfp4_round(w.transpose(1, 2)).transpose(1, 2)


def stored(stream):
    return ttnn.to_torch(prepare.host_stream(stream)).float().reshape(stream.shape)


def stored_sharded(stream):
    return stored(stream.to(torch.bfloat16).float())


@pytest.mark.parametrize("what", ["shared_layer1", "dense_layer0"])
def test_prerounded_host(what):
    if not os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"):
        pytest.skip("host arithmetic: run under the mock Galaxy")
    if not os.environ.get("HY4_HF_MODEL"):
        pytest.skip("HY4_HF_MODEL is not set")
    share = L.tp_share(32)
    if what == "shared_layer1":
        layer, first = 1, 256
        g, u, d = prepare.checkpoint_tensors(
            [f"model.layers.1.mlp.shared_experts.{n}_proj.weight" for n in ("gate", "up", "down")]
        )
        gate_up, down = torch.cat((g, u), dim=0)[None].float(), d[None].float()
    else:
        layer, first = 0, 0
        gate_up, down = prepare.dense_experts(0, terms=1)
        gate_up, down = gate_up.float(), down.float()
    assert torch.equal(gate_up, gate_up.to(torch.bfloat16).float()), "the checkpoint's weights are BF16"

    def streams(terms):
        gu, dn = terms(gate_up), terms(down)
        gate, up, dnc = prepare.chip_weights(gu, dn, "tp32", CHIP)
        return L.gate_up_stream(gate, up, share), L.down_stream(dnc, share)

    as_prepared = streams(L.two_terms)

    def pre_rounded(w):
        q1 = round_tiles(w)
        return torch.stack((q1, round_tiles(w - q1)), dim=1).reshape(2 * w.shape[0], *w.shape[1:])

    rounded = streams(pre_rounded)
    for name, a, b in zip(("gate|up", "down"), as_prepared, rounded):
        held, held_sharded, held_rounded = stored(a), stored_sharded(a), stored(b)
        same = torch.equal(held, held_rounded)
        sharded_same = torch.equal(held_sharded, held)
        residual_bf16 = torch.equal(a, a.to(torch.bfloat16).float())
        logger.info(
            f"HY4_MK9 two-term streams {what} chip {CHIP} {name}: the prepared stream holds the pre-rounded terms "
            f"in every entry: {same}; a sharded tensor of the same floats would hold the same: {sharded_same}; "
            f"the floats given to the stream are BF16 values: {residual_bf16} ({a.numel()} values)"
        )
        assert same and sharded_same
    # the file on disk
    paths = prepare.stream_paths(STREAMS, layer, "tp32", CHIP)
    if paths[0].is_file():
        n_rows = as_prepared[0].shape[0]
        for name, path, a in zip(("gate|up", "down"), paths, as_prepared):
            disk = ttnn.to_torch(ttnn.load_tensor(str(path))).float().reshape(-1, a.shape[1])
            rows = a.shape[0]
            stride = share.gate_up_stride if name == "gate|up" else share.down_stride
            row0 = first * stride // L.BFP4_TILE_BYTES * L.TILE
            equal = torch.equal(disk[row0 : row0 + rows], stored(a))
            logger.info(f"HY4_MK9 two-term streams {what} chip {CHIP} {name}: {path.name} holds these values: {equal}")
            assert equal
