# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Decode PCC of ``DeepSeekV41Engram`` -- lookup on the host, the rest on device -- against the checkpoint's ``Engram``.

The two Engram layers (1 and 14) run on two devices, the halves of a 1x2 mesh, and each receives its
rows over its own H2D socket. Both n-gram tables are read into host RAM (~203 GB; the test skips
without 1.25x that free), and ``EngramHostLookup`` does token -> n-gram hash -> fp8 rows -> bf16. The
device does ``wkv``, the gate and the residual write. Each step feeds every user one token at its
position, as decode does, with random residual streams.

The rows are checked bit-exact against the reference's own embedding; the layer output by PCC, on the
whole output and on what the Engram adds to the streams (``out - x``).

Set ``DEEPSEEK_V41_CACHE_DIR`` to keep the converted ttnn weights across runs.

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_engram.py
"""

import math
import os

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v41_flash.tests.reference import (
    load_reference,
    reference_engram,
    reference_engram_hash,
)
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR, engram_weights, load_config
from models.experimental.deepseek_v41_flash.tt.decode.engram import DeepSeekV41Engram
from models.experimental.deepseek_v41_flash.tt.engram_lookup import EngramHostLookup, load_engram_table, tensor_region
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir

ENGRAM_LAYERS = (1, 14)  # inference/config.json engram_layer_ids; layer j of the tuple runs on device j
STEPS = 16
PCC = 0.999
DELTA_PCC = 0.98
PROMPT = (
    "Engram adds a learned memory to the residual stream at two layers of the network. For every position, "
    "the last few tokens are normalized and hashed into one bucket per head for each n-gram size from two to four."
)


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


def _available_bytes():
    with open("/proc/meminfo") as f:
        return next(int(line.split()[1]) * 1024 for line in f if line.startswith("MemAvailable:"))


@pytest.fixture(scope="module")
def tables():
    """Both Engram layers' tables in host RAM, shared by every case."""
    loader = DeepseekV4WeightLoader(_snapshot())
    names = [f"layers.{i}.engram.embed.{part}" for i in ENGRAM_LAYERS for part in ("weight", "scale")]
    nbytes = sum(math.prod(tensor_region(loader.shard_of(n, translate=False), n)[1]) for n in names)  # 1 B / value
    if _available_bytes() < 1.25 * nbytes:
        pytest.skip(f"needs {nbytes / 1e9:.0f} GB of free memory for the Engram tables")
    return {i: load_engram_table(loader, i) for i in ENGRAM_LAYERS}


def _pcc(expected, actual, threshold, what):
    passing, pcc = comp_pcc(expected.reshape(-1), actual.reshape(-1), threshold)
    logger.info(f"{what}: PCC {pcc}")
    assert passing, f"{what}: PCC {pcc} < {threshold}"


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("batch", (1, 4))
@torch.no_grad()
def test_engram_decode(mesh_device, tables, batch):
    snapshot = _snapshot()
    config = load_config(snapshot)
    assert tuple(config.engram_layer_ids) == ENGRAM_LAYERS
    loader = DeepseekV4WeightLoader(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = load_reference(snapshot)
    reference_hash = reference_engram_hash(model, snapshot, tokenizer, STEPS, batch)
    reference = {i: reference_engram(model, snapshot, loader, i, *tables[i]) for i in ENGRAM_LAYERS}
    lookup = EngramHostLookup(config, snapshot, tables, batch, STEPS, tokenizer)

    cache = WeightCache(os.environ.get("DEEPSEEK_V41_CACHE_DIR"))
    engrams = {
        i: DeepSeekV41Engram(
            config,
            i,
            engram_weights(loader, i),
            mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, j)),
            batch=batch,
            cache=cache.sub(f"layers.{i}.engram"),
        )
        for j, i in enumerate(ENGRAM_LAYERS)
    }

    # A different stretch of the prompt per user, so their n-gram histories differ.
    ids = tokenizer.encode(PROMPT)
    assert len(ids) >= 3 * (batch - 1) + STEPS, f"PROMPT has {len(ids)} tokens"
    input_ids = torch.tensor([ids[3 * u : 3 * u + STEPS] for u in range(batch)])
    torch.manual_seed(0)
    streams = torch.randn(len(ENGRAM_LAYERS), STEPS, batch, 1, config.hc_mult, config.hidden_size).bfloat16().float()

    for pos in range(STEPS):
        token = input_ids[:, pos : pos + 1]
        rows = lookup(token, pos)  # {layer: [B, 1, n_hash_cols * head_dim]}
        hashes = reference_hash(token, pos)  # [B, 1, n_engram_layers, n_hash_cols]

        # Dispatch both layers first: each blocks in recv_async_h2d until its rows arrive.
        outs = {}
        for j, (i, engram) in enumerate(engrams.items()):
            x = ttnn.from_torch(streams[j, pos], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=engram.device)
            outs[i] = engram(x)
        for i, engram in engrams.items():
            engram.write_rows(rows[i])

        for j, i in enumerate(ENGRAM_LAYERS):
            layer_hashes = hashes[:, :, reference_hash.layout.layer_ids.index(i)]
            expected_rows = reference[i].embed.module(layer_hashes).flatten(-2)
            assert torch.equal(rows[i], expected_rows), f"layer {i} pos {pos}: host rows differ from the reference's"

            x = streams[j, pos]
            expected = reference[i](x, layer_hashes)
            actual = ttnn.to_torch(outs[i], mesh_composer=ttnn.ConcatMeshToTensor(engrams[i].device, dim=0)).float()
            assert actual.shape == expected.shape, (tuple(actual.shape), tuple(expected.shape))
            what = f"layer {i} pos {pos} batch {batch}"
            _pcc(expected, actual, PCC, what)
            _pcc(expected - x, actual - x, DELTA_PCC, f"{what} (out - x)")
