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

``test_engram_decode_s5`` feeds one session's positions as ``S``-row blocks, streams ``[1, S, hc, D]``
(DSpark's parallel verify): one host lookup over the block's tokens, one layer call, each row checked
against the reference stepped one token at a time.

The converted ttnn weights are cached under ``CACHE_DIR`` (``$DEEPSEEK_V4_CACHE_DIR``, default ``../cache``, per checkpoint).

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_engram.py
"""

import contextlib
from concurrent.futures import ThreadPoolExecutor

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
from models.experimental.deepseek_v41_flash.tt.config import CACHE_DIR, DEFAULT_MODEL_DIR, engram_weights, load_config
from models.experimental.deepseek_v41_flash.tt.decode.engram import DeepSeekV41Engram
from models.experimental.deepseek_v41_flash.tt.engram_lookup import (
    EngramHostLookup,
    engram_ram_needed,
    load_engram_table,
)
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir
from tests.ttnn.unit_tests.operations.prefetcher_common import tensor_prefetcher_session

ENGRAM_LAYERS = (1, 14)  # inference/config.json engram_layer_ids; layer j of the tuple runs on device j
STEPS = 16
# DSpark's block: the token plus four drafts.
S = 5
NUM_BLOCKS = 3
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
    nbytes = engram_ram_needed(loader, ENGRAM_LAYERS)
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

    if not ttnn.experimental.is_tensor_prefetcher_supported(mesh_device):
        pytest.skip("Engram's wkv needs the tensor prefetcher (programmable DRAM cores)")
    cache = WeightCache(CACHE_DIR)
    # wkv's ring is private, so each device's mapping starts empty and the layer builds it.
    engrams = {
        i: DeepSeekV41Engram(
            config,
            i,
            engram_weights(loader, i),
            mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, j)),
            {},
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

    writer = ThreadPoolExecutor(max_workers=1)
    with contextlib.ExitStack() as sessions:
        # A failed step can leave a write blocked on the device for good; do not wait on it.
        sessions.callback(writer.shutdown, wait=False)
        for engram in engrams.values():
            sessions.enter_context(tensor_prefetcher_session(engram.device))
            ttnn.experimental.wait_for_cq_on_tensor_prefetcher(engram.device, cq_id=0)

        for pos in range(STEPS):
            token = input_ids[:, pos : pos + 1]
            rows = lookup(token, pos)  # {layer: [B, 1, n_hash_cols * head_dim]}
            hashes = reference_hash(token, pos)  # [B, 1, n_engram_layers, n_hash_cols]

            # Each layer's recv_async_h2d stalls the device queue until its rows arrive, and a step's rows can
            # outnumber the socket's pages, so they go in from another thread while the step is dispatched.
            outs, writes = {}, []
            for j, (i, engram) in enumerate(engrams.items()):
                writes.append(writer.submit(engram.write_rows, rows[i]))
                engram.wkv.fetch_weights()
                x = ttnn.from_torch(
                    streams[j, pos],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=engram.device,
                    memory_config=engram.streams_memory_config,
                )
                outs[i] = engram(x)
            for write in writes:
                write.result()

            for j, i in enumerate(ENGRAM_LAYERS):
                layer_hashes = hashes[:, :, reference_hash.layout.layer_ids.index(i)]
                expected_rows = reference[i].embed.module(layer_hashes).flatten(-2)
                assert torch.equal(
                    rows[i], expected_rows
                ), f"layer {i} pos {pos}: host rows differ from the reference's"

                x = streams[j, pos]
                expected = reference[i](x, layer_hashes)
                actual = ttnn.to_torch(outs[i], mesh_composer=ttnn.ConcatMeshToTensor(engrams[i].device, dim=0)).float()
                assert actual.shape == expected.shape, (tuple(actual.shape), tuple(expected.shape))
                what = f"layer {i} pos {pos} batch {batch}"
                _pcc(expected, actual, PCC, what)
                _pcc(expected - x, actual - x, DELTA_PCC, f"{what} (out - x)")


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@torch.no_grad()
def test_engram_decode_s5(mesh_device, tables):
    snapshot = _snapshot()
    config = load_config(snapshot)
    assert tuple(config.engram_layer_ids) == ENGRAM_LAYERS
    loader = DeepseekV4WeightLoader(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = load_reference(snapshot)
    tokens = NUM_BLOCKS * S
    reference_hash = reference_engram_hash(model, snapshot, tokenizer, tokens)
    reference = {i: reference_engram(model, snapshot, loader, i, *tables[i]) for i in ENGRAM_LAYERS}
    lookup = EngramHostLookup(config, snapshot, tables, 1, tokens, tokenizer)

    if not ttnn.experimental.is_tensor_prefetcher_supported(mesh_device):
        pytest.skip("Engram's wkv needs the tensor prefetcher (programmable DRAM cores)")
    cache = WeightCache(CACHE_DIR)
    engrams = {
        i: DeepSeekV41Engram(
            config,
            i,
            engram_weights(loader, i),
            mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, j)),
            {},
            cache=cache.sub(f"layers.{i}.engram"),
            verify_rows=S,
        )
        for j, i in enumerate(ENGRAM_LAYERS)
    }

    ids = tokenizer.encode(PROMPT)
    assert len(ids) >= tokens, f"PROMPT has {len(ids)} tokens"
    input_ids = torch.tensor([ids[:tokens]])
    hc, hidden = config.hc_mult, config.hidden_size
    torch.manual_seed(0)
    streams = torch.randn(len(ENGRAM_LAYERS), tokens, hc, hidden).bfloat16().float()

    writer = ThreadPoolExecutor(max_workers=1)
    with contextlib.ExitStack() as sessions:
        # A failed step can leave a write blocked on the device for good; do not wait on it.
        sessions.callback(writer.shutdown, wait=False)
        for engram in engrams.values():
            sessions.enter_context(tensor_prefetcher_session(engram.device))
            ttnn.experimental.wait_for_cq_on_tensor_prefetcher(engram.device, cq_id=0)

        for start in range(0, tokens, S):
            positions = range(start, start + S)
            # One lookup over the block, as a verify step hashes its run of one session.
            rows = lookup(input_ids[:, start : start + S], start)  # {layer: [1, S, n_hash_cols * head_dim]}
            # The reference hashes token by token, as S single-row steps would.
            hashes = [reference_hash(input_ids[:, pos : pos + 1], pos) for pos in positions]

            # The block's rows outnumber the socket's pages, so they go in while the step is dispatched.
            outs, writes = {}, []
            for j, (i, engram) in enumerate(engrams.items()):
                writes.append(writer.submit(engram.write_rows, rows[i]))
                engram.wkv.fetch_weights()
                x = ttnn.from_torch(
                    streams[j, start : start + S].reshape(1, S, hc, hidden),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=engram.device,
                    memory_config=engram.verify_streams_memory_config,
                )
                outs[i] = engram(x)
            for write in writes:
                write.result()

            for j, i in enumerate(ENGRAM_LAYERS):
                actual = ttnn.to_torch(outs[i], mesh_composer=ttnn.ConcatMeshToTensor(engrams[i].device, dim=0)).float()
                assert tuple(actual.shape) == (1, S, hc, hidden), tuple(actual.shape)
                for k, pos in enumerate(positions):
                    layer_hashes = hashes[k][:, :, reference_hash.layout.layer_ids.index(i)]
                    expected_rows = reference[i].embed.module(layer_hashes).flatten(-2)
                    assert torch.equal(
                        rows[i][:, k : k + 1], expected_rows
                    ), f"layer {i} pos {pos} (row {k}): host rows differ from the reference's"

                    x = streams[j, pos].reshape(1, 1, hc, hidden)
                    expected = reference[i](x, layer_hashes)
                    got = actual[:, k : k + 1]
                    what = f"layer {i} pos {pos} (row {k} of the block at {start})"
                    _pcc(expected, got, PCC, what)
                    _pcc(expected - x, got - x, DELTA_PCC, f"{what} (out - x)")
