# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host timing of the Engram lookup -- token -> n-gram hash -> fp8 table rows -- through the checkpoint's own code.

Each step feeds one token at its position, as decode does: ``NgramHashState`` hashes it together with
its history, and layer ``layer_id``'s ``ParallelEngramEmbedding`` gathers the 24 fp8 E4M3 rows and E8M0
scales those hashes name and dequantizes them to bf16. The ~100 GB table is either read into memory
(``ram``, as the model holds it) or left memory-mapped in the safetensors file (``disk``: the pages the
run will read are dropped from the page cache first, so every lookup goes to the drive).

No device is opened. The ``ram`` case needs ~101 GB of free memory and skips otherwise.

    pytest models/experimental/deepseek_v41_flash/tests/test_engram_lookup.py
"""

import math
import mmap
import os
import time
import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from safetensors import safe_open
from transformers import AutoTokenizer

from models.experimental.deepseek_v41_flash.tests.reference import (
    load_reference,
    reference_engram_embedding,
    reference_engram_hash,
)
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR
from models.experimental.deepseek_v41_flash.tt.engram_lookup import read_into_memory, tensor_region
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir

ITERATIONS = 100
ENGRAM_LAYERS = (1, 14)  # inference/config.json engram_layer_ids
PROMPT = (
    "Engram adds a learned memory to the residual stream at two layers of the network. For every position, "
    "the last few tokens are normalized, so that case and accents do not matter, and hashed into one bucket per "
    "head for each n-gram size from two to four. Each bucket names a row of a very large table that is stored in "
    "eight bit floating point with one shared exponent per block of thirty two values. The rows are projected into "
    "a key for every hyper-connection copy and one shared value, and a gate that compares the key with the stream "
    "decides how much of that value is written back. Because the lookup depends only on token ids, it can run on "
    "the host while the device works on earlier layers."
)


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


def _mapped(path, offset, shape, dtype):
    """The tensor at ``offset`` as a read-only memory map of ``path``: every first touch of a row reads the file."""
    nbytes = math.prod(shape) * dtype.itemsize
    start = offset // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
    with open(path, "rb") as f:
        mapping = mmap.mmap(f.fileno(), offset - start + nbytes, access=mmap.ACCESS_READ, offset=start)
    mapping.madvise(mmap.MADV_RANDOM)  # a lookup touches one row: no read-around of neighbouring pages
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # torch warns that the mapping is not writable
        return torch.from_numpy(np.frombuffer(mapping, np.uint8, nbytes, offset - start)).view(dtype).reshape(shape)


def _drop_from_page_cache(path, offsets, nbytes):
    """Evict the pages holding ``nbytes`` at each of ``offsets`` from the page cache (pages mapped by a running
    process stay). ``POSIX_FADV_DONTNEED`` only drops whole pages, so each range is widened to page bounds."""
    page = mmap.PAGESIZE
    fd = os.open(path, os.O_RDONLY)
    try:
        for offset in offsets:
            start = offset // page * page
            os.posix_fadvise(fd, start, -(-(offset + nbytes) // page) * page - start, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _available_bytes():
    with open("/proc/meminfo") as f:
        return next(int(line.split()[1]) * 1024 for line in f if line.startswith("MemAvailable:"))


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("layer_id", ENGRAM_LAYERS, ids=lambda i: f"layer{i}")
@pytest.mark.parametrize("storage", ("ram", "disk"))
@torch.no_grad()
def test_engram_lookup(storage, layer_id):
    snapshot = _snapshot()
    model = load_reference(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    input_ids = torch.tensor([tokenizer.encode(PROMPT)])[:, :ITERATIONS]
    assert input_ids.shape[1] == ITERATIONS, f"PROMPT has {input_ids.shape[1]} tokens, need {ITERATIONS}"

    hash_state = reference_engram_hash(model, snapshot, tokenizer, ITERATIONS)
    layout = hash_state.layout
    assert layout.layer_ids == ENGRAM_LAYERS
    layer = layout.layer_ids.index(layer_id)
    rows = layout.num_embeddings[layer]

    # The whole prompt in one call, as prefill hashes it: what the per-token steps below must reproduce.
    expected = hash_state(input_ids, 0)[:, :, layer]  # [1, ITERATIONS, 24]
    low = hash_state.offsets[layer]
    high = low + hash_state.primes[layer].reshape(-1)
    assert high[-1] == rows, "the 24 bucket ranges tile the table"
    assert ((expected >= low) & (expected < high)).all(), "every hash lands in its own column's bucket range"

    loader = DeepseekV4WeightLoader(snapshot)
    names = {part: f"layers.{layer_id}.engram.embed.{part}" for part in ("weight", "scale")}
    path = loader.shard_of(names["weight"], translate=False)
    assert loader.shard_of(names["scale"], translate=False) == path
    (w_offset, w_shape), (s_offset, s_shape) = (tensor_region(path, names[part]) for part in ("weight", "scale"))
    dtypes = {"weight": torch.float8_e4m3fn, "scale": torch.float8_e8m0fnu}
    row_bytes = {part: shape[1] for part, shape in (("weight", w_shape), ("scale", s_shape))}

    start = time.perf_counter()
    if storage == "ram":
        table_bytes = sum(math.prod(shape) for shape in (w_shape, s_shape))
        if _available_bytes() < 1.25 * table_bytes:
            pytest.skip(f"needs {table_bytes / 1e9:.0f} GB of free memory for the table")
        weight = read_into_memory(path, w_offset, w_shape, dtypes["weight"])
        scale = read_into_memory(path, s_offset, s_shape, dtypes["scale"])
    else:
        weight = _mapped(path, w_offset, w_shape, dtypes["weight"])
        scale = _mapped(path, s_offset, s_shape, dtypes["scale"])
        ids = expected.reshape(-1).tolist()
        _drop_from_page_cache(path, (w_offset + i * row_bytes["weight"] for i in ids), row_bytes["weight"])
        _drop_from_page_cache(path, (s_offset + i * row_bytes["scale"] for i in ids), row_bytes["scale"])
    logger.info(f"layer {layer_id} table ({rows} rows) in {storage}: ready in {time.perf_counter() - start:.1f} s")
    embed = reference_engram_embedding(model, layout, layer_id, weight, scale)

    hash_ns, lookup_ns, looked_up = [], [], []
    for pos in range(ITERATIONS):
        t0 = time.perf_counter_ns()
        hashes = hash_state(input_ids[:, pos : pos + 1], pos)[:, :, layer]
        t1 = time.perf_counter_ns()
        values = embed(hashes)
        t2 = time.perf_counter_ns()
        hash_ns.append(t1 - t0)
        lookup_ns.append(t2 - t1)
        assert torch.equal(hashes, expected[:, pos : pos + 1]), f"pos {pos}: decode hash differs from prefill hash"
        assert values.shape == (1, 1, hashes.shape[-1], layout.head_dim) and values.dtype == torch.bfloat16
        looked_up.append(values)

    # Untimed: the fp8 bytes the lookup read are the checkpoint's, and its bf16 output is their dequantization.
    ids = expected.reshape(-1)
    with safe_open(str(path), framework="pt") as f:
        stored = {part: f.get_slice(names[part]) for part in names}
        stored = {part: torch.cat([s[i : i + 1] for i in ids.tolist()]) for part, s in stored.items()}
    for part, table in (("weight", embed.weight), ("scale", embed.scale)):
        assert stored[part].dtype == dtypes[part] and table.dtype == dtypes[part]
        gathered = F.embedding(ids, table)
        assert torch.equal(gathered.view(torch.uint8), stored[part].view(torch.uint8)), f"{part} rows differ"
    block = layout.head_dim // stored["scale"].shape[-1]
    dequantized = stored["weight"].float().unflatten(-1, (-1, block)) * stored["scale"].float().unsqueeze(-1)
    assert torch.equal(torch.cat(looked_up).reshape(ids.numel(), -1), dequantized.flatten(-2).to(torch.bfloat16))

    hash_ms, lookup_ms = np.array(hash_ns) / 1e6, np.array(lookup_ns) / 1e6
    total_ms = hash_ms + lookup_ms
    logger.info(
        f"layer {layer_id}, table in {storage}, {ITERATIONS} tokens ({ids.unique().numel()} distinct rows): "
        f"mean per token -- hash {hash_ms.mean():.3f} ms, lookup {lookup_ms.mean():.3f} ms, "
        f"total {total_ms.mean():.3f} ms (median {np.median(total_ms):.3f}, max {total_ms.max():.3f})"
    )
