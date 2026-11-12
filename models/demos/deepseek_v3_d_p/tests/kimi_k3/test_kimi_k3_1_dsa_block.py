# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3.1's Gated DSA layers run inside the Kimi-K3 stack: finite and deterministic on 8x4.

K3.1 is K3 with its 24 full-attention layers turned into Gated DSA (`ttMLA` with K3.1's NoPE
indexer, scaled-FP8 KVPE cache and a bf8 indexer key cache). `ttMLA` itself is covered against the
CPU truth by `sparse_mla/test_sparse_mla.py`; this test covers the seam K3.1 adds to the K3 stack:
`TtKimiK3Transformer` threading the sparse cache format and the index cache through
`TtKimiK3Block` to `TtK3MlaAttention`, in the rank-local slot space K3's schedule assigns.

There is no K3.1 CPU reference or golden, so the bar is the K3 layer tests' structural one: two
5120-token chunks of chunked prefill (the second chunk's indexer selects its top-2048 from a 10240
token prefix and reads the first chunk's cache), every output and cache finite, and a second run of
the same sequence bit-identical to the first. In the DSA-only window, chunk 1's output must change when only
chunk 0's tokens do: the stack-level proof that a later chunk reads the earlier chunks' kvpe cache. It does
not prove the index-key read (an indexer ignoring its cached keys would still pass); that is left to
`sparse_mla/test_sparse_mla.py`'s chunked kimi_k3_1 case against the CPU reference (CI: the Kimi-K3 accuracy
suite, `k3_accuracy_suite` section [5/5] `k31_dsa_mla`, in tests/pipeline_reorg/blaze_models_prefill_tests.yaml).

Synthetic weights from `synthetic.py`, with its one substitution: every layer's FFN is K3's dense FFN
(a synthetic LatentMoE layer is ~59 GB; the MoE is unchanged from K3 and covered by K3's tests).

Windows: the first DSA layer alone (layer 3, a first rank with its embedding); the same layer as a
pipeline stage's KV-only last layer (layer 92's role); layers 3-7, two DSA layers in KV slots 0 and 1
of both caches, so a slot collision leaves a zero row; and layers 0-3, so the KDA carries and the DSA
index cache meet in one forward.

The caches and KDA slabs come from the serving allocator (`KimiK31Adapter.allocate_kv_cache`: TP-deduped
scaled-FP8 KVPE and bf8 index-key caches, one slot per DSA layer of the rank). The capacity must be a
multiple of 8192 tokens (the allocator's paging alignment) and of the chunk (the cache write), so the two
chunks fill 10240 of a 40960-token cache.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.kimi_k3.synthetic import (
    CACHE_LEN,
    CHUNK,
    HF_CONFIGS,
    build_stack,
    compose,
    placements,
    release,
    synthetic_state_dict,
    token_ids,
    upload_tokens,
)

NUM_CHUNKS = 2
# (first_layer_idx, num_layers, kv_only_last_layer)
WINDOWS = {
    "dsa_L3": (3, 1, False),
    "dsa_L3_kv_only": (3, 1, True),
    "dsa_L3_L7": (3, 5, False),
    "hybrid_L0_3": (0, 4, False),
}


def _shards(mesh_device, tensor) -> torch.Tensor:
    """Every device's shard, as stored (no dtype conversion): what bit-identity compares."""
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape))
    return ttnn.to_torch(tensor, mesh_composer=composer)


def _run_sequence(model, mesh_device, ids, kvpe, index) -> list[torch.Tensor | None]:
    outputs = []
    for chunk in range(NUM_CHUNKS):
        start = chunk * CHUNK
        tokens_tt = upload_tokens(mesh_device, ids[start : start + CHUNK])
        logger.info(f"chunk {chunk}: forward at actual_start={start}")
        out = model.forward(
            tokens_tt,
            kvpe_cache=kvpe,
            index_kv_cache=index,
            actual_start=start,
            actual_end=start + CHUNK,
        )
        ttnn.deallocate(tokens_tt)
        if out is None:
            outputs.append(None)
        else:
            outputs.append(compose(mesh_device, out, CHUNK))
            ttnn.deallocate(out)
    return outputs


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "model, mesh_device, device_params", placements(["kimi_k3_1"]), indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("window", list(WINDOWS), ids=list(WINDOWS))
def test_k31_dsa_block_finite_and_deterministic(model, mesh_device, device_params, window):
    first_layer_idx, num_layers, kv_only = WINDOWS[window]
    logger.info(f"{window}: building synthetic weights for layers [{first_layer_idx}, {first_layer_idx + num_layers})")
    state_dict = synthetic_state_dict(HF_CONFIGS[model](CACHE_LEN), first_layer_idx, num_layers)
    logger.info(f"{window}: building the model")
    stack, caches = build_stack(mesh_device, model, state_dict, first_layer_idx, num_layers, kv_only_last_layer=kv_only)
    dsa_layers = [layer for layer in stack.layers if layer.attention.writes_kv]
    assert dsa_layers and all(layer.attention.mla._has_indexer for layer in dsa_layers), "K3.1 layers must be DSA"
    kvpe, index = caches.kvpe, caches.index

    ids = token_ids(NUM_CHUNKS * CHUNK)
    runs = []
    try:
        for run in range(2):
            logger.info(f"{window}: run {run}")
            outputs = _run_sequence(stack, mesh_device, ids, kvpe, index)
            runs.append((outputs, _shards(mesh_device, kvpe.storage), _shards(mesh_device, index)))
        # Prefix dependence, where nothing but the caches crosses a chunk: a window of DSA layers only (no
        # KDA carry; AttnRes opens a fresh walk per chunk). Change only chunk 0's tokens: chunk 1's output
        # can move only through its attention reading chunk 0 from the kvpe cache (the index-key read is not
        # proven here; see the module docstring).
        reads_prefix_only_from_caches = not kv_only and not stack.schedule.rank_kda_layer_ids()
        if reads_prefix_only_from_caches:
            logger.info(f"{window}: chunk 0 re-tokenized, chunk 1 unchanged")
            other_prefix = token_ids(CHUNK, seed=8) + ids[CHUNK:]
            prefix_outputs = _run_sequence(stack, mesh_device, other_prefix, kvpe, index)
        l1_small = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL)
        logger.info(f"{window}: L1_SMALL after the runs: {l1_small.total_bytes_allocated_per_bank} B/bank allocated")
    finally:
        release(stack, caches)

    (outputs, kvpe_raw, index_raw), (outputs_again, kvpe_raw_again, index_raw_again) = runs

    for chunk, out in enumerate(outputs):
        if kv_only:
            assert out is None, "a KV-only last layer produces no output"
            continue
        assert torch.isfinite(out).all(), f"chunk {chunk}: non-finite output"
        assert out.float().std() > 0, f"chunk {chunk}: constant output"

    kvpe_logical = kvpe.unpack_host(kvpe_raw)
    assert torch.isfinite(kvpe_logical).all(), "non-finite KVPE cache"
    assert torch.isfinite(index_raw.float()).all(), "non-finite index cache"
    # Both chunks of every DSA slot were written: each chunk fills its first CHUNK / (SP*TP) rows on every
    # chip (TP-deduped, block-cyclic), so no all-zero row survives below that; the rest is still empty.
    written = NUM_CHUNKS * CHUNK // mesh_device.get_num_devices()
    for name, cache in (("KVPE", kvpe_logical), ("index", index_raw.float())):
        assert (cache[..., :written, :].abs().amax(dim=-1) > 0).all(), f"a written {name} cache row is zero"
        assert not cache[..., written:, :].any(), f"{name} cache rows past the two chunks were written"

    for chunk, (out, again) in enumerate(zip(outputs, outputs_again)):
        if out is not None:
            assert torch.equal(out, again), f"chunk {chunk}: output differs between identical runs"
    assert torch.equal(kvpe_raw.view(torch.uint8), kvpe_raw_again.view(torch.uint8)), "KVPE cache not deterministic"
    assert torch.equal(index_raw, index_raw_again), "index cache not deterministic"
    if reads_prefix_only_from_caches:
        assert not torch.equal(prefix_outputs[0], outputs[0]), "chunk 0's output ignores chunk 0's tokens"
        assert not torch.equal(prefix_outputs[1], outputs[1]), "chunk 1 does not read chunk 0 from the DSA caches"
