# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P2 — the runtime's chunk-range contract, and that ``compile()`` does not change the answer.

An out-of-contract chunk does not raise on its own: ``update_padded_kv_cache`` happily writes at a
misaligned offset and the result is a cache full of plausible numbers at the wrong positions. So
every part of ``[actual_start, actual_end)`` is asserted, and this file is where those asserts are
pinned — the model is depth-reduced to 2 layers because what is under test is the runtime's
arithmetic, not the stack.

``compile()`` warms every KV-length bucket by prefilling zeros into slot 0, so the second test checks
the thing that would make that unsafe: a real prefill after a compile must produce exactly the same
cache as one without it.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama_3_1_8b.tests.common import assert_pcc, cfg_full, galaxy_mesh, pcc
from models.demos.llama_3_1_8b.tt.attention.kv_cache import read_slot_kv
from models.demos.llama_3_1_8b.tt.model_config import random_state_dict
from models.demos.llama_3_1_8b.tt.runners.prefill_kv_validation import naturalize
from models.demos.llama_3_1_8b.tt.tt_prefill_runtime import PrefillRuntimeConfig, TtPrefillRuntime

LAYERS = 2
VOCAB = 8192
CHUNK = 5120


def _cfg():
    cfg = cfg_full()
    cfg.num_hidden_layers = LAYERS
    cfg.vocab_size = VOCAB
    return cfg


def _runtime(mesh_device, cfg, sd, *, max_seq_len, chunk_size):
    from models.demos.llama_3_1_8b.conftest import CCL_TOPOLOGY

    return TtPrefillRuntime(
        mesh_device,
        cfg,
        sd,
        PrefillRuntimeConfig(
            num_layers=LAYERS,
            max_seq_len=max_seq_len,
            chunk_size=chunk_size,
            mesh_shape=tuple(mesh_device.shape),
            topology=CCL_TOPOLOGY,
        ),
    )


def _kv(runtime, cache, n_tokens):
    k_blk, v_blk = read_slot_kv(runtime.mesh_device, cache, 0, LAYERS)
    c = runtime.config
    return [
        (
            naturalize(k_blk[L], n_tokens, c.sp, c.chunk_size, c.max_seq_len),
            naturalize(v_blk[L], n_tokens, c.sp, c.chunk_size, c.max_seq_len),
        )
        for L in range(LAYERS)
    ]


@galaxy_mesh()
def test_chunk_range_contract_is_asserted(mesh_device, device_params):
    """Every way a caller can get ``[actual_start, actual_end)`` wrong must fail loudly."""
    cfg = _cfg()
    torch.manual_seed(0)
    sd = random_state_dict(cfg, num_layers=LAYERS, dtype=torch.float16, seed=61)
    total = 2 * CHUNK
    runtime = _runtime(mesh_device, cfg, sd, max_seq_len=total, chunk_size=CHUNK)
    cache = runtime.allocate_kv_cache()
    ids = [1] * CHUNK

    cases = {
        "start not on a chunk boundary": dict(actual_start=32, actual_end=CHUNK),
        "chunk runs past the cache": dict(actual_start=total, actual_end=total + CHUNK),
        "end before start": dict(actual_start=CHUNK, actual_end=CHUNK),
        "end past one chunk": dict(actual_start=0, actual_end=CHUNK + 32),
        "slot out of range": dict(actual_start=0, actual_end=CHUNK, slot_id=3),
    }
    for name, kwargs in cases.items():
        with pytest.raises(AssertionError):
            runtime.prefill_chunk(runtime.make_chunk_input(ids), cache, **kwargs)
        logger.info(f"rejected: {name}")

    # A short chunk input must also be rejected rather than silently padded somewhere downstream.
    with pytest.raises(AssertionError, match="exactly chunk_size"):
        runtime.make_chunk_input(ids[:-32])

    # And the in-contract call still works, so the asserts are not just refusing everything.
    runtime.prefill_chunk(runtime.make_chunk_input(ids), cache, actual_start=0, actual_end=CHUNK)


@galaxy_mesh()
def test_runtime_config_rejects_misaligned_shapes(mesh_device, device_params):
    """``max_seq_len`` and ``chunk_size`` are the block-cyclic addressing period — guard both."""
    cfg = _cfg()
    sd = random_state_dict(cfg, num_layers=LAYERS, dtype=torch.float16, seed=62)
    with pytest.raises(AssertionError, match="multiple of chunk_size"):
        _runtime(mesh_device, cfg, sd, max_seq_len=CHUNK + 256, chunk_size=CHUNK)
    with pytest.raises(AssertionError, match="TILE_SIZE\\*sp"):
        _runtime(mesh_device, cfg, sd, max_seq_len=2 * 5152, chunk_size=5152)


@galaxy_mesh()
def test_compile_then_run_equals_run(mesh_device, device_params, topology_name):
    """Warming every KV bucket must leave no residue a real prefill can see."""
    cfg = _cfg()
    total = 2 * CHUNK
    torch.manual_seed(0)
    sd = random_state_dict(cfg, num_layers=LAYERS, dtype=torch.float16, seed=63)
    ids = torch.randint(0, VOCAB, (total,)).tolist()

    out = {}
    for label in ("cold", "compiled"):
        runtime = _runtime(mesh_device, cfg, sd, max_seq_len=total, chunk_size=CHUNK)
        cache = runtime.allocate_kv_cache()
        if label == "compiled":
            runtime.compile(cache)
            assert runtime.compiled
        runtime.prefill_sequence(ids, cache)
        out[label] = _kv(runtime, cache, total)

    for L in range(LAYERS):
        for name, i in (("K", 0), ("V", 1)):
            p = pcc(out["cold"][L][i], out["compiled"][L][i])
            logger.info(f"compile-then-run vs run, layer {L} {name}: PCC {p:.6f}")
            assert_pcc(f"compile_idempotent[{topology_name}] L{L} {name}", p, target=0.9999)
