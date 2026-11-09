# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Runtimes for the quad prefill-runner hardware screen (adapters/quad_screen.py).

They run inside the real prefill runner, so the H2D producer, the inter-galaxy D2D sockets, the layer acks and the
KV table all take their production path. What each stage computes is fixed and weight-free:

- passthrough: the stage forwards what it received through ``ttnn.clone`` "layers"; stage 0 starts from a known
  pattern, so every stage can compare what it holds with that pattern exactly.
- block-random: every stage runs K2.7 layer 0 (dense) and layer 1 (MoE) with block-random's seeded weights on
  block-random's seeded hidden. Each stage computes on its own copy of that hidden, not on what it received:
  random-weight outputs chained through four galaxies drift away from embedding scale, which skews MoE routing
  and could fail a healthy quad. What it received is still read back and checked, which is what tests the link.

Every stage reads back, per device, the input it received and the output it produced (StageMonitor).
"""

from __future__ import annotations

import dataclasses
import os

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.tt.runners.quad_stage_monitor import StageMonitor, device_shards
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

RESULTS_ENV = "PREFILL_QUAD_RESULTS_DIR"
HIDDEN_SEED = 1234
WEIGHT_SEED = 42


def embedding_like_hidden(total_len: int, hidden_size: int, seed: int = HIDDEN_SEED) -> torch.Tensor:
    """block-random's input: randn rows plus one offset shared by every row, at real token-embedding scale; the
    shared offset skews MoE routing the way real input does."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(total_len, hidden_size, generator=g) * 0.0072 + torch.randn(hidden_size, generator=g) * 0.0021
    return x.to(torch.bfloat16)


def _monitor(config, mode: str) -> StageMonitor:
    results_dir = os.environ.get(RESULTS_ENV)
    if not results_dir:
        raise RuntimeError(f"{RESULTS_ENV} must name a directory shared by every rank")
    rank = int(ttnn.distributed_context_get_rank()) if ttnn.distributed_context_is_initialized() else 0
    return StageMonitor(
        rank=rank,
        mode=mode,
        results_dir=results_dir,
        mesh_cols=config.mesh_shape[config.tp_axis],
        fabric=os.environ.get("PREFILL_FABRIC_MODE", "2d_torus_xy"),
    )


def _shard_dims(config) -> tuple:
    dims = [None, None]
    dims[config.sp_axis] = -2
    dims[config.tp_axis] = -1
    return tuple(dims)


def _host_shards(chunk: torch.Tensor, config) -> list[torch.Tensor]:
    """Per-device slices of a [rows, hidden] chunk, in mesh row-major order: rows over SP (mesh rows), hidden
    over TP (mesh columns), as ShardTensor2dMesh places them."""
    assert (config.sp_axis, config.tp_axis) == (0, 1), "the quad screen shards SP on mesh rows, TP on columns"
    sp, tp = config.mesh_shape
    rows, cols = chunk.shape[0] // sp, chunk.shape[1] // tp
    return [
        chunk[r * rows : (r + 1) * rows, c * cols : (c + 1) * cols].reshape(1, 1, rows, cols).clone()
        for r in range(sp)
        for c in range(tp)
    ]


class _StageInputs:
    """block-random's hidden, one device tensor per chunk of the per-user cache, plus its host shards."""

    def __init__(self, mesh_device, hf_config, config):
        n_chunks = config.max_seq_len // config.chunk_size
        hidden = embedding_like_hidden(n_chunks * config.chunk_size, hf_config.hidden_size)
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(config.mesh_shape), dims=_shard_dims(config))
        self.device, self.host = [], []
        for c in range(n_chunks):
            rows = hidden[c * config.chunk_size : (c + 1) * config.chunk_size]
            self.device.append(
                ttnn.from_torch(
                    rows.reshape(1, 1, config.chunk_size, -1),
                    device=mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mapper,
                )
            )
            self.host.append(_host_shards(rows, config))

    def clone(self, chunk: int) -> ttnn.Tensor:
        return ttnn.clone(self.device[chunk], memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _chunk_index(config, actual_start: int) -> int:
    return actual_start // config.chunk_size


def _make_chunk_input(runtime, config, token_ids):
    if config.is_first_rank:
        return prepare_prefill_input_tensor(
            token_ids, runtime.mesh_device, config.sp_factor, False, config.mesh_shape, config.sp_axis
        )
    return runtime.make_placeholder_activation()


class QuadBlockRandomRuntime:
    """Wraps a middle-stage TtPrefillRuntime (no embedding, no norm, returns its hidden) and gives the runner the
    rank's real first/last role. Everything the runner asks besides a chunk goes to the inner runtime."""

    def __init__(self, mesh_device, hf_config, inner_config, state_dict, *, is_first_rank: bool, is_last_rank: bool):
        self.inner = TtPrefillRuntime(mesh_device=mesh_device, hf_config=hf_config, state_dict=state_dict, config=inner_config)
        self.mesh_device = mesh_device
        self.config = dataclasses.replace(inner_config, is_first_rank=is_first_rank, is_last_rank=is_last_rank)
        self.inputs = _StageInputs(mesh_device, hf_config, inner_config)
        self.monitor = _monitor(self.config, "block_random")

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def make_chunk_input(self, token_ids: list[int]) -> ttnn.Tensor:
        return _make_chunk_input(self.inner, self.config, token_ids)

    def prefill_chunk(self, input_tensor, kv_caches, *, slot_id, actual_start, actual_end, request_id=0, **kwargs):
        chunk = _chunk_index(self.config, actual_start)
        try:
            self.monitor.check("input", chunk, device_shards(input_tensor))
            ttnn.deallocate(input_tensor)
            out = self.inner.prefill_chunk(
                self.inputs.clone(chunk),
                kv_caches,
                slot_id=slot_id,
                actual_start=actual_start,
                actual_end=actual_end,
                request_id=request_id,
                **kwargs,
            )
            self.monitor.check("output", chunk, device_shards(out))
        except Exception as e:
            self.monitor.fail(f"{type(e).__name__}: {e}")
            raise
        if self.config.is_last_rank:
            ttnn.deallocate(out)
            return None
        return out

    def release_trace(self) -> None:
        self.monitor.finish()
        self.inner.release_trace()


class QuadPassthroughRuntime(TtPrefillRuntime):
    """No model: each of the rank's layers is a ttnn.clone of the activation, acked like a real layer. Stage 0
    starts from block-random's hidden; every stage compares what it received and what it sends with it exactly."""

    def _build_model(self, state_dict: dict) -> None:
        self.model = None
        self.model_built = True
        self.inputs = _StageInputs(self.mesh_device, self.hf_config, self.config)
        self.monitor = None

    def compile(self, kv_caches) -> None:
        super().compile(kv_caches)
        self.monitor = _monitor(self.config, "passthrough")

    def make_chunk_input(self, token_ids: list[int]) -> ttnn.Tensor:
        return _make_chunk_input(self, self.config, token_ids)

    def prefill_chunk(self, input_tensor, kv_caches, slot_id, actual_start, actual_end, request_id=0, **kwargs):
        chunk = _chunk_index(self.config, actual_start)
        expected = self.inputs.host[chunk]
        try:
            if self.monitor is not None:
                self.monitor.check("input", chunk, device_shards(input_tensor), None if self.config.is_first_rank else expected)
            if self.config.is_first_rank:
                ttnn.deallocate(input_tensor)
                h = self.inputs.clone(chunk)
            else:
                h = input_tensor
            on_layer_complete = self._layer_complete_cb(slot_id, actual_start, actual_end)
            for i in range(self.config.num_layers):
                nxt = ttnn.clone(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(h)
                h = nxt
                if on_layer_complete is not None:
                    on_layer_complete(self.config.first_layer_idx + i)
            if self.monitor is not None:
                self.monitor.check("output", chunk, device_shards(h), expected)
        except Exception as e:
            if self.monitor is not None:
                self.monitor.fail(f"{type(e).__name__}: {e}")
            raise
        if self.config.is_last_rank:
            ttnn.deallocate(h)
            return None
        return h

    def release_trace(self) -> None:
        if self.monitor is not None:
            self.monitor.finish()
        super().release_trace()


def random_block_state_dict(variant, hf_config) -> dict:
    """K2.7 layers 0 (dense) and 1 (MoE) with block-random's seeded random weights, in TtPrefillTransformer form."""
    from models.demos.deepseek_v3_d_p.utils.transformer_helpers import create_hf_model, extract_layer_state_dict

    torch.manual_seed(WEIGHT_SEED)
    hf = create_hf_model(variant, hf_config, 2)
    sd = hf.state_dict()
    layers = [extract_layer_state_dict(variant, sd, i, hf.layers[i]) for i in range(2)]
    logger.info("quad screen: built random K2.7 layer 0 (dense) and layer 1 (MoE) weights")
    return {"layers": layers}
