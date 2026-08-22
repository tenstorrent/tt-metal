# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Explicit P150x2 gate for chunk-major streaming prefill.

The production adapter executes a cold 8,256-token prompt as two complete model
calls: an 8,192-token outer chunk followed by a 64-token tail at absolute
position 8,192.  This test isolates one production full-attention/dense layer
and compares that sequential execution with the established monolithic
8,256-token call, which internally uses the same ``(8192, 64)`` layer chunks.

Run only on physical chips 0 and 1, from a neutral working directory::

    cd /tmp
    env -u TT_METAL_HOME \
      TT_VISIBLE_DEVICES=0,1 \
      LAGUNA_PROFILE=p150x2 \
      TT_LAGUNA_PIPE_CHUNK=2048 \
      TT_LAGUNA_PREFILL_FAST=1 \
      TT_LAGUNA_PREFILL_FAST_CHUNK=8192 \
      TT_LAGUNA_STREAMING_PREFILL=1 \
      TT_LAGUNA_RUN_STREAMING_PREFILL_HW=1 \
      PYTHONPATH=/home/ttuser/dev/laguna/tt-metal \
      /home/ttuser/dev/laguna/tt-metal/models/autoports/poolside_laguna_xs_2_1/.venv/bin/python \
      -m pytest -q -s --timeout=1200 \
      /home/ttuser/dev/laguna/tt-metal/models/autoports/poolside_laguna_xs_2_1/tests/test_streaming_prefill_hardware.py

The full-stack/LM-head gate is separate so the fast layer gate stays practical::

    cd /tmp
    env -u TT_METAL_HOME \
      TT_VISIBLE_DEVICES=0,1 \
      LAGUNA_PROFILE=p150x2 \
      TT_LAGUNA_PIPE_CHUNK=2048 \
      TT_LAGUNA_PREFILL_FAST=1 \
      TT_LAGUNA_PREFILL_FAST_CHUNK=8192 \
      TT_LAGUNA_STREAMING_PREFILL=1 \
      TT_LAGUNA_RUN_STREAMING_PREFILL_FULL_STACK_HW=1 \
      PYTHONPATH=/home/ttuser/dev/laguna/tt-metal \
      /home/ttuser/dev/laguna/tt-metal/models/autoports/poolside_laguna_xs_2_1/.venv/bin/python \
      -m pytest -q -s --timeout=1200 -k full_stack \
      /home/ttuser/dev/laguna/tt-metal/models/autoports/poolside_laguna_xs_2_1/tests/test_streaming_prefill_hardware.py

The first gate intentionally isolates one layer and remains fast. The second
fails unless all forty layers and the LM head build. Reported wall times include
host dispatch and synchronization. Its first oracle/stream pair compiles and
checks accuracy; three subsequent alternating repetitions forbid program-cache
misses and report warm per-run values, medians, and their ratio.

The bucket-cliff gate proves the intended performance effect at 16,400 real
tokens, where the rollback adapter computes a 32,768-token bucket while the
streaming adapter computes three canonical 8,192-query programs (24,576 rows)::

    cd /tmp
    env -u TT_METAL_HOME \
      TT_VISIBLE_DEVICES=0,1 \
      LAGUNA_PROFILE=p150x2 \
      TT_LAGUNA_PIPE_CHUNK=2048 \
      TT_LAGUNA_PREFILL_FAST=1 \
      TT_LAGUNA_PREFILL_FAST_CHUNK=8192 \
      TT_LAGUNA_STREAMING_PREFILL=1 \
      TT_LAGUNA_RUN_STREAMING_PREFILL_CLIFF_HW=1 \
      PYTHONPATH=/home/ttuser/dev/laguna/tt-metal \
      /home/ttuser/dev/laguna/tt-metal/models/autoports/poolside_laguna_xs_2_1/.venv/bin/python \
      -m pytest -q -s --timeout=1800 -k bucket \
      /home/ttuser/dev/laguna/tt-metal/models/autoports/poolside_laguna_xs_2_1/tests/test_streaming_prefill_hardware.py
"""

from __future__ import annotations

import os
import statistics
import time

import pytest
import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_reference as R
from models.autoports.poolside_laguna_xs_2_1.tests import test_multichip_decoder as D
from models.autoports.poolside_laguna_xs_2_1.tests.laguna_test_utils import close_mesh, open_mesh, resolve_profile
from models.autoports.poolside_laguna_xs_2_1.tt.model import LagunaModel
from models.autoports.poolside_laguna_xs_2_1.tt.prefill_runtime import PrefillRuntimeOffsets

HARDWARE_GATE_ENV = "TT_LAGUNA_RUN_STREAMING_PREFILL_HW"
FULL_STACK_HARDWARE_GATE_ENV = "TT_LAGUNA_RUN_STREAMING_PREFILL_FULL_STACK_HW"
CLIFF_HARDWARE_GATE_ENV = "TT_LAGUNA_RUN_STREAMING_PREFILL_CLIFF_HW"
OUTER = 8192
TAIL = 64
TOTAL = OUTER + TAIL
CLIFF_REAL = 2 * OUTER + 16
CLIFF_TAIL_BUCKET = OUTER
CLIFF_LEGACY_BUCKET = 4 * OUTER
BLOCK_SIZE = 64
TRACE_REGION_SIZE = 200_000_000
WARM_REPETITIONS = 3
CLIFF_WARM_REPETITIONS = 2


@pytest.fixture(scope="module")
def p150x2_mesh():
    # Validate the physical target before any API can open hardware.
    assert (
        os.environ.get("TT_VISIBLE_DEVICES") == "0,1"
    ), "streaming-prefill hardware proof is pinned to TT_VISIBLE_DEVICES=0,1"
    assert "TT_METAL_HOME" not in os.environ, "run with env -u TT_METAL_HOME"
    assert os.environ.get("TT_LAGUNA_STREAMING_PREFILL", "1") == "1"
    profile = resolve_profile("p150x2", trace_region_size=TRACE_REGION_SIZE)
    mesh = open_mesh(ttnn, profile)
    try:
        assert mesh.get_num_devices() == 2
        yield mesh
    finally:
        close_mesh(ttnn, mesh)


@pytest.fixture(scope="module")
def hf_config():
    return R.build_config()


def _runtime(dec, mesh, *, start: int, chunk_lengths: tuple[int, ...]):
    """Allocate and populate the indexed-RoPE/runtime-start contract used by the adapter."""
    position_ids = []
    chunk_starts = []
    rope_outputs = []
    offset = 0
    for length in chunk_lengths:
        positions = torch.arange(start + offset, start + offset + length, dtype=torch.int32).reshape(1, length)
        position_ids.append(D._int(positions, mesh, ttnn.uint32))
        chunk_starts.append(D._int(torch.tensor([start + offset], dtype=torch.int32), mesh))
        rope_outputs.append(
            (
                D._tt(torch.zeros((1, 1, length, dec.cfg.rotary_dim)), mesh),
                D._tt(torch.zeros((1, 1, length, dec.cfg.rotary_dim)), mesh),
            )
        )
        offset += length

    runtime = PrefillRuntimeOffsets(
        bucket_len=sum(chunk_lengths),
        chunk_offsets=tuple(sum(chunk_lengths[:i]) for i in range(len(chunk_lengths))),
        chunk_lengths=chunk_lengths,
        position_ids=tuple(position_ids),
        chunk_start_idxs=tuple(chunk_starts),
        rope_outputs={dec.cfg.attention_type: tuple(rope_outputs)},
    )
    return runtime


def _model_runtime(model, mesh, *, start: int, chunk_lengths: tuple[int, ...]):
    """Allocate indexed-RoPE slots for every attention kind in the complete model."""
    position_ids = []
    chunk_starts = []
    offset = 0
    for length in chunk_lengths:
        positions = torch.arange(start + offset, start + offset + length, dtype=torch.int32).reshape(1, length)
        position_ids.append(D._int(positions, mesh, ttnn.uint32))
        chunk_starts.append(D._int(torch.tensor([start + offset], dtype=torch.int32), mesh))
        offset += length

    rotary_dims = {}
    for dec in model.layers:
        kind = dec.cfg.attention_type
        prior = rotary_dims.setdefault(kind, dec.cfg.rotary_dim)
        assert prior == dec.cfg.rotary_dim
    rope_outputs = {
        kind: tuple(
            (
                D._tt(torch.zeros((1, 1, length, rotary_dim)), mesh),
                D._tt(torch.zeros((1, 1, length, rotary_dim)), mesh),
            )
            for length in chunk_lengths
        )
        for kind, rotary_dim in rotary_dims.items()
    }
    return PrefillRuntimeOffsets(
        bucket_len=sum(chunk_lengths),
        chunk_offsets=tuple(sum(chunk_lengths[:i]) for i in range(len(chunk_lengths))),
        chunk_lengths=chunk_lengths,
        position_ids=tuple(position_ids),
        chunk_start_idxs=tuple(chunk_starts),
        rope_outputs=rope_outputs,
    )


def _indexed_rope(dec, runtime):
    """Build the preallocated per-chunk RoPE matrices exactly as LagunaModel does."""
    mats = []
    outputs = runtime.rope_outputs[dec.cfg.attention_type]
    for positions, (cos_output, sin_output) in zip(runtime.position_ids, outputs):
        cos = dec._rope_prefill_indexed(positions, output_tensor=cos_output)
        sin = dec._rope_prefill_indexed(positions, sin=True, output_tensor=sin_output)
        mats.append((cos, sin))
    return tuple(mats)


def _page_tables_for_capacity(
    mesh,
    *,
    capacity: int,
    real_end: int,
    fill_start: int,
    fill_end: int,
    scratch_block: int,
):
    """Build the adapter's scratch-protected attention and column-zero rebased fill rows."""
    real_blocks = (capacity + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Width is the logical compute horizon.  The scratch block is an adapter-private
    # physical row stored *as a table value*; it does not consume a table column.
    width = ((real_blocks + 7) // 8) * 8
    scheduler = torch.arange(real_blocks, dtype=torch.int32).reshape(1, real_blocks)

    attention = torch.full((1, width), scratch_block, dtype=torch.int32)
    visible_blocks = (real_end + BLOCK_SIZE - 1) // BLOCK_SIZE
    attention[:, :visible_blocks] = scheduler[:, :visible_blocks]

    assert fill_start % BLOCK_SIZE == 0
    fill = torch.full((1, width), -1, dtype=torch.int32)
    first_block = fill_start // BLOCK_SIZE
    last_block = (fill_end + BLOCK_SIZE - 1) // BLOCK_SIZE
    fill[:, : last_block - first_block] = scheduler[:, first_block:last_block]
    return D._int(attention, mesh), D._int(fill, mesh)


def _page_tables(mesh, *, real_end: int, fill_start: int, fill_end: int, scratch_block: int):
    return _page_tables_for_capacity(
        mesh,
        capacity=TOTAL,
        real_end=real_end,
        fill_start=fill_start,
        fill_end=fill_end,
        scratch_block=scratch_block,
    )


def _prefix_snapshot(cache, prefix_blocks: int):
    return [ttnn.to_torch(tensor)[:prefix_blocks].clone() for tensor in ttnn.get_device_tensors(cache)]


def _assert_prefix_unchanged(cache, snapshots, *, name: str):
    device_tensors = ttnn.get_device_tensors(cache)
    assert len(device_tensors) == len(snapshots) == 2
    for chip, (tensor, expected) in enumerate(zip(device_tensors, snapshots)):
        actual = ttnn.to_torch(tensor)[: expected.shape[0]]
        assert torch.equal(actual, expected), f"tail prefill overwrote prefix {name.upper()} on mesh device {chip}"


def _cache_block_pccs(actual_cache, reference_cache, block: int):
    actual_devices = ttnn.get_device_tensors(actual_cache)
    reference_devices = ttnn.get_device_tensors(reference_cache)
    assert len(actual_devices) == len(reference_devices) == 2
    return tuple(
        D._pcc(
            ttnn.to_torch(actual)[block : block + 1].float(),
            ttnn.to_torch(reference)[block : block + 1].float(),
        )
        for actual, reference in zip(actual_devices, reference_devices)
    )


def _dram_snapshot(mesh):
    """Return the synchronized per-device DRAM view used by serving gates."""

    ttnn.synchronize_device(mesh)
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    banks = int(view.num_banks)
    return {
        "used_mib": int(view.total_bytes_allocated_per_bank) * banks / 2**20,
        "free_mib": int(view.total_bytes_free_per_bank) * banks / 2**20,
        "free_fraction": (
            float(view.total_bytes_free_per_bank) / float(view.total_bytes_per_bank)
            if int(view.total_bytes_per_bank)
            else 0.0
        ),
        "largest_contiguous_mib_per_bank": int(view.largest_contiguous_bytes_free_per_bank) / 2**20,
    }


@pytest.mark.skipif(
    os.environ.get(HARDWARE_GATE_ENV) != "1",
    reason=f"set {HARDWARE_GATE_ENV}=1 to run the explicit P150x2 streaming-prefill layer gate",
)
@torch.inference_mode()
def test_d2_sequential_8192_plus_64_matches_monolithic_8256_and_preserves_prefix(p150x2_mesh, hf_config):
    """Cross the adapter outer-call boundary with production D2 layer programs."""
    mesh = p150x2_mesh
    dec = D._decoder(hf_config, D.FULL_DENSE, mesh)
    assert dec.D == 2
    assert dec.PIPE_CHUNK == 2048, "run the production TT_LAGUNA_PIPE_CHUNK=2048 profile"
    assert dec.PREFILL_FAST, "run with TT_LAGUNA_PREFILL_FAST=1"
    assert dec._prefill_pipe_chunk == OUTER

    generator = torch.Generator().manual_seed(20260822)
    hidden = (torch.randn((1, TOTAL, D.HIDDEN), generator=generator) * 0.5).to(torch.bfloat16)
    real_blocks = (TOTAL + BLOCK_SIZE - 1) // BLOCK_SIZE
    scratch_block = real_blocks
    cache_len = (real_blocks + 1) * BLOCK_SIZE

    # Monolithic control: one layer call, internally chunked as (8192, 64).
    oracle_kv = dec.alloc_kv_cache(max_users=1, max_seq_len=cache_len, block_size=BLOCK_SIZE)
    oracle_pt, oracle_fill = _page_tables(
        mesh,
        real_end=TOTAL,
        fill_start=0,
        fill_end=TOTAL,
        scratch_block=scratch_block,
    )
    oracle_runtime = _runtime(dec, mesh, start=0, chunk_lengths=(OUTER, TAIL))
    ttnn.synchronize_device(mesh)
    oracle_started = time.perf_counter()
    oracle_rope = _indexed_rope(dec, oracle_runtime)
    oracle_output = dec.prefill_forward(
        D._tt(hidden, mesh),
        oracle_kv,
        oracle_pt,
        fill_page_table=oracle_fill,
        fill_page_table_base_pos=0,
        user_id=0,
        start_pos=0,
        rope_mats=oracle_rope,
        runtime_offsets=oracle_runtime,
    )
    ttnn.synchronize_device(mesh)
    oracle_seconds = time.perf_counter() - oracle_started
    oracle_tail = D._compose0(oracle_output, mesh).float().reshape(1, TOTAL, D.HIDDEN)[:, -TAIL:].clone()

    # Streaming candidate: the same chunks cross a production layer-call boundary.
    streamed_kv = dec.alloc_kv_cache(max_users=1, max_seq_len=cache_len, block_size=BLOCK_SIZE)
    prefix_pt, prefix_fill = _page_tables(
        mesh,
        real_end=OUTER,
        fill_start=0,
        fill_end=OUTER,
        scratch_block=scratch_block,
    )
    prefix_runtime = _runtime(dec, mesh, start=0, chunk_lengths=(OUTER,))
    ttnn.synchronize_device(mesh)
    prefix_started = time.perf_counter()
    prefix_rope = _indexed_rope(dec, prefix_runtime)
    prefix_output = dec.prefill_forward(
        D._tt(hidden[:, :OUTER], mesh),
        streamed_kv,
        prefix_pt,
        fill_page_table=prefix_fill,
        fill_page_table_base_pos=0,
        user_id=0,
        start_pos=0,
        rope_mats=prefix_rope,
        runtime_offsets=prefix_runtime,
    )
    ttnn.synchronize_device(mesh)
    prefix_seconds = time.perf_counter() - prefix_started
    del prefix_output

    prefix_blocks = OUTER // BLOCK_SIZE
    snapshots = {name: _prefix_snapshot(streamed_kv[name], prefix_blocks) for name in ("k", "v")}

    tail_pt, tail_fill = _page_tables(
        mesh,
        real_end=TOTAL,
        fill_start=OUTER,
        fill_end=TOTAL,
        scratch_block=scratch_block,
    )
    tail_runtime = _runtime(dec, mesh, start=OUTER, chunk_lengths=(TAIL,))
    ttnn.synchronize_device(mesh)
    tail_started = time.perf_counter()
    tail_rope = _indexed_rope(dec, tail_runtime)
    tail_output = dec.prefill_forward(
        D._tt(hidden[:, OUTER:], mesh),
        streamed_kv,
        tail_pt,
        fill_page_table=tail_fill,
        fill_page_table_base_pos=OUTER,
        user_id=0,
        start_pos=OUTER,
        rope_mats=tail_rope,
        runtime_offsets=tail_runtime,
    )
    ttnn.synchronize_device(mesh)
    tail_seconds = time.perf_counter() - tail_started
    streamed_tail = D._compose0(tail_output, mesh).float().reshape(1, TAIL, D.HIDDEN)

    tail_pcc = D._pcc(streamed_tail, oracle_tail)
    rmse = torch.sqrt(torch.mean((streamed_tail - oracle_tail) ** 2)).item()
    ref_rms = torch.sqrt(torch.mean(oracle_tail**2)).item()
    relative_rmse = rmse / max(ref_rms, 1e-8)

    for name in ("k", "v"):
        _assert_prefix_unchanged(streamed_kv[name], snapshots[name], name=name)
    tail_k_pccs = _cache_block_pccs(streamed_kv["k"], oracle_kv["k"], prefix_blocks)
    tail_v_pccs = _cache_block_pccs(streamed_kv["v"], oracle_kv["v"], prefix_blocks)

    print(
        "STREAMING_PREFILL_HW_RESULT "
        f"oracle_8256_s={oracle_seconds:.6f} "
        f"stream_prefix_8192_s={prefix_seconds:.6f} "
        f"stream_tail_64_s={tail_seconds:.6f} "
        f"stream_total_s={prefix_seconds + tail_seconds:.6f} "
        f"tail_pcc={tail_pcc:.8f} "
        f"tail_relative_rmse={relative_rmse:.8f} "
        f"tail_k_pcc_chip0={tail_k_pccs[0]:.8f} "
        f"tail_k_pcc_chip1={tail_k_pccs[1]:.8f} "
        f"tail_v_pcc_chip0={tail_v_pccs[0]:.8f} "
        f"tail_v_pcc_chip1={tail_v_pccs[1]:.8f} "
        "logits_pcc=not_covered_single_layer "
        "prefix_kv_preserved=true",
        flush=True,
    )
    assert tail_pcc >= D.PCC_BAR, f"streamed final-tail PCC {tail_pcc:.8f} < {D.PCC_BAR}"
    assert relative_rmse <= 0.05, f"streamed final-tail relative RMSE {relative_rmse:.8f} > 0.05"
    assert min(*tail_k_pccs, *tail_v_pccs) >= D.PCC_BAR


@pytest.mark.skipif(
    os.environ.get(FULL_STACK_HARDWARE_GATE_ENV) != "1",
    reason=(f"set {FULL_STACK_HARDWARE_GATE_ENV}=1 to run the explicit P150x2 " "streaming-prefill full-stack gate"),
)
@torch.inference_mode()
def test_d2_full_stack_streamed_8192_plus_64_matches_monolithic_8256_logits(p150x2_mesh, hf_config):
    """Qualify the outer-call boundary through all 40 layers and the production LM head."""
    mesh = p150x2_mesh
    build_started = time.perf_counter()
    model = LagunaModel.from_pretrained(mesh, hf_config=hf_config, max_seq_len=TOTAL)
    ttnn.synchronize_device(mesh)
    build_seconds = time.perf_counter() - build_started

    expected_layers = int(hf_config.num_hidden_layers)
    assert expected_layers == 40
    assert len(model.layers) == expected_layers
    assert model.meta["layer_indices"] == list(range(expected_layers))
    assert model.D == 2
    assert all(dec.PIPE_CHUNK == 2048 for dec in model.layers)
    assert all(dec.PREFILL_FAST and dec._prefill_pipe_chunk == OUTER for dec in model.layers)

    real_blocks = (TOTAL + BLOCK_SIZE - 1) // BLOCK_SIZE
    scratch_block = real_blocks
    cache_len = (real_blocks + 1) * BLOCK_SIZE
    cache_started = time.perf_counter()
    # Reuse this cache after the oracle. Every streamed visible block is overwritten
    # before it is read, avoiding a second full-model cache allocation at peak residency.
    kv_cache = model.alloc_kv_cache(max_users=1, max_seq_len=cache_len, block_size=BLOCK_SIZE)
    ttnn.synchronize_device(mesh)
    cache_seconds = time.perf_counter() - cache_started
    assert len(kv_cache) == expected_layers

    generator = torch.Generator().manual_seed(20260822)
    tokens = torch.randint(0, model.cfg.vocab, (1, TOTAL), generator=generator, dtype=torch.int64)
    oracle_pt, oracle_fill = _page_tables(
        mesh,
        real_end=TOTAL,
        fill_start=0,
        fill_end=TOTAL,
        scratch_block=scratch_block,
    )
    oracle_runtime = _model_runtime(model, mesh, start=0, chunk_lengths=(OUTER, TAIL))
    oracle_input = D._int(tokens.to(torch.int32), mesh, ttnn.uint32)
    prefix_pt, prefix_fill = _page_tables(
        mesh,
        real_end=OUTER,
        fill_start=0,
        fill_end=OUTER,
        scratch_block=scratch_block,
    )
    prefix_runtime = _model_runtime(model, mesh, start=0, chunk_lengths=(OUTER,))
    prefix_input = D._int(tokens[:, :OUTER].to(torch.int32), mesh, ttnn.uint32)
    tail_pt, tail_fill = _page_tables(
        mesh,
        real_end=TOTAL,
        fill_start=OUTER,
        fill_end=TOTAL,
        scratch_block=scratch_block,
    )
    tail_runtime = _model_runtime(model, mesh, start=OUTER, chunk_lengths=(TAIL,))
    tail_input = D._int(tokens[:, OUTER:].to(torch.int32), mesh, ttnn.uint32)

    def oracle_forward():
        hidden = model.prefill_layers(
            model.embed_prefill(oracle_input),
            kv_cache,
            oracle_pt,
            fill_page_table=oracle_fill,
            fill_page_table_base_pos=0,
            user_id=0,
            start_pos=0,
            runtime_offsets=oracle_runtime,
        )
        last = ttnn.slice(hidden, [0, TOTAL - 1, 0], [1, TOTAL, model.cfg.hidden])
        return hidden, model.lm_head_shards_prefill(last)

    def prefix_forward():
        return model.prefill_layers(
            model.embed_prefill(prefix_input),
            kv_cache,
            prefix_pt,
            fill_page_table=prefix_fill,
            fill_page_table_base_pos=0,
            user_id=0,
            start_pos=0,
            runtime_offsets=prefix_runtime,
        )

    def tail_forward():
        hidden = model.prefill_layers(
            model.embed_prefill(tail_input),
            kv_cache,
            tail_pt,
            fill_page_table=tail_fill,
            fill_page_table_base_pos=OUTER,
            user_id=0,
            start_pos=OUTER,
            runtime_offsets=tail_runtime,
        )
        last = ttnn.slice(hidden, [0, TAIL - 1, 0], [1, TAIL, model.cfg.hidden])
        return hidden, model.lm_head_shards_prefill(last)

    # Compile and check each complete schedule once before measuring either one.
    ttnn.synchronize_device(mesh)
    oracle_started = time.perf_counter()
    oracle_hidden, oracle_logit_shards = oracle_forward()
    ttnn.synchronize_device(mesh)
    oracle_seconds = time.perf_counter() - oracle_started
    oracle_tail = (
        D._compose0(ttnn.slice(oracle_hidden, [0, OUTER, 0], [1, TOTAL, model.cfg.hidden]), mesh)
        .float()
        .reshape(1, TAIL, model.cfg.hidden)
    )
    oracle_logits = model.logits_to_host(oracle_logit_shards).float().reshape(model.cfg.vocab)
    del oracle_hidden, oracle_logit_shards

    ttnn.synchronize_device(mesh)
    prefix_started = time.perf_counter()
    prefix_hidden = prefix_forward()
    ttnn.synchronize_device(mesh)
    prefix_seconds = time.perf_counter() - prefix_started
    del prefix_hidden
    ttnn.synchronize_device(mesh)
    tail_started = time.perf_counter()
    streamed_hidden, streamed_logit_shards = tail_forward()
    ttnn.synchronize_device(mesh)
    tail_seconds = time.perf_counter() - tail_started
    streamed_tail = D._compose0(streamed_hidden, mesh).float().reshape(1, TAIL, model.cfg.hidden)
    streamed_logits = model.logits_to_host(streamed_logit_shards).float().reshape(model.cfg.vocab)

    hidden_pcc = D._pcc(streamed_tail, oracle_tail)
    logits_pcc = D._pcc(streamed_logits, oracle_logits)
    hidden_rmse = torch.sqrt(torch.mean((streamed_tail - oracle_tail) ** 2)).item()
    hidden_rms = torch.sqrt(torch.mean(oracle_tail**2)).item()
    hidden_relative_rmse = hidden_rmse / max(hidden_rms, 1e-8)
    logits_rmse = torch.sqrt(torch.mean((streamed_logits - oracle_logits) ** 2)).item()
    logits_rms = torch.sqrt(torch.mean(oracle_logits**2)).item()
    logits_relative_rmse = logits_rmse / max(logits_rms, 1e-8)
    oracle_argmax = int(torch.argmax(oracle_logits))
    streamed_argmax = int(torch.argmax(streamed_logits))

    assert hidden_pcc >= D.PCC_BAR
    assert hidden_relative_rmse <= 0.05
    assert logits_pcc >= D.PCC_BAR
    assert logits_relative_rmse <= 0.05
    assert streamed_argmax == oracle_argmax
    del streamed_hidden, streamed_logit_shards

    # Alternate the two already-compiled paths on the same persistent inputs and
    # cache. Both schedules overwrite every visible cache block before reading it,
    # so each repetition is idempotent and independent of the preceding schedule.
    warm_oracle_seconds = []
    warm_stream_seconds = []
    ttnn.synchronize_device(mesh)
    warm_program_cache_entries = int(mesh.num_program_cache_entries())
    mesh.set_program_cache_misses_allowed(False)
    try:
        for _ in range(WARM_REPETITIONS):
            ttnn.synchronize_device(mesh)
            started = time.perf_counter()
            warm_oracle_hidden, warm_oracle_logits = oracle_forward()
            ttnn.synchronize_device(mesh)
            warm_oracle_seconds.append(time.perf_counter() - started)
            del warm_oracle_hidden, warm_oracle_logits

            ttnn.synchronize_device(mesh)
            started = time.perf_counter()
            warm_prefix_hidden = prefix_forward()
            warm_stream_hidden, warm_stream_logits = tail_forward()
            ttnn.synchronize_device(mesh)
            warm_stream_seconds.append(time.perf_counter() - started)
            del warm_prefix_hidden, warm_stream_hidden, warm_stream_logits
    finally:
        mesh.set_program_cache_misses_allowed(True)
    assert int(mesh.num_program_cache_entries()) == warm_program_cache_entries

    oracle_median = statistics.median(warm_oracle_seconds)
    stream_median = statistics.median(warm_stream_seconds)
    stream_over_oracle = stream_median / oracle_median
    oracle_over_stream = oracle_median / stream_median
    oracle_samples = ",".join(f"{seconds:.6f}" for seconds in warm_oracle_seconds)
    stream_samples = ",".join(f"{seconds:.6f}" for seconds in warm_stream_seconds)

    print(
        "STREAMING_PREFILL_FULL_STACK_HW_RESULT "
        f"layers={len(model.layers)} "
        f"build_s={build_seconds:.6f} "
        f"cache_alloc_s={cache_seconds:.6f} "
        f"compile_oracle_8256_s={oracle_seconds:.6f} "
        f"compile_stream_prefix_8192_s={prefix_seconds:.6f} "
        f"compile_stream_tail_64_s={tail_seconds:.6f} "
        f"compile_stream_total_s={prefix_seconds + tail_seconds:.6f} "
        f"warm_repetitions={WARM_REPETITIONS} "
        f"warm_program_cache_entries={warm_program_cache_entries} "
        f"warm_oracle_samples_s={oracle_samples} "
        f"warm_stream_samples_s={stream_samples} "
        f"warm_oracle_median_s={oracle_median:.6f} "
        f"warm_stream_median_s={stream_median:.6f} "
        f"warm_stream_over_oracle_ratio={stream_over_oracle:.6f} "
        f"warm_oracle_over_stream_speedup={oracle_over_stream:.6f} "
        f"hidden_tail_pcc={hidden_pcc:.8f} "
        f"hidden_tail_relative_rmse={hidden_relative_rmse:.8f} "
        f"logits_pcc={logits_pcc:.8f} "
        f"logits_relative_rmse={logits_relative_rmse:.8f} "
        f"oracle_argmax={oracle_argmax} "
        f"streamed_argmax={streamed_argmax} "
        f"argmax_equal={str(oracle_argmax == streamed_argmax).lower()}",
        flush=True,
    )
    assert len(warm_oracle_seconds) == len(warm_stream_seconds) == WARM_REPETITIONS


@pytest.mark.skipif(
    os.environ.get(CLIFF_HARDWARE_GATE_ENV) != "1",
    reason=(f"set {CLIFF_HARDWARE_GATE_ENV}=1 to run the explicit P150x2 " "streaming-prefill bucket-cliff gate"),
)
@torch.inference_mode()
def test_d2_full_stack_streamed_16400_beats_legacy_32768_bucket(p150x2_mesh, hf_config):
    """Measure the production win immediately above the historical 16K bucket.

    The rollback adapter rounds 16,400 real rows to one 32,768-row model call.
    Streaming executes two complete 8,192-row chunks plus a canonical 8,192-row
    tail containing 16 real rows.  The streamed and rollback tails therefore use
    the same SDPA query shape/reduction family. Both paths use identical real
    tokens, absolute RoPE positions, page-table width, model/cache weights, and
    last-real-token LM head. Padding is causal and cannot affect the compared row.
    """

    mesh = p150x2_mesh
    model = LagunaModel.from_pretrained(
        mesh,
        hf_config=hf_config,
        max_seq_len=CLIFF_LEGACY_BUCKET,
    )
    assert len(model.layers) == 40 and model.D == 2
    assert all(dec._prefill_pipe_chunk == OUTER for dec in model.layers)

    physical_blocks = CLIFF_LEGACY_BUCKET // BLOCK_SIZE
    scratch_block = physical_blocks
    cache_len = (physical_blocks + 1) * BLOCK_SIZE
    kv_cache = model.alloc_kv_cache(
        max_users=1,
        max_seq_len=cache_len,
        block_size=BLOCK_SIZE,
    )
    memory_after_cache = _dram_snapshot(mesh)

    generator = torch.Generator().manual_seed(20260822)
    real_tokens = torch.randint(
        0,
        model.cfg.vocab,
        (1, CLIFF_REAL),
        generator=generator,
        dtype=torch.int64,
    )
    legacy_tokens = torch.zeros((1, CLIFF_LEGACY_BUCKET), dtype=torch.int64)
    legacy_tokens[:, :CLIFF_REAL] = real_tokens
    tail_tokens = torch.zeros((1, CLIFF_TAIL_BUCKET), dtype=torch.int64)
    tail_real = CLIFF_REAL - 2 * OUTER
    assert tail_real == 16
    tail_tokens[:, :tail_real] = real_tokens[:, 2 * OUTER :]

    legacy_input = D._int(legacy_tokens.to(torch.int32), mesh, ttnn.uint32)
    chunk0_input = D._int(real_tokens[:, :OUTER].to(torch.int32), mesh, ttnn.uint32)
    chunk1_input = D._int(real_tokens[:, OUTER : 2 * OUTER].to(torch.int32), mesh, ttnn.uint32)
    tail_input = D._int(tail_tokens.to(torch.int32), mesh, ttnn.uint32)

    def tables(real_end: int, fill_start: int, fill_end: int):
        return _page_tables_for_capacity(
            mesh,
            capacity=CLIFF_LEGACY_BUCKET,
            real_end=real_end,
            fill_start=fill_start,
            fill_end=fill_end,
            scratch_block=scratch_block,
        )

    legacy_pt, legacy_fill = tables(CLIFF_REAL, 0, CLIFF_REAL)
    chunk0_pt, chunk0_fill = tables(OUTER, 0, OUTER)
    chunk1_pt, chunk1_fill = tables(2 * OUTER, OUTER, 2 * OUTER)
    tail_pt, tail_fill = tables(CLIFF_REAL, 2 * OUTER, CLIFF_REAL)

    legacy_runtime = _model_runtime(
        model,
        mesh,
        start=0,
        chunk_lengths=(OUTER, OUTER, OUTER, OUTER),
    )
    chunk0_runtime = _model_runtime(model, mesh, start=0, chunk_lengths=(OUTER,))
    chunk1_runtime = _model_runtime(model, mesh, start=OUTER, chunk_lengths=(OUTER,))
    tail_runtime = _model_runtime(
        model,
        mesh,
        start=2 * OUTER,
        chunk_lengths=(CLIFF_TAIL_BUCKET,),
    )

    def last_logits(hidden, row: int):
        selected = ttnn.slice(hidden, [0, row, 0], [1, row + 1, model.cfg.hidden])
        return selected, model.lm_head_shards_prefill(selected)

    def legacy_forward():
        hidden = model.prefill_layers(
            model.embed_prefill(legacy_input),
            kv_cache,
            legacy_pt,
            fill_page_table=legacy_fill,
            fill_page_table_base_pos=0,
            user_id=0,
            start_pos=0,
            runtime_offsets=legacy_runtime,
        )
        selected, logits = last_logits(hidden, CLIFF_REAL - 1)
        del hidden
        return selected, logits

    def stream_forward():
        hidden = model.prefill_layers(
            model.embed_prefill(chunk0_input),
            kv_cache,
            chunk0_pt,
            fill_page_table=chunk0_fill,
            fill_page_table_base_pos=0,
            user_id=0,
            start_pos=0,
            runtime_offsets=chunk0_runtime,
        )
        del hidden
        hidden = model.prefill_layers(
            model.embed_prefill(chunk1_input),
            kv_cache,
            chunk1_pt,
            fill_page_table=chunk1_fill,
            fill_page_table_base_pos=OUTER,
            user_id=0,
            start_pos=OUTER,
            runtime_offsets=chunk1_runtime,
        )
        del hidden
        hidden = model.prefill_layers(
            model.embed_prefill(tail_input),
            kv_cache,
            tail_pt,
            fill_page_table=tail_fill,
            fill_page_table_base_pos=2 * OUTER,
            user_id=0,
            start_pos=2 * OUTER,
            runtime_offsets=tail_runtime,
        )
        selected, logits = last_logits(hidden, tail_real - 1)
        del hidden
        return selected, logits

    # Compile both complete schedules, then establish last-real-row accuracy.
    ttnn.synchronize_device(mesh)
    legacy_selected, legacy_logits_tt = legacy_forward()
    ttnn.synchronize_device(mesh)
    stream_selected, stream_logits_tt = stream_forward()
    ttnn.synchronize_device(mesh)

    legacy_hidden = D._compose0(legacy_selected, mesh).float().reshape(model.cfg.hidden)
    stream_hidden = D._compose0(stream_selected, mesh).float().reshape(model.cfg.hidden)
    legacy_logits = model.logits_to_host(legacy_logits_tt).float().reshape(model.cfg.vocab)
    stream_logits = model.logits_to_host(stream_logits_tt).float().reshape(model.cfg.vocab)
    hidden_pcc = D._pcc(stream_hidden, legacy_hidden)
    logits_pcc = D._pcc(stream_logits, legacy_logits)
    hidden_rmse = torch.sqrt(torch.mean((stream_hidden - legacy_hidden) ** 2)).item()
    hidden_rms = torch.sqrt(torch.mean(legacy_hidden**2)).item()
    hidden_relative_rmse = hidden_rmse / max(hidden_rms, 1e-8)
    logits_rmse = torch.sqrt(torch.mean((stream_logits - legacy_logits) ** 2)).item()
    logits_rms = torch.sqrt(torch.mean(legacy_logits**2)).item()
    logits_relative_rmse = logits_rmse / max(logits_rms, 1e-8)
    legacy_argmax = int(torch.argmax(legacy_logits))
    stream_argmax = int(torch.argmax(stream_logits))
    legacy_top10 = torch.topk(legacy_logits, k=10).indices.tolist()
    stream_top10 = torch.topk(stream_logits, k=10).indices.tolist()
    top10_overlap = len(set(legacy_top10) & set(stream_top10))
    legacy_top2 = torch.topk(legacy_logits, k=2).values
    stream_top2 = torch.topk(stream_logits, k=2).values
    legacy_top1_margin = float(legacy_top2[0] - legacy_top2[1])
    stream_top1_margin = float(stream_top2[0] - stream_top2[1])
    del legacy_selected, legacy_logits_tt, stream_selected, stream_logits_tt
    memory_after_compile = _dram_snapshot(mesh)

    legacy_seconds = []
    stream_seconds = []
    entries = int(mesh.num_program_cache_entries())
    mesh.set_program_cache_misses_allowed(False)
    try:
        for _ in range(CLIFF_WARM_REPETITIONS):
            ttnn.synchronize_device(mesh)
            started = time.perf_counter()
            selected, logits = legacy_forward()
            ttnn.synchronize_device(mesh)
            legacy_seconds.append(time.perf_counter() - started)
            del selected, logits

            ttnn.synchronize_device(mesh)
            started = time.perf_counter()
            selected, logits = stream_forward()
            ttnn.synchronize_device(mesh)
            stream_seconds.append(time.perf_counter() - started)
            del selected, logits
    finally:
        mesh.set_program_cache_misses_allowed(True)
    assert int(mesh.num_program_cache_entries()) == entries

    legacy_median = statistics.median(legacy_seconds)
    stream_median = statistics.median(stream_seconds)
    speedup = legacy_median / stream_median
    print(
        "STREAMING_PREFILL_CLIFF_HW_RESULT "
        f"real_tokens={CLIFF_REAL} "
        f"legacy_compute_tokens={CLIFF_LEGACY_BUCKET} "
        f"stream_compute_tokens={2 * OUTER + CLIFF_TAIL_BUCKET} "
        f"legacy_samples_s={','.join(f'{x:.6f}' for x in legacy_seconds)} "
        f"stream_samples_s={','.join(f'{x:.6f}' for x in stream_seconds)} "
        f"legacy_median_s={legacy_median:.6f} "
        f"stream_median_s={stream_median:.6f} "
        f"speedup={speedup:.6f} "
        f"hidden_pcc={hidden_pcc:.8f} "
        f"hidden_relative_rmse={hidden_relative_rmse:.8f} "
        f"logits_pcc={logits_pcc:.8f} "
        f"logits_relative_rmse={logits_relative_rmse:.8f} "
        f"legacy_argmax={legacy_argmax} "
        f"stream_argmax={stream_argmax} "
        f"argmax_equal={str(stream_argmax == legacy_argmax).lower()} "
        f"top10_overlap={top10_overlap}/10 "
        f"legacy_top1_margin={legacy_top1_margin:.8f} "
        f"stream_top1_margin={stream_top1_margin:.8f} "
        f"cache_used_mib={memory_after_cache['used_mib']:.1f} "
        f"cache_free_mib={memory_after_cache['free_mib']:.1f} "
        f"cache_free_fraction={memory_after_cache['free_fraction']:.4f} "
        f"cache_largest_contiguous_mib_per_bank="
        f"{memory_after_cache['largest_contiguous_mib_per_bank']:.1f} "
        f"compile_used_mib={memory_after_compile['used_mib']:.1f} "
        f"compile_free_mib={memory_after_compile['free_mib']:.1f} "
        f"compile_free_fraction={memory_after_compile['free_fraction']:.4f} "
        f"compile_largest_contiguous_mib_per_bank="
        f"{memory_after_compile['largest_contiguous_mib_per_bank']:.1f} "
        f"program_cache_entries={entries}",
        flush=True,
    )
    assert len(legacy_seconds) == len(stream_seconds) == CLIFF_WARM_REPETITIONS
    assert hidden_pcc >= D.PCC_BAR, f"stream hidden PCC {hidden_pcc:.8f} < {D.PCC_BAR}"
    assert logits_pcc >= D.PCC_BAR, f"stream logits PCC {logits_pcc:.8f} < {D.PCC_BAR}"
    assert stream_argmax == legacy_argmax, f"stream argmax {stream_argmax} != legacy argmax {legacy_argmax}"
    assert stream_median < legacy_median, (
        f"streamed 16,400-token prefill {stream_median:.6f}s did not beat "
        f"the legacy 32,768-token bucket {legacy_median:.6f}s"
    )
