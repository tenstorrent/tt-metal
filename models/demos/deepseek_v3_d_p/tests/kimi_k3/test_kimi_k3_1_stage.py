# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Each Kimi-K3.1 pipeline stage on one galaxy, traced, at its recorded maximum num_users.

A stage is built as `prefill_runner` builds that rank (`compute_layer_split` over the adapter's split
boundaries, `build_runtime` / `allocate_kv_cache` / `compile` / `capture_trace` with the K3 manifest's
knobs: 8x4 torus, TP-deduped KV, the adapter's scaled-FP8 kvpe, trace on, kv-only last layer on the last
rank, no lm_head) from the K3.1 TTNN weight cache. The inter-stage D2D endpoints keep their DRAM: the
receiver's and sender's activation backing tensors are allocated where the runner allocates them, after
compile and before capture. What is replaced is what moves data: the input activation (the live stream
plus the AttnRes snapshots sealed upstream) is a random tensor of the size the socket delivers, the
per-chunk metadata is a packed record instead of the socket's, as `test_prefill_chunk_traced_vs_eager.py`
does, and no H2D service, layer ack or migration export is built.

It gates the recorded capacity (`STAGE_MAX_USERS`): weights, KV caches for every slot, D2D backings,
trace and a replay must fit one chip's DRAM. Replays go to the highest slot from the start of the per-user
cache to its end, so the whole allocation is addressed. DRAM per chip is logged after each phase
(`STAGE_DRAM` lines). One more user fails in compile: the forward's buffers find no contiguous block.

Overrides for a capacity sweep: `$K31_STAGE_USERS` (num_users) and `$K31_STAGE_WEIGHTS`:

* `stub` (default): every weight is allocated on device with the spec and mesh topology of its cache
  file (the file is mapped and only its header read) and left uninitialized, so a run costs no weight
  load. A layer not yet cached takes the files of a cached layer of the same kind (full attention or
  KDA, dense or MoE FFN): the shapes depend on the kind alone. Byte-identical to `real` in DRAM and in
  the pass/fail boundary (stages 0 and 3).
* `real`: the stage's cached weights, to confirm a stub result; every layer of the stage must be cached.

    K31_STAGE_USERS=77 scripts/run_safe_pytest.sh \
        "models/demos/deepseek_v3_d_p/tests/kimi_k3/test_kimi_k3_1_stage.py::test_kimi_k3_1_stage_traced[blackhole-stage0-torus-xy-8x4]"
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.common.prefill.runners.runner_utils import compute_layer_split, d2d_activation_rows
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.test_prefill_chunk_traced_vs_eager import _packed_metadata_msg
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert

ADAPTER = get_adapter("kimi_k3_1")
MESH = (8, 4)
NUM_STAGES = 4
# The K3 manifest's context; a multiple of 40960 (allocator 8192 x chunk 5120).
MAX_SEQ_LEN = 1_064_960
CHUNK = 5120
TRACE_REGION_SIZE = 256 * 1024 * 1024  # the runner's PREFILL_TRACE_REGION_SIZE default
# Replay positions on one slot: the first two chunks, then deeper prefixes up to the last chunk of the
# per-user cache, so the DSA indexer's on-device prefix length is exercised and timed at each depth.
REPLAY_STARTS = (0, CHUNK, 51_200, 512_000, MAX_SEQ_LEN - CHUNK)


# (first layer, layer count) of each PP4 rank: 0-23 / 24-47 / 48-71 / 72-92.
STAGES = compute_layer_split(
    KimiK31Config.NUM_LAYERS, NUM_STAGES, ADAPTER.layer_split_boundaries(KimiK31Config.NUM_LAYERS)
)
# Measured on bh-glx-110-c10u14 (ao3i.1.6): the most users each stage fits at MAX_SEQ_LEN; one more fails.
# The PP4 system maximum is their minimum, 76.
STAGE_MAX_USERS = {0: 76, 1: 77, 2: 77, 3: 93}


def _layer_kind(layer_idx: int) -> tuple[bool, bool]:
    """What a layer's weight shapes depend on: full attention (vs KDA), and dense FFN (vs MoE)."""
    return layer_idx in KimiK31Config.mla_layer_ids(), layer_idx < KimiK31Config.NUM_DENSE_LAYERS


def _stub_cache(real: Path, stub: Path) -> Path:
    """A cache directory of symlinks in which every layer resolves: a cached layer to its own files, an
    uncached one to those of the first cached layer of its kind, renamed. Shapes only; see `_header_only`."""
    marker = ".layer_{}.complete"
    cached = sorted(i for i in range(KimiK31Config.NUM_LAYERS) if (real / marker.format(i)).is_file())
    layer_files = {}
    for path in real.iterdir():
        name = path.name
        if name.startswith("layer_") or name.startswith("attn_res.layers."):
            layer = int(name.removeprefix("layer_").removeprefix("attn_res.layers.").split(".")[0])
            layer_files.setdefault(layer, []).append(name)
        elif not name.startswith(".layer_"):
            (stub / name).symlink_to(path)  # model-level tensors: embedding, final norm, AttnRes output
    for layer in range(KimiK31Config.NUM_LAYERS):
        source = next((c for c in cached if _layer_kind(c) == _layer_kind(layer)), None)
        source = layer if layer in cached else source
        assert source is not None, f"no cached layer of layer {layer}'s kind {_layer_kind(layer)} in {real}"
        for name in layer_files[source]:
            renamed = name.replace(f"layer_{source}.", f"layer_{layer}.", 1).replace(
                f"attn_res.layers.{source}.", f"attn_res.layers.{layer}.", 1
            )
            (stub / renamed).symlink_to(real / name)
        (stub / marker.format(layer)).symlink_to(real / marker.format(source))
    logger.info(f"stub weight cache {stub}: layers {cached} cached, the rest stand in by kind")
    return stub


def _allocate_like(host: ttnn.Tensor, device, memory_config=None) -> ttnn.Tensor:
    """An uninitialized device tensor with the host tensor's per-chip spec and mesh topology."""
    spec = host.spec
    if memory_config is None:
        tensor = ttnn.allocate_tensor_on_device(spec, device)
    else:  # placed with another memory config, as the routed experts are (32x32 tiles)
        tensor = ttnn.allocate_tensor_on_device(spec.shape, spec.dtype, spec.layout, device, memory_config)
    tensor.update_tensor_topology(host.tensor_topology())
    return tensor


@contextlib.contextmanager
def _header_only():
    """Weights are allocated on device, uninitialized, instead of uploaded. A cache load to host maps the
    file and reads only its header, so spec and topology cost no data read. Two upload paths: a cache
    load straight to device, and the routed experts', which load to host and place with `ttnn.to_device`
    (stubbed only inside that loader, so no other host->device copy of the build is touched)."""
    tensor_module = ttnn._ttnn.tensor
    load = tensor_module.load_tensor_flatbuffer
    load_experts = TtRoutedExpert.__dict__["_convert_and_cache_expert_weights"]

    def load_header(file_name, device=None):
        host = load(file_name, None)
        return host if device is None else _allocate_like(host, device)

    def load_experts_header(*args, **kwargs):
        to_device = ttnn.to_device
        ttnn.to_device = lambda host, device, memory_config=None, **_: _allocate_like(host, device, memory_config)
        try:
            return load_experts.__func__(*args, **kwargs)
        finally:
            ttnn.to_device = to_device

    tensor_module.load_tensor_flatbuffer = load_header
    TtRoutedExpert._convert_and_cache_expert_weights = staticmethod(load_experts_header)
    try:
        yield
    finally:
        tensor_module.load_tensor_flatbuffer = load
        TtRoutedExpert._convert_and_cache_expert_weights = load_experts


def _dram_per_chip(mesh_device) -> tuple[int, int]:
    """(allocated, largest contiguous free) bytes per chip; the mesh allocator is identical on every chip."""
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return view.total_bytes_allocated_per_bank * view.num_banks, view.largest_contiguous_bytes_free_per_bank


def _log_dram(mesh_device, phase: str) -> int:
    allocated, largest_free = _dram_per_chip(mesh_device)
    l1_small = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL).total_bytes_allocated_per_bank
    logger.info(
        f"STAGE_DRAM {phase}: {allocated} B ({allocated / 2**30:.3f} GiB) allocated per chip, largest free "
        f"block {largest_free / 2**20:.1f} MiB per bank; L1_SMALL {l1_small} B/bank"
    )
    return allocated


def _d2d_backings(runtime, mesh_device) -> list[ttnn.Tensor]:
    """The DRAM the runner's D2D receiver (non-first rank) and sender (non-last rank) each allocate for the
    activation they move: its per-chip slice of `activation_global_spec`, planes taken at that boundary."""
    config = runtime.config
    rows = d2d_activation_rows(CHUNK, sp_factor=MESH[0]) // MESH[0]
    width = runtime.hf_config.hidden_size // (MESH[1] if ADAPTER.pipeline_activation_emb_tp_sharded else 1)
    boundaries = []
    if not config.is_first_rank:
        boundaries.append(config.first_layer_idx)
    if not config.is_last_rank:
        boundaries.append(config.first_layer_idx + config.num_layers)
    return [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, ADAPTER.pipeline_activation_planes(boundary), rows, width]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        for boundary in boundaries
    ]


def _random_activation(runtime, mesh_device, seed: int) -> ttnn.Tensor:
    """What the upstream rank sends, at the socket's size, with random values: zeros would route every token
    to the same experts. A fresh tensor per chunk: the traced path frees its input after staging it."""
    placeholder = runtime.make_placeholder_activation()
    shape = tuple(placeholder.shape)
    ttnn.deallocate(placeholder)
    generator = torch.Generator().manual_seed(seed)
    return ttnn.from_torch(
        torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3.1 runs on Blackhole only")
@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            MESH,
            torus_xy_device_params(
                fabric_payload_size=ADAPTER.fabric_payload_size(),
                l1_small_size=ADAPTER.l1_small_size,
                trace_region_size=TRACE_REGION_SIZE,
            ),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=MESH, topology="mesh-8x4"),
            id="torus-xy-8x4",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("stage", range(NUM_STAGES), ids=[f"stage{s}" for s in range(NUM_STAGES)])
def test_kimi_k3_1_stage_traced(stage, mesh_device, device_params, tmp_path):
    first_layer_idx, num_layers = STAGES[stage]
    num_users = int(os.getenv("K31_STAGE_USERS", STAGE_MAX_USERS[stage]))
    weights = os.getenv("K31_STAGE_WEIGHTS", "stub")
    assert weights in ("real", "stub"), f"$K31_STAGE_WEIGHTS={weights!r}: real or stub"
    weight_cache_path = ADAPTER.weight_cache_path(MESH)
    if not any(weight_cache_path.resolve().glob(".layer_*.complete")):
        pytest.skip(f"no K3.1 TTNN weight cache at {weight_cache_path}")
    if weights == "stub":
        weight_cache_path = _stub_cache(weight_cache_path.resolve(), tmp_path)
    is_last_rank = stage == NUM_STAGES - 1
    hf_config = ADAPTER.load_hf_config()
    hf_config.max_seq_len = MAX_SEQ_LEN
    params = PrefillRunParams(
        mesh_shape=MESH,
        num_layers=num_layers,
        first_layer_idx=first_layer_idx,
        is_first_rank=first_layer_idx == 0,
        is_last_rank=is_last_rank,
        max_seq_len=MAX_SEQ_LEN,
        chunk_size=CHUNK,
        num_users=num_users,
        capacity_factor=8,  # the runner's PREFILL_CAPACITY_FACTOR default
        num_links=2,
        gate_mode_name=ADAPTER.default_gate_mode,
        kv_only_last_layer=is_last_rank,
        weight_cache_path=weight_cache_path,
        tp_shard_kv=True,
        sparse_kv_cache_format=ADAPTER.default_sparse_kv_cache_format,
        use_trace=True,
    )
    logger.info(
        f"K3.1 stage {stage} layers [{first_layer_idx}, {first_layer_idx + num_layers}) is_last={is_last_rank} "
        f"num_users={num_users} weights={weights} max_seq_len={MAX_SEQ_LEN} kvpe={params.sparse_kv_cache_format} "
        f"cache={params.weight_cache_path}"
    )

    _log_dram(mesh_device, "opened")
    t0 = time.perf_counter()
    with _header_only() if weights == "stub" else contextlib.nullcontext():
        runtime = ADAPTER.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    logger.info(f"STAGE_TIME weights {time.perf_counter() - t0:.1f} s")
    weight_bytes = _log_dram(mesh_device, "weights")
    kv_caches = ADAPTER.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    caches = _log_dram(mesh_device, f"kv_caches({num_users} users)") - weight_bytes
    logger.info(f"STAGE_KV {caches} B per chip for {num_users} users ({caches / num_users / 1e6:.1f} MB per user)")
    try:
        t0 = time.perf_counter()
        runtime.compile(kv_caches)
        d2d_backings = _d2d_backings(runtime, mesh_device)
        _log_dram(mesh_device, f"d2d backings ({len(d2d_backings)})")
        runtime.capture_trace(kv_caches)
        logger.info(f"STAGE_TIME compile+capture {time.perf_counter() - t0:.1f} s")
        _log_dram(mesh_device, "traced")

        slot = num_users - 1
        for start in REPLAY_STARTS:
            if runtime.config.is_first_rank:
                chunk_input = runtime.make_chunk_input([1] * CHUNK)
            else:
                chunk_input = _random_activation(runtime, mesh_device, seed=start)
            msg = _packed_metadata_msg(mesh_device, slot, start, start + CHUNK)
            t0 = time.perf_counter()
            runtime.prefill_chunk(
                chunk_input,
                kv_caches,
                slot_id=slot,
                actual_start=start,
                actual_end=start + CHUNK,
                request_id=0,
                metadata_msg=msg,
            )
            ttnn.synchronize_device(mesh_device)
            logger.info(f"STAGE_TIME replay slot {slot} [{start}, {start + CHUNK}) {time.perf_counter() - t0:.3f} s")
            ttnn.deallocate(msg)
        _log_dram(mesh_device, "replayed")
        view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        logger.info(f"STAGE_DRAM capacity {view.total_bytes_per_bank * view.num_banks} B per chip")
    finally:
        # The captured trace buffers must go before the mesh_device fixture closes the device.
        runtime.release_trace()
