# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Each Kimi-K3.1 pipeline stage on one galaxy, traced, at its recorded maximum num_users
(`test_kimi_k3_1_stage_traced`), the four stages run in sequence through saved handoffs
(`test_kimi_k3_1_stage_handoff`), and a stage's kvpe cache read whole at two users (`test_kimi_k3_1_stage_kvpe`);
the last two are described at their sections below.

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
import functools
import json
import os
import struct
import threading
import time
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors.torch import load_file

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.common.prefill.prefill_chunk import metadata_size_bytes
from models.demos.common.prefill.runners.migration import setup_kv_chunk_table
from models.demos.common.prefill.runners.runner_utils import MTP_PAD_TOKEN_ID, compute_layer_split, d2d_activation_rows
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import load_checkpoint_tensors
from models.demos.deepseek_v3_d_p.tests.kimi_k3.stage_handoff import (
    compose_activation,
    load_activation,
    save_activation,
)
from models.demos.deepseek_v3_d_p.tests.test_prefill_chunk_traced_vs_eager import _packed_metadata_msg
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

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


# The runner's mesh: 8x4 torus, the adapter's fabric payload and L1_SMALL, the default trace region.
PLACEMENT = [
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
]


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


def _weight_cache_path() -> Path:
    """The K3.1 TTNN weight cache the runner reads; skips when there is none."""
    path = ADAPTER.weight_cache_path(MESH)
    if not any(path.resolve().glob(".layer_*.complete")):
        pytest.skip(f"no K3.1 TTNN weight cache at {path}")
    return path.resolve()


def _stage_params(
    stage: int,
    num_users: int,
    weight_cache_path: Path,
    full_last_layer: bool = False,
    *,
    num_layers: int | None = None,
    max_seq_len: int = MAX_SEQ_LEN,
    use_trace: bool = True,
):
    """(hf_config, PrefillRunParams) for PP4 rank `stage`, with the K3 manifest's knobs, as the runner sets them.
    `full_last_layer`: the last rank builds its last layer whole, plus the final norm, instead of kv-only, so
    the model produces its final hidden state (the runner's last rank produces none). The depth ladder
    (`test_transformer_depth.py`) builds the stage's first `num_layers` layers, eager, at a short `max_seq_len`;
    fewer layers than the stage holds is a single-rank stack (last rank), as no PP4 rank ends there."""
    first_layer_idx, stage_layers = STAGES[stage]
    num_layers = stage_layers if num_layers is None else num_layers
    is_last_rank = stage == NUM_STAGES - 1 or num_layers < stage_layers
    hf_config = ADAPTER.load_hf_config()
    hf_config.max_seq_len = max_seq_len
    params = PrefillRunParams(
        mesh_shape=MESH,
        num_layers=num_layers,
        first_layer_idx=first_layer_idx,
        is_first_rank=first_layer_idx == 0,
        is_last_rank=is_last_rank,
        max_seq_len=max_seq_len,
        chunk_size=CHUNK,
        num_users=num_users,
        capacity_factor=8,  # the runner's PREFILL_CAPACITY_FACTOR default
        num_links=2,
        gate_mode_name=ADAPTER.default_gate_mode,
        kv_only_last_layer=is_last_rank and not full_last_layer,
        weight_cache_path=weight_cache_path,
        tp_shard_kv=True,
        sparse_kv_cache_format=ADAPTER.default_sparse_kv_cache_format,
        use_trace=use_trace,
    )
    logger.info(
        f"K3.1 stage {stage} layers [{first_layer_idx}, {first_layer_idx + num_layers}) is_last={is_last_rank} "
        f"num_users={num_users} max_seq_len={max_seq_len} trace={use_trace} kvpe={params.sparse_kv_cache_format} "
        f"kv_only_last_layer={params.kv_only_last_layer} cache={weight_cache_path}"
    )
    return hf_config, params


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


@contextlib.contextmanager
def _built_stage(
    stage: int,
    num_users: int,
    weight_cache_path: Path,
    mesh_device,
    *,
    stub=False,
    before_capture=None,
    full_last_layer=False,
):
    """PP4 rank `stage` built as the runner builds it, traced; yields (runtime, kv_caches, final_hidden). Weights
    (header-only stubs if `stub`), every slot's KV caches, compile, the D2D backings, `before_capture(runtime,
    kv_caches)` for what the runner registers between compile and capture, then the capture. DRAM is logged after
    each step. `full_last_layer` (last rank only, see `_stage_params`): `final_hidden()` returns the traced
    forward's output, the final hidden state, which the runtime drops on the last rank; it is the capture's
    output tensor, so each replay refreshes it in place. Otherwise `final_hidden` is None."""
    assert not full_last_layer or stage == NUM_STAGES - 1, "full_last_layer is for the last rank"
    hf_config, params = _stage_params(stage, num_users, weight_cache_path, full_last_layer)
    _log_dram(mesh_device, "opened")
    t0 = time.perf_counter()
    with _header_only() if stub else contextlib.nullcontext():
        runtime = ADAPTER.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    logger.info(f"STAGE_TIME weights {time.perf_counter() - t0:.1f} s")
    weight_bytes = _log_dram(mesh_device, "weights")
    kv_caches = ADAPTER.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    caches = _log_dram(mesh_device, f"kv_caches({num_users} users)") - weight_bytes
    logger.info(f"STAGE_KV {caches} B per chip for {num_users} users ({caches / num_users / 1e6:.1f} MB per user)")
    try:
        t0 = time.perf_counter()
        runtime.compile(kv_caches)
        logger.info(f"STAGE_TIME compile {time.perf_counter() - t0:.1f} s")
        _log_dram(mesh_device, "compiled")
        d2d_backings = _d2d_backings(runtime, mesh_device)
        _log_dram(mesh_device, f"d2d backings ({len(d2d_backings)})")
        if before_capture is not None:
            before_capture(runtime, kv_caches)
        final_hidden = None
        if full_last_layer:
            kept = {}
            forward = runtime.model.forward

            def keep_output(*args, **kwargs):
                kept["hidden"] = forward(*args, **kwargs)
                return kept["hidden"]

            runtime.model.forward = keep_output  # an instance attribute; the runtime ends with this context
            final_hidden = lambda: kept["hidden"]
        t0 = time.perf_counter()
        runtime.capture_trace(kv_caches)
        logger.info(f"STAGE_TIME capture {time.perf_counter() - t0:.1f} s")
        _log_dram(mesh_device, "traced")
        yield runtime, kv_caches, final_hidden
    finally:
        # The captured trace buffers must go before the mesh_device fixture closes the device.
        runtime.release_trace()


def _replay(runtime, mesh_device, chunk_input, kv_caches, slot: int, start: int, end: int):
    """One traced chunk, synchronized and timed (`STAGE_TIME replay`, from the trace's dispatch to the sync; the
    metadata record is built before); returns the rank's outbound activation (None on the last rank)."""
    msg = _packed_metadata_msg(mesh_device, slot, start, end)
    t0 = time.perf_counter()
    out = runtime.prefill_chunk(
        chunk_input, kv_caches, slot_id=slot, actual_start=start, actual_end=end, request_id=0, metadata_msg=msg
    )
    ttnn.synchronize_device(mesh_device)
    logger.info(f"STAGE_TIME replay slot {slot} [{start}, {end}) {(time.perf_counter() - t0) * 1e3:.3f} ms")
    ttnn.deallocate(msg)
    return out


@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3.1 runs on Blackhole only")
@pytest.mark.timeout(0)
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENT, indirect=True)
@pytest.mark.parametrize("stage", range(NUM_STAGES), ids=[f"stage{s}" for s in range(NUM_STAGES)])
def test_kimi_k3_1_stage_traced(stage, mesh_device, device_params, tmp_path):
    num_users = int(os.getenv("K31_STAGE_USERS", STAGE_MAX_USERS[stage]))
    weights = os.getenv("K31_STAGE_WEIGHTS", "stub")
    assert weights in ("real", "stub"), f"$K31_STAGE_WEIGHTS={weights!r}: real or stub"
    weight_cache_path = _weight_cache_path()
    if weights == "stub":
        weight_cache_path = _stub_cache(weight_cache_path, tmp_path)
    logger.info(f"K3.1 stage {stage} weights={weights}")

    stage_built = _built_stage(stage, num_users, weight_cache_path, mesh_device, stub=weights == "stub")
    with stage_built as (runtime, kv_caches, _):
        slot = num_users - 1
        for start in REPLAY_STARTS:
            if runtime.config.is_first_rank:
                chunk_input = runtime.make_chunk_input([1] * CHUNK)
            else:
                chunk_input = _random_activation(runtime, mesh_device, seed=start)
            _replay(runtime, mesh_device, chunk_input, kv_caches, slot, start, start + CHUNK)
        _log_dram(mesh_device, "replayed")
        view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        logger.info(f"STAGE_DRAM capacity {view.total_bytes_per_bank * view.num_banks} B per chip")


# --- The four stages in sequence on one galaxy, through saved handoffs -----------------------------------
#
# Without an SC4 the PP4 pipeline runs one stage per process: each stage is built as its rank is (as above,
# real weights, every slot of the system maximum), takes its input from the previous stage's saved handoff
# (`stage_handoff`: the D2D socket's payload, bit for bit) and saves its own for the next. The runner pieces
# the capacity test leaves out are built here, where the runner builds them, between compile and capture: the
# D2H layer-ack service, so every layer's ack op runs inside the replays and each ack record is read back and
# checked, and the migration export. That export runs in the runner's mock mode, one process per stage, so its
# all-gather of KV stage layouts sees this rank only: each stage writes its own device map, and the chunk table
# (written by rank 0) covers stage 0's layers only. No H2D socket: stage 0 takes the chunk's tokens as the
# runtime's chunk input.
#
# The prompt is the token sequence of Miguel's 10K vLLM trace (`TRACE_10K`): its 8192 prompt ids and the 2048
# completion ids it decoded, teacher-forced, 10240 tokens = two chunks. The model is causal, so every position
# up to the prompt's last (8191) sees what it sees in the 8192-token prefill, and the later positions match
# the trace's decode steps. One difference from the runner's last rank: the last stage builds layer 92 whole
# and the final norm (`full_last_layer`), not kv-only, so the model has a final hidden state, saved per chunk.
# The host then checks the next token (`_check_next_token`) at the prompt's last position and at the last
# position the trace scored. The kv-only last rank is covered by the capacity, kvpe and prompt perf tests.
#
# The prompt runs twice, on the highest slot and then on slot 0: outputs and the slot's cache rows (index,
# KDA) must agree bit for bit. A whole repeat run (all four stages again into another directory) must
# reproduce every saved file byte for byte (`cmp`). Hidden states and caches are checked finite here; their
# accuracy against the trace streams belongs to the accuracy suite. The kvpe cache is too large to read at
# the system maximum; `test_kimi_k3_1_stage_kvpe` covers it.
#
#     for s in 0 1 2 3; do K31_HANDOFF_DIR=/path/run1 scripts/run_safe_pytest.sh \
#         "models/demos/deepseek_v3_d_p/tests/kimi_k3/test_kimi_k3_1_stage.py::test_kimi_k3_1_stage_handoff[blackhole-stage$s-torus-xy-8x4]"; done
HANDOFF_CHUNKS = 2  # chunk 1's DSA layers read chunk 0's KV; its KDA layers start from chunk 0's carry
# Miguel's 10K K3.1-Pro vLLM trace, v2 capture: the same weights, vLLM build and token sequence as the first
# capture (kimi_k31_pro_10k, identical ids and top-1 at every step) plus the DSA indexer streams the accuracy
# suite compares against.
TRACE_10K = Path("/mnt/weka/model-weights/llm/ref-data/debug_traces/kimi_k31_pro_10k_v2/vllm-d06690a4-10240tok")
# Miguel's 128K K3.1-Pro vLLM trace (126,976 prompt tokens): the 128K perf scenario's prompt.
TRACE_128K = Path("/mnt/weka/model-weights/llm/ref-data/debug_traces/kimi_k31_pro_128k_v2/vllm-4984af7c-131072tok")
SYSTEM_MAX_USERS = min(STAGE_MAX_USERS.values())
ACK_FIFO_BYTES = 65536  # the K3 manifest's PREFILL_LAYER_ACK_FIFO_BYTES; holds the warm pass's acks
ACK_TIMEOUT_S = 60


def _handoff_path(handoff_dir: Path, stage: int, chunk: int) -> Path:
    return handoff_dir / f"stage{stage}.chunk{chunk}.activation.bin"


def _final_hidden_path(handoff_dir: Path, chunk: int) -> Path:
    """The last stage's final hidden state (after the final norm), bf16 [1, 1, chunk, hidden], token order."""
    return handoff_dir / f"stage{NUM_STAGES - 1}.chunk{chunk}.final_hidden.bin"


@functools.cache
def _trace_metadata(trace_dir: Path) -> dict:
    """A trace's metadata (token ids, step counts), read once per trace; skips when the trace is absent (a skip
    raises, so it is not cached)."""
    if not (trace_dir / "metadata.json").is_file():
        pytest.skip(f"no K3.1 trace at {trace_dir}")
    with (trace_dir / "metadata.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _trace_stream(name: str) -> torch.Tensor:
    """A single-shard stream of `TRACE_10K` (the top-k streams: one row per vLLM step)."""
    with (TRACE_10K / "index.json").open(encoding="utf-8") as handle:
        (shard,) = json.load(handle)["tensor_streams"][name]["chunks"]
    return load_file(TRACE_10K / shard["path"])[name]


def _teacher_forced_prompt() -> list[int]:
    """The handoff test's prompt: the trace's prompt then its decoded completion, `HANDOFF_CHUNKS` chunks."""
    metadata = _trace_metadata(TRACE_10K)
    ids = metadata["token_ids"] + metadata["completion_token_ids"]
    assert len(ids) == HANDOFF_CHUNKS * CHUNK, f"{TRACE_10K.name}: {len(ids)} tokens, not {HANDOFF_CHUNKS} chunks"
    return ids


def _chunk_input(runtime, mesh_device, handoff_dir: Path, stage: int, chunk: int, prompt: list[int]) -> ttnn.Tensor:
    """Stage 0: the chunk's token ids of `prompt` (whole chunks, padded). Later stages: what the previous stage
    sent for this chunk."""
    if stage == 0:
        return runtime.make_chunk_input(prompt[chunk * CHUNK : (chunk + 1) * CHUNK])
    path = _handoff_path(handoff_dir, stage - 1, chunk)
    if not path.is_file():
        pytest.fail(f"{path} is missing: run stage {stage - 1} into $K31_HANDOFF_DIR={handoff_dir} first")
    first_layer_idx = runtime.config.first_layer_idx
    return load_activation(
        mesh_device,
        path,
        planes=ADAPTER.pipeline_activation_planes(first_layer_idx),
        rows=d2d_activation_rows(CHUNK, sp_factor=MESH[0]),
        hidden=runtime.hf_config.hidden_size,
    )


def _read_acks(d2h_service, count: int) -> list[tuple[int, int, int]]:
    """`count` layer-ack records off the D2H service, each the chunk's (slot, start, end); fails on a stall
    instead of hanging, since read_metadata blocks."""
    records = []

    def read():
        for _ in range(count):
            records.append(struct.unpack_from("<3I", d2h_service.read_metadata()))

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    reader.join(ACK_TIMEOUT_S)
    assert not reader.is_alive(), f"layer acks stalled after {len(records)} of {count}"
    return records


def _slot_state(runtime, kv_caches, slot: int) -> dict[str, torch.Tensor]:
    """What the prompt wrote into `slot`, raw as stored, every chip: the index cache rows of the first
    HANDOFF_CHUNKS chunks (block-cyclic over sp*tp stripes, so per-chip rows [0, chunks * chunk/32)) and the
    slot's KDA recurrent and convolution state. Not the kvpe rows: the packed scaled-FP8 cache (ND-sharded,
    row-major) has no device slice, and the whole cache is ~10 GB per chip at 76 users. Its rows are read by
    chunk 1's DSA attention, so the stage output and the state downstream of it depend on them."""
    mesh_device = runtime.mesh_device
    rows = HANDOFF_CHUNKS * CHUNK // (MESH[0] * MESH[1])
    num_users = runtime.config.num_users
    cache_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=MESH)
    state_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=MESH)

    def read(tensor, composer, seq_rows=None):
        shape = list(tensor.shape)
        per_slot = shape[0] // num_users
        end = [(slot + 1) * per_slot, *shape[1:]]
        if seq_rows is not None:
            end[2] = seq_rows
        # DRAM_MEMORY_CONFIG: slicing an ND-sharded cache into another ND shard misplaces it (read_slot_kv).
        piece = ttnn.slice(
            tensor, [slot * per_slot] + [0] * (len(shape) - 1), end, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        host = ttnn.to_torch(piece, mesh_composer=composer)
        ttnn.deallocate(piece)
        return host

    state = {} if kv_caches.index is None else {"index": read(kv_caches.index, cache_composer, rows)}
    state["kda_recurrent"] = read(kv_caches.kda_states.recurrent, state_composer)
    state["kda_convolution"] = read(kv_caches.kda_states.convolution, state_composer)
    return state


def _write_raw(tensor: torch.Tensor, path: Path) -> None:
    tensor.contiguous().view(torch.uint8).numpy().tofile(path)


def _check_next_token(final_hidden: dict[int, torch.Tensor]) -> None:
    """The next token on host: checkpoint `lm_head.weight` times the final hidden state (already after the final
    norm), float32, argmax, against the trace's recorded top-1 (`topk_indices`, vLLM's own logits).

    The trace sampled (temperature 1, top-p 0.95), so the token it fed next is a draw, not its argmax; it is
    reported against the host top-k only. Its top-k rows are one per vLLM step: each prefill step's last
    position, then each decode step's. Two positions are checked: the prompt's last (row 1, the 8192-token
    prefill's next token) and the last position the trace scored (the last row, the second-to-last token)."""
    metadata = _trace_metadata(TRACE_10K)
    ids = _teacher_forced_prompt()
    n_prompt = metadata["n_prompt_tokens"]
    assert metadata["prefill_steps"] == 2, "top-k row 1 is the prompt's end only for 2 prefill steps"
    trace_ids, trace_logprobs = _trace_stream("topk_indices"), _trace_stream("topk_logprobs").float()
    assert len(trace_ids) == len(ids) - n_prompt + 1, f"{len(trace_ids)} top-k rows for {len(ids) - n_prompt} decodes"
    checked = {n_prompt - 1: 1, len(ids) - 2: len(trace_ids) - 1}  # position -> top-k row

    checkpoint = Path(os.getenv("PREFILL_HF_MODEL") or ADAPTER.hf_model_default)
    lm_head = load_checkpoint_tensors(checkpoint, ["lm_head.weight"])["lm_head.weight"].float()
    mismatches = []
    for position, row in checked.items():
        hidden = final_hidden[position // CHUNK][0, 0, position % CHUNK].float()
        logits = lm_head @ hidden
        logprobs = torch.log_softmax(logits, dim=-1)
        top = torch.topk(logits, trace_ids.shape[-1])
        host_top, trace_top = top.indices.tolist(), trace_ids[row].tolist()
        argmax, expected, sampled = host_top[0], trace_top[0], ids[position + 1]
        margin = float(logits[argmax] - logits[expected])
        sampled_rank = host_top.index(sampled) if sampled in host_top else None
        logger.info(
            f"NEXT_TOKEN position {position} (trace top-k row {row}): host argmax {argmax}, trace top-1 {expected}, "
            f"logit margin argmax - trace top-1 {margin:.4f}, "
            f"host top1-top2 logit gap {float(top.values[0] - top.values[1]):.4f}; "
            f"host top-5 {host_top[:5]} logprobs {[round(float(logprobs[i]), 4) for i in host_top[:5]]}; "
            f"trace top-5 {trace_top[:5]} logprobs {[round(v, 4) for v in trace_logprobs[row, :5].tolist()]}; "
            f"top-5 overlap {len(set(host_top[:5]) & set(trace_top[:5]))}/5; "
            f"host logprob of trace top-1 {float(logprobs[expected]):.4f}; "
            f"sampled next id {sampled} at host top-{len(host_top)} rank {sampled_rank}"
        )
        if argmax != expected:
            mismatches.append(f"position {position}: host {argmax}, trace {expected}, logit margin {margin:.4f}")
    assert not mismatches, f"next token differs from the trace's top-1: {mismatches}"


def _cached_stage(stage: int) -> Path:
    """The weight cache, skipping when any layer of the stage is not cached yet."""
    weight_cache_path = _weight_cache_path()
    first_layer_idx, num_layers = STAGES[stage]
    layers = range(first_layer_idx, first_layer_idx + num_layers)
    uncached = [i for i in layers if not (weight_cache_path / f".layer_{i}.complete").is_file()]
    if uncached:
        pytest.skip(f"layers {uncached} are not in the K3.1 TTNN cache yet")
    return weight_cache_path


def _handoff_dir() -> Path:
    handoff_dir = Path(os.getenv("K31_HANDOFF_DIR", "generated/kimi_k3_1_handoff"))
    handoff_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"K3.1 handoff dir {handoff_dir.resolve()}")
    return handoff_dir


@contextlib.contextmanager
def _served_stage(stage: int, num_users: int, handoff_dir: Path, mesh_device, monkeypatch, full_last_layer=False):
    """PP4 rank `stage` with real weights and the runner pieces built between compile and capture: the D2H
    layer-ack service and the migration export (see above). Yields (runtime, kv_caches, d2h_service, acks per
    chunk, final_hidden) once the capture's warm-pass acks are read; `full_last_layer` and `final_hidden` as in
    `_built_stage`."""
    weight_cache_path = _cached_stage(stage)
    first_layer_idx, num_layers = STAGES[stage]
    d2h_service = None

    def before_capture(runtime, kv_caches):
        nonlocal d2h_service
        d2h_service = ttnn.D2HStreamService(
            mesh_device,
            global_spec=None,
            fifo_size_bytes=ACK_FIFO_BYTES,
            worker_cores=ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),  # the runner's SYNC_WORKER_CORES
            metadata_size_bytes=metadata_size_bytes(1),
            metadata_ring=True,
        )
        runtime.set_d2h_ack_service(d2h_service)
        _log_dram(mesh_device, "d2h ack service")
        # Per-process gather (see above): a device map per stage, since each export removes the maps of its
        # path; the table is stage 0's.
        monkeypatch.setenv("PREFILL_ENABLE_MIGRATION", "0")
        monkeypatch.setenv("PREFILL_MIGRATION_TABLE_PATH", str(handoff_dir / "kv_chunk_table.pb"))
        monkeypatch.setenv("PREFILL_MIGRATION_DEVICE_MAP_PATH", str(handoff_dir / f"stage{stage}.kv_device_map.json"))
        setup_kv_chunk_table(
            runtime,
            kv_caches,
            mesh_device,
            MESH,
            rank=stage,
            num_ranks=NUM_STAGES,
            first_layer_idx=first_layer_idx,
            num_my_layers=num_layers,
        )
        _log_dram(mesh_device, "migration export")

    stage_built = _built_stage(
        stage, num_users, weight_cache_path, mesh_device, before_capture=before_capture, full_last_layer=full_last_layer
    )
    with stage_built as (runtime, kv_caches, final_hidden):
        acks_per_chunk = runtime.warmup_ack_count()
        assert acks_per_chunk >= num_layers, f"{acks_per_chunk} acks per chunk for {num_layers} layers"
        _read_acks(d2h_service, acks_per_chunk)  # the capture's warm pass
        yield runtime, kv_caches, d2h_service, acks_per_chunk, final_hidden


@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3.1 runs on Blackhole only")
@pytest.mark.timeout(0)
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENT, indirect=True)
@pytest.mark.parametrize("stage", range(NUM_STAGES), ids=[f"stage{s}" for s in range(NUM_STAGES)])
def test_kimi_k3_1_stage_handoff(stage, mesh_device, device_params, monkeypatch):
    handoff_dir = _handoff_dir()
    num_users = int(os.getenv("K31_STAGE_USERS", SYSTEM_MAX_USERS))
    prompt = _teacher_forced_prompt()  # skips before the weight load when the trace is absent
    is_last_rank = stage == NUM_STAGES - 1
    served = _served_stage(stage, num_users, handoff_dir, mesh_device, monkeypatch, full_last_layer=is_last_rank)
    with served as (runtime, kv_caches, d2h_service, acks_per_chunk, final_hidden):
        outputs, states = {}, {}
        for slot in (num_users - 1, 0):
            for chunk in range(HANDOFF_CHUNKS):
                start, end = chunk * CHUNK, (chunk + 1) * CHUNK
                chunk_input = _chunk_input(runtime, mesh_device, handoff_dir, stage, chunk, prompt)
                out = _replay(runtime, mesh_device, chunk_input, kv_caches, slot, start, end)
                acks = _read_acks(d2h_service, acks_per_chunk)
                assert set(acks) == {
                    (slot, start, end)
                }, f"ack records {sorted(set(acks))} for chunk {(slot, start, end)}"
                if is_last_rank:
                    assert out is None
                    host = compose_activation(mesh_device, final_hidden())
                    hidden_shape = (1, 1, CHUNK, runtime.hf_config.hidden_size)
                    assert tuple(host.shape) == hidden_shape, f"final hidden {tuple(host.shape)}, not {hidden_shape}"
                    _write_raw(host, _final_hidden_path(handoff_dir, chunk))
                else:
                    host = save_activation(mesh_device, out, _handoff_path(handoff_dir, stage, chunk))
                assert torch.isfinite(host).all(), f"stage {stage} chunk {chunk} slot {slot}: non-finite output"
                if chunk in outputs:
                    assert torch.equal(outputs[chunk], host), f"stage {stage} chunk {chunk}: slot {slot} differs"
                outputs[chunk] = host
            states[slot] = _slot_state(runtime, kv_caches, slot)
        _log_dram(mesh_device, "replayed")

        first, again = states[num_users - 1], states[0]
        for name, raw in first.items():
            assert torch.equal(raw.view(torch.uint8), again[name].view(torch.uint8)), f"{name}: slot 0 differs"
            _write_raw(raw, handoff_dir / f"stage{stage}.{name}.bin")
        for name, values in first.items():
            assert torch.isfinite(values.float()).all(), f"stage {stage}: non-finite {name}"
        if "index" in first:
            unwritten = first["index"].abs().amax(dim=-1).eq(0).sum()
            assert not unwritten, f"stage {stage}: {unwritten} index cache rows of the prompt are unwritten"
        logger.info(f"STAGE_HANDOFF stage {stage}: {sorted(p.name for p in handoff_dir.glob(f'stage{stage}.*'))}")
    if is_last_rank:
        _check_next_token(outputs)


# --- The kvpe cache of a stage, read whole, at two users ------------------------------------------------
#
# The handoff test reads no kvpe rows (no device slice of the packed scaled-FP8 cache, and ~10 GB per chip at
# the system maximum). Here the stage is built with two slots, ~0.26 GB of kvpe per chip, read whole as stored
# (every chip's shard, raw bytes). The prompt runs on slot 1, then slot 0, from the handoff test's inputs:
# - every kvpe row of the prompt (both chunks, every DSA layer, the last rank's kv-only layer too) is written
#   (differs from the allocator's zero fill) and is the same on both slots;
# - a negative control, on a non-last rank: chunk 0 on slot 1, then the kvpe cache alone reset to its zero
#   fill (index cache and KDA carry kept), then chunk 1: its output must differ from the full prompt's, so
#   chunk 1's output depends on the kvpe rows chunk 0 wrote. The prompt rerun on that slot then restores it.
KVPE_USERS = 2


def _shards(mesh_device, tensor) -> torch.Tensor:
    """Every chip's shard as stored, raw bytes: [rows * per-chip dim 0, cols * per-chip dim 1, ...]."""
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=MESH)
    return ttnn.to_torch(tensor, mesh_composer=composer).view(torch.uint8)


def _kvpe_prompt_rows(raw: torch.Tensor, slot: int) -> torch.Tensor:
    """One slot's kvpe rows of the prompt's chunks, every chip: [mesh rows, slabs, mesh cols, rows, row bytes].
    Slabs are slot-major on each chip; with TP-deduped KV a chip holds rows [0, chunks * chunk / (sp * tp))."""
    per_chip_users_slabs = raw.shape[0] // MESH[0]
    slabs = per_chip_users_slabs // KVPE_USERS
    by_slot = raw.reshape(MESH[0], KVPE_USERS, slabs, *raw.shape[1:])
    return by_slot[:, slot, :, :, : HANDOFF_CHUNKS * CHUNK // (MESH[0] * MESH[1])].clone()


@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3.1 runs on Blackhole only")
@pytest.mark.timeout(0)
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENT, indirect=True)
@pytest.mark.parametrize("stage", range(NUM_STAGES), ids=[f"stage{s}" for s in range(NUM_STAGES)])
def test_kimi_k3_1_stage_kvpe(stage, mesh_device, device_params):
    handoff_dir = _handoff_dir()
    weight_cache_path = _cached_stage(stage)
    prompt = _teacher_forced_prompt()
    with _built_stage(stage, KVPE_USERS, weight_cache_path, mesh_device) as (runtime, kv_caches, _):
        kvpe = kv_caches.kvpe.storage
        fresh = kvpe.cpu()  # the allocator's zero fill

        def run(slot, reset_kvpe_before=()):
            """The prompt's chunks on `slot`; the stage's outbound activations on host (None on the last rank)."""
            outputs = []
            for chunk in range(HANDOFF_CHUNKS):
                if chunk in reset_kvpe_before:
                    ttnn.copy_host_to_device_tensor(fresh, kvpe)
                chunk_input = _chunk_input(runtime, mesh_device, handoff_dir, stage, chunk, prompt)
                start, end = chunk * CHUNK, (chunk + 1) * CHUNK
                out = _replay(runtime, mesh_device, chunk_input, kv_caches, slot, start, end)
                outputs.append(None if out is None else compose_activation(mesh_device, out))
            return outputs

        fresh_rows = _kvpe_prompt_rows(_shards(mesh_device, kvpe), 1)
        assert fresh_rows.shape[-1] == kvpe.shape[-1], f"kvpe read back as {fresh_rows.dtype}, not raw bytes"
        outputs = run(1)
        written = _kvpe_prompt_rows(_shards(mesh_device, kvpe), 1)
        unwritten = (written == fresh_rows).all(dim=-1).sum()
        assert not unwritten, f"stage {stage}: {unwritten} of {written[..., 0].numel()} kvpe rows unwritten"
        again = run(0)
        assert torch.equal(_kvpe_prompt_rows(_shards(mesh_device, kvpe), 0), written), "kvpe: slot 0 differs"
        for chunk, (first, second) in enumerate(zip(outputs, again)):
            assert first is None or torch.equal(first, second), f"stage {stage} chunk {chunk}: slot 0 differs"
        logger.info(f"STAGE_KVPE stage {stage}: {written[..., 0].numel()} prompt rows written, equal on both slots")

        if runtime.config.is_last_rank:
            assert outputs == [None] * HANDOFF_CHUNKS
            return
        control = run(1, reset_kvpe_before=(1,))[1]
        assert not torch.equal(control, outputs[1]), f"stage {stage}: chunk 1 does not read chunk 0's kvpe rows"
        mismatched = (control != outputs[1]).float().mean().item()
        assert torch.equal(run(1)[1], outputs[1]), f"stage {stage}: the prompt rerun after the reset differs"
        logger.info(f"STAGE_KVPE stage {stage}: kvpe reset before chunk 1 changes {mismatched:.1%} of its output")


# --- The 10K and 128K scenarios, stage by stage: traced chunk periods and a device-profiler record per program ----
#
# The prompt of one of Miguel's vLLM traces (10K: 8,192 tokens, two chunks; 128K: 126,976 tokens, 25 chunks; the last
# chunk padded past the prompt as the producer pads it) through each stage as it is served (`_served_stage`), stage 0
# from the prompt's token ids and each later stage from the previous stage's saved handoff, so every stage routes its
# experts on the real prompt. Run the stages of one prompt in order into one $K31_HANDOFF_DIR.
#
# Timing follows K3's `test_prefill_perf`: whole passes of the prompt (every chunk, in order, as a request
# arrives), each traced replay synchronized and timed on host (`STAGE_TIME replay` lines; one stage's chunk time is
# its retire period in the PP4 pipeline). Then one more pass under the real-time device profiler, one program
# record per chip, and, on the 128K prompt, the same for one chunk at the deepest position of the per-user cache (the
# 1M probe of ao3i.1.6), after the slot's rows between the prompt and the probe are filled by the model (below).
# Each profiled chunk's programs go to `stage{s}.{label}.programs.json` in dispatch order with each chip's
# [start, end] ns on that chip's clock, for attribution off device: programs overlap (sub-device regions), so
# durations alone do not partition the chunk, and a program without records (the DSA indexer's top-k on its
# sub-device) shows only as a gap in a chip's timeline.
#
#     for s in 0 1 2 3; do K31_HANDOFF_DIR=/path/perf128k scripts/run_safe_pytest.sh \
#         "models/demos/deepseek_v3_d_p/tests/kimi_k3/test_kimi_k3_1_stage.py::test_kimi_k3_1_stage_prompt_perf[blackhole-stage$s-128k-torus-xy-8x4]"; done
PROMPT_TRACES = {"10k": TRACE_10K, "128k": TRACE_128K}
PERF_PASSES = 5  # after the capture's warm pass; K3's test_prefill_perf times 5 iterations
DEEP_START = MAX_SEQ_LEN - CHUNK


def _trace_prompt(trace_dir: Path) -> tuple[list[int], int]:
    """(the trace's prompt token ids padded to whole chunks with the producer's pad id, the prompt's length);
    skips when the trace is absent."""
    metadata = _trace_metadata(trace_dir)
    ids = metadata["token_ids"]
    assert len(ids) == metadata["n_prompt_tokens"], f"{len(ids)} token ids for {metadata['n_prompt_tokens']} tokens"
    return ids + [MTP_PAD_TOKEN_ID] * (-len(ids) % CHUNK), len(ids)


def _profiled_replay(mesh_device, run, path: Path) -> None:
    """`run` (one traced chunk) under the real-time profiler; writes its programs (dispatch order, per-chip ns) to
    `path`. Fails rather than write a partial file: on no records or dropped ones (the profiler raises), and on a
    set cut short at the profiler's collection deadline (some program without a record from every chip)."""
    programs = {}
    _, records = profile_realtime_program(mesh_device, run, collect_all=True)
    for record in records:  # arrival order is dispatch order
        program = programs.setdefault(
            record["runtime_id"], {"runtime_id": record["runtime_id"], "kernel_sources": record["kernel_sources"]}
        )
        program.setdefault("chip_ns", {})[record["chip_id"]] = (record["start_ns"], record["end_ns"])
    chips = mesh_device.get_num_devices()
    assert len(records) == len(programs) * chips, f"{len(records)} records for {len(programs)} programs x {chips} chips"
    path.write_text(json.dumps(list(programs.values())))
    critical_path = sum(max(end - start for start, end in p["chip_ns"].values()) for p in programs.values())
    logger.info(
        f"STAGE_PROFILE {path.name}: {len(programs)} programs, {len(records)} records, "
        f"sum of per-program max over chips {critical_path / 1e6:.3f} ms"
    )


@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3.1 runs on Blackhole only")
@pytest.mark.timeout(0)
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENT, indirect=True)
@pytest.mark.parametrize("trace_dir", PROMPT_TRACES.values(), ids=PROMPT_TRACES.keys())
@pytest.mark.parametrize("stage", range(NUM_STAGES), ids=[f"stage{s}" for s in range(NUM_STAGES)])
def test_kimi_k3_1_stage_prompt_perf(stage, trace_dir, mesh_device, device_params, monkeypatch):
    handoff_dir = _handoff_dir()
    prompt, prompt_len = _trace_prompt(trace_dir)
    chunks = [(c * CHUNK, min((c + 1) * CHUNK, prompt_len)) for c in range(len(prompt) // CHUNK)]
    with _served_stage(stage, SYSTEM_MAX_USERS, handoff_dir, mesh_device, monkeypatch) as (
        runtime,
        kv_caches,
        d2h_service,
        acks_per_chunk,
        _,
    ):
        slot = SYSTEM_MAX_USERS - 1

        def replay(chunk, start, end, profile_label=None):
            """One chunk, timed (`STAGE_TIME replay`) or profiled (`STAGE_PROFILE` follows its replay line), its layer
            acks checked; returns its output (None when profiled or on the last rank)."""
            chunk_input = _chunk_input(runtime, mesh_device, handoff_dir, stage, chunk, prompt)
            run = lambda: _replay(runtime, mesh_device, chunk_input, kv_caches, slot, start, end)
            out = None
            if profile_label is None:
                out = run()
            else:
                _profiled_replay(mesh_device, run, handoff_dir / f"stage{stage}.{profile_label}.programs.json")
            acks = _read_acks(d2h_service, acks_per_chunk)
            assert set(acks) == {(slot, start, end)}, f"ack records {sorted(set(acks))} for chunk {(slot, start, end)}"
            return out

        for index in range(PERF_PASSES):
            for chunk, (start, end) in enumerate(chunks):
                out = replay(chunk, start, end)
                if out is not None and index == 0:
                    host = save_activation(mesh_device, out, _handoff_path(handoff_dir, stage, chunk))
                    assert torch.isfinite(host).all(), f"stage {stage} chunk {chunk}: non-finite output"
        for chunk, (start, end) in enumerate(chunks):
            replay(chunk, start, end, profile_label=f"chunk{chunk}")
        # The deep probe: the last chunk of the per-user cache, fed chunk 0's input (timing, not values). First the
        # rows up to it are written by the model: the prompt's chunks again, cyclically, as whole chunks at those
        # positions, so the probe's indexer scores ~1M keys and KV rows the model wrote. Left zero-filled, ~930K keys
        # score exactly 0 and every query short of 2,048 positive scores selects the same tied rows, which slows
        # sparse_sdpa (ao3i.2.6). The fill's replays are logged like any other. Not on the 10K prompt: cycling two
        # chunks gives every key ~100 copies on one DRAM bank, an artifact of its own (ao3i.2.8).
        if trace_dir != TRACE_128K:
            return
        for start in range(len(chunks) * CHUNK, DEEP_START, CHUNK):
            replay(start // CHUNK % len(chunks), start, start + CHUNK)
        for _ in range(PERF_PASSES):
            replay(0, DEEP_START, DEEP_START + CHUNK)
        replay(0, DEEP_START, DEEP_START + CHUNK, profile_label="deep")
