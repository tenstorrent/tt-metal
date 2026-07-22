# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Model-agnostic utilities for the prefill runner.

Only metal-native helpers live here — pure ttnn, no upward dependencies
on blaze (`_migration`, `_mpi_test_helpers`) and no dependency on any specific
model package. Migration-coupled diagnostics live in blaze at
`disaggregation/migration/python/prefill_runner_util.py`.

Per-model plumbing (which model to build, where its weights/config/trace live,
how to allocate its KV cache, how to call chunked prefill) lives behind the
PrefillModelAdapter (../adapter.py), NOT here. Model-specific KV diagnostics /
PCC live in the model package's own runner_utils.
"""

import os
from pathlib import Path

from loguru import logger

import ttnn


def _create_fabric_router_config(max_payload_size):
    """FabricRouterConfig with a custom max payload size. Inlined here (a 3-line
    ttnn wrapper) so this common module needs no model-package import."""
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


# ---------------------------------------------------------------------------
# Device / H2D-service setup
# ---------------------------------------------------------------------------
def open_mesh_device(
    mesh_shape: tuple, model_cfg: type, l1_small_size: int = 0, trace_region_size: int = 0
) -> ttnn.MeshDevice:
    """Configure fabric and open the mesh device.

    Default fabric is 1D for sp<=8, else 2D. PREFILL_FABRIC_MODE (1d|2d) overrides
    this: the D2D-socket pipeline needs 2D even at sp=8 because a MeshSocket routes
    over 2D fabric, and set_fabric_config is one global config for the whole run.

    `l1_small_size` > 0 carves an L1_SMALL region (needed when an op routes its
    semaphores there, e.g. the Kimi MoE routing all-gather with use_l1_small_for_semaphores).

    `trace_region_size` > 0 reserves device DRAM for ttnn trace capture — needed when the runtime
    replays a captured forward (TtPrefillRuntime use_trace). 0 = no trace region (default)."""
    sp = mesh_shape[0]
    fabric_mode = os.environ.get("PREFILL_FABRIC_MODE", "").strip().lower()
    if fabric_mode == "2d":
        fabric_config = ttnn.FabricConfig.FABRIC_2D
    elif fabric_mode == "1d":
        fabric_config = ttnn.FabricConfig.FABRIC_1D
    elif fabric_mode:
        raise ValueError(f"PREFILL_FABRIC_MODE must be '1d' or '2d', got {fabric_mode!r}")
    else:
        fabric_config = ttnn.FabricConfig.FABRIC_1D if sp <= 8 else ttnn.FabricConfig.FABRIC_2D
    logger.info(f"Fabric config: {fabric_config} (sp={sp}, PREFILL_FABRIC_MODE={fabric_mode or 'unset'})")

    fabric_router_config = _create_fabric_router_config(
        max_payload_size=model_cfg.FABRIC_PAYLOAD_SIZE,
    )

    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.RELAXED_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        fabric_router_config,
    )
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape), l1_small_size=l1_small_size, trace_region_size=trace_region_size
    )


def make_global_spec(mesh_shape: tuple, chunk_size: int) -> ttnn.TensorSpec:
    """Per-push input spec used by `build_h2d_service` to set the service's
    global tensor shape (the producer matches it on the host side). One push carries one
    chunk_size-token chunk. Shape `(sp_factor, 1, chunk_size // sp_factor)` uint32 ROW_MAJOR DRAM."""
    sp_factor = mesh_shape[0]
    isl_per_chip = chunk_size // sp_factor
    return ttnn.TensorSpec(
        shape=ttnn.Shape([sp_factor, 1, isl_per_chip]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def build_h2d_service(
    mesh_device: ttnn.MeshDevice,
    *,
    mesh_shape: tuple,
    chunk_size: int,
    mapper_config: ttnn.MeshMapperConfig,
    worker_cores: ttnn.CoreRange,
    metadata_size_bytes: int,
) -> ttnn.H2DStreamService:
    """Construct an H2DStreamService whose per-shard backing tensor matches
    what `prepare_prefill_input_tensor` would have produced. Each push carries one chunk_size-token
    chunk (chunked prefill streams one chunk per push), not the full sequence.

    Per-shard target: `(1, 1, chunk_size // sp_factor)` uint32 ROW_MAJOR DRAM.
    Achieved by setting global_spec.shape = `(sp_factor, 1, chunk_size // sp_factor)` and
    mapping `[Shard(0), Replicate]` on a `(sp, tp)` mesh — first axis of the
    tensor is sharded across mesh rows (sp), nothing else is split.
    """
    sp_factor, tp_factor = mesh_shape
    assert chunk_size % sp_factor == 0, f"chunk_size={chunk_size} must be divisible by sp_factor={sp_factor}"
    isl_per_chip = chunk_size // sp_factor
    per_chip_bytes = isl_per_chip * 4  # uint32

    global_spec = make_global_spec(mesh_shape, chunk_size)
    mapper = ttnn.create_mesh_mapper(mesh_device, mapper_config)
    # worker_cores set so the service-core kernel multicasts a data-ready inc
    # after each transfer; inbound_socket_service_sync() waits on that on-device, which
    # avoids the host-side barrier() round-trip per iteration.
    # metadata_size_bytes set so the producer can ship per-iter control bytes
    # (slot_id, actual_start, actual_end) inline with the token push.
    service = ttnn.H2DStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        fifo_size_bytes=8 * per_chip_bytes,  # 8 in-flight pages of headroom (0 would auto-size)
        max_socket_page_size_bytes=per_chip_bytes,  # cap socket page at one tensor page (0 = auto/coalesced)
        mapper=mapper,
        worker_cores=worker_cores,
        metadata_size_bytes=metadata_size_bytes,
    )
    logger.info(
        f"[h2d] H2DStreamService built: global_shape=({sp_factor},1,{isl_per_chip}) "
        f"uint32 ROW_MAJOR DRAM, per_chip_bytes={per_chip_bytes}, worker_cores={worker_cores}"
    )
    return service


def activation_global_spec(chunk_size: int, hidden_size: int) -> ttnn.TensorSpec:
    """Global spec of the inter-rank hidden state carried over the D2D pipeline socket:
    [1, 1, chunk_size, hidden_size] bf16 TILE DRAM. The caller's mesh mapper shards it (seq across SP
    rows, emb across TP cols) to match the embedding output layout the downstream model consumes."""
    return ttnn.TensorSpec(
        shape=ttnn.Shape([1, 1, chunk_size, hidden_size]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def resolve_trace_dir(path) -> Path:
    """Resolve a trace dir to the one holding metadata.json. vllm traces nest metadata.json + kv_cache
    under a single run-hash subdir, so if `path` itself has no metadata.json, descend into the sole
    subdir that does."""
    path = Path(path)
    if (path / "metadata.json").exists():
        return path
    subs = [d for d in sorted(path.iterdir()) if d.is_dir() and (d / "metadata.json").exists()]
    if len(subs) != 1:
        raise FileNotFoundError(f"no metadata.json in {path} or a unique subdir (found {len(subs)} candidates)")
    return subs[0]


def load_trace_token_ids(trace_dir, total_len=None) -> list:
    """Input token_ids from a resolved trace's metadata.json (optionally truncated to total_len)."""
    import json

    with open(Path(trace_dir) / "metadata.json") as f:
        md = json.load(f)
    tids = list(md["token_ids"])
    return tids[:total_len] if total_len is not None else tids
