# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Capacity-probe launcher: the stock GLM-5.3 prefill runner, patched before it starts (tt_metal_tracker-q171.3.5).

Always: logs DRAM per chip (``STAGE_DRAM``: allocated and largest free block per bank) after the weights are built
and after the KV caches are allocated, so probes can be compared footprint for footprint.

``CAP_STUB_WEIGHTS=1``: capacity-sweep weights (user ruling 2026-11-07, knowledge note "Capacity sweeps allocate
weight-shaped tensors instead of loading real weights"; K3.1's eac952cbf43). While the adapter builds the runtime,
every cached weight is allocated on device with the spec and mesh topology of its cache file (the file is mapped
and only its header read) and left uninitialized: a cache load straight to device is intercepted at
``load_tensor_flatbuffer``, and the routed experts, which load to host and place with ``ttnn.to_device``, are
stubbed only inside their loader. Weight bytes are unchanged; values are garbage, so use it for memory only.

``CAP_G2_STANDIN=1``: GLM-5.4 G2 stand-in on GLM-5.3 layers 26-53, reshaped to G2's memory. GLM-5.4's loop pass 2
does not exist yet (q171.2.5 / q171.2.6 / q171.2.7):

* every layer of the probe's range is ``shared`` (pass 2 runs no indexer, A5): no indexer weights load, no index-K
  cache and no indexer score buffer are allocated, and the offset first rank feeds the layers placeholder top-k
  (``TtPrefillRuntime.make_placeholder_indices``, passed to ``prefill_chunk``), the way G2 would consume the sets
  G1 forwards;
* next to the KVPE cache (G2's bank A: the pass-1 copy, same shape as G1's), it allocates what G2 adds that the
  runner does not: the pass-2 window, 128 rows x 704 B per loop layer per user, replicated on every chip (an upper
  bound), and one chunk's receive buffers for the 7 top-k sets and the 28 layers' pass-1 KVPE rows.
  Not represented: the two-bank attention's own working buffers and the damp gate (scalar, negligible). Needs
  ``PREFILL_PP_LAYER_OFFSET=26 PREFILL_NUM_LAYERS=28``.

GLM-5.4 (``STAGE_PROBE_MODEL=glm54``, manifest glm54.json; tt_metal_tracker-q171.3.18): the launcher applies the
patches above that fit (stub weights, ``STAGE_DRAM``), also logs ``STAGE_DRAM`` after the trace is captured and when it
is released (after the request), and then runs trunk_probe_runner, so the run keeps its push-reuse check.

Use with ``STAGE_PROBE_RUNNER_MODULE=models.demos.deepseek_v3_d_p.scripts.stage_probe.capacity_probe_runner``.
"""

from __future__ import annotations

import contextlib
import os
import runpy

from loguru import logger

from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.reference.glm_5_4_config import GLM54Config

LOOP = GLM54Config.loop_layers()
WINDOW = GLM54Config.LOOP_WINDOW_SIZE
LOOP_FULL_LAYERS = sum(1 for i in LOOP if GLM53Config.indexer_types()[i] == "full")

# Allocations kept alive for the runner's lifetime.
_EXTRA = []


def _all_shared_loop() -> None:
    types = GLM53Config.indexer_types()
    GLM53Config.INDEX_TOPK_PATTERN = "".join(
        "S" if i in LOOP else ("F" if t == "full" else "S") for i, t in enumerate(types)
    )
    logger.info(f"[g2-standin] indexer map forced shared on layers {LOOP.start}-{LOOP.stop - 1}")


def _patch_allocate_kv_cache() -> None:
    import ttnn
    from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter
    from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import MlaKvCaches
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheGeometry, init_mla_kv_cache

    def allocate_kv_cache(self, *, mesh_device, hf_config, params):
        first, last = params.first_layer_idx, params.first_layer_idx + params.num_layers - 1
        assert (first, last) == (LOOP.start, LOOP.stop - 1) and not params.mtp_levels, "G2 stand-in is layers 26-53"
        slots = self.kv_cache_slots(params)
        kvpe = init_mla_kv_cache(
            cache_format=self.resolve_sparse_kv_cache_format(params.sparse_kv_cache_format),
            hf_config=hf_config,
            mesh_device=mesh_device,
            seq_len=params.max_seq_len,
            mesh_shape=list(params.mesh_shape),
            sp_axis=params.sp_axis,
            num_kvpe_cache_layers=params.num_layers,
            num_users=slots,
            tp_axis=params.tp_axis,
            bank_aligned=True,
        )
        geometry = MlaKvCacheGeometry.from_config(hf_config)
        row = -(-geometry.packed_row_bytes // 64) * 64
        sp, tp = params.mesh_shape[params.sp_axis], params.mesh_shape[params.tp_axis]
        q_rows = params.chunk_size // (sp * tp)

        def alloc(shape, dtype, what):
            t = ttnn.allocate_tensor_on_device(
                ttnn.Shape(shape), dtype, ttnn.ROW_MAJOR_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
            )
            _EXTRA.append(t)
            logger.info(f"[g2-standin] {what}: {list(shape)} {dtype} per chip")

        # Window: 128 rows of one packed row per loop layer per user, bytes as uint8.
        alloc([slots * params.num_layers, 1, WINDOW, row], ttnn.uint8, "pass-2 window")
        alloc([LOOP_FULL_LAYERS, 1, q_rows, GLM53Config.INDEX_TOPK], ttnn.uint32, "top-k receive (7 sets)")
        alloc([params.num_layers, 1, q_rows, row], ttnn.uint8, "pass-1 KVPE receive")
        return MlaKvCaches(kvpe=kvpe, index=None)

    GLM53Adapter.allocate_kv_cache = allocate_kv_cache


def _patch_placeholder_indices() -> None:
    """prefill_runner passes top-k only from an upstream rank's D2D segment, and an offset first rank has none,
    yet ``prefill_chunk`` requires them on a rank that starts on a shared layer. Hand it a fresh placeholder set
    (zeros, ``make_placeholder_indices``) per chunk: ``prefill_chunk`` consumes (copies, then deallocates) the
    set it is given, as it does a received D2D segment."""
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    prefill_chunk = TtPrefillRuntime.prefill_chunk

    def prefill_chunk_with_indices(self, *args, **kwargs):
        if self.receives_indexer_indices and kwargs.get("indexer_indices") is None:
            kwargs["indexer_indices"] = self.make_placeholder_indices()
        return prefill_chunk(self, *args, **kwargs)

    TtPrefillRuntime.prefill_chunk = prefill_chunk_with_indices


def _allocate_like(host, device, memory_config=None):
    """An uninitialized device tensor with the host tensor's per-chip spec and mesh topology."""
    import ttnn

    spec = host.spec
    if memory_config is None:
        tensor = ttnn.allocate_tensor_on_device(spec, device)
    else:  # placed with another memory config, as the routed experts are
        tensor = ttnn.allocate_tensor_on_device(spec.shape, spec.dtype, spec.layout, device, memory_config)
    tensor.update_tensor_topology(host.tensor_topology())
    return tensor


@contextlib.contextmanager
def _header_only():
    import ttnn
    from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert

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


def _log_dram(mesh_device, phase: str) -> None:
    import ttnn

    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    allocated = view.total_bytes_allocated_per_bank * view.num_banks
    largest = view.largest_contiguous_bytes_free_per_bank
    logger.info(
        f"STAGE_DRAM {phase}: {allocated} B allocated per chip, largest free block {largest} B per bank "
        f"({largest / 2**20:.1f} MiB), free {view.total_bytes_free_per_bank} B per bank of "
        f"{view.total_bytes_per_bank} B x {view.num_banks} banks"
    )


def _patch_build_and_logging(stub: bool) -> None:
    from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter

    build_runtime = GLM53Adapter.build_runtime
    allocate_kv_cache = GLM53Adapter.allocate_kv_cache

    def build(self, *, mesh_device, **kwargs):
        with _header_only() if stub else contextlib.nullcontext():
            runtime = build_runtime(self, mesh_device=mesh_device, **kwargs)
        _log_dram(mesh_device, f"weights ({'stub' if stub else 'real'})")
        return runtime

    def allocate(self, *, mesh_device, params, **kwargs):
        caches = allocate_kv_cache(self, mesh_device=mesh_device, params=params, **kwargs)
        _log_dram(mesh_device, f"kv_caches ({params.num_users} users)")
        return caches

    GLM53Adapter.build_runtime = build
    GLM53Adapter.allocate_kv_cache = allocate


def _patch_runtime_logging() -> None:
    """``STAGE_DRAM`` once the trace is captured (steady state for the request) and when it is released."""
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    capture = TtPrefillRuntime.capture_trace
    release = TtPrefillRuntime.release_trace

    def capture_trace(self, kv_caches):
        captured = getattr(self, "_trace_captured", False)
        capture(self, kv_caches)
        if not captured and getattr(self, "_trace_captured", False):
            _log_dram(self.mesh_device, "trace captured")

    def release_trace(self):
        if getattr(self, "_trace_captured", False):
            _log_dram(self.mesh_device, "before trace release (request done)")
        return release(self)

    TtPrefillRuntime.capture_trace = capture_trace
    TtPrefillRuntime.release_trace = release_trace


def main() -> None:
    glm54 = os.environ.get("PREFILL_MANIFEST", "").endswith("glm54.json")
    assert glm54 or os.environ.get("PREFILL_MODEL", "glm_5_3") == "glm_5_3"
    if os.environ.get("CAP_G2_STANDIN") == "1":
        assert not glm54, "CAP_G2_STANDIN stands GLM-5.3 layers in for G2; GLM-5.4 runs its own G2"
        _all_shared_loop()
        _patch_allocate_kv_cache()
        _patch_placeholder_indices()
    _patch_build_and_logging(stub=os.environ.get("CAP_STUB_WEIGHTS") == "1")
    if not glm54:
        runpy.run_module("models.demos.common.prefill.runners.prefill_runner", run_name="__main__")
        return
    _patch_runtime_logging()
    from models.demos.deepseek_v3_d_p.scripts.stage_probe import trunk_probe_runner

    trunk_probe_runner.main()


if __name__ == "__main__":
    main()
