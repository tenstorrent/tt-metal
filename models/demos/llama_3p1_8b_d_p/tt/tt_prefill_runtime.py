# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-rank eager runtime with whole-chunk device completion before LayerAck.

The engine owns tokens, metadata, and KV. A model callable and completion function
can be supplied for host contract tests without importing TTNN or loading weights.
"""

from dataclasses import dataclass

from .prefill_geometry import PREFILL_LAYOUT as layout
from .prefill_geometry import PrefillGeometry
from .runners.kv_layout import integer


@dataclass(frozen=True)
class TtPrefillRuntimeConfig:
    max_seq_len: int
    chunk_size: int
    num_users: int
    num_layers: int = 32
    first_layer_idx: int = 0
    is_first_rank: bool = True
    is_last_rank: bool = True
    mesh_shape: tuple = layout.mesh_shape
    use_trace: bool = False

    def __post_init__(self):
        for name in ("max_seq_len", "chunk_size", "num_users", "num_layers"):
            integer(name, getattr(self, name), 1)
        if self.use_trace:
            raise NotImplementedError("Llama runtime uses eager execution")
        PrefillGeometry(self.max_seq_len)
        if self.chunk_size != layout.chunk_size or self.num_users != layout.num_users:
            raise ValueError(f"Llama runtime requires chunk_size={layout.chunk_size} and num_users={layout.num_users}")
        if (self.num_layers, self.first_layer_idx, self.is_first_rank, self.is_last_rank, self.mesh_shape) != (
            32,
            0,
            True,
            True,
            layout.mesh_shape,
        ):
            raise ValueError("Llama runtime supports one SP4/TP8 rank containing all 32 layers")


class TtPrefillRuntime:
    def __init__(self, mesh_device, *, config, model, synchronize, upload):
        if model.num_layers != config.num_layers:
            raise ValueError("model layer count differs from runtime configuration")
        if model.max_seq_len != config.max_seq_len:
            raise ValueError("model capacity differs from runtime configuration")
        self.geometry = PrefillGeometry(config.max_seq_len)
        self.mesh_device = mesh_device
        self.config = config
        self.model = model
        self._synchronize = synchronize
        self._upload = upload
        self._layer_completion_sink = None
        self._active = False
        self._failed = False
        self._last_request_id = -1
        self.compiled = False

    def _check_cache(self, cache):
        self.geometry.validate_cache_metadata(cache)

    def _check_ready(self):
        if self._failed:
            raise RuntimeError("runtime failed; restart the worker before further requests")
        if self._active:
            raise RuntimeError("a prefill call is active")

    def make_chunk_input(self, token_ids, *, actual_start=0):
        actual_end = actual_start + len(token_ids)
        self.geometry.validate_chunk_range(actual_start, actual_end)
        return self._upload(token_ids, actual_start=actual_start, actual_end=actual_end)

    def compile(self, kv_cache):
        """Warm both first-chunk and continuation programs before installing the sink."""
        self._check_ready()
        self._check_cache(kv_cache)
        if self.compiled:
            return
        if self._layer_completion_sink is not None:
            raise RuntimeError("compile must precede completion sink installation")
        self._active = True
        try:
            for start in range(0, min(self.config.max_seq_len, 2 * self.config.chunk_size), self.config.chunk_size):
                tokens = self.make_chunk_input([0] * self.config.chunk_size, actual_start=start)
                try:
                    self._run(tokens, kv_cache, 0, start, start + self.config.chunk_size, None, None)
                finally:
                    tokens.deallocate(True)
            # Compilation executes real cache writes. The cache is externally owned, so those
            # zero-token warmups must not become an advertised logical prefix for the first request.
            kv_cache.truncate_prefix(0, 0)
            self.compiled = True
        except BaseException:
            self._failed = True
            raise
        finally:
            self._active = False

    def _run(self, tokens, cache, slot, start, end, request, sink):
        output = self.model.prefill_chunk(
            tokens, cache, slot_idx=slot, actual_start=start, actual_end=end, skip_lm_head=True
        )
        try:
            # Persistent H2D kernels launch directly on claimed service cores and bypass
            # normal fast-dispatch queue accounting (distributed.cpp:EnqueueMeshWorkload).
            # Drain queued model work before certifying any layer to the migration manager.
            self._synchronize(self.mesh_device)
            if output is not None:
                completed, output = output, None
                completed.deallocate(True)
            if sink is not None:
                for layer in range(self.config.num_layers):
                    sink(self.config.first_layer_idx + layer, request)
        finally:
            if output is not None:
                output.deallocate(True)

    def prefill_chunk(
        self,
        input_tensor,
        kv_cache,
        *,
        slot_id,
        actual_start,
        actual_end,
        request_id=0,
        d2h_service=None,
        metadata_msg=None,
    ):
        """Borrow already reshuffled H2D input and metadata until return.

        request_id is the common runner's worker-global chunk counter, not a user ID.
        """
        self._check_ready()
        if not self.compiled:
            raise RuntimeError("compile must complete before serving")
        if d2h_service is not None:
            raise NotImplementedError("Llama eager runtime requires PREFILL_LAYER_ACK_D2H=0")
        self._check_cache(kv_cache)
        for name, value in (
            ("slot_id", slot_id),
            ("actual_start", actual_start),
            ("actual_end", actual_end),
            ("request_id", request_id),
        ):
            integer(name, value)
        if slot_id >= self.config.num_users:
            raise ValueError("slot_id is outside the allocated cache")
        if actual_start == actual_end:
            raise ValueError("chunk bounds must be nonempty")
        self.geometry.validate_chunk_range(actual_start, actual_end)
        if request_id <= self._last_request_id:
            raise ValueError("request_id must increase; replay would duplicate LayerAck")
        sink = self._layer_completion_sink
        self._active = True
        try:
            self._run(input_tensor, kv_cache, slot_id, actual_start, actual_end, request_id, sink)
            self._last_request_id = request_id
        except BaseException:
            # Partial cache writes or partial acknowledgments require a fresh worker generation.
            self._failed = True
            raise
        finally:
            self._active = False
        return None

    def set_layer_completion_sink(self, sink):
        """Replace or remove the sink between calls; an active call keeps one immutable sink."""
        self._check_ready()
        if not self.compiled:
            raise RuntimeError("compile must complete before sink installation")
        if sink is not None and not callable(sink):
            raise TypeError("sink must be callable or None")
        self._layer_completion_sink = sink

    def kv_migration_stages(self, kv_cache, first_layer_idx=None, num_my_layers=None):
        from models.demos.common.prefill.runners.migration import KvCacheStage

        self._check_cache(kv_cache)
        if first_layer_idx not in (None, 0) or num_my_layers not in (None, self.config.num_layers):
            raise ValueError("Llama migration supports one full-model rank")
        return tuple(
            KvCacheStage(int(t.buffer_address()), 0, self.config.num_layers)
            for t in (kv_cache.k, kv_cache.v)
            for _ in range(8)
        )

    def build_kv_chunk_table(self, kv_cache, path, *, first_layer_idx=0, num_my_layers=None, stage_layouts=None):
        from .runners.kv_chunk_table import build_and_serialize_kv_chunk_table

        self._check_cache(kv_cache)
        if first_layer_idx != 0 or num_my_layers not in (None, self.config.num_layers):
            raise ValueError("Llama migration supports one full-model rank")
        if stage_layouts is not None:
            stages = self.kv_migration_stages(kv_cache)
            if len(stage_layouts) != len(stages):
                raise ValueError("expected 16 gathered K/V config layouts")
            for gathered, stage in zip(stage_layouts, stages):
                if len(gathered) != 1 or any(
                    gathered[0][key] != value
                    for key, value in (
                        ("first_layer", stage.first_layer),
                        ("count", stage.count),
                        ("base_addr", stage.base_addr),
                    )
                ):
                    raise ValueError("gathered layout must describe this single-rank cache")
        return build_and_serialize_kv_chunk_table(
            mesh_device=self.mesh_device, kv_cache=kv_cache, chunk_size=self.config.chunk_size, path=path
        )


def config_from_params(params):
    """Validate shared geometry before allocation; capacity support still needs a live gate."""
    config = TtPrefillRuntimeConfig(
        max_seq_len=params.max_seq_len,
        chunk_size=params.chunk_size,
        num_users=params.num_users,
        num_layers=params.num_layers,
        first_layer_idx=params.first_layer_idx,
        is_first_rank=params.is_first_rank,
        is_last_rank=params.is_last_rank,
        mesh_shape=tuple(params.mesh_shape),
    )
    if (params.sp_axis, params.tp_axis) != (0, 1):
        raise ValueError("Llama requires SP on rows and TP on columns")
    if params.use_trace or params.dflash_enabled:
        raise NotImplementedError("Llama eager runtime does not support trace or DFlash")
    return config


def build_runtime(mesh_device, *, params, checkpoint_path):
    """Bind the eager device model; reject unsupported capacities before loading weights."""
    config = config_from_params(params)
    import ttnn

    from .input import upload_token_chunk
    from .model import PrefillModel

    model = PrefillModel(
        mesh_device,
        checkpoint_path,
        num_layers=config.num_layers,
        enable_lm_head=False,
        cache_dtype=ttnn.bfloat8_b,
        max_seq_len=config.max_seq_len,
    )
    return TtPrefillRuntime(
        mesh_device,
        config=config,
        model=model,
        synchronize=ttnn.synchronize_device,
        upload=lambda ids, **kwargs: upload_token_chunk(mesh_device, ids, max_seq_len=config.max_seq_len, **kwargs),
    )
