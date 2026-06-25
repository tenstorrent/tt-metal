# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The runner ↔ model seam.

`PrefillModelAdapter` is everything the generic runner needs from a model; a model package provides
one implementation per model and registers it (see `registry.py`). The runner core depends only on
these Protocols + `ttnn` — never on a model package.

Design notes:
  * The KV cache is NOT on this seam. It is created and owned entirely inside the concrete runtime
    (whatever its layout — merged kv+pe in one tensor, or separate regular K/V; replicated across TP,
    or TP-head-sharded). The runner core never touches it; every cache-touching operation is an
    adapter method that reaches into its OWN runtime.
  * `gate_fallback_mode` crosses the seam as a plain string — the adapter converts it to whatever
    enum its runtime wants, so the core never imports a model's MoE gate enum.
  * Token input layout is model-owned (`prepare_prefill_input_tensor` + `h2d_mapper_config`): how
    token IDs shard across the mesh can differ per model.
"""

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import ttnn


@runtime_checkable
class PrefillRuntime(Protocol):
    """What `PrefillModelAdapter.build_runtime` returns. `TtDeepSeekPrefillRuntime` already
    satisfies this. The KV cache is intentionally absent — it stays an internal detail of the
    concrete runtime, reached only by the adapter's own methods."""

    mesh_device: ttnn.MeshDevice
    # `config` exposes: sp_factor, sp_axis, tp_factor, chunk_size, num_users,
    #                   max_seq_len, num_layers, mesh_shape
    config: object

    def compile(self) -> None:
        ...

    def prefill(self, input_tensor: ttnn.Tensor, slot_id: int, actual_start: int, actual_end: int) -> None:
        ...

    def set_layer_ack_channel(self, layer_ack_channel) -> None:
        ...


@runtime_checkable
class PrefillModelAdapter(Protocol):
    """Per-model plumbing the runner needs to build and drive a prefill model. One instance per
    model; the registry maps a model name to a factory that returns one of these."""

    # --- static knobs (no device needed) ---
    name: str  # model name; matches the weight-cache dir prefix {name}_{arch}_{N}dev
    default_gate_mode: str  # GateComputeMode name (string); PREFILL_GATE_FALLBACK_MODE overrides
    uses_l1_small_semaphores: bool  # carve an L1_SMALL region when opening the mesh (e.g. Kimi routing)
    fabric_payload_size: int  # max fabric packet payload (model_cfg.FABRIC_PAYLOAD_SIZE)
    h2d_mapper_config: ttnn.MeshMapperConfig  # how a token push shards across the mesh (SP/TP)
    supports_migration: bool  # whether build_and_serialize_kv_chunk_table is implemented

    # --- resource resolution ---
    def load_hf_config(self, max_seq_len: int):
        """Load (and unwrap) the HF config and set max_seq_len. Returned object is opaque to the core
        — it is only handed back to build_runtime."""
        ...

    def resolve_weight_cache_path(self, mesh_shape: tuple) -> Optional[Path]:
        ...

    def resolve_trace_dir(self) -> Path:
        """Golden trace dir holding metadata.json (PREFILL_TRACE_DIR overrides the model default)."""
        ...

    def load_trace_token_ids(self, trace_dir, total_len=None) -> list:
        ...

    # --- runtime build (owns KV cache creation) ---
    def build_runtime(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        hf_config,
        mesh_shape: tuple,
        num_layers: int,
        max_seq_len: int,
        chunk_size: int,
        num_users: int,
        capacity_factor: int,
        gate_fallback_mode: str,
        weight_cache_path: Optional[Path],
        kv_only_last_layer: bool,
    ) -> PrefillRuntime:
        ...

    # --- token input layout (model-owned) ---
    def prepare_prefill_input_tensor(
        self,
        token_ids: list,
        mesh_device: ttnn.MeshDevice,
        sp_factor: int,
        is_balanced: bool,
        mesh_shape: tuple,
        sp_axis: int,
    ) -> ttnn.Tensor:
        ...

    # --- validation (opaque cache: reaches into the runtime) ---
    def kv_cache_pcc_check(self, runtime: PrefillRuntime, slot_id: int, n_chunks: int, trace_dir=None) -> float:
        ...

    # --- disaggregation / migration (optional capability) ---
    def build_and_serialize_kv_chunk_table(self, runtime: PrefillRuntime, path: str) -> str:
        """Build the KV chunk address table from the runtime's own KV layout and serialize it to
        `path`. Reads dims off runtime.config and the cache tensor(s) off the runtime directly.
        Only called when `supports_migration` is True."""
        ...
