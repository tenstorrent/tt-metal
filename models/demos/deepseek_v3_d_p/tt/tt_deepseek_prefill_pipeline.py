# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TtDeepSeekPrefillPipeline — high-level wrapper around TtPrefillTransformer + KV cache
for disaggregated prefill/decode inference.

Designed to be easy to use from the tt-inference-server's prefill runner:

    # Server startup (all MPI ranks):
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(32, 4))
    config = TtPrefillPipelineConfig(num_layers=61, max_seq_len=102400, ...)
    pipeline = TtDeepSeekPrefillPipeline(mesh_device, hf_config, state_dict, config,
                                         migration_layer=migration_layer)
    pipeline.compile()  # warmup once

    # Per request (rank 0 reads SHM and drives the call; other ranks participate
    # in the collective via their own pipeline.prefill() call):
    first_token = pipeline.prefill(token_ids=..., slot_id=...)

Follows the pattern established by TtSDXLPipeline:
  - Caller opens the mesh and loads the torch model; pipeline just uses them
  - Static params live in a dataclass config
  - State flags track initialization progress; asserts enforce ordering
  - Explicit method chain (compile → prefill) rather than an opaque __call__
  - __del__ releases device resources
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class BoundMigrationEndpoint:
    """Wraps a MigrationLayerEndpoint with a fixed remote_endpoint_id.

    Passed as migration_layer into TtDeepSeekPrefillPipeline so that the
    per-layer callback in MLA never needs to know about endpoint topology.
    """

    def __init__(self, endpoint, remote_endpoint_id: int):
        self._endpoint = endpoint
        self._remote_id = remote_endpoint_id

    def migrate_layer(self, layer: int, pos_start: int, pos_end: int, src_slot: int, dst_slot: int):
        """Trigger a per-layer migration. Returns a uuid (int) for tracking.

        Non-blocking: the underlying endpoint posts the migration to the
        sender pipeline and returns immediately. Use wait() with the
        returned uuid to block on completion.
        """
        return self._endpoint.migrate_layer(self._remote_id, layer, pos_start, pos_end, src_slot, dst_slot)

    def wait(self, uuid) -> None:
        """Block until the migration with the given uuid is fully sent + acked."""
        self._endpoint.wait_migration_send_completion(uuid)


import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import create_balanced_chunk_order, reorder_tensor_chunks
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache


@dataclass
class TtPrefillPipelineConfig:
    """Static configuration for the prefill pipeline.

    Set once at startup — values that change per request (token_ids, slot_id,
    actual_isl) are passed to prefill() instead.
    """

    num_layers: int
    max_seq_len: int  # maximum sequence length the KV cache is sized for
    mesh_shape: tuple = (32, 4)  # global (SP, TP) mesh
    is_balanced: bool = True  # use zigzag / balanced ring attention
    sp_axis: int = 0
    tp_axis: int = 1
    num_links: int = 1
    topology: ttnn.Topology = ttnn.Topology.Linear
    capacity_factor: int = 2
    gate_fallback_mode: GateComputeMode = GateComputeMode.HOST_ALL
    routed_expert_activations_dtype: ttnn.DataType = ttnn.bfloat8_b
    routed_expert_weights_dtype: ttnn.DataType = ttnn.bfloat4_b
    shared_expert_activations_dtype: ttnn.DataType = ttnn.bfloat16
    shared_expert_weights_dtype: ttnn.DataType = ttnn.bfloat8_b
    weight_cache_path: Optional[Path] = None

    @property
    def sp_factor(self) -> int:
        return self.mesh_shape[self.sp_axis]

    @property
    def tp_factor(self) -> int:
        return self.mesh_shape[self.tp_axis]


class TtDeepSeekPrefillPipeline:
    """Owns the prefill model, KV cache, and optional migration layer.

    One instance per prefill process. Call compile() once before the first
    prefill() call. The class is stateful but per-request state (tokens,
    slot_id) is ephemeral — only the model, cache, and migration handles
    persist across requests.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        hf_config: PretrainedConfig,
        state_dict: dict,
        config: TtPrefillPipelineConfig,
        migration_layer=None,
    ):
        """
        Args:
            mesh_device: TTNN mesh device, pre-opened by the caller
                (e.g. ttnn.open_mesh_device(MeshShape(32, 4))).
            hf_config: HuggingFace PretrainedConfig for the model.
            state_dict: DeepSeek weights in TtPrefillTransformer format.
                May be empty {} if loading from weight_cache_path.
            config: Static pipeline parameters (see TtPrefillPipelineConfig).
            migration_layer: Optional MigrationLayer for KV cache transfer to decode.
                When None, prefill runs without migration (useful for testing).
        """
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.config = config
        self.migration_layer = migration_layer

        # State flags — enforced by asserts in prefill()
        self.model_built = False
        self.kv_cache_allocated = False
        self.compiled = False

        self._build_model(state_dict)
        self._allocate_kv_cache()

    # ----------------------------------------------------------------
    # Setup (called in __init__)
    # ----------------------------------------------------------------

    def _build_model(self, state_dict: dict) -> None:
        logger.info(
            f"Building TtDeepSeekPrefillPipeline model: "
            f"num_layers={self.config.num_layers}, max_seq_len={self.config.max_seq_len}, "
            f"mesh_shape={self.config.mesh_shape}, is_balanced={self.config.is_balanced}"
        )
        self.model = TtPrefillTransformer(
            mesh_device=self.mesh_device,
            config=self.hf_config,
            state_dict=state_dict,
            num_layers=self.config.num_layers,
            seq_len=self.config.max_seq_len,
            num_links=self.config.num_links,
            topology=self.config.topology,
            sp_axis=self.config.sp_axis,
            tp_axis=self.config.tp_axis,
            is_balanced=self.config.is_balanced,
            gate_fallback_mode=self.config.gate_fallback_mode,
            routed_expert_activations_dtype=self.config.routed_expert_activations_dtype,
            routed_expert_weights_dtype=self.config.routed_expert_weights_dtype,
            shared_expert_activations_dtype=self.config.shared_expert_activations_dtype,
            shared_expert_weights_dtype=self.config.shared_expert_weights_dtype,
            weight_cache_path=self.config.weight_cache_path,
        )
        self.model_built = True

    def _allocate_kv_cache(self) -> None:
        kvpe_head_dim = self.hf_config.qk_rope_head_dim + self.hf_config.kv_lora_rank
        self.kvpe_cache = init_kvpe_cache(
            kvpe_cache_head_dim=kvpe_head_dim,
            mesh_device=self.mesh_device,
            seq_len=self.config.max_seq_len,
            mesh_shape=list(self.config.mesh_shape),
            sp_axis=self.config.sp_axis,
            num_kvpe_cache_layers=self.config.num_layers,
            zero_init=True,
        )
        self.kv_cache_allocated = True

        # Diagnostic: dump the cache's actual shard layout — both the high-level
        # memory_config AND the per-page physical (bank_id, core_x, core_y,
        # page_address) reported by ttnn._ttnn.reports.get_buffer_pages. This
        # IS the tensor's own source of truth for where each shard physically
        # lives. Compare these to the migration table entries logged at
        # [migration][prefill][table][init]:
        #   table says   (layer=L, pos=P) → (bank_id=B,  noc_addr=base+off)
        #   tensor says  page_index=K     → (bank_id=B', page_address=A')
        # If B != B' or noc_addr's local part != A' for the same logical page,
        # migration is reading from the wrong physical location.
        try:
            cache_addr = self.kvpe_cache.buffer_address()
            print(
                f"[verify-layout] kvpe_cache shape={self.kvpe_cache.shape} "
                f"buffer_address={cache_addr} "
                f"dtype={self.kvpe_cache.dtype} layout={self.kvpe_cache.layout}",
                flush=True,
            )
            print(
                f"[verify-layout] kvpe_cache memory_config={self.kvpe_cache.memory_config()}",
                flush=True,
            )

            # Per-device buffer addresses — KEY DIAGNOSTIC for the migration
            # poison-bug. The migration table is built from ONE address
            # (kvpe_cache.buffer_address()), but for a MeshDevice-backed
            # sharded tensor, each per-device tensor inside the mesh has its
            # OWN buffer_address that the per-device allocator assigned.
            # If those addresses are not all equal to cache_addr, the table
            # is reading from the wrong physical address on devices whose
            # local buffer_address differs from the mesh-level one.
            try:
                device_tensors = ttnn.get_device_tensors(self.kvpe_cache)
                addr_set = set()
                for i, dt in enumerate(device_tensors):
                    try:
                        dt_addr = dt.buffer_address()
                        try:
                            dev_id = dt.device().id() if dt.device() is not None else None
                        except Exception:
                            dev_id = None
                        addr_set.add(dt_addr)
                        # Print first and last 4, plus any that diverge from cache_addr.
                        if i < 4 or i >= len(device_tensors) - 4 or dt_addr != cache_addr:
                            print(
                                f"[verify-layout] device_tensors[{i}] device_id={dev_id} "
                                f"buffer_address={dt_addr} (delta vs mesh={dt_addr - cache_addr})",
                                flush=True,
                            )
                    except Exception as e:
                        print(
                            f"[verify-layout] device_tensors[{i}] buffer_address FAILED: " f"{type(e).__name__}: {e}",
                            flush=True,
                        )
                print(
                    f"[verify-layout] PER-DEVICE buffer_address: "
                    f"{len(addr_set)} unique value(s) across {len(device_tensors)} device tensors. "
                    f"Mesh-level cache.buffer_address()={cache_addr}. "
                    f"{'CONSISTENT' if addr_set == {cache_addr} else 'MISMATCH — migration table built from mesh address but per-device addresses differ!'}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[verify-layout] per-device address dump FAILED: " f"{type(e).__name__}: {e}",
                    flush=True,
                )
            # Per-page physical addresses — the source of truth.
            try:
                import ttnn as _ttnn_mod

                all_pages = _ttnn_mod._ttnn.reports.get_buffer_pages(self.mesh_device)
                # Filter to pages belonging to kvpe_cache (match by buffer base address).
                cache_pages = [p for p in all_pages if p.address == cache_addr]
                print(
                    f"[verify-layout] kvpe_cache has {len(cache_pages)} pages across mesh; "
                    f"sampling first 16 (and pages whose page_index hits global_pos 0/32/64/96):",
                    flush=True,
                )
                # Sort for determinism: by (device_id, bank_id, page_index)
                cache_pages.sort(key=lambda p: (p.device_id, p.bank_id, p.page_index))
                for i, p in enumerate(cache_pages[:16]):
                    print(
                        f"[verify-layout] page[{i}]: device_id={p.device_id} "
                        f"bank_id={p.bank_id} core=({p.core_x},{p.core_y}) "
                        f"page_index={p.page_index} page_address={p.page_address} "
                        f"page_size={p.page_size}",
                        flush=True,
                    )
            except Exception as e:
                print(f"[verify-layout] page dump FAILED: {type(e).__name__}: {e}", flush=True)
        except Exception as e:
            print(f"[verify-layout] dump FAILED: {type(e).__name__}: {e}", flush=True)

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    def compile(self) -> None:
        """Warmup pass to JIT-compile kernels. Call once before the first prefill().

        TODO: currently a no-op placeholder. Could run a dummy forward with a
        small sequence to trigger JIT compilation.
        """
        assert self.model_built and self.kv_cache_allocated
        logger.info("TtDeepSeekPrefillPipeline.compile() — no-op (TODO: dummy forward)")
        self.compiled = True

    # ----------------------------------------------------------------
    # Per-request entry point
    # ----------------------------------------------------------------

    def prefill(self, token_ids: list[int], slot_id: int, actual_isl: Optional[int] = None) -> int:
        """Run prefill for one request.

        All MPI ranks must call prefill() collectively with the same arguments;
        TTNN's distributed runtime handles cross-host CCL internally.

        Args:
            token_ids: Full input sequence in the user's original token order.
                This function does the zigzag reorder internally.
            slot_id: KV cache slot assigned by the inference server.
            actual_isl: Number of real (non-padded) tokens. Defaults to len(token_ids).

        Returns:
            First generated token ID.
        """
        assert self.compiled, "Call compile() before prefill()"
        if actual_isl is None:
            actual_isl = len(token_ids)

        tt_token_ids = self._prepare_input_tensor(token_ids)
        on_layer_complete = self._build_migration_callback(slot_id, actual_isl)

        first_token_id, _first_token_prob, _ = self.model.forward(
            tt_token_ids,
            self.kvpe_cache,
            number_of_non_padded_tokens=actual_isl,
            on_layer_complete=on_layer_complete,
            temperature=0.0,  # greedy argmax; expose via prefill kwarg if/when sampling is needed
        )
        return int(first_token_id)

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _prepare_input_tensor(self, token_ids: list[int]) -> ttnn.Tensor:
        """Zigzag-reorder tokens (if is_balanced) and shard to the global mesh."""
        sp_factor = self.config.sp_factor
        isl_per_chip = len(token_ids) // sp_factor

        if self.config.is_balanced:
            # Reorder into zigzag chunk order so each SP device gets one chunk
            # from the front and one from the back of the sequence.
            # reorder_tensor_chunks requires a 4D tensor with seq_dim=2.
            chunk_order = create_balanced_chunk_order(sp_factor)
            t = torch.tensor(token_ids, dtype=torch.int64).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            # [1, 1, seq_len, 1]
            t = reorder_tensor_chunks(t, chunk_order, seq_dim=2)
            token_ids_sharded = t.squeeze(0).squeeze(-1).reshape(sp_factor, 1, isl_per_chip)
        else:
            token_ids_sharded = torch.tensor(token_ids, dtype=torch.int64).reshape(sp_factor, 1, isl_per_chip)

        return ttnn.from_torch(
            token_ids_sharded,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                mesh_shape=self.config.mesh_shape,
                dims=(self.config.sp_axis, None),
            ),
        )

    def setup_migration(self, endpoint, remote_endpoint_id: int) -> None:
        """Wire up the migration endpoint after compile().

        Wraps endpoint in BoundMigrationEndpoint so that the per-layer
        callback never needs to know the remote endpoint ID directly.

        Call this on rank 0 only. Worker ranks leave migration_layer=None
        so their _build_migration_callback returns None and MLA skips migration.

        Args:
            endpoint: MigrationLayerEndpoint from make_mpi_endpoint_device().
            remote_endpoint_id: The decode side's endpoint ID (pre-agreed at startup).
        """
        assert self.compiled, "Call compile() before setup_migration()"
        self.migration_layer = BoundMigrationEndpoint(endpoint, remote_endpoint_id)

    def _build_migration_callback(self, slot_id: int, actual_isl: int):
        """Build the per-layer migration callback passed to MLA via forward().

        MLA invokes this after fill_cache_for_user_(). MLA also handles zeroing
        padding pages before fill (gated on this callback being set).

        Strategy: every layer fires migrate_layer (non-blocking). The LAST layer
        also waits on its uuid before returning. The transport delivers in order,
        so waiting on the last layer's uuid guarantees all earlier migrations
        have also been sent + acked by the decode side. By the time forward()
        returns and the runner emits the first token over SHM, the entire KV
        cache has been migrated.

        Returns None when no migration_layer is configured — MLA then skips both
        the zero-out and the post-fill callback.
        """
        if self.migration_layer is None:
            return None

        mesh_device = self.mesh_device
        migration_layer = self.migration_layer
        kvpe_cache = self.kvpe_cache
        last_layer_idx = self.config.num_layers - 1

        def _dump_cache_readback(layer_idx: int) -> None:
            """Read kvpe_cache via the shard-spec path (ttnn.to_torch on per-device
            tensors) and dump the first 16 bytes at sample positions. Use this to
            compare against what migration's raw NOC read returns. If they disagree,
            the table's (bank_id, offset) encoding doesn't match the cache's actual
            shard layout."""
            try:
                import torch

                device_tensors = ttnn.get_device_tensors(kvpe_cache)
                # Print device 0's identity + buffer_address so we can correlate
                # with the migration sender's "[scan-banks] device_id=N
                # local_addr=M" log. If buffer_address(dev_tensors[0]) differs
                # from kvpe_cache.buffer_address(), the migration table is reading
                # from the wrong physical address.
                try:
                    dev0_phys_id = device_tensors[0].device().id() if device_tensors[0].device() is not None else None
                except Exception:
                    dev0_phys_id = None
                try:
                    dev0_buf_addr = device_tensors[0].buffer_address()
                except Exception:
                    dev0_buf_addr = None
                try:
                    mesh_buf_addr = kvpe_cache.buffer_address()
                except Exception:
                    mesh_buf_addr = None
                print(
                    f"[verify-readback] device_tensors[0]: device_id={dev0_phys_id} "
                    f"buffer_address={dev0_buf_addr} "
                    f"(mesh.buffer_address={mesh_buf_addr}, "
                    f"delta={dev0_buf_addr - mesh_buf_addr if (dev0_buf_addr is not None and mesh_buf_addr is not None) else 'n/a'})",
                    flush=True,
                )
                # Cache is sharded across SP devices along seq_len; each device holds
                # seq_len_local = seq_len_total / sp_factor tokens. Print bytes from
                # device 0 (which holds global positions 0..seq_len_local-1).
                dev0 = ttnn.to_torch(device_tensors[0])  # [num_layers, 1, seq_len_local, head_dim]
                seq_len_local = dev0.shape[2]
                # Sample positions matching the migration's read_issue_pos values
                # we suspected of poison (0, 32, 64, 96 in early; 1024, 1056 in padding).
                sample_positions = [0, 32, 64, 96, 128, 1024, 1056, 1088, 1120]
                for global_pos in sample_positions:
                    if global_pos >= seq_len_local:
                        continue  # would need a different device tensor
                    row = dev0[layer_idx, 0, global_pos, :]
                    head_bytes = row.contiguous().view(torch.uint8)[:16].tolist()
                    head_uint32 = row.contiguous().view(torch.uint32)[:4].tolist()
                    print(
                        f"[verify-readback] layer={layer_idx} dev=0 global_pos={global_pos} "
                        f"bytes[0..16]={head_bytes} uint32[0..4]={head_uint32}",
                        flush=True,
                    )
            except Exception as e:
                print(f"[verify-readback] FAILED layer={layer_idx}: {type(e).__name__}: {e}", flush=True)

        def on_layer_complete(layer_idx: int) -> None:
            ttnn.synchronize_device(mesh_device)
            end_pos = math.ceil(actual_isl / 128) * 128
            print(
                f"[migration][prefill] on_layer_complete, migrating layer {layer_idx} from slot {slot_id} to slot {slot_id}. Start_pos={0}, End_pos={end_pos}"
            )
            # Diagnostic: only on layer 0 (one-shot per prefill) to compare cache contents
            # via shard-spec readback against migration's raw-NOC reads in the same window.
            if layer_idx == 0:
                _dump_cache_readback(layer_idx)
            uuid = migration_layer.migrate_layer(layer_idx, 0, end_pos, slot_id, slot_id)
            ## Wait for each one for initial bringup
            # if layer_idx == last_layer_idx:
            print(f"[migration][prefill] wait for migrate layer completion")
            migration_layer.wait(uuid)
            print(f"[migration][prefill] done migrate layer")

        return on_layer_complete

    # ----------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------

    def __del__(self):
        # Device tensors are freed when their refs drop. ttnn handles the
        # underlying buffer lifecycle; we just clear our references here.
        try:
            self.kvpe_cache = None
            self.model = None
        except Exception:
            pass
