# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2-style MLP module for the Wormhole Galaxy (8, 4) mesh.

Single unified MLP2D class with separate forward methods:
  - decode_forward(): For decode mode
  - prefill_forward(): For prefill mode
  - forward(x, mode): Dispatcher that calls the appropriate method

Execution paths:
  - Unified: linear → linear → reduce_scatter(×2) → mul+silu → all_gather → linear → all_reduce

"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight

WH_GALAXY_MESH_SHAPE = (8, 4)
PrefillProgramConfigFactory = Callable[[int], Any]

# =============================================================================
# Top-level config dataclass
# =============================================================================


@dataclass(frozen=True)
class MLP2DConfig:
    """
    Central configuration for MLP2D - the single source of truth for all settings.

    None fields are populated with derived defaults during config resolution
    (inside ``MLP2D.__init__`` or ``MLP2D.from_config``).

    Minimal usage requires resolved Galaxy collective resources:
        config = MLP2DConfig(w1, w2, w3, tt_ccl=galaxy_ccl)

    Override any field:
        config = MLP2DConfig(w1, w2, w3, max_batch_size=64)

    Full customization:
        config = MLP2DConfig(
            w1, w2, w3,
            mesh_device=custom_device,
            decode_w1_w3_prg_config=my_program_config,
            ...
        )
    """

    # Required: weights (LazyWeight)
    w1: LazyWeight
    w2: LazyWeight
    w3: LazyWeight
    prefill_w1: LazyWeight | None = None
    prefill_w2: LazyWeight | None = None
    prefill_w3: LazyWeight | None = None

    # Optional: device and collectives
    mesh_device: ttnn.MeshDevice | None = None
    tt_ccl: Any = None
    decode_ccl_context: Any = None
    prefill_ccl_context: Any = None
    decode_reduce_scatter_resources: Any = None
    decode_all_gather_resources: Any = None
    decode_all_reduce_resources: Any = None
    prefill_reduce_scatter_resources: Any = None
    prefill_all_gather_resources: Any = None
    prefill_all_reduce_resources: Any = None
    collective_resource_selector: Callable[[Any, str, int, Any, Any], Any] | None = None
    topology: ttnn.Topology | None = None  # None = auto-detect
    num_reduce_scatter_links: int = 1
    num_all_gather_links: int = 2
    ccl_chunks_per_sync: int = 10
    ccl_num_workers_per_link: int = 2
    ccl_num_buffers_per_channel: int = 2
    decode_prefetch_context: Any = None
    prefill_prefetch_context: Any = None

    # Optional: derived from weights if None
    dim: int | None = None
    hidden_dim: int | None = None

    # Optional: sensible defaults
    max_batch_size: int = 32
    mlp_activation_type: ttnn.UnaryOpType = ttnn.UnaryOpType.SILU

    # Optional: power-user overrides (None = compute defaults)
    w1_w3_memcfg: ttnn.MemoryConfig | None = None
    w2_memcfg: ttnn.MemoryConfig | None = None

    # Decode settings
    decode_input_memcfg: ttnn.MemoryConfig | None = None
    decode_w2_input_memcfg: ttnn.MemoryConfig | None = None
    decode_w1_w3_prg_config: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig | None = None
    decode_w2_prg_config: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig | None = None
    decode_w1_w3_output_memcfg: ttnn.MemoryConfig | None = None
    decode_w2_output_memcfg: ttnn.MemoryConfig | None = None
    ff1_out_reduce_scatter_memcfg: ttnn.MemoryConfig | None = None
    ff2_out_reduce_scatter_memcfg: ttnn.MemoryConfig | None = None
    sharded_attn_input_memcfg: ttnn.MemoryConfig | None = None

    # Prefill settings
    prefill_input_memcfg: ttnn.MemoryConfig | None = None
    prefill_w1_w3_prg_config: PrefillProgramConfigFactory | None = None
    prefill_w2_prg_config: PrefillProgramConfigFactory | None = None
    prefill_w1_w3_output_memcfg: ttnn.MemoryConfig | None = None
    prefill_w2_output_memcfg: ttnn.MemoryConfig | None = None

    # Dtypes & Kernels
    w1_w3_dtype: ttnn.DataType | None = None
    w2_dtype: ttnn.DataType | None = None
    activation_dtype: ttnn.DataType | None = None
    ccl_dtype: ttnn.DataType | None = None
    mul_dtype: ttnn.DataType | None = None

    ff1_3_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    ff2_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    decode_ff1_3_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    decode_ff2_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    prefill_ff1_3_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    prefill_ff2_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    decode_activation_dtype: ttnn.DataType | None = None
    decode_ccl_dtype: ttnn.DataType | None = None
    decode_mul_dtype: ttnn.DataType | None = None
    prefill_activation_dtype: ttnn.DataType | None = None
    prefill_ccl_dtype: ttnn.DataType | None = None
    prefill_mul_dtype: ttnn.DataType | None = None

    prefill_len_cutoff: int | None = None

    def is_resolved(self) -> bool:
        """Check if all fields except optional ones are resolved."""
        # Collaborators are optional until the shared Prefetcher2D interface lands.
        optional = {
            "decode_prefetch_context",
            "prefill_prefetch_context",
            "decode_w1_w3_prg_config",
            "decode_w2_prg_config",
            "collective_resource_selector",
        }

        if self.collective_resource_selector is not None:
            optional.update(
                {
                    "decode_reduce_scatter_resources",
                    "decode_all_gather_resources",
                    "decode_all_reduce_resources",
                    "prefill_reduce_scatter_resources",
                    "prefill_all_gather_resources",
                    "prefill_all_reduce_resources",
                }
            )

        return all(getattr(self, f) is not None for f in self.__dataclass_fields__ if f not in optional)


# =============================================================================
# MLP2D - Unified MLP for 2D-topology devices (Galaxy)
# =============================================================================


class MLP2D(LightweightModule):
    """
    MLP for the Wormhole Galaxy (8, 4) mesh.

    Execution paths:
      Unified: linear → linear → reduce_scatter(×2) → mul+silu → all_gather → linear → all_reduce
    """

    def __init__(
        self,
        w1: LazyWeight,
        w2: LazyWeight,
        w3: LazyWeight,
        *,
        tt_ccl: Any,
        mesh_device: ttnn.MeshDevice | None = None,
    ):
        """
        Derive tensor policy from weights and an explicit Galaxy CCL owner.

        Args:
            w1: Gate projection weight (dim, hidden_dim)
            w2: Down projection weight (hidden_dim, dim)
            w3: Up projection weight (dim, hidden_dim)
        """
        super().__init__()
        self.config = _resolve_mlp2d_config(MLP2DConfig(w1=w1, w2=w2, w3=w3, tt_ccl=tt_ccl, mesh_device=mesh_device))
        self._loaded_weight_modes: set[str] = set()

    @classmethod
    def from_config(cls, config: MLP2DConfig):
        """
        Construct from a fully customizable config.
        """
        # bypass the __init__ method of the base class for power users who want to customize the config
        instance = object.__new__(cls)
        super(MLP2D, instance).__init__()
        instance.config = _resolve_mlp2d_config(config)
        instance._loaded_weight_modes = set()
        return instance

    def load_device_weights(self, mode: str | None = None):
        """Materialize LazyWeights onto device. Called automatically on first forward; idempotent."""
        assert self.config.is_resolved(), "config must be resolved before loading device weights!"
        modes = ("decode", "prefill") if mode is None else (mode,)
        for selected_mode in modes:
            if selected_mode in self._loaded_weight_modes:
                continue
            if selected_mode == "decode":
                self.w1 = self.config.w1.get_device_weight()
                self.w2 = self.config.w2.get_device_weight()
                self.w3 = self.config.w3.get_device_weight()
            elif selected_mode == "prefill":
                self.prefill_w1 = self.config.prefill_w1.get_device_weight()
                self.prefill_w2 = self.config.prefill_w2.get_device_weight()
                self.prefill_w3 = self.config.prefill_w3.get_device_weight()
            else:
                raise ValueError(f"mode must be 'decode' or 'prefill', got {selected_mode}")
            self._loaded_weight_modes.add(selected_mode)

    def _all_reduce_tg(
        self,
        input_tensor: ttnn.Tensor,
        cluster_axis: int,
        dim: int,
        sharded: bool,
        memory_config: Any,
        reduce_scatter_memory_config: Any = None,
        ccl_dtype: ttnn.DataType | None = None,
        mode: str = "decode",
    ) -> ttnn.Tensor:
        """
        All-reduce for Galaxy devices along the specified cluster axis.
        """
        cfg = self.config
        ccl_context = cfg.decode_ccl_context if mode == "decode" else cfg.prefill_ccl_context
        original_input = input_tensor
        # Ensure dim 0 and 1 are 1
        original_shape = input_tensor.shape
        if original_shape[0] != 1 or original_shape[1] != 1:
            input_tensor = ttnn.reshape(
                input_tensor, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
            )

        # Cast to CCL dtype
        ccl_dtype = ccl_dtype or cfg.ccl_dtype
        if input_tensor.dtype != ccl_dtype:
            input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG, ccl_dtype)
            if sharded and memory_config is not None:
                input_tensor = ttnn.to_memory_config(input_tensor, memory_config, ccl_dtype)

        if not sharded:
            input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

        if mode == "prefill":
            scattered_tensor = self._reduce_scatter(
                input_tensor,
                memory_config,
                mode,
                cluster_axis=cluster_axis,
                sequence_key="final",
                persistent=False,
            )
            reduced_tensor = self._all_gather(
                scattered_tensor,
                memory_config,
                mode,
                cluster_axis=cluster_axis,
                sequence_key="final",
                persistent=False,
            )
            if input_tensor is not original_input:
                ttnn.deallocate(input_tensor)
            return ttnn.reshape(reduced_tensor, original_shape)

        resources = _select_collective_resources(
            cfg,
            mode=mode,
            collective="all_reduce",
            cluster_axis=cluster_axis,
            tensor=input_tensor,
        )

        reduced_tensor = ttnn.experimental.all_reduce_async(
            input_tensor,
            resources.persistent_output_buffers[0],
            cluster_axis=cluster_axis,
            mesh_device=cfg.mesh_device,
            multi_device_global_semaphore=_next_semaphore(ccl_context, resources),
            num_links=resources.num_links,
            memory_config=memory_config,
            dtype=ccl_dtype,
            topology=resources.topology,
            subdevice_id=ccl_context.worker_sub_device_id,
            use_optimal_ccl_for_llama=True,
        )
        if input_tensor is not original_input:
            ttnn.deallocate(input_tensor)

        reduced_tensor = ttnn.reshape(reduced_tensor, original_shape)
        return reduced_tensor

    def _reduce_scatter(
        self,
        tensor: ttnn.Tensor,
        memory_config: Any,
        mode: str,
        *,
        cluster_axis: int,
        sequence_key: Any = None,
        persistent: bool = True,
    ) -> ttnn.Tensor:
        """Reduce scatter along an explicit Galaxy mesh axis."""
        cfg = self.config
        ccl_context = cfg.decode_ccl_context if mode == "decode" else cfg.prefill_ccl_context
        resources = _select_collective_resources(
            cfg,
            mode=mode,
            collective="reduce_scatter",
            cluster_axis=cluster_axis,
            tensor=tensor,
            sequence_key=sequence_key,
        )
        if not persistent:
            return ttnn.reduce_scatter(
                tensor,
                3,
                cluster_axis=cluster_axis,
                memory_config=memory_config,
                topology=resources.topology,
                num_links=resources.num_links,
                subdevice_id=ccl_context.worker_sub_device_id,
            )
        kwargs = dict(
            persistent_output_buffers=[*resources.intermediate_output_buffers, *resources.persistent_output_buffers],
            dim=3,
            multi_device_global_semaphore=_next_semaphore(ccl_context, resources),
            barrier_semaphore=_next_barrier(ccl_context, resources),
            num_links=resources.num_links,
            cluster_axis=cluster_axis,
            memory_config=memory_config,
            topology=resources.topology,
            subdevice_id=ccl_context.worker_sub_device_id,
        )
        if mode == "prefill":
            sequence_length = int(tensor.shape[1]) * int(tensor.shape[-2])
            kwargs["num_workers_per_link"] = 1 if sequence_length <= 128 else 4
        else:
            kwargs.update(
                intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                chunks_per_sync=cfg.ccl_chunks_per_sync,
                num_workers_per_link=cfg.ccl_num_workers_per_link,
                num_buffers_per_channel=cfg.ccl_num_buffers_per_channel,
            )
        return ttnn.experimental.reduce_scatter_minimal_async(tensor, **kwargs)

    def _reduce_scatter_axis1(
        self, tensor: ttnn.Tensor, memory_config: Any, mode: str, sequence_key: Any = None
    ) -> ttnn.Tensor:
        """Reduce scatter along cluster axis 1."""
        return self._reduce_scatter(tensor, memory_config, mode, cluster_axis=1, sequence_key=sequence_key)

    def _all_gather(
        self,
        tensor: ttnn.Tensor,
        memory_config: Any,
        mode: str,
        *,
        cluster_axis: int,
        sequence_key: Any = None,
        persistent: bool = True,
    ) -> ttnn.Tensor:
        """All gather along an explicit Galaxy mesh axis."""
        cfg = self.config
        ccl_context = cfg.decode_ccl_context if mode == "decode" else cfg.prefill_ccl_context
        resources = _select_collective_resources(
            cfg,
            mode=mode,
            collective="all_gather",
            cluster_axis=cluster_axis,
            tensor=tensor,
            sequence_key=sequence_key,
        )
        if not persistent:
            return ttnn.all_gather(
                tensor,
                3,
                cluster_axis=cluster_axis,
                memory_config=memory_config,
                topology=resources.topology,
                num_links=resources.num_links,
                subdevice_id=ccl_context.worker_sub_device_id,
            )
        return ttnn.experimental.all_gather_async(
            tensor,
            3,
            multi_device_global_semaphore=_next_semaphore_window(ccl_context, resources),
            num_links=resources.num_links,
            cluster_axis=cluster_axis,
            mesh_device=cfg.mesh_device,
            topology=resources.topology,
            memory_config=memory_config,
            persistent_output_tensor=resources.persistent_output_buffers[0],
            barrier_semaphore=None,
            subdevice_id=ccl_context.worker_sub_device_id,
            use_optimal_ccl_for_llama=mode == "decode",
        )

    def _all_gather_axis1(
        self, tensor: ttnn.Tensor, memory_config: Any, mode: str, sequence_key: Any = None
    ) -> ttnn.Tensor:
        """All gather along cluster axis 1."""
        return self._all_gather(tensor, memory_config, mode, cluster_axis=1, sequence_key=sequence_key)

    def _double_matmul_reduce_scatter_axis1(self, input_tensor: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run decode W1/W3 matmuls and fuse the W1 axis-1 reduction."""
        cfg = self.config
        context = cfg.decode_ccl_context
        resources = _select_collective_resources(
            cfg,
            mode="decode",
            collective="reduce_scatter",
            cluster_axis=1,
            tensor=(1, 1, cfg.max_batch_size, cfg.hidden_dim // WH_GALAXY_MESH_SHAPE[0]),
        )
        semaphore = _next_semaphore(context, resources)
        if isinstance(semaphore, (tuple, list)):
            semaphore = semaphore[0]
        outputs = ttnn.experimental.llama_rs_matmul(
            input_tensor,
            self.w1,
            resources.intermediate_output_buffers[0],
            3,
            semaphore,
            1,
            cfg.mesh_device,
            resources.num_links,
            context.worker_sub_device_id,
            second_weight_tensor=self.w3,
            topology=resources.topology,
            memory_config_rs=cfg.ff1_out_reduce_scatter_memcfg,
            memory_config_mm=cfg.decode_w1_w3_output_memcfg,
            compute_kernel_config=cfg.decode_ff1_3_compute_kernel_cfg,
            dtype=cfg.decode_activation_dtype,
            program_config=cfg.decode_w1_w3_prg_config,
            global_cb=getattr(cfg.decode_prefetch_context, "global_cb", None),
        )
        if len(outputs) != 3:
            raise RuntimeError(f"llama_rs_matmul returned {len(outputs)} outputs; expected 3")
        first_projection, w3_projection, w1_reduced = outputs
        ttnn.deallocate(first_projection)
        return w1_reduced, w3_projection

    def _llama_reduce_scatter_axis1(self, tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Reduce the decode W3 projection with the fused path's padded geometry."""
        cfg = self.config
        context = cfg.decode_ccl_context
        resources = _select_collective_resources(
            cfg,
            mode="decode",
            collective="reduce_scatter",
            cluster_axis=1,
            tensor=(1, 1, cfg.max_batch_size, cfg.hidden_dim // WH_GALAXY_MESH_SHAPE[0]),
        )
        semaphore = _next_semaphore(context, resources)
        if isinstance(semaphore, (tuple, list)):
            semaphore = semaphore[0]
        return ttnn.experimental.llama_reduce_scatter(
            tensor,
            resources.intermediate_output_buffers[0],
            3,
            semaphore,
            context.worker_sub_device_id,
            cluster_axis=1,
            mesh_device=cfg.mesh_device,
            num_links=resources.num_links,
            memory_config=cfg.ff1_out_reduce_scatter_memcfg,
            topology=resources.topology,
        )

    def decode_forward(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        """
        Wormhole Galaxy decode forward.

        Unified Path: linear → linear → reduce_scatter(×2) → mul+silu → all_gather → linear → all_reduce
        """
        self.load_device_weights("decode")
        owns_input = isinstance(x, LazyWeight)
        x = _load_input_device_tensor(x, self.config, mode="decode")
        cfg = self.config
        prefetch_kwargs = _prefetch_kwargs(cfg.decode_prefetch_context)

        # --- STAGE 1-2: Fused W1/W3 ring matmuls and axis-1 reduce-scatter ---
        w1_out, w3_projection = self._double_matmul_reduce_scatter_axis1(x)
        if owns_input:
            ttnn.deallocate(x)

        # llama_rs_matmul reduces W1 only; W3 is returned as a projection.
        w3_out = self._llama_reduce_scatter_axis1(w3_projection)
        ttnn.deallocate(w3_projection)

        # --- STAGE 3: Activation + Multiply ---
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[cfg.mlp_activation_type],
            dtype=cfg.decode_mul_dtype,
            memory_config=cfg.ff1_out_reduce_scatter_memcfg,
        )

        # --- STAGE 4: All-gather before W2 ---
        gated = w2_in
        w2_in = self._all_gather_axis1(gated, cfg.decode_w2_input_memcfg, "decode")
        ttnn.deallocate(gated)

        # --- STAGE 5: W2 Linear ---
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=cfg.decode_ff2_compute_kernel_cfg,
            dtype=cfg.decode_ccl_dtype,
            program_config=cfg.decode_w2_prg_config,
            memory_config=cfg.decode_w2_output_memcfg,
            core_grid=None,
            **prefetch_kwargs,
        )
        # --- STAGE 6: Final All-Reduce ---
        w2_out_reduced = self._all_reduce_tg(
            w2_out,
            cluster_axis=0,
            dim=3,
            sharded=True,
            memory_config=cfg.ff2_out_reduce_scatter_memcfg,
            reduce_scatter_memory_config=cfg.ff2_out_reduce_scatter_memcfg,
            ccl_dtype=cfg.decode_ccl_dtype,
            mode="decode",
        )
        ttnn.deallocate(w2_out)

        # --- STAGE 7: Reshape + Final memory config ---
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )
        if cfg.sharded_attn_input_memcfg is not None:
            w2_out_reduced = ttnn.to_memory_config(w2_out_reduced, cfg.sharded_attn_input_memcfg)

        return w2_out_reduced

    def prefill_forward(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        """
        Wormhole Galaxy prefill forward.

        Unified Path: [reshape] → linear → linear → reduce_scatter(×2) → mul+silu → all_gather → linear → all_reduce → reshape
        """
        self.load_device_weights("prefill")
        owns_input = isinstance(x, LazyWeight)
        x = _load_input_device_tensor(x, self.config, mode="prefill")
        cfg = self.config
        seq_len = x.shape[-2]
        prefetch_kwargs = _prefetch_kwargs(cfg.prefill_prefetch_context)

        # Seq_len-dependent: reshape for long sequences
        if seq_len >= cfg.prefill_len_cutoff:
            assert (
                seq_len % cfg.prefill_len_cutoff == 0
            ), f"seq_len ({seq_len}) must be divisible by prefill_len_cutoff ({cfg.prefill_len_cutoff})"
            x = ttnn.reshape(x, [1, seq_len // cfg.prefill_len_cutoff, cfg.prefill_len_cutoff, -1])
            owns_input = True

        # Seq_len-dependent: get program configs (None = let TTNN pick defaults)
        pc_w1_w3 = cfg.prefill_w1_w3_prg_config(seq_len) if cfg.prefill_w1_w3_prg_config else None
        pc_w2 = cfg.prefill_w2_prg_config(seq_len) if cfg.prefill_w2_prg_config else None

        # --- STAGE 1: W1/W3 Linear (DRAM) ---
        w1_out = ttnn.linear(
            x,
            self.prefill_w1,
            dtype=cfg.prefill_activation_dtype,
            core_grid=None,
            compute_kernel_config=cfg.prefill_ff1_3_compute_kernel_cfg,
            program_config=pc_w1_w3,
            memory_config=cfg.prefill_w1_w3_output_memcfg,
            **prefetch_kwargs,
        )
        w3_out = ttnn.linear(
            x,
            self.prefill_w3,
            dtype=cfg.prefill_activation_dtype,
            core_grid=None,
            compute_kernel_config=cfg.prefill_ff1_3_compute_kernel_cfg,
            program_config=pc_w1_w3,
            memory_config=cfg.prefill_w1_w3_output_memcfg,
            **prefetch_kwargs,
        )
        if owns_input:
            ttnn.deallocate(x)

        # --- STAGE 2: CCL after W1/W3 (reduce_scatter for prefill) ---
        input_mem_cfg = w1_out.memory_config()

        w1_projection, w3_projection = w1_out, w3_out
        w1_out = self._reduce_scatter_axis1(w1_projection, None, "prefill", "w1")
        w3_out = self._reduce_scatter_axis1(w3_projection, None, "prefill", "w3")
        ttnn.deallocate(w1_projection)
        ttnn.deallocate(w3_projection)

        # --- STAGE 3: Activation + Multiply ---
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[cfg.mlp_activation_type],
            dtype=cfg.prefill_mul_dtype,
            memory_config=w1_out.memory_config(),
        )

        # --- STAGE 4: All-gather before W2 ---
        gated = w2_in
        w2_in = self._all_gather_axis1(gated, input_mem_cfg, "prefill", "gated")
        ttnn.deallocate(gated)
        # No L1 conversion for prefill

        # --- STAGE 5: W2 Linear ---
        w2_out = ttnn.linear(
            w2_in,
            self.prefill_w2,
            compute_kernel_config=cfg.prefill_ff2_compute_kernel_cfg,
            dtype=cfg.prefill_ccl_dtype,
            program_config=pc_w2,
            memory_config=cfg.prefill_w2_output_memcfg,
            core_grid=None,
            **prefetch_kwargs,
        )
        # --- STAGE 6: Final All-Reduce ---
        w2_out_reduced = self._all_reduce_tg(
            w2_out,
            cluster_axis=0,
            dim=3,
            sharded=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ccl_dtype=cfg.prefill_ccl_dtype,
            mode="prefill",
        )
        ttnn.deallocate(w2_out)

        # --- STAGE 7: Reshape (no final memory config change for prefill) ---
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

        return w2_out_reduced

    def forward(self, x: ttnn.Tensor | LazyWeight, mode: str) -> ttnn.Tensor:
        """Dispatch to the appropriate forward method based on mode."""
        if mode == "decode":
            return self.decode_forward(x)
        if mode == "prefill":
            return self.prefill_forward(x)
        raise ValueError(f"mode must be 'decode' or 'prefill', got {mode}")


# =============================================================================
# Helper functions
# =============================================================================


# todo)) work with the CCL team to find opportunity to simplify this --> e.g., build into TTNN APIs?


def _compute_kernel_config_hifi2_fp16() -> ttnn.WormholeComputeKernelConfig:
    """Default compute kernel config for MLP (HiFi2 with FP16 accumulation)."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _default_prefill_program_config(seq_len: int) -> None:
    """Explicit policy selecting TTNN's sequence-aware program-config resolution."""
    assert seq_len > 0
    return None


def _prefetch_kwargs(context: Any) -> dict[str, Any]:
    """Translate the pending Prefetcher2D context contract into TTNN kwargs."""
    if context is None:
        return {}
    return {
        "global_cb": getattr(context, "global_cb", None),
        "sub_device_id": getattr(context, "worker_sub_device_id", getattr(context, "sub_device_id", None)),
    }


def _resolve_ccl_context(context: Any, *, tt_ccl: Any, mode: str, mesh_device: Any) -> Any:
    if context is None:
        factory = getattr(tt_ccl, "context", None)
        if not callable(factory):
            raise TypeError("Galaxy CCL collaborator must provide context(mode)")
        context = factory(mode)
    if getattr(context, "mesh_device", None) is not mesh_device:
        raise ValueError(f"{mode} CCL context must belong to the configured mesh")
    if getattr(context, "mode", None) != mode:
        raise ValueError(f"{mode} CCL context has mode={getattr(context, 'mode', None)}")
    for method in ("resources", "next_semaphore_handles", "next_semaphore_window", "next_barrier_semaphore_handle"):
        if not callable(getattr(context, method, None)):
            raise TypeError(f"{mode} CCL context must provide {method}()")
    if getattr(context, "worker_sub_device_id", None) is None:
        raise ValueError(f"{mode} CCL context requires worker_sub_device_id")
    return context


def _resolve_collective_resources(context: Any, *, mode: str, collective: str, cluster_axis: int) -> Any:
    resources = context.resources(collective, cluster_axis)
    if resources is None or resources.cluster_axis != cluster_axis:
        raise ValueError(f"{mode} {collective} resources must target cluster_axis={cluster_axis}")
    if resources.topology is None or resources.num_links < 1:
        raise ValueError(f"{mode} {collective} topology and num_links must be resolved")
    if collective == "reduce_scatter":
        if not resources.intermediate_output_buffers or not resources.persistent_output_buffers:
            raise ValueError(f"{mode} reduce_scatter requires persistent intermediate and output buffers")
    elif not resources.persistent_output_buffers:
        raise ValueError(f"{mode} {collective} requires a persistent output buffer")
    if getattr(resources, "key", None) is None:
        raise ValueError(f"{mode} {collective} resources require an exact resource key")
    return resources


def _select_collective_resources(
    config: MLP2DConfig,
    *,
    mode: str,
    collective: str,
    cluster_axis: int,
    tensor: Any,
    sequence_key: Any = None,
) -> Any:
    selector = config.collective_resource_selector
    if selector is not None:
        context = config.decode_ccl_context if mode == "decode" else config.prefill_ccl_context
        resources = selector(context, collective, cluster_axis, tensor, sequence_key)
        return _validate_collective_resources(resources, mode=mode, collective=collective, cluster_axis=cluster_axis)
    return getattr(config, f"{mode}_{collective}_resources")


def _validate_collective_resources(resources: Any, *, mode: str, collective: str, cluster_axis: int) -> Any:
    if resources is None or resources.cluster_axis != cluster_axis:
        raise ValueError(f"{mode} {collective} resources must target cluster_axis={cluster_axis}")
    if resources.topology is None or resources.num_links < 1:
        raise ValueError(f"{mode} {collective} topology and num_links must be resolved")
    if collective == "reduce_scatter":
        if not resources.intermediate_output_buffers or not resources.persistent_output_buffers:
            raise ValueError(f"{mode} reduce_scatter requires persistent intermediate and output buffers")
    elif not resources.persistent_output_buffers:
        raise ValueError(f"{mode} {collective} requires a persistent output buffer")
    if getattr(resources, "key", None) is None:
        raise ValueError(f"{mode} {collective} resources require an exact resource key")
    return resources


def _next_semaphore(context: Any, resources: Any) -> Any:
    key = resources.key
    return context.next_semaphore_handles(key.operation, key.cluster_axis, key.geometry, key.sequence_key)


def _next_semaphore_window(context: Any, resources: Any) -> Any:
    key = resources.key
    return context.next_semaphore_window(
        key.operation,
        key.cluster_axis,
        key.geometry,
        key.sequence_key,
        count=2,
    )


def _next_barrier(context: Any, resources: Any) -> Any:
    key = resources.key
    return context.next_barrier_semaphore_handle(key.operation, key.cluster_axis, key.geometry, key.sequence_key)


def _resolve_mlp2d_config(config: MLP2DConfig) -> MLP2DConfig:
    """Materialize the config to known good defaults using replace pattern."""
    to_set = {}

    if not isinstance(config.mlp_activation_type, ttnn.UnaryOpType):
        raise TypeError("mlp_activation_type must be a ttnn.UnaryOpType")
    if config.collective_resource_selector is not None and not callable(config.collective_resource_selector):
        raise TypeError("collective_resource_selector must be callable")
    for field_name in ("prefill_w1_w3_prg_config", "prefill_w2_prg_config"):
        factory = getattr(config, field_name)
        if factory is not None and not callable(factory):
            raise TypeError(f"{field_name} must be callable")
    if config.prefill_len_cutoff is not None and config.prefill_len_cutoff <= 0:
        raise ValueError("prefill_len_cutoff must be positive")

    # --- Phase 1: Foundational fields ---

    # Derive dimensions
    dim = config.dim
    if config.dim is None:
        dim = config.w1.source.shape[-2]
        to_set["dim"] = dim

    hidden_dim = config.hidden_dim
    if config.hidden_dim is None:
        hidden_dim = config.w1.source.shape[-1]
        to_set["hidden_dim"] = hidden_dim

    prefill_weights = (
        config.prefill_w1 or config.w1,
        config.prefill_w2 or config.w2,
        config.prefill_w3 or config.w3,
    )
    for field_name, weight in zip(("prefill_w1", "prefill_w2", "prefill_w3"), prefill_weights):
        if getattr(config, field_name) is None:
            to_set[field_name] = weight

    # Derive mesh_device
    mesh_device = config.mesh_device
    if mesh_device is None:
        mesh_device = config.w1.device
    if mesh_device is None:
        mesh_device = ttnn.GetDefaultDevice()
    if config.mesh_device is None:
        to_set["mesh_device"] = mesh_device

    assert mesh_device is not None, "mesh_device must be available"
    cluster_shape = tuple(mesh_device.shape)
    assert (
        cluster_shape == WH_GALAXY_MESH_SHAPE
    ), f"MLP2D requires WH Galaxy mesh {WH_GALAXY_MESH_SHAPE}, got {cluster_shape}"
    assert mesh_device.get_num_devices() == 32, "MLP2D requires exactly 32 devices"
    assert mesh_device.arch() == ttnn.device.Arch.WORMHOLE_B0, "MLP2D requires Wormhole"

    assert dim % cluster_shape[1] == 0, f"dim={dim} must be divisible by Galaxy columns={cluster_shape[1]}"
    assert (
        hidden_dim % cluster_shape[0] == 0
    ), f"hidden_dim={hidden_dim} must be divisible by Galaxy rows={cluster_shape[0]}"
    assert tuple(config.w1.source.shape[-2:]) == (dim, hidden_dim), "w1 must have shape (dim, hidden_dim)"
    assert tuple(config.w3.source.shape[-2:]) == (dim, hidden_dim), "w3 must have shape (dim, hidden_dim)"
    assert tuple(config.w2.source.shape[-2:]) == (hidden_dim, dim), "w2 must have shape (hidden_dim, dim)"
    assert tuple(prefill_weights[0].source.shape[-2:]) == (dim, hidden_dim), "prefill_w1 must match w1 shape"
    assert tuple(prefill_weights[2].source.shape[-2:]) == (dim, hidden_dim), "prefill_w3 must match w3 shape"
    assert tuple(prefill_weights[1].source.shape[-2:]) == (hidden_dim, dim), "prefill_w2 must match w2 shape"

    for weight in (config.w1, config.w2, config.w3, *prefill_weights):
        weight_device = getattr(weight, "device", None)
        assert weight_device is None or weight_device is mesh_device, "all weights must belong to the configured mesh"
    for mode, context in (
        ("decode", config.decode_prefetch_context),
        ("prefill", config.prefill_prefetch_context),
    ):
        context_mesh = getattr(context, "mesh_device", mesh_device)
        assert context_mesh is mesh_device, "prefetch context must belong to the configured mesh"
        context_mode = getattr(context, "mode", mode)
        assert context_mode == mode, f"{mode} prefetch context has mode={context_mode}"

    # Resolve the model-owned Galaxy CCL collaborator before any hot path runs.
    tt_ccl = config.tt_ccl
    if tt_ccl is None:
        raise ValueError("MLP2D requires an injected Galaxy CCL collaborator")
    ccl_mesh = getattr(tt_ccl, "mesh_device", mesh_device)
    assert ccl_mesh is mesh_device, "CCL collaborator must belong to the configured mesh"
    for mode in ("decode", "prefill"):
        context = _resolve_ccl_context(
            getattr(config, f"{mode}_ccl_context"), tt_ccl=tt_ccl, mode=mode, mesh_device=mesh_device
        )
        to_set[f"{mode}_ccl_context"] = context
        if config.collective_resource_selector is None:
            for collective, axis in (("reduce_scatter", 1), ("all_gather", 1), ("all_reduce", 0)):
                to_set[f"{mode}_{collective}_resources"] = _resolve_collective_resources(
                    context, mode=mode, collective=collective, cluster_axis=axis
                )

    # Galaxy's supported physical routes use a linear topology.
    topology = config.topology
    if config.topology is None:
        topology = ttnn.Topology.Linear
        to_set["topology"] = topology

    # --- Phase 2: Dtypes and Tunings ---

    w1_w3_dtype = config.w1_w3_dtype or ttnn.bfloat8_b
    to_set["w1_w3_dtype"] = w1_w3_dtype
    w2_dtype = config.w2_dtype or ttnn.bfloat8_b
    to_set["w2_dtype"] = w2_dtype

    activation_dtype = config.activation_dtype or ttnn.bfloat8_b
    ccl_dtype = config.ccl_dtype or ttnn.bfloat8_b
    mul_dtype = config.mul_dtype or activation_dtype
    to_set.update(
        activation_dtype=activation_dtype,
        ccl_dtype=ccl_dtype,
        mul_dtype=mul_dtype,
        decode_activation_dtype=config.decode_activation_dtype or activation_dtype,
        decode_ccl_dtype=config.decode_ccl_dtype or ccl_dtype,
        decode_mul_dtype=config.decode_mul_dtype or mul_dtype,
        prefill_activation_dtype=config.prefill_activation_dtype or activation_dtype,
        prefill_ccl_dtype=config.prefill_ccl_dtype or ccl_dtype,
        prefill_mul_dtype=config.prefill_mul_dtype or mul_dtype,
    )

    if config.prefill_len_cutoff is None:
        to_set["prefill_len_cutoff"] = 1024

    # Compute kernel configs
    if config.ff1_3_compute_kernel_cfg is None:
        to_set["ff1_3_compute_kernel_cfg"] = _compute_kernel_config_hifi2_fp16()
    if config.ff2_compute_kernel_cfg is None:
        to_set["ff2_compute_kernel_cfg"] = _compute_kernel_config_hifi2_fp16()
    ff1_kernel = config.ff1_3_compute_kernel_cfg or to_set["ff1_3_compute_kernel_cfg"]
    ff2_kernel = config.ff2_compute_kernel_cfg or to_set["ff2_compute_kernel_cfg"]
    to_set.update(
        decode_ff1_3_compute_kernel_cfg=config.decode_ff1_3_compute_kernel_cfg or ff1_kernel,
        decode_ff2_compute_kernel_cfg=config.decode_ff2_compute_kernel_cfg or ff2_kernel,
        prefill_ff1_3_compute_kernel_cfg=config.prefill_ff1_3_compute_kernel_cfg or ff1_kernel,
        prefill_ff2_compute_kernel_cfg=config.prefill_ff2_compute_kernel_cfg or ff2_kernel,
    )

    # --- Phase 2.5: Input Memory Configs ---

    if config.decode_input_memcfg is None:
        to_set["decode_input_memcfg"] = ttnn.L1_MEMORY_CONFIG
    if config.decode_w2_input_memcfg is None:
        to_set["decode_w2_input_memcfg"] = ttnn.L1_MEMORY_CONFIG

    if config.prefill_input_memcfg is None:
        to_set["prefill_input_memcfg"] = ttnn.DRAM_MEMORY_CONFIG

    for field_name, default in (
        ("decode_w1_w3_output_memcfg", ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG),
        ("decode_w2_output_memcfg", ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG),
        ("ff1_out_reduce_scatter_memcfg", ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG),
        ("ff2_out_reduce_scatter_memcfg", ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG),
        ("sharded_attn_input_memcfg", ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG),
        ("prefill_w1_w3_output_memcfg", ttnn.DRAM_MEMORY_CONFIG),
        ("prefill_w2_output_memcfg", ttnn.DRAM_MEMORY_CONFIG),
    ):
        if getattr(config, field_name) is None:
            to_set[field_name] = default

    # --- Phase 3: Prefill Program Configs ---
    # Factories are always resolved, even when the selected policy delegates geometry to TTNN.
    if config.prefill_w1_w3_prg_config is None:
        to_set["prefill_w1_w3_prg_config"] = _default_prefill_program_config
    if config.prefill_w2_prg_config is None:
        to_set["prefill_w2_prg_config"] = _default_prefill_program_config

    # --- Phase 4: Resolve Weights (always 2D sharded for MLP2D) ---

    # Galaxy weights use DRAM interleaved (no shard memory config on weights themselves).
    w1_w3_memcfg = config.w1_w3_memcfg or ttnn.DRAM_MEMORY_CONFIG
    to_set["w1_w3_memcfg"] = w1_w3_memcfg
    w2_memcfg = config.w2_memcfg or ttnn.DRAM_MEMORY_CONFIG
    to_set["w2_memcfg"] = w2_memcfg

    # MLP2D ALWAYS uses 2D sharding - this is fundamental to how 2D mesh MLP works.
    # w1/w3: shard dims (-1, -2) = N sharded on mesh axis 0, K sharded on mesh axis 1
    # w2: shard dims (-2, -1) = K sharded on mesh axis 0, N sharded on mesh axis 1
    w1_w3_shard_dims = (-1, -2)
    w2_shard_dims = (-2, -1)

    def get_weight_mesh_mapper(lazy_weight: LazyWeight, shard_dims: tuple[int, int]):
        """Return existing mesh_mapper_config if set, else create 2D shard mapper."""
        existing = getattr(lazy_weight, "mesh_mapper_config", None)
        if existing is not None:
            return existing
        # Default: apply 2D sharding
        return ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(shard_dims[0]), ttnn.PlacementShard(shard_dims[1])],
            mesh_shape_override=ttnn.MeshShape(cluster_shape),
        )

    to_set["w1"] = resolve_lazy_weight(
        config.w1,
        device=mesh_device,
        memory_config=w1_w3_memcfg,
        mesh_mapper_config=get_weight_mesh_mapper(config.w1, w1_w3_shard_dims),
        layout=ttnn.TILE_LAYOUT,
        dtype=w1_w3_dtype,
    )

    to_set["w2"] = resolve_lazy_weight(
        config.w2,
        device=mesh_device,
        memory_config=w2_memcfg,
        mesh_mapper_config=get_weight_mesh_mapper(config.w2, w2_shard_dims),
        layout=ttnn.TILE_LAYOUT,
        dtype=w2_dtype,
    )

    to_set["w3"] = resolve_lazy_weight(
        config.w3,
        device=mesh_device,
        memory_config=w1_w3_memcfg,
        mesh_mapper_config=get_weight_mesh_mapper(config.w3, w1_w3_shard_dims),
        layout=ttnn.TILE_LAYOUT,
        dtype=w1_w3_dtype,
    )

    # Prefill matmul requires interleaved DRAM weights, while decode may use a
    # ring-specific sharded DRAM layout. Keep the two materializations distinct.
    to_set["prefill_w1"] = resolve_lazy_weight(
        prefill_weights[0],
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper_config=get_weight_mesh_mapper(prefill_weights[0], w1_w3_shard_dims),
        layout=ttnn.TILE_LAYOUT,
        dtype=w1_w3_dtype,
    )
    to_set["prefill_w2"] = resolve_lazy_weight(
        prefill_weights[1],
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper_config=get_weight_mesh_mapper(prefill_weights[1], w2_shard_dims),
        layout=ttnn.TILE_LAYOUT,
        dtype=w2_dtype,
    )
    to_set["prefill_w3"] = resolve_lazy_weight(
        prefill_weights[2],
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper_config=get_weight_mesh_mapper(prefill_weights[2], w1_w3_shard_dims),
        layout=ttnn.TILE_LAYOUT,
        dtype=w1_w3_dtype,
    )

    resolved_config = replace(config, **to_set)
    assert resolved_config.is_resolved(), "Config must be resolved!"
    return resolved_config


def _load_input_device_tensor(x: ttnn.Tensor | LazyWeight, config: MLP2DConfig, mode: str) -> ttnn.Tensor:
    """Resolve the input tensor to ttnn tensor if x is a LazyWeight, otherwise return as is."""
    assert mode in ["decode", "prefill"], "mode must be one of decode or prefill!"
    mem_cfg = config.decode_input_memcfg if mode == "decode" else config.prefill_input_memcfg
    if isinstance(x, LazyWeight):
        # For MLP2D, input must be sharded to match weight sharding:
        # - w1/w3 shard dims = (-1, -2): K sharded on mesh axis 1
        # - Input [batch, 1, seq, K]: shard K (dim -1) on mesh axis 1, replicate on axis 0
        cluster_shape = list(config.mesh_device.shape)
        input_mesh_mapper = ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)],
            mesh_shape_override=ttnn.MeshShape(cluster_shape),
        )
        resolved_x = resolve_lazy_weight(
            x,
            device=config.mesh_device,
            memory_config=mem_cfg,
            mesh_mapper_config=input_mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )
        return resolved_x.get_device_weight()

    assert isinstance(x, ttnn.Tensor), "x must be a ttnn tensor at this point!"
    return x
