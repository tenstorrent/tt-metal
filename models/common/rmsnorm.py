# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import math
import os

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import copy_to_buffer
from models.tt_transformers.tt.common import Mode

TILE = 32
SHARD_HEIGHT = TILE  # Current ttnn.rms_norm implementation requires shard height to be a single tile


class RMSNorm(LightweightModule):
    """
    RMSNorm supporting replication over a MeshDevice and sharding within devices.

    This class implements a Root Mean Square Normalization (RMSNorm) that can be
    distributed across multiple devices and cores. If the `device` parameter is a
    MeshDevice, the weights and computations are replicated across all devices in
    the mesh. Expects an interleaved input tensor, can optionally output a sharded tensor.

    Args:
        device: The device or MeshDevice on which to perform the computations.
        state_dict: The state dictionary containing the model parameters.
        dim: Input dimension (e.g. model hidden dimension size).
        layer_num: The layer number to determine the weight key in the state dictionary.
        weight_key: The key for retrieving the weight from the state dictionary.
        weight_cache_path: Optional path for caching the tilized weights.
        weight_memory_config: Configuration for the weight memory, default is DRAM_MEMORY_CONFIG.
        weight_dtype: The data type for the tensors, bfp8_b hits >0.999 PCC in the models we tested.
        model_config: Optional configuration dictionary for the model.
        eps (float): Small value to avoid division by zero in normalization, default is 1e-05.

    If model_config is provided, it must specify SHARDED_NORM_INPUT_MEMCFG, SHARDED_NORM_PRGM_CFG
    and SHARDED_NORM_OUTPUT_MEMCFG. If not provided, default configurations will be generated.
    """

    def __init__(
        self,
        device,
        dim,
        state_dict,
        weight_key,
        layer_num=None,
        state_dict_prefix=None,
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat16,
        is_distributed=None,
        eps: float = 1e-05,
        add_unit_offset=False,
        sharded_program_config=None,
        sharded_output_config=None,
        output_mem_config=None,
        ccl_topology=ttnn.Topology.Ring,
        tt_ccl=None,
        fp32_dest_acc_en=True,
    ):
        super().__init__()
        self.device = device
        self.eps = eps
        self.is_distributed = is_distributed
        self.ccl_topology = ccl_topology
        self.tt_ccl = tt_ccl
        self.add_unit_offset = add_unit_offset

        if state_dict_prefix:
            weight_name = f"{state_dict_prefix}{weight_key}.weight"
        else:
            if layer_num is None:
                weight_name = f"{weight_key}.weight"
            else:
                weight_name = f"layers.{layer_num}.{weight_key}.weight"

        torch_weight = (
            state_dict[weight_name].unsqueeze(0).view(1, 1, dim).reshape([1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT])
        )

        # Add offset before caching
        if add_unit_offset:
            torch_weight = torch_weight + 1.0

        # Compatibility with models that don't use mesh devices (e.g. single-chip Mistral-7b)
        is_mesh_device = device.__class__.__name__ == "MeshDevice"

        self.weight = ttnn.as_tensor(
            torch_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=None if weight_cache_path is None else weight_cache_path / weight_name,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        if self.is_distributed:
            self.weight_distributed = ttnn.as_tensor(
                torch_weight,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=weight_memory_config,
                cache_file_name=(
                    None if weight_cache_path is None else weight_cache_path / (weight_name + "_distributed")
                ),
                mesh_mapper=(
                    ttnn.ShardTensor2dMesh(device, dims=(None, 2), mesh_shape=list(device.shape))
                    if is_mesh_device
                    else None
                ),
            )

        self.sharded_output_config = sharded_output_config
        self.sharded_program_config = sharded_program_config
        self.output_mem_config = output_mem_config

        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=True,
        )

    def update(self, *, weight: ttnn.Tensor) -> None:
        """In-place replace the RMSNorm gamma via ``ttnn.copy``.

        HF-format input: ``weight`` is HF ``...norm.weight``, shape
        ``(1, 1, 1, dim)``, bf16, TILE, DRAM-interleaved, replicated.

        ``copy_to_buffer`` reshapes to the storage shape
        ``(1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT)`` and TILE -> ROW_MAJOR to
        match ``self.weight``. ``add_unit_offset`` is not supported (see
        assert): the caller must ship a gamma that already includes the +1.

        When ``self.weight_distributed`` (the column-sharded mirror) exists it's
        kept in sync on device: project ``self.weight`` into the sharded layout
        via ``ttnn.mesh_partition`` (the inverse of the constructor's
        ``ShardTensor2dMesh(dims=(None, 2))``, hence ``dim=2, cluster_axis=1``)
        and ``ttnn.copy`` into it. Both buffers keep their address, so captured
        traces and the prefetcher's recorded addresses stay valid.
        """
        assert not self.add_unit_offset, "RMSNorm.update does not support add_unit_offset=True"
        copy_to_buffer(weight, self.weight, self.weight.dtype)

        if getattr(self, "weight_distributed", None) is not None:
            partitioned = ttnn.mesh_partition(
                self.weight,
                memory_config=self.weight_distributed.memory_config(),
                dim=2,
                cluster_axis=1,
            )
            copy_to_buffer(partitioned, self.weight_distributed, self.weight_distributed.dtype)

    # Rows per chunk of the PCIe-mesh prefill norm (see _pcie_prefill_norm); fixes its per-core L1 footprint.
    PCIE_PREFILL_NORM_ROWS = 512

    def _pcie_prefill_norm(self, x, weight):
        """Prefill RMSNorm on a PCIe P2P mesh with a bounded L1 footprint.

        There the residual is replicated, so each device normalizes the full hidden width, and the interleaved norm
        keeps whole rows in static circular buffers: ~1.4 MB per core at hidden 5120, which only fits while nothing
        else holds L1 and clashes with PcieCCL's persistent decode landing buffers once a decode has run (any second
        request). Here each chunk of <= 128 rows is width-sharded over a core rectangle that divides the width, so the
        footprint is per-core slices only. Same accuracy as the interleaved norm (PCC vs fp32 reference 0.99999).
        Returns None (caller falls back) for shapes it does not cover."""
        w = x.padded_shape[-1]
        rows = math.prod(x.padded_shape) // w
        if x.layout != ttnn.TILE_LAYOUT or w % 32 or rows % 32:
            return None
        wt = w // 32
        grid = self.device.compute_with_storage_grid_size()
        best = None
        for gy in range(1, grid.y + 1):
            for gx in range(1, grid.x + 1):
                n = gx * gy
                if wt % n == 0 and (best is None or n > best[0] * best[1]) and n <= 64:
                    best = (gx, gy)
        if best is None:
            return None
        gx, gy = best
        n = gx * gy
        bw = wt // n
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
        out_mc = x.memory_config()
        shape = list(x.shape)
        x2 = ttnn.reshape(x, [1, 1, rows, w]) if list(x.padded_shape) != [1, 1, rows, w] else x
        outs = []
        for r0 in range(0, rows, self.PCIE_PREFILL_NORM_ROWS):
            r1 = min(rows, r0 + self.PCIE_PREFILL_NORM_ROWS)
            piece = ttnn.slice(x2, [0, 0, r0, 0], [1, 1, r1, w]) if (r0, r1) != (0, rows) else x2
            sh = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(core_grid, [r1 - r0, w // n], ttnn.ShardOrientation.ROW_MAJOR),
            )
            # Not deallocated: a row slice can share the caller's buffer (the residual is still in use).
            xs = ttnn.to_memory_config(piece, sh)
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                subblock_w=bw,
                block_h=(r1 - r0) // 32,
                block_w=bw,
                inplace=False,
            )
            ys = ttnn.rms_norm(
                xs,
                epsilon=self.eps,
                weight=weight,
                program_config=pc,
                memory_config=sh,
                compute_kernel_config=self.compute_kernel_config_hifi2,
            )
            ttnn.deallocate(xs)
            outs.append(ttnn.to_memory_config(ys, out_mc))
            ttnn.deallocate(ys)
        y = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2, memory_config=out_mc)
        if len(outs) > 1:
            for o in outs:
                ttnn.deallocate(o)
        return ttnn.reshape(y, shape) if list(y.shape) != shape else y

    def forward(
        self,
        x: ttnn.Tensor,
        mode: Mode | str,
        in_sharded=False,
        out_sharded=False,
        norm_config=None,
    ) -> ttnn.Tensor:
        if isinstance(mode, str):
            try:
                mode = Mode(mode)
            except ValueError:
                raise ValueError(f"Invalid mode: {mode}")
        elif not isinstance(mode, Mode):
            raise ValueError(f"Invalid mode: {mode}")

        sharded_program_config = norm_config.get("sharded_program_config") if norm_config else None
        sharded_output_config = norm_config.get("sharded_output_config") if norm_config else None
        output_mem_config = norm_config.get("output_mem_config") if norm_config else None
        # Optional L1 placement for the distributed 3-op outputs (pre/gather/post); None -> DRAM default.
        distributed_out_mc = norm_config.get("distributed_output_mem_config") if norm_config else None

        # If input is sharded do sharded RMSNorm and optionally return sharded output
        program_config = sharded_program_config if in_sharded else None
        memory_config = sharded_output_config if out_sharded else None
        # PCIe P2P mesh: the input is replicated (full hidden dim), so the plain norm with the full weight is the
        # right one; the distributed variant all-gathers partial stats over the fabric, which does not exist there.
        distributed = self.is_distributed and self.is_distributed(mode) and getattr(self.tt_ccl, "pcie", None) is None
        weight = self.weight_distributed if distributed else self.weight

        if in_sharded:
            assert not distributed, "Distributed RMSNorm does not support sharded inputs"
        else:
            assert not out_sharded, "Non-sharded version of RMSNorm cannot output a sharded tensor"

        if (
            not distributed
            and not in_sharded
            and mode == Mode.PREFILL
            and getattr(self.tt_ccl, "pcie", None) is not None
            and os.environ.get("TT_PCIE_PREFILL_NORM_SHARDED", "1") == "1"
        ):
            y = self._pcie_prefill_norm(x, weight)
            if y is not None:
                if output_mem_config is not None:
                    y = ttnn.to_memory_config(y, output_mem_config)
                return y

        if distributed:
            x = self._distributed_rmsnorm(
                x,
                epsilon=self.eps,
                weight=weight,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                output_memory_config=distributed_out_mc,
            )
        else:
            x = ttnn.rms_norm(
                x,
                epsilon=self.eps,
                weight=weight,
                program_config=program_config,
                memory_config=memory_config,
                compute_kernel_config=self.compute_kernel_config_hifi2,
            )

        if in_sharded and not out_sharded:
            return ttnn.sharded_to_interleaved(x)
        else:
            if output_mem_config is not None:
                x = ttnn.to_memory_config(x, output_mem_config)
            return x

    def _distributed_rmsnorm(
        self,
        inp,
        epsilon=None,
        weight=None,
        program_config=None,
        memory_config=None,
        compute_kernel_config=None,
        output_memory_config=None,
    ):
        assert program_config is None, "Distributed RMSNorm does not support sharded inputs"
        assert memory_config is None, "Distributed RMSNorm does not support sharded outputs"
        assert self.tt_ccl is not None, "Distributed RMSNorm requires tt_ccl"

        # Interleaved output placement for the 3 ops; default DRAM (matches the prior hardcoded behavior).
        mc = output_memory_config if output_memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

        # Run distributed rmsnorm part 1
        tt_stats = ttnn.rms_norm_pre_all_gather(
            inp, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16, memory_config=mc
        )
        # AllGather stats
        tt_stats = ttnn.experimental.all_gather_async(
            tt_stats,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=1,
            topology=self.ccl_topology,
            memory_config=mc,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        # Run distributed rmsnorm part 2
        tt_out = ttnn.rms_norm_post_all_gather(
            inp,
            tt_stats,
            epsilon=epsilon,
            weight=weight,
            compute_kernel_config=compute_kernel_config,
            memory_config=mc,
        )
        tt_stats.deallocate(True)

        return tt_out
