# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_distributed_rmsnorm, tt_sharded_distributed_rmsnorm
from models.tt_transformers.tt.common import Mode


class DistributedNorm(LightweightModule):
    def __init__(self, norm, args, tt_ccl, prefetcher=None, TG=False, ag_config_key=None, enable_all_gather=True):
        self.norm = norm
        self.args = args
        self.tt_ccl = tt_ccl
        self.prefetcher = prefetcher
        self.ag_config_key = ag_config_key

        # Flag to control whether all_gather is performed after distributed norm (can be disabled when output should remain sharded)
        self.enable_all_gather = enable_all_gather

        self.use_higgs_fused_rms_all_gather = (
            getattr(args, "higgs_fused_rms_all_gather", False) and not TG and args.is_multichip
        )
        self.higgs_fused_rms_stats = None
        if self.use_higgs_fused_rms_all_gather:
            cluster_axis = getattr(args, "higgs_norm_all_gather_cluster_axis", 1)
            if cluster_axis not in (0, 1):
                raise ValueError(f"Fused RMS/all-gather requires cluster axis 0 or 1, got {cluster_axis}")
            mesh_shape = list(args.mesh_device.shape)
            num_devices = mesh_shape[cluster_axis]
            if num_devices <= 1:
                self.use_higgs_fused_rms_all_gather = False
            else:
                stats_mem_cfg = ttnn.create_sharded_memory_config(
                    shape=(32, 32),
                    core_grid=ttnn.CoreGrid(y=1, x=1),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                mapper_dims = (3, None) if cluster_axis == 0 else (None, 3)
                self.higgs_fused_rms_stats = ttnn.from_torch(
                    torch.zeros((1, 1, 32, 32 * num_devices), dtype=torch.bfloat16),
                    device=args.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=stats_mem_cfg,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        mesh_device=args.mesh_device,
                        dims=mapper_dims,
                        mesh_shape=mesh_shape,
                    ),
                )
        if TG:
            core_grid_ln = (
                min(4, args.dim // 4 // 32 // 8),
                8,
            )  # dividing by 4 and 8 for num_cols and num_rows of mesh, and 32 for tile size
            num_cores_ln = core_grid_ln[0] * core_grid_ln[1]
            hidden_size_per_device_distributed_ln = args.dim // 4
            self.gather_in_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(1, 1, 32, hidden_size_per_device_distributed_ln),
                core_grid=ttnn.CoreGrid(y=core_grid_ln[0], x=core_grid_ln[1]),
                strategy=ttnn.ShardStrategy.WIDTH,
            )
            self.ln_prg_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(core_grid_ln[1], core_grid_ln[0]),
                subblock_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
                block_h=1,
                block_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
                inplace=False,
            )
            self.ln_sharded_stats_memcfg = ttnn.create_sharded_memory_config(
                shape=[1, 1, 32, 32 * 4],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
            )
            self.ln_cfg = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
        self.TG = TG

    @staticmethod
    def _higgs_fused_rms_program_config(input_mem_config):
        shard_spec = input_mem_config.shard_spec
        grid_size = shard_spec.grid.bounding_box().grid_size()
        block_w = shard_spec.shape[1] // ttnn.TILE_SIZE
        subblock_w = min(4, block_w)
        while subblock_w > 1 and block_w % subblock_w != 0:
            subblock_w -= 1
        return ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(grid_size.x, grid_size.y),
            subblock_w=subblock_w,
            block_h=shard_spec.shape[0] // ttnn.TILE_SIZE,
            block_w=block_w,
            inplace=False,
        )

    def _higgs_fused_rms_semaphore(self, cluster_axis):
        semaphore = self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis)
        return semaphore[0] if isinstance(semaphore, list) else semaphore

    def forward(self, x, mode: Mode, norm_config=None):
        """Apply a norm, possibly gathering inputs if required."""

        sharded_output_config = norm_config.get("sharded_output_config") if norm_config else None

        if self.TG:
            if mode == Mode.DECODE:
                return tt_sharded_distributed_rmsnorm(
                    x,
                    epsilon=self.norm.eps,
                    gamma=self.norm.weight_distributed,
                    mesh_device=self.args.mesh_device,
                    tt_ccl=self.tt_ccl,
                    ln_sharded_input_memcfg=self.gather_in_mem_cfg,
                    ln_sharded_progcfg=self.ln_prg_cfg,
                    ln_sharded_stats_memcfg=self.ln_sharded_stats_memcfg,
                )
            else:
                return tt_distributed_rmsnorm(
                    x,
                    epsilon=self.norm.eps,
                    gamma=self.norm.weight_distributed,
                    mesh_device=self.args.mesh_device,
                    tt_ccl=self.tt_ccl,
                    compute_kernel_config=self.ln_cfg,
                )

        input_mem_cfg = sharded_output_config if mode == Mode.DECODE else ttnn.DRAM_MEMORY_CONFIG

        # Distributed norm already performs a gather
        if self.args.is_multichip and not self.args.is_distributed_norm(mode):
            cluster_axis = getattr(self.args, "higgs_norm_all_gather_cluster_axis", None)
            if self.use_higgs_fused_rms_all_gather and mode == Mode.DECODE and cluster_axis is not None:
                compute_kernel_config = None
                if getattr(self.args, "higgs_fused_rms_lofi", False):
                    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                        math_approx_mode=True,
                        fp32_dest_acc_en=False,
                        packer_l1_acc=False,
                    )
                return ttnn.fused_rms_minimal(
                    x,
                    self._higgs_fused_rms_program_config(x.memory_config()),
                    cluster_axis,
                    self.args.mesh_device,
                    self._higgs_fused_rms_semaphore(cluster_axis),
                    topology=self.args.ccl_topology(),
                    num_links=(
                        self.args.model_config[self.ag_config_key]["num_links"]
                        if self.ag_config_key
                        else self.tt_ccl.get_num_links(cluster_axis)
                    ),
                    memory_config=sharded_output_config,
                    epsilon=self.norm.eps,
                    weight=self.norm.weight_distributed,
                    stats=self.higgs_fused_rms_stats,
                    dtype=ttnn.bfloat8_b,
                    compute_kernel_config=compute_kernel_config,
                    use_noc1_only=False,
                    subdevice_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
                )
            all_gather_kwargs = {}
            if cluster_axis is not None:
                all_gather_kwargs["cluster_axis"] = cluster_axis
            use_decode_ag_config = self.ag_config_key and (
                mode == Mode.DECODE if cluster_axis is not None else mode == "decode"
            )
            ag_semaphore = (
                self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis)
                if cluster_axis is not None
                else self.tt_ccl.get_and_cycle_ag_semaphore_handles()
            )
            barrier_semaphore = (
                self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis)
                if cluster_axis is not None
                else self.tt_ccl.get_and_cycle_barrier_semaphore_handle()
            )
            default_num_links = self.tt_ccl.get_num_links(cluster_axis if cluster_axis is not None else 1)
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=ag_semaphore,
                num_links=(
                    self.args.model_config[self.ag_config_key]["num_links"]
                    if use_decode_ag_config
                    else default_num_links
                ),
                topology=self.args.ccl_topology(),
                memory_config=input_mem_cfg,
                barrier_semaphore=barrier_semaphore,
                chunks_per_sync=(
                    self.args.model_config[self.ag_config_key]["chunks_per_sync"] if use_decode_ag_config else 10
                ),
                num_workers_per_link=(
                    self.args.model_config[self.ag_config_key]["num_workers_per_link"] if use_decode_ag_config else 2
                ),
                num_buffers_per_channel=2,
                subdevice_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
                **all_gather_kwargs,
            )
        else:
            x = ttnn.to_memory_config(x, input_mem_cfg)

        x = self.norm(
            x, mode=mode, in_sharded=(mode == Mode.DECODE), out_sharded=(mode == Mode.DECODE), norm_config=norm_config
        )

        # Distributed norm requires a gather
        if self.args.is_distributed_norm(mode) and self.enable_all_gather:
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=self.tt_ccl.get_num_links(1),
                topology=self.args.ccl_topology(),
                memory_config=x.memory_config(),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

        return x
