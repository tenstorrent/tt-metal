# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple
import ttnn
import torch


class ParallelConfig(NamedTuple):
    mesh_shape: tuple[int, int]
    factor: int
    mesh_axis: int

    # TODO: factor as property


class DiTParallelConfig(NamedTuple):
    cfg_parallel: ParallelConfig
    tensor_parallel: ParallelConfig
    sequence_parallel: ParallelConfig
    ring_parallel: ParallelConfig
    ulysses_parallel: ParallelConfig
    topology: ttnn.Topology


def create_dit_parallel_config(
    mesh_shape: tuple[int, int],
    cfg_parallel: ParallelConfig,
    tensor_parallel: ParallelConfig,
    topology: ttnn.Topology,
    sequence_parallel: ParallelConfig,
    ring_parallel: ParallelConfig,
    ulysses_parallel: ParallelConfig,
) -> DiTParallelConfig:
    # validate cfg config
    assert cfg_parallel.factor in [1, 2]
    assert cfg_parallel.mesh_axis in [0, 1]
    assert cfg_parallel.mesh_shape[cfg_parallel.mesh_axis] == mesh_shape[cfg_parallel.mesh_axis] // cfg_parallel.factor
    assert cfg_parallel.mesh_shape[1 - cfg_parallel.mesh_axis] == mesh_shape[1 - cfg_parallel.mesh_axis]

    # validate sequence and tensor config
    assert tensor_parallel.mesh_axis == 1 - sequence_parallel.mesh_axis

    # validate sequence config
    assert sequence_parallel.mesh_axis in [0, 1]
    assert (
        sequence_parallel.mesh_shape[sequence_parallel.mesh_axis]
        == cfg_parallel.mesh_shape[sequence_parallel.mesh_axis] // sequence_parallel.factor
    )
    # assert (
    #     sequence_parallel.mesh_shape[1 - sequence_parallel.mesh_axis]
    #     == cfg_parallel.mesh_shape[1 - sequence_parallel.mesh_axis] // tensor_parallel.factor
    # )

    # validate tensor config
    assert tensor_parallel.mesh_axis in [0, 1]
    assert (
        tensor_parallel.mesh_shape[tensor_parallel.mesh_axis]
        == cfg_parallel.mesh_shape[tensor_parallel.mesh_axis] // tensor_parallel.factor
    )
    # assert (
    #     tensor_parallel.mesh_shape[1 - tensor_parallel.mesh_axis]
    #     == cfg_parallel.mesh_shape[1 - tensor_parallel.mesh_axis] // sequence_parallel.factor
    # )

    # validate ring config
    assert ring_parallel.mesh_axis in [0, 1]
    assert ring_parallel.mesh_shape == sequence_parallel.mesh_shape
    assert ring_parallel.mesh_axis == sequence_parallel.mesh_axis
    assert ring_parallel.factor == sequence_parallel.factor

    # validate ulysses config
    assert ulysses_parallel.mesh_axis in [0, 1]
    assert ulysses_parallel.mesh_shape == tensor_parallel.mesh_shape
    assert ulysses_parallel.mesh_axis == tensor_parallel.mesh_axis
    assert ulysses_parallel.factor == tensor_parallel.factor

    return DiTParallelConfig(
        cfg_parallel=cfg_parallel,
        tensor_parallel=tensor_parallel,
        sequence_parallel=sequence_parallel,
        ring_parallel=ring_parallel,
        ulysses_parallel=ulysses_parallel,
        topology=topology,
    )


class StableDiffusionParallelManager:
    def __init__(
        self,
        mesh_device,
        cfg_factor,
        sp_factor,
        tp_factor,
        rp_factor,
        up_factor,
        topology,
        cfg_axis=1,
        sp_axis=0,
        tp_axis=1,
        num_links=1,
    ):
        self.mesh_device = mesh_device
        self.num_links = num_links
        mesh_shape = tuple(mesh_device.shape)
        cfg_shape = list(mesh_shape)
        cfg_shape[cfg_axis] = cfg_shape[cfg_axis] // cfg_factor
        cfg_parallel = ParallelConfig(mesh_shape=tuple(cfg_shape), factor=cfg_factor, mesh_axis=cfg_axis)
        sp_mesh = list(cfg_parallel.mesh_shape)
        sp_mesh[sp_axis] = sp_mesh[sp_axis] // sp_factor
        sequence_parallel = ParallelConfig(
            mesh_shape=sp_mesh,
            factor=sp_factor,
            mesh_axis=sp_axis,
        )
        tp_mesh = list(cfg_parallel.mesh_shape)
        tp_mesh[tp_axis] = tp_mesh[tp_axis] // tp_factor
        tensor_parallel = ParallelConfig(
            mesh_shape=tp_mesh,
            factor=tp_factor,
            mesh_axis=tp_axis,
        )
        ring_parallel = ParallelConfig(
            mesh_shape=sp_mesh,
            factor=rp_factor,
            mesh_axis=sp_axis,
        )
        ulysses_parallel = ParallelConfig(
            mesh_shape=tp_mesh,
            factor=up_factor,
            mesh_axis=tp_axis,
        )
        self.dit_parallel_config = create_dit_parallel_config(
            mesh_shape=mesh_shape,
            cfg_parallel=cfg_parallel,
            sequence_parallel=sequence_parallel,
            tensor_parallel=tensor_parallel,
            ring_parallel=ring_parallel,
            ulysses_parallel=ulysses_parallel,
            topology=topology,
        )

        # Set up submeshes for CFG parallelism
        self._init_submeshes()

        # Setup semaphores
        self._init_subdevice()

        # SD35-specific semaphores
        semaphore_names = [("ping_pong", 4), ("ring_sdpa", 2), ("rs_ping_pong", 3 * 2 * num_links)]
        self._init_semaphores(semaphore_names)
        self.ping_pong_idx = 0
        self.rs_ping_pong_idx = 0

        self.persistent_buffers = [{} for _ in range(self.dit_parallel_config.cfg_parallel.factor)]

    def _init_submeshes(self):
        # Set up submeshes for CFG parallelism
        self.submesh_devices = (
            self.mesh_device.create_submeshes(ttnn.MeshShape(*self.dit_parallel_config.cfg_parallel.mesh_shape))
            if isinstance(self.mesh_device, ttnn.MeshDevice) and self.cfg_factor > 1
            else [self.mesh_device]
        )

    def _init_subdevice(self):
        compute_grid_size = self.mesh_device.compute_with_storage_grid_size()
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )

        _worker_sub_device = ttnn.SubDevice(
            [
                self.ccl_cores,
            ]
        )
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)

    def _init_semaphores(self, semaphore_names):
        # Initialize semaphores for each necessary CCL, None if a single semaphore else length of semaphore list
        cfg_semaphores = [{} for _ in range(self.cfg_factor)]
        for cfg_idx, cfg_sem in enumerate(cfg_semaphores):
            for sem_name, len_sem in semaphore_names:
                n_sems = 1 if len_sem is None else len_sem
                sems = [
                    ttnn.create_global_semaphore(self.submesh_devices[cfg_idx], self.ccl_cores, 0)
                    for _ in range(n_sems)
                ]
                if len_sem is None:
                    sems = sems[0]
                cfg_sem[sem_name] = sems

        self.cfg_semaphores = cfg_semaphores

    def maybe_init_persistent_buffers(self, KV_shape, spatial_shape, prompt_shape):
        """
        Create persistent buffers for RingJointAttention given expected KV shape

        KV_shape: [B, H, N, D] where H is fractured by `ulysses_parallel` factor
        """

        spatial_buffer_shape = spatial_shape[:]
        spatial_buffer_shape[3] *= self.dit_parallel_config.tensor_parallel.factor
        prompt_buffer_shape = prompt_shape[:]
        prompt_buffer_shape[3] *= self.dit_parallel_config.tensor_parallel.factor

        spatial_layernorm_buffer_shape = spatial_shape[:]
        spatial_layernorm_buffer_shape[3] = 64 * self.dit_parallel_config.tensor_parallel.factor
        prompt_layernorm_buffer_shape = prompt_shape[:]
        prompt_layernorm_buffer_shape[3] = 64 * self.dit_parallel_config.tensor_parallel.factor

        spatial_seq_gather_buffer_shape = spatial_shape[:]
        spatial_seq_gather_buffer_shape[2] *= self.dit_parallel_config.sequence_parallel.factor
        spatial_tensor_gather_buffer_shape = spatial_seq_gather_buffer_shape[:]
        spatial_tensor_gather_buffer_shape[3] *= self.dit_parallel_config.tensor_parallel.factor

        spatial_rs_intermediate_buffer_shape = [2] + spatial_buffer_shape
        spatial_rs_output_buffer_shape = spatial_shape[:]
        prompt_rs_intermediate_buffer_shape = [2] + prompt_buffer_shape
        prompt_rs_output_buffer_shape = prompt_shape[:]

        ping_pong_buffers = [
            "spatial_buffer",
            "prompt_buffer",
            "spatial_seq_gather_buffer",
            "spatial_tensor_gather_buffer",
            "spatial_layernorm_buffer",
            "prompt_layernorm_buffer",
            "spatial_rs_feed_forward_intermediate",
            "spatial_rs_feed_forward_output",
            "prompt_rs_feed_forward_intermediate",
            "prompt_rs_feed_forward_output",
        ]
        self._ping_pong_buffer_indices = [
            {buffer_name: 0 for buffer_name in ping_pong_buffers}
            for _ in range(self.dit_parallel_config.cfg_parallel.factor)
        ]

        if len(self.persistent_buffers[0]) == 0 or KV_shape != self.persistent_buffers[0]["K_gathered"].shape:
            print("Generating persistent buffers for RingJointAttention")
            for cfg_idx in range(self.dit_parallel_config.cfg_parallel.factor):
                sm = self.submesh_devices[cfg_idx]
                for buffer_name in ["K_gathered", "V_gathered"]:
                    self.persistent_buffers[cfg_idx][buffer_name] = ttnn.from_torch(
                        torch.empty(KV_shape),
                        device=sm,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16,
                        mesh_mapper=ttnn.ShardTensor2dMesh(sm, mesh_shape=tuple(sm.shape), dims=[None, None]),
                    )

        for cfg_idx in range(self.dit_parallel_config.cfg_parallel.factor):
            sm = self.submesh_devices[cfg_idx]
            for buffer_name, buffer_shape in zip(
                ping_pong_buffers,
                [
                    spatial_buffer_shape,
                    prompt_buffer_shape,
                    spatial_seq_gather_buffer_shape,
                    spatial_tensor_gather_buffer_shape,
                    spatial_layernorm_buffer_shape,
                    prompt_layernorm_buffer_shape,
                    spatial_rs_intermediate_buffer_shape,
                    spatial_rs_output_buffer_shape,
                    prompt_rs_intermediate_buffer_shape,
                    prompt_rs_output_buffer_shape,
                ],
            ):
                if (
                    buffer_name not in self.persistent_buffers[cfg_idx]
                    or buffer_shape != self.persistent_buffers[cfg_idx][buffer_name][0].shape
                ):
                    self.persistent_buffers[cfg_idx][buffer_name] = [
                        ttnn.from_torch(
                            torch.empty(buffer_shape),
                            device=sm,
                            layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16,
                            mesh_mapper=ttnn.ShardTensor2dMesh(sm, mesh_shape=tuple(sm.shape), dims=[None, None]),
                        )
                        for _ in range(2)
                    ]  # double buffer

    def get_ping_pong_semaphore(self, cfg_index):
        cur_idx = self.ping_pong_idx
        self.ping_pong_idx = 2 - self.ping_pong_idx
        return self.cfg_semaphores[cfg_index]["ping_pong"][cur_idx : cur_idx + 2]

    def get_rs_ping_pong_semaphore(self, cfg_index):
        cur_idx = self.rs_ping_pong_idx
        n_sems = 3 * self.num_links
        self.rs_ping_pong_idx = (cur_idx + 1) % 2
        return self.cfg_semaphores[cfg_index]["rs_ping_pong"][cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_ping_pong_buffer(self, cfg_index, buffer_name):
        cur_idx = self._ping_pong_buffer_indices[cfg_index][buffer_name]
        self._ping_pong_buffer_indices[cfg_index][buffer_name] = (
            1 - self._ping_pong_buffer_indices[cfg_index][buffer_name]
        )
        return self.persistent_buffers[cfg_index][buffer_name][cur_idx]

    def reset_global_semaphores(self):
        for cfg_idx in range(self.dit_parallel_config.cfg_parallel.factor):
            for sems in self.cfg_semaphores[cfg_idx].values():
                if isinstance(sems, list):
                    for sem in sems:
                        ttnn.reset_global_semaphore_value(sem, 0)
                else:
                    ttnn.reset_global_semaphore_value(sems, 0)

    @property
    def is_ring_parallel(self):
        return self.dit_parallel_config.ring_parallel.factor > 1

    @property
    def is_sequence_parallel(self):
        return self.dit_parallel_config.sequence_parallel.factor > 1

    @property
    def is_tensor_parallel(self):
        return self.dit_parallel_config.tensor_parallel.factor > 1

    @property
    def is_cfg_parallel(self):
        return self.dit_parallel_config.cfg_parallel.factor > 1

    @property
    def is_ulysses_parallel(self):
        return self.dit_parallel_config.ulysses_parallel.factor > 1

    @property
    def cfg_factor(self):
        return self.dit_parallel_config.cfg_parallel.factor
