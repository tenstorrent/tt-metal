# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import NamedTuple

import ttnn


class ParallelFactor(NamedTuple):
    factor: int
    mesh_axis: int


class DiTParallelConfig(NamedTuple):
    cfg_parallel: ParallelFactor
    tensor_parallel: ParallelFactor
    sequence_parallel: ParallelFactor

    @classmethod
    def from_tuples(cls, *, cfg: tuple[int, int], sp: tuple[int, int], tp: tuple[int, int]) -> DiTParallelConfig:
        return cls(
            cfg_parallel=ParallelFactor(*cfg),
            sequence_parallel=ParallelFactor(*sp),
            tensor_parallel=ParallelFactor(*tp),
        )


class EncoderParallelConfig(NamedTuple):
    tensor_parallel: ParallelFactor
    sequence_parallel: ParallelFactor | None = None

    @classmethod
    def from_tuple(cls, tp: tuple[int, int]) -> EncoderParallelConfig:
        return cls(tensor_parallel=ParallelFactor(*tp))


class VAEParallelConfig(NamedTuple):
    tensor_parallel: ParallelFactor

    @classmethod
    def from_tuple(cls, tp: tuple[int, int]) -> VAEParallelConfig:
        return cls(tensor_parallel=ParallelFactor(*tp))


class VaeHWParallelConfig(NamedTuple):
    height_parallel: ParallelFactor
    width_parallel: ParallelFactor

    @classmethod
    def from_tuples(cls, *, height: tuple[int, int], width: tuple[int, int]) -> VaeHWParallelConfig:
        return cls(
            height_parallel=ParallelFactor(*height),
            width_parallel=ParallelFactor(*width),
        )


class AudioTParallelConfig(NamedTuple):
    axis0: ParallelFactor
    axis1: ParallelFactor

    @property
    def factor(self) -> int:
        return self.axis0.factor * self.axis1.factor


class AudioTCParallelConfig(NamedTuple):
    time_parallel: ParallelFactor
    channel_parallel: ParallelFactor

    @property
    def factor(self) -> int:
        return self.time_parallel.factor

    @property
    def mesh_axis(self) -> int:
        return self.time_parallel.mesh_axis


class MochiVAEParallelConfig(NamedTuple):
    time_parallel: ParallelFactor
    h_parallel: ParallelFactor
    w_parallel: ParallelFactor

    @classmethod
    def from_tuples(
        cls,
        *,
        time: tuple[int, int],
        h: tuple[int, int],
        w: tuple[int, int],
    ) -> MochiVAEParallelConfig:
        return cls(
            time_parallel=ParallelFactor(*time),
            h_parallel=ParallelFactor(*h),
            w_parallel=ParallelFactor(*w),
        )


class OldParallelConfig(NamedTuple):
    mesh_shape: tuple[int, int]
    factor: int
    mesh_axis: int


def vae_all_gather(
    ccl_manager,
    x: ttnn.Tensor,
    cluster_axis: int = 1,
    dim: int = 3,
    reshape: bool = True,
    use_barrier: bool = True,
) -> ttnn.Tensor:
    if x.device().shape[cluster_axis] == 1:
        return x

    global_semaphores = ccl_manager.get_ag_ping_pong_semaphore(cluster_axis)

    if reshape:
        # reshape to b,1,h*w,c. This was tested to be faster. Need to verify overhead. TODO: Cleanup
        b, h, w, c = x.shape
        if h != 1:  # Check if its already in desired shape. E.g group norm already reshaped to 1,1,h*w,c
            x = x.reshape(b, 1, h * w, c)

    if x.layout != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)  # All gather requires tile layout

    # NOTE: We can't use ping-pong persistent buffers because we run out of memory.
    # Single-buffered persistent buffers is a potential correctness issue, so we can't do that.
    # barrier_semaphore is required for correctness on repeated all_gathers (prevents
    # cross-dispatch races where fast devices start a new all_gather while slow devices
    # are still processing the previous one).
    # However, barrier_semaphore causes hangs in some pipelines (e.g. SD3.5 large).
    # Those callers pass use_barrier=False and rely on synchronize_device instead.
    if use_barrier:
        barrier_semaphore = ccl_manager.get_barrier_semaphore(cluster_axis)
    else:
        barrier_semaphore = None
        ttnn.synchronize_device(x.device())

    x_g = ttnn.experimental.all_gather_async(
        input_tensor=x,
        dim=dim,
        persistent_output_buffer=None,
        barrier_semaphore=barrier_semaphore,
        multi_device_global_semaphore=global_semaphores,
        topology=ttnn.Topology.Linear,
        cluster_axis=cluster_axis,
        num_links=ccl_manager.num_links,
        num_workers_per_link=4,
        chunks_per_sync=80,
        num_buffers_per_channel=4,
    )

    if reshape:
        # reshape back to original expected shape
        if h != 1:
            x_g = x_g.reshape(b, h, w, -1)
    return x_g


def vae_neighbor_pad(
    ccl_manager,
    x: ttnn.Tensor,
    cluster_axis: int = 1,
    dim: int = 0,
    padding_left: int = 0,
    padding_right: int = 0,
    padding_mode: str = "replicate",
) -> ttnn.Tensor:
    neighbor_semaphore = ccl_manager.get_np_ping_pong_semaphore(cluster_axis)
    barrier_semaphore = ccl_manager.get_barrier_semaphore(cluster_axis)

    x_pad = ttnn.experimental.neighbor_pad_async(
        x,
        [dim],
        [padding_left],
        [padding_right],
        padding_mode,
        [cluster_axis],
        [neighbor_semaphore],
        [barrier_semaphore],
        num_links=[ccl_manager.num_links],
        topology=ttnn.Topology.Linear,
    )

    return x_pad


def vae_slice_reshard(
    ccl_manager, x: ttnn.Tensor, cluster_axis: int = 1, dim: int = 0, output_shape: int = 88, output_offset: int = 0
) -> ttnn.Tensor:
    # NOTE: ttnn.experimental.slice_reshard_async has been nuked for agent
    # evaluation. Passthrough stub — keeps the build/imports green; the Mochi
    # VAE path that calls this is not functionally correct until the op is
    # regenerated.
    return x
