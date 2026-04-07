# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from ..utils.tensor import bf16_tensor, local_device_to_torch


class CCLManager:
    """
    Manages parallelization of DiT model.

        - stores mesh device, num links, topology
        - caches ping pong buffers and semaphores
        - sets up one SubDevice spanning all compute cores
    """

    def __init__(
        self,
        mesh_device,
        num_links=1,
        topology=None,
    ):
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology

        # Cache for ping pong buffers: key = (shape_tuple, dim, mesh_axis), value = [buffer1, buffer2]
        self._ping_pong_buffer_cache = {}
        self._ping_pong_buffer_indices = {}

        # Setup semaphores
        self._init_subdevice()

        # Initialize semaphores for reduce scatter and all gather and neighbor pad
        self._init_semaphores()
        self.rs_ping_pong_idx = [0, 0]
        self.rs_ping_pong_idx_fused = [0, 0]
        self.ag_ping_pong_idx = [0, 0]
        self.np_ping_pong_idx = [0, 0]
        self.sr_ping_pong_idx = [0, 0]
        self.barrier_idx = [0, 0]

    def _init_subdevice(self):
        compute_grid_size = self.mesh_device.compute_with_storage_grid_size()
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )

        _worker_sub_device = ttnn.SubDevice([self.ccl_cores])
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)

        # Sub-devices for halo-parallel mode:
        # Sub-device 0: fabric cores (row 0, first N cols) — for NeighborPad fabric-only on CQ1
        # Sub-device 1: all remaining cores — for conv3d on CQ0
        #
        # NP uses num_links*2 cores per halo direction. For 2D (H+W) NP the total is
        # num_links*2 (H) + num_links*2 (W) = num_links*4. We allocate that many cores
        # in row 0 so the halo sub-device covers all NP cores regardless of direction count.
        num_fabric_cores = self.num_links * 4
        self.fabric_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_fabric_cores - 1, 0))}
        )
        self.conv3d_cores = self.ccl_cores.subtract(self.fabric_cores)
        self._fabric_sub_device = ttnn.SubDevice([self.fabric_cores])
        self._conv3d_sub_device = ttnn.SubDevice([self.conv3d_cores])
        self._halo_sub_device_mgr = None
        self._fabric_sd_id = None
        self._conv3d_sd_id = None
        self._pending_np_event = None  # CQ1 event: NP done, conv3d may start on CQ0
        self._pending_cq0_event = None  # CQ0 event: conv3d done (for future use)

    def _init_semaphores(self):
        # Initialize semaphores for reduce scatter ping pong - separate for each mesh axis
        rs_n_sems = 3 * 2  # 3 semaphores * 2 for ping pong
        self.rs_ping_pong_semaphores = {
            0: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(rs_n_sems)],
            1: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(rs_n_sems)],
        }

        # 3 * 2 for ping pong semaphores for fused reduce scatter
        self.rs_ping_pong_semaphores_fused = {
            0: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(rs_n_sems)],
            1: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(rs_n_sems)],
        }

        # Initialize semaphores for all gather ping pong - separate for each mesh axis
        ag_n_sems = 2 * 2  # 2 semaphores * 2 for ping pong (2 buffers)
        self.ag_ping_pong_semaphores = {
            0: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(ag_n_sems)],
            1: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(ag_n_sems)],
        }

        # Initialize neighbor pad semaphores
        np_n_sems = 1 * 2
        self.np_ping_pong_semaphores = {
            0: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(np_n_sems)],
            1: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(np_n_sems)],
        }

        # Initialize slice reshard semaphores
        sr_n_sems = 1 * 2
        self.sr_ping_pong_semaphores = {
            0: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(sr_n_sems)],
            1: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(sr_n_sems)],
        }

        # Initialize barrier semaphore
        barrier_n_sems = 1 * 2
        self.barrier_semaphores = {
            0: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(barrier_n_sems)],
            1: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(barrier_n_sems)],
        }

        # Progress semaphores for NeighborPad→Conv3d pipelining (one per axis, ping-pong)
        np_progress_n_sems = 2
        self.np_progress_semaphores = {
            0: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(np_progress_n_sems)],
            1: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(np_progress_n_sems)],
        }
        self._np_progress_sem_idx = {0: 0, 1: 0}

    def activate_halo_sub_devices(self):
        """
        Load a 2-sub-device manager for halo-parallel execution:
        - SD0 (fabric_sd_id): 4 fabric cores — for NP on CQ1
        - SD1 (conv3d_sd_id): 116 compute cores — for conv3d on CQ0

        Sub-devices are non-overlapping (required by TT-Metal).
        Regular ops (RMSNorm, add, etc.) must call deactivate_halo_sub_devices() first
        to restore the full-grid default manager before being dispatched.

        Returns (fabric_sd_id, conv3d_sd_id).
        """
        if self._halo_sub_device_mgr is None:
            # fabric_sub_device (4 cores) and conv3d_sub_device (116 cores) are
            # already non-overlapping (conv3d_cores = all_cores - fabric_cores).
            self._halo_sub_device_mgr = self.mesh_device.create_sub_device_manager(
                [self._fabric_sub_device, self._conv3d_sub_device], 0
            )
            self._fabric_sd_id = ttnn.SubDeviceId(0)  # 4 fabric cores for NP
            self._conv3d_sd_id = ttnn.SubDeviceId(1)  # 116 compute cores for conv3d
            self._pending_np_event = None
        self.mesh_device.load_sub_device_manager(self._halo_sub_device_mgr)
        return self._fabric_sd_id, self._conv3d_sd_id

    def deactivate_halo_sub_devices(self):
        """
        Return to the default full-grid sub-device manager for regular ops.
        Must be called after conv3d before any op that uses all cores.
        """
        self.mesh_device.clear_loaded_sub_device_manager()

    def get_rs_ping_pong_buffer(self, shape, dim, mesh_axis):
        """
        Get or create ping pong buffers for reduce scatter operations.
        Caches buffers based on shape, dim, and mesh_axis.

        Args:
            shape: Tensor shape tuple
            dim: Dimension for the operation
            mesh_axis: Mesh axis for parallelization

        Returns:
            Current ping pong buffer (alternates between two buffers)
        """
        # Create cache key from the parameters
        cache_key = (tuple(shape), dim, mesh_axis)

        # Create buffers if not cached
        if cache_key not in self._ping_pong_buffer_cache:
            # Synchronize devices to ensure all are ready to allocate and proceed
            ttnn.synchronize_device(self.mesh_device)
            # Create two buffers for ping pong
            buffers = []
            output_buffer_shape = list(shape)
            output_buffer_shape[dim] //= self.mesh_device.shape[mesh_axis]

            intermediate_buffer_shape = list(shape)
            intermediate_buffer_shape = [2] + intermediate_buffer_shape
            for _ in range(2):
                intermediate_buffer = bf16_tensor(torch.empty(intermediate_buffer_shape), device=self.mesh_device)
                output_buffer = bf16_tensor(torch.empty(output_buffer_shape), device=self.mesh_device)
                buffers.append([intermediate_buffer, output_buffer])

            self._ping_pong_buffer_cache[cache_key] = buffers
            self._ping_pong_buffer_indices[cache_key] = 0
            ttnn.synchronize_device(self.mesh_device)

        # Get current buffer and alternate index
        current_idx = self._ping_pong_buffer_indices[cache_key]
        self._ping_pong_buffer_indices[cache_key] = 1 - current_idx

        return self._ping_pong_buffer_cache[cache_key][current_idx]

    def get_ag_ping_pong_buffer(self, shape, dim, mesh_axis, dtype=ttnn.bfloat16):
        """
        Get or create ping pong buffers for all gather operations.
        Caches buffers based on shape, dim, and mesh_axis.

        Args:
            shape: Tensor shape tuple
            dim: Dimension for the operation
            mesh_axis: Mesh axis for parallelization

        Returns:
            Current ping pong buffer (alternates between two buffers)
        """
        # Create cache key from the parameters (use different namespace than rs)
        cache_key = ("ag", tuple(shape), dim, mesh_axis, dtype)

        # Create buffers if not cached
        if cache_key not in self._ping_pong_buffer_cache:
            # Synchronize devices to ensure all are ready to allocate and proceed
            ttnn.synchronize_device(self.mesh_device)
            # Create two buffers for ping pong
            buffers = []
            output_buffer_shape = list(shape)
            output_buffer_shape[dim] *= self.mesh_device.shape[mesh_axis]  # All gather increases size
            for _ in range(2):
                output_buffer = ttnn.from_torch(
                    torch.empty(output_buffer_shape),
                    layout=ttnn.TILE_LAYOUT,
                    dtype=dtype,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    device=self.mesh_device,
                )
                buffers.append(output_buffer)

            self._ping_pong_buffer_cache[cache_key] = buffers
            self._ping_pong_buffer_indices[cache_key] = 0
            ttnn.synchronize_device(self.mesh_device)

        # Get current buffer and alternate index
        current_idx = self._ping_pong_buffer_indices[cache_key]
        self._ping_pong_buffer_indices[cache_key] = 1 - current_idx

        return self._ping_pong_buffer_cache[cache_key][current_idx]

    def get_rs_ping_pong_semaphore(self, mesh_axis):
        """
        Get semaphores for reduce scatter ping pong operations.

        Args:
            mesh_axis: The mesh axis (0 or 1) to get semaphores for

        Returns:
            List of 3 semaphores for the current ping pong cycle
        """
        cur_idx = self.rs_ping_pong_idx[mesh_axis]
        n_sems = 3
        self.rs_ping_pong_idx[mesh_axis] = (cur_idx + 1) % 2
        return self.rs_ping_pong_semaphores[mesh_axis][cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_rs_ping_pong_semaphore_fused(self, mesh_axis):
        """
        Get semaphores for reduce scatter ping pong operations.

        Args:
            mesh_axis: The mesh axis (0 or 1) to get semaphores for

        Returns:
            List of 3 semaphores for the current ping pong cycle
        """
        cur_idx = self.rs_ping_pong_idx_fused[mesh_axis]
        n_sems = 3
        self.rs_ping_pong_idx_fused[mesh_axis] = (cur_idx + 1) % 2
        return self.rs_ping_pong_semaphores_fused[mesh_axis][cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_ag_ping_pong_semaphore(self, mesh_axis):
        """
        Get semaphores for all gather ping pong operations.

        Args:
            mesh_axis: The mesh axis (0 or 1) to get semaphores for

        Returns:
            List of 2 semaphores for the current ping pong cycle
        """
        cur_idx = self.ag_ping_pong_idx[mesh_axis]
        n_sems = 2
        self.ag_ping_pong_idx[mesh_axis] = (cur_idx + 1) % 2
        return self.ag_ping_pong_semaphores[mesh_axis][cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_np_ping_pong_semaphore(self, mesh_axis):
        """
        Get semaphores for neighbor pad operations.
        """
        cur_idx = self.np_ping_pong_idx[mesh_axis]
        n_sems = 1
        self.np_ping_pong_idx[mesh_axis] = (cur_idx + 1) % 2
        return self.np_ping_pong_semaphores[mesh_axis][cur_idx]

    def get_sr_ping_pong_semaphore(self, mesh_axis):
        """
        Get semaphores for slice reshard operations.
        """
        cur_idx = self.sr_ping_pong_idx[mesh_axis]
        n_sems = 1
        self.sr_ping_pong_idx[mesh_axis] = (cur_idx + 1) % 2
        return self.sr_ping_pong_semaphores[mesh_axis][cur_idx]

    def get_np_progress_semaphore(self, mesh_axis):
        """Get a ping-pong progress semaphore for NeighborPad→Conv3d pipelining."""
        cur_idx = self._np_progress_sem_idx[mesh_axis]
        self._np_progress_sem_idx[mesh_axis] = (cur_idx + 1) % 2
        return self.np_progress_semaphores[mesh_axis][cur_idx]

    def get_np_halo_buffer(self, input_shape, dim, padding, dtype=ttnn.bfloat16, dim2=None, padding2=0):
        """
        Get or create a ping-pong compact halo buffer for fabric-only NeighborPad.
        Handles H halo and optionally W halo.

        Layout: [H-top | H-bot | W-left | W-right] where H sections are
        outer_dim × padding_h × W_dev sticks, and W sections are
        outer_dim × padding_w × (H_dev + 2*padding_h) sticks (extended for corner fix).
        """
        import torch

        outer_dim_size = 1
        for d in range(dim):
            outer_dim_size *= input_shape[d]
        # H sticks per halo row = product of dims after dim (excluding last C dim)
        h_sticks = 1
        for d in range(dim + 1, len(input_shape) - 1):
            h_sticks *= input_shape[d]
        # W sticks per halo col = H_dev + 2*padding_h (extended to include H-padded rows for corner fix)
        w_sticks = (input_shape[dim] + 2 * padding) if dim2 is not None else 0

        h_halo_total = outer_dim_size * 2 * padding * h_sticks
        w_halo_total = outer_dim_size * 2 * padding2 * w_sticks if dim2 is not None else 0
        total_sticks = h_halo_total + w_halo_total
        # Each "stick" = C channels × 2 bytes (BF16). The compact buffer must have shape
        # [total_sticks, C] so that each 2D row = C elements = one stick page (= C×2 bytes),
        # matching the input tensor's aligned_page_size used as stick_size in the NP factory.
        C = input_shape[-1]  # channel dimension

        cache_key = ("np_halo", tuple(input_shape), dim, padding, dim2, padding2, dtype)
        if cache_key not in self._ping_pong_buffer_cache:
            bufs = []
            for _ in range(2):
                buf = ttnn.from_torch(
                    torch.zeros([total_sticks, C], dtype=torch.bfloat16),
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=dtype,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    device=self.mesh_device,
                )
                bufs.append(buf)
            self._ping_pong_buffer_cache[cache_key] = bufs
            self._ping_pong_buffer_indices[cache_key] = 0

        cur = self._ping_pong_buffer_indices[cache_key]
        self._ping_pong_buffer_indices[cache_key] = 1 - cur
        return self._ping_pong_buffer_cache[cache_key][cur]

    def dispatch_on_cq(self, cq_id: int, fn, *args, **kwargs):
        """Dispatch an op to a specific command queue for concurrent execution."""
        from ttnn.decorators import pop_current_command_queue_id_for_thread, push_current_command_queue_id_for_thread

        push_current_command_queue_id_for_thread(ttnn.QueueId(cq_id))
        try:
            result = fn(*args, **kwargs)
        finally:
            pop_current_command_queue_id_for_thread()
        return result

    def record_cq0_event(self):
        """Record completion of the current CQ0 (conv3d) work as an event."""
        if self._conv3d_sd_id is not None:
            self._pending_cq0_event = ttnn.record_event(
                self.mesh_device,
                cq_id=ttnn.QueueId(0),
                sub_device_ids=[self._conv3d_sd_id],
            )

    def wait_cq0_event_on_cq1(self):
        """On CQ1, wait for the last CQ0 event before dispatching NeighborPad."""
        if self._pending_cq0_event is not None:
            ttnn.wait_for_event(cq_id=ttnn.QueueId(1), mesh_event=self._pending_cq0_event)
            self._pending_cq0_event = None

    def neighbor_pad_halo_only(
        self,
        tensor: ttnn.Tensor,
        /,
        *,
        dims: list,
        pad_left: list,
        pad_right: list,
        padding_mode: str,
        axes: list,
        neighbor_sems: list,
        num_links: list,
        progress_semaphore=None,
        progress_t_batch_size: int = 0,
    ) -> ttnn.Tensor:
        """
        Fabric-only NeighborPad: only writes halo rows to a compact buffer (no interior copy).
        Returns the compact halo tensor. Conv3d reads interior from the original tensor.
        Handles both H and W halos (dims can have 1 or 2 elements).

        Dispatches NP on CQ0 targeting SD0 (fabric cores), same CQ as conv3d (SD1).
        Like all_gather_matmul: single CQ drives both CCL writers (SD0) and compute (SD1)
        concurrently. Dispatch fires go to SD0 then SD1; they run on different cores.
        In-kernel semaphore handles T-batch ordering. Deactivate then correctly waits for
        BOTH NP and conv3d completions before sending the reset go_signal to SD0.
        """
        # Load halo sub-device manager (SD0=fabric, SD1=compute).
        fabric_sd_id, _ = self.activate_halo_sub_devices()

        barrier_sem = self.get_barrier_semaphore(axes[0])
        dim2 = dims[1] if len(dims) > 1 else None
        padding2 = pad_left[1] if len(dims) > 1 else 0
        halo_buf = self.get_np_halo_buffer(
            tensor.shape, dims[0], pad_left[0], dtype=tensor.get_dtype(), dim2=dim2, padding2=padding2
        )

        # When progress semaphore is active, force num_links=1 so a SINGLE direction-0
        # fabric core handles all T outer_dims. This ensures local outer_dim == global
        # T-slice: the signal condition (outer_dim+1)%T_batch==0 fires at correct boundaries.
        # With num_links=2, two direction-0 cores split outer_dims; core 1's
        # outer_dim_offset_start_id is in sticks (not T-slices) making global T-batch
        # computation wrong — core 1 never fires signals, causing reader deadlock.
        effective_num_links = [1] * len(num_links) if progress_t_batch_size > 0 else num_links

        # Dispatch NeighborPad on CQ0 (same CQ as conv3d) targeting SD0 (fabric cores).
        # Single-CQ dispatch: NP (SD0) and conv3d (SD1) run concurrently on device.
        # deactivate_halo_sub_devices() naturally waits for both NP+conv3d before reset.
        result = self.dispatch_on_cq(
            0,
            ttnn.experimental.neighbor_pad_async,
            tensor,
            dims,
            pad_left,
            pad_right,
            padding_mode,
            axes,
            neighbor_sems,
            [barrier_sem],
            num_links=effective_num_links,
            topology=self.topology,
            persistent_output_buffer=halo_buf,
            progress_semaphore=progress_semaphore,
            progress_t_batch_size=progress_t_batch_size,
            fabric_only=True,
            sub_device_id=fabric_sd_id,
        )

        self._pending_np_event = None
        return result

    def neighbor_pad_conv3d_fused(
        self,
        tensor: ttnn.Tensor,
        weight: ttnn.Tensor,
        bias,
        *,
        dims: list,
        pad_left: list,
        pad_right: list,
        axes: list,
        neighbor_sems: list,
        num_links: list,
        conv_config,
        output_channels: int,
        kernel_size,
        stride,
        padding,
        dilation,
        padding_mode: str,
        dtype,
        compute_kernel_config=None,
    ) -> ttnn.Tensor:
        """
        Fused NeighborPad + Conv3d: ONE program dispatch, no sub-device manager.
        Equivalent to neighbor_pad_halo_only() + conv3d() but as a single C++ op.
        Requires conv_config.use_h_halo_buffer=True and input_progress_t_batch_size>0.
        """
        barrier_sem = self.get_barrier_semaphore(axes[0])
        dim2 = dims[1] if len(dims) > 1 else None
        padding2 = pad_left[1] if dim2 is not None else 0
        halo_buf = self.get_np_halo_buffer(
            tensor.shape, dims[0], pad_left[0], dtype=tensor.get_dtype(), dim2=dim2, padding2=padding2
        )

        # Update halo buffer fields in conv_config before dispatch
        H_dev = tensor.shape[2]
        W_dev = tensor.shape[3]
        T_dev = tensor.shape[1]
        ph = pad_left[0]
        pw = pad_left[1] if dim2 is not None and len(pad_left) > 1 else 0
        conv_config.h_halo_buffer_addr = halo_buf.buffer_address()
        conv_config.h_halo_outer_dim_size = T_dev
        conv_config.h_halo_H = H_dev + 2 * ph
        conv_config.h_halo_W = W_dev
        conv_config.h_halo_padding_h = ph
        conv_config.h_halo_padding_w = pw

        # Progress sem: GlobalSemaphore for conv3d readers, reset before dispatch.
        # W-writer cores signal this once each after completing W-halo work.
        progress_sem = self.get_np_progress_semaphore(axes[0])
        ttnn.reset_global_semaphore_value(progress_sem, 0)
        conv_config.input_progress_sem_addr = ttnn.get_global_semaphore_address(progress_sem)

        w_sem = neighbor_sems[1] if len(neighbor_sems) > 1 else neighbor_sems[0]
        np_pad2_left = pad_left[1] if dim2 is not None and len(pad_left) > 1 else 0
        np_pad2_right = pad_right[1] if dim2 is not None and len(pad_right) > 1 else 0
        # nanobind expects int for np_pad_dim2 and np_pad2_cluster_axis (not Optional).
        # Use 0 as sentinel when not in 2D mode — the factory checks is_2d via np_pad_dim2 != 0.
        np_pad_dim2_val = dim2 if dim2 is not None else 0
        np_pad2_cluster_axis_val = axes[1] if dim2 is not None and len(axes) > 1 else 0
        np_pad2_num_links = num_links[1] if dim2 is not None and len(num_links) > 1 else 1

        return ttnn.experimental.neighbor_pad_conv3d(
            tensor,
            weight,
            bias,
            halo_buf,
            np_padding_h=pad_left[0],
            np_padding_w=pw,
            np_cluster_axis=axes[0],
            np_num_links=num_links[0],
            np_topology=self.topology,
            h_neighbor_semaphore=neighbor_sems[0],
            barrier_semaphore=barrier_sem,
            w_neighbor_semaphore=w_sem,
            np_pad_dim2=np_pad_dim2_val,
            np_pad2_left=np_pad2_left,
            np_pad2_right=np_pad2_right,
            np_pad2_cluster_axis=np_pad2_cluster_axis_val,
            np_pad2_num_links=np_pad2_num_links,
            conv_config=conv_config,
            output_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode=padding_mode,
            dtype=dtype,
            compute_kernel_config=compute_kernel_config,
        )

    def get_barrier_semaphore(self, mesh_axis):
        """
        Get semaphore for barrier operations.
        """
        cur_idx = self.barrier_idx[mesh_axis]
        n_sems = 1
        self.barrier_idx[mesh_axis] = (cur_idx + 1) % 2
        return self.barrier_semaphores[mesh_axis][cur_idx]

    def get_np_ping_pong_buffer(self, input_shape, dims, pad_left, pad_right, dtype=ttnn.bfloat16):
        """
        Get or create ping pong buffers for neighbor pad operations.
        Caches buffers based on output shape and dtype.

        Args:
            input_shape: Input tensor shape
            dims: List of dimensions being padded
            pad_left: List of left padding amounts per dim
            pad_right: List of right padding amounts per dim
            dtype: Tensor dtype

        Returns:
            Current ping pong buffer (alternates between two buffers)
        """
        output_shape = list(input_shape)
        for i, dim in enumerate(dims):
            output_shape[dim] += pad_left[i] + pad_right[i]

        cache_key = ("np", tuple(output_shape), dtype)

        if cache_key not in self._ping_pong_buffer_cache:
            ttnn.synchronize_device(self.mesh_device)
            buffers = []
            for _ in range(2):
                output_buffer = ttnn.from_torch(
                    torch.zeros(output_shape),
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=dtype,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    device=self.mesh_device,
                )
                buffers.append(output_buffer)

            self._ping_pong_buffer_cache[cache_key] = buffers
            self._ping_pong_buffer_indices[cache_key] = 0
            ttnn.synchronize_device(self.mesh_device)

        current_idx = self._ping_pong_buffer_indices[cache_key]
        self._ping_pong_buffer_indices[cache_key] = 1 - current_idx

        return self._ping_pong_buffer_cache[cache_key][current_idx]

    def neighbor_pad_persistent_buffer(
        self,
        tensor: ttnn.Tensor,
        /,
        *,
        dims: list,
        pad_left: list,
        pad_right: list,
        padding_mode: str,
        axes: list,
        neighbor_sems: list,
        num_links: list,
        progress_semaphore=None,
        progress_t_batch_size: int = 0,
    ) -> ttnn.Tensor:
        """
        Helper function to neighbor-pad a tensor with a persistent output buffer.
        """
        return self.neighbor_pad(
            tensor,
            dims=dims,
            pad_left=pad_left,
            pad_right=pad_right,
            padding_mode=padding_mode,
            axes=axes,
            neighbor_sems=neighbor_sems,
            num_links=num_links,
            use_persistent_buffer=True,
            progress_semaphore=progress_semaphore,
            progress_t_batch_size=progress_t_batch_size,
        )

    def neighbor_pad(
        self,
        tensor: ttnn.Tensor,
        /,
        *,
        dims: list,
        pad_left: list,
        pad_right: list,
        padding_mode: str,
        axes: list,
        neighbor_sems: list,
        num_links: list,
        use_persistent_buffer: bool = False,
        progress_semaphore=None,
        progress_t_batch_size: int = 0,
    ) -> ttnn.Tensor:
        barrier_sem = self.get_barrier_semaphore(axes[0])

        persistent_buf = None
        if use_persistent_buffer:
            persistent_buf = self.get_np_ping_pong_buffer(
                tensor.shape, dims, pad_left, pad_right, dtype=tensor.get_dtype()
            )

        return ttnn.experimental.neighbor_pad_async(
            tensor,
            dims,
            pad_left,
            pad_right,
            padding_mode,
            axes,
            neighbor_sems,
            [barrier_sem],
            num_links=num_links,
            topology=self.topology,
            persistent_output_buffer=persistent_buf,
            progress_semaphore=progress_semaphore,
            progress_t_batch_size=progress_t_batch_size,
        )

    def reset_global_semaphores(self):
        """Reset all global semaphores to 0"""
        for axis in [0, 1]:
            for sem in self.np_ping_pong_semaphores[axis]:
                ttnn.reset_global_semaphore_value(sem, 0)
            for sem in self.sr_ping_pong_semaphores[axis]:
                ttnn.reset_global_semaphore_value(sem, 0)
            for sem in self.rs_ping_pong_semaphores[axis]:
                ttnn.reset_global_semaphore_value(sem, 0)
            for sem in self.ag_ping_pong_semaphores[axis]:
                ttnn.reset_global_semaphore_value(sem, 0)
            for sem in self.np_progress_semaphores[axis]:
                ttnn.reset_global_semaphore_value(sem, 0)

    def all_gather_persistent_buffer(
        self, tensor: ttnn.Tensor, /, *, dim: int, mesh_axis: int | None, use_hyperparams: bool = False
    ) -> ttnn.Tensor:
        """
        Helper function to all-gather a tensor with a persistent output buffer.
        """
        return self.all_gather(
            tensor,
            dim=dim,
            mesh_axis=mesh_axis,
            use_hyperparams=use_hyperparams,
            use_persistent_buffer=True,
        )

    def all_gather(
        self,
        tensor: ttnn.Tensor,
        /,
        *,
        dim: int,
        mesh_axis: int | None,
        use_hyperparams: bool,
        use_persistent_buffer: bool = False,
    ) -> ttnn.Tensor:
        if mesh_axis is None or self.mesh_device.shape[mesh_axis] == 1:
            return tensor

        rank = len(tensor.shape)
        if dim < 0:
            dim += rank

        # all_gather_async currently supports tensors of rank 4 only
        if rank < 4:
            shape = [1] * (4 - rank) + list(tensor.shape)
            tensor = ttnn.reshape(tensor, shape)
            dim += 4 - rank

        params = self.get_ag_hyperparams(tensor.shape) if use_hyperparams else {}

        tensor = ttnn.experimental.all_gather_async(
            tensor,
            persistent_output_buffer=(
                self.get_ag_ping_pong_buffer(tensor.shape, dim, mesh_axis, dtype=tensor.get_dtype())
                if use_persistent_buffer
                else None
            ),
            barrier_semaphore=self.get_barrier_semaphore(mesh_axis) if not use_persistent_buffer else None,
            dim=dim,
            multi_device_global_semaphore=self.get_ag_ping_pong_semaphore(mesh_axis),
            num_links=self.num_links,
            topology=self.topology,
            cluster_axis=mesh_axis,
            **params,
        )

        if rank < 4:
            shape = list(tensor.shape)[4 - rank :]
            tensor = ttnn.reshape(tensor, shape)

        return tensor

    def reduce_scatter_persistent_buffer(
        self, tensor: ttnn.Tensor, /, *, dim: int, mesh_axis: int | None
    ) -> ttnn.Tensor:
        self.reduce_scatter(tensor, dim=dim, mesh_axis=mesh_axis, use_persistent_buffer=True)

    def reduce_scatter(
        self,
        tensor: ttnn.Tensor,
        /,
        *,
        dim: int,
        mesh_axis: int | None,
        use_persistent_buffer: bool = False,
    ) -> ttnn.Tensor:
        if mesh_axis is None or self.mesh_device.shape[mesh_axis] == 1:
            return tensor

        rank = len(tensor.shape)
        if dim < 0:
            dim += rank

        if rank < 4:
            shape = [1] * (4 - rank) + list(tensor.shape)
            tensor = ttnn.reshape(tensor, shape)
            dim += 4 - rank

        tensor = ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            persistent_output_buffers=(
                self.get_rs_ping_pong_buffer(tensor.shape, dim, mesh_axis) if use_persistent_buffer else None
            ),
            barrier_semaphore=self.get_barrier_semaphore(mesh_axis) if not use_persistent_buffer else None,
            dim=dim,
            multi_device_global_semaphore=self.get_rs_ping_pong_semaphore(mesh_axis),
            num_links=self.num_links,
            memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
            topology=self.topology,
            cluster_axis=mesh_axis,
            **self.get_rs_hyperparams(tensor.shape),
        )

        if rank < 4:
            shape = list(tensor.shape)[4 - rank :]
            tensor = ttnn.reshape(tensor, shape)

        return tensor

    def get_ag_hyperparams(self, shape):
        if shape[2] > 512:
            return {
                "chunks_per_sync": 16,
                "num_workers_per_link": 3,
                "num_buffers_per_channel": 2,
            }
        else:
            return {
                "chunks_per_sync": 10,
                "num_workers_per_link": 2,
                "num_buffers_per_channel": 2,
            }

    def get_rs_hyperparams(self, shape):
        return {
            "chunks_per_sync": 2,
            "num_workers_per_link": 2,
            "num_buffers_per_channel": 2,
        }

    # TODO: Merge with utils.tensor.to_torch
    def device_to_host(
        self, tensor: ttnn.Tensor, mesh_dims: list[int], use_persistent_buffer: bool = True
    ) -> torch.Tensor:
        """Move a ttnn device tensor to a torch host tensor.
        Args:
            tensor: The ttnn tensor to move to host
            mesh_dims: The dimension to gather per mesh axis. use None to skip gathering for that mesh axis. e.g [None,2] will gather along the second dimension for the mesh axis 1.
            use_persistent_buffer: Whether to use a persistent buffer for the all gather operation.
        Returns:
            The torch host tensor
        """
        device_tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)  # Workaround for bug in Row Major layout
        for mesh_axis, mesh_dim in enumerate(mesh_dims):
            if mesh_dim is not None:
                device_tensor = self.all_gather(
                    device_tensor,
                    dim=mesh_dim,
                    mesh_axis=mesh_axis,
                    use_hyperparams=True,
                    use_persistent_buffer=use_persistent_buffer,
                )
        return local_device_to_torch(device_tensor)
