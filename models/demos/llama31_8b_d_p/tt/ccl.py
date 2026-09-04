# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`CCLManager` — every persistent CCL resource for Llama-3.1-8B prefill, allocated **once**.

**HF anchor:** none — this file holds no model math. It is the resource half of the pair
`bringup_log/04_CCL_PLAN.md` §2 describes: `MeshConfig` (`tt/config.py`) owns the parallelism
decision and the collective wrappers, this class owns the semaphores, the CCL core range and the
ring-gather scratch. One instance serves all 32 layers and every chunk of a request.

**Template.** `models/demos/gpt_oss_d_p/tt/ccl.py:17`, itself mirroring
`models/demos/minimax_m3/tt/ccl.py:9`. Three properties are load-bearing and must not be
"improved":

1. **The CCL core range derives from the real compute grid**
   (`models/demos/gpt_oss_d_p/tt/ccl.py:44`). On this Blackhole Galaxy that grid is **(12, 10)**,
   not 8x8, so the ring-attention offset below is `x = 11`. Hard-coding 8x8 here breaks the ring
   SDPA's grid-offset assert. This is the **opposite** of the SDPA *program* grid, which stays a
   pinned 8x8 (`BRINGUP_RECIPE.md:1409-1417`) — the two grids look alike and must not be unified.
2. **Semaphores are allocated once**, never per layer or per chunk
   (`bringup_log/04_CCL_PLAN.md` §3: 6 RS + 4 AG + 2 barrier + 2 ring-attention = **14**).
3. **Handing one out cycles a ping-pong index**, so back-to-back collectives never reuse a
   semaphore that may still be in flight. This is the single most common source of
   nondeterministic multi-device PCC failures (`BRINGUP_RECIPE.md:1135-1137`).

**Deletions for Llama:** none of the semaphore or scratch state — there is no `ep_axis` and no
MoE-specific state in this file, and the ring-gather scratch is used by P8's SP path. Four dead
pieces of the template are dropped (`DEC-029`): the unused `_ping_pong_buffer_cache` /
`_ping_pong_buffer_indices` dicts, the `_worker_sub_device` local that is constructed and
discarded, and the `ccl_sub_device_id` attribute no caller reads.

**Barrier depth is 2 and is deliberately not deepened** (`DEC-026`): RS takes `barrier[0]`, the AG
that follows takes `barrier[1]`, the next RS takes `barrier[0]` again — a one-op gap, 64 reuses per
32-layer forward under residual scheme A. `G-RACE` is the measurement; deepening 2 -> 4 is its
documented first move if it fails, and pre-emptively changing the count would blind the gate.
`reset_global_semaphores` likewise keeps the template's behaviour of **not** resetting the barrier
or ring-attention sets.
"""

import torch

import ttnn


class CCLManager:
    """Persistent CCL state for one model. Allocate once; hand out with a ping-pong index."""

    # Semaphore inventory, asserted by `G-SEMAPHORE`. The failure this exists to catch is any of
    # these becoming `n_layers x` the constant (`bringup_log/04_CCL_PLAN.md` §3).
    RS_SEMAPHORES_PER_CALL = 3
    AG_SEMAPHORES_PER_CALL = 2
    PING_PONG_DEPTH = 2
    RING_ATTENTION_SEMAPHORES = 2

    def __init__(self, mesh_device, num_links, topology=ttnn.Topology.Ring):
        """
        Args:
            mesh_device: the open mesh (or submesh) these resources belong to.
            num_links: fabric links per collective; from
                `models/demos/gpt_oss_d_p/utils/general_utils.py:27` `get_default_num_links`
                (`DEC-013`). Note a single-row mesh gets 1 link regardless of arch (`:33`), so a
                `(1, N)` gate never exercises the deployment link count.
            topology: `ttnn.Topology.Ring` by default, and it must match the fabric config the
                harness set — a Ring collective on a plain `FABRIC_1D` fabric **hangs** rather than
                erroring, and a hang on this box poisons it until `tt-smi -r`. `DEC-027` couples the
                two behind one variable, `PREFILL_TOPOLOGY`.
        """
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology

        # Persistent ring-gather scratch for the ring SDPA (P8's SP path), allocated once and
        # reused across every layer/chunk (key -> tensor). See `get_ring_gather_buffer`.
        self._ring_gather_buffers = {}

        self._init_ccl_cores()
        self._init_semaphores()

        self.rs_ping_pong_idx = 0
        self.ag_ping_pong_idx = 0
        self.barrier_idx = 0

    def _init_ccl_cores(self):
        """Derive the CCL core range from the REAL compute grid, and the ring-attention offset from it.

        Blackhole's grid is wider than 8x8 — (12, 10) on this box — and both the ring-attention CCL
        offset and the ring SDPA's own grid must be consistent with the op's
        `ccl_core_grid_offset.x >= sdpa_grid.x` assert
        (`ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp:421`).
        The offset is `grid.x - 1 = 11`, and the SDPA *program* grid is pinned at 8 so `11 >= 8`
        holds; deriving that program grid from this one would give `11 >= 12` and fail — but only
        at SP > 1, i.e. only in P8. Mirrors `models/demos/gpt_oss_d_p/tt/ccl.py:44`, `:61`.
        """
        compute_grid_size = self.mesh_device.compute_with_storage_grid_size()
        self.compute_grid_size = compute_grid_size
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        # Ring-attention CCL workers live in the LAST compute column; the ring SDPA's compute uses
        # the remaining columns, because the op requires CCL and SDPA cores to be non-overlapping.
        self.ring_attention_ccl_core_grid_offset = (compute_grid_size.x - 1, 0)

    def _init_semaphores(self):
        """Allocate all 14 global semaphores. Called exactly once, from `__init__`."""
        rs_n_sems = self.RS_SEMAPHORES_PER_CALL * self.PING_PONG_DEPTH  # 3 * 2 = 6
        self.rs_ping_pong_semaphores = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(rs_n_sems)
        ]

        ag_n_sems = self.AG_SEMAPHORES_PER_CALL * self.PING_PONG_DEPTH  # 2 * 2 = 4
        self.ag_ping_pong_semaphores = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(ag_n_sems)
        ]

        # One barrier semaphore per collective, 2 deep — the thin one (`DEC-026`).
        self.barrier_semaphore = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(self.PING_PONG_DEPTH)
        ]

        # A forward/backward PAIR for the ring SDPA (P8's SP path), not a ping-pong ring.
        self.ring_attention_ccl_semaphore_handles = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0)
            for _ in range(self.RING_ATTENTION_SEMAPHORES)
        ]

    def get_rs_ping_pong_semaphore(self):
        """The next reduce-scatter semaphore triple (3 per call), cycling a 2-deep ping-pong."""
        cur_idx = self.rs_ping_pong_idx
        n_sems = self.RS_SEMAPHORES_PER_CALL
        self.rs_ping_pong_idx = (cur_idx + 1) % self.PING_PONG_DEPTH
        return self.rs_ping_pong_semaphores[cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_ag_ping_pong_semaphore(self):
        """The next all-gather semaphore pair (2 per call), cycling a 2-deep ping-pong."""
        cur_idx = self.ag_ping_pong_idx
        n_sems = self.AG_SEMAPHORES_PER_CALL
        self.ag_ping_pong_idx = (cur_idx + 1) % self.PING_PONG_DEPTH
        return self.ag_ping_pong_semaphores[cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_barrier_semaphore(self):
        """The next barrier semaphore (1 per collective), cycling a 2-deep ping-pong."""
        cur_idx = self.barrier_idx
        self.barrier_idx = (cur_idx + 1) % self.PING_PONG_DEPTH
        return self.barrier_semaphore[cur_idx]

    def get_ring_gather_buffer(self, key, n_kv, seq, head_dim, dtype):
        """Persistent ring-gather scratch for the ring SDPA — allocated ONCE, reused every layer/chunk.

        Replaces a per-call `from_torch(zeros)` that churns host and DRAM on every dense attention.
        The op treats it as pure scratch: it fills the gathered region and masks the invalid tail,
        so reuse without re-zeroing is safe. `key` separates buffers that are live simultaneously
        (`"k"` vs `"v"` in one op call); shape and dtype key the rest. Heads shard on the TP
        columns, seq replicated across the SP rows (`dims=[None, 1]`) — the layout the ring op
        reconstructs into. Template: `models/demos/gpt_oss_d_p/tt/ccl.py:108`.
        """
        cache_key = (key, n_kv, seq, head_dim, str(dtype))
        if cache_key not in self._ring_gather_buffers:
            rows, cols = tuple(self.mesh_device.shape)
            self._ring_gather_buffers[cache_key] = ttnn.from_torch(
                torch.zeros(1, n_kv, seq, head_dim),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=(rows, cols), dims=[None, 1]),
            )
        return self._ring_gather_buffers[cache_key]

    def reset_global_semaphores(self):
        """Reset the reduce-scatter / all-gather ping-pong semaphores to 0.

        This deliberately does **not** reset the barrier or ring-attention semaphores, matching
        `models/demos/gpt_oss_d_p/tt/ccl.py:132`. That upstream comment justifies the omission with
        "one-shot prefill never reuses a CCLManager across runs", which is **false** for chunked
        prefill — which is exactly why `DEC-026` logs the decision to ship it unchanged and makes
        `G-RACE` the measurement rather than assuming either way.
        """
        for sem in self.rs_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
        for sem in self.ag_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
