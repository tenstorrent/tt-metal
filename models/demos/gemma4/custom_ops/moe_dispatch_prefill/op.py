# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``MoeDispatchPrefill`` / ``MoeCombinePrefill``: the token dispatch and combine of the 26B's routed-expert prefill
(``tt/experts/dispatch_prefill.py``) as model-local ``ttnn.generic_op`` programs.

They are the DeepSeek prefill ``dispatch`` / ``combine`` programs (``ttnn.experimental.deepseek_prefill``) with their
kernels copied under ``kernels/`` and the two program factories rewritten in Python, so an upstream change to the
fabric headers or to the shared op cannot break the model (tt-metal main >= 964dc40f2a8 added a
``template <Fabric2DAxis AXIS>`` to ``fabric_2d_route_interface.h`` that the shared factories' ``-DAXIS=<n>`` kernel
define collides with: ``reader_worker_dispatch`` / ``writer_sender_dispatch`` stop compiling and the 26B cannot
boot). The kernel logic is the shared op's; the only edits are the ``AXIS`` define renamed ``DISPATCH_AXIS`` and the
op-local headers included by name. Program-cache and trace contract: fixed output shapes, no host reads, the
cross-device handshake semaphores created once per op instance (the kernels reset them), buffer addresses re-applied
on every call through the runtime args of the cached per-chip programs.

Dispatch (one program per chip; ``cluster_axis`` 0: the mesh rows of one column are the dispatch group):
| tensor | per-chip shape / dtype / layout | role |
|---|---|---|
| ``x`` | ``[1, S, H]`` bf16 ROW_MAJOR (or TILE) DRAM | this chip's slice of the chunk's tokens |
| ``indices`` | ``[1, S, K]`` uint16 ROW_MAJOR DRAM | the top-k expert ids (``num_routed_experts`` = the padding sentinel) |
| ``expert_offsets`` | ``[1, E]`` int32 / uint32 ROW_MAJOR DRAM | where this chip's run starts in each expert's region (``offset_cumsum``) |
| ``dispatch_table`` | ``[1, E + 1]`` int32 ROW_MAJOR DRAM | expert id -> chip index along the dispatch axis, -1 = not in this group |
| ``padding_config`` | ``[1, 2]`` uint32 ROW_MAJOR DRAM, optional | ``[real tokens, pad side]``: bounds the token loop |
| ``buffer`` (out) | ``[1, 1, M, H]`` bf16 ROW_MAJOR DRAM | the flat expert-centric dispatch buffer (``M = max_dispatch_buffer_token_size``) |
| ``metadata`` (out) | ``[1, 1, M, metadata_len]`` int32 ROW_MAJOR DRAM | per routed token ``[source chip (linearized), token, top-k slot]`` |

Combine: ``dispatched_buffer`` ``[1, 1, M, H]`` bf16 (ROW_MAJOR or TILE) or bfp8 (TILE), ``metadata`` as above,
``expert_token_counts`` / ``expert_region_offsets`` ``[.., E]`` int32 / uint32 ROW_MAJOR -> ``[1, 1, S, K, H]`` bf16
ROW_MAJOR: every routed token's expert output back at the token's ``[token, slot]`` on its source chip; slots of
experts outside this dispatch group are untouched (``init_zeros`` zeroes the output first).
"""

from pathlib import Path

import torch

import ttnn
from models.demos.gemma4.custom_ops.grid import compute_grid

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32
METADATA_LEN = 3  # [linearized source chip, token index, top-k slot]
MAX_WORKER_CORES = 4  # one sender core per fabric link, at most 4 links
MAX_UNTILIZERS_PER_SENDER = 4  # the combine's per-core semaphore budget (16 slots)
SLOTS_PER_UNTILIZER = 16  # the combine sender's receive ring depth per untilizer (a power of two)
SPARSE_MCAST_MAX_HOPS = 16  # the 1D sparse-multicast hop mask is 16 bits wide
NOC_MAX_BURST_BYTES = {"blackhole": 16384, "wormhole_b0": 8192}
_ELEM_BYTES = {ttnn.bfloat16: 2, ttnn.uint16: 2, ttnn.int32: 4, ttnn.uint32: 4, ttnn.float32: 4, ttnn.uint8: 1}
_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.int32: 4096, ttnn.uint32: 4096}
_TOPOLOGY_VALUE = {ttnn.Topology.Linear: 1, ttnn.Topology.Ring: 2}  # tt::tt_fabric::Topology
_2D_FABRIC = ("FABRIC_2D", "FABRIC_2D_TORUS_XY", "FABRIC_2D_TORUS_X", "FABRIC_2D_TORUS_Y")


# ── host helpers (the shared factories' detail:: namespace) ───────────────────────────────────────────────────────


def _volume(shape) -> int:
    v = 1
    for d in shape:
        v *= int(d)
    return v


def tensor_pages(t: ttnn.Tensor):
    """``(num_pages, page_size, aligned_page_size)`` of an interleaved tensor's buffer: one row per page in
    ROW_MAJOR (``width x element bytes``), one tile per page in TILE; pages aligned to the buffer's memory alignment
    (DRAM 64 B on Blackhole / 32 B on Wormhole, L1 16 B)."""
    mc = t.memory_config()
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, f"interleaved tensors only ({mc})"
    align = ttnn.get_dram_alignment() if mc.buffer_type == ttnn.BufferType.DRAM else ttnn.get_l1_alignment()
    shape = [int(d) for d in t.padded_shape]
    if t.layout == ttnn.ROW_MAJOR_LAYOUT:
        page = shape[-1] * _ELEM_BYTES[t.dtype]
        pages = _volume(shape[:-1])
    else:
        page = _TILE_BYTES[t.dtype]
        pages = _volume(shape) // (TILE * TILE)
    return pages, page, -(-page // align) * align


def _round_up(v: int, a: int) -> int:
    return -(-int(v) // int(a)) * int(a)


def _stringify(values) -> str:
    """``ccl::common::stringify``: a brace list the kernels take as an array initializer."""
    return "{" + "".join(f"{int(v)}, " for v in values) + "}"


def _crs(cores) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _cb(core_ranges, index: int, total: int, page: int, dtype) -> ttnn.CBDescriptor:
    return ttnn.CBDescriptor(
        total_size=int(total),
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=int(index), data_format=dtype, page_size=int(page))],
    )


def is_2d_fabric() -> bool:
    """Whether the opened fabric routes in 2D (``FABRIC_2D*``): the kernels then take the portable routing-plane
    connections (``setup_routing_plane_connection`` injects their ``FABRIC_2D`` define)."""
    return ttnn.get_fabric_config().name in _2D_FABRIC


def mesh_neighbors(mesh_shape, coord, topology, axis: int):
    """``ccl::common::get_neighbors`` along one axis: ``(neighbors, directions)`` with ``directions`` the
    ``[East, West, North, South]`` flags the kernels' ``DIRECTIONS`` define carries (axis 1: East then West, axis 0:
    North then South; a Ring / Torus axis wraps)."""
    wrap = topology in (ttnn.Topology.Ring, ttnn.Topology.Torus)
    neighbors, directions = [], [False, False, False, False]

    def add(direction: int, offset: int):
        dim = int(mesh_shape[axis])
        v = int(coord[axis]) + offset
        if wrap:
            v %= dim
        elif not 0 <= v < dim:
            return
        n = list(coord)
        n[axis] = v
        neighbors.append(tuple(n))
        directions[direction] = True

    if axis == 1:
        add(0, +1)
        add(1, -1)
    else:
        add(2, -1)
        add(3, +1)
    assert neighbors, "no neighbors along the dispatch axis"
    return neighbors, directions


def semaphore_buffer_type(mesh_device):
    """L1_SMALL when the mesh has that region (``tt/ccl.py::ccl_semaphores_in_l1_small``: a region a trace replay's
    intermediates never touch), else worker L1 (the shared op's default)."""
    from models.demos.gemma4.tt.ccl import ccl_semaphores_in_l1_small

    return ttnn.BufferType.L1_SMALL if ccl_semaphores_in_l1_small(mesh_device) else ttnn.BufferType.L1


def _block_ct_dim(hidden: int) -> int:
    """Largest divisor of ``hidden / 32`` that is <= 8 (the pack_untilize block width)."""
    full = hidden // TILE
    b = 8
    while full % b != 0:
        b -= 1
    return b


class _FabricPlan:
    """The per-op mesh facts every per-chip program shares."""

    def __init__(self, mesh_device, cluster_axis: int, num_links: int, topology):
        assert cluster_axis == 0, f"cluster_axis must be 0 (the shared op's tested value), got {cluster_axis}"
        assert topology in _TOPOLOGY_VALUE, f"topology must be Linear or Ring, got {topology}"
        assert 1 <= int(num_links) <= MAX_WORKER_CORES, f"num_links must be 1..{MAX_WORKER_CORES}, got {num_links}"
        self.mesh_device = mesh_device
        self.rows, self.cols = (int(d) for d in mesh_device.shape)
        assert (
            self.rows * self.cols > 1 and self.rows > 1
        ), "the dispatch group (the mesh rows) needs more than one chip"
        self.axis, self.num_links, self.topology = int(cluster_axis), int(num_links), topology
        self.is_2d = is_2d_fabric()
        self.l1_alignment = int(ttnn.get_l1_alignment())
        self.packet_header_bytes = int(ttnn.get_tt_fabric_packet_header_size_bytes())
        self.fabric_max_packet = int(ttnn.get_tt_fabric_max_payload_size_bytes())
        self.dest_mesh_id, self.dest_chip_id = [], []
        for r in range(self.rows):
            for c in range(self.cols):
                node = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
                self.dest_mesh_id.append(int(node.mesh_id))
                self.dest_chip_id.append(int(node.chip_id))
        gx, gy = compute_grid(mesh_device)
        self.grid = (gx, gy)
        self.worker_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
        self.row_cores = [ttnn.CoreCoord(x, 0) for x in range(gx)]  # the worker line: the grid's first row
        self.arch = "blackhole" if "blackhole" in ttnn.get_arch_name() else "wormhole_b0"

    def noc(self, core):
        v = self.mesh_device.worker_core_from_logical_core(core)
        return int(v.x), int(v.y)

    def node(self, coord):
        return self.mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))

    def fabric_defines(self, coord):
        """``DEST_CHIP_ID`` / ``DEST_MESH_ID`` / ``DIRECTIONS`` (+ the axis) of one chip's kernels."""
        _, directions = mesh_neighbors((self.rows, self.cols), coord, self.topology, self.axis)
        return [
            ("DEST_CHIP_ID", _stringify(self.dest_chip_id)),
            ("DEST_MESH_ID", _stringify(self.dest_mesh_id)),
            ("DIRECTIONS", _stringify(directions)),
            ("DISPATCH_AXIS", str(self.axis)),
        ]

    def connect(self, program, writer_kernel_idx: int, coord, core, core_link: int, label: str):
        """The fabric connection runtime args of one sender core (appended after the kernel's own args): one
        per-direction connection per dispatch-axis neighbor under FABRIC_1D (``setup_fabric_connection``), one
        routing-plane connection per physical first-hop direction of the group's peers under FABRIC_2D (the
        kernel reads the connection count first)."""
        src = self.node(coord)
        if self.is_2d:
            dst_nodes, used = [], set()
            for r in range(self.rows):
                for c in range(self.cols):
                    peer = (r, c)
                    same_group = peer[1] == coord[1] if self.axis == 0 else peer[0] == coord[0]
                    if peer == tuple(coord) or not same_group:
                        continue
                    node = self.node(peer)
                    direction = ttnn.get_eth_forwarding_direction(src, node)
                    assert direction is not None, f"no Fabric2D forwarding direction from {coord} to {peer}"
                    if direction not in used:
                        used.add(direction)
                        dst_nodes.append(node)
            links = []
            for node in dst_nodes:
                valid = list(ttnn.get_forwarding_link_indices(self.mesh_device, src, node))
                assert valid, f"no forwarding links from {coord} to its {label} neighbor"
                links.append(int(valid[core_link % len(valid)]))
            args = [len(dst_nodes)]
            args += list(ttnn.setup_routing_plane_connection(src, dst_nodes, links, program, writer_kernel_idx, core))
            return args
        neighbors, _ = mesh_neighbors((self.rows, self.cols), coord, self.topology, self.axis)
        args = []
        for n in neighbors:
            if n == tuple(coord):
                continue
            args += list(ttnn.setup_fabric_connection(src, self.node(n), core_link, program, core))
        return args


def _set_runtime_args(program, templates, addresses):
    """Write every core's runtime args of a cached program from its templates (tensor roles -> this call's buffer
    addresses)."""
    for kernel_idx, core, template in templates:
        program.kernels[kernel_idx].runtime_args[core.x][core.y] = [
            addresses[v] if isinstance(v, str) else int(v) for v in template
        ]


def _check_runtime_args(program, templates, addresses):
    """Build-time check that the cached program's runtime args are written in place (a copy would leave the
    first call's addresses on every later call)."""
    for kernel_idx, core, template in templates:
        got = list(program.kernels[kernel_idx].runtime_args[core.x][core.y])
        want = [addresses[v] if isinstance(v, str) else int(v) for v in template]
        assert got == want, f"runtime args of kernel {kernel_idx} core {core} not applied in place"


# ── dispatch ──────────────────────────────────────────────────────────────────────────────────────────────────────


class MoeDispatchPrefill:
    """The prefill token dispatch (``ttnn.experimental.deepseek_prefill.dispatch``) as a model-local program."""

    CB_INPUT, CB_INDICES, CB_OFFSETS, CB_TABLE = 0, 1, 3, 9
    CB_SIGNAL, CB_UNTILIZE_OUT, CB_METADATA, CB_PLAN, CB_ROUTE_SCRATCH = 10, 11, 13, 14, 15
    CB_PACKET_HEADER = 8
    CB_PAD_READER, CB_PAD_WRITER = 16, 17
    WRITER_CB_SETS = {0: (4, 5, 6), 1: (16, 17, 18)}  # sender-side ring CBs per worker; more take 19, 20, 21, ...

    def __init__(
        self,
        mesh_device,
        *,
        dispatch_group_size: int,
        experts_per_chip: int,
        num_routed_experts: int,
        num_experts_per_tok: int,
        metadata_len: int,
        max_dispatch_buffer_token_size: int,
        seq_len_per_chip: int,
        emb_dim: int,
        cluster_axis: int = 0,
        num_links: int = 1,
        topology=ttnn.Topology.Linear,
        num_workers_per_sender: int = 2,
    ):
        assert metadata_len == METADATA_LEN, "the bf16 path carries the three routing fields (no fp8 scale tail)"
        assert num_workers_per_sender >= 1
        self.plan = _FabricPlan(mesh_device, cluster_axis, num_links, topology)
        assert dispatch_group_size == self.plan.rows, (dispatch_group_size, self.plan.rows)
        self.mesh_device = mesh_device
        self.dgs, self.experts_per_chip = int(dispatch_group_size), int(experts_per_chip)
        self.num_experts, self.top_k = int(num_routed_experts), int(num_experts_per_tok)
        self.metadata_len, self.max_buf = int(metadata_len), int(max_dispatch_buffer_token_size)
        self.seq_per_chip, self.hidden = int(seq_len_per_chip), int(emb_dim)
        self.num_workers = int(num_workers_per_sender)
        self.num_cores = min(self.plan.num_links, MAX_WORKER_CORES)  # sender cores, one per link
        cores_per_sender = 1 + self.num_workers
        assert len(self.plan.row_cores) >= cores_per_sender * self.num_cores, (
            f"the worker line has {len(self.plan.row_cores)} cores for {self.num_cores} senders of "
            f"{cores_per_sender} cores each"
        )
        self.sender_cores, self.worker_groups = [], []
        for s in range(self.num_cores):
            self.sender_cores.append(self.plan.row_cores[cores_per_sender * s])
            self.worker_groups.append(
                [self.plan.row_cores[cores_per_sender * s + 1 + u] for u in range(self.num_workers)]
            )
        # the cross-device handshake semaphores, allocated once (the kernels wait on and reset them per call)
        sem_type = semaphore_buffer_type(mesh_device)
        self.init_sem, self.exit_sem, self.cross_sem = (
            ttnn.create_global_semaphore(mesh_device, self.plan.worker_grid, 0, sem_type) for _ in range(3)
        )
        ttnn.synchronize_device(mesh_device)
        self._programs: dict = {}  # (layout, has_padding) -> {coord: (program, rt templates)}

    @staticmethod
    def shard_expert_dispatch_table(mesh_device, table: torch.Tensor, dispatch_axis: int = 0) -> ttnn.Tensor:
        """The ``[groups, E + 1]`` dispatch table on the mesh: its rows over the group axis, replicated along the
        dispatch axis (``TtDispatchModule.shard_expert_dispatch_table``)."""
        dims = (None, 0) if dispatch_axis == 0 else (0, None)
        return ttnn.from_torch(
            table,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=dims),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.int32,
        )

    @staticmethod
    def golden(x, indices, offsets, table, lin_coords, mesh_cols: int, max_buf: int, real_tokens=None):
        """torch model of the dispatch over one dispatch group (axis 0): per source chip ``i`` (group position),
        ``x[i]`` [S, H], ``indices[i]`` [S, K] (long), ``offsets[i]`` [E] (this chip's run start per expert),
        ``table`` [E + 1] (expert -> group position, -1 = elsewhere), ``lin_coords[i]`` the chip's linearized mesh
        index, ``real_tokens[i]`` the unpadded rows. Returns per destination position ``(buffer [max_buf, H],
        metadata [max_buf, 3], written [max_buf] bool)``."""
        n = len(x)
        H = x[0].shape[-1]
        buffers = [torch.zeros(max_buf, H, dtype=x[0].dtype) for _ in range(n)]
        metadata = [torch.full((max_buf, METADATA_LEN), -1, dtype=torch.int64) for _ in range(n)]
        written = [torch.zeros(max_buf, dtype=torch.bool) for _ in range(n)]
        for i in range(n):
            counter = offsets[i].clone().long()
            S = x[i].shape[0] if real_tokens is None else int(real_tokens[i])
            for t in range(S):
                for k in range(indices[i].shape[1]):
                    e = int(indices[i][t, k])
                    d = int(table[e])
                    if d < 0:
                        continue
                    page = int(counter[e])
                    counter[e] += 1
                    if page >= max_buf:
                        continue  # the kernel's bounds check drops it
                    buffers[d][page] = x[i][t]
                    metadata[d][page] = torch.tensor([lin_coords[i], t, k])
                    written[d][page] = True
        return buffers, metadata, written

    # ── program build ──

    def _build(self, coord, x, indices, offsets, table, out, meta, padding):
        plan = self.plan
        is_row_major = x.layout != ttnn.TILE_LAYOUT
        has_padding = padding is not None
        r, c = coord
        lin = r * plan.cols + c
        src = plan.node(coord)
        src_mesh_id, src_chip_id = int(src.mesh_id), int(src.chip_id)
        tokens = _volume([int(d) for d in x.shape]) // self.hidden
        read_batch = 8 if is_row_major else 32
        total_batches = -(-tokens // read_batch)
        block_ct = _block_ct_dim(self.hidden)
        l1a = plan.l1_alignment
        writer_cb_size = read_batch
        in_pages, in_page, in_apage = tensor_pages(x)
        idx_pages, idx_page, idx_apage = tensor_pages(indices)
        off_pages, off_page, off_apage = tensor_pages(offsets)
        out_pages, out_page, out_apage = tensor_pages(out)
        meta_pages, meta_page, meta_apage = tensor_pages(meta)
        tab_pages, tab_page, tab_apage = tensor_pages(table)
        if is_row_major:
            assert in_apage == out_apage, f"row-major dispatch is a byte copy: page {in_apage} != {out_apage}"
        all_workers = [u for g in self.worker_groups for u in g]
        sender_grid, worker_grid = _crs(self.sender_cores), _crs(all_workers)

        # semaphores: per-entry credits between a worker and its sender, the sender's address handshake, the
        # workers' offsets baton
        sems, next_id = [], [0]

        def add_sema(crs, init=0):
            sid = next_id[0]
            next_id[0] += 1
            sems.append(ttnn.SemaphoreDescriptor(sid, ttnn.CoreType.WORKER, crs, int(init)))
            return sid

        data_avail = [add_sema(sender_grid) for _ in range(self.num_workers)]
        space_avail = add_sema(worker_grid, writer_cb_size)
        addr_ready = add_sema(worker_grid)
        cross_addr = add_sema(worker_grid)
        turn = add_sema(worker_grid)

        # worker-core CBs
        cbs = []
        if is_row_major:
            cbs.append(_cb(worker_grid, self.CB_INPUT, 2 * read_batch * out_apage, out_apage, out.dtype))
        else:
            cbs.append(_cb(worker_grid, self.CB_INPUT, 2 * block_ct * in_apage, in_apage, x.dtype))
        cbs.append(_cb(worker_grid, self.CB_INDICES, read_batch * idx_apage, idx_apage, indices.dtype))
        cbs.append(_cb(worker_grid, self.CB_OFFSETS, off_pages * off_apage, off_apage, offsets.dtype))
        cbs.append(_cb(worker_grid, self.CB_TABLE, tab_pages * tab_apage, tab_apage, table.dtype))
        if not is_row_major:
            cbs.append(_cb(worker_grid, self.CB_SIGNAL, 2 * l1a, l1a, ttnn.uint32))
            cbs.append(_cb(worker_grid, self.CB_UNTILIZE_OUT, 2 * read_batch * out_apage, out_apage, out.dtype))
        cbs.append(
            _cb(worker_grid, self.CB_METADATA, (read_batch * self.top_k + 1) * meta_apage, meta_apage, meta.dtype)
        )
        # 1D sparse multicast: co-directional destinations of a token grouped into one record of 64 B
        topology_wraps = plan.topology in (ttnn.Topology.Ring, ttnn.Topology.Torus)

        def max_hops(dim):
            return dim // 2 if topology_wraps else max(dim - 1, 0)

        hops_fit = max_hops(plan.rows) + max_hops(plan.cols) <= SPARSE_MCAST_MAX_HOPS
        sparse_mcast = (not plan.is_2d) and plan.num_links > 0 and hops_fit
        route_stride = _round_up(64, l1a) if sparse_mcast else l1a
        cbs.append(_cb(worker_grid, self.CB_ROUTE_SCRATCH, route_stride, route_stride, ttnn.uint32))
        plan_page = _round_up(16 + read_batch * self.top_k * 48, l1a)  # PlanHeader + PlanEntry[] (dispatch_plan.hpp)
        cbs.append(_cb(worker_grid, self.CB_PLAN, 2 * plan_page, plan_page, ttnn.uint32))
        # sender-core CBs: one (route, payload, metadata) ring per worker
        writer_cb_ids, next_free = [], 19
        for s in range(self.num_workers):
            ids = self.WRITER_CB_SETS.get(s)
            if ids is None:
                ids = (next_free, next_free + 1, next_free + 2)
                next_free += 3
            writer_cb_ids.append(ids)
            cbs.append(_cb(sender_grid, ids[0], writer_cb_size * route_stride, route_stride, ttnn.uint32))
            cbs.append(_cb(sender_grid, ids[1], writer_cb_size * out_apage, out_apage, out.dtype))
            cbs.append(_cb(sender_grid, ids[2], writer_cb_size * meta_apage, meta_apage, meta.dtype))
        cbs.append(
            _cb(sender_grid, self.CB_PACKET_HEADER, 2 * plan.packet_header_bytes, plan.packet_header_bytes, ttnn.uint8)
        )
        if has_padding:
            pad_pages, pad_page, pad_apage = tensor_pages(padding)
            cbs.append(_cb(worker_grid, self.CB_PAD_READER, pad_apage, pad_apage, padding.dtype))
            cbs.append(_cb(worker_grid, self.CB_PAD_WRITER, pad_apage, pad_apage, padding.dtype))

        # compile-time args shared by the sender writer (its own tail appended below)
        ct = [
            self.CB_INPUT, self.CB_INDICES, self.CB_OFFSETS, 4, 5, 6, 7, self.CB_PACKET_HEADER, self.CB_TABLE,
            in_pages, idx_pages, off_pages, out_pages, meta_pages, tab_pages,
            in_page, idx_page, off_page, out_page, meta_page, tab_page,
            plan.rows * plan.cols, self.hidden, self.experts_per_chip, self.num_experts, self.top_k,
            self.metadata_len, tokens,
            src_mesh_id, src_chip_id, plan.rows, plan.cols, lin,
            in_apage, idx_apage, off_apage, out_apage, meta_apage, tab_apage,
            plan.fabric_max_packet, l1a, plan.num_links, _TOPOLOGY_VALUE[plan.topology],
            read_batch, self.max_buf,
        ]  # fmt: skip
        for t in (x, indices, offsets, out, meta, table):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        fabric_defines = plan.fabric_defines(coord)
        if sparse_mcast:
            fabric_defines.append(("SPARSE_MCAST_DISPATCH", "1"))

        kernels, templates = [], []
        # the sender writer: drains the workers' rings and sends over the fabric
        writer_idx = len(kernels)
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_sender_dispatch.cpp"),
                core_ranges=sender_grid,
                compile_time_args=ct + [writer_cb_size, self.num_workers, route_stride],
                defines=list(fabric_defines),
                runtime_args=ttnn.RuntimeArgs(),
                config=ttnn.WriterConfigDescriptor(),
            )
        )
        # worker cores: reader (routing, plan) + writer (local DRAM writes, the sender's rings)
        worker_defines = list(fabric_defines)
        if has_padding:
            worker_defines.append(("HAS_PADDING_CONFIG", "1"))
        reader_defines = list(worker_defines)
        if is_row_major:
            reader_defines.append(("ROW_MAJOR_INPUT", "1"))
        reader_idx, worker_writer_idx = [], []
        for j, core in enumerate(all_workers):
            s, u = j // self.num_workers, j % self.num_workers
            rct = [
                self.CB_INPUT, self.CB_SIGNAL, self.hidden, in_apage, total_batches, u, self.num_workers,
                self.CB_INDICES, self.CB_OFFSETS, self.CB_TABLE, self.CB_PLAN, read_batch,
                idx_apage, off_apage, tab_apage, off_pages, tab_pages, self.top_k, self.num_experts, self.max_buf,
                s, self.num_cores, plan.rows * plan.cols, plan.rows, plan.cols, lin,
                _TOPOLOGY_VALUE[plan.topology], block_ct,
            ]  # fmt: skip
            for t in (x, indices, offsets, table):
                rct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
            if has_padding:
                rct += ttnn.TensorAccessorArgs(padding).get_compile_time_args() + [self.CB_PAD_READER]
            wct = [
                self.CB_INPUT if is_row_major else self.CB_UNTILIZE_OUT, read_batch, out_apage, total_batches, u,
                self.num_workers, self.CB_METADATA, meta_apage, self.CB_PLAN, lin, route_stride, writer_cb_size,
                self.CB_ROUTE_SCRATCH, read_batch * self.top_k,
            ]  # fmt: skip
            for t in (out, meta):
                wct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
            if has_padding:
                wct += ttnn.TensorAccessorArgs(padding).get_compile_time_args() + [self.CB_PAD_WRITER]
            reader_idx.append(len(kernels))
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "reader_worker_dispatch.cpp"),
                    core_ranges=_crs([core]),
                    compile_time_args=rct,
                    defines=list(reader_defines),
                    runtime_args=ttnn.RuntimeArgs(),
                    config=ttnn.ReaderConfigDescriptor(),
                )
            )
            worker_writer_idx.append(len(kernels))
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "writer_worker_dispatch.cpp"),
                    core_ranges=_crs([core]),
                    compile_time_args=wct,
                    defines=list(worker_defines),
                    runtime_args=ttnn.RuntimeArgs(),
                    config=ttnn.WriterConfigDescriptor(),
                )
            )
        if not is_row_major:
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "untilize_dispatch.cpp"),
                    core_ranges=worker_grid,
                    compile_time_args=[
                        self.CB_SIGNAL,
                        self.CB_UNTILIZE_OUT,
                        self.CB_INPUT,
                        self.hidden,
                        read_batch,
                        block_ct,
                    ],
                    runtime_args=ttnn.RuntimeArgs(),
                    config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4),
                )
            )
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)

        # runtime args: tensor roles are substituted per call
        sender_noc = [plan.noc(sc) for sc in self.sender_cores]
        init_addr = int(ttnn.get_global_semaphore_address(self.init_sem))
        exit_addr = int(ttnn.get_global_semaphore_address(self.exit_sem))
        cross_addr_sem = int(ttnn.get_global_semaphore_address(self.cross_sem))
        for s, sc in enumerate(self.sender_cores):
            args = ["x", "idx", "off", "out", "meta", "table", cross_addr_sem, init_addr, 0, tokens, s, self.num_cores]
            args += [exit_addr, addr_ready, cross_addr, space_avail]
            for u, uc in enumerate(self.worker_groups[s]):
                nx, ny = plan.noc(uc)
                args += [writer_cb_ids[u][0], writer_cb_ids[u][1], writer_cb_ids[u][2], nx, ny, data_avail[u]]
            args += plan.connect(program, writer_idx, coord, sc, s % plan.num_links, "dispatch-axis")
            templates.append((writer_idx, sc, args))
        for j, core in enumerate(all_workers):
            s, u = j // self.num_workers, j % self.num_workers
            group = self.worker_groups[s]
            ox, oy = plan.noc(group[0])
            nx, ny = plan.noc(group[(u + 1) % self.num_workers])
            rargs = ["x", "idx", "off", "table", 0, tokens, ox, oy, nx, ny, turn]
            wargs = [
                sender_noc[s][0],
                sender_noc[s][1],
                addr_ready,
                cross_addr,
                data_avail[u],
                space_avail,
                "out",
                "meta",
            ]
            if has_padding:
                rargs.append("pad")
                wargs.append("pad")
            templates.append((reader_idx[j], core, rargs))
            templates.append((worker_writer_idx[j], core, wargs))
        return program, templates

    def _addresses(self, x, indices, offsets, table, out, meta, padding):
        a = {
            "x": x.buffer_address(),
            "idx": indices.buffer_address(),
            "off": offsets.buffer_address(),
            "table": table.buffer_address(),
            "out": out.buffer_address(),
            "meta": meta.buffer_address(),
        }
        if padding is not None:
            a["pad"] = padding.buffer_address()
        return a

    def op(self, x, indices, expert_offsets, dispatch_table, padding_config=None):
        """``(buffer [1, 1, M, H] bf16 ROW_MAJOR, metadata [1, 1, M, metadata_len] int32 ROW_MAJOR)``, DRAM."""
        assert x.dtype == ttnn.bfloat16, f"x: bf16 (got {x.dtype})"
        assert int(x.shape[-1]) == self.hidden, (x.shape, self.hidden)
        assert indices.layout == ttnn.ROW_MAJOR_LAYOUT and indices.dtype == ttnn.uint16, "indices: uint16 ROW_MAJOR"
        assert expert_offsets.layout == ttnn.ROW_MAJOR_LAYOUT and expert_offsets.dtype in (ttnn.int32, ttnn.uint32)
        assert dispatch_table.layout == ttnn.ROW_MAJOR_LAYOUT and dispatch_table.dtype == ttnn.int32
        if padding_config is not None:
            assert padding_config.layout == ttnn.ROW_MAJOR_LAYOUT and int(padding_config.shape[-1]) == 2
            assert padding_config.dtype in (ttnn.uint32, ttnn.int32)
        dram = ttnn.DRAM_MEMORY_CONFIG
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.max_buf, self.hidden]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.mesh_device, dram
        )
        meta = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.max_buf, self.metadata_len]),
            ttnn.int32,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            dram,
        )
        key = (x.layout == ttnn.TILE_LAYOUT, padding_config is not None, tuple(int(d) for d in x.shape))
        addresses = self._addresses(x, indices, expert_offsets, dispatch_table, out, meta, padding_config)
        programs = self._programs.get(key)
        if programs is None:
            programs = self._programs[key] = {}
            for r in range(self.plan.rows):
                for c in range(self.plan.cols):
                    program, templates = self._build(
                        (r, c), x, indices, expert_offsets, dispatch_table, out, meta, padding_config
                    )
                    _set_runtime_args(program, templates, addresses)
                    _check_runtime_args(program, templates, addresses)
                    programs[(r, c)] = (program, templates)
        mesh_program = ttnn.MeshProgramDescriptor()
        for (r, c), (program, templates) in programs.items():
            _set_runtime_args(program, templates, addresses)
            coord = ttnn.MeshCoordinate(r, c)
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program
        io = [x, indices, expert_offsets, dispatch_table]
        if padding_config is not None:
            io.append(padding_config)
        ttnn.generic_op(io + [out, meta], mesh_program)
        return out, meta


# ── combine ───────────────────────────────────────────────────────────────────────────────────────────────────────


class MoeCombinePrefill:
    """The prefill combine (``ttnn.experimental.deepseek_prefill.combine``) as a model-local program."""

    CB_BUFFER, CB_METADATA, CB_COUNTS, CB_ROUTE, CB_PACKET_HEADER = 0, 1, 2, 3, 5
    CB_ZERO_UNTILIZER, CB_ZERO_SENDER, CB_REGION_OFFSETS, CB_META_BATCH = 6, 7, 8, 9
    CB_UNTILIZE = 2  # on the untilizer cores
    CB_RECEIVE, CB_META_RING = 18, 19

    def __init__(
        self,
        mesh_device,
        *,
        dispatch_group_size: int,
        experts_per_chip: int,
        num_experts_per_tok: int,
        seq_len_per_chip: int,
        emb_dim: int,
        cluster_axis: int = 0,
        num_links: int = 1,
        topology=ttnn.Topology.Linear,
        init_zeros: bool = False,
    ):
        self.plan = _FabricPlan(mesh_device, cluster_axis, num_links, topology)
        assert dispatch_group_size == self.plan.rows, (dispatch_group_size, self.plan.rows)
        self.mesh_device = mesh_device
        self.dgs, self.experts_per_chip = int(dispatch_group_size), int(experts_per_chip)
        self.top_k, self.seq_per_chip, self.hidden = int(num_experts_per_tok), int(seq_len_per_chip), int(emb_dim)
        self.init_zeros = bool(init_zeros)
        self.num_cores = min(self.plan.num_links, MAX_WORKER_CORES)
        line = self.plan.row_cores
        assert len(line) > self.num_cores, "the worker line needs an untilizer core per sender"
        # the line split into per-sender groups: the sender first, then its untilizers (at most 4 kept)
        base, extra = len(line) // self.num_cores, len(line) % self.num_cores
        self.sender_cores, self.untilizer_groups = [], []
        pos = 0
        for s in range(self.num_cores):
            size = base + (1 if s >= self.num_cores - extra else 0)
            self.sender_cores.append(line[pos])
            self.untilizer_groups.append(line[pos + 1 : pos + size][:MAX_UNTILIZERS_PER_SENDER])
            pos += size
        assert all(self.untilizer_groups), "every sender needs an untilizer core"
        sem_type = semaphore_buffer_type(mesh_device)
        self.init_sem, self.exit_sem = (
            ttnn.create_global_semaphore(mesh_device, self.plan.worker_grid, 0, sem_type) for _ in range(2)
        )
        ttnn.synchronize_device(mesh_device)
        self._programs: dict = {}

    @staticmethod
    def golden(buffers, metadata, counts, region_offsets, local_experts, seq_per_chip: int, top_k: int):
        """torch model over the chips of one dispatch group: ``buffers[d]`` [M, H], ``metadata[d]`` [M, 3],
        ``counts[d]`` / ``region_offsets[d]`` [E], ``local_experts[d]`` the global ids hosted by chip ``d``.
        Returns per chip ``(out [S, K, H], written [S, K] bool)`` keyed by the metadata's source index: the source
        field is the chip's linearized mesh index, so ``lin_to_pos`` is applied by the caller."""
        n = len(buffers)
        H = buffers[0].shape[-1]
        out = {}
        written = {}
        for d in range(n):
            for e in local_experts[d]:
                start, count = int(region_offsets[d][e]), int(counts[d][e])
                for i in range(count):
                    page = start + i
                    src, t, k = (int(v) for v in metadata[d][page])
                    if src not in out:
                        out[src] = torch.zeros(seq_per_chip, top_k, H, dtype=buffers[0].dtype)
                        written[src] = torch.zeros(seq_per_chip, top_k, dtype=torch.bool)
                    out[src][t, k] = buffers[d][page]
                    written[src][t, k] = True
        return out, written

    def _build(self, coord, buf, meta, counts, region, out):
        plan = self.plan
        is_tile = buf.layout == ttnn.TILE_LAYOUT
        r, c = coord
        lin = r * plan.cols + c
        src = plan.node(coord)
        src_mesh_id, src_chip_id = int(src.mesh_id), int(src.chip_id)
        l1a = plan.l1_alignment
        max_buf = int(buf.shape[-2])
        buf_pages, buf_page, buf_apage = tensor_pages(buf)
        meta_pages, meta_page, meta_apage = tensor_pages(meta)
        cnt_pages, cnt_page, cnt_apage = tensor_pages(counts)
        reg_pages, reg_page, reg_apage = tensor_pages(region)
        out_pages, out_page, out_apage = tensor_pages(out)
        num_experts = int(counts.shape[-1])
        ndg = num_experts // (self.experts_per_chip * self.dgs)
        assert ndg * self.experts_per_chip * self.dgs == num_experts, (num_experts, self.experts_per_chip, self.dgs)
        read_batch = TILE
        full_ct = self.hidden // TILE
        block_ct = _block_ct_dim(self.hidden)
        all_untilizers = [u for g in self.untilizer_groups for u in g]
        sender_map = [s for s, g in enumerate(self.untilizer_groups) for _ in g]
        num_untilizers = len(all_untilizers)
        sender_grid, untilizer_grid = _crs(self.sender_cores), _crs(all_untilizers)
        counter_offset = c * (self.experts_per_chip * plan.rows) + r * self.experts_per_chip

        # semaphores: the lowest id free on every core of the set (disjoint sets share ids; 16 per core)
        sems, slots = [], {}

        def add_sema(cores, init=0):
            for sid in range(16):
                if all(sid not in slots.setdefault((cc.x, cc.y), set()) for cc in cores):
                    for cc in cores:
                        slots[(cc.x, cc.y)].add(sid)
                    sems.append(ttnn.SemaphoreDescriptor(sid, ttnn.CoreType.WORKER, _crs(cores), int(init)))
                    return sid
            raise RuntimeError("no free L1 semaphore slot on the requested cores")

        output_init_complete = add_sema(self.sender_cores)
        output_init_barrier = add_sema(self.sender_cores)

        # sender-core CBs (the untilizers' c_1 is allocated first on their cores below: same L1 address as here)
        cbs = [
            _cb(sender_grid, self.CB_METADATA, read_batch * meta_apage, meta_apage, meta.dtype),
            _cb(sender_grid, self.CB_COUNTS, (cnt_pages + 1) * cnt_apage, cnt_apage, counts.dtype),
            _cb(sender_grid, self.CB_REGION_OFFSETS, reg_pages * reg_apage, reg_apage, region.dtype),
        ]
        for s, sc in enumerate(self.sender_cores):
            k_s = len(self.untilizer_groups[s])
            cbs.append(_cb(_crs([sc]), self.CB_RECEIVE, k_s * SLOTS_PER_UNTILIZER * out_apage, out_apage, out.dtype))
            cbs.append(
                _cb(_crs([sc]), self.CB_META_RING, k_s * SLOTS_PER_UNTILIZER * meta_apage, meta_apage, meta.dtype)
            )
        merged_page = l1a + out_apage  # route info + one output row
        cbs.append(_cb(sender_grid, self.CB_ROUTE, 2 * merged_page, merged_page, ttnn.uint8))
        cbs.append(
            _cb(sender_grid, self.CB_PACKET_HEADER, 2 * plan.packet_header_bytes, plan.packet_header_bytes, ttnn.uint8)
        )

        # compile-time args shared by the sender reader and writer
        ct = [
            self.CB_BUFFER, self.CB_METADATA, self.CB_COUNTS, self.CB_ROUTE, self.CB_PACKET_HEADER,
            buf_pages, meta_pages, cnt_pages, out_pages,
            buf_page, meta_page, cnt_page, out_page,
            self.dgs, self.experts_per_chip, self.top_k, self.seq_per_chip,
            self.hidden,
            buf_apage, meta_apage, cnt_apage, out_apage,
            src_mesh_id, src_chip_id, plan.rows, plan.cols, lin,
            plan.fabric_max_packet, l1a, plan.num_links, _TOPOLOGY_VALUE[plan.topology],
            read_batch, ndg,
            self.CB_REGION_OFFSETS, reg_pages, reg_page, reg_apage,
            max_buf,
        ]  # fmt: skip
        for t in (buf, meta, counts, out, region):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        fabric_defines = plan.fabric_defines(coord)
        init_zeros = ("INIT_ZEROS", "1" if self.init_zeros else "0")
        tile_define = ("IS_TILE_LAYOUT", "1" if is_tile else "0")
        reader_defines = fabric_defines + [init_zeros, tile_define]
        writer_defines = fabric_defines + [init_zeros]

        # sender <-> untilizer handshake semaphores
        counter_ready = add_sema(self.sender_cores + all_untilizers)
        data_ready, credits = [], []
        for s, sc in enumerate(self.sender_cores):
            data_ready.append([add_sema([sc, uc]) for uc in self.untilizer_groups[s]])
            credits.append([add_sema([uc]) for uc in self.untilizer_groups[s]])

        # untilizer-core CBs
        cbs.append(_cb(untilizer_grid, self.CB_METADATA, (cnt_pages + 1) * cnt_apage, cnt_apage, counts.dtype))
        if is_tile:
            cbs.append(_cb(untilizer_grid, self.CB_BUFFER, 2 * block_ct * buf_apage, buf_apage, buf.dtype))
        cbs.append(_cb(untilizer_grid, self.CB_UNTILIZE, 2 * read_batch * out_apage, out_apage, out.dtype))
        cbs.append(_cb(untilizer_grid, self.CB_META_BATCH, 2 * read_batch * meta_apage, meta_apage, meta.dtype))

        kernels = []
        # reader_untilize: one per untilizer core
        reader_untilize_idx = []
        for u, uc in enumerate(all_untilizers):
            rct = [
                self.CB_METADATA, cnt_pages, self.experts_per_chip, counter_offset, self.CB_BUFFER, self.CB_UNTILIZE,
                self.hidden, read_batch, buf_apage, TILE, TILE, max_buf,
                out_apage, cnt_apage, self.CB_META_BATCH, meta_apage, block_ct, cnt_pages + 1,
            ]  # fmt: skip
            for t in (buf, meta):
                rct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
            reader_untilize_idx.append(len(kernels))
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "reader_untilize.cpp"),
                    core_ranges=_crs([uc]),
                    compile_time_args=rct,
                    defines=[tile_define],
                    runtime_args=ttnn.RuntimeArgs(),
                    config=ttnn.ReaderConfigDescriptor(),
                )
            )
        # output zeroing (init_zeros): every core clears a slice of the output before the combine starts
        output_init_done = pages_per_core = remainder = 0
        if self.init_zeros:
            burst = NOC_MAX_BURST_BYTES[plan.arch]
            cbs.append(_cb(sender_grid, self.CB_ZERO_SENDER, burst, burst, ttnn.uint8))
            cbs.append(_cb(untilizer_grid, self.CB_ZERO_UNTILIZER, burst, burst, ttnn.uint8))
            total_init_cores = self.num_cores + num_untilizers
            pages_per_core, remainder = out_pages // total_init_cores, out_pages % total_init_cores
            output_init_done = add_sema(self.sender_cores + all_untilizers)

        def page_range(idx):
            start = idx * pages_per_core + min(idx, remainder)
            return start, start + pages_per_core + (1 if idx < remainder else 0)

        # writer_untilize on the untilizer grid
        wct = [out_apage, self.num_cores if self.init_zeros else 0, self.CB_ZERO_UNTILIZER if self.init_zeros else 0]
        wct += ttnn.TensorAccessorArgs(out).get_compile_time_args()
        wct += [
            self.CB_UNTILIZE, self.CB_METADATA, cnt_pages, cnt_apage, read_batch, self.CB_META_BATCH, self.top_k,
            meta_apage, lin, self.experts_per_chip, counter_offset, max_buf, full_ct, cnt_pages + 1, SLOTS_PER_UNTILIZER,
        ]  # fmt: skip
        writer_untilize_idx = len(kernels)
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_untilize.cpp"),
                core_ranges=untilizer_grid,
                compile_time_args=wct,
                defines=[tile_define, init_zeros],
                runtime_args=ttnn.RuntimeArgs(),
                config=ttnn.WriterConfigDescriptor(),
            )
        )
        # reader_combine: one per sender (its own untilizer count)
        reader_base = list(ct)
        if self.init_zeros:
            reader_base += [self.CB_ZERO_SENDER, num_untilizers]
        reader_idx = []
        for s, sc in enumerate(self.sender_cores):
            reader_idx.append(len(kernels))
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "reader_combine.cpp"),
                    core_ranges=_crs([sc]),
                    compile_time_args=reader_base
                    + [len(self.untilizer_groups[s]), self.CB_RECEIVE, self.CB_META_RING, SLOTS_PER_UNTILIZER],
                    defines=list(reader_defines),
                    runtime_args=ttnn.RuntimeArgs(),
                    config=ttnn.ReaderConfigDescriptor(),
                )
            )
        writer_idx = len(kernels)
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_combine.cpp"),
                core_ranges=sender_grid,
                compile_time_args=list(ct),
                defines=list(writer_defines),
                runtime_args=ttnn.RuntimeArgs(),
                config=ttnn.WriterConfigDescriptor(),
            )
        )
        compute_idx = None
        if is_tile:
            compute_idx = len(kernels)
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "untilize_combine.cpp"),
                    core_ranges=untilizer_grid,
                    compile_time_args=[
                        self.CB_UNTILIZE,
                        self.CB_BUFFER,
                        self.CB_METADATA,
                        cnt_pages,
                        self.experts_per_chip,
                        counter_offset,
                        max_buf,
                        read_batch,
                        full_ct,
                        block_ct,
                        cnt_pages + 1,
                    ],  # fmt: skip
                    runtime_args=ttnn.RuntimeArgs(),
                    config=ttnn.ComputeConfigDescriptor(),
                )
            )
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)

        # the untilizers' global batch positions: sender-minor interleave [S0.U0, S1.U0, S0.U1, ...]
        max_k = max(len(g) for g in self.untilizer_groups)
        pos_by_rank = [[0] * len(g) for g in self.untilizer_groups]
        pos = 0
        for rank in range(max_k):
            for s in range(self.num_cores):
                if rank < len(self.untilizer_groups[s]):
                    pos_by_rank[s][rank] = pos
                    pos += 1
        rank_counter = [0] * self.num_cores
        global_pos = []
        for u in range(num_untilizers):
            s = sender_map[u]
            global_pos.append(pos_by_rank[s][rank_counter[s]])
            rank_counter[s] += 1

        templates = []
        sender_noc = [plan.noc(sc) for sc in self.sender_cores]
        init_addr = int(ttnn.get_global_semaphore_address(self.init_sem))
        exit_addr = int(ttnn.get_global_semaphore_address(self.exit_sem))
        for u, uc in enumerate(all_untilizers):
            s = sender_map[u]
            local_id = sum(1 for j in range(u) if sender_map[j] == s)
            args = ["out"]
            if self.init_zeros:
                start, end = page_range(self.num_cores + u)
                args += [start, end, output_init_done]
                for nx, ny in sender_noc:
                    args += [nx, ny]
            args += [counter_ready, sender_noc[s][0], sender_noc[s][1], data_ready[s][local_id], credits[s][local_id]]
            args += [local_id, 0, self.experts_per_chip, global_pos[u], num_untilizers]
            templates.append((writer_untilize_idx, uc, args))
            templates.append(
                (
                    reader_untilize_idx[u],
                    uc,
                    [counter_ready, "buf", 0, self.experts_per_chip, "meta", global_pos[u], num_untilizers],
                )
            )
            if is_tile:
                k_s = len(self.untilizer_groups[s])
                templates.append(
                    (compute_idx, uc, [0, self.experts_per_chip, local_id, k_s, global_pos[u], num_untilizers])
                )
        for s, sc in enumerate(self.sender_cores):
            group = self.untilizer_groups[s]
            nocs = [plan.noc(uc) for uc in group]
            xs, ys = [n[0] for n in nocs], [n[1] for n in nocs]
            rargs = [
                "buf",
                "meta",
                "counts",
                "region",
                "out",
                output_init_complete,
                output_init_barrier,
                self.num_cores,
            ]
            rargs += [0, self.experts_per_chip]
            if self.init_zeros:
                start, end = page_range(s)
                rargs += [start, end, output_init_done]
            rargs += [counter_ready, min(xs), min(ys), max(xs), max(ys)]
            for cidx, (nx, ny) in enumerate(nocs):
                rargs += [data_ready[s][cidx], credits[s][cidx], nx, ny]
            templates.append((reader_idx[s], sc, rargs))
            wargs = ["buf", "meta", "counts", "region", "out", output_init_complete, init_addr, exit_addr]
            wargs += [output_init_barrier, self.num_cores, 0, self.experts_per_chip]
            for nx, ny in sender_noc:
                wargs += [nx, ny]
            wargs += plan.connect(program, writer_idx, coord, sc, s % plan.num_links, "combine-axis")
            templates.append((writer_idx, sc, wargs))
        return program, templates

    def op(self, dispatched_buffer, dispatched_metadata, expert_token_counts, expert_region_offsets):
        """``[1, 1, S, K, H]`` bf16 ROW_MAJOR DRAM: the routed outputs back at their ``[token, slot]``."""
        buf, meta, counts, region = dispatched_buffer, dispatched_metadata, expert_token_counts, expert_region_offsets
        assert buf.dtype == ttnn.bfloat16 or (buf.dtype == ttnn.bfloat8_b and buf.layout == ttnn.TILE_LAYOUT), buf.dtype
        assert meta.layout == ttnn.ROW_MAJOR_LAYOUT and meta.dtype == ttnn.int32, "metadata: int32 ROW_MAJOR"
        assert counts.layout == ttnn.ROW_MAJOR_LAYOUT and counts.dtype in (ttnn.int32, ttnn.uint32)
        assert region.layout == ttnn.ROW_MAJOR_LAYOUT and region.dtype in (ttnn.int32, ttnn.uint32)
        assert [int(d) for d in region.shape] == [int(d) for d in counts.shape], (region.shape, counts.shape)
        assert int(buf.shape[-1]) == self.hidden and int(buf.shape[-2]) == int(meta.shape[-2]), (buf.shape, meta.shape)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.seq_per_chip, self.top_k, self.hidden]),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        key = (buf.layout == ttnn.TILE_LAYOUT, buf.dtype, int(buf.shape[-2]), int(counts.shape[-1]))
        addresses = {
            "buf": buf.buffer_address(),
            "meta": meta.buffer_address(),
            "counts": counts.buffer_address(),
            "region": region.buffer_address(),
            "out": out.buffer_address(),
        }
        programs = self._programs.get(key)
        if programs is None:
            programs = self._programs[key] = {}
            for r in range(self.plan.rows):
                for c in range(self.plan.cols):
                    program, templates = self._build((r, c), buf, meta, counts, region, out)
                    _set_runtime_args(program, templates, addresses)
                    _check_runtime_args(program, templates, addresses)
                    programs[(r, c)] = (program, templates)
        mesh_program = ttnn.MeshProgramDescriptor()
        for (r, c), (program, templates) in programs.items():
            _set_runtime_args(program, templates, addresses)
            coord = ttnn.MeshCoordinate(r, c)
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program
        ttnn.generic_op([buf, meta, counts, region, out], mesh_program)
        return out


__all__ = ["MoeDispatchPrefill", "MoeCombinePrefill", "tensor_pages", "mesh_neighbors", "is_2d_fabric"]
