# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-side handoff: SP prefill (SPPrefillSC on the 4 unit submeshes of a 1x4 mesh) -> TP=4 decode model on the
PARENT 1x4 mesh, in ONE process, with no host round trip of the state.

Used only by tests/test_sp_e2e_tp4.py (nothing else imports it; the SP and TP model code paths are unchanged).

Co-existence of the parent mesh and its unit submeshes (probed on a 1x4 p300c mesh):
  * Allocators: the parent and every submesh have INDEPENDENT allocators over the same physical memory (both hand
    out the same addresses). ``reserve(target, source)`` (ttnn._ttnn.multi_device.reserve_allocator_regions) marks
    every region the source has allocated as reserved in the target (DRAM, L1, L1_SMALL, TRACE), leaving the
    target's own allocations intact. The e2e test reserves in both directions after every allocation phase and
    orders the phases so that nothing is allocated or compiled on one side after the other side parked a trace.
  * Dispatch: at every parent <-> submesh switch the side being LEFT is quiesced (``MeshSwitch``); quiescing while
    both sides are in use hangs (Finish waits on an event the other side's reset renumbered).
  * Views: ``view(t, mesh, page_offset, shape)`` (ttnn._ttnn.multi_device.unit_mesh_view_pages) is a non-owning
    tensor over pages [page_offset, ...) of an interleaved tensor, on the tensor's own mesh, its parent or a unit
    submesh; used to address the TP model's (parent) buffers from submesh ops and to page-range sub-tensors.

Handoff (M1: explicit transfers into the TP model's own persistent buffers). Die d = submesh d = TP device d. After
SP prefill die d's paged cache holds positions [0, p_d * 64) of BOTH kv heads (p_d = (d + 1) * 16 blocks at
T = 4096, uniform spans), die 3 also holds the final GDN recurrent + conv state of every GDN layer. TP device d
holds kv head h_d = (d * n_kv) // 4 = [0, 0, 1, 1][d] and GDN value heads [4d, 4d + 4).
  * TP paged caches: [192, 1, 64, 256] per device (bf8, like SP). Slots [0, 2 * p_d) = a copy of die d's SP cache
    blocks [0, p_d) (both heads, one local page-range copy per cache); slots [128, ...) = the missing head-h_d blocks
    [p_d, 64) (tails) followed by the decode blocks. A per-device (sharded) page table maps logical block i ->
    2i + h_d (i < p_d) or 128 + (i - p_d).
  * Tails (static-dst sends straight into the TP cache slots, over backward sockets; logical dies 0-1-2-3-0 form a
    physical ring, 1D sockets need the same physical row / column so 3->1 and 2->0 are forwarded):
      die 3 -> 2: h1 [48, 64)                  die 3 -> 0: h0 [48, 64)   (die 0 forwards it -> 1)
      die 2 -> 1: h0 [32, 48) (die 1 forwards it -> 0)                  die 1 -> 0: h0 [16, 32)
  * GDN rec: die 3 concatenates the 18 layers' states, slices value heads [4d, 4d + 4) per die and sends each slice
    into BIG_REC (parent [n_gdn, 4, 128, 128] fp32; the TP layers' rec_state are rebound to its page views); die
    1's slice goes through a staging buffer on die 2.
  * GDN conv: die 3 regroups the 18 conv states [1, 3, 6144] (channels [q|k|v]) into [4, 3 * n_gdn, 1536] (per die
    [q_d|k_d|v_d]), sends block d into CONV_LAND (parent [1, 3 * n_gdn, 1536]); each die then expands it into
    BIG_CONV (parent [3 * n_gdn, 1, 1536]; the TP layers' conv_states[1..3] are rebound to its page views).
"""
import threading
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.sp_prefill import BLOCK_SIZE, _build_socket_pair

_MD = ttnn._ttnn.multi_device


def reserve(target, source):
    """Reserve in target's allocator everything source's allocator holds (see module docstring)."""
    return _MD.reserve_allocator_regions(target, source)


def reserve_all(mesh, subs, direction):
    """direction 'parent': parent <- every sub; 'subs': every sub <- parent."""
    if direction == "parent":
        return sum(reserve(mesh, s) for s in subs)
    assert direction == "subs"
    return sum(reserve(s, mesh) for s in subs)


def view(t, mesh, page_offset, shape):
    return _MD.unit_mesh_view_pages(t, mesh, int(page_offset), ttnn.Shape(list(shape)))


def num_pages(t):
    """Pages of an interleaved TILE tensor (tile count; batch dims x padded tile rows x tile cols)."""
    s = list(t.padded_shape)
    n = 1
    for v in s[:-2]:
        n *= v
    return n * (s[-2] // 32) * (s[-1] // 32)


class MeshSwitch:
    """Parent <-> submesh dispatch switch: quiesce only the side being left (sub quiesces run in parallel threads;
    quiesce_devices releases the GIL).


    The parent and the submesh command queues also keep separate host-side prefetcher-cache managers over each
    device's one prefetcher cache: after the other side ran programs, an eager program re-enqueued from a stale
    manager replays clobbered cached kernel data (device hang). A read or an execute_trace resets the manager, so
    ``to(side, eager=True)`` does one tiny read per command queue of the new side; pass eager=False when the first
    op on the new side is an execute_trace."""

    def __init__(self, mesh, subs, side="parent", parallel=True):
        self.mesh, self.subs, self.side, self.parallel = mesh, subs, side, parallel
        self._tiny = {}

    def _tiny_read(self, dev):
        t = self._tiny.get(id(dev))
        if t is None:
            t = ttnn.from_torch(
                torch.zeros(1, 1, 32, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
            )
            self._tiny[id(dev)] = t
        ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))

    def to(self, side, eager=True):
        t0 = time.perf_counter()
        if side == self.side:
            return 0.0
        if self.side == "parent":
            self.mesh.quiesce_devices()
        elif self.parallel:
            ths = [threading.Thread(target=s.quiesce_devices) for s in self.subs]
            for th in ths:
                th.start()
            for th in ths:
                th.join()
        else:
            for s in self.subs:
                s.quiesce_devices()
        self.side = side
        if eager:
            for dev in [self.mesh] if side == "parent" else self.subs:
                self._tiny_read(dev)
        return time.perf_counter() - t0


class _SubMeshProxy:
    """Hands SPPrefillSC the pre-created unit submeshes (so the caller can reserve their allocators before the SP
    models are built); everything else forwards to the real mesh."""

    def __init__(self, mesh, subs):
        self._mesh, self._subs = mesh, subs

    def create_submeshes(self, shape):
        assert tuple(shape) == (1, 1)
        return list(self._subs)

    def __getattr__(self, name):
        return getattr(self._mesh, name)


def sp_mesh_proxy(mesh, subs):
    return _SubMeshProxy(mesh, subs)


# --------------------------------------------------------------------------------------------------------------
# TP model state rebinding (before any trace capture)
# --------------------------------------------------------------------------------------------------------------
def tp_bind_contiguous_gdn_state(model):
    """Rebind the TP model's per-layer GDN rec_state [1, Nv_tp, Dk, Dv] (fp32) and conv_states[1..K-1]
    [1, 1, D_tp] (bf16) to page views of two contiguous parent buffers (BIG_REC [n_gdn, Nv_tp, Dk, Dv],
    BIG_CONV [(K-1) * n_gdn, 1, D_tp]), so one transfer fills every layer. Values are copied over (zeros)."""
    mesh = model.device
    gdn = [layer.attention for layer in model.layers if not layer.is_full_attention]
    a0 = gdn[0]
    rec_shape = list(a0.rec_state.shape)
    assert rec_shape[0] == 1, rec_shape
    K = len(a0.conv_states)
    conv_shape = list(a0.conv_states[1].shape)
    assert conv_shape[:2] == [1, 1], conv_shape
    n = len(gdn)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    big_rec = ttnn.from_torch(
        torch.zeros([n] + rec_shape[1:], dtype=torch.float32),
        dtype=a0.rec_state.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    big_conv = ttnn.from_torch(
        torch.zeros([(K - 1) * n, 1, conv_shape[-1]], dtype=torch.bfloat16),
        dtype=a0.conv_states[1].dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    rp = num_pages(a0.rec_state)
    cp = num_pages(a0.conv_states[1])
    for i, dn in enumerate(gdn):
        old = dn.rec_state
        dn.rec_state = view(big_rec, mesh, i * rp, rec_shape)
        ttnn.deallocate(old)
        for m in range(1, K):
            old = dn.conv_states[m]
            dn.conv_states[m] = view(big_conv, mesh, ((K - 1) * i + (m - 1)) * cp, conv_shape)
            ttnn.deallocate(old)
    return big_rec, big_conv


def tp_conv_landing(model):
    """Parent landing buffer [1, (K-1) * n_gdn, D_tp] bf16 for the regrouped conv states (allocate in the parent
    phase, before the submesh allocations)."""
    mesh = model.device
    gdn = [layer.attention for layer in model.layers if not layer.is_full_attention]
    rows = (len(gdn[0].conv_states) - 1) * len(gdn)
    return ttnn.from_torch(
        torch.zeros(1, rows, gdn[0].qkv_dim_tp, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def tp_page_table(num_devices, p_blocks, heads, width, slot_base=128, n_slots=192):
    """Per-device page tables [num_devices, width] int32 (see module docstring)."""
    pt = torch.full((num_devices, width), n_slots - 1, dtype=torch.int32)
    for d in range(num_devices):
        p, h = p_blocks[d], heads[d]
        for i in range(width):
            s = 2 * i + h if i < p else slot_base + (i - p)
            if s < n_slots:
                pt[d, i] = s
    return pt


# --------------------------------------------------------------------------------------------------------------
# the handoff engine
# --------------------------------------------------------------------------------------------------------------
class SPTPHandoff:
    # socket pairs (sender die, receiver die); index i -> sender cores (2i, 7), (2i+1, 7), receiver (2i, 6), (2i+1, 6)
    PAIRS = [(3, 2), (3, 0), (2, 1), (1, 0), (0, 1)]

    def __init__(self, mesh, subs, sp, tp_model, *, big_rec, big_conv, conv_land, tp_caches, fifo_pages=None):
        self.mesh, self.subs, self.sp, self.tp = mesh, subs, sp, tp_model
        n = len(subs)
        assert n == 4 and sp.n_spans == 4, "SPTPHandoff: 4 dies"
        assert sp.span_len is not None and sp.total_len == 4096, "SPTPHandoff: uniform spans, T = 4096 only"
        self.nkv, self.hd = sp.nkv, sp.hd
        assert self.nkv == 2
        self.p_blocks = [(sp.span_starts[d] + sp.spans[d]) // BLOCK_SIZE for d in range(n)]  # [16, 32, 48, 64]
        self.heads = [(d * self.nkv) // n for d in range(n)]  # [0, 0, 1, 1]
        self.big_rec, self.big_conv, self.tp_caches = big_rec, big_conv, tp_caches  # tp_caches: flat [K0,V0,K1,..]
        m3 = sp.models[3]
        self.fa_idx = [li for li, layer in enumerate(m3.layers) if layer.is_full_attention]
        self.gdn_idx = [li for li, layer in enumerate(m3.layers) if not layer.is_full_attention]
        self.n_gdn = len(self.gdn_idx)
        # SP caches per die, flat [K(fa0), V(fa0), K(fa1), ...] (same order as tp_caches)
        self.sp_caches = [
            [
                t
                for li in self.fa_idx
                for t in (
                    sp.models[d].layers[li].attention.paged_kv_cache_key,
                    sp.models[d].layers[li].attention.paged_kv_cache_value,
                )
            ]
            for d in range(n)
        ]
        assert len(self.sp_caches[0]) == len(tp_caches)
        c0 = self.sp_caches[0][0]
        self.nb_sp = c0.shape[0]
        assert list(c0.shape) == [self.nb_sp, 2, BLOCK_SIZE, self.hd], c0.shape
        # TP caches may be bf16 while SP's are bf8: the KV copies / slices typecast on the source die (2x bytes).
        self.kv_cast = tp_caches[0].dtype != c0.dtype
        assert not self.kv_cast or tp_caches[0].dtype == ttnn.bfloat16, (tp_caches[0].dtype, c0.dtype)
        self.bpp = (BLOCK_SIZE // 32) * (self.hd // 32)  # pages per (block, head): 16
        self.slot_base = 2 * 64
        # GDN shapes
        dn3 = m3.layers[self.gdn_idx[0]].attention
        self.rec_full_shape = list(dn3.recurrent_state.shape)  # [1, 16, 128, 128]
        self.conv_full_shape = list(dn3.fused_conv_state.shape)  # [1, 3, 6144]
        self.nv = self.rec_full_shape[1]
        self.nv_tp = self.nv // n
        self.km1 = self.conv_full_shape[1]
        a = tp_model.args
        self.kd, self.vd = a.gdn_key_dim, a.gdn_value_dim
        self.kp, self.vp = self.kd // n, self.vd // n
        self.d_tp = 2 * self.kp + self.vp
        assert self.conv_full_shape[-1] == 2 * self.kd + self.vd
        self.conv_rows = self.km1 * self.n_gdn  # 54

        # landing (parent, allocated by the caller in the parent phase) / staging (die 2) buffers
        self.conv_land = conv_land
        assert list(conv_land.shape) == [1, self.conv_rows, self.d_tp], conv_land.shape
        self.stage_rec = ttnn.zeros(
            [self.n_gdn, self.nv_tp] + self.rec_full_shape[2:],
            dtype=big_rec.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=subs[2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.stage_conv = ttnn.zeros(
            [1, self.conv_rows, self.d_tp],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=subs[2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # per-sub views of the parent buffers
        self.v_rec = [view(big_rec, s, 0, big_rec.shape) for s in subs]
        self.v_conv = [view(big_conv, s, 0, big_conv.shape) for s in subs]
        self.v_land = [view(self.conv_land, s, 0, self.conv_land.shape) for s in subs]

        # sockets (static-dst only; FIFO = fifo_pages x 64 B, >= one request's transfers per socket)
        n_c = len(tp_caches)
        per_socket = {(3, 2): 4 + n_c, (3, 0): 2 + n_c, (2, 1): 2 + n_c, (1, 0): 2 * n_c, (0, 1): n_c}
        self.socks = {}
        for i, (e, d) in enumerate(self.PAIRS):
            pages = fifo_pages or (per_socket[(e, d)] + 2)
            self.socks[(e, d)] = _build_socket_pair(
                subs[e],
                subs[d],
                [ttnn.CoreCoord(2 * i, 7), ttnn.CoreCoord(2 * i + 1, 7)],
                [ttnn.CoreCoord(2 * i, 6), ttnn.CoreCoord(2 * i + 1, 6)],
                ttnn.BufferType.L1,
                64 * pages,
            )
        logger.info(
            f"[SPTPHandoff] p_blocks={self.p_blocks} heads={self.heads} n_caches={n_c} n_gdn={self.n_gdn} "
            f"rec {self.rec_full_shape} conv {self.conv_full_shape} -> d_tp={self.d_tp}; sockets {list(self.socks)}"
        )
        self.trace_ids = [None] * n
        self.pc_entries = None

    # ---------------------------------------------------------------------------------------------- helpers
    def _send(self, e, d, src, dst):
        """Static-dst send of src (die e) into dst's address on die d (same page layout)."""
        assert num_pages(src) == num_pages(dst), (src.shape, dst.shape)
        ttnn.experimental.send_direct_async(src, self.socks[(e, d)][0], static_dst_address=dst.buffer_address())

    def _recv(self, e, d, dst):
        ttnn.experimental.recv_direct_async(dst, self.socks[(e, d)][1], wait_only=True)

    def _tp_slots(self, c, d, slot0, nblk):
        """View of TP cache c on die d: slots [slot0, slot0 + nblk) as [nblk, 1, 64, hd]."""
        return view(self.tp_caches[c], self.subs[d], slot0 * self.bpp, [nblk, 1, BLOCK_SIZE, self.hd])

    def _tail_slot(self, d, blk):
        """TP cache slot of logical block blk (>= p_d) on die d."""
        return self.slot_base + (blk - self.p_blocks[d])

    def _sp_head_slice(self, e, c, b0, b1, h):
        t = ttnn.slice(
            self.sp_caches[e][c], (b0, h, 0, 0), (b1, h + 1, BLOCK_SIZE, self.hd), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if self.kv_cast:
            t16 = ttnn.typecast(t, self.tp_caches[c].dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(t)
            t = t16
        return t

    def _local_kv(self, d):
        """Die d: TP cache slots [0, 2 p_d) <- its SP cache blocks [0, p_d) (both heads)."""
        p = self.p_blocks[d]
        for c in range(len(self.tp_caches)):
            src = view(self.sp_caches[d][c], self.subs[d], 0, [p, 2, BLOCK_SIZE, self.hd])
            dst = view(self.tp_caches[c], self.subs[d], 0, [p, 2, BLOCK_SIZE, self.hd])
            if self.kv_cast:
                t16 = ttnn.typecast(src, dst.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.copy(t16, dst)
                ttnn.deallocate(t16)
            else:
                ttnn.copy(src, dst)

    def _conv_expand(self, d, land):
        """Die d: land [1, 3 n_gdn, D_tp] (rows (layer, m-1)) -> BIG_CONV [3 n_gdn, 1, D_tp] (one tile row each)."""
        x = ttnn.reshape(land, [self.conv_rows, 1, self.d_tp])
        ttnn.copy(x, self.v_conv[d])
        ttnn.deallocate(x)

    # ---------------------------------------------------------------------------------------------- programs
    def _prog_die3(self):
        e = 3
        sub = self.subs[e]
        m3 = self.sp.models[e]
        n_c = len(self.tp_caches)
        # GDN conv regroup: [n_gdn, 3, 6144] (q|k|v) -> [4, 3 n_gdn, 1536] (die-major [q_d|k_d|v_d])
        convs = [m3.layers[li].attention.fused_conv_state for li in self.gdn_idx]
        cat = ttnn.concat(convs, dim=0)  # [n_gdn, 3, 6144] TILE
        rm = ttnn.to_layout(cat, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(cat)
        r5 = ttnn.reshape(rm, [self.n_gdn, self.km1, 3, 4, self.kp])
        pm = ttnn.permute(r5, (3, 0, 1, 2, 4))  # [4, n_gdn, 3, 3, 512]
        ttnn.deallocate(rm)
        r3 = ttnn.reshape(pm, [4, self.conv_rows, self.d_tp])
        y = ttnn.to_layout(r3, ttnn.TILE_LAYOUT)  # [4, 54, 1536]
        ttnn.deallocate(pm)
        blk = num_pages(self.conv_land)  # 96
        yd = [view(y, sub, d * blk, [1, self.conv_rows, self.d_tp]) for d in range(4)]
        # GDN rec: [n_gdn, 16, 128, 128] -> per-die value-head slices
        recs = [m3.layers[li].attention.recurrent_state for li in self.gdn_idx]
        rcat = ttnn.concat(recs, dim=0)
        rs = [
            ttnn.slice(
                rcat,
                (0, d * self.nv_tp, 0, 0),
                (self.n_gdn, (d + 1) * self.nv_tp, self.rec_full_shape[2], self.rec_full_shape[3]),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for d in range(4)
        ]
        ttnn.deallocate(rcat)
        # die 1 first (longest chain, through die 2), then die 2, die 0
        self._send(3, 2, rs[1], self.stage_rec)
        self._send(3, 2, yd[1], self.stage_conv)
        self._send(3, 2, rs[2], self.big_rec)
        self._send(3, 2, yd[2], self.conv_land)
        self._send(3, 0, rs[0], self.big_rec)
        self._send(3, 0, yd[0], self.conv_land)
        for c in range(n_c):  # KV tails [48, 64): h0 -> die 0 (forwarded to 1), h1 -> die 2
            t0 = self._sp_head_slice(3, c, 48, 64, 0)
            self._send(3, 0, t0, self._tp_slots(c, 0, self._tail_slot(0, 48), 16))
            ttnn.deallocate(t0)
            t1 = self._sp_head_slice(3, c, 48, 64, 1)
            self._send(3, 2, t1, self._tp_slots(c, 2, self._tail_slot(2, 48), 16))
            ttnn.deallocate(t1)
        # local
        ttnn.copy(rs[3], self.v_rec[3])
        self._conv_expand(3, yd[3])
        self._local_kv(3)
        for t in rs:
            ttnn.deallocate(t)
        ttnn.deallocate(y)

    def _prog_die2(self):
        n_c = len(self.tp_caches)
        for c in range(n_c):  # h0 [32, 48) -> die 1
            t = self._sp_head_slice(2, c, 32, 48, 0)
            self._send(2, 1, t, self._tp_slots(c, 1, self._tail_slot(1, 32), 16))
            ttnn.deallocate(t)
        self._local_kv(2)
        self._recv(3, 2, self.stage_rec)
        self._send(2, 1, self.stage_rec, self.big_rec)
        self._recv(3, 2, self.stage_conv)
        self._send(2, 1, self.stage_conv, self.conv_land)
        self._recv(3, 2, self.v_rec[2])
        self._recv(3, 2, self.v_land[2])
        self._conv_expand(2, self.v_land[2])
        for c in range(n_c):
            self._recv(3, 2, self._tp_slots(c, 2, self._tail_slot(2, 48), 16))

    def _prog_die1(self):
        n_c = len(self.tp_caches)
        for c in range(n_c):  # h0 [16, 32) -> die 0
            t = self._sp_head_slice(1, c, 16, 32, 0)
            self._send(1, 0, t, self._tp_slots(c, 0, self._tail_slot(0, 16), 16))
            ttnn.deallocate(t)
        self._local_kv(1)
        for c in range(n_c):  # h0 [32, 48) from die 2, forwarded -> die 0
            land = self._tp_slots(c, 1, self._tail_slot(1, 32), 16)
            self._recv(2, 1, land)
            self._send(1, 0, land, self._tp_slots(c, 0, self._tail_slot(0, 32), 16))
        self._recv(2, 1, self.v_rec[1])
        self._recv(2, 1, self.v_land[1])
        self._conv_expand(1, self.v_land[1])
        for c in range(n_c):  # h0 [48, 64) from die 0
            self._recv(0, 1, self._tp_slots(c, 1, self._tail_slot(1, 48), 16))

    def _prog_die0(self):
        n_c = len(self.tp_caches)
        self._local_kv(0)
        for c in range(n_c):
            self._recv(1, 0, self._tp_slots(c, 0, self._tail_slot(0, 16), 16))
        self._recv(3, 0, self.v_rec[0])
        self._recv(3, 0, self.v_land[0])
        for c in range(n_c):  # h0 [48, 64) from die 3, forwarded -> die 1
            land = self._tp_slots(c, 0, self._tail_slot(0, 48), 16)
            self._recv(3, 0, land)
            self._send(0, 1, land, self._tp_slots(c, 1, self._tail_slot(1, 48), 16))
        for c in range(n_c):
            self._recv(1, 0, self._tp_slots(c, 0, self._tail_slot(0, 32), 16))
        self._conv_expand(0, self.v_land[0])

    def _programs(self, dies):
        progs = {0: self._prog_die0, 1: self._prog_die1, 2: self._prog_die2, 3: self._prog_die3}
        for d in dies:
            progs[d]()

    # ---------------------------------------------------------------------------------------------- run
    def run_eager(self):
        """Eager pass (compiles every program; valid handoff when run after a prefill). Synchronizes."""
        self._programs([0, 1, 2, 3])
        for s in self.subs:
            ttnn.synchronize_device(s)

    def capture(self):
        """One trace per submesh (after an eager run compiled everything; asserts no compile)."""
        n0 = [s.num_program_cache_entries() for s in self.subs]
        opened = {}
        try:
            for d, s in enumerate(self.subs):
                opened[d] = ttnn.begin_trace_capture(s, cq_id=0)
            self._programs([0, 1, 2, 3])
            for d, s in enumerate(self.subs):
                ttnn.end_trace_capture(s, opened[d], cq_id=0)
                self.trace_ids[d] = opened.pop(d)
        finally:
            for d, tid in opened.items():
                try:
                    ttnn.end_trace_capture(self.subs[d], tid, cq_id=0)
                except Exception as ex:  # pragma: no cover
                    logger.warning(f"[SPTPHandoff] could not close trace on die {d}: {ex!r}")
                self.trace_ids[d] = tid
        n1 = [s.num_program_cache_entries() for s in self.subs]
        assert n1 == n0, f"SPTPHandoff.capture compiled: program cache entries {n0} -> {n1}"
        self.pc_entries = n0

    def execute(self, dies):
        for d in dies:
            ttnn.execute_trace(self.subs[d], self.trace_ids[d], cq_id=0, blocking=False)

    def check_no_compile(self):
        n = [s.num_program_cache_entries() for s in self.subs]
        assert n == self.pc_entries, f"SPTPHandoff: program cache entries {self.pc_entries} -> {n}"

    def release(self):
        for d, tid in enumerate(self.trace_ids):
            if tid is not None:
                ttnn.release_trace(self.subs[d], tid)
        self.trace_ids = [None] * len(self.subs)
