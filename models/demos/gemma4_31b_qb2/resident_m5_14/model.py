# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident Gemma-4-31B decode prototype (step 2: real attention for one user): one program per chip per decode step
of M rows (1 <= M <= 8: plain decode or a user's verify rows at positions p .. p + M - 1).

Per chip (TP4 on the 1 x 4 mesh), adapted from changh95/qwen3.8-27b-p300x2 code/qwen36_resident/model.py
(Apache-2.0):
  - 32 streamer cores (4 next to each DRAM bank) stream their column slices of every layer's weights through a weight
    ring in L1 and run the projections (M <= 8: custom_mm on M x 32 tiles; M > 8: matmul on 32 x 32 tiles), the
    GELU-tanh gating and the LM head, and scan their logits for every row's argmax;
  - K slice cores own the residual stream by hidden columns (sandwich norms, residual, layer scalar, the next
    pre-norm, exchanging per-row partial sums of squares) and multicast their columns of each layer half's input h;
  - one hub core carries every round's partials over fabric to the peer chips' slices, reduces the per-row argmax
    across streamers and chips, and applies the serving feed mode (tokens / positions / status, engine.py);
  - 32 attention cores (attention.py, kernels/attn_*): the streamers' q|k|v columns go to 4 group leaders (one per
    sliding kv head; leader 0 for the full layers), which norm / rope them, write the new K / V rows into the ttnn
    paged BF8 cache and, with their workers, attend over it; the O rows go back into every streamer's o_in.
The slices can export the residual after chosen layers (taps: DFlash conditioning, per-layer checks).
M5 (phase 1, bench/resident/m5/M5_FOUNDATION.md): the M rows are M / block user blocks of `block` rows; the hub
validates, flags and feeds each user's block, the attention serves user 0's block with per-unit table reads.
"""
import math
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.gemma4_31b_qb2.resident_m5_14 import PKG_DIR
from models.demos.gemma4_31b_qb2.resident_m5_14.attention import Attention, norm_tiles
from models.demos.gemma4_31b_qb2.resident_m5_14.k2a import K2aAttention
from models.demos.gemma4_31b_qb2.resident_m5_14.common import (
    E_HEAD,
    ENTRIES,
    EPS,
    HIDDEN,
    HT,
    IT,
    N_CHIPS,
    PER_BANK,
    Q_F,
    Q_S,
    RING_BYTES,
    TILE,
    TILE_BYTES,
    VOCAB,
    block_geometry,
    entry_core_cols,
    f32_bits,
    is_full,
    rect_cover,
    streamer_cores,
    seam_split,
)
from models.demos.gemma4_31b_qb2.resident_m5_14.weights import load_host

KDIR = f"{PKG_DIR}/kernels/"
SEM_SLOTS, SEM_O, SEM_ACT, SEM_GATHER, SEM_REC, SEM_EXCH, SEM_HADDR, SEM_STAGE, SEM_SLOTADDR, SEM_RXADDR, SEM_CTRL = range(11)
SEM_MCGO, SEM_MCDONE, SEM_MCSRC = 11, 12, 13  # (M2b F2) the slices' second h multicaster (slice_mc.cpp)
SLICE_MC2 = os.environ.get("GRES_SLICE_MC2", "1") == "1"  # (M2b F2) split the slices' h multicast over two NoCs
# (M2b F10 / F12) slice views per DEST pass and per h push / multicast chunk (8: all of DEST, fp32 in full sync)
SLICE_CHUNK = int(os.environ.get("GRES_SLICE_CHUNK", "8"))
ISO = os.environ.get("GRES_ISO", "1") == "1"  # (m5_11, D43) the slices' norm isolates rows from non-finite co-tenants
ISO_SKIPFOLD = os.environ.get("GRES_ISO_SKIPFOLD", "0") == "1"  # (m5_11) skip the R = 32 fold (not bitwise vs m5_9)
SEM_L1_SMALL = os.environ.get("GRES_SEM_L1_SMALL", "1") == "1"  # (m5_13) global semaphores in L1_SMALL (0: m5_11's L1)
OT_S, OT_F = Q_S // TILE, Q_F // TILE  # 64, 128
TS_WORDS = 24  # streamer timeline words per layer (M4: the writer's [8, 24), marks 6 .. 9 split matmul / gather)
HUB_TS, SLICE_TS = 2, 8  # timeline words per round (M5 D7 L0: the slices' 8, slice_dm.cpp)
OUT_WORDS = 80


def tile_rows(m):
    """R: the rows of the activation tiles for M rows"""
    return next(r for r in (1, 2, 4, 8, 32) if r >= m)


def tile_order(mat):
    """[R, N] -> the elements in device order of R x 32 tiles laid end to end (R = 32: 32 x 32 tiles)"""
    R, N = mat.shape
    if R == 32:
        return mat.reshape(2, 16, N // 32, 2, 16).permute(2, 0, 3, 1, 4).reshape(-1)
    return mat.reshape(R, N // 32, 2, 16).permute(1, 2, 0, 3).reshape(-1)


def tile_unorder(flat, R):
    N = flat.numel() // R
    if R == 32:
        return flat.reshape(N // 32, 2, 2, 16, 16).permute(1, 3, 0, 2, 4).reshape(R, N)
    return flat.reshape(N // 32, 2, R, 16).permute(2, 0, 1, 3).reshape(R, N)


class ResidentGemma:
    def __init__(
        self,
        mesh,
        cache,
        serve,
        layers=60,
        rows=1,
        slices=None,
        timeline=False,
        dump_rounds=False,
        links=int(os.environ.get("GRES_LINKS", "2")),
        packet_bytes=int(os.environ.get("GRES_PACKET_BYTES", "8192")),  # M3 (F0): 8 KB, bitwise, R = 8 -0.42 ms
        ring_bytes=RING_BYTES,
        embed_rows=None,
        fidelity=ttnn.MathFidelity.HiFi2,
        taps=(),
        shared=None,
        attn_fidelity=(ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2),
        l1_top=int(os.environ.get("GRES_L1_TOP", "1400000")),
        acc_fp32=True,
        block=None,
        stub_attention=False,
        layer_offset=0,
        bf8_partials=False,
        k2a_r8=False,
    ):
        """fidelity: the streamers' program fidelity: the BF8 LM head (and the R = 32 projections) run at it, the
        R <= 8 projections run custom_mm (LoFi only). serve: the serving engine's I/O (engine.ServeIO: the feed,
        table, logits, status and taps tensors, the KV pools and RoPE tables, capacity / max_users / groups); taps:
        layers whose output
        residual each step exports (taps()); shared: an engine whose weights this one reuses (one copy on device);
        l1_top: where the streamers' CBs must end (below L1 buffers another model in the process keeps); acc_fp32: the
        streamers' DEST accumulation (False: bf16, as the ttnn decode matmuls' fp32_dest_acc_en=False); block (M5): rows
        per user block (default: all M rows one user's block, as before)"""
        self.mesh, self.layers, self.cache = mesh, layers, cache
        self.serve = serve
        self.taps_layers = sorted(set(taps))
        assert all(0 <= l < layers for l in self.taps_layers)
        n = self.n = mesh.get_num_devices()
        assert n == N_CHIPS
        banks = self.banks = mesh.dram_grid_size().x
        self.rounds = rounds = 2 * layers
        self.timeline, self.dump_rounds = timeline, dump_rounds
        M = self.M = rows
        R = self.R = tile_rows(M)
        # (M4 step 0) stub_attention: no attention cores; the streamers' writers gather their q columns into every
        # streamer's o_in (o = q, as step 1's stub), so the dense path runs up to 32 rows
        self.stub = stub_attention
        assert M <= 8 or stub_attention or R == 32, "step 2: up to 8 rows; R = 32: K2a or the M4 stub"
        Mb = self.Mb = block or M  # (M5: user blocks of Mb rows)
        assert 1 <= Mb <= M and M % Mb == 0, (M, Mb)
        TB = self.TB = 64 * R
        # (M5 D6) R = 32: the round partials (streamers' cb_pout -> the slices' slots and the hub's staging -> fabric)
        # as BF8 32x32 tiles (1,088 B) instead of bf16 (2,048 B): 47% fewer bytes per round; not bitwise (the 4-chip
        # sum adds BF8-rounded partials); row-independent (one exponent per face row), so Q2 holds
        self.bf8p = bf8p = bool(bf8_partials) and R == 32
        PTB = self.PTB = 1088 if bf8p else TB
        K = self.K = slices or (1 if R == 1 else 2 if R <= 8 else 4)  # the fastest measured (M = 8: K 2, M = 32: K 4)
        assert banks % K == 0
        T = self.T = HT // K
        V = self.V = -(-T * R // 32)
        P = -(-T * PTB // packet_bytes)

        # ---- cores: streamers, the hub and K slices (free cores in the streamers' bounding box)
        groups, used = streamer_cores(mesh)
        cores = [c for grp in groups for c in grp]  # streamer i = bank * PER_BANK + j
        S = self.S = len(cores)
        xs, ys = [c.x for c in cores], [c.y for c in cores]
        free = [
            ttnn.CoreCoord(x, y)
            for y in range(min(ys), max(ys) + 1)
            for x in range(min(xs), max(xs) + 1)
            if (x, y) not in used
        ]
        hub, slices_c = free[0], free[1 : 1 + K]
        C = lambda cs: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cs])
        grid, hub_grid, slice_grid = C(cores), C([hub]), C(slices_c)
        all_grid = grid.merge(hub_grid).merge(slice_grid)
        phys = lambda c: mesh.worker_core_from_logical_core(c)
        sphys, hub_p, slice_p = [phys(c) for c in cores], phys(hub), [phys(c) for c in slices_c]
        self.cores, self.hub, self.slices = cores, hub, slices_c
        logger.info(f"resident: M {M} (tiles of {R} rows), {S} streamers, hub {hub}, {K} slices of {T} tiles")
        taken = used | {(hub.x, hub.y)} | {(c.x, c.y) for c in slices_c}
        # (M5 K2a) R = 32: U = 32 / Mb users on head + unit cores (k2a.py); R <= 8: the step-2 / M1L attention
        # (M5 D16, opt-in k2a_r8) and R = 8 with blocks of M < 8 rows (multi-user R = 8; a block of 8 rows is one user:
        # M1L as before; without k2a_r8 every R = 8 program is m5_6's M1L, bitwise)
        self.k2a = (R == 32 or (R == 8 and Mb < 8 and k2a_r8)) and not stub_attention
        if self.k2a:
            # (K2b step 3) one DRAM partials buffer per engine: kept on the weight owner, reused by every K2a model
            holder = shared if shared is not None else self
            att = self.att = K2aAttention(mesh, Mb, layers, serve.capacity, taken, S, serve.layer_group, timeline=timeline,
                                          R=R, parts_dram=getattr(holder, "_k2b_parts", None))
            holder._k2b_parts = att.parts_dram
        else:
            att = self.att = None if stub_attention else Attention(
                mesh, R, Mb, layers, serve.capacity, taken, S, serve.layer_group, users=M // Mb, timeline=timeline,
            )
        if self.k2a:
            logger.info(f"attention: K2a, {Mb} rows x {32 // Mb} users, heads {[(c.x, c.y) for c in att.heads]}, L1 {att.l1_bytes()}")
        elif att is not None:
            logger.info(f"attention: groups {[[(c.x, c.y) for c in g][:2] for g in att.cores]}, L1 per role {att.l1_bytes()}, NVf {att.NVf}")
        else:
            logger.info("attention: STUB (o = q; M4 step 0)")

        rep = ttnn.ReplicateTensorToMesh(mesh)
        dram = ttnn.DRAM_MEMORY_CONFIG
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})

        def settled(t):
            ttnn.synchronize_device(mesh)
            return t

        def l1(t, grid_, shard, dtype=ttnn.bfloat16, tile=None, layout=ttnn.TILE_LAYOUT):
            return settled(
                ttnn.from_torch(
                    t,
                    dtype=dtype,
                    layout=layout,
                    device=mesh,
                    tile=(tile or ttnn.Tile([R, TILE])) if layout == ttnn.TILE_LAYOUT else None,
                    mesh_mapper=rep,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                        ttnn.BufferType.L1,
                        ttnn.ShardSpec(grid_, shard, ttnn.ShardOrientation.ROW_MAJOR),
                    ),
                )
            )

        u32 = lambda grid_, ncores, words: l1(
            torch.zeros(ncores, words, dtype=torch.int32), grid_, [1, words], ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )

        def dram_t(t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
            return settled(ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, memory_config=dram, mesh_mapper=rep))

        # ---- column assignment per core (bank-major, then over PER_BANK cores; see common.entry_core_cols)
        cols = [entry_core_cols(e, banks) for e in range(len(ENTRIES))]
        per_core = lambda e, i: cols[e][i // PER_BANK][i % PER_BANK]
        self.per_core = per_core
        nq_max = max(len(per_core(e, i)) for e in (0, 1) for i in range(S))
        nd_max = max(len(per_core(5, i)) for i in range(S))
        ng_max = max(len(per_core(4, i)) // 2 for i in range(S))
        nv_max = max(len(per_core(E_HEAD, i)) for i in range(S))
        kblk = (32 // R) * -(-max(nq_max, ng_max) * R // 32)
        kact = max(IT, OT_F, nv_max)
        self.nv_max = nv_max

        # ---- DRAM weights: per entry the copies back to back (fixed per-bank stride)
        kinds = [is_full(l) for l in range(layers)]
        layer_of = [
            [l for l in range(layers) if not kinds[l]],
            [l for l in range(layers) if kinds[l]],
            [l for l in range(layers) if not kinds[l]],
            [l for l in range(layers) if kinds[l]],
            list(range(layers)),
            list(range(layers)),
            [0],
        ]
        self.geo = [block_geometry(Kt, TILE_BYTES[dt]) for _, Kt, dt, _ in ENTRIES]
        self.w_entries, self._keep = [], []
        self.weight_bytes = 0  # real (unpadded) weight bytes streamed per step per chip
        t0 = time.time()
        for e, (name, Kt, dt, N) in enumerate(ENTRIES if shared is None else []):
            if not layer_of[e]:
                self.w_entries.append((None, 0, 1))
                continue
            width = PER_BANK * max(len(c) for bank in cols[e] for c in bank)
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram_grid, [Kt * TILE, width * TILE], ttnn.ShardOrientation.ROW_MAJOR),
            )
            ts = []
            for l in layer_of[e]:
                host = ttnn.from_host_shards(load_host(cache, l, e), ttnn.MeshShape(1, n))
                ts.append(settled(ttnn.to_device(host, mesh, memory_config=mc)))
                self.weight_bytes += Kt * N * TILE_BYTES[dt]
            addrs = [t.buffer_address() for t in ts]
            stride = (addrs[1] - addrs[0]) & 0xFFFFFFFF if len(addrs) > 1 else 0
            assert all(((b - a) & 0xFFFFFFFF) == stride for a, b in zip(addrs, addrs[1:])), (name, addrs[:4])
            self._keep += ts
            self.w_entries.append((ts[0], stride, len(ts)))
        if shared is not None:
            assert shared.layers == layers and shared.banks == banks
            self.w_entries, self._keep, self.weight_bytes = shared.w_entries, shared._keep, shared.weight_bytes
        logger.info(f"weights on device: {self.weight_bytes / 1e9:.2f} GB / chip, {time.time() - t0:.0f} s")

        # the activation buffers grow with R: the weight ring gives up their L1
        act_l1 = TB * (HT + kact + 4 * kblk + nd_max) + (layers * TS_WORDS * 4 if timeline else 0)
        lcm = math.lcm(*TILE_BYTES.values())
        # l1_top is an L1 address: CBs are laid out from the allocator's L1 base (step 3: absolute, so cb_top() and
        # the serving L1 guard agree; step 2 treated it as a size, about 112 KB higher)
        l1_base = ttnn.get_allocator_base_address(mesh, ttnn.BufferType.L1)
        ring_bytes = min(ring_bytes, l1_top - l1_base - act_l1 - 1024)
        self.ring_bytes = ring_bytes = (ring_bytes // lcm) * lcm
        logger.info(f"weight ring {ring_bytes} B per streamer, activations {act_l1} B")

        # ---- L1 tensors (addresses the same on every core of their grid and on every chip)
        # (the big buffers are CBs, each role's laid out alike on its cores: L1 tensors take their address range on
        # every core, CBs only on their own; the kernels publish the addresses they need at launch)
        self.ts_t = u32(grid, S, layers * TS_WORDS) if timeline else None
        # (M5 D14) the hub's records / candidates scratch: one pair per engine, sized for R = 32 and shared by every
        # program (per-launch scratch, one launch at a time; the L1 allocator reserves the range on every core, so a
        # program whose hub sits elsewhere uses the same address safely). m5_4 kept one pair per program: ~9.4 KB of
        # L1 floor each (S1_REQUEST_M5_KERNELS_1)
        rw = -(-32 * 8 // 16) * 4
        if shared is not None and getattr(shared, "rec_t", None) is not None:
            self.rec_t, self.cand_t = shared.rec_t, shared.cand_t
        else:
            self.rec_t = u32(hub_grid, 1, S * rw)
            self.cand_t = u32(hub_grid, 1, n * rw)
        self.hub_ts_t = u32(hub_grid, 1, 4 + HUB_TS * rounds) if timeline else None
        self.slice_ts_t = u32(slice_grid, K, 2 + SLICE_TS * rounds) if timeline else None

        # ---- DRAM tensors
        self.out_t = dram_t(torch.zeros(1, OUT_WORDS, dtype=torch.int32), ttnn.uint32)
        # [0] final h, [1 + r] round r; (M5 K1, dump_rounds) [2 + rounds + l] x after layer l, raw (the taps' source)
        self.dump_rows_n = 2 + rounds + (layers if dump_rounds else 0)
        self.dump_t = dram_t(torch.zeros(self.dump_rows_n, K * V * 1024), ttnn.bfloat16)
        self.logits_t = dram_t(torch.zeros(S, nv_max * TB // 2), ttnn.bfloat16) if dump_rounds else None
        vectors = self.vectors = torch.load(f"{cache}/vectors.pt")
        self.wpost_t = dram_t(self._post_rows(vectors), ttnn.bfloat16)
        # raw (unfolded) weights: the slices apply each round's pre-norm weight, as the ttnn path's rms_norm(x, w)
        self.pre_weighted = not vectors.get("folded", True)
        self.wpre_t = dram_t(self._pre_rows(vectors), ttnn.bfloat16) if self.pre_weighted else None
        logger.info(f"weights: {'raw (pre-norm weights on the slices)' if self.pre_weighted else 'folded norms'}")
        if shared is None:
            emb = torch.load(f"{cache}/embed_scaled.pt", mmap=True)
            self.emb_host = emb
            # (M5 E) with the engine's x0 (the ttnn embedding chain) there is no resident table: -2.82 GB per chip
            self.embed_t = None if serve.x0 is not None else dram_t(
                emb if embed_rows is None else emb[:embed_rows].clone(), ttnn.bfloat16)
            self.attn_w_t = settled(
                ttnn.from_torch(norm_tiles(layers), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                memory_config=dram, mesh_mapper=rep)
            )
        else:
            self.emb_host, self.embed_t, self.attn_w_t = shared.emb_host, shared.embed_t, shared.attn_w_t
        if self.taps_layers:
            assert list(self.taps_layers) == list(serve.tap_layers) and len(serve.taps) == 2, "taps: the engine's buffers"
        self.odump_t = dram_t(torch.zeros(layers, 128 * TB // 2), ttnn.bfloat16) if dump_rounds else None
        ptmax = max(att.PTs, att.PTf) if att is not None and not self.k2a else 1
        self.pdump_t = (
            dram_t(torch.zeros(layers * 32, ptmax * 1024), ttnn.bfloat16) if os.environ.get("GRES_PDUMP") else None
        )
        logger.info(f"tensors on device, {time.time() - t0:.0f} s")

        # (m5_13, M7: LEAD_DECISIONS_M7 M5, bench/resident/m7/L1_FLOOR_ANSWER.md) the global semaphores live in L1_SMALL
        # (the region l1_small_size reserves at the top of L1, outside the main heap), so they no longer lower the
        # process-wide L1 buffer floor (64 B each, 5 per block model); the kernels see the same kind of L1 address
        sem_bt = ttnn.BufferType.L1_SMALL if SEM_L1_SMALL else ttnn.BufferType.L1
        self.hub_sems = [ttnn.create_global_semaphore(mesh, hub_grid, 0, sem_bt) for _ in range(2)]  # cand, launch
        self.slice_sems = [ttnn.create_global_semaphore(mesh, slice_grid, 0, sem_bt) for _ in range(3)]  # slot 0/1, launch
        hsem = [ttnn.get_global_semaphore_address(s) for s in self.hub_sems]
        ssem = [ttnn.get_global_semaphore_address(s) for s in self.slice_sems]

        # ---- fabric links
        self.links = 1
        if links > 1:
            node = lambda c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, c))  # noqa: E731
            trained = min(
                len(ttnn.get_forwarding_link_indices(mesh, node(c), node(c + s)))
                for c in range(n)
                for s in (1, -1)
                if 0 <= c + s < n
            )
            self.links = max(1, min(links, trained))

        # ---- programs
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        NCRISC, BRISC = ttnn.DataMovementProcessor.RISCV_1, ttnn.DataMovementProcessor.RISCV_0
        NOC0, NOC1 = ttnn.NOC.RISCV_0_default, ttnn.NOC.RISCV_1_default

        def compute_cfg(fid, fp32, full_sync, unpack_fp32=()):
            c = ttnn.ComputeConfigDescriptor()
            c.math_fidelity = fid
            c.fp32_dest_acc_en = fp32
            c.dst_full_sync_en = full_sync
            if unpack_fp32:
                modes = type(c.unpack_to_dest_mode)([ttnn.UnpackToDestMode.Default] * 64)
                for i in unpack_fp32:
                    modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
                c.unpack_to_dest_mode = modes
            return c

        def cb(idx, grid_, pages, dt=ttnn.bfloat16, page=2048, rows=None):
            fd = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)
            if rows is not None:
                fd = ttnn.CBFormatDescriptor(
                    buffer_index=idx, data_format=dt, page_size=page, tile=ttnn.TileDescriptor(rows, TILE)
                )
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=grid_, format_descriptors=[fd])

        def aliased(tile_idx, full_idx, tiles):
            return ttnn.CBDescriptor(
                total_size=tiles * TB,
                core_ranges=grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=tile_idx, data_format=ttnn.bfloat16, page_size=TB, tile=ttnn.TileDescriptor(R, TILE)
                    ),
                    ttnn.CBFormatDescriptor(buffer_index=full_idx, data_format=ttnn.bfloat16, page_size=2048),
                ],
            )

        ring_fmt = [
            ttnn.CBFormatDescriptor(
                buffer_index=e, data_format=ttnn.bfloat4_b if dt == "bf4" else ttnn.bfloat8_b, page_size=TILE_BYTES[dt]
            )
            for e, (_, _, dt, _) in enumerate(ENTRIES)
        ]
        hdrs = 2 * K * P * 2 + 2
        f32 = (ttnn.float32, 4096)
        tile_fmt = lambda idx: ttnn.CBFormatDescriptor(
            buffer_index=idx, data_format=ttnn.bfloat16, page_size=TB, tile=ttnn.TileDescriptor(R, TILE)
        )
        cbs = [
            ttnn.CBDescriptor(total_size=ring_bytes, core_ranges=grid, format_descriptors=ring_fmt),
            cb(7, grid, 1, ttnn.uint32, 16),
            cb(8, grid, HT, page=TB, rows=R),
            cb(9, grid, kblk, page=TB, rows=R),
            # o_in | act | logits: one buffer (o_in is consumed before any act lands, act before the next o_in)
            ttnn.CBDescriptor(total_size=kact * TB, core_ranges=grid, format_descriptors=[tile_fmt(i) for i in (10, 18, 19)]),
            cb(11, grid, nd_max, ttnn.bfloat8_b if bf8p else ttnn.bfloat16, page=PTB, rows=R),  # (D6)
            aliased(12, 13, kblk),
            aliased(14, 15, kblk),
            aliased(16, 17, kblk),
            # hub
            cb(0, hub_grid, 2 * HT, ttnn.bfloat8_b if bf8p else ttnn.bfloat16, page=PTB, rows=R),  # (D6)
            cb(6, hub_grid, 1, ttnn.uint32, 512),
            cb(7, hub_grid, 1, ttnn.uint32, hdrs * 160),
            # slices
            cb(0, slice_grid, 2 * n * V, ttnn.bfloat8_b if bf8p else ttnn.bfloat16, page=PTB if bf8p else 2048),
            cb(1, slice_grid, 2 * V),
            cb(2, slice_grid, 2 * V),
        ] + ([
            # (M5 K0b, dflash-builder's m4x0 at R = 32 only) y | x0: one buffer (x0 is the seed's only, consumed into x
            # before the seed's h goes out; y is first packed by round 1's slot sum, which needs the slots that h
            # produces; the slots CB would not do: a peer chip's round 1 partials need only its own seed). It brings the
            # K = 4 slice top under floor - 32 KB; R <= 8 keeps m3_3's separate CBs (layout unchanged)
            ttnn.CBDescriptor(
                total_size=V * 2048,
                core_ranges=slice_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048) for i in (3, 5)
                ],
            ),
            cb(4, slice_grid, V),
        ] if R == 32 else [
            cb(3, slice_grid, V),
            cb(4, slice_grid, V),
            cb(5, slice_grid, V),
        ]) + [
            cb(6, slice_grid, 3),
            cb(7, slice_grid, 1, *f32),
            cb(8, slice_grid, 2 * K, *f32),
            cb(9, slice_grid, 1, *f32),
            cb(10, slice_grid, 2, *f32),
            cb(11, slice_grid, V),
        ] + (att.cbs() if att is not None else [])
        # each semaphore only on the cores that own it (the hub's fabric connections take free ids)
        owners = {
            SEM_SLOTS: grid,
            SEM_O: grid,
            SEM_ACT: grid,
            SEM_GATHER: hub_grid,
            SEM_REC: hub_grid,
            SEM_EXCH: slice_grid,
            SEM_HADDR: slice_grid,
            SEM_STAGE: grid,
            SEM_SLOTADDR: grid.merge(hub_grid),
            SEM_RXADDR: grid,
            SEM_CTRL: grid,
            SEM_MCGO: slice_grid,
            SEM_MCDONE: slice_grid,
            SEM_MCSRC: slice_grid,
        }
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=g, initial_value=0) for i, g in owners.items()]
        sems += att.semaphores() if att is not None else []  # ids 0..5 on the attention cores

        ct = [layers, S, HT, IT, 0, ring_bytes, ng_max, nd_max, nq_max, nv_max, kblk, kact]
        ct += [SEM_SLOTS, SEM_O, SEM_ACT, OT_S, OT_F, R, M, K]
        for (_, Kt, _, _), (sb, pages, page, block) in zip(ENTRIES, self.geo):
            ct += [Kt, sb, pages, page, block]
        ct.append(int(stub_attention))  # (M4) 20 + 5 kEntries: the o = q stub
        # (M4 teacher forcing, stub only) 21 + 5 kEntries: program layer l is real layer l + layer_offset (the caller also
        # shifts this module's is_full and points the cache at those layers)
        assert layer_offset == 0 or stub_attention
        ct.append(int(layer_offset))
        ct.append(int(self.k2a))  # (M5 K2a) 22 + 5 kEntries: q|k|v to the 8 head cores, O from all 8
        ct.append(int(bf8p))  # (M5 D6) 23 + 5 kEntries: BF8 round partials
        ts_addr = self.ts_t.buffer_address() if timeline else 0
        hub_ts = self.hub_ts_t.buffer_address() if timeline else 0
        slice_ts = self.slice_ts_t.buffer_address() if timeline else 0
        layer_scalars = [f32_bits(vectors["scalar"][l]) for l in range(layers)]
        tap_mask = sum(1 << l for l in self.taps_layers)
        rope_addrs = [t.buffer_address() for kind in ("sliding_attention", "full_attention") for t in serve.rope[kind]]
        kv_addrs = [(k.buffer_address(), v.buffer_address()) for k, v in serve.kv[:layers]]
        slot_bytes = V * (PTB if bf8p else 2048)  # (D6: R = 32 views are one tile each)
        banks_per_slice = banks // K

        mesh_pd = ttnn.MeshProgramDescriptor()
        self.programs = []
        peers = [v for p in sphys for v in (p.x, p.y)]
        slice_xy = [v for p in slice_p for v in (p.x, p.y)]
        rects = rect_cover(mesh, cores, seam_split(self.M))
        kernel = lambda src, cr_, ct_, rt_, cfg: ttnn.KernelDescriptor(
            kernel_source=KDIR + src,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cr_,
            compile_time_args=ct_,
            runtime_args=rt_,
            config=cfg,
        )
        for chip in range(n):
            coord = ttnn.MeshCoordinate(0, chip)
            sct = list(ct)
            sct[4] = chip
            rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for i, c in enumerate(cores):
                bank, j = divmod(i, PER_BANK)
                ent = []
                for e, (t, stride, copies) in enumerate(self.w_entries):
                    ent += [len(per_core(e, i)) if t is not None else 0, t.buffer_address() if t else 0, stride, copies]
                qs, qf, o, g, d, v = (per_core(e, i) for e in (0, 1, 2, 4, 5, 6))
                ng = len(g) // 2
                assert o == d
                first_index = chip * (VOCAB // n) + (v[0] * TILE if v else 0)
                rr[c.x][c.y] = [bank, i & 0x3, j, PER_BANK] + ent + [ts_addr] + [
                    hub_p.x,
                    hub_p.y,
                    self.rec_t.buffer_address(),
                    SEM_REC,
                    len(v),
                    first_index,
                    i,
                    serve.control.buffer_address() if serve.logits_raw is not None else 0,
                    SEM_CTRL,
                ]
                k = bank // banks_per_slice
                wr[c.x][c.y] = [
                    ng,
                    g[0] if ng else 0,
                    len(d),
                    d[0] if d else 0,
                    hub_p.x,
                    hub_p.y,
                    SEM_RXADDR,
                    self.rec_t.buffer_address(),
                    SEM_GATHER,
                    SEM_REC,
                    len(qs),
                    qs[0] if qs else 0,
                    len(qf),
                    qf[0] if qf else 0,
                    serve.logits_raw.buffer_address() if serve.logits_raw is not None else 0,
                    len(v),
                    first_index,
                    i,
                    ts_addr,
                    slice_p[k].x,
                    slice_p[k].y,
                    SEM_STAGE,
                    ssem[0],
                    ssem[1],
                    (d[0] if d else 0) - k * T,
                    slot_bytes,
                    SEM_SLOTADDR,
                    SEM_HADDR,
                    self.logits_t.buffer_address() if self.logits_t is not None else 0,
                    SEM_CTRL,
                ] + peers + slice_xy + ([att.xy(c_) for c_ in att.leaders] if att is not None else [0] * 4)
                cr[c.x][c.y] = [ng, len(d), len(qs), len(qf), len(v)]
            hub_rt = ttnn.RuntimeArgs()
            hub_rt[hub.x][hub.y] = [
                SEM_STAGE,
                SEM_SLOTADDR,
                ssem[0],
                ssem[1],
                slot_bytes,
                self.rec_t.buffer_address(),
                self.cand_t.buffer_address(),
                hsem[0],
                hsem[1],
                serve.tokens.buffer_address(),
                serve.positions.buffer_address(),
                serve.slots.buffer_address(),
                self.out_t.buffer_address(),
                serve.control.buffer_address(),
                serve.status.buffer_address(),
                att.capacity if self.k2a else serve.capacity,  # (K2a: the R = 32 program's capacity, k2a.MAX_POS)
                serve.max_users,
                serve.argmax_ids.buffer_address(),  # (M5 A4)
                hub_ts,
            ] + rects + slice_xy + ([att.xy(c_) for c_ in att.flat] if att is not None else [])
            hub_ct = [n, chip, R, M, HT, K, T, packet_bytes, rounds, S, SEM_GATHER, SEM_REC, self.links,
                      len(att.flat) if att is not None else 0, Mb, int(self.k2a), int(bf8p)]  # (D6: 16)
            sdm_rts, sc_rt = [], ttnn.RuntimeArgs()
            for k, c in enumerate(slices_c):
                sdm_rt = ttnn.RuntimeArgs()
                sdm_rts.append(sdm_rt)
                sdm_rt[c.x][c.y] = [
                    SEM_HADDR,
                    ssem[0],
                    ssem[1],
                    ssem[2],
                    serve.tokens.buffer_address(),
                    self.wpost_t.buffer_address(),
                    SEM_SLOTADDR,
                    hub_p.x | (hub_p.y << 16),
                    self.dump_t.buffer_address(),
                    slice_ts,
                    (serve.x0 if serve.x0 is not None else self.embed_t).buffer_address(),  # (M5 E: x0 tiles)
                    serve.status.buffer_address(),
                ] + rects + slice_xy + [
                    serve.taps[0].buffer_address() if self.taps_layers else 0,
                    serve.taps[1].buffer_address() if self.taps_layers else 0,
                    tap_mask & 0xFFFFFFFF,
                    tap_mask >> 32,
                ] + [
                    self.wpre_t.buffer_address() if self.wpre_t is not None else 0
                ]
                sc_rt[c.x][c.y] = layer_scalars
            kernels = [
                kernel("streamer_reader.cpp", grid, sct, rr, dm(NCRISC, NOC0)),
                kernel("streamer_writer.cpp", grid, sct, wr, dm(BRISC, NOC1)),
                kernel("streamer_compute.cpp", grid, sct, cr, compute_cfg(fidelity, acc_fp32, True)),
                kernel("hub_dm.cpp", hub_grid, hub_ct, hub_rt, dm(BRISC, NOC0)),
            ]
            for k, c in enumerate(slices_c):
                sc_ct = [n, chip, k, K, R, T, V, rounds, PER_BANK * banks_per_slice, P, SEM_SLOTS, SEM_EXCH]
                sc_ct += [int(dump_rounds), int(self.pre_weighted), M]
                nrect = rects[0]
                mc_split = (nrect + 1) // 2 if SLICE_MC2 and nrect >= 2 else 0
                sc_ct += [mc_split, SEM_MCGO, SEM_MCDONE, SEM_MCSRC, SLICE_CHUNK]
                sc_ct += [int(serve.x0 is not None)]  # (M5 E) 20: x0 from the engine's TILE x0
                sc_ct += [int(bf8p)]  # (M5 D6) 21: BF8 slots
                kernels.append(kernel("slice_dm.cpp", C([c]), sc_ct, sdm_rts[k], dm(BRISC, NOC0)))
                if mc_split:
                    mc_rt = ttnn.RuntimeArgs()
                    mc_rt[c.x][c.y] = [nrect - mc_split] + rects[1 + 5 * mc_split :]
                    mc_ct = [rounds, T * TB, k, SEM_MCGO, SEM_MCDONE, SEM_MCSRC, SEM_HADDR, SEM_SLOTS, V, SLICE_CHUNK]
                    kernels.append(kernel("slice_mc.cpp", C([c]), mc_ct, mc_rt, dm(NCRISC, NOC1)))
            kernels.append(
                kernel(
                    "slice_compute.cpp",
                    slice_grid,
                    [n, V, rounds, K, f32_bits(EPS), f32_bits(1 / HIDDEN), int(self.pre_weighted), SLICE_CHUNK, int(ISO), R,
                     int(ISO_SKIPFOLD)],
                    sc_rt,
                    compute_cfg(ttnn.MathFidelity.HiFi4, True, True, unpack_fp32=(7, 8)),
                )
            )
            kernels += [] if att is None else att.kernels(
                kernel, dm, compute_cfg, cores, serve, rope_addrs, self.attn_w_t.buffer_address(), kv_addrs, SEM_O,
                SEM_RXADDR, *attn_fidelity,
            ) if self.k2a else att.kernels(
                kernel,
                dm,
                compute_cfg,
                cores,
                serve,
                rope_addrs,
                self.attn_w_t.buffer_address(),
                kv_addrs,
                SEM_O,
                SEM_RXADDR,
                self.odump_t.buffer_address() if self.odump_t is not None else 0,
                self.pdump_t.buffer_address() if self.pdump_t is not None else 0,
                *attn_fidelity,
            )
            program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
            args = program.kernels[3].runtime_args[hub.x][hub.y]
            for link in range(self.links):
                for nb in (chip + 1, chip - 1):
                    if 0 <= nb < n:
                        args.append(1)
                        args.extend(
                            ttnn.setup_fabric_connection(
                                mesh,
                                mesh.get_fabric_node_id(coord),
                                mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, nb)),
                                link,
                                program,
                                hub,
                            )
                        )
                    else:
                        args.append(0)
            mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = program
            self.programs.append(program)
        self.mesh_pd = mesh_pd
        io = [self.rec_t, self.cand_t, self.dump_t, self.wpost_t] + self._keep
        io += [t for t in (self.embed_t, serve.x0) if t is not None]  # (M5 E: the engine's x0 instead of the table)
        io += [serve.tokens, serve.positions, serve.slots, serve.control, serve.status, serve.tables, serve.scratch_t]
        io += [serve.argmax_ids]  # (M5 A4)
        io += [serve.logits_raw] if serve.logits_raw is not None else []
        io += list(serve.taps) if self.taps_layers else []
        io += [self.wpre_t] if self.wpre_t is not None else []
        io += [self.attn_w_t] + [t for pair in serve.kv[:layers] for t in pair]
        io += att.io if self.k2a else []
        io += [att.tl_l] if not self.k2a and att is not None and att.tl_l is not None else []  # (CP1 timing)
        io += [t for t in (self.odump_t, self.pdump_t) if t is not None]
        io += [t for kind in ("sliding_attention", "full_attention") for t in serve.rope[kind]]
        io += [self.logits_t] if self.logits_t is not None else []
        io += [t for t in (self.ts_t, self.hub_ts_t, self.slice_ts_t) if t is not None]
        io.append(self.out_t)
        self.io = io
        self.trace_id = None

    # ---- host-side inputs
    def _view_rows(self, vecs):
        """[len(vecs), K V 1024] bf16: each vector on R rows, in the slices' tile order, each slice's tiles zero padded
        to V views"""
        R, K, T, V, TB = self.R, self.K, self.T, self.V, self.TB
        rows = torch.zeros(len(vecs), K * V * 1024, dtype=torch.bfloat16)
        for i, w in enumerate(vecs):
            flat = tile_order(w.reshape(1, -1).bfloat16().expand(R, -1).contiguous())
            for k in range(K):
                seg = flat[k * T * TB // 2 : (k + 1) * T * TB // 2]
                rows[i, k * V * 1024 : k * V * 1024 + seg.numel()] = seg
        return rows

    def _pre_rows(self, vectors):
        """row r: the weight of round r's h (r = 0 seed; 2l + 1 after layer l's mixer half; 2l + 2 after its MLP half):
        input norm of the next layer / pre-feedforward norm / the final norm after the last layer"""
        pre, L = vectors["pre"], self.layers
        vecs = [pre[0][0]]
        for l in range(L):
            vecs.append(pre[l][1])
            vecs.append(pre[l + 1][0] if l + 1 < L else vectors["final_norm"])
        return self._view_rows(vecs)

    def _post_rows(self, vectors):
        """[rounds, K V 1024] bf16: per all-reduce round its post-norm weight (post-attention, post-feedforward per
        layer) on R rows, in the slices' tile order, each slice's tiles zero padded to V views"""
        R, K, T, V, TB = self.R, self.K, self.T, self.V, self.TB
        rows = torch.zeros(self.rounds, K * V * 1024, dtype=torch.bfloat16)
        for l in range(self.layers):
            for half, w in enumerate(vectors["post"][l]):
                flat = tile_order(w.reshape(1, -1).expand(R, -1).contiguous())  # Ht tiles, TB / 2 elements each
                for k in range(K):
                    seg = flat[k * T * TB // 2 : (k + 1) * T * TB // 2]
                    rows[2 * l + half, k * V * 1024 : k * V * 1024 + seg.numel()] = seg
        return rows

    def _write(self, host, dev, dtype):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(host, dtype=dtype, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)), dev
        )

    def reset(self):
        """zero the launch counters and global semaphores (device idle)"""
        for s in self.hub_sems + self.slice_sems:
            ttnn.reset_global_semaphore_value(s, 0)
        ttnn.synchronize_device(self.mesh)

    def launch(self):
        if self.trace_id is not None:
            ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
        else:
            ttnn.generic_op(self.io, self.mesh_pd)

    def capture_trace(self):
        self.trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        ttnn.generic_op(self.io, self.mesh_pd)
        ttnn.end_trace_capture(self.mesh, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh)

    def release_trace(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh, self.trace_id)
            self.trace_id = None

    # ---- read back
    def _per_chip(self, t):
        return [ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)]

    def out(self):
        """per chip: ([M tokens], [M keys], (launch, position, M))"""
        res = []
        for o in self._per_chip(self.out_t):
            w = o.reshape(-1).to(torch.int64) & 0xFFFFFFFF
            res.append(([int(w[2 * r]) for r in range(self.M)], [int(w[2 * r + 1]) for r in range(self.M)], tuple(int(v) for v in w[64:67])))
        return res

    def tokens(self):
        return self.out()[0][0]

    def _dump_rows(self, row):
        """[n, M, HIDDEN] from dump row `row` (the slices' views, in tile order)"""
        out = []
        for d in self._per_chip(self.dump_t):
            d = d.reshape(self.dump_rows_n, self.K, self.V * 1024)[row]
            flat = torch.cat([d[k, : self.T * self.TB // 2] for k in range(self.K)])
            out.append(tile_unorder(flat, self.R)[: self.M].float())
        return torch.stack(out)

    def taps(self, k):
        """{layer: [n, M, HIDDEN]}: the residual stream after each tapped layer (the step's rows), from taps buffer k"""
        out = {}
        chips = [ttnn.to_torch(x).reshape(32, -1) for x in ttnn.get_device_tensors(self.serve.taps[k])]
        for i, l in enumerate(self.taps_layers):
            out[l] = torch.stack([c[: self.M, i * HIDDEN : (i + 1) * HIDDEN].float() for c in chips])
        return out

    def attn_out(self):
        """[n, layers, M, 8 heads x head_dim] (dump_rounds): every layer's attention output (the o_proj input)"""
        out = []
        for d in self._per_chip(self.odump_t):
            d = d.reshape(self.layers, -1)
            rows = []
            for l in range(self.layers):
                n = (OT_F if is_full(l) else OT_S) * self.TB // 2
                o = tile_unorder(d[l, :n].float(), self.R)[: self.M]
                rows.append(torch.nn.functional.pad(o, (0, Q_F - o.shape[1])))
            out.append(torch.stack(rows))
        return torch.stack(out)

    def partials(self, chip=0):
        """(GRES_PDUMP) [layers, 32 cores, PTmax, 32, 32]: each core's own partial (O tiles, m, l) as 32x32 tiles"""
        d = self._per_chip(self.pdump_t)[chip].reshape(self.layers, 32, -1, 2, 2, 16, 16)
        return d.permute(0, 1, 2, 3, 5, 4, 6).reshape(self.layers, 32, -1, 32, 32).float()

    def final_h(self):
        """[n, M, HIDDEN]: rmsnorm(x) after the last layer (the LM head input, final norm weight folded) per chip"""
        return self._dump_rows(0)

    def round_x(self):
        """(M5 K1, dump_rounds) [n, M, layers, HIDDEN]: the residual x after every layer, from the slices' raw copy of
        the tiles the taps write reads (the reference of the taps layout gate, M4_M5_GATES site 2)"""
        return torch.stack([self._dump_rows(2 + self.rounds + l) for l in range(self.layers)], dim=2)

    def round_h(self):
        """[n, M, rounds + 1, HIDDEN] (dump_rounds): h of the seed and every round"""
        return torch.stack([self._dump_rows(1 + r) for r in range(self.rounds + 1)], dim=2)

    def logits(self):
        """[M, VOCAB] logits of the last step (bf16 values as float; dump_rounds), from every chip's streamers"""
        out = torch.zeros(self.M, VOCAB)
        for chip, raw in enumerate(self._per_chip(self.logits_t)):
            raw = raw.reshape(self.S, -1)
            for i in range(self.S):
                v = self.per_core(E_HEAD, i)
                if v:
                    lg = tile_unorder(raw[i].float(), self.R)[: self.M]
                    c0 = chip * (VOCAB // self.n) + v[0] * TILE
                    out[:, c0 : c0 + len(v) * TILE] = lg[:, : len(v) * TILE]
        return out

    def timelines(self):
        """(streamer [n, S, layers, 16], hub [n, words], slices [n, K, words]) raw wall-clock words"""
        st = torch.stack([t.reshape(self.S, self.layers, TS_WORDS) for t in self._per_chip(self.ts_t)])
        hub = torch.stack([t.reshape(-1) for t in self._per_chip(self.hub_ts_t)])
        sl = torch.stack([t.reshape(self.K, -1) for t in self._per_chip(self.slice_ts_t)])
        m = lambda x: x.to(torch.int64) & 0xFFFFFFFF
        return m(st), m(hub), m(sl)
