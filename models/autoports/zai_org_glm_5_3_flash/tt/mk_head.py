# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14b (PRODUCTION_PATH.md M5): the model head as ONE megakernel program per chip, run right after the layer
stack on its stacked residual:

  H1  final norm      64 owners (kernels/head)   res [t,1,4,4096] fp32 -> h = RMSNorm(mean of the 4 streams) * w,
                                                 [1,1,t,4096] bf16 (one all-gather of the owners' partial sums of
                                                 squares; the readers signalled, count 64)
  H2  lm_head         the DS bank readers        h -> logits, the chip's vocab columns in `splits` DS matmul phases
                      (7a's reader kernels)      (bfloat8_b weights streamed through the ring, HiFi2, fp32 accumulate)
                                                 -> [1,1,t,cw_pad] bf16 DRAM per split
  H3  top-K (topk>0)  the readers' activation    every logits tile scanned in L1 as it is written out (kernels/mk/
                      mover + one select core    topk.hpp): each reader's K best (key, vocab id) per row -> the select
                                                 core (kernels/head/topk_select.cpp) merges -> [4, 64] uint32: row r =
                                                 the K best vocab ids (word 0 = the greedy token) | their bf16 keys

Vocab layout (Stage 14's tp8_head.py contract): the 00c cache's lm_head.weight [4096, 154,880] bfloat8_b, padded to
`tp_padded_vocab`; chip r holds the tile-aligned columns [W r, W (r + 1)), W = Vpad / tp, as `splits` phases of
cw = W / splits columns (each DS-padded); logit column c of split j on chip r is vocab id W r + cw j + c. The slices are
lossless (bfloat8_b blocks are 16-value face rows inside a tile)."""

from __future__ import annotations

import os
import struct

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre import merged_ranges
from models.autoports.zai_org_glm_5_3_flash.tt.mk_dense import rect_placement
from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import MMPhase, ReaderPhases
from models.autoports.zai_org_glm_5_3_flash.tt.rt_guard import checked
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import VOCAB_SIZE, tp_padded_vocab

BF16, FP32 = ttnn.bfloat16, ttnn.float32
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
TILE = 32
HIDDEN = 4096
SEM_HEAD = 0  # owners: the partial-sum all-gather
SEM_HN = 4  # readers: h landed (64 owners)
SEM_TOPK = 0  # the select core: one increment per reader
TOPK_CB = 37  # readers: the top-K record scratch (no reader phase uses 37)
KDIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels")


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def lm_head_host(source) -> torch.Tensor:
    """the cache's lm_head [4096, 154,880] (the bfloat8_b values, exactly)"""
    e = source.tensors["lm_head.weight"]
    return ttnn.to_torch(source._load(e)).reshape(e["shape"]).float()


def norm_host(source) -> torch.Tensor:
    e = source.tensors["norm.weight"]
    return ttnn.to_torch(source._load(e)).reshape(-1).float()


MAX_CHUNKS = 8  # 4-row residual chunks of t <= 32


def _chunk_addrs(chunks) -> list:
    """the residual chunks 1.. addresses, zero-padded to a FIXED length: generic_op's program cache keys on the
    descriptor without the runtime args, so the arg count must not vary with the batch (a cached program with fewer
    args reused for more chunks would leave the extra args unwritten)"""
    a = [r_.buffer_address() for r_ in chunks[1:]]
    assert len(a) < MAX_CHUNKS, len(chunks)
    return a + [0] * (MAX_CHUNKS - 1 - len(a))


class MkHead:
    """final norm + lm_head (this chip's vocab shard) as ONE program. __call__(res) -> [logits chunk per split]."""

    def __init__(self, device, lm_w: torch.Tensor, norm_w: torch.Tensor, *, t: int, eps: float,
                 splits: int | None = None, kb: int = 8, ocfg=None, topk: int = 0, fabric=None,
                 n_prod: int | None = None, sample: bool = False):  # fmt: skip
        """device: one chip, or a mesh (TP over the vocab: chip r = rank r in row-major mesh order holds columns
        [W r, W (r + 1)); the program differs per chip only in the vocab base of the top-K candidates).
        fabric (TP + top-K): the layers' FabricPhase (tt/fabric_ar.py, "cb" mode) -> the chips' candidates are merged
        IN the program (kernels/head/topk_select.cpp TOPK_FAB): every chip ends with the same global top-K rows (the
        next step's token on every chip); n_prod = the layers' producers per all-reduce (the lanes' OWN_READY step).
        sample (top-K): device sampling in the select core (kernels/head/topk_select.cpp TOPK_SAMPLE) from the per-row
        params page `samp_par` (k, p, 1 / temperature, flags) and RNG page `samp_rng` (xoshiro128** states), both
        [1, 128] uint32 DRAM written by the host between steps (the RNG page also advanced by the kernel)."""
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import DSLinear, OptimizedDecoderConfig
        from models.autoports.zai_org_glm_5_3_flash.tt.precision import hifi2_fp32

        assert 1 <= t <= 32, t
        tp = int(device.get_num_devices())
        self.device, self.t, self.eps, self.tp = device, t, eps, tp
        shard_map = (lambda: ttnn.ShardTensorToMesh(device, dim=-1)) if tp > 1 else (lambda: None)  # noqa: E731
        repl = (lambda: ttnn.ReplicateTensorToMesh(device)) if tp > 1 else (lambda: None)  # noqa: E731
        self.vpad = tp_padded_vocab(VOCAB_SIZE, tp)
        self.w = self.vpad // tp
        self.splits = splits or max(1, 8 // tp)  # <= ~605 tiles per DS phase (the readers' output CB)
        assert self.w % (TILE * self.splits) == 0, (self.w, self.splits)
        self.cw = self.w // self.splits
        ocfg = ocfg or OptimizedDecoderConfig()
        ocfg.ds_readers_per_role = ";".join(filter(None, [ocfg.ds_readers_per_role, "lm_head:3"]))
        full = torch.nn.functional.pad(lm_w, (0, self.vpad - lm_w.shape[-1]))
        self.ds = []
        for j in range(self.splits):
            cols = [full[:, self.w * r + self.cw * j : self.w * r + self.cw * (j + 1)] for r in range(tp)]
            shard = ttnn.from_torch(torch.cat(cols, dim=-1).contiguous(), dtype=ttnn.bfloat8_b,
                                    layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM,
                                    mesh_mapper=shard_map())  # fmt: skip
            self.ds.append(DSLinear(device, shard, role="lm_head", cfg=ocfg, ckc=hifi2_fp32(device), dtype=BF16))
        del full
        self.readers = ReaderPhases(device, [MMPhase(d, kb=kb, wait=(SEM_HN, 64) if j == 0 else None)
                                             for j, d in enumerate(self.ds)])  # fmt: skip
        self.place = rect_placement(device, {(int(c.x), int(c.y)) for c, _, _ in self.readers.assign})
        wt = torch.zeros(TILE, HIDDEN)
        wt[0] = norm_w.float()
        self.norm_w = ttnn.from_torch(wt, dtype=FP32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM,
                                      mesh_mapper=repl())  # fmt: skip
        self.owner_noc = [
            (int(w.x), int(w.y)) for w in (device.worker_core_from_logical_core(c) for c in self.place["k"])
        ]
        assert 0 <= topk <= 32, topk
        self.topk = topk
        grid = device.compute_with_storage_grid_size()
        used = {(int(c.x), int(c.y)) for c, _, _ in self.readers.assign} | {
            (int(c.x), int(c.y)) for c in self.place["k"]
        }
        used.add((int(self.place["root"].x), int(self.place["root"].y)))
        self.fabric, self.n_prod = fabric, n_prod
        if fabric is not None:  # the lanes are the fabric phase's cores
            assert topk and tp > 1 and n_prod, "the in-program candidate gather is for TP + top-K"
            # topk_select.cpp puts row r at row r % brows of row group r // brows (packed rows, or whole tiles at
            # brows 32), 6 bf16 fields per candidate per chip: every group must fit lane 0's 64 tiles
            assert -(-t // fabric.brows) * tp * topk * 6 <= 64 * 32, ("candidates exceed lane 0", t, fabric.brows, topk)
            used |= {tuple(int(v) for v in lane) for lane in fabric.lanes}
        self.select = next((x, y) for y in range(int(grid.y)) for x in range(int(grid.x)) if (x, y) not in used)
        self.select_noc = device.worker_core_from_logical_core(ttnn.CoreCoord(*self.select))
        self.topk_out = self.topk_land = self.sampled = None
        if topk:  # row r: the K best vocab ids | their bf16 keys (one 256-B page per row)
            mk_rows = lambda: ttnn.from_torch(torch.zeros(32, 64, dtype=torch.int32), dtype=ttnn.uint32,  # noqa: E731
                                              layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM,
                                              mesh_mapper=repl())  # fmt: skip
            self.topk_out = mk_rows()  # the next step's token input (row t word 0)
            self.sampled = mk_rows()  # the readback copy (same rows; differs where the host rewrote topk_out)
        assert not sample or topk, "device sampling draws from the top-K candidates"
        self.sample = bool(sample)
        self.samp_par = self.samp_rng = None
        if self.sample:  # [1, 128] uint32: 32 rows x 4 words, one 512-B DRAM page (all rows greedy until written)
            mk_page = lambda: ttnn.from_torch(torch.zeros(1, 128, dtype=torch.int32), dtype=ttnn.uint32,  # noqa: E731
                                              layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM,
                                              mesh_mapper=repl())  # fmt: skip
            self.samp_par, self.samp_rng = mk_page(), mk_page()
        self._repl = repl
        self.h = None
        self.outs = None
        self.probe = None  # GLM53_HEAD_PROBE=1: [2 NREAD + 1, 16] uint32 phase marks (kernels/mk/probe.hpp)

    @classmethod
    def width_view(cls, base: "MkHead", t: int, view) -> "MkHead":
        """the head of a native width bucket (Stage 14b width traces: rows 0..t-1 live): base's weights, readers,
        select core, fabric phase, token / sampler pages and landing buffer; h and the logits outputs as t-row views
        of base's (the same buffers: no upload, no new L1). The norm / lm_head / top-K / sampling / the position
        advance cover rows < t only (the 32-row head's top-K scan is ~linear in its rows)"""
        assert base.h is not None, "the base head runs (eagerly) before its width views"
        assert 1 <= t <= base.t, (t, base.t)
        o = cls.__new__(cls)
        o.__dict__.update(base.__dict__)
        o.t = t
        o.h = view(base.h, t)
        o.outs = [view(x, t) for x in base.outs]
        return o

    def __call__(self, res, pos=None, pos_copies=None):
        """res: the stacked residual [t, 1, 4, 4096] (t <= 4), or the list of 4-row chunk residuals (t <= 32); pos (top-K
        only): the decode positions [1, 32] int32 ROW_MAJOR, advanced (+1 on rows >= 0) after the step's sampling;
        pos_copies: per-chunk [1, 32] int32 tensors that get rows 4c..4c+3 of the advanced positions at rows 0..3 (the
        sparse_dsa programs of chunk c read their positions there)"""
        copies = list(pos_copies or [])
        assert len(copies) <= 8, len(copies)
        chunks = list(res) if isinstance(res, (list, tuple)) else [res]
        res = chunks[0]
        flat = len(chunks) == 1 and int(res.shape[-1]) == 16384  # the flat residual [1, 1, t, 16384] (native t)
        assert flat or len(chunks) == -(-self.t // 4), (len(chunks), self.t)
        dev, rd, t = self.device, self.readers, self.t
        if self.h is None:  # persistent (allocated by the first, eager call: never inside a capture)
            self.h = ttnn.empty((1, 1, t, HIDDEN), BF16, ttnn.TILE_LAYOUT, dev, L1)
            self.outs = rd.outputs(t, DRAM)
        h, outs = self.h, self.outs
        acc = lambda tt: list(ttnn.TensorAccessorArgs(tt).get_compile_time_args())  # noqa: E731
        owners = merged_ranges([(int(c.x), int(c.y)) for c in self.place["k"]])
        ks = self.place["k"]
        flat_r = [v for xy in rd.noc_coords() for v in xy]
        flat_o = [v for xy in self.owner_noc for v in xy]
        rt0, rt1 = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for k, c in enumerate(ks):
            rt0[c.x][c.y] = [res.buffer_address(), self.norm_w.buffer_address(), h.buffer_address(), 2 * k, t,
                             len(rd.assign)] + flat_r + _chunk_addrs(chunks)  # fmt: skip
            rt1[c.x][c.y] = [t, k] + flat_o
        defs = [("HEAD_SEM", str(SEM_HEAD)), ("HEAD_HN_SEM", str(SEM_HN))] + ([("HEAD_RES_FLAT", "1")] if flat else [])
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)  # noqa: E731
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True)  # fmt: skip
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for cb in (0, 1, 2, 4):  # CB_X, CB_W, CB_M, CB_G (kernels/head/head_common.hpp): exact fp32 to DST
            modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=os.path.join(KDIR, "head", src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=owners, compile_time_args=ct, runtime_args=rt, defines=defs, config=cfg)  # fmt: skip
        kernels = [
            kd("head_dm0.cpp", acc(res) + acc(self.norm_w) + acc(h), rt0,
               dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)),
            kd("head_dm1.cpp", [], rt1, dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default)),
            kd("head_c.cpp", [f32_bits(1.0 / HIDDEN), f32_bits(self.eps), f32_bits(0.25)], ttnn.RuntimeArgs(), cc),
        ]  # fmt: skip
        cb = lambda i, dt, page, n: ttnn.CBDescriptor(total_size=page * n, core_ranges=owners, format_descriptors=[  # noqa: E731
            ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)])  # fmt: skip
        cbs = [cb(0, FP32, 4096, 8), cb(1, FP32, 4096, 2), cb(2, FP32, 4096, 2), cb(3, FP32, 4096, 1),
               cb(4, FP32, 4096, 2), cb(5, BF16, 2048, 2), cb(6, ttnn.uint32, 64 * 128, 1), cb(7, ttnn.uint32, 160, 1)]  # fmt: skip
        sems = [ttnn.SemaphoreDescriptor(id=SEM_HEAD, core_ranges=owners, initial_value=0),
                ttnn.SemaphoreDescriptor(id=SEM_HN, core_ranges=rd.cores, initial_value=0)]  # fmt: skip
        io = chunks + [self.norm_w, h] + [d.weight for d in self.ds] + list(outs)
        if not self.topk:
            r_kernels, r_cbs = rd.parts([h] * self.splits, outs)
        else:
            K, nread = self.topk, len(rd.assign)
            # GLM53_HEAD_PROBE=1 (diagnostics, never in a measured row): reader i's weight streamer / activation mover
            # marks -> probe rows 2 i / 2 i + 1 (the scan's span inside the mover), the select core's -> row 2 NREAD
            probe = os.environ.get("GLM53_HEAD_PROBE") == "1"
            if probe and self.probe is None:
                self.probe = ttnn.from_torch(torch.zeros(2 * nread + 1, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                             layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=DRAM,
                                             mesh_mapper=self._repl())  # fmt: skip
            raw, r_cbs = rd.parts_raw([h] * self.splits, outs,
                                      mk_probe=(self.probe.buffer_address(), 0) if probe else None)  # fmt: skip
            sel = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*self.select), ttnn.CoreCoord(*self.select))])
            page = max(64, -(-32 * K * 8 // 64) * 64)  # one reader's rows (32 x K x (key, id)), 64-B DRAM pages
            land_bytes = nread * page
            if self.topk_land is None:
                self.topk_land = ttnn.from_torch(torch.zeros(nread, page // 4, dtype=torch.int32), dtype=ttnn.uint32,
                                                 layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=DRAM,
                                                 mesh_mapper=self._repl())  # fmt: skip
            _, a_ct, a_rt, a_defs, a_cfg = raw["a"]
            n_a = max(len(v) for v in a_rt.values())
            assert all(len(v) == n_a for v in a_rt.values()), "uniform activation-mover args"
            slot = {(int(c.x), int(c.y)): i for i, (c, _, _) in enumerate(rd.assign)}
            land = self.topk_land.buffer_address()
            a_rt_r = [{xy: list(v) + [t, self.cw, self.w * r, int(self.select_noc.x), int(self.select_noc.y), land,
                                      slot[xy]] for xy, v in a_rt.items()} for r in range(self.tp)]  # fmt: skip
            # the readers' heap composites (kernels/mk/topk.hpp): local column = (phase slot << 12) | column within the
            # reader's tiles of the phase -> <= 15 phases, <= 128 tiles per reader per phase
            assert self.splits <= 15 and rd.nr_max <= 128, (self.splits, rd.nr_max)
            a_defs = list(a_defs) + [("MK_TOPK_K", str(K)), ("MK_TOPK_SEM", str(SEM_TOPK)), ("MK_TOPK_CB", str(TOPK_CB)),
                                     ("MK_TOPK_RT", str(n_a)), ("MK_TOPK_PAGE", str(page))]  # fmt: skip
            r_kernels = [
                rd.kernel(*raw["w"]),
                rd.kernel(raw["a"][0], a_ct, a_rt_r[0], a_defs, a_cfg),
                rd.kernel(*raw["c"]),
            ]
            a_per_rank = [rd.kernel(raw["a"][0], a_ct, a_rt_r[r], a_defs, a_cfg) for r in range(self.tp)]
            scratch = 32 * K * 8 + 128  # the running rows + 32 counts (kernels/mk/topk.hpp)
            r_cbs = list(r_cbs) + [ttnn.CBDescriptor(total_size=scratch, core_ranges=rd.cores, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=TOPK_CB, data_format=ttnn.uint32, page_size=scratch)])]  # fmt: skip
            land_cb, out_cb = 0, 1
            cbs += [ttnn.CBDescriptor(total_size=land_bytes, core_ranges=sel, format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=land_cb, data_format=ttnn.uint32, page_size=land_bytes)]),
                    ttnn.CBDescriptor(total_size=256, core_ranges=sel, format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=out_cb, data_format=ttnn.uint32, page_size=256)])]  # fmt: skip
            srt = ttnn.RuntimeArgs()
            s_args = [t, self.topk_out.buffer_address(), land, self.sampled.buffer_address(),
                      pos.buffer_address() if pos is not None else 0] + [c_.buffer_address() for c_ in copies]  # fmt: skip
            cdefs = [("TOPK_POS_COPIES", str(len(copies))), ("TOPK_POS_COPY_RT", "5")] if copies else []
            if self.sample:  # the sampler pages' addresses after every other select arg (TOPK_SAMPLE_RT per path)
                cbs.append(ttnn.CBDescriptor(total_size=1152, core_ranges=sel, format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=3, data_format=ttnn.uint32, page_size=1152)]))  # fmt: skip
                cdefs += [("TOPK_SAMPLE", "1"), ("TOPK_SAMPLE_CB", "3"), ("TOPK_SAMPLE_RT", str(len(s_args)))]
                s_args = s_args + [self.samp_par.buffer_address(), self.samp_rng.buffer_address()]
            if probe:  # the select core's marks: probe row 2 NREAD (args last; 64-B scratch CB 31)
                cbs.append(ttnn.CBDescriptor(total_size=64, core_ranges=sel, format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=31, data_format=ttnn.uint32, page_size=64)]))  # fmt: skip
                cdefs += [("MK_PROBE_ARG", str(len(s_args))), ("MK_PROBE_CB", "31")]
                s_args = s_args + [self.probe.buffer_address(), 2 * nread]
            srt[self.select[0]][self.select[1]] = s_args
            kernels.append(ttnn.KernelDescriptor(
                kernel_source=os.path.join(KDIR, "head", "topk_select.cpp"),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=sel,
                compile_time_args=[K, nread] + acc(self.topk_out), runtime_args=srt,
                defines=[("TOPK_SEM", str(SEM_TOPK)), ("TOPK_LAND_CB", str(land_cb)), ("TOPK_OUT_CB", str(out_cb)),
                         ("TOPK_PAGE", str(page))] + cdefs,
                config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)))  # fmt: skip
            sems.append(ttnn.SemaphoreDescriptor(id=SEM_TOPK, core_ranges=sel, initial_value=0))
            io += [self.topk_out, self.topk_land, self.sampled] + ([pos] if pos is not None else []) + copies
            io += [self.samp_par, self.samp_rng] if self.sample else []
        if self.tp == 1 or not self.topk:
            ttnn.generic_op(io, checked(ttnn.ProgramDescriptor(kernels=kernels + r_kernels, semaphores=sems,
                                                               cbs=cbs + r_cbs)))  # fmt: skip
            return outs
        # TP with top-K: the chips' programs differ in the readers' vocab base (rank r = row-major mesh order) and, with
        # the in-program gather, the select core's rank + the lane kernels of the mesh coordinate
        mpd = ttnn.MeshProgramDescriptor()
        cols = int(self.device.shape[1])
        fab = self.fabric
        if fab is not None:
            sel = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*self.select), ttnn.CoreCoord(*self.select))])
            sel_k = kernels.pop()  # the select kernel: per chip below (rank, the TOPK_FAB defines and args)
            img_bytes = 2 * 64 * fab.v
            cbs = cbs + [ttnn.CBDescriptor(total_size=img_bytes, core_ranges=sel, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=2, data_format=ttnn.uint32, page_size=img_bytes)])]  # fmt: skip
            sems = sems + [ttnn.SemaphoreDescriptor(id=SEM_TOPK + 1, core_ranges=sel, initial_value=0)]
            io = io + fab.io()
        for r in range(self.tp):
            ks_r = kernels + [r_kernels[0], a_per_rank[r], r_kernels[2]]
            m = ttnn.MeshCoordinate(r // cols, r % cols)
            if fab is None:
                mpd[ttnn.MeshCoordinateRange(m, m)] = ttnn.ProgramDescriptor(kernels=ks_r, semaphores=sems,
                                                                             cbs=cbs + r_cbs)  # fmt: skip
                continue
            srt = ttnn.RuntimeArgs()
            base_args = ([t, self.topk_out.buffer_address(), self.topk_land.buffer_address(),
                          self.sampled.buffer_address(), pos.buffer_address() if pos is not None else 0, r]
                         + fab.producer_args() + fab.consumer_args()[4:5] + [fab.v, fab.ch, self.n_prod])  # fmt: skip
            f_args = base_args + [c_.buffer_address() for c_ in copies]
            sel_defs = [d for d in sel_k.defines if d[0] not in ("TOPK_POS_COPY_RT", "TOPK_SAMPLE_RT", "MK_PROBE_ARG")]
            sel_defs += [("TOPK_POS_COPY_RT", str(len(base_args)))] if copies else []
            if self.sample:
                sel_defs += [("TOPK_SAMPLE_RT", str(len(f_args)))]
                f_args = f_args + [self.samp_par.buffer_address(), self.samp_rng.buffer_address()]
            if self.probe is not None and os.environ.get("GLM53_HEAD_PROBE") == "1":
                sel_defs += [("MK_PROBE_ARG", str(len(f_args)))]
                f_args = f_args + [self.probe.buffer_address(), 2 * len(rd.assign)]
            srt[self.select[0]][self.select[1]] = f_args
            ks_r = ks_r + [ttnn.KernelDescriptor(
                kernel_source=sel_k.kernel_source, source_type=sel_k.source_type, core_ranges=sel,
                compile_time_args=list(sel_k.compile_time_args), runtime_args=srt,
                defines=sel_defs + [("TOPK_FAB", "1"), ("TOPK_FAB_SEM", str(SEM_TOPK + 1)),
                                               ("TOPK_IMG_CB", "2"), ("TOPK_NCHIP", str(self.tp))],
                config=sel_k.config)]  # fmt: skip
            sel_xy = [(int(self.select_noc.x), int(self.select_noc.y))]
            lk, lcbs = fab.lane_parts(r // cols, r % cols, 1, self.n_prod, sel_xy, cons_sem=SEM_TOPK + 1)
            prog = ttnn.ProgramDescriptor(kernels=ks_r + lk, semaphores=sems, cbs=cbs + r_cbs + lcbs)
            fab.add_connections(prog, r // cols, r % cols, send_kernel_index=len(ks_r))
            mpd[ttnn.MeshCoordinateRange(m, m)] = prog
        ttnn.generic_op(io, mpd)
        return outs

    def probe_rows(self) -> list:
        """GLM53_HEAD_PROBE: per chip [2 NREAD + 1, 16] uint32 marks (rows 2 i / 2 i + 1: reader i's weight streamer /
        activation mover; the last row: the select core)"""
        devs = ttnn.get_device_tensors(self.probe) if self.tp > 1 else [self.probe]
        return [(ttnn.to_torch(d).to(torch.int64) & 0xFFFFFFFF) for d in devs]

    def gather(self, outs) -> torch.Tensor:
        """host: the logits [t, Vpad] in vocab order (every chip's shard, DS padding dropped)"""
        per_rank = [[ttnn.to_torch(d).float().reshape(-1, d.shape[-1])[: self.t, : self.cw]
                     for d in ttnn.get_device_tensors(o)] for o in outs]  # fmt: skip
        return torch.cat([torch.cat([per_rank[j][r] for j in range(self.splits)], -1) for r in range(self.tp)], -1)

    def topk_host(self) -> torch.Tensor:
        """host: every chip's top-K rows -> [tp, t, K] vocab ids (each chip: its own vocab shard's candidates)"""
        return torch.stack([ttnn.to_torch(d).to(torch.int64).reshape(32, 64)[: self.t, : self.topk]
                            for d in ttnn.get_device_tensors(self.topk_out)])  # fmt: skip


def embed_host(source) -> torch.Tensor:
    """the cache's embed_tokens [154,880, 4096] (bf16 values)"""
    e = source.tensors["embed_tokens.weight"]
    return ttnn.to_torch(source._load(e)).reshape(e["shape"]).to(torch.bfloat16)


class MkEmbed:
    """The step's input as ONE program on 64 cores: the device-resident tokens [4, 64] uint32 (row t word 0; the
    previous step's MkHead top-K output, or a host write) -> the stacked residual [t,1,4,4096] fp32 = the token's
    embedding in all 4 hyper-connection streams (kernels/head/embed_dm0.cpp)."""

    def __init__(self, device, table: torch.Tensor, *, t: int, place: dict):
        assert 1 <= t <= 32, t
        self.device, self.t, self.place = device, t, place
        repl = ttnn.ReplicateTensorToMesh(device) if int(device.get_num_devices()) > 1 else None
        self.table = ttnn.from_torch(table.reshape(table.shape[0], HIDDEN), dtype=BF16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                     device=device, memory_config=DRAM, mesh_mapper=repl)  # fmt: skip

    def __call__(self, tokens, res):
        """tokens [32, 64] uint32 (row t word 0); res: the stacked residual (t <= 4) or the 4-row chunk residuals"""
        chunks = list(res) if isinstance(res, (list, tuple)) else [res]
        res = chunks[0]
        acc = lambda tt: list(ttnn.TensorAccessorArgs(tt).get_compile_time_args())  # noqa: E731
        ks = self.place["k"]
        cores = merged_ranges([(int(c.x), int(c.y)) for c in ks])
        rt = ttnn.RuntimeArgs()
        for k, c in enumerate(ks):
            rt[c.x][c.y] = ([tokens.buffer_address(), self.table.buffer_address(), res.buffer_address(), 2 * k, self.t]
                            + _chunk_addrs(chunks))  # fmt: skip
        kd = ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, "head", "embed_dm0.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=cores,
            compile_time_args=acc(tokens) + acc(self.table) + acc(res), runtime_args=rt,
            defines=[("EMB_FLAT", "1")] if len(chunks) == 1 and int(res.shape[-1]) == 16384 else [],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                     noc=ttnn.NOC.RISCV_0_default))  # fmt: skip
        cb = ttnn.CBDescriptor(total_size=4096 + 384 + 64, core_ranges=cores, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=8, data_format=ttnn.uint32, page_size=4096 + 384 + 64)])  # fmt: skip
        desc = checked(ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cb]))
        ttnn.generic_op([tokens, self.table] + chunks[1:] + [res], desc)
        return res
