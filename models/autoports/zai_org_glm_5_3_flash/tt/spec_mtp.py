# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 17 (PRODUCTION_PATH.md §8c): MTP speculative decoding INSIDE the persistent decode step of
tt/stack_model.StackBackend — design of record: doc/ttnn/17-mtp-spec-decode/design.md.

One speculative step of ONE user (slot 0) with K drafts = one trace replay, no host work inside:

  verify   MkEmbed (t = K+1, the token rows [b, d1..dK]) -> the 45 layers at width K+1 in VERIFY mode (KDA rows chained
           through the state / conv window, kda_pre KDA_VERIFY + kda_rec REC_CHAIN with a device-selected source row;
           MLA causal inside the block, MLA_ATT_VERIFY + a page table whose rows 0..K repeat the user's row) -> a
           verify head (MkHead clone: top-1, no sampling, no position advance) -> a_0..a_K
  accept   SpecCtl ACCEPT (one core): n = leading a_{j-1} == d_j, bonus a_n -> token row 0, sel = n, positions
  draft    MTP catch-up over the K+1 rows (h_j = the main head's h row j, token a_j; vLLM MTP semantics): MkEh (enorm /
           hnorm + eh_proj) -> EhExpand -> layer 45's TPMla + TPMoE (identity mHC: mtp_mhc_args) -> an MTP head
           (shared_head.norm + the shared lm_head, top-1) -> d'1 = row n; K-1 chained single rows (h = the MTP block's
           raw output) -> d'2..d'K; SpecCtl GATHER -> token rows 1..K.

The bootstrap (after a prefill / plain step) is the same trace without the verify part (SpecCtl BOOT: n = 0, the plain
head's h row 0). State rollback (design.md §2): the verify writes CANDIDATE KDA states / conv histories to rows 0..K of
the shared 32-row state; the next step reads row `sel` = n; latent rows past the committed position are rewritten
before any query attends them.
"""

from __future__ import annotations

import contextlib
import math
import os

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.config import DENSE_KDA, SPARSE_KDA
from models.autoports.zai_org_glm_5_3_flash.tt.generator import (  # noqa: F401 (ttnn-free)
    mtp_prefill_span,
    spec_multi_groups,
    spec_widths,
)
from models.autoports.zai_org_glm_5_3_flash.tt.rt_guard import checked

MTP_L = 45  # conventions.mtp_layer (weights/manifest.json)
DRAM = ttnn.DRAM_MEMORY_CONFIG
BF16, FP32 = ttnn.bfloat16, ttnn.float32
TILE = 32
HIDDEN = 4096
KDIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels")
SPEC_REV = 4  # 4: SPEC_GROUPS ACCEPT addresses user u's ring at slot u (K + 1) of each layer's tensor (R bases, not R x G); 3: ACCEPT moves the DSA rings (mk_mla rev 56); a define on every Stage 17 kernel (a long-lived process caches FILE_PATH binaries by path + args + defines)
SEM_HEAD, SEM_HN = 0, 4  # kernels/head/head_common.hpp roles (owners: partial-sum all-gather; readers: hcat landed)


def _repl(mesh):
    return ttnn.ReplicateTensorToMesh(mesh) if int(mesh.get_num_devices()) > 1 else None


def dram_rows(mesh, host: torch.Tensor, dtype=ttnn.uint32) -> ttnn.Tensor:
    """a small ROW_MAJOR DRAM tensor replicated on every chip (token / position / selector / record pages)"""
    return ttnn.from_torch(host.to(torch.int32), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                           memory_config=DRAM, mesh_mapper=_repl(mesh))  # fmt: skip


def write_rows(mesh, t: ttnn.Tensor, host: torch.Tensor, dtype=ttnn.uint32) -> None:
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(host.to(torch.int32), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                    mesh_mapper=_repl(mesh)), t)  # fmt: skip


def read_rows(mesh, t: ttnn.Tensor) -> torch.Tensor:
    d = ttnn.get_device_tensors(t)[0] if int(mesh.get_num_devices()) > 1 else t
    return ttnn.to_torch(d).to(torch.int64)


@contextlib.contextmanager
def verify_allocator(pool, w: int, enable: bool, mod=None, pools=()):
    """GLM53_SPEC_PC_DRAM for the verify's STACK BUILDS only: patches the pools' allocators (AllocPool.orig, the
    pool and its child, plus every pool in `pools` -- the sparse_dsa layers' DSA pool -- and their children) with
    _pc_dram and, on exit, restores them AND the module's `empty` (ttnn.empty). The restore is
    needed because AllocPool.__exit__ sets ttnn.empty = pool.orig: a stack build under the patch leaves the patched
    allocator GLOBAL (it leaked into the plain decode: a later plain step at the verify's width allocated its post /
    comb tiles in DRAM = a new program key = a compile inside the trace capture: TT_FATAL).
    The handle's reset() restores `empty` between calls (the verify's MkMla calls between stack builds). Re-entrant:
    each use restores exactly what it saw at entry, also on an exception. tests/s17/test_verify_allocator.py."""
    mod = ttnn if mod is None else mod
    saved = mod.empty
    targets = [q for p in (pool, *pools) for q in (p, getattr(p, "_child", None))]
    origs = [(o, o.orig) for o in targets if o is not None and hasattr(o, "orig")] if enable else []
    for o, f in origs:  # (new allocations only: the first call builds the programs; later calls reuse the pool's)
        o.orig = _pc_dram(f, w)

    class _Handle:
        @staticmethod
        def reset():
            mod.empty = saved

    try:
        yield _Handle()
    finally:
        for o, f in reversed(origs):
            o.orig = f
        mod.empty = saved


def _pc_dram(orig, w: int):
    """ttnn.empty of the verify stacks' pool with the t = w stacked mHC post / comb tiles in DRAM (GLM53_SPEC_PC_DRAM,
    default on): (w,1,4,1) / (w,1,4,4) fp32, padded (w,1,32,32), are written by mhc_pre's root through tensor
    accessors (kernels/mhc_pre/sync.cpp) and read by no kernel on the packed t <= 4 path (mhc_post reads the fused coef
    tile). In L1 they take 4 x 4 KB per bank below the served native-B=32 construction's floor (K=2 / K=3 were 128 B /
    1.8 KB short on the TP8 mesh)."""
    dram = ttnn.DRAM_MEMORY_CONFIG

    def empty(shape, *a, **kw):
        mem = a[3] if len(a) > 3 else kw.get("memory_config")
        dt = a[0] if a else kw.get("dtype")
        if (tuple(int(v) for v in shape) in ((w, 1, 32, 32), (w, 1, 4, 1), (w, 1, 4, 4)) and dt == ttnn.float32 and mem is not None
                and mem.buffer_type == ttnn.BufferType.L1 and not mem.is_sharded()):  # fmt: skip
            if len(a) > 3:
                a = a[:3] + (dram,) + a[4:]
            else:
                kw["memory_config"] = dram
        return orig(shape, *a, **kw)

    return empty


# the knob attributes the TP KDA layer constructors write onto a layer's shared attention object
SHARED_ATTN_KNOBS = ("kda_rec_ugroups", "kda_rec_spread", "kda_pre_rsplit", "kda_pre_spread", "kda_rec_rms_sem")


@contextlib.contextmanager
def env(**kv):
    """set GLM53_* knobs around a build / call (read at program build; None = unset); always restored (memory: env
    leak via resident builds)"""
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def mtp_mhc_args(source, cfg, sub: str) -> dict:
    """MhcPre arguments that make the mHC sublayer wrappers a PLAIN pre-norm residual (layer 45 has no hc_* tensors;
    design.md §3): fn = 0 -> z = bias; pre = sigmoid(ln 1/3) = 1/4 per stream (4 identical streams -> h = x), post =
    2 sigmoid(0) = 1, comb = softmax(0) = 1/4 (doubly stochastic, a Sinkhorn fixed point) with hc_eps = 0 -> every
    stream X' = y + x."""
    n, d = 4, cfg.hidden_size
    norm = "input_layernorm.weight" if sub == "attn" else "post_attention_layernorm.weight"
    return dict(
        fn_t=torch.zeros(n * d, (2 + n) * n), scale=torch.ones((2 + n) * n),
        bias=torch.tensor([math.log(1.0 / 3.0)] * n + [0.0] * n + [0.0] * (n * n)),
        norm_w=source.host(MTP_L, norm).float().reshape(-1),
        hc_eps=0.0, x_eps=cfg.rms_norm_eps, h_eps=cfg.rms_norm_eps, iters=cfg.hc_sinkhorn_iters, hidden=d, n=n,
    )  # fmt: skip


def norm_tile(mesh, w: torch.Tensor) -> ttnn.Tensor:
    """a [4096] norm weight as the head's [32, 4096] fp32 row-0 tile tensor (replicated)"""
    wt = torch.zeros(TILE, HIDDEN)
    wt[0] = w.float().reshape(-1)
    return ttnn.from_torch(wt, dtype=FP32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM,
                           mesh_mapper=_repl(mesh))  # fmt: skip


_L1_SEEN: dict = {}


def l1_debug(tag: str, mesh=None) -> None:
    """GLM53_SPEC_L1_DEBUG=1 (diagnostics): the L1 buffers allocated since the previous call (ttnn reports: address,
    bytes per bank), the lowest live L1 address (the static-CB ceiling of every later program) and the Python owners of
    the new ones (gc walk: shape, dtype, layout)"""
    if os.environ.get("GLM53_SPEC_L1_DEBUG") != "1" or mesh is None:
        return
    import gc

    bufs = {(int(b.address), int(b.max_size_per_bank)) for b in ttnn._ttnn.reports.get_buffers(mesh)
            if b.buffer_type == ttnn.BufferType.L1}  # fmt: skip
    new = sorted(bufs - set(_L1_SEEN.get("bufs", set())))
    owners = {}
    if new:
        gc.collect()
        want = {a for a, _ in new}
        # (nanobind tensors are not gc-tracked: walk the values of the tracked containers, name the referring key)
        for o in gc.get_objects():
            items = o.items() if isinstance(o, dict) else enumerate(o) if isinstance(o, (list, tuple)) else ()
            try:
                for k, v in items:
                    if isinstance(v, ttnn.Tensor) and v.storage_type() == ttnn.StorageType.DEVICE and v.is_allocated():
                        a = int(v.buffer_address())
                        if a in want and a not in owners:
                            owners[a] = (str(k)[:40], tuple(int(x) for x in v.padded_shape), str(v.dtype).split(".")[-1],
                                         str(v.memory_config().memory_layout).split(".")[-1])  # fmt: skip
            except Exception:  # noqa: BLE001 (gc walk)
                continue
    low = min(bufs)[0] if bufs else None
    print(f"SPEC_L1 {tag}: {len(bufs)} L1 buffers, lowest {low}; new {[(a, n, owners.get(a)) for a, n in new][:10]}",
          flush=True)  # fmt: skip
    _L1_SEEN["bufs"] = bufs


class MtpLayer:
    """Layer 45 (the MTP block's attention + MoE) as Stage 7b's TPMla + Stage 14b's TPMoE on every chip, the identity
    mHC wrappers (mtp_mhc_args), its own paged latent cache; the fabric phases of a main sparse_dsa layer (`share`)."""

    def __init__(self, mesh, cfg, source, ocfg, *, ts, share=None):
        from models.autoports.zai_org_glm_5_3_flash.tt import ep_experts as EPX
        from models.autoports.zai_org_glm_5_3_flash.tt import tp_dsa as TPD
        from models.autoports.zai_org_glm_5_3_flash.tt import tp_mla as TPM

        self.mesh = mesh
        self.attn = TPD.build_mla_attn(mesh, cfg, source, MTP_L, ocfg)
        # the K blocks of the main sparse_dsa layers (TPDsaLayer._build_mla): on the 32-chip replica (2 heads per chip)
        # TP8's blocks give W_uv (K block 16) q_b's 16-tile bfp8 ring at W_uv's 280-KB budget instead of sharing W_uk's
        # 8-tile one: +136 KB of reader CBs -- the 8x4 MTP rows' clash (mk137: the MTP MLA program's CB end 1291456 on
        # its reader cores, 999,488 B of CBs vs the main layers' 886,656 there)
        kb = TPD.MLA_KB_SHARD if int(mesh.get_num_devices()) <= 8 else TPD.MLA_KB_SHARD_WIDE
        self.mla = TPM.TPMla(mesh, mtp_mhc_args(source, cfg, "attn"), self.attn, kb=kb,
                             rings=TPD.MLA_RINGS_SHARD, fabrics=share.mla.fabrics if share else None, ts=ts)  # fmt: skip
        for name in ("ds_qb", "ds_o"):  # (TPDsaLayer: the per-chip slices live in the head shard)
            ttnn.deallocate(getattr(self.attn, name).weight)
        for name in ("w_uk", "w_uv"):
            ttnn.deallocate(getattr(self.attn, name))
        moe = EPX.build_moe_ep(mesh, cfg, source, MTP_L, ocfg)
        # (GLM53_MOE_SHARE_M, as TPDsaLayer: the main layers' shared-expert intermediate m, not an L1 buffer of its own)
        shm = share.moe.mk.persist if share is not None and os.environ.get("GLM53_MOE_SHARE_M", "1") == "1" else None
        self.moe = TPD.TPMoE(mesh, mtp_mhc_args(source, cfg, "ffn"), moe, ts=ts,
                             fabrics=share.moe.fabrics if share else None, persist=shm)  # fmt: skip


class MkEh:
    """x0 = eh_proj(cat[enorm(e), hnorm(h)]) for t rows (design.md §3), replicated on every chip:
    program A (64 owners: kernels/spec/eh_dm0.cpp EH_IN=1 + kernels/head/head_dm1.cpp + head_c.cpp QUARTER = 1) writes
    enorm(E[token]) into hcat columns 0..4095; program B (the same owners on the h input, EH_IN=2 the main head's bf16
    h / EH_IN=3 a stacked residual row, + 7a's DS readers on eh_proj [8192, 4096] bfloat8_b) writes hnorm(h) into hcat
    columns 4096.. and streams the projection -> out [1, 1, t, 4608] bf16 DRAM (columns >= 4096: DS padding)."""

    def __init__(self, mesh, source, cfg, ocfg, *, kb: int = 8, readers: int = 3):
        import copy

        from models.autoports.zai_org_glm_5_3_flash.tt.mk_dense import rect_placement
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import MMPhase, ReaderPhases
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import DSLinear
        from models.autoports.zai_org_glm_5_3_flash.tt.precision import hifi2_fp32

        self.mesh, self.eps = mesh, cfg.rms_norm_eps
        ocfg = copy.copy(ocfg)
        ocfg.ds_readers_per_role = ";".join(filter(None, [ocfg.ds_readers_per_role, f"eh_proj:{readers}"]))
        # [8192, 4096] bfloat8_b, replicated ([in, out]); the cache's own dtype (the generic dtype map names bf16 for
        # every *_proj outside the attention / MLP groups: the manifest entry is the authority here)
        w = source._load(source._entry(MTP_L, "eh_proj.weight"), mesh)
        assert [int(v) for v in w.shape][-2:] == [2 * HIDDEN, HIDDEN], w.shape
        self.ds = DSLinear(mesh, w, role="eh_proj", cfg=ocfg, ckc=hifi2_fp32(mesh), dtype=BF16)
        self.readers = ReaderPhases(mesh, [MMPhase(self.ds, kb=kb, wait=(SEM_HN, 64))], readers_per_bank=readers)
        self.place = rect_placement(mesh, {(int(c.x), int(c.y)) for c, _, _ in self.readers.assign})
        self.owners = self.place["k"]
        from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre import merged_ranges

        self.ocores = merged_ranges([(int(c.x), int(c.y)) for c in self.owners])
        self.owner_noc = [(int(v.x), int(v.y)) for v in (mesh.worker_core_from_logical_core(c) for c in self.owners)]
        self.w_e = norm_tile(mesh, source.host(MTP_L, "enorm.weight"))
        self.w_h = norm_tile(mesh, source.host(MTP_L, "hnorm.weight"))
        self.hcat, self.out = {}, {}

    def buffers(self, t: int):
        """hcat [1, 1, t, 8192] bf16 and the projection output (DRAM; allocated once per t, before any capture)"""
        if t not in self.hcat:
            self.hcat[t] = ttnn.empty((1, 1, t, 2 * HIDDEN), BF16, ttnn.TILE_LAYOUT, self.mesh, DRAM)
            self.out[t] = self.readers.outputs(t, DRAM)[0]
        return self.hcat[t], self.out[t]

    def _owner_parts(self, *, src, mode: int, w, hcat, t: int, page0: int, table=None, sel=None, readers=None,
                     row0=None, sel_rows: bool = False, flat: bool = False):  # fmt: skip
        """sel_rows (GLM53_SPEC_MULTI, eh_dm0.cpp EH_SEL_ROWS): output row u's source row = selector word u (the chained
        MTP launch: row u = user u, its catch-up row u (K + 1) + n_u); flat (EH_FLAT, mode 3): the source is a FLAT
        residual [1, 1, T', 16384] (an MTP launch wider than 4 rows)"""
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_head import f32_bits

        acc = lambda tt: list(ttnn.TensorAccessorArgs(tt).get_compile_time_args())  # noqa: E731
        flat_o = [v for xy in self.owner_noc for v in xy]
        flat_r = [v for xy in (readers or []) for v in xy]
        rt0, rt1 = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for k, c in enumerate(self.owners):
            rt0[c.x][c.y] = [src.buffer_address(), w.buffer_address(), hcat.buffer_address(), 2 * k, t, page0,
                             table.buffer_address() if table is not None else 0,
                             sel.buffer_address() if sel is not None else 0, len(readers or [])] + flat_r  # fmt: skip
            if row0 is not None:  # EH_ROW0: the row-offset page, after the readers' NoC pairs
                rt0[c.x][c.y] = list(rt0[c.x][c.y]) + [row0.buffer_address()]
            rt1[c.x][c.y] = [t, k] + flat_o
        defs = [("HEAD_SEM", str(SEM_HEAD)), ("HEAD_HN_SEM", str(SEM_HN)), ("EH_IN", str(mode)),
                ("SPEC_REV", str(SPEC_REV)), ("EH_VOCAB", str(int(table.shape[-2]) if table is not None else 1 << 30))
                ] + ([("EH_SEL", "1")] if sel is not None else []) + ([("EH_ROW0", "1")] if row0 is not None else [])  # fmt: skip
        if sel_rows:
            assert sel is not None, "EH_SEL_ROWS: the selector page"
            defs.append(("EH_SEL_ROWS", "1"))
        if flat:
            assert mode == 3, ("EH_FLAT: the stacked-residual source mode", mode)
            defs.append(("EH_FLAT", "1"))
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)  # noqa: E731
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True)  # fmt: skip
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for cb in (0, 1, 2, 4):  # CB_X, CB_W, CB_M, CB_G: exact fp32 to DST (as MkHead)
            modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        kd = lambda path, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=os.path.join(KDIR, path), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.ocores, compile_time_args=ct, runtime_args=rt, defines=defs, config=cfg)  # fmt: skip
        kernels = [
            kd("spec/eh_dm0.cpp", acc(src) + acc(w) + acc(hcat) + acc(table if table is not None else src), rt0,
               dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)),
            kd("head/head_dm1.cpp", [], rt1, dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default)),
            kd("head/head_c.cpp", [f32_bits(1.0 / HIDDEN), f32_bits(self.eps), f32_bits(1.0)], ttnn.RuntimeArgs(), cc),
        ]  # fmt: skip
        cb = lambda i, dt, page, n: ttnn.CBDescriptor(total_size=page * n, core_ranges=self.ocores,  # noqa: E731
                                                      format_descriptors=[ttnn.CBFormatDescriptor(
                                                          buffer_index=i, data_format=dt, page_size=page)])  # fmt: skip
        cbs = [cb(0, FP32, 4096, 8), cb(1, FP32, 4096, 2), cb(2, FP32, 4096, 2), cb(3, FP32, 4096, 1),
               cb(4, FP32, 4096, 2), cb(5, BF16, 2048, 2), cb(6, ttnn.uint32, 64 * 128, 1), cb(7, ttnn.uint32, 160, 1),
               cb(8, ttnn.uint32, 4096 + 512 + 64, 1)]  # fmt: skip
        sems = [ttnn.SemaphoreDescriptor(id=SEM_HEAD, core_ranges=self.ocores, initial_value=0)]
        io = [src, w, hcat] + ([table] if table is not None else []) + ([sel] if sel is not None else [])
        io += [row0] if row0 is not None else []
        return kernels, cbs, sems, io

    def norm_e(self, tokens, table, *, t: int, sel=None, sel_rows: bool = False):
        """program A: hcat[:, :4096] = enorm(E[token of row t (or of row `sel`; sel_rows: of row sel[t])])"""
        hcat, _ = self.buffers(t)
        kernels, cbs, sems, io = self._owner_parts(src=tokens, mode=1, w=self.w_e, hcat=hcat, t=t, page0=0,
                                                   table=table, sel=sel, sel_rows=sel_rows)  # fmt: skip
        ttnn.generic_op(io, checked(ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)))

    def norm_h_proj(self, src, *, mode: int, t: int, sel=None, row0=None, sel_rows: bool = False):
        """program B: hcat[:, 4096:] = hnorm(h rows), then out = hcat @ eh_proj (the DS readers wait for the 64 owners)
        -> out [1, 1, t, 4608] bf16 DRAM. row0 (mode 2, kernels/spec/eh_dm0.cpp EH_ROW0): a [1, 32] uint32 DRAM page
        whose word 0 = r0 -> rows r0 .. r0 + t - 1 of a taller bf16 source (the TP prefill's final-norm tap), within one
        tile row (the caller's rule); device-resident, so a captured launch serves every offset. Mode 3 on a FLAT
        residual (GLM53_SPEC_MULTI: an MTP launch wider than 4 rows) -> EH_FLAT; sel_rows -> EH_SEL_ROWS"""
        assert row0 is None or mode == 2, (mode, row0)
        hcat, out = self.buffers(t)
        rd = self.readers
        flat = mode == 3 and int(src.shape[-1]) == 4 * HIDDEN
        kernels, cbs, sems, io = self._owner_parts(src=src, mode=mode, w=self.w_h, hcat=hcat, t=t,
                                                   page0=HIDDEN // TILE, sel=sel, readers=rd.noc_coords(),
                                                   row0=row0, sel_rows=sel_rows, flat=flat)  # fmt: skip
        # the DS readers read GLM53_MK_W_DELAY_US per call: unset here, else this program's key follows whichever layer
        # kind's knobs the previous program left in the env (the bootstrap right after a step's MTP layer would launch
        # a key no warm-up call compiled -> "not yet in program cache" inside the capture)
        with env(GLM53_MK_W_DELAY_US=None):
            r_kernels, r_cbs = rd.parts([hcat], [out])
        sems = sems + [ttnn.SemaphoreDescriptor(id=SEM_HN, core_ranges=rd.cores, initial_value=0)]
        io = io + [self.ds.weight, out]
        ttnn.generic_op(io, checked(ttnn.ProgramDescriptor(kernels=kernels + r_kernels, semaphores=sems,
                                                           cbs=cbs + r_cbs)))  # fmt: skip
        return out

    def expand(self, x0, res, *, t: int):
        """EhExpand: x0 [1, 1, t, 4608] bf16 -> the stacked residual [t, 1, 4, 4096] fp32 (x0 in all 4 streams); a FLAT
        residual [1, 1, t, 16384] (GLM53_SPEC_MULTI: an MTP launch wider than 4 rows) -> eh_expand.cpp EH_FLAT"""
        acc = lambda tt: list(ttnn.TensorAccessorArgs(tt).get_compile_time_args())  # noqa: E731
        rt = ttnn.RuntimeArgs()
        for k, c in enumerate(self.owners):
            rt[c.x][c.y] = [x0.buffer_address(), res.buffer_address(), 2 * k, t]
        defs = [("SPEC_REV", str(SPEC_REV))]
        if int(res.shape[-1]) == 4 * HIDDEN:
            defs.append(("EH_FLAT", "1"))
        kd = ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, "spec", "eh_expand.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.ocores,
            compile_time_args=acc(x0) + acc(res), runtime_args=rt, defines=defs,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                     noc=ttnn.NOC.RISCV_0_default))  # fmt: skip
        cb = ttnn.CBDescriptor(total_size=2048 + 4096 + 64, core_ranges=self.ocores, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=8, data_format=ttnn.uint32, page_size=2048 + 4096 + 64)])  # fmt: skip
        ttnn.generic_op([x0, res], checked(ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cb])))


class SpecCtl:
    """kernels/spec/spec_ctl.cpp on one core per chip: ACCEPT / BOOT / GATHER (fixed-length runtime args)."""

    MODES = dict(accept=1, boot=2, gather=3)

    def __init__(self, mesh, K: int, groups: int = 0):
        self.mesh, self.K = mesh, K
        # GLM53_SPEC_MULTI: groups = G users (spec_ctl.cpp SPEC_GROUPS; 0 = the single-user kernel, unchanged)
        self.groups = int(groups)
        assert self.groups == 0 or 1 <= self.groups * (K + 1) <= 32, (self.groups, K)
        self.core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    NARGS = 27  # 10 + the DSA ring count + 16 ring bases (ACCEPT; kernels/spec/spec_ctl.cpp; SPEC_GROUPS alike)

    def __call__(self, mode: str, args: list, io: list, rings: list = ()):
        """rings: the slot-0 base of every main sparse_dsa layer's ring tensor (<= 16); with groups (SPEC_REV 4) the
        kernel addresses user u's ring at slot u (K + 1) of each tensor -- R args for any G (was R x G <= 64: G 8 x 11
        layers did not fit)"""
        nmax = 16
        assert len(args) <= 10 and len(rings) <= nmax, (len(args), len(rings))
        per = len(rings)
        rt = ttnn.RuntimeArgs()
        rt[0][0] = ([int(a) for a in args] + [0] * (10 - len(args)) + [per] + [int(a) for a in rings]
                    + [0] * (nmax - len(rings)))  # fmt: skip
        defs = [("SPEC_MODE", str(self.MODES[mode])), ("SPEC_REV", str(SPEC_REV))]
        if self.groups:
            defs.append(("SPEC_GROUPS", str(self.groups)))
        scr = 2048 if self.groups else 1024  # (SPEC_GROUPS: the assembled pages past the single-user slots)
        kd = ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, "spec", "spec_ctl.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.core,
            compile_time_args=[self.K], runtime_args=rt,
            defines=defs,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                     noc=ttnn.NOC.RISCV_0_default))  # fmt: skip
        cb = ttnn.CBDescriptor(total_size=scr + 64, core_ranges=self.core, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=scr + 64)])  # fmt: skip
        ttnn.generic_op(io, checked(ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cb])))


def head_clone(base, t: int, *, norm_w=None, view=None):
    """an MkHead on the base head's weights / readers / select core / fabric phase (as MkHead.width_view): rows 0..t-1,
    top-1 (greedy, lowest-id tie break), no sampling, NO position advance (call with pos=None), its OWN token / sampled /
    landing pages (DRAM), h / outs = row views of the base's buffers (no new L1); norm_w: another final-norm weight
    (the MTP head: shared_head.norm)"""
    from models.autoports.zai_org_glm_5_3_flash.tt.mk_head import MkHead

    assert base.h is not None, "the base head runs (eagerly) before its clones"
    o = MkHead.__new__(MkHead)
    o.__dict__.update(base.__dict__)
    o.t = t
    o.h = view(base.h, t)
    o.outs = [view(x, t) for x in base.outs]
    o.topk, o.sample, o.samp_par, o.samp_rng, o.probe = 1, False, None, None, None
    mesh = base.device
    o.topk_out = dram_rows(mesh, torch.zeros(32, 64))
    o.sampled = dram_rows(mesh, torch.zeros(32, 64))
    o.topk_land = None  # allocated by the first (eager) call
    if norm_w is not None:
        o.norm_w = norm_w
    return o


class SpecMTP:
    """The speculative step of ONE user (slot 0) with K drafts on a StackBackend (TP mesh, native B = 32 with a
    weight cache — the served construction). Build after allocate_kv_cache + allocate_decode_io; prepare() (eager
    compile, before the backend's first trace capture: StackBackend.prepare_decode calls it); capture(); then the host
    API enter() / step() / read() / leave(). GLM53_SPEC_MULTI=1 (groups = G > 0): G users on the group bases `slots`
    in the same one-replay step -- enter(rows) / set_page_tables(rows) take the users' rows, read() / leave() return
    their G records (group order)."""

    def __init__(self, be, kv, *, K: int, drafter: str = "mtp"):
        from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (
            FdTileCacheSource,
            OptimizedDecoderConfig,
        )

        # drafter "mtp": the MTP head drafts on device (K <= 3); "none" (GLM53_SPEC_DRAFTER): the step is the verify +
        # ACCEPT only, the K drafts come from outside (host-forced rows in the identity jobs; the DFlash2 block drafter)
        # -- K <= 7: the t = K+1 <= 8 verify on the native w8 programs (flat residual, the full width's fabric phases);
        # "dflash2": the DFlash2 block drafter INSIDE the step (tt/dflash2_kind.py: the verify's taps on the 9 tap
        # layers -> the drafter's context K / V -> the block -> the drafts into token rows 1..K), K <= 7, VERIFY_W=8
        assert drafter in ("mtp", "none", "dflash2"), drafter
        # K <= 7 for "none" / "dflash2": the verify is t = K + 1 <= 8 rows (the native w8 programs) and the DFlash2
        # drafter's block is 8 rows = the bonus + 7 mask rows (its training block); K = 8 would need a 9-row verify
        assert 1 <= K <= (3 if drafter == "mtp" else 7), (K, drafter)
        assert (
            be.fl is not None and be.chunk == be.max_batch and be.max_batch >= K + 1
        ), "speculative step: the TP stack with the shared 32-row state (native_t / B <= 4 single chunk)"
        # GLM53_SPEC_MULTI (doc/perf/mtp_multi_user_design.md section 6, batch 2: fixed G): G = GLM53_SPEC_MULTI_USERS
        # users in ONE step -- user u in generator slot u (K + 1) (its group base), its verify group rows u (K + 1)
        # .. u (K + 1) + K (token rows, positions c_u + j, its page-table row, candidate KDA states), selector word u,
        # record page u, chained MTP row u, its own private MTP cache; built with the batch-1 kernels' groups
        # (REC_CHAIN_GROUPS, KDA_VERIFY_GROUPS, SPEC_GROUPS, MLA_DSA_GROUP_ROWS) + MLA_ATT_GROUP_ROWS / EH_SEL_ROWS /
        # EH_FLAT. 0 = the single-user step (every program and page below unchanged)
        self.groups = G = spec_multi_groups()
        if G:
            assert drafter == "mtp", ("GLM53_SPEC_MULTI: the MTP drafter", drafter)
            assert (
                os.environ.get("GLM53_SPEC_MTP_PREFILL", "1") != "launch"
            ), "GLM53_SPEC_MULTI: the in-trace MTP-cache fill (the launch-mode prefill indexes one user's rows)"
        self.be, self.kv, self.K, self.w = be, kv, K, (K + 1) * max(1, G)
        self.slots = tuple(u * (K + 1) for u in range(max(1, G)))  # the speculating users' generator slots
        self.drafter = drafter
        # GLM53_SPEC_VERIFY_W: the verify's program width wv. Unset: K+1 (the narrow t = K+1 programs). "plain": the lone
        # SPECULATIVE plain width plain_w = the smallest decode width >= K+1 (n29h6: with a width-1 bucket the lone plain
        # width is 1 < K+1, so a speculable lone user's plain steps run at plain_w instead -- tt/generator.py
        # spec_lone_width; w1 serves only the non-speculable rows and MTP off); a number must equal plain_w. With wv
        # = the plain width, the verify's KDA / MLA / MoE launches have the plain decode's row count, tiling and knobs;
        # rows K+1 .. wv-1 are inactive (SpecCtl writes pos_v -1 past row K; masks 0; the verify MoE routes the live rows
        # of pos_v). The KDA layers still run their own verify-mode programs (KDA_VERIFY / REC_CHAIN) and the sparse_dsa
        # MkMla runs MLA_DSA_ONE_SLOT on one index core, so equality with the plain decode is a measured property
        # (s3_identity rollback / DSA-state rows), not an identity of programs. GLM53_MKX_XSL_SEED (a per-kind knob:
        # GLM53_STACK_KNOB_SET "s:" / "a:" entries) makes a row's routed-expert sum independent of the co-rows' routing.
        # (spec_widths: the widths' rules and their GLM53_SPEC_ON_SHARD defaults)
        sw = spec_widths(K, be.chunk, tuple(be.narrow_widths) + tuple(be.native_widths), groups=G)
        self.plain_w = sw["lone"]  # the speculative lone user's plain decode width (the lossless twin of its verify)
        self.wv = sw["wv"]
        # the MTP launches' widths: catch-up rows 0..K at cu (rows past K inactive), each chained row at ch (row 0 live)
        # (GLM53_SPEC_MULTI: cu = wv = G (K + 1), every group's rows; ch >= G, chained row u = user u)
        self.mw_cu, self.mw_ch, self.mtp_mode = sw["cu"], sw["ch"], sw["mode"]
        if G:
            assert self.w == self.wv <= be.max_batch, ("GLM53_SPEC_MULTI verify width", self.wv, self.w, be.max_batch)
        else:
            assert self.w <= self.wv <= 8 and be.max_batch >= self.wv, ("verify width", self.wv, self.w, be.max_batch)
        self.flat = self.wv > 4  # t > 4: the flat residual [1, 1, t, 16384] and the native-width programs
        self.mesh = be.mesh
        mesh, w, wv = self.mesh, self.w, self.wv
        # the verify MoE's live rows (MKX_LIVE, as the plain decode's routed experts): pos_v itself (rows 0..K >= 0)
        self.vlive = (
            True if os.environ.get("GLM53_MOE_LIVE", "1") == "1" and be.n_chunks == 1 else None
        )  # (pos_v below)
        self.source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        self.ocfg = OptimizedDecoderConfig()
        cfg = be.cfg
        l1_debug("spec init start", mesh)
        # device pages (DRAM, replicated): token rows = the backend's token tensor (shared with plain decode)
        self.tin = be.head.topk_out
        z32 = lambda fill: torch.full((1, 32), fill, dtype=torch.int32)  # noqa: E731
        self.sel = dram_rows(mesh, z32(0))
        self.rec = dram_rows(mesh, torch.zeros(max(1, G), 64, dtype=torch.int32))  # (one record page per user)
        self.pos_v = dram_rows(mesh, z32(-1), ttnn.int32)
        if self.vlive is not None:
            self.vlive = self.pos_v
        self.mpos_cu = dram_rows(mesh, z32(-1), ttnn.int32)
        self.mpos_ch = [dram_rows(mesh, z32(-1), ttnn.int32) for _ in range(K - 1)] if drafter == "mtp" else []
        io = kv.extra["io"]
        # GLM53_DSA_SHARD (GLM53_SPEC_ON_SHARD=1, Stage 14 dsa/spec_on_shard_design.md): the main sparse_dsa layers'
        # verify reads every chip's LOCAL table (pt_v [32, bps_local] per chip: rows 0..K = slot 0's local row, written
        # from the physical row by set_page_table); the MTP layer keeps its replicated dense cache over the PHYSICAL
        # global block ids (pt_m, C x nb_local blocks; v0 scope: the dense MTP cap -- design section 4 at long context)
        self.shard = kv.extra.get("dsa_shard")
        # the MTP layer's latent cache PRIVATE to the speculating slot (GLM53_SPEC_MTP_CACHE=private, the default):
        # position p in block p // 64 of a cache sized by the context speculation runs at (GLM53_SPEC_MAX_CTX, else the
        # cache length) -- pt_m a constant identity row; "pool" = M0's cache over the request pool's PHYSICAL block ids
        # (C x nb_local blocks: 4 GiB per chip at 1 M x 4; pt_m = the slot's physical row)
        self.mtp_private = os.environ.get("GLM53_SPEC_MTP_CACHE", "private") != "pool"
        if self.shard is not None:
            # the one-slot verify on the shard keeps the plain width's token lanes (mk_mla rev 62, the lane baton; only
            # one-slot builds read it: the plain programs are unchanged) -- MkMla refuses the t <= 4 verify on the
            # default 4 lanes (GLM53_MLA_DSA_LANES_T4) without it
            os.environ.setdefault("GLM53_MLA_DSA_BATON", "1")
            self.pt_v = ttnn.from_torch(torch.zeros(self.shard[0] * 32, kv.extra["bps_local"], dtype=torch.int32),
                                        dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM,
                                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))  # fmt: skip
            self.pt_m = dram_rows(mesh, torch.zeros(32, kv.blocks_per_seq, dtype=torch.int32), ttnn.int32)
        else:
            self.pt_v = dram_rows(mesh, torch.zeros(32, kv.blocks_per_seq, dtype=torch.int32), ttnn.int32)
            self.pt_m = self.pt_v
            if self.mtp_private:
                self.pt_m = dram_rows(mesh, torch.zeros(32, kv.blocks_per_seq, dtype=torch.int32), ttnn.int32)
        # the chained MTP launches' page table: the catch-up's (pt_m) for one user; GLM53_SPEC_MULTI: its own (pt_mc row
        # u = user u's MTP row, while pt_m row u (K + 1) + j = user u's: the two launches index the table by launch row)
        self.pt_mc = self.pt_m
        if G and drafter == "mtp":
            self.pt_mc = dram_rows(mesh, torch.zeros(32, kv.blocks_per_seq, dtype=torch.int32), ttnn.int32)
        self._pt_rows = None
        lb = float(be._kda0().lower_bound)
        m = torch.ones(1, 1, wv, 1)
        m[:, :, w:] = 0  # (padded verify rows: inactive, as the plain decode's idle rows)
        mk = lambda v, dt: ttnn.from_torch(
            v, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM, mesh_mapper=_repl(mesh)  # noqa: E731
        )
        self.masks_v = (mk(m, FP32), mk(m, BF16), mk(m * lb, FP32))
        stacked = lambda t: ttnn.from_torch(
            torch.zeros(t, 1, 4, HIDDEN),
            dtype=FP32,
            layout=ttnn.TILE_LAYOUT,  # noqa
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=_repl(mesh),
        )
        if (
            self.flat
        ):  # the verify residual = the width's own flat residual (the plain programs' buffer: one step at a time)
            wres = kv.extra.get("wres", {}).get(wv)
            self.R_v = kv.res[0] if be.chunk == wv else wres if wres is not None else ttnn.from_torch(
                torch.zeros(1, 1, wv, 4 * HIDDEN), dtype=FP32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM,
                mesh_mapper=_repl(mesh))  # fmt: skip
        else:
            self.R_v = stacked(wv)
        cu, ch = self.mw_cu, self.mw_ch
        if G:  # (GLM53_SPEC_MULTI: an MTP launch wider than 4 rows runs the flat residual, as the native widths)
            stacked = lambda t, _s=stacked: _s(t) if t <= 4 else ttnn.from_torch(  # noqa: E731
                torch.zeros(1, 1, t, 4 * HIDDEN), dtype=FP32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM,
                mesh_mapper=_repl(mesh))  # fmt: skip
        self.R_m = stacked(cu) if drafter == "mtp" else None
        self.R_c = [stacked(ch) for _ in range(K - 1)] if drafter == "mtp" else []
        # the MTP-cache prefill's token rows (prefill_mtp: row j = the prompt token after the launch's position j)
        self.pf_tok = dram_rows(mesh, torch.zeros(32, 64)) if drafter == "mtp" else None
        self.pf_row0 = dram_rows(mesh, torch.zeros(1, 32)) if drafter == "mtp" else None  # (MkEh EH_ROW0: word 0)
        # diagnosis only (GLM53_SPEC_DBG_IDS=1): every MTP row's routed experts copied out of the MoE program (catch-up
        # rows, then each chained row) -> host comparison with the CPU oracle's routing; off in every measured run
        self.dbg_ids = None
        if os.environ.get("GLM53_SPEC_DBG_IDS") == "1":
            ids = lambda t: ttnn.from_torch(torch.zeros(1, 1, t, 8, dtype=torch.int32), dtype=ttnn.uint16,  # noqa
                                            layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM,
                                            mesh_mapper=_repl(mesh))  # fmt: skip
            self.dbg_ids = [ids(cu)] + [ids(ch) for _ in range(K - 1)]
        dsa = [i for i, e in enumerate(be.seq) if e["sparse"] == "dsa"]
        share = be.seq[dsa[0]]["tp"] if dsa else None
        self.mtp = self.mtp_cache = self.eh = self.w_mtp = None
        self.pf_scratch, self._pfw = (0, 0), None
        self.mtp_cap_blocks = 0  # (GLM53_SPEC_MULTI + the private MTP cache: the blocks of each user's)
        if drafter == "mtp":
            # (GLM53_SPEC_MULTI: a width t > 4 other than the full width has no sparse_dsa fabric phase of its own -- it
            # runs on the full width's, TPMla.fabric -- so only the shared phases are named)
            mts = tuple(sorted({cu, ch})) if not G else tuple(sorted(t for t in {cu, ch} if t <= 4 or t == be.chunk))
            self.mtp = MtpLayer(mesh, cfg, self.source, self.ocfg, ts=mts, share=share)
            l1_debug("after MtpLayer", self.mesh)
            # (+ scratch pages past the request blocks: the TP prefill's in-trace MTP fill writes an unarmed chunk's
            # rows there -- TPPrefill._mtp_write; one bucket's worth of the largest GLM53_PREFILL_BUCKETS bucket)
            blk = int(self.mtp.attn.block_size)
            if self.mtp_private:
                cap = min(
                    int(kv.max_seq_len), int(os.environ.get("GLM53_SPEC_MAX_CTX", "0") or 0) or int(kv.max_seq_len)
                )
                nb = -(-cap // blk)
                if G:  # one private cache per user: user u's blocks u nb .. u nb + nb - 1 (G x GLM53_SPEC_MAX_CTX rows)
                    self.mtp_cap_blocks = nb
                    nb = G * nb
            else:
                nb = kv.num_blocks if self.shard is None else self.shard[0] * self.shard[1]
            bmax = max(int(v) for v in os.environ.get("GLM53_PREFILL_BUCKETS", "256,512,2048").split(",") if v.strip())
            self.pf_scratch = (int(nb), max(1, -(-bmax // blk)))
            self.mtp_cache = self.mtp.attn.allocate_cache(nb + self.pf_scratch[1])
            if (
                self.mtp_private and not G
            ):  # (the constant row: positions past the cache's blocks -> scratch, never read)
                write_rows(mesh, self.pt_m, self.mtp_row(None).reshape(1, -1).repeat(32, 1), ttnn.int32)
            elif self.mtp_private:  # (GLM53_SPEC_MULTI: catch-up row u (K + 1) + j / chained row u = user u's cache)
                pm, pc = self._mtp_private_tables()
                write_rows(mesh, self.pt_m, pm, ttnn.int32)
                write_rows(mesh, self.pt_mc, pc, ttnn.int32)
            self.eh = MkEh(mesh, self.source, cfg, self.ocfg)
            for t in sorted({cu, ch}):
                self.eh.buffers(t)
            self.w_mtp = norm_tile(mesh, self.source.host(MTP_L, "shared_head.norm.weight"))
        self.ctl = SpecCtl(mesh, K, groups=G)
        l1_debug("after MkEh / ctl", self.mesh)
        self.df2 = None
        # the positions the speculative step may touch (None: the backend's max_seq_len); the dflash2 drafter's own
        # cache capacity GLM53_DF2_CACHE_CAP (the 1M line: 131,072, coordinator 2026-10-23 10:14:52Z) bounds it --
        # past it the request decodes plainly (generator spec_enter declines, the adapter leaves before the step).
        # GLM53_SPEC_MAX_CTX (the MTP drafter): the context bound past which speculation stops paying -- the MTP layer's
        # attention is dense over every position (+~2.4 ms per step per 32 K positions on the 8x4, chain_mk201), so a
        # long-context profile speculates below the bound and decodes plainly above it
        self.pos_cap = int(os.environ.get("GLM53_SPEC_MAX_CTX", "0") or 0) or None
        self._build_verify_stacks()
        l1_debug("after verify stacks", self.mesh)
        if drafter == "dflash2":
            from models.autoports.zai_org_glm_5_3_flash.tt.dflash2_kind import build_dflash2_kind

            assert self.flat, "dflash2: the verify runs at the lone user's plain w8 width (GLM53_SPEC_VERIFY_W=8)"
            # the drafter's own paged K / V cache covers the user's positions (+ the garbage rows past the step), up to
            # GLM53_DF2_CACHE_CAP positions when that is smaller than the backend's max_seq_len
            cap = int(kv.max_seq_len)
            ccap = int(os.environ.get("GLM53_DF2_CACHE_CAP", "0") or 0)
            if 0 < ccap < cap:
                cap = self.pos_cap = ccap
            self.df2 = build_dflash2_kind(mesh, K=K, cap=cap, tin=self.tin, rec=self.rec,
                                          log=lambda *a: print("S17-DF2", *a, flush=True))  # fmt: skip
            l1_debug("after dflash2 kind", self.mesh)
        self.heads = None  # clones made in prepare() (the base head's h exists after its first call)
        self.traces = {}
        self.counters = dict(steps=0, boots=0, captures=0, enters=0, leaves=0, slot_copies=0, prefill_rows=0,
                             prefill_launches=0)  # fmt: skip

    # ------------------------------------------------------------------ construction
    def _build_verify_stacks(self):
        """the main model's KDA layers at t = K+1 in VERIFY mode (programs only, the cached weights; kda_verify_sel set
        on the resident attention objects only while each layer's programs are built), the narrow width's fabric
        phases when that width exists (else one per KDA kind), the shared 32-row state (rows 0..K), the full width's L1
        pool aliased; the sparse_dsa layers are the full width's objects (fabric phases for t <= 4)"""
        from models.autoports.zai_org_glm_5_3_flash.tt import fabric_ar as FAR
        from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as SM
        from models.autoports.zai_org_glm_5_3_flash.tt import tp8_layer as TPL
        from models.autoports.zai_org_glm_5_3_flash.tt import tp8_sparse as TPSP
        from models.autoports.zai_org_glm_5_3_flash.tt import tp_stack as TPS

        be, kv, w = self.be, self.kv, self.wv  # (the verify's program width)
        wc = be._wc
        sliced = os.environ.get("GLM53_TP_XSL") == "1" and os.environ.get("GLM53_TP_XSL_SLICED") == "1"
        nseq = be.nseq.get(w)
        fabs, seq = {}, []
        for i, e in enumerate(be.seq):
            L = e["L"]
            if e["sparse"] == "dsa":
                # (t > 4: TPMla / TPMoE run on the full width's 32-row phases, as the native widths)
                assert self.flat or w in e["tp"].mla.fabrics, (
                    "sparse_dsa fabric phase for t",
                    w,
                    sorted(e["tp"].mla.fabrics),
                )
                seq.append(e)
                continue
            kind = SM.layer_kind(be.cfg, L)
            if self.flat:  # t > 4: the native width's knobs (StackBackend._build_width)
                SM.set_knobs(kind == SPARSE_KDA, be.tp, w)
            else:
                SM._narrow_knobs(kind == SPARSE_KDA, be.tp, w)
            if kind not in fabs:
                if self.flat:  # the full width's 32-row phase of this kind (whole-tile packets: the t > 4 consumer)
                    fabs[kind] = e["tp"].fabric
                elif nseq is not None:  # the narrow width w's phase of this kind
                    fabs[kind] = nseq[i]["tp"].fabric
                elif e["tp"].fabric is not None and e["tp"].fabric.b == w:  # a B = w backend: its own phase
                    fabs[kind] = e["tp"].fabric
                else:
                    f = FAR.FabricPhase(be.mesh, w, lanes=e["tp"].fabric.lanes, mode="cb")
                    f.probe()
                    fabs[kind] = f
            wts = wc[("dense", L)] if kind == DENSE_KDA else wc[("sparse", L)]
            wts.attn.kda_verify_sel = self.sel
            if self.groups:  # GLM53_SPEC_MULTI: kda_pre KDA_VERIFY_GROUPS = G (an int: kda_rec reads it as a flag)
                wts.attn.kda_verify_groups = self.groups
            # the TP layer constructors write their knobs onto the shared attention object: restored afterwards
            keep = {a: getattr(wts.attn, a) for a in SHARED_ATTN_KNOBS if hasattr(wts.attn, a)}
            try:
                # the verify rows are ONE user's chained rows (REC_CHAIN: one user group, whatever the served
                # width's GLM53_REC_UGROUPS, e.g. 14b's TP2 knob set "s:GLM53_REC_UGROUPS=2")
                # (GLM53_REC_DEPTH=s would build a second, unchained kda_rec lazily at call time, after kda_verify_sel)
                # GLM53_SPEC_MULTI: one user group per user (REC_CHAIN_GROUPS: each group chained from sel[u])
                assert os.environ.get("GLM53_REC_DEPTH", "1") != "s", "spec verify: GLM53_REC_DEPTH=s is not chained"
                with env(GLM53_REC_UGROUPS=str(self.groups) if self.groups else "1"):
                    if kind == DENSE_KDA:
                        tp = TPL.TPDenseKDALayer(be.mesh, be.cfg, L, self.source, self.ocfg, b=w, weights=wts,
                                                 fabric=fabs[kind])  # fmt: skip
                    else:
                        tp = TPSP.TPSparseKDALayer(be.mesh, be.cfg, L, self.source, self.ocfg, b=w,
                                                   moe=wc[("moe_rs", L) if sliced else ("moe", L)], weights=wts,
                                                   fabric=fabs[kind])  # fmt: skip
            finally:
                del wts.attn.kda_verify_sel
                if self.groups:
                    del wts.attn.kda_verify_groups
                for a, v in keep.items():
                    setattr(wts.attn, a, v)
            assert tp.mk.rec.verify_sel is self.sel and tp.mk.kda.verify_sel is self.sel, L
            assert not self.groups or (tp.mk.rec.chain_groups and tp.mk.rec.ug == self.groups
                                       and tp.mk.kda.verify_groups == self.groups), (L, "GLM53_SPEC_MULTI groups")  # fmt: skip
            seq.append(dict(L=L, mk=tp.mk, sparse=kind == SPARSE_KDA, kda=tp.attn, tp=tp))
        self.vseq = seq
        pool = kv.extra["pool"]
        per = []
        # alias mode (padded-shape keys) only when the full width is wider (native B = 32): a B = K+1 backend's full
        # width is this width -> the same (unpadded) request keys
        pool.alias = be.chunk != w
        try:
            for r in self._verify_runs():
                if seq[r[0]]["sparse"] == "dsa":
                    per.append((r, None))
                    continue
                # the verify layers on the full width's persistent hand-off buffers (one L1 set; programs of different
                # widths never run concurrently) — as StackBackend's native widths: a new L1 buffer here lowers the
                # floor under the existing programs' static CBs (a CB clash)
                for i in r:
                    SM._alias_persist(seq[i]["mk"], be.seq[r[0]]["mk"], w, be.chunk)
                    if hasattr(seq[i]["mk"], "experts"):
                        SM._alias_persist(seq[i]["mk"].experts, be.seq[i]["mk"].experts, w, be.chunk)
                cls = TPS.TPSparseStack if seq[r[0]]["sparse"] else TPS.TPStack
                per.append((r, cls(be.mesh, [seq[i]["mk"] for i in r], seq[r[0]]["tp"], t=w, res=self.R_v, pool=pool,
                                   live=self.vlive if seq[r[0]]["sparse"] else None)))  # fmt: skip
        finally:
            pool.alias = False
        self.vstacks = per

    def _make_heads(self):
        from models.autoports.zai_org_glm_5_3_flash.tt import mk_head as MKH
        from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import _rows_view

        be = self.be
        em = MKH.MkEmbed.__new__(MKH.MkEmbed)
        em.__dict__.update(be.embed.__dict__)
        em.t = self.wv
        self.embed_v = em
        self.heads = dict(verify=head_clone(be.head, self.wv, view=_rows_view))
        if self.drafter != "mtp":
            return
        self.heads.update(
            mtp_cu=head_clone(be.head, self.mw_cu, norm_w=self.w_mtp, view=_rows_view),
            mtp_ch=[head_clone(be.head, self.mw_ch, norm_w=self.w_mtp, view=_rows_view) for _ in range(self.K - 1)],
        )

    def _seed_persist(self):
        """the persistent L1 hand-off buffers the verify / MTP calls would allocate (MkMoE's ("m", t)): row views of the
        full width's (the native B = 32 programs sit within KB of the L1 floor; a new L1 buffer clashes with their
        static CBs) — every main sparse_dsa layer's MLA / MoE objects at t = K+1, and the MTP layer's MLA / MoE on the
        first main sparse_dsa layer's buffers (same shapes: layer 45 = layer 43's)"""
        from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import _rows_view, _seed_width

        be, full = self.be, self.be.chunk
        dsa = [e for e in be.seq if e["sparse"] == "dsa"]
        for e in dsa:
            for o in (getattr(e["mla"], "mk", None), getattr(e["moe"], "mk", None)):
                if o is not None:
                    _seed_width(o, self.wv, full)
                    _seed_width(o, self.w, full)  # (the MTP rows' width shares nothing here; harmless when wv == w)
        # the MTP layer's objects: (name, t) keys as t-row views of the first main sparse_dsa layer's (name, full); its
        # other keys (t-independent: MkMoE's shared intermediate per TILE-ROW count ("m", 32), the routed experts'
        # ("m", m_tiles)) as that layer's own tensors (the layers never run concurrently; each call writes the buffer
        # before reading it)
        if self.mtp is None:  # (drafter "none": no MTP layer)
            return
        pairs = (
            [
                (getattr(dsa[0]["moe"].mk, "persist", None), getattr(self.mtp.moe.mk, "persist", None), True),
                (
                    getattr(getattr(dsa[0]["moe"].mk, "experts", None), "persist", None),
                    getattr(getattr(self.mtp.moe.mk, "experts", None), "persist", None),
                    False,
                ),  # (keyed by m tiles)
                (getattr(dsa[0]["mla"].mk, "persist", None), getattr(self.mtp.mla.mk, "persist", None), True),
            ]
            if dsa
            else []
        )
        for src, dst, views in pairs:
            if src is None or dst is None or src is dst:
                continue
            for k, v in list(src.items()) if views else []:
                if isinstance(k, tuple) and len(k) == 2 and k[1] == full:
                    for t in sorted({self.mw_cu, self.mw_ch}):
                        dst.setdefault((k[0], t), _rows_view(v, t))
            for k, v in list(src.items()):
                dst.setdefault(k, v)

    def _verify_runs(self) -> list:
        """the verify's layer runs: the backend's (same-kind runs; sparse_dsa alone), split after every DFlash2 tap
        layer when the drafter is "dflash2" (the tap = the completed residual after that layer, read from the run's
        output residual; +5 launches per step: taps 5, 9, 24, 28, 33 sit inside 3-layer sparse_kda runs. The
        production path is an mhc_post output on the tap layers, no split)"""
        runs = self.be._runs()
        if self.drafter != "dflash2":
            return runs
        from models.autoports.zai_org_glm_5_3_flash.tt.dflash2_kind import TAP_LAYERS

        out = []
        for r in runs:
            cur = []
            for i in r:
                cur.append(i)
                if self.be.seq[i]["L"] in TAP_LAYERS and i != r[-1]:
                    out.append(cur)
                    cur = []
            if cur:
                out.append(cur)
        return out

    def _tap(self, r: list):
        """after the verify run r: its last layer's completed residual -> the drafter's tap slot"""
        from models.autoports.zai_org_glm_5_3_flash.tt.dflash2_kind import TAP_LAYERS

        L = self.be.seq[r[-1]]["L"]
        if self.df2 is not None and L in TAP_LAYERS:
            self.df2.tap(TAP_LAYERS.index(L), self.R_v)

    # ------------------------------------------------------------------ the device step (eager here; the same calls
    # run inside the captured traces)
    def _dsa(self, e, res, cache, pt, pos, t: int, verify: bool, dbg: int | None = None, dsa_state=None,
             live="verify"):  # fmt: skip
        """live: the MoE's live-row page (MKX_LIVE); "verify" = pos_v on a verify-mode call (the plain decode's rule), else
        none; a page tensor = that page (the MTP rows at the verify width: their own positions, rows < 0 route nowhere)
        """
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_mla import ATT_NKMAX
        from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import set_knobs

        set_knobs("dsa", self.be.tp, t)
        # the MTP layer's dense attention at t = 1 (the K >= 2 chain rows) past the one-lane dense capacity (a
        # GLM53_STACK_DSA backend's max_seq_len; MkMla: 'longer contexts need dsa='): 2 token lanes = the batched path
        mk = getattr(e["mla"], "mk", None)
        lanes = (t == 1 and dsa_state is None and mk is not None and mk.dsa is None
                 and int(pt.shape[-1]) * mk.attn.block_size > mk.chunks * TILE * ATT_NKMAX)  # fmt: skip
        with contextlib.ExitStack() as es:
            if verify:
                es.enter_context(env(GLM53_MLA_ATT_VERIFY="1"))
                if self.groups:  # GLM53_SPEC_MULTI: G row groups (mk_mla MLA_DSA_GROUP_ROWS / MLA_ATT_GROUP_ROWS)
                    es.enter_context(env(GLM53_MLA_VERIFY_GROUPS=str(self.groups)))
            if lanes:
                es.enter_context(env(GLM53_MLA_ATT_LANES_T4="2"))
            # dsa_state: a main sparse_dsa layer's DSA state under GLM53_STACK_DSA (None = the dense MkMla / the MTP
            # layer); the verify's rows all address slot 0 (mk_mla rev 56 MLA_DSA_ONE_SLOT), ACCEPT moves the rings
            pool = self.be._dsa_pools(self.kv)  # GLM53_DSA_POOL: the pooled intermediates (stack_model._dsa_layer)
            if pool is None:
                s1 = e["mla"](res, cache, pt, pos, t=t, keep=("s1",), stack=True, dsa_state=dsa_state)["s1"]
            else:
                # a narrow row count (the verify's w, the draft's w and 1) is served as views of the full-width
                # intermediates (alias), whatever the main pool's flag is at this moment (the draft runs outside
                # _verify's window): otherwise the pool keys on the exact shape and allocates a second L1 set per width
                pool.alias = self.be.chunk != t
                with pool.child():  # s1 from the persistent-hand-off pool (never aliased by a build intermediate)
                    s1 = e["mla"].mk.s1_buffer(t)
                with pool:
                    e["mla"](res, cache, pt, pos, t=t, stack=True, dsa_state=dsa_state, s1=s1)
        if isinstance(live, str):
            live = self.vlive if verify else None
        elif self.vlive is None:  # (GLM53_MOE_LIVE=0 / chunked steps: no MKX_LIVE program anywhere)
            live = None
        if pool is None:
            r = e["moe"](s1, t=t, out=res, live=live)
        else:
            with pool:
                r = e["moe"](s1, t=t, out=res, live=live)
        if dbg is not None and self.dbg_ids is not None:
            ttnn.copy(r["ids"], self.dbg_ids[dbg])
        if pool is None:
            ttnn.deallocate(s1)  # L1: must not stay allocated under the next program's CB region
        del s1

    def _verify(self):
        from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import _narrow_knobs, set_knobs

        be, kv, w = self.be, self.kv, self.wv  # (the verify's program width)
        self.embed_v(self.tin, [self.R_v])
        l1_debug("verify: start", self.mesh)
        pool = kv.extra["pool"]
        pool.alias = be.chunk != w  # (see _build_verify_stacks)
        try:
            dsa_pool = kv.extra.get("dsa_pool")  # the sparse_dsa layers' pool: its post / comb tiles go to DRAM too
            with verify_allocator(pool, w, os.environ.get("GLM53_SPEC_PC_DRAM", "1") == "1",
                                  pools=() if dsa_pool is None else (dsa_pool,)) as va:  # fmt: skip
                for r, stk in self.vstacks:
                    if stk is None:
                        self._dsa(be.seq[r[0]], self.R_v, kv.extra["latent"][r[0]], self.pt_v, self.pos_v, w, True,
                                  dsa_state=be._dsa_st(kv, r[0], 0))  # fmt: skip
                        self._tap(r)
                        continue
                    (set_knobs if self.flat else _narrow_knobs)(self.vseq[r[0]]["sparse"], be.tp, w)
                    stk([kv.states[0][i][1] for i in r], self.masks_v, [kv.states[0][i][0] for i in r])
                    va.reset()  # a stack BUILD left the patched allocator global: the MkMla calls see the real one
                    self._tap(r)
                    l1_debug(f"verify: after stack {r}", self.mesh)
        finally:
            pool.alias = False
        self.heads["verify"]([self.R_v], None)

    def _ctl_accept(self, boot: bool):
        h = self.heads["verify"]
        pos_src = self.be.pos if boot else self.pos_v
        mp = [m.buffer_address() for m in self.mpos_ch] + [0] * (3 - len(self.mpos_ch))
        args = [self.tin.buffer_address(), h.topk_out.buffer_address(), pos_src.buffer_address(),
                self.pos_v.buffer_address(), self.mpos_cu.buffer_address(), self.sel.buffer_address(),
                self.rec.buffer_address()] + mp  # fmt: skip
        # (distinct tensors by construction: never list.remove / `in` on ttnn tensors — == is the device EQ op)
        io = [self.tin, h.topk_out] + ([self.be.pos] if boot else []) + [self.pos_v, self.mpos_cu, self.sel, self.rec]
        # ACCEPT: the main sparse_dsa layers' slot-0 rings (GLM53_STACK_DSA; the verify's later-pool rows -> rows 0..3)
        self.ctl("boot" if boot else "accept", args, io + list(self.mpos_ch), rings=[] if boot else self._dsa_rings())

    def _dsa_rings(self) -> list:
        """slot 0's ring base address of every main sparse_dsa layer's DSA state (empty without DSA); GLM53_SPEC_MULTI:
        the same R bases -- spec_ctl.cpp SPEC_GROUPS (SPEC_REV 4) pages user u's ring at slot u (K + 1) = its group base
        (StackBackend._dsa_st: DSA slot = generator slot)"""
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_mla import dsa_slot_addrs

        return [dsa_slot_addrs(self.be._dsa_st(self.kv, i, 0), 1)[1] for i in sorted(self.kv.extra.get("dsa", {}))]

    def _draft(self):
        """the MTP catch-up (K+1 rows) + K-1 chained rows + GATHER (design.md §1 rows 6-21)"""
        self._catchup()
        for i in range(self.K - 1):
            self._chain(i)
        self._gather()

    def _catchup(self):
        """tokens a_j = the verify head's rows (BOOT: row 0 = b), h = the main head's h rows (views of one buffer); rows
        0..K at mpos_cu (c + j), launched at mw_cu rows (the rows past K: positions -1, inactive)"""
        be, w, eh, hd = self.be, self.mw_cu, self.eh, self.heads
        eh.norm_e(hd["verify"].topk_out, be.embed.table, t=w)
        x0 = eh.norm_h_proj(be.head.h, mode=2, t=w)
        eh.expand(x0, self.R_m, t=w)
        # (MoE live rows: pos_v -- rows 0..K live, as the catch-up's; the verify-width mode reads mpos_cu itself;
        # GLM53_SPEC_MULTI: every group's K + 1 rows at mpos_cu; the MTP attention per group: MLA_ATT_GROUP_ROWS)
        live = self.mpos_cu if self.mtp_mode in ("verify", "multi") else "verify"
        self._dsa(dict(mla=self.mtp.mla, moe=self.mtp.moe), self.R_m, self.mtp_cache, self.pt_m, self.mpos_cu, w, True,
                  dbg=0, live=live)  # fmt: skip
        l1_debug("catch-up: after the MTP layer", self.mesh)
        hd["mtp_cu"]([self.R_m], None)

    def _chain(self, i: int):
        """chained row i (draft depth i + 2): row 0 = d'(i+1) at MTP position c + n + 1 + i"""
        eh, hd = self.eh, self.heads
        if i == 0:  # d'1 and the raw block output of row n (the device selector)
            tok, src, sel = hd["mtp_cu"].topk_out, self.R_m, self.sel
        else:
            tok, src, sel = hd["mtp_ch"][i - 1].topk_out, self.R_c[i - 1], None
        t = self.mw_ch  # (> 1: the verify-width mode -- row 0 is the chained row, rows 1.. inactive: positions -1)
        # (GLM53_SPEC_MULTI: row u = user u -- the first chained launch reads catch-up row sel[u] = u (K + 1) + n_u of
        # the catch-up head's tokens / the MTP block's output (EH_SEL_ROWS), on the chain's page table pt_mc)
        rows = bool(self.groups) and sel is not None
        eh.norm_e(tok, self.be.embed.table, t=t, sel=sel, sel_rows=rows)
        x0 = eh.norm_h_proj(src, mode=3, t=t, sel=sel, sel_rows=rows)
        eh.expand(x0, self.R_c[i], t=t)
        live = self.mpos_ch[i] if self.mtp_mode in ("verify", "multi") else "verify"
        self._dsa(dict(mla=self.mtp.mla, moe=self.mtp.moe), self.R_c[i], self.mtp_cache, self.pt_mc, self.mpos_ch[i],
                  t, False, dbg=i + 1, live=live)  # fmt: skip
        l1_debug(f"chain {i}: after the MTP layer", self.mesh)
        hd["mtp_ch"][i]([self.R_c[i]], None)

    def _mtp_prefill_body(self, src):
        """the device part of one MTP-cache prefill launch of mw_cu rows (eager, or captured per tap: capture()'s "pf<c>"): row j
        = MTP position mpos_cu[j] (< 0: inactive) with the token in pf_tok row j and the final-norm hidden row r0 + j of
        src (bf16 [1, 1, C, 4096] DRAM; r0 = word 0 of pf_row0: MkEh EH_ROW0)"""
        be, eh, T = self.be, self.eh, self.mw_cu
        eh.norm_e(self.pf_tok, be.embed.table, t=T)
        x0 = eh.norm_h_proj(src, mode=2, t=T, row0=self.pf_row0)
        eh.expand(x0, self.R_m, t=T)
        # (verify mode: the launch's rows are consecutive positions of one user, causal inside the launch; the MoE's live
        # rows = these positions)
        self._dsa(dict(mla=self.mtp.mla, moe=self.mtp.moe), self.R_m, self.mtp_cache, self.pt_m, self.mpos_cu, T, True,
                  live=self.mpos_cu)  # fmt: skip

    def _mtp_prefill_launch(self, src, row0: int, toks: list, pos0: int, trace=None):
        """one MTP-cache prefill launch: rows j < len(toks) = MTP position pos0 + j, token toks[j], hidden row row0 + j of
        src (the rest inactive) -- the host writes three small pages, then the body runs (eager) or `trace` replays"""
        T = self.mw_cu
        n = len(toks)
        assert n <= T and row0 % TILE + T <= TILE, (n, T, row0)
        tok = torch.zeros(32, 64, dtype=torch.int32)
        pos = torch.full((1, 32), -1, dtype=torch.int32)
        if n:
            tok[:n, 0] = torch.tensor(toks, dtype=torch.int32)
            pos[0, :n] = int(pos0) + torch.arange(n, dtype=torch.int32)
        r0 = torch.zeros(1, 32, dtype=torch.int32)
        r0[0, 0] = int(row0)
        write_rows(self.mesh, self.pf_tok, tok)
        write_rows(self.mesh, self.mpos_cu, pos, ttnn.int32)
        write_rows(self.mesh, self.pf_row0, r0)
        if trace is None:
            self._mtp_prefill_body(src)
        elif isinstance(trace, tuple):  # mock
            trace[1]()
        else:
            ttnn.execute_trace(self.mesh, trace, cq_id=0, blocking=False)

    def mtp_row(self, phys_row, slot: int = 0) -> torch.Tensor:
        """the MTP cache's page row of the speculating slot: the private identity row (blocks 0 .. nb - 1, a scratch
        page past them) or, with GLM53_SPEC_MTP_CACHE=pool, the slot's PHYSICAL row. GLM53_SPEC_MULTI (private): user
        u's cache = blocks u nb .. u nb + nb - 1 for the group base `slot` = u (K + 1) (another slot: scratch only)"""
        if not self.mtp_private:
            return torch.as_tensor(phys_row).reshape(-1).to(torch.int64)
        nb, scr = self.pf_scratch
        j = torch.arange(int(self.kv.blocks_per_seq), dtype=torch.int64)
        if not self.groups:
            return torch.where(j < nb, j, torch.full_like(j, nb))
        cap, u = self.mtp_cap_blocks, self.group_of(slot)
        if u is None:
            return torch.full_like(j, nb)
        return torch.where(j < cap, u * cap + j, torch.full_like(j, nb))

    def group_of(self, slot) -> int | None:
        """GLM53_SPEC_MULTI: the user (group) whose base is generator slot `slot` (None: not a group base)"""
        if slot is None:
            return None
        s, g = int(slot), self.K + 1
        return s // g if self.groups and s % g == 0 and s // g < self.groups else None

    def _mtp_private_tables(self) -> tuple:
        """GLM53_SPEC_MULTI with the private MTP caches: the constant [32, blocks] page tables of the catch-up (row
        u (K + 1) + j = user u's row) and of the chained launches (row u = user u's); rows past the users: scratch"""
        g = self.K + 1
        rows = [self.mtp_row(None, u * g) for u in range(self.groups)]
        scratch = self.mtp_row(None, -1)
        pm = torch.stack([rows[r // g] if r < self.w else scratch for r in range(32)])
        pc = torch.stack([rows[r] if r < self.groups else scratch for r in range(32)])
        return pm, pc

    def pf_scratch_pages(self, n: int) -> torch.Tensor:
        """n MTP cache pages no request holds (the in-trace MTP fill of an unarmed chunk)"""
        assert 0 < int(n) <= self.pf_scratch[1], ("MTP fill: a bucket past the scratch pages (GLM53_PREFILL_BUCKETS)",
                                                  n, self.pf_scratch)  # fmt: skip
        return self.pf_scratch[0] + torch.arange(int(n), dtype=torch.int64)

    def pf_weights(self) -> dict:
        """the weights of the TP prefill's in-trace MTP fill (TPPrefill._mtp_fill): the embedding table, enorm / hnorm /
        the MTP block's input_layernorm as [1, 1, 1, 4096] bf16 rows, eh_proj (MkEh's DSLinear), the MTP attention's
        q_a | kv_a DSLinear and kv_a_layernorm (pf_mm / rms_norm, as MlaPrefill); loaded at the first (eager, compile)
        call, before any capture"""
        if self._pfw is None:
            assert self.drafter == "mtp" and self.mtp is not None, "the MTP fill: the MTP drafter"
            row = lambda name: ttnn.from_torch(self.source.host(MTP_L, name).float().reshape(1, 1, 1, HIDDEN),  # noqa
                                               dtype=BF16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                               memory_config=DRAM, mesh_mapper=_repl(self.mesh))  # fmt: skip
            a = self.mtp.attn
            self._pfw = dict(table=self.be.embed.table, enorm=row("enorm.weight"), hnorm=row("hnorm.weight"),
                             in_norm=row("input_layernorm.weight"), eh=self.eh.ds, qa_kva=a.ds_qa_kva,
                             kva_norm=a.kva_norm, q_lora=int(a.q_lora), kv_lora=int(a.kv_lora))  # fmt: skip
        return self._pfw

    def _pf_taps(self) -> dict:
        """{bucket: tap} of the TP prefill's MTP taps (allocated by its compile forwards; none: {})"""
        tpp = getattr(self.be, "tpp", None)
        if self.drafter != "mtp" or tpp is None or not getattr(tpp, "mtp_tap_on", False):
            return {}
        return dict(getattr(tpp, "_mtp_taps", {}))

    def prefill_mtp(self, tap, tokens_next, pos0: int, n: int, bucket: int | None = None):
        """the MTP-cache prefill (M0): MTP positions pos0 .. pos0 + n - 1 from rows 0 .. n - 1 of `tap` (the TP prefill's
        final-norm hidden of the chunk at pos0: TPPrefill.mtp_tap) and tokens_next[i] = the prompt token at position
        pos0 + i + 1 (mtp_prefill_span) -- vLLM MTP semantics, the catch-up's: the drafts attend the real prompt, not
        a zero cache. Launches of mw_cu rows after the chunk's replay (each launch's rows inside one tap tile row; the
        slot's MTP page-table row = pt_m, set with the page table): the bucket's captured launch (capture(): "pf<c>";
        GLM53_SPEC_PF_TRACE=0 or no trace: eager); nothing allocated, the programs compiled in prepare(). The page table
        must cover the positions (the generator's rows)."""
        assert self.drafter == "mtp" and self.mtp is not None, "MTP-cache prefill: the MTP drafter"
        assert not self.groups, "GLM53_SPEC_MULTI: the in-trace MTP fill (TPPrefill._mtp_write), not these launches"
        assert TILE % self.mw_cu == 0, ("MTP-cache prefill: a launch width dividing the tap's 32-row tiles", self.mw_cu)
        toks = [int(v) for v in torch.as_tensor(tokens_next).reshape(-1)[: int(n)].tolist()]
        assert len(toks) == int(n), (len(toks), n)
        trace = None
        if bucket is not None and os.environ.get("GLM53_SPEC_PF_TRACE", "1") == "1":
            trace = self.traces.get(f"pf{int(bucket)}")  # (captured at warm-up with the step; never here)
        r = 0
        while r < len(toks):  # (launch r covers tap rows r .. r + mw_cu - 1: one tile row, r a multiple of mw_cu)
            m = min(self.mw_cu, len(toks) - r)
            self._mtp_prefill_launch(tap, r, toks[r : r + m], int(pos0) + r, trace=trace)
            self.counters["prefill_launches"] += 1
            r += m
        self.counters["prefill_rows"] += len(toks)

    def _gather(self):
        hd = self.heads
        couts = [h.topk_out for h in hd["mtp_ch"]]
        args = [self.tin.buffer_address(), hd["mtp_cu"].topk_out.buffer_address(), self.rec.buffer_address()]
        args += [c.buffer_address() for c in couts] + [0] * (3 - len(couts))
        self.ctl("gather", args, [self.tin, hd["mtp_cu"].topk_out, self.rec] + couts)

    def _step(self):
        self._verify()
        self._ctl_accept(boot=False)
        if self.drafter == "mtp":  # ("none": the next step's drafts are written into token rows 1..K from outside)
            self._draft()
        elif self.drafter == "dflash2":  # PREP -> the context K / V of this step's rows -> the block -> token rows 1..K
            self.df2.step()

    def _boot(self):
        self._ctl_accept(boot=True)
        if self.drafter == "mtp":
            self._draft()
        elif self.drafter == "dflash2":  # (the prompt's context: seeded before enter(), DFlash2Kind.seed_ctx)
            self.df2.boot()

    # ------------------------------------------------------------------ lifecycle
    def prepare(self):
        """compile every spec program (one eager boot + step, every row inactive: positions -1, so no latent row is
        written) BEFORE the backend's first trace capture; the caller (StackBackend.prepare_decode) restores the
        slots' KDA state and its token / position pages afterwards"""
        if self.heads is None:
            self._make_heads()
        self._seed_persist()
        l1_debug("prepare: heads + seeds", self.mesh)
        # far-negative positions: c = pos - 1 and every derived position (c + n + 1 + j, c + j) stay < 0 -> no latent
        # row of the main or the MTP cache is written (an inactive row only reads row 0 of its first page)
        z = torch.full((1, 32), -1, dtype=torch.int32)
        far = torch.full((1, 32), -100, dtype=torch.int32)
        keep = read_rows(self.mesh, self.be.pos)
        write_rows(self.mesh, self.be.pos, far, ttnn.int32)
        self._boot()
        l1_debug("prepare: after boot", self.mesh)
        for t_ in [self.pos_v, self.mpos_cu] + self.mpos_ch:
            write_rows(self.mesh, t_, far, ttnn.int32)
        self._step()
        l1_debug("prepare: after step", self.mesh)
        if self.drafter == "mtp" and os.environ.get("GLM53_SPEC_MTP_PREFILL", "1") == "launch":
            # M0's MTP-cache prefill launches (MkEh EH_ROW0), every row inactive: compiled here, before any capture (a
            # first compile at the first request would allocate after the captures); the default in-trace fill
            # (TPPrefill._mtp_fill) compiles with the prefill buckets
            self._mtp_prefill_launch(self.eh.buffers(self.mw_cu)[0], 0, [], 0)
            l1_debug("prepare: after the MTP prefill launch", self.mesh)
        ttnn.synchronize_device(self.mesh)
        write_rows(self.mesh, self.be.pos, keep.reshape(1, 32), ttnn.int32)
        write_rows(self.mesh, self.sel, torch.zeros(1, 32, dtype=torch.int32))
        write_rows(self.mesh, self.rec, torch.zeros(max(1, self.groups), 64, dtype=torch.int32))
        for t_ in [self.pos_v, self.mpos_cu] + self.mpos_ch:
            write_rows(self.mesh, t_, z, ttnn.int32)

    def capture(self):
        """the spec-step and bootstrap traces (a deferred first capture of prepared programs)"""
        if self.traces:
            return self.traces
        from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import MOCK, l1_report

        if os.environ.get("GLM53_L1_REPORT") == "1":  # Stage 9 / 12 cf3f: the L1 the verify / draft programs meet
            l1_report(self.mesh, "spec capture (before the step / boot compile)", root=self.be)
        # (+ the MTP-cache prefill's launch per TP prefill tap: "pf<c>", replayed between the prefill's chunks)
        pf = [(f"pf{c}", (lambda tap=tap: self._mtp_prefill_body(tap))) for c, tap in sorted(self._pf_taps().items())]
        for name, fn in [("step", self._step), ("boot", self._boot)] + pf:
            if MOCK:
                if os.environ.get("GLM53_MOCK_CAPTURE", "0") == "1":
                    # Stage 9 cf3f (coordinator 18:11:19Z acceptance): the step runs once here, so its programs'
                    # static CBs are validated against every L1 buffer allocated before the first spec request
                    fn()
                self.traces[name] = ("mock", fn)
                continue
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            try:
                fn()
            except BaseException:
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                ttnn.release_trace(self.mesh, tid)
                self.release()  # all or nothing: a later call must not find a half-captured pair
                raise
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.be.traces.append(tid)
            self.traces[name] = tid
            self.counters["captures"] += 1
        return self.traces

    def release(self):
        for name, tid in list(self.traces.items()):
            if not isinstance(tid, tuple) and tid in self.be.traces:
                ttnn.release_trace(self.mesh, tid)
                self.be.traces.remove(tid)
        self.traces = {}

    def _replay(self, name: str, blocking: bool = False):
        tr = self.capture()[name]
        if isinstance(tr, tuple):
            tr[1]()
        else:
            ttnn.execute_trace(self.mesh, tr, cq_id=0, blocking=blocking)

    def set_page_table(self, row: torch.Tensor):
        """the user's page-table row -> rows 0..K of pt_v (the verify / MTP launches index the page table by launch row)"""
        row = row.reshape(-1).to(torch.int32)
        if self._pt_rows is not None and torch.equal(self._pt_rows, row):
            return
        host = row.reshape(1, -1).repeat(32, 1)
        if self.shard is None:
            write_rows(self.mesh, self.pt_v, host, ttnn.int32)
        else:  # row = slot 0's PHYSICAL row: chip r's local row in rows 0..31 of its pt_v; the MTP layer's global rows
            local = self.be.shard_local_rows(row.reshape(1, -1), 1)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(
                torch.cat([lt.reshape(1, -1).repeat(32, 1) for lt in local]).to(torch.int32).contiguous(),
                dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0)),
                self.pt_v)  # fmt: skip
            if not self.mtp_private:  # (the private MTP cache's row is constant)
                write_rows(self.mesh, self.pt_m, host, ttnn.int32)
        self._pt_rows = row.clone()

    def set_page_tables(self, rows) -> None:
        """GLM53_SPEC_MULTI: the users' page-table rows (group order = self.slots) -> pt_v rows u (K + 1) + j = user u's
        (the shard: each chip's LOCAL rows of the PHYSICAL ones); GLM53_SPEC_MTP_CACHE=pool: the MTP tables likewise
        (catch-up rows by group, chained row u = user u); the private MTP tables are constant (_mtp_private_tables)"""
        assert self.groups and len(rows) == self.groups, (self.groups, len(rows))
        key = torch.stack([torch.as_tensor(r).reshape(-1).to(torch.int32) for r in rows])
        old = self._pt_rows
        if old is not None and tuple(old.shape) == tuple(key.shape) and torch.equal(old, key):
            return
        g, G = self.K + 1, self.groups
        by_row = [min(r // g, G - 1) for r in range(32)]  # launch row -> user (rows past the groups: inactive)
        host = key[by_row]
        if self.shard is None:
            write_rows(self.mesh, self.pt_v, host, ttnn.int32)
        else:
            local = self.be.shard_local_rows(key, G)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(
                torch.cat([torch.as_tensor(lt).reshape(G, -1)[by_row] for lt in local]).to(torch.int32).contiguous(),
                dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0)),
                self.pt_v)  # fmt: skip
        if not self.mtp_private and self.drafter == "mtp":
            if self.pt_m is not self.pt_v:
                write_rows(self.mesh, self.pt_m, host, ttnn.int32)
            write_rows(self.mesh, self.pt_mc, key[[min(r, G - 1) for r in range(32)]], ttnn.int32)
        self._pt_rows = key.clone()

    def set_table(self, value: torch.Tensor) -> None:
        """StackBackend's page-table write (every slot's row): the speculating rows -- slot 0's (set_page_table), or
        with GLM53_SPEC_MULTI the group bases' (set_page_tables)"""
        if not self.groups:
            self.set_page_table(value[0])
            return
        if int(value.shape[0]) > max(self.slots):
            self.set_page_tables([value[s] for s in self.slots])

    def enter(self, page_row):
        """start speculating for the user in slot 0 after a plain step / prefill (its committed state in row 0, the next
        input token in token row 0, be.pos[0] = its position): the bootstrap replay drafts d'1..d'K on device.
        GLM53_SPEC_MULTI: page_row = the users' rows (group order); every user on its group base the same way"""
        if self.groups:
            self.set_page_tables(page_row)
        else:
            self.set_page_table(page_row)
        self._replay("boot")
        self.counters["enters"] += 1
        self.counters["boots"] += 1

    def step(self, blocking: bool = False):
        self._replay("step", blocking=blocking)
        self.counters["steps"] += 1

    def read(self):
        """the step record: n (accepted drafts), c (the step's base position), a (the K+1 verify argmax = the committed
        tokens a[0..n]), drafts (the next step's), steps. GLM53_SPEC_MULTI: the G users' records (group order)"""
        return self._parse_pages(read_rows(self.mesh, self.rec))

    def _parse_pages(self, r: torch.Tensor):
        if not self.groups:
            return self._parse(r)
        r = r.reshape(-1)[: self.groups * 64].reshape(self.groups, 64)
        return [self._parse(r[u]) for u in range(self.groups)]

    def read_async(self):
        """the record's device -> host copy enqueued behind the replays submitted so far (non-blocking) + an event
        (Glm53Generator.spec_step_submit: the served verify's deferred readback); finish_read parses it once the event
        completed. The mock cluster reads blocking."""
        from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import MOCK

        if MOCK:
            return self.read(), []
        return ttnn.from_device(self.rec, blocking=False), [ttnn.record_event(self.mesh)]

    def finish_read(self, pending) -> dict:
        if isinstance(pending, dict):  # (a blocking read's record)
            return pending
        d = ttnn.get_device_tensors(pending)[0] if int(self.mesh.get_num_devices()) > 1 else pending
        return self._parse_pages(ttnn.to_torch(d).to(torch.int64))

    def _parse(self, r: torch.Tensor) -> dict:
        r = r.reshape(-1) & 0xFFFFFFFF
        K = self.K
        n = int(r[0])
        return dict(n=n, c=int(r[1]) - (1 << 32) * (int(r[1]) >= (1 << 31)), a=[int(v) for v in r[2 : 3 + K]],
                    committed=[int(v) for v in r[2 : 3 + n]], drafts=[int(v) for v in r[16 : 16 + K]],
                    steps=int(r[31]))  # fmt: skip

    def abort(self) -> None:
        """the speculating request ended without a leave (released): the selector back to row 0 (its candidate rows are
        dead state; the next request's prefill rewrites the slot)"""
        write_rows(self.mesh, self.sel, torch.zeros(1, 32, dtype=torch.int32))

    def leave(self) -> dict:
        """stop speculating: the committed KDA state / conv history of row sel -> row 0 (SlotOps, the stack's existing
        between-step mechanism), the plain position row 0 = the next input's position; token row 0 already holds the
        bonus. Plain decode then continues from exactly the committed state."""
        rec = self.read()
        be, kv = self.be, self.kv
        if self.groups:  # GLM53_SPEC_MULTI: per user, candidate row u (K + 1) + n_u -> its base row, its plain position
            pos = read_rows(self.mesh, be.pos).reshape(-1)[:32].clone()
            for s, r in zip(self.slots, rec):
                if r["n"] != 0:
                    be.slot_ops.run(be._state_ops(kv, s, s + r["n"], dsa=False))
                    self.counters["slot_copies"] += 1
                pos[s] = r["c"] + r["n"] + 1
            write_rows(self.mesh, self.sel, torch.zeros(1, 32, dtype=torch.int32))
            be.write(kv.extra["io"].cur_pos, pos[: kv.max_batch])
            self.counters["leaves"] += 1
            return rec
        n, c = rec["n"], rec["c"]
        if n != 0:
            # KDA S / conv rows only: the user's DSA pooled keys / ring already live in slot 0 (the one-slot verify
            # writes slot 0 and ACCEPT moves its ring rows); copying slot n's DSA state would overwrite them
            be.slot_ops.run(be._state_ops(kv, 0, n, dsa=False))
            self.counters["slot_copies"] += 1
        write_rows(self.mesh, self.sel, torch.zeros(1, 32, dtype=torch.int32))
        pos = read_rows(self.mesh, be.pos).reshape(-1)[:32].clone()
        pos[0] = c + n + 1
        be.write(kv.extra["io"].cur_pos, pos[: kv.max_batch])
        self.counters["leaves"] += 1
        return rec
