# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14b (PRODUCTION_PATH.md M5, the sparse_dsa kind on the TP mesh): a sparse_dsa layer as TWO programs per chip,
Stage 7b's TPMla (the MLA attention half, tt/tp_mla.py) + TPMoE here (the MoE FFN half: 7a's MkMoE on every chip + the
in-program fabric all-reduce of the routed experts' partial, as Stage 14's TPSparseKDALayer does for sparse_kda):

  router + router tail: replicated (every chip routes every row identically);
  routed experts: EP — chip r runs experts [E_loc r, E_loc (r + 1)) (ReaderExperts' e_lo, per chip at compile time;
      tt/ep_experts.py's EP-sharded weights, or the replicated ones) -> its routed PARTIAL -> the 2 fabric lanes
      (kernels/mkx/xa.cpp MKX_FAB_*), which all-reduce over the mesh and signal the owners;
  shared expert: replicated (y2 complete on every chip);
  P10 mhc_post on the owners: y = the reduced rows from the lanes (mhc_post/reader_p.cpp MK_POST_Y_FABRIC), y2 local;
      it waits for the shared down readers (nr) + the 2 lanes on SEM_Y.

s1 = TPMla(...)["s1"] and out = TPMoE(s1) are identical on every chip. Decode rows t <= 4 (the stacked residual, the
packed mhc_post, the lanes' packed rows)."""

from __future__ import annotations

import os

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import fabric_ar as FAR
from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MK
from models.autoports.zai_org_glm_5_3_flash.tt import mk_moe as MKM
from models.autoports.zai_org_glm_5_3_flash.tt import tp_mla as TPM
from models.autoports.zai_org_glm_5_3_flash.tt.tp8_layer import _Capture, _rt_dict, _src_is
from models.autoports.zai_org_glm_5_3_flash.tt.tp8_sparse import XW_E_LO_CT, _LocalExperts

DRAM = ttnn.DRAM_MEMORY_CONFIG
# TPMla's head-shard reader phases (8 heads per chip): Stage 7b's validated TP mesh config (doc/ttnn/07b-mla-megakernel
# tp_mla_mesh_vs_full, rev 24 at 2026-10-20 05:43Z: the rev-24 defaults + dev_mk_tp_mesh.py's shard K blocks / rings)
MLA_KB_SHARD = (8, 16, 8, 16, 16)
MLA_RINGS_SHARD = (140, 105, 128, 280)


def mhc_host_args(source, cfg, layer: int, sub: str) -> dict:
    """MhcPre arguments (host tensors) of `layer`'s attention ("attn") or FFN ("ffn") hyper-connection (tt/tp8.py
    TPShards.mhc_args for a layer without a KDA shard)."""
    n = cfg.hc_mult
    scale = source.host(layer, f"hc_{sub}_scale").float().reshape(-1)
    norm = "input_layernorm.weight" if sub == "attn" else "post_attention_layernorm.weight"
    return dict(
        fn_t=source.host(layer, f"hc_{sub}_fn").reshape(n * cfg.hidden_size, (2 + n) * n),
        scale=torch.cat([scale[0].repeat(n), scale[1].repeat(n), scale[2].repeat(n * n)]),
        bias=source.host(layer, f"hc_{sub}_base").float().reshape(-1),
        norm_w=source.host(layer, norm).float().reshape(-1),
        hc_eps=cfg.hc_eps, x_eps=cfg.rms_norm_eps, h_eps=cfg.rms_norm_eps, iters=cfg.hc_sinkhorn_iters,
        hidden=cfg.hidden_size, n=n,
    )  # fmt: skip


def build_mla_attn(mesh, cfg, source, layer: int, ocfg):
    """`layer`'s MLA attention replicated on `mesh` (tests/s7b/mla_common.MeshMla: FusedMLA as FusedDecoderLayer builds
    it -> OptimizedMLA), the input of MeshHeadShard / TPMla and the owner of the latent-cache layout."""
    from models.autoports.zai_org_glm_5_3_flash.tt.functional_decoder import BLOCK_SIZE
    from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedMLA, _row
    from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import OptimizedMLA, _upgrade

    dev = lambda n: source.device(layer, n, mesh)  # noqa: E731
    attn = FusedMLA(
        mesh,
        w_qa_kva=ttnn.concat([dev("self_attn.q_a_proj.weight"), dev("self_attn.kv_a_proj_with_mqa.weight")], dim=-1,
                             memory_config=DRAM),
        qa_norm=_row(dev("self_attn.q_a_layernorm.weight")), kva_norm=_row(dev("self_attn.kv_a_layernorm.weight")),
        w_qb=dev("self_attn.q_b_proj.weight"), w_uk=dev("self_attn.kv_b_proj.W_uk"),
        w_uv=dev("self_attn.kv_b_proj.W_uv"), w_o=dev("self_attn.o_proj.weight"),
        num_heads=cfg.num_attention_heads, q_lora_rank=cfg.q_lora_rank, kv_lora_rank=cfg.kv_lora_rank,
        qk_head_dim=cfg.qk_head_dim, v_head_dim=cfg.v_head_dim, eps=cfg.rms_norm_eps, block_size=BLOCK_SIZE,
    )  # fmt: skip
    attn.fcfg = ocfg
    _upgrade(attn, OptimizedMLA, ocfg)
    return attn


def _real_head_weights(shard, source, layer: int) -> int:
    """A DRAM-image replay hands MeshHeadShard all-zero W_uk / W_uv host copies: tp_image's _host0 of an unread weight
    returns zeros, and MLA keeps a ttnn.reshape of the cache-loaded W_uk, so the replay cannot map it back to the file.
    The TP prefill, attached after the restore, uploads these copies -> re-read them from the weight FILE
    (FdTileCacheSource.host: a real host read in a replay too; coordinator 2026-10-22 09:42Z defect). The tensor path's
    copies are already the weights (no zeros): untouched. Returns the number of per-rank tensors replaced."""
    hw = getattr(shard, "host_weights", None)
    if not hw or os.environ.get("GLM53_IMAGE_HOST_WEIGHTS_FILE", "1") != "1":
        return 0
    n = 0
    for k, name in (("w_uk", "self_attn.kv_b_proj.W_uk"), ("w_uv", "self_attn.kv_b_proj.W_uv")):
        if all(bool(w.abs().max() > 0) for w in hw[k]):
            continue
        hh = sum(int(w.shape[0]) for w in hw[k])
        full = source.host(layer, name).float().reshape(hh, *[int(v) for v in hw[k][0].shape[1:]])
        hw[k] = [full[s] for s in shard.heads]
        n += len(hw[k])
    return n


class TPMoE:
    """The MoE FFN half of a sparse_dsa layer, TP / EP over every chip of `mesh` (decode rows t <= 4): MkMoE with this
    chip's experts + the in-program fabric all-reduce of the routed partial. Same call / outputs as MkMoE (`y` is NOT
    written on a fabric mesh: the reduced rows only live in the lanes' result buffer, read by mhc_post)."""

    def __init__(self, mesh, mhc_args: dict, moe, *, ts=(1, 2, 3, 4), fabrics: dict | None = None, **mk_kwargs):
        """moe: the layer's OptimizedMoE on `mesh` (tt/ep_experts.build_moe_ep or tt/tp8_sparse.build_moe); ts: the row
        counts this instance serves (one FabricPhase each: the packed row count is a compile-time value of the
        readers); fabrics: {t: FabricPhase} of an earlier TPMoE with the same lane cores (shared lane buffers and
        counters: every sparse_dsa layer's MoE program places its cores identically)."""
        self.mesh = mesh
        self.R, self.C = int(mesh.shape[0]), int(mesh.shape[1])
        ndev = self.R * self.C
        self.e_all = int(moe.num_experts)
        assert self.e_all % ndev == 0, (self.e_all, ndev)
        self.e_loc = self.e_all // ndev
        self.mlp = moe
        self.mk = MKM.MkMoE(mesh, mhc_args, _LocalExperts(moe, 0, self.e_loc), e_lo=0, num_experts=self.e_all,
                            **mk_kwargs)  # fmt: skip
        self.nofab = ndev == 1
        self.owner_noc = self.mk.owner_noc
        self.n_readers = len(self.mk.readers.assign)
        if not self.nofab:
            # the per-chip expert address table: chip r = experts [E_loc r, E_loc (r + 1)) (EP-sharded weights: the
            # slot addresses, identical rows; replicated weights: each chip's own experts)
            ex = self.mk.experts
            rows = []
            for r in range(ndev):
                lo = self.e_loc * r
                rows.append([g.buffer_address() for g in moe.gate_up[lo : lo + self.e_loc]]
                            + [d.buffer_address() for d in moe.down[lo : lo + self.e_loc]])  # fmt: skip
            tab = torch.tensor(rows, dtype=torch.int64).to(torch.int32).reshape(ndev, 1, 1, -1)
            ttnn.deallocate(ex.table)
            ex.table = ttnn.from_torch(tab, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                       memory_config=DRAM, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))  # fmt: skip
        # lanes: the first 2 cores no role of the program takes (readers, owners, root; the router tail runs on owners)
        grid = mesh.compute_with_storage_grid_size()
        used = {(int(c.x), int(c.y)) for c, _, _ in self.mk.readers.assign}
        used |= {(int(c.x), int(c.y)) for c in self.mk.place["k"]}
        used.add((int(self.mk.place["root"].x), int(self.mk.place["root"].y)))
        free = [(x, y) for y in range(int(grid.y)) for x in range(int(grid.x)) if (x, y) not in used]
        self.lanes = free[:2]
        assert len(self.lanes) == 2, "two free cores for the fabric lanes"
        if self.nofab:
            self.fabrics = {}
        elif fabrics is not None:
            assert all(list(f.lanes) == list(self.lanes) for f in fabrics.values()), "shared fabrics: other lane cores"
            assert set(ts) <= set(fabrics), (ts, sorted(fabrics))
            self.fabrics = fabrics
        else:  # built and probed HERE (a launch + read-back), never on a call that may sit inside a trace capture
            self.fabrics = {t: self._new_fabric(t) for t in ts}
        self.calls = 0

    def _new_fabric(self, t: int):
        f = FAR.FabricPhase(self.mesh, t, lanes=self.lanes, mode="cb")
        f.probe()
        return f

    def _hooked(self, io, desc, t: int):
        """The captured MkMoE program + the fabric phase -> MeshProgramDescriptor (one program per chip: the lanes'
        arguments and the weight streamer's e_lo differ)."""
        # a native-t width bucket (t > 4 without its own phase): the 32-row phase (whole tiles: the t > 4 consumer)
        fab = self.fabrics[t] if t in self.fabrics or t <= 4 or 32 not in self.fabrics else self.fabrics[32]
        nr = self.n_readers
        base, wkern, wdesc, seen = [], None, None, set()
        for i, kd in enumerate(desc.kernels):  # (ProgramDescriptor.kernels returns copies)
            if _src_is(kd, "mk/reader_moe_a.cpp"):  # the routed partial -> the lanes (the lanes signal the owners)
                defs = [d for d in kd.defines if d[0] != "MKX_SIGNAL_SEM"]
                n = max(len(v) for v in _rt_dict(kd).values())
                defs += [("MKX_FAB_RT", str(n)), ("MKX_FAB_BROWS", str(fab.brows))]
                base.append(TPM._with_rt(kd, fab.producer_args(), defs)[0])
                seen.add("a")
            elif _src_is(kd, "mk/reader_moe_w.cpp"):
                wkern, wdesc = len(base), i
                base.append(kd)
                seen.add("w")
            elif _src_is(kd, "mk/owner_moe_dm0.cpp"):  # P10: y from the lanes; shared down readers + 2 lanes
                n = max(len(v) for v in _rt_dict(kd).values())
                defs = [d for d in kd.defines if d[0] != "MK_POST_WAIT_COUNT"]
                defs += [("MK_POST_WAIT_COUNT", str(nr + 2)), ("MK_POST_Y_FABRIC", "1"), ("MK_POST_FAB_RT", str(n))]
                base.append(TPM._with_rt(kd, fab.consumer_args(), defs)[0])
                seen.add("o")
            else:
                base.append(kd)
        assert seen == {"a", "w", "o"}, (seen, [str(k.kernel_source) for k in desc.kernels])
        xct = dict(base[wkern].defines).get("MK_X_CT")
        assert xct is not None, "reader_moe_w.cpp carries MK_X_CT"
        e_lo_idx = int(xct) + XW_E_LO_CT
        assert int(base[wkern].compile_time_args[e_lo_idx]) == 0, "the captured program is chip 0's (e_lo 0)"
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.R):
            for c in range(self.C):
                rank = r * self.C + c
                kernels = list(base)
                if rank:  # a fresh copy of the weight streamer with this chip's e_lo
                    kw = desc.kernels[wdesc]
                    ct = list(kw.compile_time_args)
                    ct[e_lo_idx] = self.e_loc * rank
                    kw.compile_time_args = ct
                    kernels[wkern] = kw
                lk, lcbs = fab.lane_parts(r, c, 1, nr, self.owner_noc, cons_sem=MKM.SEM_Y)
                prog = ttnn.ProgramDescriptor(kernels=kernels + lk, semaphores=list(desc.semaphores),
                                              cbs=list(desc.cbs) + lcbs)  # fmt: skip
                fab.add_connections(prog, r, c, send_kernel_index=len(kernels))
                m = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(m, m)] = prog
        return list(io) + fab.io(), mpd

    def __call__(self, s1, *, t: int, out=None, live=None):
        """MkMoE's call on every chip (+ the all-reduce of the routed partial) -> MkMoE's outputs dict. live: the
        stack's position page (idle rows route to no expert, MkMoE / ReaderExperts MKX_LIVE)."""
        self.calls += 1
        if self.nofab:
            return self.mk(s1, t=t, out=out, live=live)
        with _Capture() as cap:
            res = self.mk(s1, t=t, out=out, live=live)
        ((io, desc),) = cap.calls
        io2, mpd = self._hooked(io, desc, t)
        ttnn.generic_op(io2, mpd)
        return res


class TPDsaLayer:
    """One sparse_dsa layer TP / EP over every chip of `mesh` (decode rows t <= 4): attn (the replicated OptimizedMLA:
    latent-cache layout), mla (TPMla), moe (TPMoE). share: an earlier TPDsaLayer of the same mesh / ts (its fabric
    phases: every sparse_dsa layer's programs place their cores identically). dsa (Stage 14, the stack's
    GLM53_STACK_DSA): TPMla(dsa=DsaIndexer) -- the replicated indexer, top-512 sparse attention past 2,051 rows; the
    call then takes dsa_state (the stack's per-layer store, mk_mla.DsaIndexer.stack_state)."""

    def __init__(self, mesh, cfg, layer: int, source, ocfg, *, moe, ts=(1, 2, 3, 4), share: "TPDsaLayer" | None = None,
                 dsa: bool = False, defer: bool = False, shard: bool = False):  # fmt: skip
        """defer (GLM53_STACK_DSA_DEFER, with dsa): build the DENSE attention half now (the allocation sequence a DRAM
        image recorded without the DSA stack: tt/tp_image.py replays only an identical upload sequence) and swap in the
        DSA one with attach_dsa() after the restore (StackBackend.allocate_kv_cache: before any KV / program state).
        shard (GLM53_DSA_SHARD, with dsa): the DSA half is tp_dsa_shard.TPMlaShard (the sequence-sharded pool's split-K
        decode: the caller's latent cache / page table / pooled-key store are per-chip LOCAL)"""
        assert dsa or not shard, "shard needs dsa"
        self.mesh, self.layer, self.shard = mesh, layer, bool(shard)
        self.dsa_pending = bool(dsa and defer)
        self._attach = (cfg, source, ocfg, tuple(ts)) if self.dsa_pending else None
        self.attn, self.mla = self._build_mla(cfg, source, ocfg, ts, share, dsa and not defer)
        self.moe = TPMoE(mesh, mhc_host_args(source, cfg, layer, "ffn"), moe, ts=ts,
                         fabrics=share.moe.fabrics if share else None)  # fmt: skip

    # ---- the slice sidecar (GLM53_DSA_SLICE_CACHE=<local dir>; unset: off): MeshHeadShard's fp32 host copies of q_b /
    # o_proj / W_uk / W_uv (what the per-rank uploads and host_weights are cut from), saved by the first attach_dsa of a
    # layer and loaded (mmap) by later restarts in place of 4 device read-backs + untilizes per layer. Keyed by the
    # weight manifest's sha256 of the four tensors + the mesh size: another weight cache or TP never matches; shapes
    # are checked again at construction (MeshHeadShard).
    SLICE_WEIGHTS = ("self_attn.q_b_proj.weight", "self_attn.o_proj.weight", "self_attn.kv_b_proj.W_uk",
                     "self_attn.kv_b_proj.W_uv")  # fmt: skip

    @classmethod
    def slice_sidecar(cls, source, layer: int, ndev: int) -> str | None:
        root = os.environ.get("GLM53_DSA_SLICE_CACHE")
        entry = getattr(source, "_entry", None)
        if not root or entry is None:
            return None
        import hashlib

        h = hashlib.sha256()
        try:
            for name in cls.SLICE_WEIGHTS:
                e = entry(layer, name)
                h.update(f"{name}:{e.get('sha256') or e.get('path')}:{e.get('bytes')};".encode())
        except KeyError:
            return None
        return os.path.join(root, f"dsa_slices-{h.hexdigest()[:16]}-L{layer:02d}-tp{ndev}.pt")

    @classmethod
    def deferred_paths(cls, source, layer: int) -> list:
        """the weight-cache files attach_dsa reads for `layer` (a tile-cache source; else []) + its slice sidecar when
        one exists (page-cache prefetch targets)"""
        entry, root = getattr(source, "_entry", None), getattr(source, "cache_dir", None)
        if entry is None or not root:
            return []
        out = []
        for name in cls.DEFERRED_WEIGHTS:
            try:
                out.append(os.path.join(root, entry(layer, name)["path"]))
            except KeyError:
                pass
        for ndev in (8, 4, 2):
            sc = cls.slice_sidecar(source, layer, ndev)
            if sc and os.path.isfile(sc):
                out.append(sc)
                break
        return out

    @staticmethod
    def _load_slices(path: str | None) -> dict | None:
        if not path or not os.path.isfile(path):
            return None
        try:
            d = torch.load(path, mmap=True)
        except Exception as e:  # noqa: BLE001  (a torn / foreign file: rebuilt from the device read-backs, overwritten)
            print(f"[tp_dsa] slice sidecar {path} unreadable ({type(e).__name__}): rebuilding", flush=True)
            return None
        if not isinstance(d, dict) or any(k not in d for k in TPM.MeshHeadShard.FULL_KEYS):
            return None
        # a device launch serves only device-written slices (coordinator 2026-10-23T01:09:36Z: an image mock's zero
        # read-backs were served as q_b / o_proj / W_uk / W_uv in gx2c2..gx2c5): abort on all-zero / a non-device writer
        zero = [k for k in TPM.MeshHeadShard.FULL_KEYS if not bool(d[k].any())]
        writer = (d.get("_meta") or {}).get("writer", "device")  # sidecars before 2026-10-23 carry no _meta
        if zero or writer != "device":
            raise RuntimeError(
                f"DSA slice sidecar {path}: all-zero {zero} / writer {writer!r} -- not device-written "
                "weights; move it aside (the next device attach rebuilds it)"
            )
        return d

    @staticmethod
    def sidecar_writable() -> bool:
        """an image mock's device read-backs carry no data: it never writes the sidecar (it only loads one)"""
        return os.environ.get("GLM53_MOCK") != "1"

    @staticmethod
    def _save_slices(path: str, full: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        meta = {"writer": "device", "run": os.environ.get("GLM53_RUN"), "pid": os.getpid()}
        torch.save({**{k: full[k].contiguous() for k in TPM.MeshHeadShard.FULL_KEYS}, "_meta": meta}, tmp)
        os.replace(tmp, path)

    def _build_mla(self, cfg, source, ocfg, ts, share, dsa: bool, *, host_full: dict | None = None,
                   keep_full: bool = False):  # fmt: skip
        mesh, layer = self.mesh, self.layer
        attn = build_mla_attn(mesh, cfg, source, layer, ocfg)
        ix, rings = {}, MLA_RINGS_SHARD
        if dsa:  # the indexer's reader phases (I1 w_x, I3 wq_b) under the stack's L1 floor (mock 2026-10-21 04:21Z,
            # B=1 [L2, L3]: K blocks (32, 8) + a 128-KB bf16 ring end the readers' static CBs at 0x1604c0, 58,752 B
            # above the KDA pool's 0x151f40; (16, 4) + 64 KB: 0x1404c0, one shared 2-slot ring of 16-tile blocks)
            kb = tuple(int(v) for v in os.environ.get("GLM53_STACK_DSA_KB", "16,4").split(","))
            rings = tuple(int(v) for v in os.environ.get("GLM53_STACK_DSA_RINGS", "140,105,64,280").split(","))
            ix = dict(dsa=MK.DsaIndexer(mesh, source, layer), dsa_kb=kb)
        cls = TPM.TPMla
        if dsa and self.shard:
            from models.autoports.zai_org_glm_5_3_flash.tt import tp_dsa_shard as TPDS

            cls = TPDS.TPMlaShard
        if cls is not TPM.TPMla:  # (GLM53_DSA_SHARD_SHARE_SYNC: the first layer's stream sync tensors)
            ix = dict(ix, stream_share=share.mla if share is not None and isinstance(share.mla, cls) else None)
        mla = cls(mesh, mhc_host_args(source, cfg, layer, "attn"), attn, kb=MLA_KB_SHARD,
                  rings=rings, fabrics=share.mla.fabrics if share else None,
                  ts=ts, host_full=host_full, keep_full=keep_full, **ix)  # fmt: skip
        _real_head_weights(mla.shard, source, layer)
        # the replicated full-head q_b / o_proj / W_uk / W_uv: MeshHeadShard holds the per-chip slices (MkMla re-lays
        # W_uk / W_uv out from the host slices) -> only q_a | kv_a and the norms stay replicated
        for name in ("ds_qb", "ds_o"):
            ttnn.deallocate(getattr(attn, name).weight)
        for name in ("w_uk", "w_uv"):
            ttnn.deallocate(getattr(attn, name))
        return attn, mla

    # what attach_dsa reads from the weight cache (build_mla_attn + DsaIndexer + _real_head_weights): the files a
    # restore prefetches into the page cache while the image replay runs (StackBackend, GLM53_DSA_PREFETCH)
    DEFERRED_WEIGHTS = (
        "self_attn.q_a_proj.weight",
        "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.q_a_layernorm.weight",
        "self_attn.kv_a_layernorm.weight",
        "self_attn.q_b_proj.weight",
        "self_attn.kv_b_proj.W_uk",
        "self_attn.kv_b_proj.W_uv",
        "self_attn.o_proj.weight",
        "self_attn.indexer.wk.weight",
        "self_attn.indexer.index_kpool_compress_gate",
        "self_attn.indexer.weights_proj.weight",
        "self_attn.indexer.wq_b.weight",
        "self_attn.indexer.k_norm.weight",
        "self_attn.indexer.k_norm.bias",
        "self_attn.indexer.index_kpool_compress_ape",
    )

    def attach_dsa(self, share: "TPDsaLayer" | None = None) -> None:
        """the deferred DSA attention half (defer=True): rebuilt from the weight cache (real reads: after the image
        replay) as TPMla(dsa=DsaIndexer); the dense half's tensors are released. share: the first attached layer (its
        fabric phases; the dense halves' phases are not reused: the DSA program reserves more cores)."""
        assert self.dsa_pending, "attach_dsa: nothing deferred"
        cfg, source, ocfg, ts = self._attach
        old = (self.attn, self.mla)
        sc = self.slice_sidecar(source, self.layer, int(self.mesh.get_num_devices()))
        hf = self._load_slices(sc)
        self.attn, self.mla = self._build_mla(cfg, source, ocfg, ts, share, True, host_full=hf,
                                              keep_full=sc is not None and hf is None)  # fmt: skip
        self.slices_from = "sidecar" if hf is not None else "device"
        if sc is not None and hf is None and getattr(self.mla.shard, "host_full", None):
            if self.sidecar_writable():
                try:
                    self._save_slices(sc, self.mla.shard.host_full)
                except OSError as e:
                    print(f"[tp_dsa] slice sidecar {sc} not written ({e})", flush=True)
            self.mla.shard.host_full = None
        del old
        self.dsa_pending, self._attach = False, None
