# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP-sharded, traceable DFlash / DFlash2 drafter (the production speed form of tt/dflash2.py).

Same math as DFlash2Drafter (validated against the oracle), restructured for speed:

* Weights SHARDED across the TP mesh (q/k/v/gate/up column-parallel, o/down/fc row-parallel), so
  each device reads 1/4 of the ~1.8B drafter parameters per draft instead of all of them.
  Activations stay REPLICATED full-width (8-16 rows: tiny); a row-parallel matmul is followed by
  reduce-scatter + all-gather (the same tt_all_reduce / tt_all_gather the target uses).
* The context feature fc runs ROW-PARALLEL on the taps as they already are on device — each device
  holds a dim/tp slice of every tap (the verify trace's fixed buffers, or the eager prefill clones) —
  so no tap ever round-trips through the host.
* lm_head runs on the target's vocab-sharded weight WITHOUT the vocab all-gather: top-16 per device
  over its V/4 slice, the 4x16 candidates are merged on host (exact).
* Both per-iteration forwards (draft, context extend) are trace-captured on demand: every input they
  read is a persistent fixed-address buffer staged per iteration (like the verify trace).

Indexing is identical to DFlash2Drafter: context 0..p, block [pending @ C=p+1, MASK...], K = block-1.
"""
import os

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.attention.rope_tp import apply_partial_rope_decode, apply_partial_rope_prefill
from models.demos.blackhole.qwen36.tt.dflash2 import (
    CKC,
    HD,
    NH,
    NKV,
    H,
    _expansion,
    _shift,
    load_config,
    resolve_weights_dir,
    rope_tables,
    select_path,
)
from models.tt_transformers.tt.ccl import tt_all_gather, tt_all_reduce


class DFlash2DrafterTP:
    def __init__(
        self,
        mesh,
        weights_dir,
        embed_host,
        tt_ccl,
        topology,
        lm_head_weight,
        lm_vocab_sharded=True,
        block=None,
        w_dtype=ttnn.bfloat16,
        mlp_dtype=None,
        kv_dtype=ttnn.bfloat16,
        cache_dir=None,
    ):
        import glob

        from safetensors.torch import load_file

        self.mesh = mesh
        self.nd = mesh.get_num_devices()
        self.tt_ccl = tt_ccl
        self.topo = topology
        self.embed = embed_host
        self.lm_w = lm_head_weight  # [H, V/nd] per device (vocab-sharded) or [H, V] replicated
        self.lm_sharded = bool(lm_vocab_sharded and self.nd > 1)
        self.cfg = load_config(weights_dir)
        self.theta = self.cfg["rope_theta"]
        self.block = int(block) if block else self.cfg["block"]
        assert 2 <= self.block <= self.cfg["block"]
        self.K = self.block - 1
        self.taps = self.cfg["taps"]
        self.mask_id = self.cfg["mask_id"]
        self.causal = list(self.cfg["causal"])
        _w = os.environ.get("QWEN36_DFLASH_WINDOW")
        self.windows = [(int(_w) if _w is not None else w) if w else 0 for w in self.cfg["windows"]]
        self.has_selector = self.cfg["has_selector"]
        self.topk = 16
        self.w_dtype = w_dtype
        self.mlp_dtype = mlp_dtype or w_dtype
        self.kv_dtype = kv_dtype
        assert NH % self.nd == 0 and NKV % self.nd == 0
        self.NHl, self.NKVl = NH // self.nd, NKV // self.nd
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        weights_dir = resolve_weights_dir(weights_dir)
        s = {}
        for f in sorted(glob.glob(f"{weights_dir}/*.safetensors")):
            s.update(load_file(f))
        # fc row-parallel over the DEVICE-ORDERED tap slices: device d holds columns d*dim_tp..(d+1)*dim_tp
        # of every tap, concatenated tap-major -> permute W_fc^T rows to (device, tap, col) order and
        # shard along rows.
        dim_tp = H // self.nd
        Wt = s["fc.weight"].T.contiguous()  # [ntaps*H, H]
        ntaps = Wt.shape[0] // H
        perm = torch.cat(
            [
                torch.cat([torch.arange(t * H + d * dim_tp, t * H + (d + 1) * dim_tp) for t in range(ntaps)])
                for d in range(self.nd)
            ]
        )
        self.fc = self._w(Wt[perm].contiguous(), "fc", dim=0)  # [ntaps*dim_tp, H] per device
        self.hnorm = self._rep(s["hidden_norm.weight"], "hidden_norm")
        self.fnorm = self._rep(s["norm.weight"], "norm")
        self.E = self._rep(_expansion(), "E", cache=False)  # host-built constants: never disk-cached
        self.layers = [self._load_layer(s, i) for i in range(self.cfg["num_layers"])]
        if self.has_selector:
            self.pred_cb = s["candidate_selector.predecessor_codebook"].float()
            self.succ_cb = s["candidate_selector.successor_codebook"].float()
            self.hproj = self._rep(s["candidate_selector.hidden_projection.weight"].T.contiguous(), "hproj")
        self.S = self._rep(_shift(self.block), "shift", cache=False)  # depends on the block
        self._sc1 = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, HD),
            core_grid=ttnn.CoreGrid(x=1, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        grid = mesh.compute_with_storage_grid_size()
        self._sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0
        )
        # QWEN36_DFLASH_SDPA_FP32=0: bf16 dest-acc in the decode SDPA only (HiFi4/no-approx kept). fp32 dest
        # halves the kernel's dynamic K chunk (dst_size 4 vs 8 tiles) -> 2x chunk iterations per layer,
        # bounded by the 2048 window. Default keeps the validated fp32 accumulation.
        self._sdpa_ckc = (
            CKC
            if os.environ.get("QWEN36_DFLASH_SDPA_FP32", "1") == "1"
            else ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
        )
        # per-generate state
        self.kv = None
        self.page_table = None
        self.block_size = None
        self.scratch_block = None
        self.tap_bufs = None  # list of persistent fractured [1,1,B,dim_tp] (the verify trace's taps)
        self._bufs = None  # persistent staging buffers
        self._draft_tid = None
        self._draft_out = None
        self._ext_tid = None
        self.last_hidden = None
        self.stats = {"draft_eager": 0, "draft_traced": 0, "extend_eager": 0, "extend_traced": 0}

    # ------------------------------------------------------------------ weights
    def _cache(self, name, dim, dtype):
        if not self.cache_dir:
            return None
        return f"{self.cache_dir}/{name}.d{dim}.{str(dtype).split('.')[-1]}.nd{self.nd}"

    def _w(self, t, name, dim, dtype=None):
        """Sharded weight: torch [in,out] TILE, split along `dim` (-1 column-parallel, 0 row-parallel)."""
        dtype = dtype or self.w_dtype
        mapper = ttnn.ShardTensorToMesh(self.mesh, dim=dim) if self.nd > 1 else None
        return ttnn.as_tensor(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=self._cache(name, dim, dtype),
            **({"mesh_mapper": mapper} if mapper else {}),
        )

    def _rep(self, t, name, dtype=ttnn.bfloat16, cache=True):
        if t.ndim == 1:
            t = t.reshape(1, -1)
        mapper = ttnn.ReplicateTensorToMesh(self.mesh) if self.nd > 1 else None
        return ttnn.as_tensor(
            t.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=self._cache(name, "r", dtype) if cache else None,
            **({"mesh_mapper": mapper} if mapper else {}),
        )

    def _load_layer(self, s, i):
        p = f"layers.{i}"
        lw = {
            "in_ln": self._rep(s[f"{p}.input_layernorm.weight"], f"l{i}.in_ln"),
            "post_ln": self._rep(s[f"{p}.post_attention_layernorm.weight"], f"l{i}.post_ln"),
            "qn": self._rep(s[f"{p}.self_attn.q_norm.weight"], f"l{i}.qn"),
            "kn": self._rep(s[f"{p}.self_attn.k_norm.weight"], f"l{i}.kn"),
        }
        for k in ("q", "k", "v"):
            lw[k] = self._w(s[f"{p}.self_attn.{k}_proj.weight"].T.contiguous(), f"l{i}.{k}", dim=-1)
        lw["o"] = self._w(s[f"{p}.self_attn.o_proj.weight"].T.contiguous(), f"l{i}.o", dim=0)
        lw["gate"] = self._w(s[f"{p}.mlp.gate_proj.weight"].T.contiguous(), f"l{i}.gate", dim=-1, dtype=self.mlp_dtype)
        lw["up"] = self._w(s[f"{p}.mlp.up_proj.weight"].T.contiguous(), f"l{i}.up", dim=-1, dtype=self.mlp_dtype)
        lw["down"] = self._w(s[f"{p}.mlp.down_proj.weight"].T.contiguous(), f"l{i}.down", dim=0, dtype=self.mlp_dtype)
        for tag, cp in (("ac", f"{p}.attention_conv"), ("mc", f"{p}.mlp_conv")):
            if f"{cp}.kernel_projection.weight" not in s:
                continue
            lw[f"{tag}_kp"] = self._rep(s[f"{cp}.kernel_projection.weight"].T.contiguous(), f"l{i}.{tag}_kp")
            bk = s[f"{cp}.base_kernel"].float()
            for a in (0, 1):
                for b in (0, 1):
                    lw[f"{tag}_b{a}{b}"] = self._rep(bk[a, b].reshape(1, 1, 1, H), f"l{i}.{tag}_b{a}{b}")
        return lw

    def free_weights(self):
        """Release every device weight (reload with another checkpoint/block must not leak)."""
        self.free()
        seen = set()

        def _free(t):
            if isinstance(t, ttnn.Tensor) and id(t) not in seen:
                seen.add(id(t))
                ttnn.deallocate(t)

        for name in ("fc", "hnorm", "fnorm", "E", "S", "hproj"):
            _free(getattr(self, name, None))
            setattr(self, name, None)
        for lw in self.layers:
            for t in lw.values():
                _free(t)
        self.layers = []
        self.dead = True

    # ------------------------------------------------------------------ primitives
    def _lin(self, x, w, **k):
        return ttnn.linear(x, w, compute_kernel_config=CKC, **k)

    def _rms(self, x, w):
        return ttnn.rms_norm(x, weight=w, epsilon=1e-6, compute_kernel_config=CKC)

    def _allreduce(self, x4):
        """Row-parallel partial [1,1,n,H] (consumed) -> replicated full [1,1,n,H]."""
        if self.nd == 1:
            return x4
        frac = tt_all_reduce(
            x4, self.mesh, self.tt_ccl, cluster_axis=0, dim=3, topology=self.topo, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        full = tt_all_gather(
            frac,
            self.mesh,
            self.tt_ccl,
            cluster_axis=None,
            dim=3,
            topology=self.topo,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(frac)
        return full

    def _rm(self, t):
        return ttnn.from_torch(
            t.to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def _dev(self, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            t, dtype=dtype, layout=layout, device=self.mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
        )

    def _stage(self, host_t, buf, dtype, layout=ttnn.TILE_LAYOUT):
        h = ttnn.from_torch(
            host_t, dtype=dtype, layout=layout, device=None, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
        )
        ttnn.copy_host_to_device_tensor(h, buf)

    def _gdc(self, values, kp, b0, b1, sel, L):
        """GroupedDynamicCausalConv on 4D [1,1,L,H] rows (replicated)."""
        off = sel * 640
        d0 = ttnn.slice(kp, (0, 0, 0, off), (1, 1, L, off + 320))
        d1 = ttnn.slice(kp, (0, 0, 0, off + 320), (1, 1, L, off + 640))
        c0 = ttnn.add(self._lin(d0, self.E), b0)
        c1 = ttnn.add(self._lin(d1, self.E), b1)
        shifted = ttnn.matmul(self.S, values, compute_kernel_config=CKC)  # [1,L,L] @ [1,1,L,H]
        out = ttnn.add(ttnn.mul(c0, values), ttnn.mul(c1, shifted))
        for t in (d0, d1, c0, c1, shifted):
            ttnn.deallocate(t)
        return out

    def _mlp(self, x, lw):
        g = self._lin(x, lw["gate"], activation="silu")
        u = self._lin(x, lw["up"])
        gu = ttnn.mul(g, u)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        part = self._lin(gu, lw["down"])
        ttnn.deallocate(gu)
        return self._allreduce(part)

    def _hctx(self, taps_dev, n):
        """Fractured taps (list of [1,1,n,dim_tp]) -> replicated hctx [1,1,n,H]."""
        x = ttnn.concat(list(taps_dev), dim=-1)  # [1,1,n,ntaps*dim_tp]
        part = self._lin(x, self.fc)
        ttnn.deallocate(x)
        full = self._allreduce(part)
        out = self._rms(full, self.hnorm)
        ttnn.deallocate(full)
        return out

    def _ctx_kv(self, hctx, lw, n):
        k = ttnn.reshape(self._lin(hctx, lw["k"]), (1, n, self.NKVl, HD))
        kn = self._rms(k, lw["kn"])
        ttnn.deallocate(k)
        v = ttnn.reshape(self._lin(hctx, lw["v"]), (1, n, self.NKVl, HD))
        return kn, v

    def _write_rows(self, li, k, v, pos_t, pt_t):
        kc, vc = self.kv[li]
        B = k.shape[1]
        nb = pt_t.shape[-1]
        k_p = ttnn.pad(k, [1, B, ttnn.TILE_SIZE, HD], [0, 0, 0, 0], 0.0)
        v_p = ttnn.pad(v, [1, B, ttnn.TILE_SIZE, HD], [0, 0, 0, 0], 0.0)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        for i in range(B):
            pos_i = ttnn.slice(pos_t, (i,), (i + 1,))
            pt_i = ttnn.slice(pt_t, (i, 0), (i + 1, nb))
            for cache, src in ((kc, k_p), (vc, v_p)):
                row = ttnn.slice(src, (0, i, 0, 0), (1, i + 1, ttnn.TILE_SIZE, HD))
                row_sh = ttnn.to_memory_config(row, self._sc1)
                ttnn.deallocate(row)
                ttnn.experimental.paged_update_cache(cache, row_sh, update_idxs_tensor=pos_i, page_table=pt_i)
                ttnn.deallocate(row_sh)
            ttnn.deallocate(pos_i)
            ttnn.deallocate(pt_i)
        ttnn.deallocate(k_p)
        ttnn.deallocate(v_p)

    # ------------------------------------------------------------------ per-generate state
    def alloc(self, page_table_torch, block_size, tap_bufs):
        """Draft KV (nb+1 blocks, block nb = scratch) + persistent staging buffers. Call BEFORE any
        trace is captured. ``tap_bufs``: the model's persistent fractured tap buffers ([1,1,B,dim_tp]
        each, one per tap layer) that the verify trace fills — the extend reads them directly."""
        assert self.layers and not getattr(
            self, "dead", False
        ), "drafter weights were released (another checkpoint/block/TP was loaded via get_drafter); rebuild the decoder"
        assert self.kv is None
        B = self.block
        nb = int(page_table_torch.shape[-1])
        self.page_table = page_table_torch.to(torch.int32)
        self.block_size = int(block_size)
        self.scratch_block = nb
        self.tap_bufs = list(tap_bufs)
        assert (
            len(self.tap_bufs) == len(self.taps) and self.tap_bufs[0].shape[-2] == B
        ), f"tap bufs {[tuple(t.shape) for t in self.tap_bufs]} vs block {B}"
        shape = [nb + 1, self.NKVl, self.block_size, HD]
        rep = ttnn.ReplicateTensorToMesh(self.mesh) if self.nd > 1 else None

        def _mk():
            return ttnn.as_tensor(
                torch.zeros(shape, dtype=torch.bfloat16),
                device=self.mesh,
                dtype=self.kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **({"mesh_mapper": rep} if rep else {}),
            )

        self.kv = [(_mk(), _mk()) for _ in range(len(self.layers))]
        z = torch.zeros
        self._bufs = {
            "ptB": self._rm(self.page_table.repeat(B, 1).contiguous()),
            "emb": self._dev(z(1, 1, B, H, dtype=torch.bfloat16)),
            "cos": self._dev(z(1, B, 1, HD, dtype=torch.bfloat16)),
            "sin": self._dev(z(1, B, 1, HD, dtype=torch.bfloat16)),
            "pos": self._rm(z(B, dtype=torch.int32)),
            "cur": self._rm(z(B, dtype=torch.int32)),
            "xpos": self._rm(z(B, dtype=torch.int32)),
            "xpt": self._rm(self.page_table.repeat(B, 1).contiguous()),
            "xcos": self._dev(z(1, B, 1, HD, dtype=torch.bfloat16)),
            "xsin": self._dev(z(1, B, 1, HD, dtype=torch.bfloat16)),
        }

    def free(self):
        ttnn.synchronize_device(self.mesh)  # a non-blocking replay may still be in flight (exception path)
        self._trace_armed = False
        for tid in (self._draft_tid, self._ext_tid):
            if tid is not None:
                ttnn.release_trace(self.mesh, tid)
        self._draft_tid = self._ext_tid = None
        self._draft_out = None
        if self.kv is not None:
            for kc, vc in self.kv:
                ttnn.deallocate(kc)
                ttnn.deallocate(vc)
            self.kv = None
        if self._bufs is not None:
            for t in self._bufs.values():
                ttnn.deallocate(t)
            self._bufs = None
        self.tap_bufs = None

    # ------------------------------------------------------------------ context: prompt fill (eager)
    def fill_context(self, taps_dev, chunk_start):
        """Prompt chunk: taps_dev = list of fractured [1,1,S,dim_tp] (S = bucket rows, block-aligned),
        chunk_start block-aligned -> k/v of hctx into the draft KV via paged_fill_cache."""
        assert self.kv is not None
        S = taps_dev[0].shape[-2]
        bs = self.block_size
        assert S % bs == 0 and chunk_start % bs == 0
        blk0 = chunk_start // bs
        blkN = min(blk0 + S // bs, int(self.page_table.shape[-1]))
        rows = (blkN - blk0) * bs
        assert rows > 0
        hctx = self._hctx(taps_dev, S)
        cos, sin = rope_tables(self.theta, torch.arange(chunk_start, chunk_start + S))
        cos = self._dev(cos.reshape(1, 1, S, HD))
        sin = self._dev(sin.reshape(1, 1, S, HD))
        chunk_pt = self._rm(self.page_table[:, blk0:blkN].contiguous())
        for li, lw in enumerate(self.layers):
            kc, vc = self.kv[li]
            k, v = self._ctx_kv(hctx, lw, S)
            kt = ttnn.transpose(k, 1, 2)
            ttnn.deallocate(k)
            k = apply_partial_rope_prefill(kt, cos, sin, self.NKVl, HD)
            ttnn.deallocate(kt)
            vt = ttnn.transpose(v, 1, 2)
            ttnn.deallocate(v)
            v = vt
            if rows < S:
                k2 = ttnn.slice(k, (0, 0, 0, 0), (1, self.NKVl, rows, HD))
                ttnn.deallocate(k)
                k = k2
                v2 = ttnn.slice(v, (0, 0, 0, 0), (1, self.NKVl, rows, HD))
                ttnn.deallocate(v)
                v = v2
            if k.dtype != self.kv_dtype:
                k = ttnn.typecast(k, self.kv_dtype)
                v = ttnn.typecast(v, self.kv_dtype)
            ttnn.experimental.paged_fill_cache(kc, k, chunk_pt, batch_idx=0)
            ttnn.experimental.paged_fill_cache(vc, v, chunk_pt, batch_idx=0)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        for t in (hctx, chunk_pt, cos, sin):
            ttnn.deallocate(t)

    # ------------------------------------------------------------------ context: extend (traceable)
    def _extend_body(self):
        b = self._bufs
        B = self.block
        hctx = self._hctx(self.tap_bufs, B)
        for li, lw in enumerate(self.layers):
            k, v = self._ctx_kv(hctx, lw, B)
            kr = apply_partial_rope_decode(k, b["xcos"], b["xsin"], self.NKVl, B, HD)
            ttnn.deallocate(k)
            self._write_rows(li, kr, v, b["xpos"], b["xpt"])
        ttnn.deallocate(hctx)

    def _stage_extend(self, slot0, n_valid):
        B = self.block
        n = int(n_valid)
        pos = torch.zeros(B, dtype=torch.int32)
        pos[:n] = torch.arange(slot0, slot0 + n, dtype=torch.int32)
        pt = self.page_table.repeat(B, 1).contiguous()
        pt[n:, :] = self.scratch_block
        cos, sin = rope_tables(self.theta, pos)
        self._stage(pos, self._bufs["xpos"], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self._stage(pt, self._bufs["xpt"], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self._stage(cos.reshape(1, B, 1, HD).bfloat16(), self._bufs["xcos"], ttnn.bfloat16)
        self._stage(sin.reshape(1, B, 1, HD).bfloat16(), self._bufs["xsin"], ttnn.bfloat16)

    def extend_context(self, slot0, n_valid, traced=True):
        """Context rows 0..n_valid-1 of the tap bufs -> slots slot0..; the rest -> scratch."""
        assert self.kv is not None
        self._stage_extend(slot0, n_valid)
        if traced and self._ext_tid is None and self._trace_armed:
            self._ext_tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._extend_body()
            ttnn.end_trace_capture(self.mesh, self._ext_tid, cq_id=0)
            logger.info("[dflash2-tp] extend trace captured")
        if traced and self._ext_tid is not None:
            ttnn.execute_trace(self.mesh, self._ext_tid, cq_id=0, blocking=False)
            self.stats["extend_traced"] += 1
        else:
            self._extend_body()
            self.stats["extend_eager"] += 1

    def extend_from_taps(self, taps_dev, slot0, n_valid):
        """Eager extend from arbitrary fractured taps (>= B rows; e.g. the seed's bucket-128 clones):
        rows 0..B-1 are copied into the persistent tap bufs first, so the body is byte-identical to
        the traced one (and compiles it, pre-capture)."""
        B = self.block
        for src, buf in zip(taps_dev, self.tap_bufs):
            if src.shape[-2] == B:
                ttnn.copy(src, buf)
            else:
                sl = ttnn.slice(src, (0, 0, 0, 0), (1, 1, B, src.shape[-1]))
                ttnn.copy(sl, buf)
                ttnn.deallocate(sl)
        self.extend_context(slot0, n_valid, traced=False)

    # ------------------------------------------------------------------ draft (traceable)
    def _attn(self, hb, lw, li, cur_t):
        b = self._bufs
        B = self.block
        q = ttnn.reshape(self._lin(hb, lw["q"]), (1, B, self.NHl, HD))
        qn = self._rms(q, lw["qn"])
        ttnn.deallocate(q)
        k = ttnn.reshape(self._lin(hb, lw["k"]), (1, B, self.NKVl, HD))
        kn = self._rms(k, lw["kn"])
        ttnn.deallocate(k)
        v = ttnn.reshape(self._lin(hb, lw["v"]), (1, B, self.NKVl, HD))
        qr = apply_partial_rope_decode(qn, b["cos"], b["sin"], self.NHl, B, HD)
        ttnn.deallocate(qn)
        kr = apply_partial_rope_decode(kn, b["cos"], b["sin"], self.NKVl, B, HD)
        ttnn.deallocate(kn)
        self._write_rows(li, kr, v, b["pos"], b["ptB"])
        kc, vc = self.kv[li]
        kw = {"sliding_window_size": self.windows[li]} if self.windows[li] else {}
        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            qr,
            kc,
            vc,
            page_table_tensor=b["ptB"],
            cur_pos_tensor=cur_t,
            scale=HD**-0.5,
            program_config=self._sdpa_cfg,
            compute_kernel_config=self._sdpa_ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **kw,
        )
        ttnn.deallocate(qr)
        o4 = ttnn.reshape(o, (1, 1, B, self.NHl * HD))
        part = self._lin(o4, lw["o"])
        ttnn.deallocate(o4)
        return self._allreduce(part)

    def _layer(self, hidden, lw, li):
        L = self.block
        cur_t = self._bufs["pos"] if self.causal[li] else self._bufs["cur"]
        h = self._rms(hidden, lw["in_ln"])
        if "ac_kp" in lw:
            kp = self._lin(h, lw["ac_kp"])
            hb = self._gdc(h, kp, lw["ac_b00"], lw["ac_b01"], 0, L)
            a0 = self._attn(hb, lw, li, cur_t)
            ttnn.deallocate(hb)
            a = self._gdc(a0, kp, lw["ac_b10"], lw["ac_b11"], 1, L)
            ttnn.deallocate(a0)
            ttnn.deallocate(kp)
        else:
            a = self._attn(h, lw, li, cur_t)
        ttnn.deallocate(h)
        h1 = ttnn.add(hidden, a)
        ttnn.deallocate(a)
        h2 = self._rms(h1, lw["post_ln"])
        if "mc_kp" in lw:
            kpm = self._lin(h2, lw["mc_kp"])
            hm = self._gdc(h2, kpm, lw["mc_b00"], lw["mc_b01"], 0, L)
            m0 = self._mlp(hm, lw)
            ttnn.deallocate(hm)
            m = self._gdc(m0, kpm, lw["mc_b10"], lw["mc_b11"], 1, L)
            ttnn.deallocate(m0)
            ttnn.deallocate(kpm)
        else:
            m = self._mlp(h2, lw)
        ttnn.deallocate(h2)
        out = ttnn.add(h1, m)
        ttnn.deallocate(m)
        ttnn.deallocate(h1)
        return out

    def _draft_body(self):
        """Persistent inputs -> (vals [1,1,B,16] local, idx [1,1,B,16] local, hp [1,1,B,256] | None, h [1,1,B,H])."""
        b = self._bufs
        B = self.block
        h = b["emb"]
        for li, lw in enumerate(self.layers):
            h_new = self._layer(h, lw, li)
            if h is not b["emb"]:
                ttnn.deallocate(h)
            h = h_new
        hn = self._rms(h, self.fnorm)
        ttnn.deallocate(h)
        logits = ttnn.linear(hn, self.lm_w)  # [1,1,B,V/nd] per device
        vals, idx = ttnn.topk(logits, self.topk, dim=-1, largest=True, sorted=True)
        ttnn.deallocate(logits)
        hp = self._lin(hn, self.hproj) if self.has_selector else None
        return vals, idx, hp, hn

    def _stage_draft(self, anchor, C):
        B = self.block
        blk = torch.tensor([[int(anchor)] + [self.mask_id] * (B - 1)])
        emb = torch.nn.functional.embedding(blk, self.embed).reshape(1, 1, B, H).to(torch.bfloat16)
        pos = torch.arange(C, C + B, dtype=torch.int32)
        cos, sin = rope_tables(self.theta, pos)
        self._stage(emb, self._bufs["emb"], ttnn.bfloat16)
        self._stage(cos.reshape(1, B, 1, HD).bfloat16(), self._bufs["cos"], ttnn.bfloat16)
        self._stage(sin.reshape(1, B, 1, HD).bfloat16(), self._bufs["sin"], ttnn.bfloat16)
        self._stage(pos, self._bufs["pos"], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self._stage(
            torch.full((B,), C + B - 1, dtype=torch.int32), self._bufs["cur"], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
        )

    _trace_armed = False

    def arm_traces(self):
        """Allow draft()/extend_context() to capture their traces on the next call. Call only once
        every program they need has been compiled (i.e. after the pre-capture warmups) and after the
        verify trace is captured."""
        self._trace_armed = True

    def draft(self, anchor, C, traced=True):
        assert self.kv is not None
        B = self.block
        self._stage_draft(anchor, C)
        if traced and self._draft_tid is None and self._trace_armed:
            self._draft_tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._draft_out = self._draft_body()
            ttnn.end_trace_capture(self.mesh, self._draft_tid, cq_id=0)
            logger.info("[dflash2-tp] draft trace captured")
        if traced and self._draft_tid is not None:
            ttnn.execute_trace(self.mesh, self._draft_tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh)
            vals, idx, hp, hn = self._draft_out
            owned = False
            self.stats["draft_traced"] += 1
        else:
            vals, idx, hp, hn = self._draft_body()
            owned = True
            self.stats["draft_eager"] += 1
        comp0 = ttnn.ConcatMeshToTensor(self.mesh, dim=0)
        if self.lm_sharded:
            V_l = self.lm_w.shape[-1]
            va = ttnn.to_torch(vals, mesh_composer=comp0).float().reshape(self.nd, B, self.topk)  # [nd,B,16]
            ia = ttnn.to_torch(idx, mesh_composer=comp0).long().reshape(self.nd, B, self.topk)
            ia = ia + torch.arange(self.nd).view(self.nd, 1, 1) * V_l
            va = va.permute(1, 0, 2).reshape(B, -1)  # [B, nd*16]
            ia = ia.permute(1, 0, 2).reshape(B, -1)
            top = va.topk(self.topk, dim=-1)
            unary = top.values[1:]
            cand = ia.gather(-1, top.indices)[1:]
        else:
            unary = ttnn.to_torch(ttnn.get_device_tensors(vals)[0]).float().reshape(B, self.topk)[1:]
            cand = ttnn.to_torch(ttnn.get_device_tensors(idx)[0]).long().reshape(B, self.topk)[1:]
        if self.has_selector:
            hph = ttnn.to_torch(ttnn.get_device_tensors(hp)[0]).float().reshape(B, -1)[1:]
            out = select_path(unary, cand, hph, anchor, self.pred_cb, self.succ_cb)
        else:
            out = [int(c[int(u.argmax())]) for u, c in zip(unary, cand)]
        self.last_hidden = ttnn.to_torch(ttnn.get_device_tensors(hn)[0]).float().reshape(1, B, H)[:, 1:]
        if owned:
            for t in (vals, idx, hp, hn):
                if t is not None:
                    ttnn.deallocate(t)
        return out
