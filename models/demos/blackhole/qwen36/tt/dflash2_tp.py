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
from models.demos.blackhole.qwen36.tt.attention.tp import KV_GROUP_WRITE_OK, _aliases
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
        # The in-block shift (GroupedDynamicCausalConv's second tap) is built per alloc(): it is
        # block-diagonal over the U users' blocks, so it depends on U as well as the block.
        self.S = None
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
        self.ring = 0  # ring_tokens (serving) or 0
        self.kv = None
        self.page_tables = None  # torch int32 [U, nb]: user u's context blocks (row u)
        self.page_table = None  # alias of page_tables (kept for single-user callers)
        self.U = 1  # users per draft/extend (rows = U * block, user-major: row u*block + j)
        self.R = self.block
        self.block_size = None
        self.scratch_block = None
        self.tap_bufs = None  # list of persistent fractured [1,1,R,dim_tp] (the verify trace's taps)
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

    def _kv_cfg(self, n):
        """HEIGHT shard for an n-row paged_update_cache input (one 32-row tile per core), cached per n."""
        cache = getattr(self, "_kv_cfg_cache", None)
        if cache is None:
            cache = self._kv_cfg_cache = {}
        if n not in cache:
            cols = next(c for c in range(min(8, n), 0, -1) if n % c == 0)
            cache[n] = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, HD),
                core_grid=ttnn.CoreGrid(x=cols, y=n // cols),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        return cache[n]

    def _write_rows(self, li, k, v, pos_t, pt_t):
        """Write the R = U*block draft/extend rows' K/V into the paged context cache.

        Rows of ONE user are consecutive positions of one sequence, so they share a cache tile and
        must never ride one paged_update_cache call (several cores would read-modify-write the same
        tile; last writer wins). Rows of DIFFERENT users live in different blocks. So at U > 1 the
        write goes out GROUPED by in-block index j: call j carries rows {u*block + j}, one row per
        user, U rows -> 2*block calls per layer instead of 2*R (TPAttention._write_kv_aliased's
        idiom: tile-view reshape [1,R,32,HD] -> [1,U,block*32,HD], tile-aligned slice, strided
        slices of the per-row position / page-table tensors). Padding rows of different users all
        name the scratch block and may collide there: that block is write-only junk.
        QWEN36_DFLASH_KV_GROUP_WRITE=0 keeps the per-row loop (A/B)."""
        kc, vc = self.kv[li]
        R = k.shape[1]
        nb = pt_t.shape[-1]
        tile = ttnn.TILE_SIZE
        k_p = ttnn.pad(k, [1, R, tile, HD], [0, 0, 0, 0], 0.0)
        v_p = ttnn.pad(v, [1, R, tile, HD], [0, 0, 0, 0], 0.0)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        U, Bk = self.U, self.block
        grouped = (
            U > 1
            and Bk >= 2
            and R == U * Bk
            and KV_GROUP_WRITE_OK
            and os.environ.get("QWEN36_DFLASH_KV_GROUP_WRITE", "1") != "0"
        )
        if not grouped:
            for i in range(R):
                pos_i = ttnn.slice(pos_t, (i,), (i + 1,))
                pt_i = ttnn.slice(pt_t, (i, 0), (i + 1, nb))
                for cache, src in ((kc, k_p), (vc, v_p)):
                    row = ttnn.slice(src, (0, i, 0, 0), (1, i + 1, tile, HD))
                    row_sh = ttnn.to_memory_config(row, self._sc1)
                    ttnn.deallocate(row)
                    ttnn.experimental.paged_update_cache(cache, row_sh, update_idxs_tensor=pos_i, page_table=pt_i)
                    ttnn.deallocate(row_sh)
                ttnn.deallocate(pos_i)
                ttnn.deallocate(pt_i)
            ttnn.deallocate(k_p)
            ttnn.deallocate(v_p)
            return
        cfg = self._kv_cfg(U)
        view = ttnn.Shape([1, U, Bk * tile, HD])
        srcs = []
        for t in (k_p, v_p):
            vw = ttnn.reshape(t, view, view)  # tile-view alias (same buffer) -- see _write_kv_aliased
            srcs.append((vw, None if _aliases(vw, t) else t))
        (k_v, k_orig), (v_v, v_orig) = srcs
        for j in range(Bk):
            r0 = j * tile
            pos_j = ttnn.slice(pos_t, (j,), (R,), (Bk,))  # rows j, j+Bk, ... = user 0..U-1 at in-block index j
            pt_j = ttnn.slice(pt_t, (j, 0), (R, nb), (Bk, 1))
            for cache, src in ((kc, k_v), (vc, v_v)):
                grp = ttnn.slice(src, (0, 0, r0, 0), (1, U, r0 + tile, HD))
                grp_sh = ttnn.to_memory_config(grp, cfg)
                ttnn.deallocate(grp)
                ttnn.experimental.paged_update_cache(cache, grp_sh, update_idxs_tensor=pos_j, page_table=pt_j)
                ttnn.deallocate(grp_sh)
            ttnn.deallocate(pos_j)
            ttnn.deallocate(pt_j)
        for t in (k_v, v_v, k_orig, v_orig):
            if t is not None:
                ttnn.deallocate(t)

    # ------------------------------------------------------------------ per-generate state
    def alloc(self, page_tables_torch, block_size, tap_bufs, ring_tokens=None):
        """Draft KV + persistent staging buffers for U users. Call BEFORE any trace is captured.

        ``ring_tokens`` (serving): treat each user's table as a CIRCULAR buffer of that many
        positions (a multiple of block_size, >= the widest sliding window + block, so every key a
        block row can attend is still resident): KV writes land at position % ring_tokens and the
        paged decode SDPA looks its keys up the same way (cache_position_modulo) while the causal /
        sliding-window bound stays on absolute positions. A user then costs ring_tokens/block_size
        blocks however long its request runs. Off (None): plain per-position tables (the demo).

        ``page_tables_torch``: torch int32 [U, nb] (a [nb] / [1, nb] table is U = 1), user u's
        context blocks in row u — the drafter mirrors the target's per-user block layout in its own
        cache, sized to the largest block id named plus one scratch block (index = that maximum + 1)
        for padding rows. Every draft/extend runs R = U * block rows, USER-MAJOR (row u*block + j is
        user u's block position j), the same layout as the verify bucket, so the verify trace's tap
        buffers ([1,1,R,dim_tp] each, one per tap layer; ``tap_bufs``) feed the extend directly.
        """
        assert self.layers and not getattr(
            self, "dead", False
        ), "drafter weights were released (another checkpoint/block/TP was loaded via get_drafter); rebuild the decoder"
        assert self.kv is None
        pt = torch.as_tensor(page_tables_torch)
        pt = pt.reshape(1, -1) if pt.dim() == 1 else pt
        assert pt.dim() == 2, f"page tables must be [U, nb], got {tuple(pt.shape)}"
        self.page_tables = pt.to(torch.int32).contiguous()
        self.page_table = self.page_tables
        U = int(self.page_tables.shape[0])
        Bk = self.block
        R = U * Bk
        assert R <= ttnn.TILE_SIZE, f"{U} users x block {Bk} = {R} draft rows exceed one decode tile"
        self.U, self.R = U, R
        nb = int(self.page_tables.shape[-1])
        self.block_size = int(block_size)
        self.scratch_block = int(self.page_tables.max()) + 1
        self.ring = int(ring_tokens) if ring_tokens else 0
        if self.ring:
            widest = max(self.windows) if any(self.windows) else 0
            assert all(
                self.windows
            ), "a ring context needs every drafter layer windowed (DFlash v1 has a full-attention layer)"
            assert (
                self.ring % self.block_size == 0
            ), f"ring {self.ring} is not a multiple of block_size {self.block_size}"
            assert (
                self.ring >= widest + Bk
            ), f"ring {self.ring} < window {widest} + block {Bk}: a block row would evict a key it attends"
            assert (
                nb * self.block_size == self.ring
            ), f"ring tables need exactly {self.ring // self.block_size} blocks per user, got {nb}"
        self.tap_bufs = list(tap_bufs)
        assert (
            len(self.tap_bufs) == len(self.taps) and self.tap_bufs[0].shape[-2] == R
        ), f"tap bufs {[tuple(t.shape) for t in self.tap_bufs]} vs {U} users x block {Bk} = {R} rows"
        shape = [self.scratch_block + 1, self.NKVl, self.block_size, HD]
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
        # Block-diagonal in-block shift: kron(I_U, shift(block)) -> [1, R, R]. Row u*Bk + j reads
        # row u*Bk + j - 1 (j > 0) and nothing else, so the conv never crosses a user boundary.
        S = torch.zeros(1, R, R)
        for u in range(U):
            S[0, u * Bk : (u + 1) * Bk, u * Bk : (u + 1) * Bk] = _shift(Bk)[0]
        if self.S is not None:
            ttnn.deallocate(self.S)
        self.S = self._rep(S, "shift", cache=False)
        # Per-ROW page tables (user-major): row u*Bk + j is user u's table.
        pt_rows = self.page_tables.repeat_interleave(Bk, dim=0).contiguous()
        z = torch.zeros
        self._bufs = {
            "ptB": self._rm(pt_rows),
            "emb": self._dev(z(1, 1, R, H, dtype=torch.bfloat16)),
            "cos": self._dev(z(1, R, 1, HD, dtype=torch.bfloat16)),
            "sin": self._dev(z(1, R, 1, HD, dtype=torch.bfloat16)),
            "pos": self._rm(z(R, dtype=torch.int32)),
            "wpos": self._rm(z(R, dtype=torch.int32)),  # KV write positions (== pos, or pos % ring)
            "cur": self._rm(z(R, dtype=torch.int32)),
            "xpos": self._rm(z(R, dtype=torch.int32)),
            "xpt": self._rm(pt_rows.clone()),
            "xcos": self._dev(z(1, R, 1, HD, dtype=torch.bfloat16)),
            "xsin": self._dev(z(1, R, 1, HD, dtype=torch.bfloat16)),
        }

    def release_traces(self):
        """Drop the draft/extend traces (and their output tensors) but keep the KV + staging buffers, so
        the next armed draft/extend re-captures against the same addresses."""
        ttnn.synchronize_device(self.mesh)
        self._trace_armed = False
        for tid in (self._draft_tid, self._ext_tid):
            if tid is not None:
                ttnn.release_trace(self.mesh, tid)
        self._draft_tid = self._ext_tid = None
        if self._draft_out is not None:
            for t in self._draft_out:
                if t is not None:
                    try:
                        ttnn.deallocate(t)
                    except Exception:
                        pass
        self._draft_out = None

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
        if self.S is not None:
            ttnn.deallocate(self.S)
            self.S = None
        self.tap_bufs = None

    # ------------------------------------------------------------------ context: prompt fill (eager)
    def fill_context(self, taps_dev, chunk_start, user=0, valid_len=None):
        """Prompt chunk of ONE user: taps_dev = list of fractured [1,1,S,dim_tp] (S = bucket rows,
        block-aligned), chunk_start block-aligned -> k/v of hctx into user ``user``'s blocks of the
        draft KV via paged_fill_cache. ``valid_len``: real rows in the chunk (the rest are bucket
        padding); writes are rounded up to whole blocks. On a ring the chunk's blocks are the
        user's table entries (chunk_start/bs + i) % nblk, and no more than the ring holds."""
        assert self.kv is not None
        assert 0 <= user < self.U, f"user {user} of {self.U}"
        S = taps_dev[0].shape[-2]
        bs = self.block_size
        assert S % bs == 0 and chunk_start % bs == 0
        nblk_tab = int(self.page_tables.shape[-1])
        n_valid = S if valid_len is None else min(S, int(valid_len))
        nblk = -(-n_valid // bs)
        blk0 = chunk_start // bs
        if self.ring:
            assert nblk <= nblk_tab, f"a {nblk}-block chunk does not fit the {nblk_tab}-block ring"
            ids = [(blk0 + i) % nblk_tab for i in range(nblk)]
            chunk_tab = self.page_tables[user, ids].reshape(1, nblk)
        else:
            nblk = min(nblk, nblk_tab - blk0)
            chunk_tab = self.page_tables[user : user + 1, blk0 : blk0 + nblk]
        rows = nblk * bs
        assert rows > 0
        hctx = self._hctx(taps_dev, S)
        cos, sin = rope_tables(self.theta, torch.arange(chunk_start, chunk_start + S))
        cos = self._dev(cos.reshape(1, 1, S, HD))
        sin = self._dev(sin.reshape(1, 1, S, HD))
        chunk_pt = self._rm(chunk_tab.contiguous())
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
        R = self.R
        hctx = self._hctx(self.tap_bufs, R)
        for li, lw in enumerate(self.layers):
            k, v = self._ctx_kv(hctx, lw, R)
            kr = apply_partial_rope_decode(k, b["xcos"], b["xsin"], self.NKVl, R, HD)
            ttnn.deallocate(k)
            self._write_rows(li, kr, v, b["xpos"], b["xpt"])
        ttnn.deallocate(hctx)

    def _per_user(self, v):
        """An int (U = 1 / broadcast) or a length-U list -> a length-U list of ints."""
        if isinstance(v, (list, tuple, torch.Tensor)):
            out = [int(x) for x in v]
            assert len(out) == self.U, f"expected {self.U} per-user values, got {len(out)}"
            return out
        return [int(v)] * self.U

    def _stage_extend(self, slot0, n_valid):
        Bk, R = self.block, self.R
        slot0s, ns = self._per_user(slot0), self._per_user(n_valid)
        pos = torch.zeros(R, dtype=torch.int32)
        pt = torch.full((R, int(self.page_tables.shape[-1])), self.scratch_block, dtype=torch.int32)
        for u, (s0, n) in enumerate(zip(slot0s, ns)):
            assert 0 <= n <= Bk, f"user {u}: extend {n} rows into a {Bk}-row block"
            r0 = u * Bk
            pos[r0 : r0 + n] = torch.arange(s0, s0 + n, dtype=torch.int32)
            pt[r0 : r0 + n, :] = self.page_tables[u]
        cos, sin = rope_tables(self.theta, pos)  # RoPE at the ABSOLUTE position
        self._stage(pos % self.ring if self.ring else pos, self._bufs["xpos"], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self._stage(pt, self._bufs["xpt"], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self._stage(cos.reshape(1, R, 1, HD).bfloat16(), self._bufs["xcos"], ttnn.bfloat16)
        self._stage(sin.reshape(1, R, 1, HD).bfloat16(), self._bufs["xsin"], ttnn.bfloat16)

    def extend_context(self, slot0, n_valid, traced=True):
        """Per user u: tap-buf rows u*block .. u*block+n_valid[u]-1 -> user u's context slots
        slot0[u]..; every other row -> the scratch block. ``slot0`` / ``n_valid`` are ints (U = 1) or
        length-U lists."""
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
        """Eager extend from arbitrary fractured taps (>= R rows; e.g. the seed's bucket-128 clones):
        rows 0..R-1 are copied into the persistent tap bufs first, so the body is byte-identical to
        the traced one (and compiles it, pre-capture). Single-user form (U = 1)."""
        R = self.R
        for src, buf in zip(taps_dev, self.tap_bufs):
            if src.shape[-2] == R:
                ttnn.copy(src, buf)
            else:
                sl = ttnn.slice(src, (0, 0, 0, 0), (1, 1, R, src.shape[-1]))
                ttnn.copy(sl, buf)
                ttnn.deallocate(sl)
        self.extend_context(slot0, n_valid, traced=False)

    def extend_seed_rows(self, taps_dev, slots):
        """Eager extend from the SEED's taps: ``taps_dev`` is one fractured [1,1,U,dim_tp] tensor per
        tap layer (row u = user u's residual at ``slots[u]``). Each user's row is placed at tap-buf
        row u*block (the user-major layout the extend body reads), the other rows are left as they
        are and routed to the scratch block, and the R-row extend body runs eagerly (pre-capture,
        so this also compiles it). Host-staged: a handful of rows, once per generation."""
        U, Bk = self.U, self.block
        slots = self._per_user(slots)
        comp = ttnn.ConcatMeshToTensor(self.mesh, dim=-1)
        shard = ttnn.ShardTensorToMesh(self.mesh, dim=-1)
        for src, buf in zip(taps_dev, self.tap_bufs):
            full = ttnn.to_torch(src, mesh_composer=comp).reshape(-1, H)[:U]  # [U, H] bf16
            host = ttnn.to_torch(buf, mesh_composer=comp).reshape(-1, H).clone()  # [R, H]
            for u in range(U):
                host[u * Bk] = full[u]
            h = ttnn.from_torch(
                host.reshape(1, 1, self.R, H).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=None,
                mesh_mapper=shard,
            )
            ttnn.copy_host_to_device_tensor(h, buf)
        self.extend_context(slots, [1] * U, traced=False)

    # ------------------------------------------------------------------ draft (traceable)
    def _attn(self, hb, lw, li, cur_t):
        b = self._bufs
        B = self.R  # every draft row (U users x block) is a pseudo-user of the paged decode SDPA
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
        self._write_rows(li, kr, v, b["wpos"], b["ptB"])
        kc, vc = self.kv[li]
        kw = {"sliding_window_size": self.windows[li]} if self.windows[li] else {}
        if self.ring:
            kw["cache_position_modulo"] = self.ring  # ring lookup; cur_pos / window bound stay absolute
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
        L = self.R
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
        """Persistent inputs -> (vals [1,1,R,16] local, idx [1,1,R,16] local, hp [1,1,R,256] | None, h [1,1,R,H])."""
        b = self._bufs
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
        """Per user u: block [anchor[u], MASK x (block-1)] at positions C[u] .. C[u]+block-1
        (user-major rows). ``anchor`` / ``C`` are ints (U = 1) or length-U lists."""
        Bk, R = self.block, self.R
        anchors, Cs = self._per_user(anchor), self._per_user(C)
        blk = torch.tensor([[a] + [self.mask_id] * (Bk - 1) for a in anchors]).reshape(1, R)
        emb = torch.nn.functional.embedding(blk, self.embed).reshape(1, 1, R, H).to(torch.bfloat16)
        pos = torch.cat([torch.arange(c, c + Bk, dtype=torch.int32) for c in Cs])
        cur = torch.cat([torch.full((Bk,), c + Bk - 1, dtype=torch.int32) for c in Cs])
        cos, sin = rope_tables(self.theta, pos)
        self._stage(emb, self._bufs["emb"], ttnn.bfloat16)
        self._stage(cos.reshape(1, R, 1, HD).bfloat16(), self._bufs["cos"], ttnn.bfloat16)
        self._stage(sin.reshape(1, R, 1, HD).bfloat16(), self._bufs["sin"], ttnn.bfloat16)
        self._stage(pos, self._bufs["pos"], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self._stage(pos % self.ring if self.ring else pos, self._bufs["wpos"], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self._stage(cur, self._bufs["cur"], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    _trace_armed = False

    def arm_traces(self):
        """Allow draft()/extend_context() to capture their traces on the next call. Call only once
        every program they need has been compiled (i.e. after the pre-capture warmups) and after the
        verify trace is captured."""
        self._trace_armed = True

    def draft(self, anchor, C, traced=True):
        """Draft K = block-1 tokens per user from ``anchor`` (the token at C-1... i.e. the pending
        token at position C[u]) over context 0..C[u]-1. Returns K ids (int anchor, U = 1) or a list
        of U lists of K ids (list anchors), in user order."""
        assert self.kv is not None
        Bk, R, U = self.block, self.R, self.U
        per_user_out = isinstance(anchor, (list, tuple))
        anchors = self._per_user(anchor)
        self._stage_draft(anchors, C)
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
            va = ttnn.to_torch(vals, mesh_composer=comp0).float().reshape(self.nd, R, self.topk)  # [nd,R,16]
            ia = ttnn.to_torch(idx, mesh_composer=comp0).long().reshape(self.nd, R, self.topk)
            ia = ia + torch.arange(self.nd).view(self.nd, 1, 1) * V_l
            va = va.permute(1, 0, 2).reshape(R, -1)  # [R, nd*16]
            ia = ia.permute(1, 0, 2).reshape(R, -1)
            top = va.topk(self.topk, dim=-1)
            unary_all = top.values
            cand_all = ia.gather(-1, top.indices)
        else:
            unary_all = ttnn.to_torch(ttnn.get_device_tensors(vals)[0]).float().reshape(R, self.topk)
            cand_all = ttnn.to_torch(ttnn.get_device_tensors(idx)[0]).long().reshape(R, self.topk)
        hph_all = ttnn.to_torch(ttnn.get_device_tensors(hp)[0]).float().reshape(R, -1) if self.has_selector else None
        hn_all = ttnn.to_torch(ttnn.get_device_tensors(hn)[0]).float().reshape(R, H)
        outs, hiddens = [], []
        for u in range(U):
            r0 = u * Bk
            unary = unary_all[r0 + 1 : r0 + Bk]  # rows 1..block-1: the K mask positions of user u
            cand = cand_all[r0 + 1 : r0 + Bk]
            if self.has_selector:
                outs.append(select_path(unary, cand, hph_all[r0 + 1 : r0 + Bk], anchors[u], self.pred_cb, self.succ_cb))
            else:
                outs.append([int(c[int(w.argmax())]) for w, c in zip(unary, cand)])
            hiddens.append(hn_all[r0 + 1 : r0 + Bk])
        self.last_hidden = torch.stack(hiddens)  # [U, K, H]
        if owned:
            for t in (vals, idx, hp, hn):
                if t is not None:
                    ttnn.deallocate(t)
        return outs if per_user_out else outs[0]
