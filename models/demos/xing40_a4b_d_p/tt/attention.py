# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Xing4.0 dense causal MLA on the 4x2 mesh (SP=4 over rows, axis 0 x TP=2 over columns, axis 1; plan.md).

    q        = q_b_proj(q_resid) [S, 32, 192] -> q_nope 128 | q_rope 64 (interleaved YaRN RoPE)
    kv       = kv_a_proj_with_mqa(attn_norm) [S, 576] -> kv_a_layernorm(latent 512, eps 1e-6) | RoPE(k_rope 64)
    q_abs    = [q_nope @ W_uk (128 -> 512 per head) | RoPE(q_rope)] (576)
    o_lat    = causal softmax(q_abs . kv_j * scale) @ latent_j (512), scale = 192^-0.5 * mscale^2
    attn_out = o_proj(concat_h(o_lat @ W_uv (512 -> 128 per head)))

This is the reference's kv_b re-expansion in absorbed form (same math). Adapted from
models/demos/deepseek_v3_d_p/tt/mla/mla.py:ttMLA's dense chunked path (_kv_stem, _q_stem, _update_kv_cache,
_chunked_attn -> ttnn.transformer.ring_mla, _o_proj_epilogue) and hy4_preview_d_p/tt/attention.py:TtHy4Attention
(ttMLA's ops without its fixed CCL buffers; _Geometry.load / read at the harness boundary). Differences to ttMLA:

- bf16 weights (as stored) and a bf16 TILE latent cache, not bfp8; every matmul, the norm, the RoPE and ring_mla at
  HiFi4 + fp32 dest (owner rule), except the ring_mla matmuls at HiFi2 (owner exception, P.1;
  XING_MLA_SDPA_FIDELITY). ring_mla through the sdpa fork (ttnn.bringup.ring_mla), whose streaming latent-V
  path runs at fp32 dest; XING_MLA_SDPA=source keeps ttnn.transformer.ring_mla at bf16 dest (owner 06:35).
- kv_a_proj: K-split fp32 partials + ttnn.all_reduce over axis 1; kv_a_layernorm through ttnn.bringup.rms_norm.
- o_proj: row-parallel fp32 partials -> ttnn.reduce_scatter over axis 1 -> attn_out [1, 1, S/4, 1792] fp32.
- RoPE tables from the reference's YaRN inv_freq (xing_ref.yarn_inv_freq), interleaved (Meta pair) order, so the
  checkpoint order needs no permutation (known issues: HF's "interleaved" RoPE).

Latent cache: [1, 1, max_seq / 4, 576] bf16 TILE per chip, block-cyclic over the 4 rows with period = the chunk
(row r holds [k chunk + r chunk/4, + chunk/4) of every chunk k), replicated over the 2 columns (both columns compute
the same all-reduced kv and write their own copy). ring_mla gathers the populated prefix around axis 0 into one
replicated [1, 1, max_seq, 576] scratch and runs causal flash attention for the chip's chunk/4 query rows.

Per chip (r, c): inputs attn_norm [1, 1, S/4, 1792] (column split, bf16) and q_resid [1, 1, S/4, 768] (row split,
replicated over axis 1, bf16); heads 16c .. 16c + 15 on chip column c. Geometry constants (block-cyclic RoPE tables,
the cache, the gather scratch, the SDPA program config) are built once per (chunk, max_seq) by ``setup``;
``__call__`` does no host work.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder, blockcyclic_positions
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

TILE = 32


# ring_mla q / k chunk sizes (tokens); XING_MLA_Q_CHUNK / XING_MLA_K_CHUNK override (L1 probing). Each is reduced
# to the largest multiple of 32 that divides the local chunk. ttMLA runs Kimi at q32 / k640 on a bfp8 cache;
# the bf16 cache doubles the K CB, so k256 here.
# P.1 (chunk 5120 after 51200, per layer, bringup ring_mla): HiFi4 q32 / k256 25.9 ms; HiFi3 q32 / k256 20.6 ms
# (q64 23.6); HiFi2 q64 / k256 17.9 ms (q32 18.7). q128 / k256, q256 / k128, q32 / k384 overflow L1 at any fidelity.
K_CHUNK = int(os.environ.get("XING_MLA_K_CHUNK", "256"))  # k512 overflows L1 with a bf16 cache (2.15 MB)


def sdpa_fidelity() -> str:
    """XING_MLA_SDPA_FIDELITY for the ring_mla matmuls only (owner exception to rule 7 for the SDPA, P.1; every
    other matmul stays HiFi4): ``HiFi2`` (default, the owner's pick) or ``HiFi4`` (the previous setting). HiFi2 fails
    the frozen C.moe.attention component test (L02 median row norm -0.0068 vs limit 0.004) but is accepted by the owner
    on end-to-end accuracy (rung last / s56320 within 1e-3 of HiFi4, top5 1.0; supervision.md P.1)."""
    f = os.environ.get("XING_MLA_SDPA_FIDELITY", "HiFi2")
    assert f in ("HiFi2", "HiFi4"), f"XING_MLA_SDPA_FIDELITY must be HiFi2 or HiFi4, got {f}"
    return f


def sdpa_q_chunk() -> int:
    return int(os.environ.get("XING_MLA_Q_CHUNK", "64" if sdpa_fidelity() == "HiFi2" else "32"))


def sdpa_impl() -> str:
    """XING_MLA_SDPA: ``fork`` (default) = ttnn.bringup.ring_mla at HiFi4 + fp32 DEST (the fork runs latent-V ring
    attention on the streaming path with fp32 accumulation); ``source`` = ttnn.transformer.ring_mla at HiFi4 + bf16
    DEST (the owner's 06:35 setting; its bf16 QK^T accumulation fails the component test's x2-input check)."""
    impl = os.environ.get("XING_MLA_SDPA", "fork")
    assert impl in ("fork", "source"), f"XING_MLA_SDPA must be fork or source, got {impl}"
    return impl


_KV_BUFS: dict = {}  # (id(mesh), max_seq, width) -> the shared ring_mla gather scratch
_SEMAPHORES: dict = {}  # id(mesh) -> (ring_mla global semaphores, ccl core offset, sdpa grid)


def _ring_ccl(mesh):
    """The ring_mla CCL resources, created once per mesh and shared by every layer (as ttMLA's TT_CCL)."""
    key = id(mesh)
    if key not in _SEMAPHORES:
        g = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
        sems = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)]
        _SEMAPHORES[key] = (sems, (g.x - 1, 0), (g.x - 1, g.y))
    return _SEMAPHORES[key]


def _largest_divisor_chunk(n: int, want: int) -> int:
    """The largest multiple of 32 <= want that divides n."""
    c = min(want, n) // TILE * TILE
    while c > TILE and n % c:
        c -= TILE
    return max(c, TILE)


class _Geometry:
    """Per (chunk, max_seq) device constants: block-cyclic RoPE tables, the bf16 latent cache, the gather scratch."""

    def __init__(self, att: "TtMlaAttention", chunk: int, max_seq: int):
        mesh, sp = att.mesh, att.sp
        assert chunk % (TILE * sp) == 0, f"chunk {chunk} must be a multiple of {TILE * sp}"
        assert max_seq % chunk == 0, f"max_seq {max_seq} must be a multiple of the chunk {chunk}"
        self.chunk, self.max_seq = chunk, max_seq
        self.chunk_local = chunk // sp  # query / key rows per SP rank
        dram = ttnn.DRAM_MEMORY_CONFIG
        # Interleaved (Meta pair) cos / sin [max_seq, 64]: entries 2i, 2i+1 hold cos / sin of pos * inv_freq[i],
        # times the YaRN attention factor (1.0 for Xing), angles in fp32 as the reference (rope_cos_sin).
        freqs = torch.arange(max_seq).float()[:, None] * att.inv_freq[None, :]
        cos = (freqs.cos() * att.rope_att).repeat_interleave(2, dim=-1)
        sin = (freqs.sin() * att.rope_att).repeat_interleave(2, dim=-1)
        shard_sp = ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, None))

        def table(t):
            t = block_cyclic_reorder(t.reshape(1, 1, max_seq, att.rope_dim), self.chunk_local, sp, seq_dim=2)
            return ttnn.from_torch(
                t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=dram, mesh_mapper=shard_sp
            )

        self.cos, self.sin = table(cos), table(sin)
        self.trans = ttnn.from_torch(
            get_rot_transformation_mat(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=dram,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # Latent cache [1, 1, max_seq / sp, 576] bf16 TILE per chip, block-cyclic over the SP rows, replicated over
        # the TP columns (tp_axis None: the dense ring_mla path reads a TP-replicated cache), zeroed on the device.
        self.cache = init_kvpe_cache(
            att.kv_width,
            mesh,
            max_seq,
            tuple(mesh.shape),
            att.sp_axis,
            1,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        self._topology = self.cache.tensor_topology()
        # ring_mla's gathered-KV scratch: replicated [1, 1, max_seq, 576], one per (mesh, max_seq) shared by every
        # layer (ring_mla rewrites the gathered prefix each call; layers run one after another). plan.md: 65 MB.
        key = (id(mesh), max_seq, att.kv_width)
        if key not in _KV_BUFS:
            _KV_BUFS[key] = ttnn.from_torch(
                torch.zeros(1, 1, max_seq, att.kv_width),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=dram,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
        self.kv_buf = _KV_BUFS[key]
        cl = self.chunk_local
        q_chunk, k_chunk = _largest_divisor_chunk(cl, sdpa_q_chunk()), _largest_divisor_chunk(cl, K_CHUNK)
        _, _, grid = _ring_ccl(mesh)
        self.sdpa_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            # XING_MLA_EXP_APPROX=1: the online-softmax correction exp uses the fp32-accurate exp (the flag's True);
            # default False (range-reduced polynomial) as before. The softmax exp itself is always the fast approx.
            exp_approx_mode=os.environ.get("XING_MLA_EXP_APPROX", "0") == "1",
        )

    # ---- harness boundary (state load / read-back); never called from __call__
    def load(self, mesh, prefix: torch.Tensor | None, sp: int, width: int) -> None:
        """Write natural-order latent rows [n, 576] (n <= max_seq, zeros past n) into the block-cyclic cache."""
        nat = torch.zeros(self.max_seq, width, dtype=torch.bfloat16)
        if prefix is not None and prefix.shape[0]:
            nat[: prefix.shape[0]] = prefix.to(torch.bfloat16)
        p = blockcyclic_positions(sp, self.chunk, self.max_seq)  # shard row -> natural position
        host = nat[p].reshape(1, 1, self.max_seq, width)  # row r's shard = rows [r max_seq/sp, (r+1) max_seq/sp)
        ht = ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, None)),
        )
        ttnn.copy_host_to_device_tensor(ht, self.cache)
        self.cache.update_tensor_topology(self._topology)

    def read(self, mesh, length: int, sp: int) -> torch.Tensor:
        """The cache in natural order [length, 576] fp32 (column 0's copy of each row)."""
        devs = ttnn.get_device_tensors(self.cache)  # row-major over the mesh
        cols = mesh.shape[1]
        flat = torch.cat([ttnn.to_torch(devs[r * cols]).float().reshape(-1, devs[0].shape[-1]) for r in range(sp)])
        p = blockcyclic_positions(sp, self.chunk, self.max_seq)
        nat = torch.empty_like(flat)
        nat[p] = flat
        return nat[:length]


class TtMlaAttention:
    """One layer's dense causal MLA. ``setup(chunk, max_seq)`` once per geometry, then
    ``__call__(attn_norm, q_resid, start)`` per chunk (start a multiple of the chunk). Writes the chunk's latent rows
    into the layer's device cache and returns attn_out [1, 1, S/4, hidden/2] fp32 (column split)."""

    def __init__(
        self,
        mesh,
        q_b: torch.Tensor,  # [H * (dn + r), q_lora]
        kv_a: torch.Tensor,  # [lat + r, hidden]
        kv_a_norm: torch.Tensor,  # [lat]
        kv_b: torch.Tensor,  # [H * (dn + dv), lat]
        o_proj: torch.Tensor,  # [hidden, H * dv]
        *,
        n_heads: int,
        nope_dim: int,
        rope_dim: int,
        v_dim: int,
        kv_lora_rank: int,
        inv_freq: torch.Tensor,  # [rope_dim / 2] YaRN
        rope_att: float,
        eps: float,
        scale: float,
        sp_axis: int = 0,
        tp_axis: int = 1,
    ):
        assert sp_axis == 0 and tp_axis == 1, "block-cyclic cache / ring_mla wiring assume sp_axis 0, tp_axis 1"
        self.mesh, self.sp_axis, self.tp_axis = mesh, sp_axis, tp_axis
        self.sp, self.tp = mesh.shape[sp_axis], mesh.shape[tp_axis]
        assert self.sp > 1, "ring_mla needs at least 2 devices on the SP axis"
        hq, dn, r, dv, lat = n_heads, nope_dim, rope_dim, v_dim, kv_lora_rank
        hidden = kv_a.shape[1]
        assert hq % self.tp == 0
        assert q_b.shape[0] == hq * (dn + r) and kv_a.shape[0] == lat + r and kv_b.shape == (hq * (dn + dv), lat)
        assert o_proj.shape == (hidden, hq * dv) and hidden % (TILE * self.tp) == 0
        self.n_heads, self.heads_local = hq, hq // self.tp
        self.nope_dim, self.rope_dim, self.v_dim, self.lat = dn, r, dv, lat
        self.kv_width = lat + r
        self.hidden = hidden
        self.inv_freq, self.rope_att = inv_freq.float(), float(rope_att)
        self.eps = float(eps)
        self.scale = float(torch.tensor(scale, dtype=torch.float32))
        self.num_links = 2 if mesh.arch() == ttnn.Arch.BLACKHOLE else 1  # as ttMLA.ccl_num_links
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # ring_mla at HiFi2 by default (owner exception for the SDPA only, XING_MLA_SDPA_FIDELITY=HiFi4 = previous);
        # fp32 dest (fork) as before. The source op runs latent V only on its streaming path, which it allows only at bf16
        # dest (owner 06:35: bf16 dest); the fork (XING_MLA_SDPA=fork, default) runs that path at fp32 dest.
        self.sdpa_impl = sdpa_impl()
        self.ring_mla = ttnn.bringup.ring_mla if self.sdpa_impl == "fork" else ttnn.transformer.ring_mla
        self.sdpa_ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, sdpa_fidelity()),
            math_approx_mode=False,
            fp32_dest_acc_en=self.sdpa_impl == "fork",
            packer_l1_acc=False,
        )
        dram = ttnn.DRAM_MEMORY_CONFIG

        def put(t, dtype, dims=None, layout=ttnn.TILE_LAYOUT):
            mapper = (
                ttnn.ReplicateTensorToMesh(mesh)
                if dims is None
                else ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims)
            )
            return ttnn.from_torch(
                t.contiguous(), dtype=dtype, layout=layout, device=mesh, memory_config=dram, mesh_mapper=mapper
            )

        q_lora = q_b.shape[1]
        # Head split: chip column c holds heads hq/tp * c .. (contiguous output columns of q_b^T, dim 1 of the
        # per-head kv_b tensors, contiguous K rows of o_proj^T), replicated over the SP rows.
        self.q_b = put(q_b.float().t().reshape(1, 1, q_lora, hq * (dn + r)), ttnn.bfloat16, (None, 3))
        kvb = kv_b.float().reshape(1, hq, dn + dv, lat)
        self.w_uk = put(kvb[:, :, :dn, :], ttnn.bfloat16, (None, 1))  # [1, H, 128, 512]: q_nope -> latent
        w_uv = kvb[:, :, dn:, :].transpose(-2, -1)  # [1, H, 512, 128]
        self.w_uv = put(w_uv, ttnn.bfloat16, (None, 1))
        self.o_proj = put(o_proj.float().t().reshape(1, 1, hq * dv, hidden), ttnn.bfloat16, (None, 2))
        # kv_a K-split: W^T [hidden, 576], rows split over the TP axis (matches attn_norm's column split).
        self.kv_a = put(kv_a.float().t().reshape(1, 1, hidden, lat + r), ttnn.bfloat16, (None, 2))
        self.kv_a_norm = put(
            kv_a_norm.float().reshape(1, 1, lat // TILE, TILE), ttnn.float32, None, ttnn.ROW_MAJOR_LAYOUT
        )
        self.sems, self.ccl_offset, _ = _ring_ccl(mesh)
        self._geoms: dict = {}
        self.geom: _Geometry | None = None
        self.slot = None  # serving binding (bind_cache); None = the geometry's own single-sequence cache

    # ---- load time / harness boundary
    def setup(self, chunk: int, max_seq: int) -> _Geometry:
        key = (chunk, max_seq)
        if key not in self._geoms:
            self._geoms[key] = _Geometry(self, chunk, max_seq)
        self.geom = self._geoms[key]
        return self.geom

    def release(self, max_seq: int) -> None:
        """Harness boundary: free every geometry built for ``max_seq`` (cache, RoPE tables, the shared gather
        scratch). The next setup() for that length rebuilds them."""
        for key in [k for k in self._geoms if k[1] == max_seq]:
            g = self._geoms.pop(key)
            if self.geom is g:
                self.geom = None
            for t in (g.cache, g.cos, g.sin, g.trans):
                ttnn.deallocate(t)
        kb = _KV_BUFS.pop((id(self.mesh), max_seq, self.kv_width), None)
        if kb is not None:
            ttnn.deallocate(kb)

    def bind_cache(self, cache, slot: int, row: int, rows: int) -> None:
        """Serving option: write / gather this layer's latent rows in an external multi-slot cache (the prefill
        engine's, tt/runners/kv_contract.py) instead of the geometry's own. ``cache`` is laid out like the geometry's
        (init_kvpe_cache, tp_axis None, same max_seq and chunk) with batch = slot * rows + row. ``unbind_cache``
        (the default) restores the geometry cache at batch 0."""
        assert tuple(cache.shape)[1:] == tuple(self.geom.cache.shape)[1:], (cache.shape, self.geom.cache.shape)
        assert 0 <= row < rows and (slot + 1) * rows <= cache.shape[0], (slot, row, rows, cache.shape)
        self.slot = (cache, int(slot), int(row), int(rows))

    def unbind_cache(self) -> None:
        self.slot = None

    def _cache(self):
        """(cache, slot_idx, layer_idx, num_layers) the chunk writes and gathers."""
        return self.slot if self.slot is not None else (self.geom.cache, 0, 0, 1)

    def load_state(self, prefix: torch.Tensor | None) -> None:
        """Harness boundary: natural-order kv_latent [n, 576] into the current geometry's cache (zeros past n)."""
        self.geom.load(self.mesh, prefix, self.sp, self.kv_width)

    def read_state(self, length: int) -> torch.Tensor:
        """Harness boundary: the latent cache [length, 576] in natural order."""
        return self.geom.read(self.mesh, length, self.sp)

    # ---- forward (device only)
    def _rope(self, x, start):
        return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            x,
            self.geom.cos,
            self.geom.sin,
            self.geom.trans,
            start,
            self.sp_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.ckc,
        )

    def _kv_stem(self, x, start: int) -> None:
        """kv = [kv_a_layernorm(latent) | RoPE(k_rope)] for this chunk -> the latent cache (this row's window)."""
        dram, s_loc = ttnn.DRAM_MEMORY_CONFIG, x.shape[2]
        part = ttnn.linear(x, self.kv_a, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=dram)
        kv = ttnn.all_reduce(part, cluster_axis=self.tp_axis, memory_config=dram)  # [1, 1, S/4, 576] fp32
        ttnn.deallocate(part)
        nope = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, s_loc, self.lat], memory_config=dram)
        rope = ttnn.slice(kv, [0, 0, 0, self.lat], [1, 1, s_loc, self.kv_width], memory_config=dram)
        ttnn.deallocate(kv)
        nn_ = ttnn.bringup.rms_norm(
            nope, weight=self.kv_a_norm, epsilon=self.eps, memory_config=dram, compute_kernel_config=self.ckc
        )
        ttnn.deallocate(nope)
        nb = ttnn.typecast(nn_, ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(nn_)
        rb = ttnn.typecast(rope, ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(rope)
        rr = self._rope(rb, start)
        ttnn.deallocate(rb)
        kvpe = ttnn.concat([nb, rr], dim=-1, memory_config=dram)  # [1, 1, S/4, 576] bf16 TILE
        ttnn.deallocate(nb)
        ttnn.deallocate(rr)
        cache, slot, row, rows = self._cache()
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache,
            kvpe,
            slot_idx=slot,
            layer_idx=row,
            num_layers=rows,
            kv_actual_global=start,
            cluster_axis=self.sp_axis,
        )
        ttnn.deallocate(kvpe)

    def _q_stem(self, qr, start: int):
        """Absorbed q [1, 16, S/4, 576] bf16: q_b_proj -> heads -> q_nope @ W_uk | RoPE(q_rope)."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        q = ttnn.linear(qr, self.q_b, dtype=ttnn.bfloat16, compute_kernel_config=self.ckc, memory_config=dram)
        q_nope, q_rope = ttnn.experimental.nlp_create_q_heads_split(
            q, num_heads=self.heads_local, split_head_dim=self.nope_dim, memory_config=dram
        )
        ttnn.deallocate(q)
        q_lat = ttnn.linear(q_nope, self.w_uk, dtype=ttnn.bfloat16, compute_kernel_config=self.ckc, memory_config=dram)
        ttnn.deallocate(q_nope)
        q_rr = self._rope(q_rope, start)
        ttnn.deallocate(q_rope)
        q_abs = ttnn.concat([q_lat, q_rr], dim=-1, memory_config=dram)
        ttnn.deallocate(q_lat)
        ttnn.deallocate(q_rr)
        return q_abs

    def __call__(self, x: ttnn.Tensor, qr: ttnn.Tensor, start: int) -> ttnn.Tensor:
        dram, g = ttnn.DRAM_MEMORY_CONFIG, self.geom
        assert g is not None, "call setup(chunk, max_seq) first"
        assert start % g.chunk == 0, f"start {start} must be chunk ({g.chunk}) aligned"
        self._kv_stem(x, start)
        q_abs = self._q_stem(qr, start)
        o = self._ring_attend(q_abs, start)
        ttnn.deallocate(q_abs)
        v = ttnn.linear(o, self.w_uv, dtype=ttnn.bfloat16, compute_kernel_config=self.ckc, memory_config=dram)
        ttnn.deallocate(o)  # [1, 16, S/4, 128]
        vc = ttnn.experimental.nlp_concat_heads(v, memory_config=dram)  # [1, 1, S/4, 16 * 128]
        ttnn.deallocate(v)
        # o_proj row-parallel (K = this chip's 16 heads x 128), fp32 partials, reduce-scatter to the column split.
        part = ttnn.linear(vc, self.o_proj, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=dram)
        ttnn.deallocate(vc)
        out = ttnn.reduce_scatter(part, dim=3, cluster_axis=self.tp_axis, memory_config=dram)
        ttnn.deallocate(part)
        return out  # [1, 1, S/4, hidden / tp] fp32

    def _ring_attend(self, q_abs, start: int):
        """ring_mla over axis 0 (ttMLA._chunked_attn): o [1, 16, S/4, 512]."""
        g = self.geom
        cache, slot, row, rows = self._cache()
        o, stats = self.ring_mla(
            q_abs,
            cache,
            persistent_output_buffer_kv=g.kv_buf,
            head_dim_v=self.lat,
            logical_n=min(start + g.chunk, g.max_seq),
            program_config=g.sdpa_pc,
            scale=self.scale,
            compute_kernel_config=self.sdpa_ckc,
            dim=2,
            multi_device_global_semaphore=self.sems,
            num_links=self.num_links,
            cluster_axis=self.sp_axis,
            mesh_device=self.mesh,
            topology=ttnn.Topology.Linear,
            ccl_core_grid_offset=self.ccl_offset,
            use_column_major_ccl=True,
            is_balanced=False,
            kv_cache_batch_idx=slot * rows + row,
            kv_actual_isl=start,
        )  # [1, 16, S/4, 512] bf16
        ttnn.deallocate(stats)
        return o


def build_attention(mesh, loader, cfg, layer: int) -> TtMlaAttention:
    from models.demos.xing40_a4b_d_p.reference.xing_ref import yarn_inv_freq

    p = f"model.layers.{layer}.self_attn."
    inv_freq, att = yarn_inv_freq(cfg)
    return TtMlaAttention(
        mesh,
        loader.get(p + "q_b_proj.weight").float(),
        loader.get(p + "kv_a_proj_with_mqa.weight").float(),
        loader.get(p + "kv_a_layernorm.weight").float(),
        loader.get(p + "kv_b_proj.weight").float(),
        loader.get(p + "o_proj.weight").float(),
        n_heads=cfg.num_attention_heads,
        nope_dim=cfg.qk_nope_head_dim,
        rope_dim=cfg.qk_rope_head_dim,
        v_dim=cfg.v_head_dim,
        kv_lora_rank=cfg.kv_lora_rank,
        inv_freq=inv_freq,
        rope_att=att,
        eps=cfg.rms_norm_eps,
        scale=cfg.attn_scale,
    )
