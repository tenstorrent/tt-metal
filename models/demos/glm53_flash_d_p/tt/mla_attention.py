# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 DSA sparse MLA (dsa_moe attention step) on the 2x2 mesh: NoPE, absorbed kv_b, latent 512.

Latent (all S rows, every chip): kv_a_proj_with_mqa [4096 -> 512] (fp32 out) -> kv_a_layernorm (TtRMSNorm, eps 1e-5)
-> bf16 -> ROW_MAJOR -> ttnn.experimental.slice_write into the replicated latent cache [1, 1, max_seq, 512] bf16
ROW_MAJOR at row start. Rows past start + S are never selected (the indices are causal).
Queries (chip d = 2 r + c takes rows d S/4 .., two ttnn.mesh_partition, as the indexer): q_b [1536 -> 64 x 256] ->
heads [1, 64, S/4, 256] -> matmul w_uk [64, 256, 512] -> q_lat [1, 64, S/4, 512] -> ROW_MAJOR ->
ttnn.bringup.sparse_sdpa(high_precision=True) (kv = the latent cache, V = its 512 columns, indices [1, 1, S/4, 2176]
uint32 with a 0xFFFFFFFF sentinel tail, scale 256^-0.5 = 1/16 explicitly: the op's default is K_DIM^-0.5) -> TILE ->
matmul w_uv [64, 512, 256] -> concat heads [S/4, 16384] -> o_proj [16384 -> 4096] -> all_gather (dim -2, axis 1) +
all_gather (dim -2, axis 0) -> [1, 1, S, 4096] replicated bf16. Every matmul HiFi4 + fp32 acc; weights bf16 (fp8
dequantized). high_precision keeps sparse_sdpa's running output / row-sum in Float32 and takes the exact exp; the
source op keeps them bf16 with a fast approximate exp (rel 0.0066 vs 0.0030 on its own output).
Reuse: models/demos/deepseek_v3_d_p/tt/mla/mla.py:ttMLA (absorbed kv_b, sparse_sdpa call) without RoPE, the 576-wide
latent and the head-to-sequence reshard (all 64 heads stay on every chip; the sequence is split instead).
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.glm53_flash_d_p.reference.weights import PREFIX
from models.demos.glm53_flash_d_p.tt.common import attn_fidelity, hifi4_config, mm_config, replicate
from models.demos.glm53_flash_d_p.tt.mm_configs import bmm, linear_config, minimal_config
from models.demos.glm53_flash_d_p.tt.rms_norm import TtRMSNorm

MC = ttnn.DRAM_MEMORY_CONFIG
IDX_W = 2176  # index row width sparse_sdpa takes (2051 padded to a multiple of 128)
K_CHUNK = 128
CONCAT_GROUP = 32  # heads per nlp_concat_heads call
# "fork": ttnn.bringup.sparse_sdpa(high_precision=True) (fp32 running state, exact exp); "source":
# ttnn.transformer.sparse_sdpa (bf16 running state, approximate exp: per-token norm ratio fails at 0.993)
SDPA_MODE = os.environ.get("GLM_MLA_SDPA", "fork")
# dtype of q, the per-head outputs and the o_proj output. bf16 (rounded) is unbiased; fp32 inputs to the next matmul are
# read at TF32 precision, which shrinks the output by about 0.05% ("fp32" for comparison)
MID_DTYPE = os.environ.get("GLM_MLA_MID", "bf16")
# explicit configs for the per-head absorb matmuls (mm_configs.bmm: one output block per core, the only correct
# multi-core-reuse batched case); o w_uv 0.899 -> 0.276 ms, q w_uk stays auto. Module attributes so
# tests/test_ab_layers.py can flip them between runs
USE_BMM = os.environ.get("GLM_MLA_BMM", "1") == "1"
CHECK_BMM = False  # debug (tests/test_ab_layers.py): also run the auto config on the same input and print the diff


def _check(tag, out, a, w, dtype, ckc, pc=None):
    ref = ttnn.matmul(a, w, dtype=dtype, compute_kernel_config=ckc, memory_config=MC)
    x = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).double()
    y = ttnn.to_torch(ttnn.get_device_tensors(ref)[0]).double()
    ttnn.deallocate(ref)
    rel = float((x - y).norm() / y.norm())
    dump = os.environ.get("GLM_MLA_CHECK_DUMP")
    if dump:  # chip 0's live inputs and both outputs, for the single-chip replay (tests/test_mla_bmm_repro.py)
        torch.save(
            {
                "a": ttnn.to_torch(ttnn.get_device_tensors(a)[0]),
                "w": ttnn.to_torch(ttnn.get_device_tensors(w)[0]),
                "cfg_out": x.float(),
                "auto_out": y.float(),
            },
            f"{dump}/{tag.replace(' ', '_')}.pt",
        )
    print(
        f"[mla-check] {tag} {tuple(a.shape)} padded {tuple(a.padded_shape)} {a.dtype} {a.memory_config().memory_layout}"
        f" x {tuple(w.shape)} {w.dtype}: config vs auto rel {rel:.3e}; cfg {pc}",
        flush=True,
    )


# head-parallel projections (GLM non-flash's MLA layout; split layout + own-row keys only): q_b / w_uk / w_uv / o_proj
# sharded by heads over mesh axis 1 (TP = 4: 16 heads per chip, 1/4 of their DRAM), computed on the mesh row's S/2
# rows; sparse_sdpa stays fully sequence-parallel (own S/8 rows, all heads) via a head <-> row all-to-all on axis 1
# around it (all_to_all_async_generic: in_dim = the gathered dim, out_dim = the split dim; tests/test_mla_tp_a2a.py);
# o_proj's head partial sums reduce-scattered over axis 1 to the own rows (MiMo fabric_reduce_scatter, bf16)
TP_HEADS = os.environ.get("GLM_MLA_TP", "1") == "1"
TP_LINKS = int(os.environ.get("GLM_MOE_LINKS", "2"))
# dtype of the 2D projection weights kv_a, q_b, o_proj (bf16 | bfp8; MiMo runs its projections on bfp8 weights)
W_DTYPE = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("GLM_MLA_WDTYPE", "bf16")]


class TtMLA:
    def __init__(self, mesh, w: dict, cfg, max_seq: int, tp: bool = False):
        self.mesh = mesh
        self.tp = tp and TP_HEADS and tuple(mesh.shape)[1] > 1
        self.nh, self.dqk, self.dv, self.r = cfg.num_attention_heads, cfg.qk_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
        assert self.dqk == self.dv == 256 and self.r == 512, "GLM-5.3 NoPE MLA geometry"
        self.ndev = mesh.get_num_devices()
        self.scale = self.dqk**-0.5
        assert float(torch.tensor(self.scale, dtype=torch.float32)) == self.scale, "scale must be fp32-exact"
        self.mm = hifi4_config(fidelity=attn_fidelity())
        self.mid = ttnn.float32 if MID_DTYPE == "fp32" else ttnn.bfloat16
        self.max_seq = max_seq
        up = lambda t: replicate(  # noqa: E731
            mesh, t.float().T.reshape(1, 1, t.shape[1], t.shape[0]).to(torch.bfloat16), dtype=W_DTYPE
        )
        self.w_kva = up(w["kv_a"])  # [4096, 512]
        self.kv_norm = TtRMSNorm(mesh, w["kv_a_norm"], cfg.rms_norm_eps)
        kv_b = w["kv_b"].float().view(self.nh, self.dqk + self.dv, self.r)
        w_uk = kv_b[:, : self.dqk].reshape(1, self.nh, self.dqk, self.r)
        w_uv = kv_b[:, self.dqk :].transpose(1, 2).reshape(1, self.nh, self.r, self.dv).contiguous()
        if self.tp:
            self.tpn = tuple(mesh.shape)[1]
            self.nh_local = self.nh // self.tpn

            def heads(t, dim, dtype=W_DTYPE):  # shard dim over mesh axis 1 (heads), replicated over axis 0
                return ttnn.from_torch(
                    t.contiguous().to(torch.bfloat16),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=MC,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, dim)),
                )

            q_b, o_proj = w["q_b"].float(), w["o_proj"].float()
            self.w_qb = heads(q_b.T.reshape(1, 1, q_b.shape[1], q_b.shape[0]), 3)  # [1536, 16 * 256] per chip
            self.w_uk = heads(w_uk, 1, ttnn.bfloat16)  # [16, 256, 512]
            self.w_uv = heads(w_uv, 1, ttnn.bfloat16)  # [16, 512, 256]
            self.w_o = heads(o_proj.T.reshape(1, 1, o_proj.shape[1], o_proj.shape[0]), 2)  # [16 * 256, 4096]
        else:
            self.w_qb = up(w["q_b"])  # [1536, 64 * 256]
            self.w_uk = replicate(mesh, w_uk.to(torch.bfloat16))
            self.w_uv = replicate(mesh, w_uv.to(torch.bfloat16))
            self.w_o = up(w["o_proj"])  # [64 * 256, 4096]
        self.qb_cfg = minimal_config(mesh, 2, 8, 8, 1, 4)
        self.cache = latent_cache(mesh, cfg, max_seq)

    def bind_cache(self, cache: ttnn.Tensor) -> None:
        """Read and write another latent cache of the same shape (latent_cache; one per serving slot)."""
        assert tuple(cache.shape) == tuple(self.cache.shape), (cache.shape, self.cache.shape)
        self.cache = cache

    def _local_rows(self, t: ttnn.Tensor) -> ttnn.Tensor:
        a = ttnn.mesh_partition(t, dim=-2, cluster_axis=0, memory_config=MC)
        b = ttnn.mesh_partition(a, dim=-2, cluster_axis=1, memory_config=MC)
        ttnn.deallocate(a)
        return b

    def _write_latent(self, x: ttnn.Tensor, start: int, x_local: bool = False) -> None:
        """x_local: x is this chip's S/n rows; the latent is computed on them and all-gathered (split layout order)."""
        s = x.shape[-2] * (self.ndev if x_local else 1)
        lat = ttnn.linear(
            x,
            self.w_kva,
            dtype=ttnn.float32,
            program_config=linear_config(x, self.w_kva, ttnn.float32),
            compute_kernel_config=mm_config(self.mm),
            memory_config=MC,
        )
        ln = self.kv_norm(lat)
        ttnn.deallocate(lat)
        lb = ttnn.typecast(ln, ttnn.bfloat16, memory_config=MC)
        ttnn.deallocate(ln)
        if x_local:
            from models.demos.glm53_flash_d_p.tt.common import gather_rows

            own = lb
            lb = gather_rows(own)
            ttnn.deallocate(own)
        lrm = ttnn.to_layout(lb, ttnn.ROW_MAJOR_LAYOUT, memory_config=MC)
        ttnn.deallocate(lb)
        ttnn.experimental.slice_write(lrm, self.cache, [0, 0, start, 0], [1, 1, start + s, self.r], [1, 1, 1, 1])
        ttnn.deallocate(lrm)

    def __call__(
        self,
        x: ttnn.Tensor,
        q_resid: ttnn.Tensor,
        idx: ttnn.Tensor,
        start: int,
        split: bool = False,
        x_local: bool = False,
    ) -> ttnn.Tensor:
        """x (attn_norm) [1, 1, S, H], q_resid [1, 1, S, 1536], both replicated bf16 TILE; idx this chip's
        [1, 1, S/4, 2176] uint32 ROW_MAJOR token ids (the indexer's output). Returns [1, 1, S, H] replicated bf16.
        split: q_resid and the output are this chip's S/4 rows (the split residual layout; no output gather).
        Writes the chunk's latent rows into the cache."""
        s = x.shape[-2] * (self.ndev if x_local else 1)
        assert start + s <= self.max_seq, f"chunk end {start + s} past max_seq {self.max_seq}"
        self._write_latent(x, start, x_local)
        if self.tp:
            assert split and x_local, "head-parallel MLA needs the split layout with own-row keys"
            return self._call_tp(q_resid, idx)

        qr = q_resid if split else self._local_rows(q_resid)
        q = ttnn.experimental.minimal_matmul(  # 0.271 -> 0.228 ms (tests/test_matmul_tune.py), same error
            qr,
            self.w_qb,
            config=self.qb_cfg,
            dtype=self.mid,
            compute_kernel_config=mm_config(self.mm),
            memory_config=MC,
        )
        if qr is not q_resid:
            ttnn.deallocate(qr)
        qh, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=self.nh, num_kv_heads=0, transpose_k_heads=False, memory_config=MC
        )
        ttnn.deallocate(q)
        ql = ttnn.matmul(
            qh,
            self.w_uk,
            dtype=ttnn.bfloat16,
            program_config=bmm(qh.device(), self.nh, qh.shape[-2], self.dqk, self.r) if USE_BMM else None,
            compute_kernel_config=mm_config(self.mm),
            memory_config=MC,
        )
        if CHECK_BMM:
            _check(
                "q w_uk",
                ql,
                qh,
                self.w_uk,
                ttnn.bfloat16,
                mm_config(self.mm),
                bmm(qh.device(), self.nh, qh.shape[-2], self.dqk, self.r),
            )
        ttnn.deallocate(qh)
        qrm = ttnn.to_layout(ql, ttnn.ROW_MAJOR_LAYOUT, memory_config=MC)
        ttnn.deallocate(ql)
        if SDPA_MODE == "source":
            o = ttnn.transformer.sparse_sdpa(
                qrm,
                self.cache,
                idx,
                self.r,
                kv_format=ttnn.transformer.SparseKVFormat.BF16,
                scale=self.scale,
                k_chunk_size=K_CHUNK,
                compute_kernel_config=self.mm,
            )
        else:
            o = ttnn.bringup.sparse_sdpa(
                qrm,
                self.cache,
                idx,
                self.r,
                kv_format=ttnn.bringup.SparseKVFormat.BF16,
                scale=self.scale,
                k_chunk_size=K_CHUNK,
                compute_kernel_config=self.mm,
                high_precision=True,
            )
        ttnn.deallocate(qrm)
        ot = ttnn.to_layout(o, ttnn.TILE_LAYOUT, memory_config=MC)
        ttnn.deallocate(o)
        oh = ttnn.matmul(
            ot,
            self.w_uv,
            dtype=self.mid,
            program_config=bmm(ot.device(), self.nh, ot.shape[-2], self.r, self.dv) if USE_BMM else None,
            compute_kernel_config=mm_config(self.mm),
            memory_config=MC,
        )
        if CHECK_BMM:
            _check(
                "o w_uv",
                oh,
                ot,
                self.w_uv,
                self.mid,
                mm_config(self.mm),
                bmm(ot.device(), self.nh, ot.shape[-2], self.r, self.dv),
            )
        ttnn.deallocate(ot)
        oc = self._concat_heads(oh)  # [1, 1, S/4, 64 * 256]
        ttnn.deallocate(oh)
        y = ttnn.linear(
            oc,
            self.w_o,
            dtype=self.mid,
            program_config=linear_config(oc, self.w_o, self.mid),
            compute_kernel_config=mm_config(self.mm),
            memory_config=MC,
        )
        ttnn.deallocate(oc)
        if y.dtype != ttnn.bfloat16:
            yb = ttnn.typecast(y, ttnn.bfloat16, memory_config=MC)
            ttnn.deallocate(y)
            y = yb
        if split:
            return y
        g1 = ttnn.all_gather(y, dim=-2, cluster_axis=1, memory_config=MC)
        ttnn.deallocate(y)
        g2 = ttnn.all_gather(g1, dim=-2, cluster_axis=0, memory_config=MC)
        ttnn.deallocate(g1)
        return g2

    def _sdpa(self, ql: ttnn.Tensor, idx: ttnn.Tensor) -> ttnn.Tensor:
        """Absorbed q [1, 64, S/8, 512] TILE -> sparse attention over the latent cache -> [1, 64, S/8, 512] TILE."""
        qrm = ttnn.to_layout(ql, ttnn.ROW_MAJOR_LAYOUT, memory_config=MC)
        if SDPA_MODE == "source":
            o = ttnn.transformer.sparse_sdpa(
                qrm,
                self.cache,
                idx,
                self.r,
                kv_format=ttnn.transformer.SparseKVFormat.BF16,
                scale=self.scale,
                k_chunk_size=K_CHUNK,
                compute_kernel_config=self.mm,
            )
        else:
            o = ttnn.bringup.sparse_sdpa(
                qrm,
                self.cache,
                idx,
                self.r,
                kv_format=ttnn.bringup.SparseKVFormat.BF16,
                scale=self.scale,
                k_chunk_size=K_CHUNK,
                compute_kernel_config=self.mm,
                high_precision=True,
            )
        ttnn.deallocate(qrm)
        ot = ttnn.to_layout(o, ttnn.TILE_LAYOUT, memory_config=MC)
        ttnn.deallocate(o)
        return ot

    def _call_tp(self, q_resid: ttnn.Tensor, idx: ttnn.Tensor) -> ttnn.Tensor:
        """Head-parallel projections, sequence-parallel attention. q_resid / idx / the output: this chip's S/8 rows."""
        from models.demos.glm53_flash_d_p.tt.common import gather_half

        qr = gather_half(q_resid)  # [1, 1, S/2, 1536]: the mesh row's rows (axis-1 all-gather)
        q = ttnn.linear(
            qr,
            self.w_qb,
            dtype=self.mid,
            program_config=linear_config(qr, self.w_qb, self.mid),
            compute_kernel_config=mm_config(self.mm),
            memory_config=MC,
        )  # [1, 1, S/2, 16 * 256]
        ttnn.deallocate(qr)
        qh, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=self.nh_local, num_kv_heads=0, transpose_k_heads=False, memory_config=MC
        )
        ttnn.deallocate(q)
        ql = ttnn.matmul(
            qh,
            self.w_uk,
            dtype=ttnn.bfloat16,
            program_config=bmm(qh.device(), self.nh_local, qh.shape[-2], self.dqk, self.r) if USE_BMM else None,
            compute_kernel_config=mm_config(self.mm),
            memory_config=MC,
        )  # [1, 16, S/2, 512]
        ttnn.deallocate(qh)
        qs = ttnn.experimental.all_to_all_async_generic(  # heads -> rows: [1, 64, S/8, 512], own rows, all heads
            ql, in_dim=1, out_dim=2, num_links=TP_LINKS, memory_config=MC, cluster_axis=1
        )
        ttnn.deallocate(ql)
        ot = self._sdpa(qs, idx)
        ttnn.deallocate(qs)
        oh_ = ttnn.experimental.all_to_all_async_generic(  # rows -> heads: [1, 16, S/2, 512]
            ot, in_dim=2, out_dim=1, num_links=TP_LINKS, memory_config=MC, cluster_axis=1
        )
        ttnn.deallocate(ot)
        oh = ttnn.matmul(
            oh_,
            self.w_uv,
            dtype=self.mid,
            program_config=bmm(oh_.device(), self.nh_local, oh_.shape[-2], self.r, self.dv) if USE_BMM else None,
            compute_kernel_config=mm_config(self.mm),
            memory_config=MC,
        )  # [1, 16, S/2, 256]
        ttnn.deallocate(oh_)
        oc = self._concat_heads(oh)  # [1, 1, S/2, 16 * 256]
        ttnn.deallocate(oh)
        y = ttnn.linear(
            oc,
            self.w_o,
            dtype=ttnn.bfloat16,
            program_config=linear_config(oc, self.w_o, ttnn.bfloat16),
            compute_kernel_config=mm_config(self.mm),
            memory_config=MC,
        )  # this chip's heads' partial sum, [1, 1, S/2, 4096]
        ttnn.deallocate(oc)
        out = ttnn.bringup.fabric_reduce_scatter(y, cluster_axis=1, num_links=TP_LINKS)  # [1, 1, S/8, 4096] own rows
        ttnn.deallocate(y)
        return out

    def _concat_heads(self, oh: ttnn.Tensor) -> ttnn.Tensor:
        """[1, 64, M, 256] -> [1, 1, M, 64 * 256]. nlp_concat_heads holds a whole row of heads per core, which does not
        fit L1 at 64 x 256 (2.2 MB), so it runs on groups of CONCAT_GROUP heads joined on the last dim."""
        m = oh.shape[-2]
        parts = []
        grp = CONCAT_GROUP if oh.dtype == ttnn.bfloat16 else CONCAT_GROUP // 2  # the op's CBs scale with the dtype
        grp = min(grp, oh.shape[1])
        for h0 in range(0, oh.shape[1], grp):
            sl = ttnn.slice(oh, (0, h0, 0, 0), (1, h0 + grp, m, self.dv), memory_config=MC)
            parts.append(ttnn.experimental.nlp_concat_heads(sl, memory_config=MC))
            ttnn.deallocate(sl)
        if len(parts) == 1:  # concat of one tensor returns it: do not free it
            return parts[0]
        out = ttnn.concat(parts, dim=-1, memory_config=MC)
        for t in parts:
            ttnn.deallocate(t)
        return out

    # ---- state at the harness boundary (prefix load / read-back; never inside the forward)
    def load_state(self, tensors: dict, length: int | None = None) -> None:
        """Latent rows [n, 512] (reference layout, rows [0, length) valid) -> the device cache."""
        lat = tensors["kv_latent"].float()
        n = lat.shape[0] if length is None else length
        host = torch.zeros(1, 1, self.max_seq, self.r)
        host[0, 0, :n] = lat[:n]
        d = replicate(self.mesh, host.to(torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy(d, self.cache)
        ttnn.deallocate(d)

    def state_torch(self) -> dict:
        """The latent cache [max_seq, 512] (chip 0's copy; replicated)."""
        t = ttnn.to_torch(ttnn.get_device_tensors(self.cache)[0])
        return {"kv_latent": t.reshape(self.max_seq, self.r).float()}


def latent_cache(mesh, cfg, max_seq: int) -> ttnn.Tensor:
    """The replicated latent cache [1, 1, max_seq, 512] bf16 ROW_MAJOR, zeroed."""
    return ttnn.zeros(
        (1, 1, max_seq, cfg.kv_lora_rank),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=MC,
    )


def idx_to_device(mesh, topk: torch.Tensor) -> ttnn.Tensor:
    """Harness boundary: reference topk int32 [S, W] (-1 = none, anywhere in the row) -> per-chip [1, 1, S/4, 2176]
    uint32 ROW_MAJOR, valid ids first and 0xFFFFFFFF as a contiguous tail (the indexer module's device format)."""
    s, w = topk.shape
    t = topk.to(torch.int32)
    order = torch.argsort((t < 0).to(torch.int8), dim=-1, stable=True)
    t = torch.gather(t, -1, order)
    out = torch.full((s, IDX_W), -1, dtype=torch.int32)
    out[:, :w] = t
    nd = mesh.get_num_devices()
    d = ttnn.from_torch(
        out.reshape(*tuple(mesh.shape), s // nd, IDX_W).contiguous(),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=MC,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape)),
    )
    return ttnn.bitcast(d, ttnn.uint32)


def build_mla(mesh, loader, cfg, layer: int, max_seq: int, tp: bool = False) -> TtMLA:
    p = f"{PREFIX}layers.{layer}.self_attn."
    w = {
        "kv_a": loader.weight(p + "kv_a_proj_with_mqa.weight"),
        "kv_a_norm": loader.weight(p + "kv_a_layernorm.weight"),
        "q_b": loader.weight(p + "q_b_proj.weight"),
        "kv_b": loader.weight(p + "kv_b_proj.weight"),
        "o_proj": loader.weight(p + "o_proj.weight"),
    }
    return TtMLA(mesh, w, cfg, max_seq, tp=tp)
