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
from models.demos.glm53_flash_d_p.tt.common import hifi4_config, replicate
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


class TtMLA:
    def __init__(self, mesh, w: dict, cfg, max_seq: int):
        self.mesh = mesh
        self.nh, self.dqk, self.dv, self.r = cfg.num_attention_heads, cfg.qk_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
        assert self.dqk == self.dv == 256 and self.r == 512, "GLM-5.3 NoPE MLA geometry"
        assert tuple(mesh.shape) == (2, 2), "query split assumes a 2x2 mesh (chip d = 2 r + c)"
        self.ndev = mesh.get_num_devices()
        self.scale = self.dqk**-0.5
        assert float(torch.tensor(self.scale, dtype=torch.float32)) == self.scale, "scale must be fp32-exact"
        self.mm = hifi4_config()
        self.mid = ttnn.float32 if MID_DTYPE == "fp32" else ttnn.bfloat16
        self.max_seq = max_seq
        up = lambda t: replicate(mesh, t.float().T.reshape(1, 1, t.shape[1], t.shape[0]).to(torch.bfloat16))  # noqa
        self.w_kva = up(w["kv_a"])  # [4096, 512]
        self.kv_norm = TtRMSNorm(mesh, w["kv_a_norm"], cfg.rms_norm_eps)
        self.w_qb = up(w["q_b"])  # [1536, 64 * 256]
        kv_b = w["kv_b"].float().view(self.nh, self.dqk + self.dv, self.r)
        self.w_uk = replicate(mesh, kv_b[:, : self.dqk].reshape(1, self.nh, self.dqk, self.r).to(torch.bfloat16))
        w_uv = kv_b[:, self.dqk :].transpose(1, 2).reshape(1, self.nh, self.r, self.dv)
        self.w_uv = replicate(mesh, w_uv.contiguous().to(torch.bfloat16))
        self.w_o = up(w["o_proj"])  # [64 * 256, 4096]
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

    def _write_latent(self, x: ttnn.Tensor, start: int) -> None:
        s = x.shape[-2]
        lat = ttnn.linear(x, self.w_kva, dtype=ttnn.float32, compute_kernel_config=self.mm, memory_config=MC)
        ln = self.kv_norm(lat)
        ttnn.deallocate(lat)
        lb = ttnn.typecast(ln, ttnn.bfloat16, memory_config=MC)
        ttnn.deallocate(ln)
        lrm = ttnn.to_layout(lb, ttnn.ROW_MAJOR_LAYOUT, memory_config=MC)
        ttnn.deallocate(lb)
        ttnn.experimental.slice_write(lrm, self.cache, [0, 0, start, 0], [1, 1, start + s, self.r], [1, 1, 1, 1])
        ttnn.deallocate(lrm)

    def __call__(self, x: ttnn.Tensor, q_resid: ttnn.Tensor, idx: ttnn.Tensor, start: int) -> ttnn.Tensor:
        """x (attn_norm) [1, 1, S, H], q_resid [1, 1, S, 1536], both replicated bf16 TILE; idx this chip's
        [1, 1, S/4, 2176] uint32 ROW_MAJOR token ids (the indexer's output). Returns [1, 1, S, H] replicated bf16.
        Writes the chunk's latent rows into the cache."""
        s = x.shape[-2]
        assert start + s <= self.max_seq, f"chunk end {start + s} past max_seq {self.max_seq}"
        self._write_latent(x, start)

        qr = self._local_rows(q_resid)
        q = ttnn.linear(qr, self.w_qb, dtype=self.mid, compute_kernel_config=self.mm, memory_config=MC)
        ttnn.deallocate(qr)
        qh, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=self.nh, num_kv_heads=0, transpose_k_heads=False, memory_config=MC
        )
        ttnn.deallocate(q)
        ql = ttnn.matmul(qh, self.w_uk, dtype=ttnn.bfloat16, compute_kernel_config=self.mm, memory_config=MC)
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
        oh = ttnn.matmul(ot, self.w_uv, dtype=self.mid, compute_kernel_config=self.mm, memory_config=MC)
        ttnn.deallocate(ot)
        oc = self._concat_heads(oh)  # [1, 1, S/4, 64 * 256]
        ttnn.deallocate(oh)
        y = ttnn.linear(oc, self.w_o, dtype=self.mid, compute_kernel_config=self.mm, memory_config=MC)
        ttnn.deallocate(oc)
        if y.dtype != ttnn.bfloat16:
            yb = ttnn.typecast(y, ttnn.bfloat16, memory_config=MC)
            ttnn.deallocate(y)
            y = yb
        g1 = ttnn.all_gather(y, dim=-2, cluster_axis=1, memory_config=MC)
        ttnn.deallocate(y)
        g2 = ttnn.all_gather(g1, dim=-2, cluster_axis=0, memory_config=MC)
        ttnn.deallocate(g1)
        return g2

    def _concat_heads(self, oh: ttnn.Tensor) -> ttnn.Tensor:
        """[1, 64, M, 256] -> [1, 1, M, 64 * 256]. nlp_concat_heads holds a whole row of heads per core, which does not
        fit L1 at 64 x 256 (2.2 MB), so it runs on groups of CONCAT_GROUP heads joined on the last dim."""
        m = oh.shape[-2]
        parts = []
        grp = CONCAT_GROUP if oh.dtype == ttnn.bfloat16 else CONCAT_GROUP // 2  # the op's CBs scale with the dtype
        for h0 in range(0, self.nh, grp):
            sl = ttnn.slice(oh, (0, h0, 0, 0), (1, h0 + grp, m, self.dv), memory_config=MC)
            parts.append(ttnn.experimental.nlp_concat_heads(sl, memory_config=MC))
            ttnn.deallocate(sl)
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
        out.reshape(2, 2, s // nd, IDX_W).contiguous(),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=MC,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape)),
    )
    return ttnn.bitcast(d, ttnn.uint32)


def build_mla(mesh, loader, cfg, layer: int, max_seq: int) -> TtMLA:
    p = f"{PREFIX}layers.{layer}.self_attn."
    w = {
        "kv_a": loader.weight(p + "kv_a_proj_with_mqa.weight"),
        "kv_a_norm": loader.weight(p + "kv_a_layernorm.weight"),
        "q_b": loader.weight(p + "q_b_proj.weight"),
        "kv_b": loader.weight(p + "kv_b_proj.weight"),
        "o_proj": loader.weight(p + "o_proj.weight"),
    }
    return TtMLA(mesh, w, cfg, max_seq)
