# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``DsaDecodeState``: the fused DSA (DeepSeek sparse attention) decode state maintenance as a model-local
``ttnn.generic_op`` (kernels in ``kernels/``; contract in README.md).

From the decode stem matmul output it computes ``q_resid = rmsnorm(stem[:, qa]) * w`` (one core per 32-user
tile row) and, one core per user b at position a: the RMSNorm'd kv_a row into row ``a % 32`` of its latent page,
the LayerNorm'd indexer k row and the raw gate row into ring slot ``a % P`` of its rings, and the pooled key of
pool ``a // P`` (softmax-weighted mean over the ring) into the pool cache -- all four caches read-modify-written in
place.
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


def f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


class DsaDecodeState:
    CB = dict(
        qa=0,
        wq=1,
        kva=2,
        kblk=3,
        gblk=4,
        wkva=5,
        wk=6,
        bk=7,
        ape=8,
        sel=9,
        lat_old=10,
        ring_old=11,
        pool_old=12,
        selrow=13,
        qout=14,
        lat_new=15,
        ring_new=16,
        pool_new=17,
        tmp=18,
        inv=19,
        normed=20,
        knorm=21,
        rows=22,
        selT=23,
        keep=24,
        ringf=25,
        pscr=26,
        prow=27,
        run=28,
        meta_w=29,
        idx=30,
    )

    def __init__(
        self,
        *,
        q_lora_rank: int,
        latent_dim: int,
        index_dim: int,
        pool_size: int,
        pool_blocks: int,
        qa_offset: int,
        kva_offset: int,
        k_offset: int,
        g_offset: int,
        eps_rms: float = 1e-5,
        eps_ln: float = 1e-6,
        math_fidelity=ttnn.MathFidelity.HiFi4,
    ):
        self.Lq, self.L, self.D, self.P = q_lora_rank, latent_dim, index_dim, pool_size
        self.pool_blocks = pool_blocks
        self.off = (qa_offset, kva_offset, k_offset, g_offset)
        self.eps_rms, self.eps_ln = eps_rms, eps_ln
        self.fidelity = math_fidelity
        assert (
            self.D == 128 and self.Lq % 128 == 0 and self.L % 128 == 0
        ), "the kernels want D = 128, Lq/L multiples of 128"
        assert all(o % TILE == 0 for o in self.off)

    @staticmethod
    def golden(
        stem,
        latent,
        pt,
        positions,
        ring_k,
        ring_g,
        pool,
        wq,
        wkva,
        wk,
        bk,
        ape,
        *,
        Lq,
        L,
        D,
        P,
        pool_blocks,
        offsets,
        eps_rms=1e-5,
        eps_ln=1e-6,
    ):
        """One step in torch (bf16-rounded like the device). stem [B, Ws]; latent [n_blocks, 1, 32, L]; pt [B, bps]
        int; positions [B] int; rings [B, 1, 32, D]; pool [B*pool_blocks, 1, 32, D]; weights [Lq]/[L]/[D]/[D]; ape
        [P, D]; offsets = (qa, kva, k, g). Returns (q_resid [B, Lq], latent, ring_k, ring_g, pool)."""
        B = stem.shape[0]
        qa_o, kva_o, k_o, g_o = offsets
        bf = lambda t: t.to(torch.bfloat16).float()
        qa = stem[:, qa_o : qa_o + Lq]
        q_resid = bf(qa * torch.rsqrt((qa * qa).mean(-1, keepdim=True) + eps_rms) * wq)
        kva = stem[:, kva_o : kva_o + L]
        lat_row = bf(kva * torch.rsqrt((kva * kva).mean(-1, keepdim=True) + eps_rms) * wkva)
        k = stem[:, k_o : k_o + D]
        mu, var = k.mean(-1, keepdim=True), k.var(-1, keepdim=True, unbiased=False)
        k_row = bf((k - mu) * torch.rsqrt(var + eps_ln) * wk + bk)
        g_row = stem[:, g_o : g_o + D]
        latent, ring_k, ring_g, pool = (t.clone() for t in (latent, ring_k, ring_g, pool))
        for b in range(B):
            a = int(positions[b])
            page = int(pt[b, a // 32])
            latent[page, 0, a % 32] = lat_row[b]
            ring_k[b, 0, a % P] = k_row[b]
            ring_g[b, 0, a % P] = g_row[b]
            prob = torch.softmax(ring_g[b, 0, :P] + ape, dim=0)
            pooled = bf((prob * ring_k[b, 0, :P]).sum(0))
            p = a // P
            pool[b * pool_blocks + p // 32, 0, p % 32] = pooled
        return q_resid, latent, ring_k, ring_g, pool

    def _cb(self, idx, dtype, page, pages, cores):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page)],
        )

    def __call__(
        self,
        stem,
        latent_cache,
        latent_page_table,
        positions,
        ring_k,
        ring_g,
        pool_cache,
        qa_norm_w,
        kva_norm_w,
        k_norm_w,
        k_norm_b,
        ape,
        selectors,
        q_resid,
    ):
        """See README.md; ``q_resid`` [1,1,B,Lq] bf16 TILE is the pre-allocated output; the four caches are
        updated in place. Returns ``q_resid``."""
        Lq, L, D = self.Lq, self.L, self.D
        B = stem.shape[-2]
        Bt = -(-B // TILE)
        Lq_t, L_t, D_t = Lq // TILE, L // TILE, D // TILE
        Ws_t = stem.padded_shape[-1] // TILE
        device = stem.device()
        grid = device.compute_with_storage_grid_size()
        NC = Bt + B
        assert NC <= grid.x * grid.y, f"dsa_decode_state needs {NC} cores, the grid has {grid.x * grid.y}"
        core_of = [ttnn.CoreCoord(c // grid.y, c % grid.y) for c in range(NC)]
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in core_of])
        f32, f16, u32 = ttnn.float32, ttnn.bfloat16, ttnn.uint32
        t32, t16 = TILE * TILE * 4, TILE * TILE * 2
        cb = self.CB
        cbs = [
            self._cb(cb["qa"], f16, t16, Lq_t, cores),
            self._cb(cb["wq"], f32, t32, 8, cores),
            self._cb(cb["kva"], f16, t16, L_t, cores),
            self._cb(cb["kblk"], f16, t16, D_t, cores),
            self._cb(cb["gblk"], f16, t16, D_t, cores),
            self._cb(cb["wkva"], f32, t32, L_t, cores),
            self._cb(cb["wk"], f32, t32, D_t, cores),
            self._cb(cb["bk"], f32, t32, D_t, cores),
            self._cb(cb["ape"], f32, t32, D_t, cores),
            self._cb(cb["sel"], f16, t16, 33, cores),
            self._cb(cb["lat_old"], f16, t16, L_t, cores),
            self._cb(cb["ring_old"], f16, t16, 2 * D_t, cores),
            self._cb(cb["pool_old"], f16, t16, D_t, cores),
            self._cb(cb["selrow"], f16, t16, 3, cores),
            self._cb(cb["qout"], f16, t16, 8, cores),
            self._cb(cb["lat_new"], f16, t16, L_t, cores),
            self._cb(cb["ring_new"], f16, t16, 2 * D_t, cores),
            self._cb(cb["pool_new"], f16, t16, D_t, cores),
            self._cb(cb["tmp"], f32, t32, 8, cores),
            self._cb(cb["inv"], f32, t32, 1, cores),
            self._cb(cb["normed"], f32, t32, L_t, cores),
            self._cb(cb["knorm"], f32, t32, D_t, cores),
            self._cb(cb["rows"], f32, t32, L_t + 2 * D_t, cores),
            self._cb(cb["selT"], f32, t32, 3, cores),
            self._cb(cb["keep"], f32, t32, 3, cores),
            self._cb(cb["ringf"], f32, t32, 2 * D_t, cores),
            self._cb(cb["pscr"], f32, t32, 2 * D_t, cores),
            self._cb(cb["prow"], f32, t32, D_t, cores),
            self._cb(cb["run"], f32, t32, 2, cores),
            self._cb(cb["meta_w"], u32, 32, 1, cores),
            self._cb(cb["idx"], u32, 512, 1, cores),
        ]
        qa_o, kva_o, k_o, g_o = self.off
        compute_ct = [
            Lq_t,
            L_t,
            D_t,
            f32_bits(self.eps_rms),
            f32_bits(self.eps_ln),
            f32_bits(1.0 / Lq),
            f32_bits(1.0 / L),
            f32_bits(1.0 / D),
        ]
        reader_ct = [
            Lq_t,
            L_t,
            D_t,
            Ws_t,
            qa_o // TILE,
            kva_o // TILE,
            k_o // TILE,
            g_o // TILE,
            self.P,
            self.pool_blocks,
            Bt,
        ]
        for t in (
            stem,
            latent_cache,
            latent_page_table,
            positions,
            ring_k,
            ring_g,
            pool_cache,
            qa_norm_w,
            kva_norm_w,
            k_norm_w,
            k_norm_b,
            ape,
            selectors,
        ):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [Lq_t, L_t, D_t]
        for t in (q_resid, latent_cache, ring_k, ring_g, pool_cache):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        lpt_page_bytes = -(-(latent_page_table.padded_shape[-1] * 4) // 32) * 32  # one page-table row (int32)
        pos_page_bytes = -(-(positions.padded_shape[-1] * 4) // 32) * 32  # the [B] int32 vector
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for c, core in enumerate(core_of):
            role, idx = (0, c) if c < Bt else (1, c - Bt)
            reader_rt[core.x][core.y] = [
                role,
                idx,
                stem.buffer_address(),
                latent_cache.buffer_address(),
                latent_page_table.buffer_address(),
                positions.buffer_address(),
                ring_k.buffer_address(),
                ring_g.buffer_address(),
                pool_cache.buffer_address(),
                qa_norm_w.buffer_address(),
                kva_norm_w.buffer_address(),
                k_norm_w.buffer_address(),
                k_norm_b.buffer_address(),
                ape.buffer_address(),
                selectors.buffer_address(),
                lpt_page_bytes,
                pos_page_bytes,
            ]
            writer_rt[core.x][core.y] = [
                role,
                idx,
                q_resid.buffer_address(),
                latent_cache.buffer_address(),
                ring_k.buffer_address(),
                ring_g.buffer_address(),
                pool_cache.buffer_address(),
            ]
            compute_rt[core.x][core.y] = [role, idx]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_dsa_decode_state.cpp"),
                core_ranges=cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_dsa_decode_state.cpp"),
                core_ranges=cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_dsa_decode_state.cpp"),
                core_ranges=cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=self.fidelity, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=False
                ),
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op(
            [
                stem,
                latent_cache,
                latent_page_table,
                positions,
                ring_k,
                ring_g,
                pool_cache,
                qa_norm_w,
                kva_norm_w,
                k_norm_w,
                k_norm_b,
                ape,
                selectors,
                q_resid,
            ],
            program,
        )
