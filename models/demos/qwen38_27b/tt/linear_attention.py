# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet (linear attention) of Qwen3.8-27B, chunked prefill on the 8x4 mesh (SP 8 on axis 0, TP 4 on axis 1).

Per chip: the chunk's 640 local rows (row r of the mesh holds positions with (p // 640) % 8 == r) and one TP column's
heads: 4 key heads and 12 value heads (conv channels q 512 | k 512 | v 1536 = 2560, z 1536).

    x [1, 1, 640, 5120] bf16
      -> qkv|z linear (column-parallel, bf16 out) and a|b linear (column-parallel, fp32 out)
      -> qkv_causal_conv1d_silu_with_carry (depthwise conv K=4 + SiLU + q/k/v split; the three predecessor rows come
         over the fabric on axis 0, the chunk's first row reads the slot's carried conv history; returns the new carry)
      -> beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)            (fp32)
      -> chunk_gated_delta_rule on the folded value v' = [0 | v] from S'_0 = [I | 0], chunk 64 on head-split q/k/v
         (q/k l2-normalised here; GDN_DELTA_CHUNK=32 runs the flat path, normalised in the kernel): final state
         [A | B] (this row's affine state transform, S_out = A S_in + B) and outputs [U | O0] (O = U S_entry + O0)
      -> [A | B] as bf16, all_gather over axis 0 -> chain_affine_transforms from the slot's carried recurrent state:
         this row's entry state S_entry and the state after the chunk
      -> O = [U | O0] @ [[S_entry], [I]]                                       (fp32)
      -> gated RMSNorm per value head (w, eps) * silu(z) -> out_proj (row-parallel) -> all_reduce axis 1 (fp32)
      -> bf16 [1, 1, 640, 5120], replicated over the columns.

The slot state (GdnState) is the conv history [1, 3, 2560] bf16 ROW_MAJOR and the recurrent state [12, 128, 128] fp32,
TP-sharded on heads and replicated over SP. ``actual_start`` is a replicated uint32 device scalar (the chunk's first
absolute position); at start 0 the caller passes a zero state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn

from .settings import settings

CONV_K = 4
EXCHANGE_ROWS = 12  # the conv's fabric history exchange buffer: predecessor, carry, outgoing and end rows, 3 each
EXCHANGE_SEMAPHORES = 3
DELTA_CHUNK = 32  # chunk_gated_delta_rule's flat (token-major) path runs only at chunk 32
AB_PAD = 32  # a and b (12 columns per TP column) each padded to one tile in the a|b projection


def compute_config(fidelity: str, fp32_acc: bool = True):
    return ttnn.init_device_compute_kernel_config(
        ttnn.device.Arch.BLACKHOLE,
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=False,
    )


@dataclass
class GdnState:
    conv: ttnn.Tensor  # [1, 3, conv_tp] bf16 ROW_MAJOR, the last three pre-conv q|k|v rows (this column's channels)
    recurrent: ttnn.Tensor  # [nv_tp, dk, dv] fp32 TILE


class _HistoryExchange:
    """The conv's fabric history exchange: a persistent DRAM buffer and three global semaphores on the op's exchange
    core. Peers write this rank's buffer and bump its semaphores, possibly before its own call starts, so both outlive
    every call; their counts carry across calls. One per (mesh, width), shared by every linear layer."""

    _by_mesh: dict = {}

    @classmethod
    def shared(cls, mesh, width: int):
        key = (mesh.id(), width)
        if key not in cls._by_mesh:
            cls._by_mesh[key] = cls(mesh, width)
        return cls._by_mesh[key]

    def __init__(self, mesh, width: int):
        core = ttnn._ttnn.operations.experimental.kda.qkv_causal_conv1d_silu_exchange_core(
            mesh.compute_with_storage_grid_size()
        )
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})
        l1_small = ttnn.get_memory_view(mesh, ttnn.BufferType.L1_SMALL)
        buffer_type = ttnn.BufferType.L1_SMALL if l1_small.total_bytes_per_bank > 0 else ttnn.BufferType.L1
        self.buffer = ttnn.from_torch(
            torch.zeros(1, EXCHANGE_ROWS, width, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.semaphores = [
            ttnn.create_global_semaphore(mesh, cores, 0, buffer_type) for _ in range(EXCHANGE_SEMAPHORES)
        ]


class TtGatedDeltaNet:
    """One linear-attention layer. ``w``: the checkpoint's ``linear_attn.*`` tensors (fp32 host), ``cfg``: the
    reference Config."""

    sp_axis, tp_axis = 0, 1

    def __init__(self, mesh, w: dict, cfg, local_rows: int):
        self.mesh = mesh
        rows, cols = tuple(mesh.shape)
        self.sp, self.tp = rows, cols
        self.local_rows = local_rows
        self.nk, self.nv = cfg.lk_heads // cols, cfg.lv_heads // cols
        self.dk, self.dv = cfg.lk_dim, cfg.lv_dim
        assert self.dk == self.dv, "the folded value [0 | v] and the [A | B] transform assume dk == dv"
        self.kd, self.vd = self.nk * self.dk, self.nv * self.dv  # 512, 1536 per column
        self.conv_tp = 2 * self.kd + self.vd  # 2560
        self.eps = cfg.eps
        self.cfg = cfg
        dram = ttnn.DRAM_MEMORY_CONFIG
        self.mm_cfg = compute_config(settings.get("GDN_FIDELITY"), settings.get("GDN_FP32_ACC"))
        self.chain_cfg = compute_config(settings.get("GDN_CHAIN_FIDELITY"), True)  # the chain requires fp32 dest
        self.exact_cfg = compute_config("HiFi4", True)  # 0/1 placement matmul: exact
        col_mapper = lambda dim: ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(None, dim))  # noqa: E731

        def up(t, dtype, mapper, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, memory_config=dram, mesh_mapper=mapper)

        rep = ttnn.ReplicateTensorToMesh(mesh)
        wqkv = w["linear_attn.in_proj_qkv.weight"].float()  # [2*key_dim + value_dim, H]
        wz = w["linear_attn.in_proj_z.weight"].float()  # [value_dim, H]
        wa = w["linear_attn.in_proj_a.weight"].float()  # [nv, H]
        wb = w["linear_attn.in_proj_b.weight"].float()
        conv_w = w["linear_attn.conv1d.weight"].float().reshape(cfg.conv_dim, -1)  # [conv_dim, K]
        key_dim = cfg.key_dim
        kd, vd, nv = self.kd, self.vd, self.nv

        def conv_cols(t):
            """Rows (channels) of a [conv_dim, ...] tensor in column order: per column c its q | k | v channels."""
            q, k, v = t[:key_dim], t[key_dim : 2 * key_dim], t[2 * key_dim :]
            return torch.cat(
                [
                    torch.cat([q[c * kd : (c + 1) * kd], k[c * kd : (c + 1) * kd], v[c * vd : (c + 1) * vd]])
                    for c in range(cols)
                ]
            )

        self._conv_cols = conv_cols
        qkv_c = conv_cols(wqkv)
        qkvz = torch.cat(
            [
                torch.cat([qkv_c[c * self.conv_tp : (c + 1) * self.conv_tp], wz[c * vd : (c + 1) * vd]])
                for c in range(cols)
            ]
        )  # [cols * (conv_tp + vd), H]
        self.w_qkvz = up(qkvz.T.contiguous(), ttnn.bfloat8_b, col_mapper(-1))

        def pad_heads(t):  # [nv_total, ...] -> per column nv rows padded to AB_PAD
            out = torch.zeros(cols * AB_PAD, *t.shape[1:], dtype=t.dtype)
            for c in range(cols):
                out[c * AB_PAD : c * AB_PAD + nv] = t[c * nv : (c + 1) * nv]
            return out

        ab = torch.cat(
            [
                torch.cat([pad_heads(wa)[c * AB_PAD : (c + 1) * AB_PAD], pad_heads(wb)[c * AB_PAD : (c + 1) * AB_PAD]])
                for c in range(cols)
            ]
        )  # [cols * 2 * AB_PAD, H]
        self.w_ab = up(ab.T.contiguous(), ttnn.bfloat8_b, col_mapper(-1))
        # bfloat8_b shares one exponent per 16 values along a tile row. The in-projections are stored [K, N] (blocks
        # along the outputs): the layer inputs carry a few huge channels (layer-3 attn_norm: one channel near the row
        # norm), whose weights then keep their bits. out_proj is stored [N, K] and read with transpose_b (blocks along
        # K): its output column 3994 is ~20x its neighbours at every linear layer, and blocks along N cost column 3993
        # rel 0.5.
        self.w_out = up(w["linear_attn.out_proj.weight"].float(), ttnn.bfloat8_b, col_mapper(-1))
        self.dt_bias = up(pad_heads(w["linear_attn.dt_bias"].float()).reshape(1, 1, -1), ttnn.float32, col_mapper(-1))
        self.neg_exp_a = up(
            pad_heads(-w["linear_attn.A_log"].float().exp()).reshape(1, 1, -1), ttnn.float32, col_mapper(-1)
        )
        self.norm_w = up(w["linear_attn.norm.weight"].float().reshape(1, 1, 1, -1), ttnn.float32, rep)
        taps = conv_cols(conv_w)  # [cols * conv_tp, K]; tap j multiplies row t - 3 + j
        self.taps = [
            up(taps[:, j].reshape(1, 1, -1).contiguous(), ttnn.bfloat16, col_mapper(-1)) for j in range(CONV_K)
        ]
        self.conv_pc = ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=settings.get("GDN_CONV_CHANNEL_CHUNK"))
        self.exchange = _HistoryExchange.shared(mesh, self.conv_tp)

        # fold constants: S'_0 = [I | 0] per head, the placement P (v @ P = per head [0 | v]) and the identity of W
        dk, dv = self.dk, self.dv
        init_fold = torch.zeros(1, nv, dk, dk + dv)
        init_fold[:, :, :, :dk] = torch.eye(dk)
        self.init_fold = up(init_fold, ttnn.float32, rep)
        place = torch.zeros(nv * dv, nv * (dk + dv))
        for h in range(nv):
            place[h * dv : (h + 1) * dv, h * (dk + dv) + dk : (h + 1) * (dk + dv)] = torch.eye(dv)
        self.place = up(place, ttnn.bfloat16, rep)
        self.fold_eye = up(torch.eye(dv).expand(nv, dv, dv).contiguous(), ttnn.float32, rep)
        self.const_tiles = self._delta_const_tiles(up, rep)

    @staticmethod
    def _delta_const_tiles(up, rep):
        """chunk_gated_delta_rule's constant tiles (identity, lower-triangular ones, ones, WY quadrant masks), built
        once: without them the op uploads them on every call."""
        C = DELTA_CHUNK
        ii, jj = torch.arange(32)[:, None], torch.arange(32)[None, :]
        lo_i, lo_j = ii < 16, jj < 16
        masks = torch.cat([(lo_i & lo_j).float(), (~lo_i & ~lo_j).float(), (~lo_i & lo_j).float()], dim=1)
        tiles = [torch.eye(C), torch.tril(torch.ones(C, C)), torch.ones(C, C), masks]
        return tuple(up(t.reshape(1, 1, *t.shape), ttnn.float32, rep) for t in tiles)

    # ---- state (host <-> device: load time and the harness boundary only)
    def state_from_host(self, conv: torch.Tensor | None, recurrent: torch.Tensor | None) -> GdnState:
        """conv [conv_dim, 3] (the reference's layout), recurrent [nv_total, dk, dv]; None = a new sequence."""
        rows, cols = self.sp, self.tp
        cfg = self.cfg
        conv = torch.zeros(cfg.conv_dim, CONV_K - 1) if conv is None else conv.float()
        recurrent = torch.zeros(cfg.lv_heads, self.dk, self.dv) if recurrent is None else recurrent.float()
        hist = self._conv_cols(conv).T.reshape(1, CONV_K - 1, -1).to(torch.bfloat16).contiguous()
        mapper = lambda dim: ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=(rows, cols), dims=(None, dim))  # noqa: E731
        return GdnState(
            conv=ttnn.from_torch(
                hist,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper(-1),
            ),
            recurrent=ttnn.from_torch(
                recurrent.contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper(0),
            ),
        )

    def scalar(self, value: int) -> ttnn.Tensor:
        """A replicated uint32 device scalar (actual_start)."""
        return ttnn.from_torch(
            torch.tensor([value], dtype=torch.int64),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    @staticmethod
    def _split_heads(t, heads: int, dim: int):
        """[1, T, heads*dim] TILE -> [1, T, heads, dim] TILE (through ROW_MAJOR: a TILE reshape would pad heads)."""
        T = t.shape[-2]
        dram = ttnn.DRAM_MEMORY_CONFIG
        r = ttnn.reshape(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram), (1, T, heads, dim))
        return ttnn.to_layout(r, ttnn.TILE_LAYOUT, memory_config=dram)

    def _l2norm_heads(self, t, heads: int, dim: int):
        """q or k [1, T, heads*dim] bf16 -> per-head l2-normalised [1, T, heads, dim] bf16: x * rsqrt(sum x^2 + 1e-6)
        in fp32, as the reference."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        x = self._split_heads(ttnn.typecast(t, ttnn.float32, memory_config=dram), heads, dim)
        ss = ttnn.sum(ttnn.multiply(x, x, memory_config=dram), dim=-1, keepdim=True, memory_config=dram)
        y = ttnn.multiply(x, ttnn.rsqrt(ttnn.add(ss, 1e-6, memory_config=dram), memory_config=dram), memory_config=dram)
        ttnn.deallocate(x)
        return ttnn.typecast(y, ttnn.bfloat16, memory_config=dram)

    # ---- forward
    def __call__(self, x, state: GdnState, actual_start) -> tuple[ttnn.Tensor, GdnState]:
        dram = ttnn.DRAM_MEMORY_CONFIG
        T = x.shape[-2]
        nk, nv, dk, dv, kd, vd = self.nk, self.nv, self.dk, self.dv, self.kd, self.vd
        x3 = ttnn.reshape(x, (1, T, x.shape[-1]))

        qkvz = ttnn.linear(x3, self.w_qkvz, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_cfg, memory_config=dram)
        ab = ttnn.linear(x3, self.w_ab, dtype=ttnn.float32, compute_kernel_config=self.mm_cfg, memory_config=dram)

        q, k, v, conv_out = ttnn.experimental.kda.qkv_causal_conv1d_silu_with_carry(
            qkvz,
            state.conv,
            *self.taps,
            kd,
            kd,
            vd,
            program_config=self.conv_pc,
            actual_start=actual_start,
            exchange_buffer=self.exchange.buffer,
            exchange_semaphores=self.exchange.semaphores,
            sequence_parallel_axis=self.sp_axis,
            memory_config=dram,
        )
        z = ttnn.slice(qkvz, (0, 0, self.conv_tp), (1, T, self.conv_tp + vd), memory_config=dram)
        ttnn.deallocate(qkvz)

        a = ttnn.slice(ab, (0, 0, 0), (1, T, AB_PAD), memory_config=dram)
        b = ttnn.slice(ab, (0, 0, AB_PAD), (1, T, 2 * AB_PAD), memory_config=dram)
        ttnn.deallocate(ab)
        g = ttnn.multiply(
            ttnn.softplus(ttnn.add(a, self.dt_bias, memory_config=dram), memory_config=dram),
            self.neg_exp_a,
            memory_config=dram,
        )
        beta = ttnn.sigmoid(b, memory_config=dram)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        g = ttnn.slice(g, (0, 0, 0), (1, T, nv), memory_config=dram)
        beta = ttnn.slice(beta, (0, 0, 0), (1, T, nv), memory_config=dram)

        # fold: v' = [0 | v] per head (exact 0/1 matmul), from S'_0 = [I | 0]
        vf = ttnn.matmul(v, self.place, dtype=ttnn.bfloat16, compute_kernel_config=self.exact_cfg, memory_config=dram)
        ttnn.deallocate(v)
        chunk = settings.get("GDN_DELTA_CHUNK")
        if chunk == DELTA_CHUNK:  # flat token-major q/k/v: the kernel l2-normalises q/k
            qa, ka, va = q, k, vf
            extra = {"num_key_heads": nk, **dict(zip(("eye", "tril", "ones", "masks"), self.const_tiles))}
        else:  # rank-4 head-split q/k/v, the only path that runs chunk 64: q/k l2-normalised here, in fp32
            qa, ka, va, extra = (
                self._l2norm_heads(q, nk, dk),
                self._l2norm_heads(k, nk, dk),
                self._split_heads(vf, nv, dk + dv),
                {},
            )
        o, fin = ttnn.transformer.chunk_gated_delta_rule(
            qa,
            ka,
            va,
            g,
            beta,
            scale=dk**-0.5,
            initial_state=self.init_fold,
            output_final_state=True,
            chunk_size=chunk,
            output_head_major=True,
            **extra,
        )  # o [nv, T, dk + dv] fp32 = [U | O0]; fin [1, nv, dk, dk + dv] fp32 = [A | B]
        for t in {id(t): t for t in (q, k, vf, qa, ka, va, g, beta)}.values():
            ttnn.deallocate(t)

        # chain the rows' transforms from the slot's carried state
        packed = ttnn.typecast(ttnn.reshape(fin, (1, nv, dk, dk + dv)), ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(fin)
        gathered = ttnn.all_gather(packed, dim=0, cluster_axis=self.sp_axis, memory_config=dram)
        ttnn.deallocate(packed)
        entry, final = ttnn.experimental.kda.chain_affine_transforms(
            gathered,
            state.recurrent,
            actual_start=actual_start,
            local_rows=T,
            memory_config=dram,
            compute_kernel_config=self.chain_cfg,
            sequence_parallel_axis=self.sp_axis,
        )
        ttnn.deallocate(gathered)

        # O = [U | O0] @ [[S_entry], [I]]
        wfold = ttnn.concat([entry, self.fold_eye], dim=1, memory_config=dram)  # [nv, dk + dv, dv]
        ttnn.deallocate(entry)
        core = ttnn.matmul(
            ttnn.reshape(o, (1, nv, T, dk + dv)),
            ttnn.reshape(wfold, (1, nv, dk + dv, dv)),
            dtype=ttnn.float32,
            compute_kernel_config=self.mm_cfg,
            memory_config=dram,
        )  # [1, nv, T, dv]
        ttnn.deallocate(o)
        ttnn.deallocate(wfold)

        # gated RMSNorm per value head, then * silu(z)
        n = ttnn.rms_norm(
            core, epsilon=self.eps, weight=self.norm_w, memory_config=dram, compute_kernel_config=self.mm_cfg
        )
        ttnn.deallocate(core)
        n = ttnn.experimental.nlp_concat_heads(n, memory_config=dram)  # [1, 1, T, vd]
        zf = ttnn.typecast(ttnn.reshape(z, (1, 1, T, vd)), ttnn.float32, memory_config=dram)
        ttnn.deallocate(z)
        gated = ttnn.multiply(n, ttnn.silu(zf, memory_config=dram), dtype=ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(n)
        ttnn.deallocate(zf)

        partial = ttnn.linear(
            gated,
            self.w_out,
            transpose_b=True,
            dtype=ttnn.float32,
            compute_kernel_config=self.mm_cfg,
            memory_config=dram,
        )
        ttnn.deallocate(gated)
        out = ttnn.all_reduce(partial, cluster_axis=self.tp_axis, memory_config=dram)
        ttnn.deallocate(partial)
        out = ttnn.typecast(out, ttnn.bfloat16, memory_config=dram)
        return out, GdnState(conv=conv_out, recurrent=final)
