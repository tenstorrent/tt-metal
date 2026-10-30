# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sequence parallelism (SP) over mesh axis 0 for qwen36 prefill on a 2D (rows, tp) mesh.

A prompt of B tokens is split into R = mesh rows contiguous slices of L = B/R tokens: row r holds tokens
[r*L, (r+1)*L). Everything token-local (embedding, MLP, norms, projections, TP CCLs) is unchanged; full attention
all-gathers K/V over the rows and GDN chains the rows' affine state transforms. From-zero-state single pass only.
"""
import os

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.tt_transformers.tt.ccl import tt_all_gather

CONV_HISTORY = 3  # conv kernel 4 -> three predecessor rows


class SeqParallel:
    """Sequence parallelism over mesh axis 0 for a from-zero-state single-pass prefill. Owns the constant
    tensors the SP layers read; allocate (construct + prepare) before any trace capture."""

    axis = 0

    def __init__(self, mesh, args, tt_ccl):
        assert tpc.is_2d_mesh(mesh), f"SeqParallel needs a 2D (rows, tp) mesh, got {tuple(mesh.shape)}"
        self.mesh = mesh
        self.args = args
        self.tt_ccl = tt_ccl
        self.rows = mesh.shape[0]
        self.Nv, self.Dk, self.Dv = args.gdn_nv_tp, args.gdn_dk, args.gdn_dv
        assert self.Dk == self.Dv, "the packed [A | B] transform and U @ S_c assume Dk == Dv"
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def up(t, dtype, layout=ttnn.TILE_LAYOUT, mapper=rep):
            return ttnn.from_torch(
                t,
                dtype=dtype,
                layout=layout,
                device=mesh,
                mesh_mapper=mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Row r reads its predecessor's last three conv-input rows out of the gathered [1, 3R, C]: rows
        # [3(r-1), 3(r-1)+3). Row 0 never reads it (the conv op uses the zero history on the rank holding token 0).
        idx = torch.tensor([[3 * max(r - 1, 0) + j for j in range(CONV_HISTORY)] for r in range(self.rows)])
        self.conv_pred_idx = up(
            idx.to(torch.int32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mapper=self.rows_mapper(0)
        )  # per device [1, 3]
        # Batch-2 GDN initial state: batch 0 zero (real inputs -> O0 and B), batch 1 identity (v = 0 -> U and A).
        init2 = torch.zeros(2, self.Nv, self.Dk, self.Dv, dtype=torch.float32)
        init2[1] = torch.eye(self.Dk, dtype=torch.float32).expand(self.Nv, self.Dk, self.Dv)
        self.gdn_init2 = up(init2, ttnn.float32)
        self.gdn_zero_state = up(torch.zeros(self.Nv, self.Dk, self.Dv, dtype=torch.float32), ttnn.float32)
        # Same constant as gdn/tp.py _kda_actual_start: the chunk's first token is at absolute position 0.
        self.actual_start = up(torch.tensor([0], dtype=torch.int64), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        # chain_affine_transforms requires HiFi2 + fp32 dest accumulation; the U @ S_c combine runs HiFi4 by default.
        self._chain_cfg = self._compute_config(ttnn.MathFidelity.HiFi2)
        # The combine is O = O0 + U @ S_c (gdn_combine and gdn_combine_fold); lower fidelity is faster, so validate accuracy
        # before changing the default. QWEN36_SP_COMBINE_FIDELITY: hifi4 (default) | hifi2 | lofi; fp32 dest acc stays on.
        combine_fidelity = os.environ.get("QWEN36_SP_COMBINE_FIDELITY", "hifi4").strip().lower()
        combine_fidelities = {
            "hifi4": ttnn.MathFidelity.HiFi4,
            "hifi2": ttnn.MathFidelity.HiFi2,
            "lofi": ttnn.MathFidelity.LoFi,
        }
        assert (
            combine_fidelity in combine_fidelities
        ), f"QWEN36_SP_COMBINE_FIDELITY={combine_fidelity!r}: expected one of {sorted(combine_fidelities)}"
        self._combine_cfg = self._compute_config(combine_fidelities[combine_fidelity])
        # Fold variant (gdn_fold_v / gdn_entry_state_fold / gdn_combine_fold): ONE batch-1 delta-op call on the augmented value
        # v' = [0 | v] from S'_0 = [I | 0] instead of the batch-2 call. Needs a library whose chunk_gated_delta_rule accepts
        # num_key_heads (flat q/k with Dk != Dv'; built 2026-10-13). Measured in the full model: the fold is faster for per-row
        # lengths T <= 256 (-2 to -3 ms per forward), the batch-2 call faster above. use_fold(T) picks per length; the fold
        # constants are built whenever fold_max_len > 0 (QWEN36_SP_GDN_FOLD_MAX, default 256; 0 disables the fold), the
        # batch-2 ones (gdn_init2, per-length _v_zero) always where they are used.
        self.fold_max_len = int(os.environ.get("QWEN36_SP_GDN_FOLD_MAX", "256"))
        if self.fold_max_len > 0:
            Nv, Dk, Dv = self.Nv, self.Dk, self.Dv
            # S'_0 = [I | 0] per head: [1, Nv, Dk, Dk + Dv] (the op's initial state is [B, HV, K, V'] with V' = Dk + Dv).
            init_fold = torch.zeros(1, Nv, Dk, Dk + Dv, dtype=torch.float32)
            init_fold[:, :, :, :Dk] = torch.eye(Dk, dtype=torch.float32)
            self.gdn_init_fold = up(init_fold, ttnn.float32)
            # [Nv, Dv, Dv] identity per head: the lower half of W = [[S_c], [I]].
            self.fold_eye = up(torch.eye(Dv, dtype=torch.float32).expand(Nv, Dv, Dv).contiguous(), ttnn.float32)
            # Placement matrix P [Nv*Dv, Nv*(Dk+Dv)], P[h*Dv + j, h*(Dk+Dv) + Dk + j] = 1: v @ P puts head h's v in the right
            # half of head h's augmented block (zeros in the left Dk columns).
            place = torch.zeros(Nv * Dv, Nv * (Dk + Dv), dtype=torch.float32)
            eye_v = torch.eye(Dv, dtype=torch.float32)
            for h in range(Nv):
                c0 = h * (Dk + Dv) + Dk
                place[h * Dv : (h + 1) * Dv, c0 : c0 + Dv] = eye_v
            self.fold_place = up(place.to(torch.bfloat16), ttnn.bfloat16)
            # HiFi4 + fp32 dest accumulation: the 0/1 placement matmul is exact (each output is one bf16 value times 1).
            self._fold_mm_cfg = self._compute_config(ttnn.MathFidelity.HiFi4)
        self._v_zero = {}  # local length L -> bf16 zeros shaped like the flat GDN v input [1, L, Nv*Dv]
        self._cu_window = {}  # local length L -> uint32 ROW_MAJOR [0, rows*L], the windowed SDPA's cu_window_seqlens

    def use_fold(self, T):
        """True if the GDN of a per-row slice of T tokens runs the Dv-fold variant (else the batch-2 one)."""
        return T <= self.fold_max_len

    def _compute_config(self, fidelity):
        return ttnn.init_device_compute_kernel_config(
            self.mesh.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )

    def prepare(self, local_lens):
        """Allocate the per-length constants for every local length L (tokens per row) that will run. The batch-2 zero v
        is only needed for the lengths that do not use the fold (L > fold_max_len). The cu_window_seqlens [0, rows*L]
        (windowed SDPA over the gathered K/V, one window = the whole sequence) is built for every L."""
        for L in local_lens:
            assert L % tpc.TILE_SIZE == 0, f"local length {L} must be a multiple of {tpc.TILE_SIZE}"
            if L not in self._cu_window:
                self._cu_window[L] = ttnn.from_torch(
                    torch.tensor([0, self.rows * L], dtype=torch.int32),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )  # 1-D [2]
            if self.use_fold(L) or L in self._v_zero:
                continue
            self._v_zero[L] = ttnn.from_torch(
                torch.zeros(1, L, self.Nv * self.Dv, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    def cu_window(self, L):
        """cu_window_seqlens [0, rows*L] (uint32 ROW_MAJOR, replicated) of the windowed SDPA for L tokens per row."""
        assert L in self._cu_window, f"SeqParallel.prepare([{L}]) must run before the first prefill of {L} local tokens"
        return self._cu_window[L]

    def rows_mapper(self, dim):
        """Mesh mapper sharding `dim` over the rows (SP axis), replicated over the TP axis."""
        return ttnn.ShardTensor2dMesh(self.mesh, dims=(dim, None), mesh_shape=tuple(self.mesh.shape))

    def gather_rows(self, x, dim):
        """All-gather x over mesh axis 0 along `dim` -> DRAM; each row's slice lands at its row index.

        Consumes x: tt_all_gather deallocates its input (when it is bf16 in DRAM), so do not read x afterwards."""
        if x.layout == ttnn.ROW_MAJOR_LAYOUT:
            # tt_all_gather's all_gather_async sends row-major through a composite (all_broadcast + concat);
            # ttnn.all_gather gathers aligned row-major pages natively (the call deepseek_v3_d_p's conv carry uses).
            out = ttnn.all_gather(x, dim=dim, cluster_axis=self.axis, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(x)
            return out
        return tt_all_gather(
            x,
            self.mesh,
            self.tt_ccl,
            cluster_axis=self.axis,
            dim=dim,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def conv_predecessor(self, xin):
        """The previous row's last three conv-input rows: [1, 3, C] ROW_MAJOR bf16 (row 0's is unused).
        xin: ROW_MAJOR bf16 [1, n, C] whose last three rows are this row's last three conv-input rows: the
        conv input's last tile row as kda_conv_prefill passes it (n = 32), or the whole input; not consumed."""
        T, C = xin.shape[1], xin.shape[2]
        outgoing = ttnn.slice(xin, (0, T - CONV_HISTORY, 0), (1, T, C), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gathered = self.gather_rows(outgoing, dim=1)  # [1, 3R, C]
        # Pick this row's predecessor rows out of the gathered table (per-device index row, no host readback).
        pred = ttnn.embedding(
            self.conv_pred_idx,
            ttnn.reshape(gathered, (CONV_HISTORY * self.rows, C)),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gathered)
        return ttnn.reshape(pred, (1, CONV_HISTORY, C))

    def gdn_batch2(self, q, k, v, beta, g):
        """Batch-2 GDN inputs: batch 0 the real q/k/v/beta/g, batch 1 the same q/k/beta/g with v = 0. With the
        gdn_init2 initial state one op call gives O0 and B (batch 0) and U and A (batch 1). Consumes the inputs."""
        L = v.shape[1]
        assert L in self._v_zero, f"SeqParallel.prepare([{L}]) must run before the first prefill of {L} local tokens"

        def cat(a, b):
            return ttnn.concat([a, b], dim=0, memory_config=a.memory_config())  # keep the input's placement

        q2, k2, beta2, g2 = cat(q, q), cat(k, k), cat(beta, beta), cat(g, g)
        v2 = cat(v, self._v_zero[L])
        for t in (q, k, v, beta, g):
            ttnn.deallocate(t)
        return q2, k2, v2, beta2, g2

    def gdn_entry_state(self, fin2, local_rows):
        """This row's entry state S_c [Nv, Dk, Dv] fp32 from the batch-2 final state fin2 [2, Nv, Dk, Dv]
        (batch 0 = B, batch 1 = A: S_out = A @ S_in + B). Chains the rows' transforms S_{r+1} = A_r @ S_r + B_r from
        a zero state on device. local_rows: tokens per row. Consumes fin2."""
        Nv, Dk, Dv = self.Nv, self.Dk, self.Dv
        dram = ttnn.DRAM_MEMORY_CONFIG
        b = ttnn.slice(fin2, (0, 0, 0, 0), (1, Nv, Dk, Dv), memory_config=dram)
        a = ttnn.slice(fin2, (1, 0, 0, 0), (2, Nv, Dk, Dv), memory_config=dram)
        ttnn.deallocate(fin2)
        # Pack [A | B] (Dk == Dv, so no reshape) and transport as bf16, the chain op's input contract.
        packed = ttnn.concat([a, b], dim=3, memory_config=dram)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        packed_bf16 = ttnn.typecast(packed, ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(packed)
        gathered = self.gather_rows(packed_bf16, dim=0)  # [R, Nv, Dk, Dk + Dv]
        entry, final = ttnn.experimental.kda.chain_affine_transforms(
            gathered,
            self.gdn_zero_state,
            actual_start=self.actual_start,
            local_rows=local_rows,
            memory_config=dram,
            compute_kernel_config=self._chain_cfg,
            sequence_parallel_axis=self.axis,
        )
        ttnn.deallocate(final)
        ttnn.deallocate(gathered)
        return entry

    def gdn_combine(self, o2, entry):
        """True outputs O = O0 + U @ S_c per head, from the batch-2 outputs o2 [2*Nv, T, Dv] fp32 (O0 then U) and
        the entry state [Nv, Dk, Dv]. Returns fp32 [Nv, T, Dv] TILE DRAM, what the single-batch op returns.
        Consumes o2 and entry."""
        Nv, Dk, Dv = self.Nv, self.Dk, self.Dv
        T = o2.shape[1]
        dram = ttnn.DRAM_MEMORY_CONFIG
        o0 = ttnn.reshape(ttnn.slice(o2, (0, 0, 0), (Nv, T, Dv), memory_config=dram), (1, Nv, T, Dv))
        u = ttnn.reshape(ttnn.slice(o2, (Nv, 0, 0), (2 * Nv, T, Dv), memory_config=dram), (1, Nv, T, Dk))
        ttnn.deallocate(o2)
        # Heads as the batch dim: [1, Nv, T, Dk] @ [1, Nv, Dk, Dv], fp32 (HiFi4, fp32 dest acc).
        us = ttnn.matmul(
            u,
            ttnn.reshape(entry, (1, Nv, Dk, Dv)),
            dtype=ttnn.float32,
            memory_config=dram,
            compute_kernel_config=self._combine_cfg,
        )
        ttnn.deallocate(u)
        ttnn.deallocate(entry)
        out = ttnn.add(o0, us, dtype=ttnn.float32, memory_config=dram)
        ttnn.deallocate(o0)
        ttnn.deallocate(us)
        return ttnn.reshape(out, (Nv, T, Dv))

    # ---- fold variant: one batch-1 call on v' = [0 | v] from S'_0 = [I | 0] (see __init__) ----------------------------------
    # The recurrence is linear in (S, v). With the augmented value v' = [0 | v] (Dk zero columns, then v; Dv' = Dk + Dv) and
    # S'_0 = [I | 0], one call yields the final state S'_T = [A | B] (the packed transform layout) and the outputs
    # o'_t = [U_t | O0_t]. With this row's entry state S_c the true outputs are O = O0 + U @ S_c = o' @ [[S_c], [I]].

    def gdn_fold_v(self, v):
        """Augmented value v' = v @ P: [1, T, Nv*Dv] bf16 -> [1, T, Nv*(Dk + Dv)] bf16, per head [0 | v]. Consumes v."""
        v2 = ttnn.matmul(
            v,
            self.fold_place,
            dtype=ttnn.bfloat16,
            memory_config=v.memory_config(),
            compute_kernel_config=self._fold_mm_cfg,
        )
        ttnn.deallocate(v)
        return v2

    def gdn_entry_state_fold(self, fin, local_rows):
        """This row's entry state S_c [Nv, Dk, Dv] fp32 from the fold call's final state fin [1, Nv, Dk, Dk + Dv] fp32
        (already the packed [A | B]: S_out = A @ S_in + B). Chains the rows' transforms from a zero state on device, as
        gdn_entry_state does. local_rows: tokens per row. Consumes fin."""
        Nv, Dk, Dv = self.Nv, self.Dk, self.Dv
        dram = ttnn.DRAM_MEMORY_CONFIG
        if tuple(fin.shape) != (1, Nv, Dk, Dk + Dv):
            fin = ttnn.reshape(fin, (1, Nv, Dk, Dk + Dv))
        # Transport as bf16, the chain op's input contract.
        fin_bf16 = ttnn.typecast(fin, ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(fin)
        gathered = self.gather_rows(fin_bf16, dim=0)  # [R, Nv, Dk, Dk + Dv]
        entry, final = ttnn.experimental.kda.chain_affine_transforms(
            gathered,
            self.gdn_zero_state,
            actual_start=self.actual_start,
            local_rows=local_rows,
            memory_config=dram,
            compute_kernel_config=self._chain_cfg,
            sequence_parallel_axis=self.axis,
        )
        ttnn.deallocate(final)
        ttnn.deallocate(gathered)
        return entry

    def gdn_combine_fold(self, o, entry):
        """True outputs O = o' @ W per head, W = [[S_c], [I]] ([Dk + Dv, Dv]), from the fold call's head-major outputs o
        [Nv, T, Dk + Dv] fp32 and the entry state [Nv, Dk, Dv]. Returns fp32 [Nv, T, Dv] TILE DRAM, what the single-batch op
        returns. Consumes o and entry."""
        Nv, Dk, Dv = self.Nv, self.Dk, self.Dv
        T = o.shape[1]
        dram = ttnn.DRAM_MEMORY_CONFIG
        w = ttnn.concat([entry, self.fold_eye], dim=1, memory_config=dram)  # [Nv, Dk + Dv, Dv]
        # Heads as the batch dim: [1, Nv, T, Dk + Dv] @ [1, Nv, Dk + Dv, Dv], fp32 (HiFi4, fp32 dest acc).
        out = ttnn.matmul(
            ttnn.reshape(o, (1, Nv, T, Dk + Dv)),
            ttnn.reshape(w, (1, Nv, Dk + Dv, Dv)),
            dtype=ttnn.float32,
            memory_config=dram,
            compute_kernel_config=self._combine_cfg,
        )
        ttnn.deallocate(o)
        ttnn.deallocate(entry)
        ttnn.deallocate(w)
        return ttnn.reshape(out, (Nv, T, Dv))
