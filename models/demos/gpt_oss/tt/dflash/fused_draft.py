# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The DFlash drafter's draft and context passes on the multi-row fused-decode ops (the same ops as the 8-row verify
pass, dflash/verify.py):

  draft pass: embeddings [last token, mask x 7] -> entry boundary (row-tile residual + input norm)
    per layer: QKV stream (+ bias) into [32 users, heads, 64] head tensors
               -> one program: q / k RMSNorm + RoPE (dflash/qk_prep.py) + K / V rows into the draft cache
                  (dflash/kv_write.py) + block attention over the window (dflash/block_attention.py, every row bounded
                  by the block's last position: not causal; it patches the block's own rows in from the head shards)
                  + o_proj (weights streamed in while attention runs) with the attn-site boundary send
               -> boundary (all-reduce + residual + post-attention norm) + dense MLP in one program
                  (dflash/moe_fused.py DenseMLPFused) with the moe-site send
    final norm -> the BFP4 8-row argmax LM head (the verify pass's prescreen head) -> on-device merge into the verify
    pass's token input.
  context pass: the accepted rows' captured target features (ctx_buf, written by the verify pass) -> fc + hidden_norm
    -> every layer's K / V rows at once -> k_norm + RoPE -> the draft cache.

Weights are streamed in BF8 (the drafter's checkpoint is BF16), the residual is the boundary's BF16 row-tile vector.
"""

import os
from pathlib import Path

import torch

import ttnn
from models.tt_transformers.tt.load_checkpoints import reverse_permute, reverse_permute_1d

from ..fused_decode.boundary import PendingBoundary, flat_shape
from ..fused_decode.config import LM_HEAD_STREAM_PREFETCH
from ..fused_decode.stream import (
    LinearStream,
    as_stream_tensor,
    mark_cache_built,
    stream_down_layout,
    stream_gate_up_layout,
    stream_linear_layout,
)
from ..fused_decode.union import act_buffer
from .argmax import ArgmaxGather
from .block_attention import BlockAttention, free_cores
from .kv_write import DraftKVWrite
from .moe_fused import DenseMLPFused
from .qk_prep import DraftQKPrep

TILE = ttnn.TILE_SIZE
ROWS = 32

# program semaphores of the drafter's attention program (clear of the boundary consumer's 0, the boundary helpers' 6
# (fused_decode/boundary.py XCHG_SEM) and the block attention's 9 .. 11): the KV write's (12, 13), KV write -> block
# attention and o_proj's weight readers (15), block attention's roots -> o_proj's writers (7), o_proj's writers -> the
# boundary sender (4)
PREP_SEM_BASE, BA_PREP_SEM, OPROJ_X_SEM, OPROJ_SEND_SEM = 12, 15, 7, 4
# The MLP weights' logical bank (b + 7) % banks on DRAM bank b, each slice's intermediate padding at its end and the
# cores whose columns are all padding dropped: the padding sits on DRAM bank 0, the slowest for the streaming readers
# (fused_decode/stream.py EXPERT_BANK_ROT)
DRAFT_BANK_ROT = 7
MLP_SLICES = 2  # the dense MLP as 2 column slices: down starts on slice 0 while gate|up computes slice 1
MLP_READERS = 4  # gate|up and down readers per DRAM bank
LM_READERS = 3  # the BFP4 LM head's readers per bank (the verify pass's prescreen head)
BA_SLICES = 24  # block attention slices per kv head (capped by the free cores and the window's chunks)
BA_MIN_CHUNKS = 4  # 32-position chunks per slice at least
CTX_KV_CORES = 4  # the context pass's 8-layer cache write split over 4 cores


class _Norm:
    def __init__(self, tt_weight):
        self.tt_weight = tt_weight


class FusedDraft:
    def __init__(self, draft, verify, dtype=ttnn.bfloat8_b):
        """draft: the DFlashDraft; verify: the FusedVerify it feeds (its boundary, partial buffer,
        token input and prescreen LM head are shared)."""
        self.d, self.v = draft, verify
        mesh = self.mesh = draft.mesh_device
        b = self.b = verify.boundary
        H, hd, nq, nkv = draft.hidden, draft.hd, draft.nq, draft.nkv
        self.M = M = draft.block
        banks = mesh.dram_grid_size().x
        I = draft.inter
        self.I_pad = I_pad = -(-I // (TILE * banks)) * TILE * banks
        self.g_ln1 = [b.flat_gamma(_Norm(L["ln1"])) for L in draft.L]
        self.g_ln2 = [b.flat_gamma(_Norm(L["ln2"])) for L in draft.L]
        self.g_final = b.flat_gamma(_Norm(draft.norm))
        self.qkv = LinearStream(
            mesh,
            H // TILE,
            (nq + 2 * nkv) * hd,
            dtype,
            readers=1,
            out_mode=3,
            heads=(nq * hd // TILE, nkv * hd // TILE),
            rows=M,
            # the readers hold all 5 of their weight columns: they stream in during the boundary
            prefetch_cols=5,
        )
        # o_proj (+ the attn-site boundary send) runs in the attention program; the prep cores sit at (1 + r % 8, r // 8)
        # and the KV write at (9, 0), clear of the boundary core; o_proj's readers stay off them and the boundary's cores
        prep_xy = [(1 + r % 8, r // 8) for r in range(M)] + [(9, 0)]
        bnd_xy = [(c.x, c.y) for c in ttnn.corerange_to_cores(b.cores, row_wise=True)]
        bnd_xy += [(h.x, h.y) for h in b.helpers]
        self.o = LinearStream(
            mesh,
            nq * hd // TILE,
            H,
            dtype,
            readers=2,
            x_pages=hd // TILE,
            out_mode=5,
            rows=M,
            gather_split=True,
            exclude=prep_xy + bnd_xy,
        )
        self.mlp_fused = DenseMLPFused(
            mesh,
            b,
            H,
            I_pad // MLP_SLICES,
            MLP_SLICES,
            M,
            dtype,
            gate_up_readers=MLP_READERS,
            down_readers=MLP_READERS,
            rot=DRAFT_BANK_ROT,
            real_inter=I // MLP_SLICES,
            real_hidden=True,
        )
        self.gu, self.down = self.mlp_fused.gate_up, self.mlp_fused.down
        self._load_weights(dtype, banks)
        # LM head: the verify pass's BFP4 prescreen head (the drafts only need the argmax to agree with the target's
        # often; the verify pass rescores its candidates in BF8)
        assert len(verify.lm_head.assignments) == LM_READERS * banks
        self.lm_w, n = verify.lm_weights, verify.lm_n4
        self.lm = LinearStream(
            mesh,
            H // TILE,
            n,
            ttnn.bfloat4_b,
            readers=LM_READERS,
            out_mode=6,
            prefetch_cols=LM_HEAD_STREAM_PREFETCH,
            rows=M,
        )
        rep = ttnn.ReplicateTensorToMesh(mesh)
        self.lists = ttnn.from_torch(
            torch.zeros(1, len(self.lm.assignments) * M * 2, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.gather = ArgmaxGather(mesh, M, len(self.lm.assignments), draft.vd, kind=0)

        def dram(t, dt=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mc=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.from_torch(t, dtype=dt, layout=layout, device=mesh, memory_config=mc, mesh_mapper=rep)

        # the per-layer q / k / v head tensors and the MLP activation in L1
        l1 = ttnn.L1_MEMORY_CONFIG
        self.q = dram(torch.zeros(1, ROWS, nq, hd), mc=l1)
        self.k = dram(torch.zeros(1, ROWS, nkv, hd), mc=l1)
        self.vv = dram(torch.zeros(1, ROWS, nkv, hd), mc=l1)
        S = MLP_SLICES
        self.act = act_buffer(mesh, I_pad // S, M, slots=S, memory_config=l1)
        u = torch.zeros(1, 64, dtype=torch.int32)
        u[0, 0] = S  # the dense MLP as S "experts" (column slices), ids 0 .. S - 1
        u[0, 1 : 1 + S] = torch.arange(S)
        self.union = dram(u, ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, l1)
        w = torch.zeros(1, M, 32)
        w[0, :, :S] = 1.0  # every row's weight of every slice
        self.weights = dram(w.reshape(1, 32 * M), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, l1)
        self._init_context(dtype, banks, dram)
        # q / k RMSNorm + RoPE + head sharding (+ the K / V cache write) as one program per layer, on the cores
        # (1 + r % 8, r // 8), the KV write at (9, 0)
        head_mc = ttnn.create_sharded_memory_config(
            shape=(TILE, hd),
            core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(8, 3))]),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.prep = DraftQKPrep(
            mesh,
            M,
            nq,
            nkv,
            hd,
            draft.eps,
            head_mc,
            draft.rope.cos_matrix,
            draft.rope.sin_matrix,
            core_offset=(1, 0),
            kv_core=ttnn.CoreCoord(9, 0),
        )
        self.prep_kv_write = DraftKVWrite(mesh, M, nkv, hd, draft.block_size, core_offset=(1, 0))
        host = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().reshape(-1)[:hd]
        self.prep_gamma = [self.prep.gamma(host(L["q_norm"]), host(L["k_norm"])) for L in draft.L]
        self.ba = self._block_attention(draft.window or 0)

    def _block_attention(self, window):
        d, mesh = self.d, self.mesh
        grid = mesh.compute_with_storage_grid_size()
        per_slice = BlockAttention.cores_needed(d.nkv, self.M, 1)
        # clear of the prep cores, the KV write core, the QKV / boundary cores and o_proj's readers (the same program)
        taken = {(c.x, c.y) for c in self.prep.core_list} | {(self.prep.kv_core.x, self.prep.kv_core.y)}
        taken |= {(c.x, c.y) for c in ttnn.corerange_to_cores(self.qkv.cores, row_wise=True)}
        taken |= {(c.x, c.y) for c in ttnn.corerange_to_cores(self.b.cores, row_wise=True)}
        taken |= {(h.x, h.y) for h in self.b.helpers}
        taken |= {(c.x, c.y) for c, *_ in self.o.assignments}
        S = min(BA_SLICES, (grid.x * grid.y - len(taken)) // per_slice)
        if window:  # a window caps the history: no more slices than its chunks give work (cores cost launch time)
            S = min(S, -(-(window // 32 + 2) // BA_MIN_CHUNKS))
        return BlockAttention(
            mesh,
            self.M,
            d.nq,
            d.nkv,
            d.hd,
            d.block_size,
            free_cores(mesh, taken, per_slice * S),
            S,
            uniform=True,
            window=window,
            q_rows=True,
            min_chunks=BA_MIN_CHUNKS,
        )

    def set_window(self, w):
        """The drafter's attention window (0: none); call with no traces captured."""
        self.ba = self._block_attention(w or 0)

    def _attention_oproj(self, i):
        """Layer i's q / k prep + K / V cache write + block attention + o_proj + the attn-site boundary send as one
        program per device: o_proj's readers buffer their weight columns once the KV write signals (after the prep's
        reads); its writers wait for block attention's roots (OPROJ_X_SEM), then notify the boundary sender."""
        d, b = self.d, self.b
        kc, vc = d.caches[i]
        o_noc = [self.mesh.worker_core_from_logical_core(c) for c, *_ in self.o.assignments]

        def fuse_fn(qo, kv_src):
            # block attention reads the block's chunks with the older ones and patches the block's rows in from the
            # head-sharded K / V (signalled once those are written) instead of waiting for the cache write
            coords = self.prep_kv_write.coords
            patch = (kv_src[0], kv_src[1], list(zip(coords[0::2], coords[1::2])))
            kern, cbs, io = self.ba.parts(
                qo,
                kc,
                vc,
                d.t_bound,
                d.page_table,
                None,
                d.hd**-0.5,
                wait=(BA_PREP_SEM, 2),
                patch=patch,
                out_notify=(o_noc, OPROJ_X_SEM),
            )
            return kern, cbs, io, self.ba.notify_order() + o_noc, BA_PREP_SEM, self.ba.semaphores()

        kv = (self.prep_kv_write, kc, vc, d.t_pos, d.page_table)
        pk, pc, ps, pio, (qr, kr, vs) = self.prep.parts(
            self.q, self.k, self.vv, self.prep_gamma[i], d.t_rot, kv=kv, fuse_fn=fuse_fn, sem_base=PREP_SEM_BASE
        )
        oprog, oio = self.o(
            self.ba.out,
            self.w_o[i],
            self.v.partial,
            build_only=True,
            spec_tt=self.ba.Tt,
            x_wait=(OPROJ_X_SEM, self.ba.nkv * self.ba.Tt),
            notify=(b.noc_xy[0], b.noc_xy[1], OPROJ_SEND_SEM),
            extra_defines=[("LIN_R_WAIT_SEM", str(BA_PREP_SEM)), ("LIN_R_WAIT_N", "2")],
        )
        n_o = self.o.cores.num_cores()
        mesh_program = ttnn.MeshProgramDescriptor()
        for coord in b.coords:
            program = ttnn.ProgramDescriptor(
                kernels=list(pk) + list(oprog.kernels),
                semaphores=list(ps)
                + list(oprog.semaphores)
                + [
                    ttnn.SemaphoreDescriptor(OPROJ_SEND_SEM, ttnn.CoreType.WORKER, self.o.cores.merge(b.cores), 0),
                    ttnn.SemaphoreDescriptor(OPROJ_X_SEM, ttnn.CoreType.WORKER, self.o.cores, 0),
                    ttnn.SemaphoreDescriptor(BA_PREP_SEM, ttnn.CoreType.WORKER, self.o.cores, 0),
                ],
                cbs=list(pc) + list(oprog.cbs),
            )
            links = min(b.links, b.chunks)
            for link in range(links):
                sender = b._sender_kernel(
                    coord, self.v.partial, "attn", producers=n_o, sem_id=OPROJ_SEND_SEM, link=link, links=links
                )
                b._add_sender(coord, program, sender, link)
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program
        io = []
        for t in [*pio, qr, kr, vs, *oio]:
            if all(t is not u for u in io):
                io.append(t)
        ttnn.generic_op(io, mesh_program)
        return qr, kr, vs

    # -- context pass ----------------------------------------------------------------------------------------------
    def _init_context(self, dtype, banks, dram):
        """Context pass on streamed ops: the verify pass copies each captured layer output (row tiles) into this
        device's whole-tile slice of the 5 x hidden context features (ctx_buf); fc (row-parallel over the slices) ->
        boundary all-reduce + hidden_norm (zero residual) inside the K / V stream of all 8 layers at once -> one
        k_norm / RoPE over the 16 K heads -> one K / V row write per layer."""
        d, b, mesh, M, H = self.d, self.b, self.mesh, self.M, self.d.hidden
        D = d.D
        tot = d.ctx_in // TILE  # 450 tiles of 5 x hidden
        # whole tiles per device, balanced (113 / 113 / 112 / 112)
        self.ctx_split = []
        base, extra = divmod(tot, D)
        st = 0
        for i in range(D):
            n = base + (1 if i < extra else 0)
            self.ctx_split.append((st, st + n))
            st += n
        self.ctx_kx = max(e - s_ for s_, e in self.ctx_split)
        pages = -(-self.ctx_kx * M * 64 // 2048)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        # DRAM, one bank's shard (contiguous: the fc stream reads it in one piece; L1 is short)
        one_bank = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                (TILE, pages * TILE),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        self.ctx_buf = ttnn.from_torch(
            torch.zeros(flat_shape(H, pages)),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=one_bank,
            mesh_mapper=rep,
        )
        self.g_hidden = b.flat_gamma(_Norm(d.hidden_norm))
        nkv, hd, L = d.nkv, d.hd, d.layers
        self.fc = LinearStream(mesh, self.ctx_kx, H, dtype, readers=2, out_mode=5, rows=M)
        self.kv = LinearStream(
            mesh,
            H // TILE,
            L * 2 * nkv * hd,
            dtype,
            readers=4,
            out_mode=3,
            heads=(0, L * nkv * hd // TILE),
            rows=M,
            prefetch_cols=3,
        )
        self.Kc = dram(torch.zeros(1, ROWS, L * nkv, hd))
        self.Vc = dram(torch.zeros(1, ROWS, L * nkv, hd))
        cache_dir = os.environ.get("TT_CACHE_PATH")
        dt = {ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4", ttnn.bfloat16: "bf16"}[dtype]
        base_name = None if cache_dir is None else f"{cache_dir}/dflash_fused_ctx_{dt}_b{banks}_k{self.ctx_kx}"
        marker = None if base_name is None else Path(base_name + ".built")
        lay = {"fc": None, "kv": None}
        knorm = torch.zeros(1, 1, L * nkv, hd)
        from safetensors.torch import load_file

        w = load_file(str(d.path / "model.safetensors"))
        for l in range(L):
            kn = reverse_permute_1d(w[f"layers.{l}.self_attn.k_norm.weight"]).float()
            knorm[0, 0, l * nkv : (l + 1) * nkv] = kn
        if marker is None or not marker.exists():
            fcT = w["fc.weight"].float().T  # [5 hidden, hidden]
            lay = {"fc": [], "kv": []}
            kh = nkv * hd
            for dv in range(D):
                s_, e = self.ctx_split[dv]
                wf = torch.zeros(self.ctx_kx * TILE, H)
                wf[: (e - s_) * TILE] = fcT[s_ * TILE : e * TILE]
                lay["fc"].append(stream_linear_layout(wf, torch.zeros(H), banks))
                ks, vs, kbs, vbs = [], [], [], []
                for l in range(L):
                    a = f"layers.{l}.self_attn."
                    k = reverse_permute(w[a + "k_proj.weight"], D * nkv, D * nkv * hd, H).float()
                    kb = reverse_permute(w[a + "k_proj.bias"][:, None], D * nkv, D * nkv * hd, 1)[:, 0].float()
                    ks.append(k[dv * kh : (dv + 1) * kh])
                    kbs.append(kb[dv * kh : (dv + 1) * kh])
                    vs.append(w[a + "v_proj.weight"].float()[dv * kh : (dv + 1) * kh])
                    vbs.append(w[a + "v_proj.bias"].float()[dv * kh : (dv + 1) * kh])
                wkv = torch.cat(ks + vs).T.contiguous()  # [H, L * 2 * kh]: K of every layer, then V
                lay["kv"].append(stream_linear_layout(wkv, torch.cat(kbs + vbs), banks))
        self.w_fc = as_stream_tensor(
            mesh, lay["fc"], self.fc.weight_rows(), dtype, None if base_name is None else base_name + "_fc"
        )
        self.w_kv = as_stream_tensor(
            mesh, lay["kv"], self.kv.weight_rows(), dtype, None if base_name is None else base_name + "_kv"
        )
        mark_cache_built(marker)
        # the context K's per-head RMSNorm (k_norm of its layer) + RoPE + head sharding and V's head sharding as one
        # DraftQKPrep program (joint mode: the L * nkv heads as rows)
        self.ctx_prep = DraftQKPrep(mesh, M, L * nkv, 0, hd, d.eps, d.head_mc, d.rope.cos_matrix, d.rope.sin_matrix)
        self.ctx_gamma = self.ctx_prep.gamma_rows(knorm[0, 0])

    def context(self):
        """Context K / V of the rows d.t_cpos marks (features already in ctx_buf, captured by the verify pass)."""
        d, v, b = self.d, self.v, self.b
        self.fc(self.ctx_buf, self.w_fc, v.partial, send=(b, "moe"))
        pending = PendingBoundary(b, None, self.g_hidden, v.partial, "moe", entry="zero", sent=True)
        self.kv(pending, self.w_kv, (self.Kc, self.Kc, self.Vc))
        pending.x.deallocate(True)
        pending.residual_out.deallocate(True)
        kr, k_unused, vs = self.ctx_prep(self.Kc, self.Kc, self.Vc, self.ctx_gamma, d.t_crot)
        k_unused.deallocate(True)
        d.kv_write.write_layers(
            kr, vs, [d.caches[l] for l in range(d.layers)], d.t_cpos, d.page_table, cores=CTX_KV_CORES
        )
        kr.deallocate(True)
        vs.deallocate(True)

    def _load_weights(self, dtype, banks):
        d = self.d
        D, H, hd, nq, nkv, I, I_pad = d.D, d.hidden, d.hd, d.nq, d.nkv, d.inter, self.I_pad
        dt = {ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4", ttnn.bfloat16: "bf16"}[dtype]
        cache_dir = os.environ.get("TT_CACHE_PATH")
        S, rot = MLP_SLICES, DRAFT_BANK_ROT
        base = None if cache_dir is None else f"{cache_dir}/dflash_fused_draft_{dt}_b{banks}_i{I_pad}_s{S}_rot{rot}"
        marker = None if base is None else Path(base + ".built")
        w = None
        if marker is None or not marker.exists():
            from safetensors.torch import load_file

            w = load_file(str(d.path / "model.safetensors"))
        self.w_qkv, self.w_o, self.w_gu, self.w_down = [], [], [], []
        for i in range(d.layers):
            lay = {"qkv": None, "o": None, "gu": None, "down": None}
            if w is not None:
                p = f"layers.{i}."
                a = p + "self_attn."
                q = reverse_permute(w[a + "q_proj.weight"], D * nq, D * nq * hd, H).float()
                k = reverse_permute(w[a + "k_proj.weight"], D * nkv, D * nkv * hd, H).float()
                v = w[a + "v_proj.weight"].float()
                qb = reverse_permute(w[a + "q_proj.bias"][:, None], D * nq, D * nq * hd, 1)[:, 0].float()
                kb = reverse_permute(w[a + "k_proj.bias"][:, None], D * nkv, D * nkv * hd, 1)[:, 0].float()
                vb = w[a + "v_proj.bias"].float()
                o = w[a + "o_proj.weight"].float().T  # [nq_all * hd, H]
                ob = w[a + "o_proj.bias"].float()
                g = w[p + "mlp.gate_proj.weight"].float()  # [I_all, H]
                u = w[p + "mlp.up_proj.weight"].float()
                dn = w[p + "mlp.down_proj.weight"].float().T  # [I_all, H]
                lay = {k_: [] for k_ in lay}
                qh, kh = nq * hd, nkv * hd
                Is, Ir = I_pad // S, I // S  # slice s = real columns [s Ir, (s + 1) Ir), padded at its end to Is
                for dv in range(D):
                    wq = torch.cat([q[dv * qh : (dv + 1) * qh], k[dv * kh : (dv + 1) * kh], v[dv * kh : (dv + 1) * kh]])
                    bq = torch.cat(
                        [qb[dv * qh : (dv + 1) * qh], kb[dv * kh : (dv + 1) * kh], vb[dv * kh : (dv + 1) * kh]]
                    )
                    lay["qkv"].append(stream_linear_layout(wq.T.contiguous(), bq, banks))
                    lay["o"].append(
                        stream_linear_layout(o[dv * qh : (dv + 1) * qh], ob if dv == 0 else torch.zeros_like(ob), banks)
                    )
                    gd, ud = (
                        torch.cat(
                            [
                                torch.nn.functional.pad(t[dv * I + s_ * Ir : dv * I + (s_ + 1) * Ir].T, (0, Is - Ir))
                                for s_ in range(S)
                            ],
                            -1,
                        )
                        for t in (g, u)
                    )
                    dd = torch.cat(
                        [
                            torch.nn.functional.pad(dn[dv * I + s_ * Ir : dv * I + (s_ + 1) * Ir], (0, 0, 0, Is - Ir))
                            for s_ in range(S)
                        ]
                    )
                    gu_s = torch.stack(
                        [
                            torch.cat([gd[:, s_ * Is : (s_ + 1) * Is], ud[:, s_ * Is : (s_ + 1) * Is]], -1)
                            for s_ in range(S)
                        ]
                    )
                    lay["gu"].append(stream_gate_up_layout(gu_s, torch.zeros(S, 2 * Is), banks, rot))
                    lay["down"].append(stream_down_layout(dd.reshape(S, Is, H), torch.zeros(S, H), banks, rot))

            def put(name, rows):
                cache = None if base is None else f"{base}_l{i}_{name}"
                return as_stream_tensor(self.mesh, lay[name], rows, dtype, cache)

            self.w_qkv.append(put("qkv", self.qkv.weight_rows()))
            self.w_o.append(put("o", self.o.weight_rows()))
            self.w_gu.append(put("gu", self.gu.weight_rows(S)))
            self.w_down.append(put("down", self.down.weight_rows(S)))
        mark_cache_built(marker)

    def forward(self):
        """The draft pass on the current inputs (d.t_toks / t_pos / t_rot / t_bound): [last token, 7 drafts] into the
        verify pass's token input."""
        d, v, b, M = self.d, self.v, self.b, self.M
        toks = ttnn.slice(d.t_toks, (0, 0), (1, M))  # the block's M rows only (the L1 embedding is 8 rows)
        emb = ttnn.embedding(
            toks, d.target.embedding_weight, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        toks.deallocate(True)
        emb = ttnn.reshape(emb, (1, 1, M, d.hidden))
        pending = PendingBoundary(b, emb, self.g_ln1[0], entry=True)
        for i in range(d.layers):
            self.qkv(pending, self.w_qkv[i], (self.q, self.k, self.vv))
            res = pending.residual_out
            pending.x.deallocate(True)
            pending.residual.deallocate(True)
            qr, kr, vs = self._attention_oproj(i)
            kr.deallocate(True)
            vs.deallocate(True)
            qr.deallocate(True)
            pending = PendingBoundary(b, res, self.g_ln2[i], v.partial, "attn", sent=True)
            self.mlp_fused(pending, self.union, self.weights, self.w_gu[i], self.act, self.w_down[i], v.partial)
            res = pending.residual_out
            pending.residual.deallocate(True)
            pending.x.deallocate(True)
            gamma = self.g_ln1[i + 1] if i + 1 < d.layers else self.g_final
            pending = PendingBoundary(b, res, gamma, v.partial, "moe", sent=True)
        self.lm(pending, self.lm_w, self.lists)
        pending.x.deallocate(True)
        pending.residual.deallocate(True)
        pending.residual_out.deallocate(True)
        self.gather(self.lists, d.t_toks, v.tokens)
