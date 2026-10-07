# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2.6-Flash decoder layer on the 4x8 mesh: RMSNorm -> MiMoAttention -> residual -> RMSNorm -> FFN -> residual.

FFN: layer 0 dense SwiGLU 16384 (TP over mesh columns, all-reduce), layers 1..47 MoE (exact router -> dispatch -> bf16-activation experts -> combine ->
all-reduce of the per-column partial sums over mesh columns). The residual stream is bf16 [1,1,N,4096] per device, rows = independent user groups (DP),
replicated over the 8 columns. Attention is called per user (one prefill chunk of S tokens each); the FFN sees all U*S tokens of the row at once.
"""
import os

import torch

import ttnn
from models.demos.blackhole.mimo_v26.reference import loader as Ld
from models.demos.blackhole.mimo_v26.tt import weights as W
from models.demos.blackhole.mimo_v26.tt.attention import BLOCK, DQK, GA, HIDDEN, NQ, SWA, MiMoAttention, convert_weights
from models.demos.blackhole.mimo_v26.tt.moe_prefill import COLS, EPC, ROWS, MiMoPrefillMoE, global_expert, make_ckc
from models.demos.blackhole.mimo_v26.tt.moe_weights import load_layer_experts
from models.demos.blackhole.mimo_v26.tt.router import MiMoGate

EPS = 1e-6
EXPERT_OP = os.environ.get("MIMO_EXPERT_OP", "mm")  # loop | mm | unified (bf16-activation paths: loop, mm)
ROUTER_MODE = os.environ.get("MIMO_ROUTER_MODE", "exact")


def _k_norm():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


def _k_ffn():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )


class MiMoDecoderLayer:
    def __init__(
        self,
        mesh,
        L,
        ck,
        wc,
        max_pos,
        users=32,
        ring_blocks=10,
        seed=0,
        expert_op=None,
        cf=4,
        wdtype=ttnn.bfloat8_b,
        build_only=False,
        page_table=None,
    ):
        self.mesh, self.L, self.users, self.max_pos = mesh, L, users, max_pos
        self.swa = Ld.is_swa(L)
        self.cfg = SWA if self.swa else GA
        self.expert_op = expert_op or EXPERT_OP
        p = f"model.layers.{L}."
        rep = W.rep_mapper(mesh)
        # ---- FFN weights first (they populate / hit the cache; build_only stops after this)
        if L == 0:
            self.dense = self._dense_weights(ck, wc)
            self.moe = self.gate = None
        else:
            self.moe = self._moe_weights(ck, wc, cf)
            self.dense = None
        if build_only:
            return

        # ---- norms (replicated, row-major [1,1,128,32])
        def norm(n):
            return ttnn.from_torch(
                W.norm_weight(ck, p + n),
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )

        self.w_in, self.w_post = norm("input_layernorm.weight"), norm("post_attention_layernorm.weight")
        # ---- attention
        qkv = Ld.dequant_qkv(
            L, ck.get(p + "self_attn.qkv_proj.weight"), ck.get(p + "self_attn.qkv_proj.weight_scale_inv")
        )
        nq, nk = NQ * DQK, self.cfg.nkv * DQK
        o_w = ck.get(p + "self_attn.o_proj.weight").float()
        sink = ck.get(p + "self_attn.attention_sink_bias") if ck.has(p + "self_attn.attention_sink_bias") else None
        hw = convert_weights(self.cfg, qkv[:nq], qkv[nq : nq + nk], qkv[nq + nk :], o_w, sink)
        nb = ring_blocks if self.swa else max_pos // BLOCK
        g = torch.Generator().manual_seed(seed + L)
        pt = (
            page_table
            if page_table is not None
            else torch.randperm(users * nb, generator=g).reshape(users, nb).to(torch.int32)
        )
        self.attn = MiMoAttention(
            mesh, self.cfg, hw, pt, max_pos, ring_blocks=ring_blocks if self.swa else 0, users=users, wdtype=wdtype
        )
        self.io = {}
        self.check_cap = True

    # ------------------------------------------------------------------ weights
    def _dense_weights(self, ck, wc):
        mesh, L = self.mesh, self.L
        cols = {}

        def mk(i):
            return lambda: cols.setdefault("w", W.load_dense_ffn(ck, L))[i].reshape(1, 1, *cols["w"][i].shape)

        out = []
        for i, (n, dim) in enumerate((("gate", 3), ("up", 3), ("down", 2))):
            out.append(wc.get(L, f"dense_{n}", mk(i), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, W.col_mapper(mesh, dim)))
        return out

    def _moe_weights(self, ck, wc, cf):
        mesh, L = self.mesh, self.L
        gw = ck.get(f"model.layers.{L}.mlp.gate.weight").float()
        gb = ck.get(f"model.layers.{L}.mlp.gate.e_score_correction_bias").float()
        self.gate = (
            None if wc.build_only else MiMoGate(mesh, gw.to(torch.bfloat16), gb, bias_shift=0.0, batch_per_device=None)
        )
        if self.gate is not None and ROUTER_MODE == "exact":
            # TTMoEGate keeps L1-sharded kernel buffers per instance; 47 layers of them exhaust L1 (rmsnorm CB clash at layer 48 build). Only the
            # hilo/plain kernel paths read them, the exact path (fp32 matmul + topk) does not.
            for n in ("tt_bias", "tt_input_indices", "tt_output", "tt_output_indices"):
                ttnn.deallocate(getattr(self.gate, n))
        experts = {}

        def host(l, i):
            def f():
                if "e" not in experts:
                    experts["e"] = load_layer_experts(L)
                ws = [[experts["e"][global_expert(r, c, l)] for c in range(COLS)] for r in range(ROWS)]
                return torch.stack([torch.stack([ws[r][c][i] for c in range(COLS)]) for r in range(ROWS)])

            return f

        wts = {"gate": [], "up": [], "down": []}
        nd = self.expert_op not in ("loop", "mm")
        for l in range(EPC):
            for i, proj in enumerate(("gate", "up", "down")):
                t = wc.get(
                    L, f"exp{l}_{proj}", host(l, i), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, W.row_col_mapper(mesh), squeeze=2
                )
                if t is not None and nd:
                    from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import routed_expert_weight_memory_config

                    t = ttnn.to_memory_config(t, routed_expert_weight_memory_config(mesh, t.shape[-1], True))
                wts[proj].append(t)
        experts.clear()
        if wc.build_only:
            return None
        return MiMoPrefillMoE(
            mesh,
            (wts["gate"], wts["up"], wts["down"]),
            ckc=make_ckc("hifi2", fp32acc=(self.expert_op != "fused"), packer_l1_acc=(self.expert_op != "fused")),
            expert_op=self.expert_op,
            cf=cf,
        )

    # ------------------------------------------------------------------ pieces
    def rms(self, x, w):
        return ttnn.rms_norm(x, epsilon=EPS, weight=w, compute_kernel_config=_k_norm())

    def all_reduce_cols(self, x):
        return ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def ffn_dense(self, x):
        g = ttnn.linear(x, self.dense[0], dtype=ttnn.bfloat16, compute_kernel_config=_k_ffn())
        u = ttnn.linear(x, self.dense[1], dtype=ttnn.bfloat16, compute_kernel_config=_k_ffn())
        h = ttnn.multiply(
            g, u, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)], dtype=ttnn.bfloat16
        )
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        o = ttnn.linear(h, self.dense[2], dtype=ttnn.bfloat16, compute_kernel_config=_k_ffn())
        ttnn.deallocate(h)
        r = self.all_reduce_cols(o)
        ttnn.deallocate(o)
        return r

    def ffn_moe(self, x, route=None):
        """x [1,1,N,4096] bf16 tile (N % 64 == 0). route: optional (weights, indices) device tensors (teacher-forced routing)."""
        N = x.shape[2]
        wts, idx = route if route is not None else self.gate.route(x, ROUTER_MODE)
        x_rm = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), [1, N, HIDDEN])
        if self.check_cap and os.environ.get("MIMO_CHECK_CAP", "1") == "1":
            self.check_capacity(idx, N)
        part = self.moe.forward(x_rm, wts, idx)
        ttnn.deallocate(x_rm)
        out = self.all_reduce_cols(part)
        ttnn.deallocate(part)
        self.last_route = (wts, idx)
        return out

    def check_capacity(self, idx, N):
        """Assert no chip's dispatch load (experts padded to 32 rows) exceeds max_buf for this N (dead/pad rows included). Host sync: debug/smoke only."""
        from models.demos.blackhole.mimo_v26.tt.moe_prefill import NUM_ROUTED

        mb = self.moe.shared(N).max_buf
        devs = ttnn.get_device_tensors(idx)
        for r in range(ROWS):
            ix = ttnn.to_torch(devs[r * COLS]).reshape(N, -1).long()
            cnt = torch.bincount(ix.flatten(), minlength=NUM_ROUTED)
            for c in range(COLS):
                load = sum(-(-int(cnt[global_expert(r, c, l)]) // 32) * 32 for l in range(EPC))
                assert (
                    load <= mb
                ), f"layer {self.L} row {r} col {c}: dispatch load {load} > max_buf {mb} (capacity overflow)"

    def ffn(self, x, route=None):
        return self.ffn_dense(x) if self.L == 0 else self.ffn_moe(x, route)

    # ------------------------------------------------------------------ prefill
    def prefill(self, h, S, U, taps=None, route=None, slot0=0):
        """h [1,1,U*S,4096] bf16 tile (U users of S tokens each, positions 0..S-1, user u uses KV slot u). Returns h_out (same shape).
        taps: dict to receive attn_in/attn_out/ffn_in/ffn_out (device tensors, caller deallocates)."""
        if S not in self.io:
            self.io[S] = self.attn.prefill_io(S)
        io = self.io[S]
        xn = self.rms(h, self.w_in)
        outs = []
        for u in range(U):
            xu = xn if U == 1 else ttnn.slice(xn, [0, 0, u * S, 0], [1, 1, (u + 1) * S, HIDDEN])
            self.attn.prefill_set(io, 0, slot0 + u)
            outs.append(self.attn.prefill_chunk(xu, io))
            if U > 1:
                ttnn.deallocate(xu)
        a = outs[0] if U == 1 else ttnn.concat(outs, dim=2)
        if U > 1:
            for o in outs:
                ttnn.deallocate(o)
        h1 = ttnn.add(h, a, dtype=ttnn.bfloat16)
        fn = self.rms(h1, self.w_post)
        f = self.ffn(fn, route)
        h2 = ttnn.add(h1, f, dtype=ttnn.bfloat16)
        if taps is not None:
            taps.update(attn_in=xn, attn_out=a, ffn_in=fn, ffn_out=f, h1=h1)
        else:
            for t in (xn, a, fn, f, h1):
                ttnn.deallocate(t)
        return h2

    def chunk(self, h, io, taps=None):
        """One prefill chunk of ONE user per mesh row (trace friendly): h [1,1,S,4096]; io = the shared per-kind io dict of tt/prefill.py (row-sharded device
        tensors). No host sync. Returns h_out [1,1,S,4096]."""
        xn = self.rms(h, self.w_in)
        a = self.attn.prefill_chunk(xn, io)
        h1 = ttnn.add(h, a, dtype=ttnn.bfloat16)
        fn = self.rms(h1, self.w_post)
        f = self.ffn(fn)
        h2 = ttnn.add(h1, f, dtype=ttnn.bfloat16)
        if taps is not None:
            taps.update(attn_out=a, ffn_out=f, h1=h1)
        else:
            for t in (xn, a, fn, f, h1):
                ttnn.deallocate(t)
        return h2

    # ------------------------------------------------------------------ decode (32 slots per row)
    def decode(self, h, cur_pos, rope_idx, taps=None, live=None):
        """h [1,1,32,4096] bf16 tile. Dead slots must hold zeros and cur_pos 0."""
        xn = self.rms(h, self.w_in)
        a = self.attn.decode(xn, cur_pos, rope_idx)
        h1 = ttnn.add(h, a, dtype=ttnn.bfloat16)
        fn = self.rms(h1, self.w_post)
        if self.L == 0:
            f = self.ffn_dense(fn)
        else:
            f = self.moe_decode(fn, live)
        h2 = ttnn.add(h1, f, dtype=ttnn.bfloat16)
        if taps is not None:
            taps.update(attn_out=a, ffn_in=fn, ffn_out=f)
        else:
            for t in (xn, a, h1, fn, f):
                ttnn.deallocate(t)
        return h2

    def moe_decode(self, x, live=None):
        """Decode MoE for the 32 slots of a mesh row. live: persistent device live-slot mask [1,1,64,1] (tt/decode.py LiveMask): traced path through
        tt/moe_decode.MiMoMoEDecode (router -> dead-slot routing spread -> dispatch -> experts -> combine -> all-reduce, no host reads). Without a mask the old
        untraceable path (zero pad rows, no spread, optional capacity check) is used."""
        if live is not None:
            if getattr(self, "moe_dec", None) is None:
                from models.demos.blackhole.mimo_v26.tt.moe_decode import MiMoMoEDecode

                self.moe_dec = MiMoMoEDecode(self.mesh, self.gate, self.moe)
            return self.moe_dec.forward(x, live)
        B = x.shape[2]
        npad = 64 - B
        # route the B real slots with the gate; the npad padding rows get spread routing (round-robin over all experts) so they cannot pile onto the experts a
        # zero hidden state would pick and overflow one chip's dispatch buffer
        wts, idx = self.gate.route(x, ROUTER_MODE)
        from models.demos.blackhole.mimo_v26.tt.moe_prefill import spread_pad_routing

        pi, pw = spread_pad_routing(ROWS, npad)
        rowmap = ttnn.ShardTensor2dMesh(self.mesh, self.mesh.shape, dims=(0, None))
        pi_t = ttnn.from_torch(
            pi.reshape(ROWS, npad, 1, 1, -1).to(torch.int32),
            device=self.mesh,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rowmap,
        )
        pw_t = ttnn.from_torch(
            pw.reshape(ROWS, npad, 1, 1, -1),
            device=self.mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rowmap,
        )
        pi_t, pw_t = ttnn.reshape(pi_t, [npad, 1, 1, -1]), ttnn.reshape(pw_t, [npad, 1, 1, -1])
        idx = ttnn.concat([idx, pi_t], dim=0)
        wts = ttnn.concat([wts, pw_t], dim=0)
        xp = ttnn.pad(x, [(0, 0), (0, 0), (0, npad), (0, 0)], 0.0)
        out = self.ffn_moe(xp, route=(wts, idx))
        ttnn.deallocate(xp)
        o = ttnn.slice(out, [0, 0, 0, 0], [1, 1, B, HIDDEN])
        ttnn.deallocate(out)
        return o
