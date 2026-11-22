# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini text model, batch-1 decode on one Blackhole p100a (1x1 mesh).

Mirrors transformers `Inkling2LoopTextModel` + `Inkling2LoopForCausalLM` (modular_inkling2_loop.py) one token at a time.
48 layer applications per token: layers 0-5, layers 6-17 three times, layers 18-23.

Pieces owned here: token embedding + embed_norm, per-layer input norms (`attn_norm`, `q/k/v_input_norm`),
the layer wiring (block boundaries, residual add in fp32, bf16 layer exit), loop restart + cross-pass messages,
final attn-res, final norm, unembed, softcap 30, vocab mask (cols >= 200058), on-device argmax, state reset.
Attention = tt/attention_fast.py, MoE = tt/moe.py, attn-res / message gate / short conv = tt/fused/.

Tensor form: "row" [1,1,1,C] TILE everywhere on the token path (r vectors: [1,1,r,C]); block-state stacks are
the in-place stack of tt/fused/attnres.py (FusedRowOps).
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.inkling2_mini.convert import cache as wcache
from models.demos.inkling2_mini.tt.attention_fast import DecodePositionsFast as DecodePositions
from models.demos.inkling2_mini.tt.attention_fast import Inkling2AttentionFast as Inkling2Attention
from models.demos.inkling2_mini.tt.dram_sharded import DramShardedLinear
from models.demos.inkling2_mini.tt.fused.attnres import FusedAttnRes, FusedRowOps
from models.demos.inkling2_mini.tt.fused.bank_linear import BankLinear
from models.demos.inkling2_mini.tt.fused.conv import FusedShortConv
from models.demos.inkling2_mini.tt.fused.gate import FusedMessageGate, FusedQKVGate
from models.demos.inkling2_mini.tt.fused.head import FusedLogitsTail
from models.demos.inkling2_mini.tt.fused.messages import FusedMessages
from models.demos.inkling2_mini.tt.fused.norm import FusedRowNorm
from models.demos.inkling2_mini.tt.moe import Inkling2MoE
from models.demos.inkling2_mini.tt.prefetch import PrefetchedLinear
from models.demos.inkling2_mini.tt.residual import RowOps, cache_index

HIDDEN = 2304
N_LAYERS = 24
LOOP_START, LOOP_END, N_PASSES = 6, 18, 3
GLOBAL_LAYERS = {5, 11, 17, 23}
BOUNDARY_LAYERS = {0, 6, 12, 18}
VOCAB_PAD = 200064
VOCAB_REAL = 200058
SOFTCAP = 30.0
UNEMBED_CHUNK = 14336  # columns per DRAM-sharded unembed matmul (7 banks x 64 tiles; fp32 out fits L1)
UNEMBED_SPLIT = 1  # bank-streamed unembed: cores per DRAM-sharded column group (1: 1183 us, 2: 1234 us)
EPS = 1e-6

# (layer, pass) of each of the 48 applications, in call order (= golden hidden_XX order)
APPS = (
    [(L, 1) for L in range(LOOP_START)]
    + [(L, p) for p in range(1, N_PASSES + 1) for L in range(LOOP_START, LOOP_END)]
    + [(L, 1) for L in range(LOOP_END, N_LAYERS)]
)


def golden_app_inputs(hiddens, embed):
    """Teacher-forcing inputs per HF dataflow: app a's input tensor, from the golden layer returns.

    Not simply hidden_{a-1}: passes 2-3 restart from the pre-loop hidden (output of app 5).
    Also returns, per pass p >= 2, the previous pass's (input of L12, output of L17) used for its messages."""
    inputs, msg_keys = [], {}
    pass_first = {1: 6, 2: 18, 3: 30}
    for a, (L, p) in enumerate(APPS):
        if a == 0:
            inputs.append(embed)
        elif L == LOOP_START and p > 1:
            inputs.append(hiddens[5])
        else:
            inputs.append(hiddens[a - 1])
    for p in (2, 3):
        prev = pass_first[p - 1]
        msg_keys[p] = (inputs[prev + (12 - LOOP_START)], hiddens[prev + (LOOP_END - 1 - LOOP_START)])
    return inputs, msg_keys


class _Layer:
    """Weights of one decoder layer (shared by its loop passes)."""

    def __init__(self, device, rows, frows, L, w, prefetcher=None, expert_layout=None):
        self.L = L
        self.is_global = L in GLOBAL_LAYERS
        self.is_loop = LOOP_START <= L < LOOP_END
        self.boundary = L in BOUNDARY_LAYERS
        attn_w = {k[len("attn.") :]: v for k, v in w.items() if k.startswith("attn.")}
        self.attn = Inkling2Attention(device, attn_w, self.is_global, prefetcher)
        self.moe = Inkling2MoE(device, w, layout=expert_layout)
        # residual path: fused kernels (one program each); frows holds the in-place block-state stack
        self.attn_res = FusedAttnRes(frows, w, "attn_res_q", 3 if self.is_global else 1) if L > 0 else None
        self.mlp_res = FusedAttnRes(frows, w, "mlp_res_q", 1)
        self.conv_attn = FusedShortConv(rows, w, "post_attn_conv")
        self.conv_mlp = FusedShortConv(rows, w, "post_mlp_conv")
        dram = ttnn.DRAM_MEMORY_CONFIG
        names = ("q_input_norm", "k_input_norm", "v_input_norm") if self.is_global else ("attn_norm",)
        self.norm = FusedRowNorm(device, [ttnn.to_device(w[n], device, memory_config=dram) for n in names])
        self.qkv_gate = FusedQKVGate(rows, w) if self.is_loop else None
        self.mlp_gate = FusedMessageGate(rows, w, "mlp") if self.is_loop else None


class Inkling2Model:
    def __init__(
        self,
        device,
        root=wcache.DEFAULT_ROOT,
        max_ctx=2048,
        kv_dtype=ttnn.bfloat8_b,
        log=print,
        layer_roots=None,
        prefetch=False,
        head="bank",
        head_fused=True,
    ):
        """layer_roots: optional {layer: cache root} override (e.g. BFP8-expert cache for kernel-vs-quant checks).
        head: unembed as one bank-streamed program ("bank"), DRAM-sharded column chunks ("ds") or the interleaved
        ttnn.linear ("linear"); "bank" and "ds" give the same logits. head_fused: softcap + argmax per unembed
        chunk (tt/fused/head.py) instead of the ttnn op chain. The other values are for A/B runs."""
        assert head in ("bank", "ds", "linear"), head
        self.device = device
        self.max_ctx = max_ctx
        # tensor prefetcher: its L1 pipes must be carved before any program runs on the ring cores
        self.prefetcher = PrefetchedLinear(device, recv_rows=3, ring_kb=768) if prefetch else None
        self.rows = RowOps(device)
        self.frows = FusedRowOps(device)
        dram = ttnn.DRAM_MEMORY_CONFIG
        self.hifi4 = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.mm_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        g = wcache.load_global(root, None)
        emb = g["embed"]
        self.embed = ttnn.to_device(ttnn.reshape(emb, [emb.shape[-2], emb.shape[-1]]), device, memory_config=dram)
        self.embed_norm = ttnn.to_device(g["embed_norm"], device, memory_config=dram)
        self.final_norm = ttnn.to_device(g["final_norm"], device, memory_config=dram)
        # Unembed: DRAM WIDTH_SHARDED column chunks, bank-local readers (tt/dram_sharded.py); 3.9 -> 2.5 ms head.
        if head == "bank":
            # one bank-streamed program for all chunks, bit-equal to the DRAM-sharded chunks below (same in0_block_w,
            # 3 readers per bank); writes the head's raw logits row itself.
            uw = ttnn.to_torch(g["unembed"])
            uw = uw.reshape(uw.shape[-2], uw.shape[-1])
            kt = uw.shape[0] // (32 * 8)  # DramShardedLinear's act_cores = 8
            self.unembed = BankLinear(
                device,
                uw,
                g["unembed"].dtype,
                UNEMBED_CHUNK,
                kb=next(b for b in range(8, 0, -1) if kt % b == 0),
                cores_per_bank=3,
                compute_kernel_config=self.mm_cfg,
                split=UNEMBED_SPLIT,
            )
            del uw
        elif head == "ds":
            self.unembed = DramShardedLinear(
                device,
                g["unembed"],
                dtype=g["unembed"].dtype,
                chunk=UNEMBED_CHUNK,
                workers=3,
                out_dtype=ttnn.float32,
                compute_kernel_config=self.mm_cfg,
            )
        else:
            w = ttnn.to_device(g["unembed"], device, memory_config=dram)
            self.unembed = lambda x: ttnn.linear(x, w, dtype=ttnn.float32, compute_kernel_config=self.mm_cfg)
        # Decode logits tail (tt/fused/head.py): softcap + argmax per unembed chunk, no full-vocab tile ops.
        self.head_tail = None
        if head_fused and isinstance(self.unembed, (DramShardedLinear, BankLinear)):
            self.head_tail = FusedLogitsTail(device, self.unembed, VOCAB_PAD, VOCAB_REAL, SOFTCAP)
        self.final_res = FusedAttnRes(self.frows, g, "final_res_q", 1)
        self.msg = FusedMessages(device, g, "msg_q")  # loop messages + per-layer split, 2 programs per pass
        mask = torch.zeros(1, 1, 1, VOCAB_PAD)
        mask[..., VOCAB_REAL:] = -1.0e4
        self.vocab_mask = ttnn.from_torch(mask, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        del g, emb

        self.overlap = None  # decode weight staging (tt/mk/overlap.py), set by Generator.capture .. release
        self.layers = []
        for L in range(N_LAYERS):
            lroot = (layer_roots or {}).get(L, root)
            w = wcache.load_layer(lroot, L, None)  # host tensors; modules move them
            lay = wcache.layouts(lroot, L).get("experts.w_gate_up")
            self.layers.append(_Layer(device, self.rows, self.frows, L, w, self.prefetcher, lay))
            del w
            log(f"layer {L} on device")

        # per-application state, indexed by HF cache index
        self.states = {}
        for L, p in APPS:
            lay = self.layers[L]
            c = cache_index(L, p)
            self.states[c] = (
                lay.attn.new_state(max_ctx, kv_dtype),
                lay.conv_attn.alloc_state(c),
                lay.conv_mlp.alloc_state(c),
            )
        self.positions = DecodePositions(device, max_ctx)
        self.tokens = ttnn.from_torch(
            torch.zeros(1, 1, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        self._zero_hist = {}
        self.phase = 0
        # allocate the zero sources now: a buffer allocated after a trace capture can alias trace scratch
        self.reset()
        if self.prefetcher is not None:
            self.prefetcher.start()  # stopped by ttnn.close_device (tt.prefetch)

    # ------------------------------------------------------------ helpers
    def _messages(self, h12_in, h17_out):
        """Cross-pass messages for all 12 loop layers: [(qkv [1,1,3,C], mlp row)], with dense twins. Cached
        tensors: the next pass's call overwrites them (each pass consumes its messages first)."""
        return self.msg(h12_in, h17_out)

    def _rms(self, x, w):
        """Weighted RMSNorm (Olmo2 style: fp32 math), result bf16."""
        return self.rows.rms(x, weight=w, eps=EPS, dtype=ttnn.bfloat16)

    def reset(self):
        """Zero every conv history (the only state read unmasked). KV caches need no reset: the position masks
        hide every slot not written in the current sequence, and stale values are finite."""
        for st, ca, cm in self.states.values():
            for t in [*st.k_hist, *st.v_hist, *ca, *cm]:
                key = (tuple(t.shape), t.dtype)
                if key not in self._zero_hist:
                    self._zero_hist[key] = ttnn.from_torch(
                        torch.zeros(list(t.shape)), dtype=t.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
                    )
                ttnn.copy(self._zero_hist[key], t)

    def set_inputs(self, token, pos):
        """Host writes before a step (outside any trace). Also sets the conv ring phase: a traced step only
        replays correctly at positions with the phase it was captured at."""
        self.phase = pos % FusedShortConv.PHASES
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([[token]], dtype=torch.int32), dtype=ttnn.uint32), self.tokens
        )
        self.positions.update(pos)

    # ------------------------------------------------------------ one layer application
    def _app(self, L, p, h, stack, n_sealed, msgs):
        """h: bf16 row (layer input). stack: split stack of n_sealed block states (None if 0). Returns (out bf16 row,
        stack, n_sealed) after this layer."""
        lay = self.layers[L]
        attn_st, conv_a, conv_m = self.states[cache_index(L, p)]

        # attention inputs. Multi-row tensors: row 0/1/2 = q/k/v input. The fused modules use frows.stack
        # (`stack` is only passed through); outputs are per-module buffers, valid until that module runs again.
        if lay.attn_res is None:
            x = h
        else:
            rep = 3 if (msgs is not None and not lay.is_global) else None  # same input in rows 0..2
            # row_out=False: x feeds only fused ops (FusedQKVGate or FusedRowNorm), which read its dense twin
            x = lay.attn_res(stack, h, n_sealed + 1, split=False, repeat_rows=rep, row_out=False)
        if msgs is not None:
            x = lay.qkv_gate(x, msgs[0])
        xs = lay.norm(x)  # one bf16 row per input row (row i uses norm min(i, n_norms - 1))
        qn, kn, vn = xs if len(xs) == 3 else xs * 3

        # resid = [h +] bf16(conv(a)), fp32 (no row form), a = attention output; the fused attention runs the conv
        # inside its o-projection program
        post = (lay.conv_attn, conv_a, self.phase, None if lay.boundary else h)
        resid = lay.attn.decode(qn, attn_st, self.positions, kn, vn, post=post)
        if lay.boundary:
            stack = self.frows.seal(stack, h, n_sealed)  # free: the attn-res above left rms(h) in that row
            n_sealed += 1

        # with messages m feeds only FusedMessageGate (no row form); otherwise the MoE reads its rows
        m = lay.mlp_res(stack, resid, n_sealed + 1, row_out=msgs is None)[0]
        if msgs is not None:
            m = lay.mlp_gate(m, msgs[1])
        y, _ = lay.moe(m)  # bf16 row (L1); conv_mlp reads it from L1
        out = lay.conv_mlp.conv_add(y, conv_m, self.phase, resid, ttnn.bfloat16)  # fp32 add, bf16 out
        return out, stack, n_sealed

    # ------------------------------------------------------------ one token
    def step(self, force=None, collect=False):
        """One decode step on (self.tokens, self.positions). Returns dict with device tensors: logits [1,1,1,200064]
        fp32, token [argmax], and (collect) embed + per-app outputs.

        force (eager only, teacher forcing): {"inputs": [48 torch [C]], "msg_keys": {p: (h12_in, h17_out)}}."""
        dev = self.device
        self.positions.prepare()
        e = ttnn.embedding(self.tokens, self.embed, layout=ttnn.TILE_LAYOUT)
        h = self._rms(ttnn.reshape(e, [1, 1, 1, HIDDEN]), self.embed_norm)
        res = {"embed": h, "apps": []} if collect else {}

        def forced(t):
            row = t.reshape(1, 1, 1, HIDDEN).to(torch.bfloat16)
            return ttnn.from_torch(row, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

        stack, n = None, 0
        pre = None
        h12_in = None
        msgs = None
        for a, (L, p) in enumerate(APPS):
            if L == LOOP_START:
                if p == 1:
                    pre = (h, stack, n)
                else:
                    keys = (h12_in, h)  # previous pass: input of L12, output of L17
                    if force is not None:
                        keys = tuple(forced(t) for t in force["msg_keys"][p])
                    all_msgs = self._messages(*keys)
                    h, stack, n = pre
            if force is not None:
                h = forced(force["inputs"][a])
            if L == 12:
                h12_in = h
            msgs = all_msgs[L - LOOP_START] if (LOOP_START <= L < LOOP_END and p > 1) else None
            h, stack, n = self._app(L, p, h, stack, n, msgs)
            if collect:
                res["apps"].append(ttnn.clone(h))  # h is a module buffer: later loop passes overwrite it

        x = self.final_res(stack, h, n + 1)[0]
        x = self._rms(x, self.final_norm)
        if self.head_tail is not None:
            res["logits"], res["token"] = self.head_tail(x)  # module buffers: the next step overwrites them
            return res
        logits = self.unembed(x)
        logits = ttnn.multiply(ttnn.tanh(ttnn.multiply(logits, 1.0 / SOFTCAP)), SOFTCAP)
        logits = ttnn.add(logits, self.vocab_mask)
        res["logits"] = logits
        # argmax on ROW_MAJOR: 0.08 ms vs 7 ms on TILE (200064 wide)
        res["token"] = ttnn.argmax(ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True)
        return res
