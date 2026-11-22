# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini text model, batch-1 decode on one Blackhole p100a (1x1 mesh).

Mirrors transformers `Inkling2LoopTextModel` + `Inkling2LoopForCausalLM` (modular_inkling2_loop.py) one token at a time.
48 layer applications per token: layers 0-5, layers 6-17 three times, layers 18-23.

Pieces owned here (review-r4 #7): token embedding + embed_norm, per-layer input norms (`attn_norm`, `q/k/v_input_norm`),
the layer wiring (block boundaries, residual add in fp32, bf16 layer exit), loop restart + cross-pass messages,
final attn-res, final norm, unembed, softcap 30, vocab mask (cols >= 200058), on-device argmax, state reset.
Attention = tt/attention.py (T3), MoE = tt/moe.py (T4), attn-res / message / gate / short conv = tt/residual.py (T5).

Tensor form: "row" [1,1,1,C] TILE everywhere on the token path (r vectors: [1,1,r,C]); block-state stacks are
residual.py split stacks (review-r5 #8b).
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.inkling2_mini.convert import cache as wcache
from models.demos.inkling2_mini.tt.attention import DecodePositions, Inkling2Attention
from models.demos.inkling2_mini.tt.moe import Inkling2MoE
from models.demos.inkling2_mini.tt.residual import (
    AttnRes,
    MessageAttnRes,
    MessageGate,
    RowOps,
    ShortConv,
    _torch,
    cache_index,
    split_k_matmul,
    split_k_weight,
    to_device,
)

HIDDEN = 2304
N_LAYERS = 24
LOOP_START, LOOP_END, N_PASSES = 6, 18, 3
GLOBAL_LAYERS = {5, 11, 17, 23}
BOUNDARY_LAYERS = {0, 6, 12, 18}
VOCAB_PAD = 200064
VOCAB_REAL = 200058
SOFTCAP = 30.0
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


class _QKVGate:
    """The q, k, v `MessageGate`s of one loop layer in one call: row i of the inputs goes through gate i
    (same math as residual.MessageGate; the down outputs are block-masked so each row only sees its own gate)."""

    def __init__(self, rows, w, rank=64):
        d = rows.device
        g3 = ("q", "k", "v")
        down = torch.cat([_torch(w[f"msg_gate.{g}.down"]).reshape(2 * HIDDEN, rank) for g in g3], -1)
        self.down = split_k_weight(down, d)  # [2C, 3*rank]
        self.forget = to_device(
            torch.cat([_torch(w[f"msg_gate.{g}.forget"]).reshape(rank, HIDDEN) for g in g3]).float(), d
        )
        self.gain = to_device(torch.cat([_torch(w[f"msg_gate.{g}.gain"]).reshape(rank, HIDDEN) for g in g3]).float(), d)
        mask = torch.zeros(3, 3 * rank)
        for i in range(3):
            mask[i, i * rank : (i + 1) * rank] = 1
        self.mask = to_device(mask, d)
        self.rows = rows

    def __call__(self, h, m):
        """h, m: fp32 [1,1,3,C], rows = (q, k, v) inputs / messages. Returns fp32 [1,1,3,C]."""
        r = self.rows
        x = ttnn.concat([r.rms(h, eps=EPS, dtype=ttnn.bfloat16), r.rms(m, eps=EPS, dtype=ttnn.bfloat16)], dim=-1)
        z = ttnn.typecast(split_k_matmul(x, self.down, r.ckc), ttnn.bfloat16)  # HF: bf16 down_proj output
        z = ttnn.multiply(z, self.mask, dtype=ttnn.float32)  # bf16 * 0/1: exact
        f = ttnn.tanh(r.mm(z, self.forget))
        g = ttnn.tanh(r.mm(z, self.gain))
        return ttnn.addcmul(ttnn.addcmul(h, f, h, value=-1.0), g, m)  # h - f*h + g*m, fp32


class _Layer:
    """Weights of one decoder layer (shared by its loop passes)."""

    def __init__(self, device, rows, L, w):
        self.L = L
        self.is_global = L in GLOBAL_LAYERS
        self.is_loop = LOOP_START <= L < LOOP_END
        self.boundary = L in BOUNDARY_LAYERS
        attn_w = {k[len("attn.") :]: v for k, v in w.items() if k.startswith("attn.")}
        self.attn = Inkling2Attention(device, attn_w, self.is_global)
        self.moe = Inkling2MoE(device, w)
        self.attn_res = AttnRes(rows, w, "attn_res_q", 3 if self.is_global else 1) if L > 0 else None
        self.mlp_res = AttnRes(rows, w, "mlp_res_q", 1)
        self.conv_attn = ShortConv(rows, w, "post_attn_conv")
        self.conv_mlp = ShortConv(rows, w, "post_mlp_conv")
        dram = ttnn.DRAM_MEMORY_CONFIG
        names = ("q_input_norm", "k_input_norm", "v_input_norm") if self.is_global else ("attn_norm",)
        self.norms = [ttnn.to_device(w[n], device, memory_config=dram) for n in names]
        self.qkv_gate = _QKVGate(rows, w) if self.is_loop else None
        self.mlp_gate = MessageGate(rows, w, "mlp") if self.is_loop else None


class Inkling2Model:
    def __init__(
        self, device, root=wcache.DEFAULT_ROOT, max_ctx=2048, kv_dtype=ttnn.bfloat8_b, log=print, layer_roots=None
    ):
        """layer_roots: optional {layer: cache root} override (e.g. BFP8-expert cache for kernel-vs-quant checks)."""
        self.device = device
        self.max_ctx = max_ctx
        self.rows = RowOps(device)
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
        self.unembed = ttnn.to_device(g["unembed"], device, memory_config=dram)
        self.final_res = AttnRes(self.rows, g, "final_res_q", 1)
        self.msg = MessageAttnRes(self.rows, g, "msg_q")
        mask = torch.zeros(1, 1, 1, VOCAB_PAD)
        mask[..., VOCAB_REAL:] = -1.0e4
        self.vocab_mask = ttnn.from_torch(mask, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        del g, emb

        self.layers = []
        for L in range(N_LAYERS):
            w = wcache.load_layer(
                (layer_roots or {}).get(L, root), L, None
            )  # host tensors; modules move them (review-r4 #8)
            self.layers.append(_Layer(device, self.rows, L, w))
            del w
            log(f"layer {L} on device")

        # per-application state, indexed by HF cache index (review-r4 #6)
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

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _row(x, i=0):
        return ttnn.slice(x, [0, 0, i, 0], [1, 1, i + 1, x.shape[-1]])

    def _messages(self, h12_in, h17_out):
        """Cross-pass messages for all 12 loop layers: [(qkv [1,1,3,C], mlp row)] (MessageAttnRes math)."""
        r, q = self.rows, self.msg
        out = q.all_messages(h12_in, h17_out)  # [1,1,48,C], row 4l+j
        return [(r.mm(q.qkv_sel(l), out), r.mm(q.mlp_sel(l), out)) for l in range(LOOP_END - LOOP_START)]

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
        self.phase = pos % ShortConv.PHASES
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

        # attention inputs. Multi-row tensors: row 0/1/2 = q/k/v input.
        if lay.attn_res is None:
            x = h
        else:
            x = lay.attn_res(stack, h, n_sealed + 1, split=False)
        if msgs is not None:
            if not lay.is_global:
                x = ttnn.repeat(x, ttnn.Shape([1, 1, 3, 1]))  # same input in rows 0..2
            x = lay.qkv_gate(x, msgs[0])
        if lay.is_global:
            qn, kn, vn = (self._rms(self._row(x, i), w) for i, w in enumerate(lay.norms))
        elif msgs is None:
            qn = kn = vn = self._rms(x, lay.norms[0])
        else:
            xn = self._rms(x, lay.norms[0])
            qn, kn, vn = (self._row(xn, i) for i in range(3))

        a = lay.attn.decode(qn, attn_st, self.positions, kn, vn)  # bf16 row
        a = ttnn.typecast(lay.conv_attn(a, conv_a, self.phase), ttnn.float32)  # conv returns bf16 (HF .to(input))

        if lay.boundary:
            stack = self.rows.seal(stack, h, n_sealed)
            n_sealed += 1
            resid = a
        else:
            resid = ttnn.add(ttnn.typecast(h, ttnn.float32), a, dtype=ttnn.float32)

        m = lay.mlp_res(stack, resid, n_sealed + 1)[0]
        if msgs is not None:
            m = lay.mlp_gate(m, msgs[1])
        y, _ = lay.moe(m)  # bf16 row
        y = lay.conv_mlp(y, conv_m, self.phase)  # bf16 row
        out = ttnn.add(resid, ttnn.typecast(y, ttnn.float32), dtype=ttnn.bfloat16)  # fp32 add, bf16 out
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
                res["apps"].append(h)

        x = self.final_res(stack, h, n + 1)[0]
        x = self._rms(x, self.final_norm)
        logits = ttnn.linear(x, self.unembed, dtype=ttnn.float32, compute_kernel_config=self.mm_cfg)
        logits = ttnn.multiply(ttnn.tanh(ttnn.multiply(logits, 1.0 / SOFTCAP)), SOFTCAP)
        logits = ttnn.add(logits, self.vocab_mask)
        res["logits"] = logits
        # argmax on ROW_MAJOR: 0.08 ms vs 7 ms on TILE (200064 wide)
        res["token"] = ttnn.argmax(ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True)
        return res
