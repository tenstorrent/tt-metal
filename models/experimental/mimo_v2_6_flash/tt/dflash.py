# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo's DFlash draft model (``dflash/`` beside the checkpoint, reference/dflash.py) as a speculative drafter on one
TP=8 replica: one traced program drafts MAX_DRAFTS tokens for the one request of the lane that speculates, in one
non-causal pass over the block [anchor, mask x (BLOCK - 1)].

Context. DFlash reads the target's residual stream after layers ``target_layer_ids`` (0, 11, 23, 35, 47) at every
committed position. The decode and verify traces copy those layers' outputs into five persistent tap buffers
(``tap``, TtMiMoModel.decode_forward_device); the draft program projects the rows the host names (the last step's
committed slots) through ``fc`` and ``hidden_norm`` into every layer's K / V (k_norm, RoPE at the row's position) and
writes them into the lane's context ring: per layer a plain [RING, 128] K and V tensor per chip (its one KV head),
position p at row p % RING, written by a one-hot matmul scatter (``ring * (1 - P^T 1) + P^T rows``: exact, and no two
rows patch one cache tile concurrently, as paged_update_cache's per-row writers would at consecutive positions).
A prefill chunk writes its last RING rows the same way (``prefill_tap``, ``prefill_context``). ``valid`` marks the
ring rows written since ``reset`` (a new request); the host keeps the ring continuous (generator_vllm.py) and resets
it when it is not.

Block. Each of the 5 layers runs over 32 rows (block rows 0..7, the rest padding) stacked with the 32 context rows
(one qkv matmul serves both: the context's K / V use the layer's own k_proj / v_proj / k_norm): q_norm / k_norm per
head, RoPE (64 of 128 dims, neox pairs, theta 10000: the q / k weights are permuted for rotary_embedding_llama's
adjacent pairs, as the target's are), then attention over [ring, block keys, one sink tile] as two matmuls and a
softmax: one additive bias per layer carries the sink logit (attention_sink_bias), the ring's validity and the
padding. Every block row sees the whole ring (the last RING committed positions) and the whole block: the reference's
window (keys > own position - 1024) differs only in the oldest few positions. The dense SiLU MLP is column / row
parallel with one all-reduce, as is o_proj. The final norm's rows 1..MAX_DRAFTS (moved to rows 0.. by a one-hot
matmul) go through the target's LM head into a GreedyHead: its token row is the drafts.

Inputs per draft (``enqueue``, host-planned): the anchor token, the context scatter P [32, RING] (row r of the taps
to ring row p % RING; zero rows skip), and the RoPE cos / sin of the block and context rows (the query's carry the
softmax scale, which rotary_embedding_llama applies linearly).
"""

import os
import time

import torch

import ttnn

from ..reference.dflash import DFlashConfig, load_dflash_weights
from ..reference.spec_decode import MAX_DRAFTS
from .common import TILE, as_weight, cache_name, compute_config
from .decoder import TtRMSNorm
from .greedy_head import GreedyHead
from .rope import MiMoRotarySetup, meta_rope_permutation, partial_rope_cos_sin
from .spec_chain import SpecChain

NEG = -30000.0  # an additive bias that removes a key from the softmax (bf16 scores stay finite)


class TtDFlashLayer:
    def __init__(self, plan, cfg, weights, index, cache_dir, weight_dtype):
        tp = plan.tp
        tag = f"dflash_tp{tp}_layer{index}"
        nh, nkv, hd, h = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.hidden_size
        assert nh % tp == 0 and nkv % tp == 0 and nkv // tp == 1, "one KV head per chip"
        self.nh_local = nh // tp
        perm = meta_rope_permutation(hd, cfg.rope_dim)
        wqkv = wo = wg = wu = wd = qn = kn = bias = None
        if weights is not None:

            def w(name):
                return weights[f"layers.{index}.{name}"]

            q = w("self_attn.q_proj.weight").view(nh, hd, h)[:, perm]
            k = w("self_attn.k_proj.weight").view(nkv, hd, h)[:, perm]
            v = w("self_attn.v_proj.weight").view(nkv, hd, h)
            nl = self.nh_local
            blocks = []
            for c in range(tp):  # chip c: its q heads, then its K and V head (nlp_create_qkv_heads' order)
                blocks += [q[c * nl : (c + 1) * nl].reshape(-1, h), k[c].reshape(-1, h), v[c].reshape(-1, h)]
            wqkv = torch.cat(blocks).T.contiguous()  # [hidden, tp * (nl + 2) * hd]
            wo = w("self_attn.o_proj.weight").T.contiguous()  # [nh * hd, hidden]: rows by head
            wg = w("mlp.gate_proj.weight").T.contiguous()
            wu = w("mlp.up_proj.weight").T.contiguous()
            wd = w("mlp.down_proj.weight").T.contiguous()
            qn = w("self_attn.q_norm.weight")[perm].reshape(1, 1, 1, hd)
            kn = w("self_attn.k_norm.weight")[perm].reshape(1, 1, 1, hd)
            # Per head: the ring and block keys open (their validity / padding is added per draft), block padding rows
            # and the sink tile closed but for its first column, the sink logit (the zero key's score is 0).
            ring = cfg.window
            bias = torch.zeros(1, nh, TILE, ring + 2 * TILE)
            bias[..., ring + cfg.block : ring + TILE] = NEG
            bias[..., ring + TILE + 1 :] = NEG
            bias[..., ring + TILE] = w("self_attn.attention_sink_bias").float().reshape(1, nh, 1)
        dt = weight_dtype
        self.wqkv = as_weight(wqkv, plan, plan.shard(-1), dt, cache_name(cache_dir, tag, "wqkv"))
        self.wo = as_weight(wo, plan, plan.shard(-2), dt, cache_name(cache_dir, tag, "wo"))
        self.w_gate = as_weight(wg, plan, plan.shard(-1), dt, cache_name(cache_dir, tag, "w_gate"))
        self.w_up = as_weight(wu, plan, plan.shard(-1), dt, cache_name(cache_dir, tag, "w_up"))
        self.w_down = as_weight(wd, plan, plan.shard(-2), dt, cache_name(cache_dir, tag, "w_down"))
        rep = plan.replicate()
        self.q_norm = as_weight(qn, plan, rep, ttnn.bfloat16, cache_name(cache_dir, tag, "q_norm"))
        self.k_norm = as_weight(kn, plan, rep, ttnn.bfloat16, cache_name(cache_dir, tag, "k_norm"))
        self.bias = as_weight(bias, plan, plan.shard(1), ttnn.bfloat16, cache_name(cache_dir, tag, "bias"))
        eps = cfg.eps
        self.input_norm = TtRMSNorm(
            plan, weights and w("input_layernorm.weight"), eps, cache_name(cache_dir, tag, "input_norm")
        )
        self.post_norm = TtRMSNorm(
            plan, weights and w("post_attention_layernorm.weight"), eps, cache_name(cache_dir, tag, "post_norm")
        )
        self.ring_k = self.ring_v = None  # [1, 1, RING, head_dim] per chip (TtDFlash.setup_cache)


class TtDFlash:
    def __init__(self, model, model_path, cache_dir=None, weight_dtype=ttnn.bfloat8_b, load=True):
        """``model``: the replica's TtMiMoModel (embedding, LM head, the decode traces' taps). Loads the draft weights
        from ``model_path``/dflash (``load``; a later replica passes False and reads the tensor cache)."""
        self.model, self.plan = model, model.plan
        cfg = self.cfg = DFlashConfig.from_model_path(model_path)
        plan, tp = self.plan, self.plan.tp
        tag = f"dflash_tp{tp}"
        done = cache_name(cache_dir, tag, "complete")
        warm = done is not None and os.path.exists(done)
        assert load or warm, "cold cache needs the checkpoint"
        weights = None if warm else load_dflash_weights(cfg)
        assert cfg.hidden_size == model.cfg.hidden_size and cfg.block <= TILE and MAX_DRAFTS < cfg.block
        self.ring = cfg.window
        self.tap_layers = list(cfg.target_layer_ids)
        self.layers = [TtDFlashLayer(plan, cfg, weights, i, cache_dir, weight_dtype) for i in range(cfg.num_layers)]
        h = cfg.hidden_size
        fc = None
        if weights is not None:
            # fc [hidden, taps * hidden] -> per tap layer its [hidden, hidden] block, column-parallel.
            full = weights["fc.weight"].T.contiguous()
            fc = [full[k * h : (k + 1) * h] for k in range(len(self.tap_layers))]
        self.fc = [
            as_weight(fc and fc[k], plan, plan.shard(-1), weight_dtype, cache_name(cache_dir, tag, f"fc{k}"))
            for k in range(len(self.tap_layers))
        ]
        self.hidden_norm = TtRMSNorm(
            plan, weights and weights["hidden_norm.weight"], cfg.eps, cache_name(cache_dir, tag, "hidden_norm")
        )
        self.final_norm = TtRMSNorm(
            plan, weights and weights["norm.weight"], cfg.eps, cache_name(cache_dir, tag, "norm")
        )
        mask = None
        if weights is not None:
            mask = weights["mask_embedding"].bfloat16().float().reshape(1, 1, 1, h).expand(1, 1, TILE, h).clone()
            mask[:, :, 0] = 0  # row 0 is the anchor's embedding
        self.mask_rows = as_weight(mask, plan, plan.replicate(), ttnn.bfloat16, cache_name(cache_dir, tag, "mask_rows"))
        del weights, fc
        if done is not None and not warm:
            open(done, "w").close()
        self.rope = MiMoRotarySetup(plan.mesh_device, TILE, cfg.head_dim, cfg.rope_dim, cfg.rope_theta)
        self.scale = cfg.head_dim**-0.5
        self.mm_config = compute_config(ttnn.MathFidelity.HiFi2)
        self.exact_config = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=True, packer_l1_acc=False)
        self.norm_config = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=True, packer_l1_acc=False)
        self.taps = None  # the decode traces' tap buffers (setup_cache)
        self.head = None
        self.inputs = None
        self.trace = None
        self.last_inputs_s = 0.0
        self.debug = None  # a list: eager runs of the draft program append a copy of the block's final normed hidden

    # ------------------------------------------------------------------ persistent buffers
    def _tensor(self, values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            values,
            device=self.plan.mesh_device,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.plan.replicate(),
        )

    def setup_cache(self):
        """The taps, the context rings, the draft program's inputs, constants and GreedyHead: persistent, made before
        any trace capture (after TtMiMoModel.setup_vllm_kv_cache)."""
        assert self.taps is None
        m, cfg = self.model, self.cfg
        b, h, hd, r = m.max_batch, cfg.hidden_size, cfg.head_dim, self.ring
        assert b == TILE, "the taps' rows are the decode slots, one tile"
        self.taps = [self._tensor(torch.zeros(1, 1, b, h)) for _ in self.tap_layers]
        for layer in self.layers:
            layer.ring_k = self._tensor(torch.zeros(1, 1, r, hd))
            layer.ring_v = self._tensor(torch.zeros(1, 1, r, hd))
        self.valid = self._tensor(torch.zeros(1, 1, TILE, r))
        self.ones_kv = self._tensor(torch.ones(1, 1, TILE, hd))
        self.ones_sq = self._tensor(torch.ones(1, 1, TILE, TILE))
        self.zeros_kv = self._tensor(torch.zeros(1, 1, TILE, hd))
        self.zeros_pad = self._tensor(torch.zeros(1, 1, TILE, 2 * TILE))
        sel = torch.zeros(1, 1, TILE, h)
        sel[:, :, 0] = 1
        self.anchor_sel = self._tensor(sel)
        shift = torch.zeros(1, 1, TILE, TILE)
        shift[0, 0, torch.arange(TILE - 1), torch.arange(1, TILE)] = 1  # row i <- row i + 1
        self.shift = self._tensor(shift)
        self.inputs = {
            "tokens": self._tensor(torch.zeros(1, TILE, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "scatter": self._tensor(torch.zeros(1, 1, TILE, r)),
            "cos_q": self._tensor(torch.zeros(1, 1, TILE, hd)),
            "sin_q": self._tensor(torch.zeros(1, 1, TILE, hd)),
            "cos_k": self._tensor(torch.zeros(1, 1, 2 * TILE, hd)),
            "sin_k": self._tensor(torch.zeros(1, 1, 2 * TILE, hd)),
        }
        self.head = GreedyHead(self.plan, m.vocab_local, MAX_DRAFTS, b)
        if m.verify_sampler is not None:
            # The verify's accept trace ends with the chain step, which writes the pass's result page
            # (tt/spec_chain.py); DFlash drafts are host-planned (no chain), so its drafts step, which reads
            # ``heads``, is never replayed.
            m.spec_chain = SpecChain(
                self.plan,
                m.spec_inputs,
                m.verify_sampler.out,
                m.greedy_head.next_tokens,
                [self.head.next_tokens] * MAX_DRAFTS,
            )

    def reset(self):
        """Forget the ring (a new request, or one whose context the ring does not continue)."""
        ttnn.fill(self.valid, 0.0, output_tensor=self.valid)

    # ------------------------------------------------------------------ taps (decode / verify traces, prefill)
    def tap(self, layer_index, y):
        """In the decode and verify traces: keep target layer ``layer_index``'s output rows when DFlash reads it."""
        if self.taps is None or layer_index not in self.tap_layers:
            return
        dst = self.taps[self.tap_layers.index(layer_index)]
        if y.memory_config().is_sharded() or y.memory_config().buffer_type != ttnn.BufferType.DRAM:
            tmp = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.copy(tmp, dst)
            tmp.deallocate(True)
        else:
            ttnn.copy(y, dst)

    def prefill_tap(self, layer_index, x, acc):
        """In a prefill chunk: ``acc`` + fc's block of target layer ``layer_index`` applied to its output rows x
        [1, 1, S, hidden] (column-parallel; None starts the sum). Returns the sum."""
        if layer_index not in self.tap_layers:
            return acc
        part = ttnn.linear(
            x,
            self.fc[self.tap_layers.index(layer_index)],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
        )
        if acc is None:
            return part
        out = ttnn.add(acc, part)
        acc.deallocate(True)
        part.deallocate(True)
        return out

    def _context_hidden(self, acc, prefill):
        """hidden_norm(fc(...)) of the summed column-parallel fc rows (consumes ``acc``)."""
        full = self.plan.all_gather(acc, dim=-1)
        acc.deallocate(True)
        h = self.hidden_norm(full, prefill=prefill)
        full.deallocate(True)
        return h

    def _scatter_matrix(self, rows, positions):
        """Host P [1, 1, rows, RING]: row i of the rows to ring row positions[i] % RING (None or < 0 skips it)."""
        p = torch.zeros(1, 1, rows, self.ring)
        for i, pos in enumerate(positions):
            if pos is not None and pos >= 0:
                p[0, 0, i, pos % self.ring] = 1
        return p

    def _cos_sin(self, positions, scale=1.0):
        pos = torch.tensor([max(int(p), 0) for p in positions])
        cos, sin = partial_rope_cos_sin(pos, self.cfg.head_dim, self.cfg.rope_dim, self.cfg.rope_theta)
        return (cos * scale).reshape(1, 1, -1, cos.shape[-1]), (sin * scale).reshape(1, 1, -1, sin.shape[-1])

    def _scatter(self, ring, rows, pt, keep):
        """ring <- ring * keep + P^T rows (in place)."""
        kept = ttnn.multiply(ring, keep)
        new = ttnn.matmul(pt, rows, dtype=ttnn.bfloat16, compute_kernel_config=self.exact_config)
        ttnn.add(kept, new, output_tensor=ring)
        kept.deallocate(True)
        new.deallocate(True)

    def _mark_valid(self, p):
        """valid <- max(valid, the ring rows P writes) (every row of ``valid`` alike)."""
        ones = self.ones_sq if p.shape[-2] == TILE else self._tensor(torch.ones(1, 1, TILE, p.shape[-2]))
        written = ttnn.matmul(ones, p, dtype=ttnn.bfloat16, compute_kernel_config=self.exact_config)
        if ones is not self.ones_sq:
            ones.deallocate(True)
        ttnn.maximum(self.valid, written, output_tensor=self.valid)
        written.deallocate(True)

    def _keep(self, pt):
        """1 - P^T 1: zero on the ring rows a scatter rewrites."""
        rows = pt.shape[-1]
        ones = self.ones_kv if rows == TILE else self._tensor(torch.ones(1, 1, rows, self.cfg.head_dim))
        hit = ttnn.matmul(pt, ones, dtype=ttnn.bfloat16, compute_kernel_config=self.exact_config)
        if ones is not self.ones_kv:
            ones.deallocate(True)
        keep = ttnn.rsub(hit, 1.0)
        hit.deallocate(True)
        return keep

    def _qkv(self, layer, x):
        xqkv = ttnn.linear(
            x,
            layer.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=layer.nh_local,
            num_kv_heads=1,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        xqkv.deallocate(True)
        return q, k, v

    def _norm_rope(self, t, weight, cos, sin):
        n = ttnn.rms_norm(
            t,
            epsilon=self.cfg.eps,
            weight=weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.norm_config,
        )
        out = ttnn.experimental.rotary_embedding_llama(
            n, cos, sin, self.rope.transformation_mat_prefill, is_decode_mode=False
        )
        n.deallocate(True)
        return out

    def prefill_context(self, acc, start_pos, real_len):
        """After a prefill chunk at ``start_pos`` with ``real_len`` real rows: write the K / V of its last RING rows
        (``acc``: the chunk's summed fc rows, consumed) into every layer's ring. Eager; keyed on the padded length."""
        rows = acc.shape[-2]
        h = self._context_hidden(acc, prefill=True)
        first = max(0, real_len - self.ring)
        positions = [start_pos + i if first <= i < real_len else None for i in range(rows)]
        p = self._tensor(self._scatter_matrix(rows, positions))
        cos, sin = (self._tensor(t) for t in self._cos_sin([start_pos + i for i in range(rows)]))
        pt = ttnn.transpose(p, -2, -1)
        keep = self._keep(pt)
        for layer in self.layers:
            q, k, v = self._qkv(layer, h)
            q.deallocate(True)
            kr = self._norm_rope(k, layer.k_norm, cos, sin)
            k.deallocate(True)
            self._scatter(layer.ring_k, kr, pt, keep)
            self._scatter(layer.ring_v, v, pt, keep)
            kr.deallocate(True)
            v.deallocate(True)
        self._mark_valid(p)
        for t in (h, p, pt, keep, cos, sin):
            t.deallocate(True)

    # ------------------------------------------------------------------ draft program
    def _body(self):
        """The draft program (persistent inputs only): the context step, the block's 5 layers, the LM head and the
        GreedyHead whose token row (rows 0 .. MAX_DRAFTS - 1) is the drafts."""
        m, cfg, inp = self.model, self.cfg, self.inputs
        # Context: hidden_norm(fc(taps)) of the 32 tap rows; the scatter writes the committed ones.
        acc = None
        for k, tap in enumerate(self.taps):
            acc = self.prefill_tap(self.tap_layers[k], tap, acc)
        hctx = self._context_hidden(acc, prefill=False)
        p = inp["scatter"]
        pt = ttnn.transpose(p, -2, -1)
        keep = self._keep(pt)
        self._mark_valid(p)
        open_ = ttnn.multiply(ttnn.subtract(self.valid, 1.0), -NEG)  # 0 on valid ring rows, NEG elsewhere
        ring_bias = ttnn.concat([open_, self.zeros_pad], dim=-1)
        open_.deallocate(True)
        # Block rows: the anchor's embedding in row 0, the mask embedding below it.
        e = ttnn.embedding(inp["tokens"], m.embedding, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        e = ttnn.reshape(e, (1, 1, TILE, cfg.hidden_size))
        anchor = ttnn.multiply(e, self.anchor_sel)
        e.deallocate(True)
        x = ttnn.add(anchor, self.mask_rows)
        anchor.deallocate(True)
        for layer in self.layers:
            hn = layer.input_norm(x)
            stacked = ttnn.concat([hn, hctx], dim=2)  # block rows 0..31, context rows 32..63
            hn.deallocate(True)
            q, k, v = self._qkv(layer, stacked)
            stacked.deallocate(True)
            qb = ttnn.slice(q, (0, 0, 0, 0), (1, layer.nh_local, TILE, cfg.head_dim))
            q.deallocate(True)
            qr = self._norm_rope(qb, layer.q_norm, inp["cos_q"], inp["sin_q"])
            qb.deallocate(True)
            kr = self._norm_rope(k, layer.k_norm, inp["cos_k"], inp["sin_k"])
            k.deallocate(True)
            kv = {}
            for name, t in (("k", kr), ("v", v)):
                kv[name] = (
                    ttnn.slice(t, (0, 0, 0, 0), (1, 1, TILE, cfg.head_dim)),
                    ttnn.slice(t, (0, 0, TILE, 0), (1, 1, 2 * TILE, cfg.head_dim)),
                )
                t.deallocate(True)
            self._scatter(layer.ring_k, kv["k"][1], pt, keep)
            self._scatter(layer.ring_v, kv["v"][1], pt, keep)
            keys = ttnn.concat([layer.ring_k, kv["k"][0], self.zeros_kv], dim=2)
            vals = ttnn.concat([layer.ring_v, kv["v"][0], self.zeros_kv], dim=2)
            for pair in kv.values():
                for t in pair:
                    t.deallocate(True)
            keys_t = ttnn.transpose(keys, -2, -1)
            keys.deallocate(True)
            s = ttnn.matmul(qr, keys_t, dtype=ttnn.bfloat16, compute_kernel_config=self.exact_config)
            qr.deallocate(True)
            keys_t.deallocate(True)
            bias = ttnn.add(layer.bias, ring_bias)
            sb = ttnn.add(s, bias)
            s.deallocate(True)
            bias.deallocate(True)
            probs = ttnn.softmax(sb, dim=-1, numeric_stable=True, compute_kernel_config=self.exact_config)
            sb.deallocate(True)
            o = ttnn.matmul(probs, vals, dtype=ttnn.bfloat16, compute_kernel_config=self.exact_config)
            probs.deallocate(True)
            vals.deallocate(True)
            cat = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            o.deallocate(True)
            x = self._residual(x, cat, layer.wo)
            h2 = layer.post_norm(x)
            g = ttnn.linear(
                h2,
                layer.w_gate,
                dtype=ttnn.bfloat16,
                activation="silu",
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.mm_config,
            )
            u = ttnn.linear(
                h2,
                layer.w_up,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.mm_config,
            )
            h2.deallocate(True)
            gu = ttnn.multiply(g, u)
            g.deallocate(True)
            u.deallocate(True)
            x = self._residual(x, gu, layer.w_down)
        for t in (hctx, pt, keep, ring_bias):
            t.deallocate(True)
        hf = self.final_norm(x)
        x.deallocate(True)
        if self.debug is not None:
            self.debug.append(ttnn.clone(hf))
        hs = ttnn.matmul(self.shift, hf, dtype=ttnn.bfloat16, compute_kernel_config=self.exact_config)
        hf.deallocate(True)
        logits = m._logits(None, h=hs)
        hs.deallocate(True)
        self.head(logits)
        logits.deallocate(True)

    def _residual(self, x, a, w):
        """x + all_reduce(a @ w) (row-parallel ``w``; consumes ``x`` and ``a``)."""
        part = ttnn.linear(
            a, w, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.mm_config
        )
        a.deallocate(True)
        full = self.plan.all_reduce(part)
        out = ttnn.add(x, full)
        x.deallocate(True)
        full.deallocate(True)
        return out

    def capture_trace(self):
        """Compile with an idle run (an all-zero scatter writes nothing), then capture the draft program."""
        assert self.trace is None and self.head is not None
        m = self.model
        self.upload([], 0, 0)
        self._body()
        ttnn.synchronize_device(m.mesh_device)
        trace_id = ttnn.begin_trace_capture(m.mesh_device, cq_id=0)
        self._body()
        ttnn.end_trace_capture(m.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(m.mesh_device)
        self.trace = trace_id

    def release_trace(self):
        if self.trace is not None:
            ttnn.release_trace(self.model.mesh_device, self.trace)
            self.trace = None

    # ------------------------------------------------------------------ host
    def host_inputs(self, context, anchor, start):
        """Host tensors of one draft: ``context`` [(tap row, position)] the committed rows to write into the ring,
        ``anchor`` the last committed token, at position ``start``."""
        cfg = self.cfg
        tokens = torch.zeros(1, TILE, dtype=torch.int32)
        tokens[0, 0] = int(anchor)
        positions = [None] * TILE
        for row, pos in context:
            positions[row] = int(pos)
        block = [start + min(j, cfg.block - 1) for j in range(TILE)]
        cos_q, sin_q = self._cos_sin(block, self.scale)
        cos_b, sin_b = self._cos_sin(block)
        cos_c, sin_c = self._cos_sin([p if p is not None else 0 for p in positions])
        rep = self.plan.replicate()

        def host(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=rep)

        return {
            "tokens": host(tokens, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "scatter": host(self._scatter_matrix(TILE, positions)),
            "cos_q": host(cos_q),
            "sin_q": host(sin_q),
            "cos_k": host(torch.cat([cos_b, cos_c], dim=2)),
            "sin_k": host(torch.cat([sin_b, sin_c], dim=2)),
        }

    def upload(self, context, anchor, start):
        t = time.perf_counter()
        host = self.host_inputs(context, anchor, start)
        self.last_inputs_s = time.perf_counter() - t
        for name, h in host.items():
            ttnn.copy_host_to_device_tensor(h, self.inputs[name])

    def enqueue(self, context, anchor, start):
        """Draft after the replica's last decode or verify step (non-blocking): writes ``context`` (see host_inputs)
        from the taps that step left, then drafts the block after ``anchor`` at ``start``."""
        if self.trace is None:
            raise RuntimeError("a DFlash draft before warmup_model_decode captured the draft trace")
        self.upload(context, anchor, start)
        ttnn.execute_trace(self.model.mesh_device, self.trace, cq_id=0, blocking=False)

    def read_drafts(self):
        """The drafts (blocks until the draft finished)."""
        row = ttnn.to_torch(ttnn.get_device_tensors(self.head.next_tokens)[0]).reshape(-1)
        return [int(t) for t in row[:MAX_DRAFTS]]
