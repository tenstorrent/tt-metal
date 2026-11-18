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

Block. Each of the 5 layers runs over 32 rows (block rows 0..7, the rest padding): q_norm / k_norm per head, RoPE
(64 of 128 dims, neox pairs, theta 10000: the q / k weights are permuted for rotary_embedding_llama's adjacent pairs,
as the target's are), then a non-causal decode SDPA of one user whose query heads are the (block row, head) pairs, over the keys [ring,
block, zero rows], with the layer's sinks and one mask per draft (the ring's validity, the block's padding). The context rows' K / V use the layer's own k_proj / v_proj /
k_norm. Every block row sees the whole ring (the last RING committed positions) and the whole block: the reference's
window (keys > own position - 1024) differs only in the oldest few positions. Every projection is a DRAM-sharded
matmul (32 rows); o_proj and the dense SiLU MLP's down projection are row-parallel with one all-reduce each. The final
norm's rows 1..MAX_DRAFTS (moved to rows 0.. by a one-hot matmul) go through the target's LM head into a GreedyHead:
its token row is the drafts.

Draft groups (MIMO_SPEC_GROUPS=2): one program drafts for a lane's two requests, group g's block in rows
g * block .. (its own decode-SDPA user over its own ring rows g * RING ..), its drafts in GreedyHead rows
g * MAX_DRAFTS ..; each group's ring is its request's (the adapter keeps a request in one group).

Inputs per draft: one packed uint32 row (reference.spec_chain.DFlashLayout) with the anchor token, the RoPE
positions of the queries, the block and the context rows (rows of the decode RoPE tables), and each context row's ring
row (the scatter P [32, RING] is rows of an identity table; a skipped row reads its zero row). The host uploads it
(``enqueue``, a host-planned draft), or the verify's accept trace writes it from what the device committed
(``enqueue_chained``, tt/spec_chain.py), and the draft program ends with the chain's drafts step, which copies the
drafts into the pass's result page.
"""

import time

import torch

import ttnn

from ..reference.dflash import DFlashConfig, load_dflash_weights
from ..reference.spec_chain import DFlashLayout, dflash_inputs
from ..reference.spec_decode import DRAFT_GROUPS, MAX_DRAFTS
from .common import TILE, DramShardedMatmul, as_weight, cache_fill, cache_name, compute_config
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
        self.nh_local = nl = nh // tp
        inter = cfg.intermediate_size // tp
        perm = meta_rope_permutation(hd, cfg.rope_dim)
        mm_config = compute_config(ttnn.MathFidelity.HiFi2)
        # Every projection reads its weight once per draft (32 rows): DRAM-sharded matmuls, one reader core per bank.
        self.qkv_mm = DramShardedMatmul(plan, h, (nl + 2) * hd, compute_kernel_config=mm_config)
        self.kv_mm = DramShardedMatmul(plan, h, 2 * hd, compute_kernel_config=mm_config)
        self.o_mm = DramShardedMatmul(plan, nl * hd, h, compute_kernel_config=mm_config)
        self.up_mm = DramShardedMatmul(plan, h, inter, compute_kernel_config=mm_config)
        self.down_mm = DramShardedMatmul(plan, inter, h, compute_kernel_config=mm_config)
        wqkv = wkv = wo = wg = wu = wd = qn = kn = sinks = None
        if weights is not None:

            def w(name):
                return weights[f"layers.{index}.{name}"]

            q = w("self_attn.q_proj.weight").view(nh, hd, h)[:, perm]
            k = w("self_attn.k_proj.weight").view(nkv, hd, h)[:, perm]
            v = w("self_attn.v_proj.weight").view(nkv, hd, h)
            qkv, kv = [], []
            for c in range(tp):  # chip c: its q heads, then its K and V head (nlp_create_qkv_heads' order)
                qkv += [q[c * nl : (c + 1) * nl].reshape(-1, h), k[c], v[c]]
                kv += [k[c], v[c]]
            wqkv = torch.cat(qkv).T.contiguous()  # [hidden, tp * (nl + 2) * hd]
            wkv = torch.cat(kv).T.contiguous()  # [hidden, tp * 2 * hd]: the context rows' K and V
            wo = w("self_attn.o_proj.weight").T.contiguous()  # [nh * hd, hidden]: rows by head
            wg = w("mlp.gate_proj.weight").T.contiguous()
            wu = w("mlp.up_proj.weight").T.contiguous()
            wd = w("mlp.down_proj.weight").T.contiguous()
            qn = w("self_attn.q_norm.weight")[perm].reshape(1, 1, 1, hd)
            kn = w("self_attn.k_norm.weight")[perm].reshape(1, 1, 1, hd)
            # The attention's queries are the (block row, head) pairs of the chip, row r * heads + h (TtDFlash._body).
            # The SDPA kernel scales the sink logit with the scores: pre-divided, in column 0 of a tile row (the
            # target's SWA decode sinks).
            s = (w("self_attn.attention_sink_bias").float() * hd**0.5).reshape(tp, 1, nl).expand(tp, cfg.block, nl)
            sinks = torch.nn.functional.pad(s.reshape(tp * cfg.block * nl, 1), (0, TILE - 1))
        dt = weight_dtype

        def sharded(t, mapper, mm, name):
            return as_weight(
                t, plan, mapper, dt, cache_name(cache_dir, tag, name), memory_config=mm.weight_memory_config
            )

        self.wqkv = sharded(wqkv, plan.shard(-1), self.qkv_mm, "wqkv_ds")
        self.wkv = sharded(wkv, plan.shard(-1), self.kv_mm, "wkv_ds")
        self.wo = sharded(wo, plan.shard(-2), self.o_mm, "wo_ds")
        self.w_gate = sharded(wg, plan.shard(-1), self.up_mm, "w_gate_ds")
        self.w_up = sharded(wu, plan.shard(-1), self.up_mm, "w_up_ds")
        self.w_down = sharded(wd, plan.shard(-2), self.down_mm, "w_down_ds")
        rep = plan.replicate()
        self.q_norm = as_weight(qn, plan, rep, ttnn.bfloat16, cache_name(cache_dir, tag, "q_norm"))
        self.k_norm = as_weight(kn, plan, rep, ttnn.bfloat16, cache_name(cache_dir, tag, "k_norm"))
        self.sinks = as_weight(sinks, plan, plan.shard(0), ttnn.bfloat16, cache_name(cache_dir, tag, "sinks"))
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
        with cache_fill(cache_dir, tag) as need:
            assert load or not need, "cold cache needs the checkpoint"
            weights = load_dflash_weights(cfg) if need else None
            assert cfg.hidden_size == model.cfg.hidden_size and cfg.block <= TILE and MAX_DRAFTS < cfg.block
            self.ring = cfg.window
            self.tap_layers = list(cfg.target_layer_ids)
            self.layers = [TtDFlashLayer(plan, cfg, weights, i, cache_dir, weight_dtype) for i in range(cfg.num_layers)]
            h = cfg.hidden_size
            fc = full = None
            if weights is not None:
                # fc [hidden, taps * hidden] -> per tap layer its [hidden, hidden] block, column-parallel.
                full = weights["fc.weight"].T.contiguous()
                fc = [full[k * h : (k + 1) * h] for k in range(len(self.tap_layers))]
            self.fc = [
                as_weight(fc and fc[k], plan, plan.shard(-1), weight_dtype, cache_name(cache_dir, tag, f"fc{k}"))
                for k in range(len(self.tap_layers))
            ]
            # The draft's 32 tap rows go through all of fc in one DRAM-sharded matmul.
            taps_width = len(self.tap_layers) * h
            self.fc_mm = DramShardedMatmul(
                plan, taps_width, h // tp, compute_kernel_config=compute_config(ttnn.MathFidelity.HiFi2)
            )
            self.fc_all = as_weight(
                full if weights is not None else None,
                plan,
                plan.shard(-1),
                weight_dtype,
                cache_name(cache_dir, tag, "fc_ds"),
                memory_config=self.fc_mm.weight_memory_config,
            )
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
            self.mask_rows = as_weight(
                mask, plan, plan.replicate(), ttnn.bfloat16, cache_name(cache_dir, tag, "mask_rows")
            )
            del weights, fc
            full = None
        self.rope = MiMoRotarySetup(plan.mesh_device, TILE, cfg.head_dim, cfg.rope_dim, cfg.rope_theta)
        self.scale = cfg.head_dim**-0.5
        self.mm_config = compute_config(ttnn.MathFidelity.HiFi2)
        self.sdpa_config = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=True, packer_l1_acc=False)
        # The block's attention is one decode-SDPA user whose query heads are the (block row, head) pairs, all on the
        # chip's one KV head: the keys (the ring, the block's keys and padding: KEYS rows, a whole number of K chunks)
        # split over the cores.
        self.keys = self.ring + 4 * TILE
        self.queries = cfg.block * self.layers[0].nh_local
        self.sdpa_program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=self.queries,
            k_chunk_size=4 * TILE,
            exp_approx_mode=False,
        )
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
        b, h, hd, r, groups = m.max_batch, cfg.hidden_size, cfg.head_dim, self.ring, DRAFT_GROUPS
        assert b == TILE, "the taps' rows are the decode slots, one tile"
        assert groups * cfg.block <= TILE and groups * MAX_DRAFTS <= 8, "the groups' blocks and drafts fit a tile"
        self.taps = [self._tensor(torch.zeros(1, 1, b, h)) for _ in self.tap_layers]
        for layer in self.layers:  # group g's ring: rows g * RING .. (g + 1) * RING - 1
            layer.ring_k = self._tensor(torch.zeros(1, 1, groups * r, hd))
            layer.ring_v = self._tensor(torch.zeros(1, 1, groups * r, hd))
        self.valid = self._tensor(torch.zeros(1, 1, TILE, groups * r))
        # reset(g): valid <- valid * keep[g], zero on group g's ring rows.
        self.keep = []
        for g in range(groups):
            keep = torch.ones(1, 1, TILE, groups * r)
            keep[..., g * r : (g + 1) * r] = 0
            self.keep.append(self._tensor(keep))
        self.ones_kv = self._tensor(torch.ones(1, 1, TILE, hd))
        self.ones_sq = self._tensor(torch.ones(1, 1, TILE, TILE))
        self.zeros_kv = self._tensor(torch.zeros(1, 1, self.keys - r - TILE, hd))
        # The attention mask's columns after the ring, per group: its block's rows open, the other rows (another
        # group's block, the padding) and the zero rows closed.
        self.mask_tail = []
        for g in range(groups):
            tail = torch.full((1, 1, TILE, self.keys - r), NEG)
            tail[..., g * cfg.block : (g + 1) * cfg.block] = 0
            self.mask_tail.append(self._tensor(tail))
        sel = torch.zeros(1, 1, TILE, h)
        sel[:, :, [g * cfg.block for g in range(groups)]] = 1  # the groups' anchor rows
        self.anchor_sel = self._tensor(sel)
        not_anchor = self._tensor(1 - sel)
        self.mask_block = ttnn.multiply(self.mask_rows, not_anchor)  # the mask embedding in every other row
        not_anchor.deallocate(True)
        shift = torch.zeros(1, 1, TILE, TILE)
        for g in range(groups):  # group g's draft k (block row g * block + 1 + k) to row g * MAX_DRAFTS + k
            for k in range(MAX_DRAFTS):
                shift[0, 0, g * MAX_DRAFTS + k, g * cfg.block + 1 + k] = 1
        self.shift = self._tensor(shift)
        # The draft's cos / sin are rows of the decode RoPE tables (the SWA theta, 10000, and 64 rotated dims of the
        # target's 192: their first 128 dims are DFlash's), its context scatter rows of an identity table (the last
        # row zeros: a row that writes nothing); one packed upload of indices (``host_inputs``) drives them.
        tables = m.rope_tables
        assert tables is not None, "DFlash reads its cos / sin from the decode RoPE tables (MIMO_DEVICE_ROPE=1)"
        swa = m.rope_setups["swa"]
        assert (swa.theta, swa.rope_dim) == (cfg.rope_theta, cfg.rope_dim) and swa.head_dim >= hd
        self.rope_positions = tables[0].shape[0]
        self.rope_tails = {}
        if tables[0].shape[-1] < hd:  # compact tables: the rotated dims, then the constant 1 (cos) / 0 (sin)
            for n in (TILE, groups * self.queries):
                tail = (torch.ones(1, 1, n, hd - tables[0].shape[-1]), torch.zeros(1, 1, n, hd - tables[0].shape[-1]))
                self.rope_tails[n] = [self._tensor(t) for t in tail]
        eye = torch.zeros(groups * r + 1, groups * r)
        eye[: groups * r] = torch.eye(groups * r)
        self.eye = self._tensor(eye, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.layout = DFlashLayout(r, cfg.block, self.queries // cfg.block, self.rope_positions - 1, groups)
        # tokens, query / block / context RoPE rows, ring rows
        self.packed = [TILE, groups * self.queries, TILE, TILE, TILE]
        assert sum(self.packed) == self.layout.words
        self.inputs = {
            "packed": self._tensor(
                torch.zeros(1, self.layout.words, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
            )
        }
        self.head = GreedyHead(self.plan, m.vocab_local, groups * MAX_DRAFTS, b)
        if m.verify_sampler is not None:
            # The verify's accept trace ends with the chain step, which writes the pass's result page and, when a
            # draft chains behind the pass, this draft's packed upload; the draft program ends with the drafts step
            # (tt/spec_chain.py).
            m.spec_chain = SpecChain(
                self.plan,
                m.spec_inputs,
                m.verify_sampler.out,
                m.greedy_head.next_tokens,
                [self.head.next_tokens],
                dflash=(self.inputs["packed"], self.layout),
            )

    def reset(self, group=0):
        """Forget draft group ``group``'s ring (a new request, or one whose context the ring does not continue)."""
        ttnn.multiply(self.valid, self.keep[group], output_tensor=self.valid)

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

    def _scatter_matrix(self, rows, positions, group):
        """Host P [1, 1, rows, groups * RING]: row i of the rows to ring row group * RING + positions[i] % RING (None or
        < 0 skips it)."""
        p = torch.zeros(1, 1, rows, DRAFT_GROUPS * self.ring)
        for i, pos in enumerate(positions):
            if pos is not None and pos >= 0:
                p[0, 0, i, group * self.ring + pos % self.ring] = 1
        return p

    def _cos_sin(self, positions):
        pos = torch.tensor([max(int(p), 0) for p in positions])
        cos, sin = partial_rope_cos_sin(pos, self.cfg.head_dim, self.cfg.rope_dim, self.cfg.rope_theta)
        return cos.reshape(1, 1, -1, cos.shape[-1]), sin.reshape(1, 1, -1, sin.shape[-1])

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

    def _split_kv(self, kv, rows):
        """[1, 1, rows, 2 * head_dim] -> K, V [1, 1, rows, head_dim] (consumes ``kv``)."""
        hd = self.cfg.head_dim
        k = ttnn.slice(kv, (0, 0, 0, 0), (1, 1, rows, hd))
        v = ttnn.slice(kv, (0, 0, 0, hd), (1, 1, rows, 2 * hd))
        kv.deallocate(True)
        return k, v

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

    def prefill_context(self, acc, start_pos, real_len, group=0):
        """After a prefill chunk at ``start_pos`` with ``real_len`` real rows: write the K / V of its last RING rows
        (``acc``: the chunk's summed fc rows, consumed) into every layer's ring of draft group ``group``. Eager; keyed
        on the padded length."""
        rows = acc.shape[-2]
        h = self._context_hidden(acc, prefill=True)
        first = max(0, real_len - self.ring)
        positions = [start_pos + i if first <= i < real_len else None for i in range(rows)]
        p = self._tensor(self._scatter_matrix(rows, positions, group))
        cos, sin = (self._tensor(t) for t in self._cos_sin([start_pos + i for i in range(rows)]))
        pt = ttnn.transpose(p, -2, -1)
        keep = self._keep(pt)
        for layer in self.layers:
            w = ttnn.to_memory_config(layer.wkv, ttnn.DRAM_MEMORY_CONFIG)
            kv = ttnn.linear(
                h, w, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.mm_config
            )
            w.deallocate(True)
            k, v = self._split_kv(kv, rows)
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
        GreedyHead whose token row is the drafts (group g's at rows g * MAX_DRAFTS ..)."""
        m, cfg, groups, r = self.model, self.cfg, DRAFT_GROUPS, self.ring
        tokens, idx_q, idx_b, idx_c, slots = self._unpack()
        cos_q, sin_q = self._rope_rows(idx_q)
        cos_b, sin_b = self._rope_rows(idx_b)
        cos_c, sin_c = self._rope_rows(idx_c)
        p = ttnn.reshape(
            ttnn.embedding(slots, self.eye, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16), (1, 1, TILE, groups * r)
        )
        for t in (idx_q, idx_b, idx_c, slots):
            t.deallocate(True)
        # Context: hidden_norm(fc(taps)) of the 32 tap rows; the scatter writes the committed ones.
        taps = ttnn.concat(self.taps, dim=-1)
        acc = self.fc_mm(taps, self.fc_all)
        taps.deallocate(True)
        hctx = self._context_hidden(acc, prefill=False)
        pt = ttnn.transpose(p, -2, -1)
        keep = self._keep(pt)
        self._mark_valid(p)
        closed = ttnn.multiply(ttnn.subtract(self.valid, 1.0), -NEG)  # 0 on valid ring rows, NEG elsewhere
        masks = []
        for g in range(groups):  # group g's user: its ring rows, its block rows; every query alike
            ring = closed if groups == 1 else ttnn.slice(closed, (0, 0, 0, g * r), (1, 1, TILE, (g + 1) * r))
            full = ttnn.concat([ring, self.mask_tail[g]], dim=-1)
            if ring is not closed:
                ring.deallocate(True)
            masks.append(ttnn.concat([full] * (self.queries // TILE), dim=2))
            full.deallocate(True)
        closed.deallocate(True)
        mask = masks[0] if groups == 1 else ttnn.concat(masks, dim=0)  # [groups, 1, queries, keys]
        if groups > 1:
            for t in masks:
                t.deallocate(True)
        # Block rows: each group's anchor embedding in its first row, the mask embedding in the others.
        e = ttnn.embedding(tokens, m.embedding, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tokens.deallocate(True)
        e = ttnn.reshape(e, (1, 1, TILE, cfg.hidden_size))
        anchor = ttnn.multiply(e, self.anchor_sel)
        e.deallocate(True)
        x = ttnn.add(anchor, self.mask_block)
        anchor.deallocate(True)
        for layer in self.layers:
            hn = layer.input_norm(x)
            xqkv = layer.qkv_mm(hn, layer.wqkv)
            hn.deallocate(True)
            hd, nl = cfg.head_dim, layer.nh_local
            # Queries: the groups' block rows' q heads as rows (g * block + r) * heads + h; the block's K / V rows.
            qrows = ttnn.slice(xqkv, (0, 0, 0, 0), (1, 1, groups * cfg.block, nl * hd))
            q = ttnn.reshape(qrows, (1, 1, groups * self.queries, hd))
            qrows.deallocate(True)
            kb = ttnn.slice(xqkv, (0, 0, 0, nl * hd), (1, 1, TILE, (nl + 1) * hd))
            vb = ttnn.slice(xqkv, (0, 0, 0, (nl + 1) * hd), (1, 1, TILE, (nl + 2) * hd))
            xqkv.deallocate(True)
            kc, vc = self._split_kv(layer.kv_mm(hctx, layer.wkv), TILE)
            qr = self._norm_rope(q, layer.q_norm, cos_q, sin_q)
            kbr = self._norm_rope(kb, layer.k_norm, cos_b, sin_b)
            kcr = self._norm_rope(kc, layer.k_norm, cos_c, sin_c)
            for t in (q, kb, kc):
                t.deallocate(True)
            self._scatter(layer.ring_k, kcr, pt, keep)
            self._scatter(layer.ring_v, vc, pt, keep)
            kcr.deallocate(True)
            vc.deallocate(True)
            keys = self._user_keys(layer.ring_k, kbr)
            vals = self._user_keys(layer.ring_v, vb)
            kbr.deallocate(True)
            vb.deallocate(True)
            if groups > 1:  # [1, groups, queries, D]: one decode user per group
                qg = ttnn.reshape(qr, (1, groups, self.queries, hd))
                qr.deallocate(True)
                qr = qg
            attn = ttnn.transformer.scaled_dot_product_attention_decode(
                qr,
                keys,
                vals,
                is_causal=False,
                attn_mask=mask,
                attention_sink=layer.sinks,
                scale=self.scale,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.sdpa_program,
                compute_kernel_config=self.sdpa_config,
            )
            for t in (qr, keys, vals):
                t.deallocate(True)
            # [1, groups, rows * heads, D] -> each block row's heads concatenated, padded back to the tile's 32 rows
            rows = ttnn.reshape(attn, (1, 1, groups * cfg.block, nl * hd))
            attn.deallocate(True)
            cat = ttnn.pad(rows, padding=((0, 0), (0, 0), (0, TILE - groups * cfg.block), (0, 0)), value=0.0)
            rows.deallocate(True)
            x = self._residual(x, cat, layer.o_mm, layer.wo)
            h2 = layer.post_norm(x)
            gate = layer.up_mm(h2, layer.w_gate)
            up = layer.up_mm(h2, layer.w_up)
            h2.deallocate(True)
            act = ttnn.mul(
                gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            gate.deallocate(True)
            up.deallocate(True)
            x = self._residual(x, act, layer.down_mm, layer.w_down)
        for t in (hctx, p, pt, keep, mask, cos_q, sin_q, cos_b, sin_b, cos_c, sin_c):
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
        if m.spec_chain is not None:
            m.spec_chain.drafts()

    def _user_keys(self, ring, block):
        """The attention's keys (or values) [groups, 1, KEYS, head_dim]: per group its ring rows, the block's 32 rows
        (its mask opens the group's own), the zero rows."""
        r, users = self.ring, []
        for g in range(DRAFT_GROUPS):
            rows = (
                ring if DRAFT_GROUPS == 1 else ttnn.slice(ring, (0, 0, g * r, 0), (1, 1, (g + 1) * r, ring.shape[-1]))
            )
            users.append(ttnn.concat([rows, block, self.zeros_kv], dim=2))
            if rows is not ring:
                rows.deallocate(True)
        if DRAFT_GROUPS == 1:
            return users[0]
        out = ttnn.concat(users, dim=0)
        for t in users:
            t.deallocate(True)
        return out

    def _unpack(self):
        """The packed upload's parts (uint32 [1, n] each): tokens, the query / block / context RoPE positions, the
        context rows' ring rows."""
        out, at = [], 0
        for n in self.packed:
            out.append(ttnn.slice(self.inputs["packed"], (0, at), (1, at + n)))
            at += n
        return out

    def _rope_rows(self, idx):
        """DFlash's cos / sin [1, 1, n, head_dim] at the positions ``idx`` (uint32 [1, n]): rows of the decode RoPE
        tables, cut to head_dim (or a compact table's rotated dims followed by the constant tail)."""
        n, hd = idx.shape[-1], self.cfg.head_dim
        out = []
        for k, table in enumerate(self.model.rope_tables[:2]):
            width = table.shape[-1]
            row = ttnn.reshape(
                ttnn.embedding(idx, table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16), (1, 1, n, width)
            )
            if width > hd:
                cut = ttnn.slice(row, (0, 0, 0, 0), (1, 1, n, hd))
            elif width < hd:
                cut = ttnn.concat([row, self.rope_tails[n][k]], dim=-1)
            else:
                cut = row
            if cut is not row:
                row.deallocate(True)
            out.append(cut)
        return out

    def _residual(self, x, a, mm, w):
        """x + all_reduce(a @ w) (row-parallel ``w`` through DramShardedMatmul ``mm``; consumes ``x`` and ``a``)."""
        part = mm(a, w)
        a.deallocate(True)
        full = self.plan.all_reduce(part)
        out = ttnn.add(x, full)
        x.deallocate(True)
        full.deallocate(True)
        return out

    def warm(self):
        """Compile the draft program with an idle run (an all-zero scatter writes nothing) before any trace capture
        (TtMiMoModel.capture_decode_trace): the tiled reshape owns a device mapping table, allocated when its program
        compiles, and a buffer allocated after a capture may sit where that trace writes on replay (a draft program
        first compiled after the decode and verify captures hung on it in serving)."""
        self.upload([])
        self._body()

    def capture_trace(self):
        """Compile with an idle run (warm: a hit by now), then capture the draft program."""
        assert self.trace is None and self.head is not None
        m = self.model
        self.upload([])
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
    def host_inputs(self, drafts):
        """The packed upload of a draft (uint32 [1, n], reference.spec_chain.dflash_inputs): ``drafts`` [(group,
        context, anchor, start, anchor_slot)], per drafting group ``context`` [(tap row, position)] the committed rows
        to write into its ring, ``anchor`` the last committed token, at position ``start``, from slot ``anchor_slot``
        (the drafts step reports the drafts of a group with one). Positions past the RoPE tables (a draft at the
        context's end, which the verify then drops) read their last row."""
        words = dflash_inputs(self.layout, drafts)
        packed = torch.tensor([[w - (1 << 32) if w >= 1 << 31 else w for w in words]], dtype=torch.int32)
        return ttnn.from_torch(
            packed, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.plan.replicate()
        )

    def upload(self, drafts):
        t = time.perf_counter()
        host = self.host_inputs(drafts)
        self.last_inputs_s = time.perf_counter() - t
        ttnn.copy_host_to_device_tensor(host, self.inputs["packed"])

    def enqueue(self, drafts):
        """Draft after the replica's last decode or verify step (non-blocking): writes each group's context rows (see
        host_inputs) from the taps that step left, then drafts its block after its anchor."""
        self._require_trace()
        self.upload(drafts)
        ttnn.execute_trace(self.model.mesh_device, self.trace, cq_id=0, blocking=False)

    def enqueue_chained(self):
        """Draft right behind a verify pass and its accept trace (non-blocking): the accept trace's chain step wrote
        the packed upload from the pass's accepted count and committed tokens, so nothing is copied."""
        self._require_trace()
        ttnn.execute_trace(self.model.mesh_device, self.trace, cq_id=0, blocking=False)

    def _require_trace(self):
        if self.trace is None:
            raise RuntimeError("a DFlash draft before warmup_model_decode captured the draft trace")

    def read_drafts(self, group=0):
        """Draft group ``group``'s drafts (blocks until the draft finished)."""
        row = ttnn.to_torch(ttnn.get_device_tensors(self.head.next_tokens)[0]).reshape(-1)
        return [int(t) for t in row[group * MAX_DRAFTS : (group + 1) * MAX_DRAFTS]]
