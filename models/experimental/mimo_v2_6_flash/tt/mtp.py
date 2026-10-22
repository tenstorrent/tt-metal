# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo's MTP layer as a speculative drafter, on one TP=8 replica (reference/mtp.py, reference/spec_decode.py).

One traced program per replica drafts MAX_DRAFTS tokens for the one request of the lane that speculates:

* fill: the MTP layer over the request's committed tokens of the last step. Slot ``s`` takes the target's final
  normed hidden row ``s`` of that step (the persistent buffer its last layer's reduction writes, TtMiMoModel
  ``norm_buffers[0]``) and the embedding of the token committed there (uploaded), and writes the layer's K / V at that
  position, slot by slot; the anchor slot's greedy token is draft 1;
* two steps on the anchor slot alone, each taking the layer's own normed output and the previous draft (on device),
  for drafts 2 and 3.

A draft copies one packed int32 tensor (tt/spec_inputs.py): the program starts by writing its token and position
inputs from it. The committed tokens are uploaded, not taken from the last step's greedy tokens on device: they are
the greedy tokens only when the plugin accepted greedily (a logits verify or a host-sampled step commits the plugin's
own samples). With the decode RoPE tables (TtMiMoModel.setup_decode_rope_tables) the cos / sin are rows of the tables
at the uploaded positions, as in decode; without them (long contexts) the draft also copies its six cos / sin.

The layer is a decoder block with SWA attention (sinks, window 128) and a dense MLP, so it is a TtMiMoDecoderLayer
at the MTP layer index; ``eh_proj`` (column-parallel) mixes cat(enorm(embedding), hnorm(hidden)) into it, and its
own final norm feeds the target's LM head. Its KV cache is a small ring addressed at mtp_cache_position(t) with RoPE
at the true position t (reference/spec_decode.py).
"""

import os
import time

import torch

import ttnn

from ..reference.mtp import load_mtp_weights
from ..reference.spec_decode import MAX_DRAFTS, SPEC_SLOTS, mtp_table_row
from .common import as_weight, cache_name, compute_config
from .decoder import TtMiMoDecoderLayer, TtRMSNorm
from .greedy_head import GreedyHead


class TtMiMoMTP:
    def __init__(self, model, ckpt=None, cache_dir=None, layer=0, weight_dtype=ttnn.bfloat8_b):
        """``model``: the replica's TtMiMoModel (embedding, LM head, RoPE setups, greedy head, final hidden buffer).
        ``ckpt`` may be None when ``cache_dir`` already holds MTP layer ``layer``'s tensors (a later replica)."""
        cfg, plan = model.cfg, model.plan
        self.model, self.cfg, self.plan = model, cfg, plan
        tag = f"mtp{layer}_tp{plan.tp}"
        done = cache_name(cache_dir, tag, "complete")
        warm = done is not None and os.path.exists(done)
        assert ckpt is not None or warm, "cold cache needs a checkpoint"
        weights = None if warm else load_mtp_weights(ckpt, cfg, layer)
        head = weights["head"] if weights is not None else {}
        self.block = TtMiMoDecoderLayer(
            plan,
            cfg,
            cfg.mtp_layer_index(layer),
            weights,
            model.rope_setups,
            cache_dir=cache_dir,
            attn_dtype=weight_dtype,
            mlp_dtype=weight_dtype,
        )
        eps = cfg.layernorm_epsilon
        self.enorm = TtRMSNorm(plan, head.get("enorm"), eps, cache_name(cache_dir, tag, "enorm"))
        self.hnorm = TtRMSNorm(plan, head.get("hnorm"), eps, cache_name(cache_dir, tag, "hnorm"))
        self.final_norm = TtRMSNorm(plan, head.get("final_layernorm"), eps, cache_name(cache_dir, tag, "final_norm"))
        eh = head["eh_proj"].T.contiguous() if weights is not None else None  # [2 * hidden, hidden]
        self.eh_proj = as_weight(eh, plan, plan.shard(-1), weight_dtype, cache_name(cache_dir, tag, "eh_proj"))
        del weights, head, eh
        if done is not None and not warm:
            open(done, "w").close()
        self.mm_config = compute_config(ttnn.MathFidelity.HiFi2)
        self.kv_cache = None
        self.page_table = None
        self.heads = None  # one GreedyHead per draft: their persistent token rows are the drafts
        self.trace = None

    def setup_cache(self):
        """The ring KV cache, its page table and the greedy heads: persistent, made before any trace capture."""
        m = self.model
        assert self.kv_cache is None and m.decode_users is not None and m.decode_users >= SPEC_SLOTS
        row, blocks = mtp_table_row(m.block_size)
        self.kv_cache = self.block.attn.allocate_kv_cache(blocks, m.block_size, m.kv_dtype)
        table = torch.zeros(m.max_batch, len(row), dtype=torch.int32)
        table[:SPEC_SLOTS] = torch.tensor(row, dtype=torch.int32)
        self.page_table = ttnn.from_torch(
            table,
            device=m.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.plan.replicate(),
        )
        self.heads = [GreedyHead(self.plan, m.vocab_local, m.decode_users, m.max_batch) for _ in range(MAX_DRAFTS)]

    def reset(self):
        """Zero the ring (a new request's prompt): its attention reads no K / V of an earlier request."""
        for t in self.kv_cache:
            ttnn.fill(t, 0.0, output_tensor=t)

    # ------------------------------------------------------------------ device
    def _pass(self, tokens_tt, hidden, pos_tt, serial, cos, sin, head):
        """One MTP layer step over the slots: tokens [1, B] u32, hidden [1, 1, B, hidden] (final normed), cache
        positions [B] i32 (-1 skips a slot), cos / sin [1, 1, B, D] of the true positions. Writes the greedy tokens
        into ``head.next_tokens`` and returns the layer's final normed output."""
        m = self.model
        e = ttnn.embedding(tokens_tt, m.embedding, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        e = ttnn.reshape(e, (1, 1, m.max_batch, self.cfg.hidden_size))
        en = self.enorm(e)
        e.deallocate(True)
        hn = self.hnorm(hidden)
        cat = ttnn.concat([en, hn], dim=-1)
        en.deallocate(True)
        hn.deallocate(True)
        xs = ttnn.linear(
            cat,
            self.eh_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
        )
        cat.deallocate(True)
        x = self.plan.all_gather(xs, dim=-1)
        xs.deallocate(True)
        rot = tuple(m.rope_setups["swa"].decode_rot_mats(cos, sin))
        positions = {"pos": pos_tt}
        if serial:
            positions["serial"] = serial
        y = self.block.forward_decode(x, {True: rot}, positions, self.page_table, self.kv_cache)
        x.deallocate(True)
        for t in rot:
            t.deallocate(True)
        h = self.final_norm(y)
        y.deallocate(True)
        logits = m._logits(None, h=h)
        head(logits)
        logits.deallocate(True)
        return h

    def _rope(self, rope):
        """The (cos, sin) of each of the three passes and the tensors made for them: the host inputs, or with the
        decode RoPE tables the table rows at the three RoPE index inputs (the SWA theta's tables)."""
        m = self.model
        if m.rope_tables is None:
            return [tuple(rope[2 * k : 2 * k + 2]) for k in range(3)], []
        pairs, made = [], []
        for idx in rope:
            pair = []
            for t in m.rope_tables[:2]:
                row = ttnn.embedding(idx, t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
                pair.append(ttnn.reshape(row, (1, 1, m.max_batch, t.shape[-1])))
            pairs.append(tuple(pair))
            made += pair
        return pairs, made

    def _body(self, inputs):
        """The draft program: its inputs written from the packed upload, then the three passes. ``inputs``: tokens,
        fill, SPEC_SLOTS serial fill positions, step2, step3, then the RoPE inputs (three table indices, or the
        three passes' host cos / sin)."""
        tokens, fill, *rest = inputs
        serial = list(rest[:SPEC_SLOTS])
        step2, step3, *rope = rest[SPEC_SLOTS:]
        tables = self.model.rope_tables is not None
        self.model.spec_inputs.derive_mtp(tokens, fill, serial, step2, step3, rope if tables else None)
        (cs1, cs2, cs3), made = self._rope(rope)
        hidden = self.model.norm_buffers[0]
        h1 = self._pass(tokens, hidden, fill, serial, *cs1, self.heads[0])
        h2 = self._pass(self.heads[0].next_tokens, h1, step2, None, *cs2, self.heads[1])
        h1.deallocate(True)
        h3 = self._pass(self.heads[1].next_tokens, h2, step3, None, *cs3, self.heads[2])
        h2.deallocate(True)
        h3.deallocate(True)
        for t in made:
            t.deallocate(True)

    # ------------------------------------------------------------------ host
    def _rope_host(self, plan=None):
        """Host cos / sin of the fill, step2 and step3 true positions (-1 idle; None: an idle run), when there are
        no decode RoPE tables."""
        m = self.model
        rope = m.rope_setups["swa"]
        out = []
        for values in (plan.fill, plan.step2, plan.step3) if plan is not None else ([],) * 3:
            t = torch.full((m.max_batch,), -1, dtype=torch.int32)
            t[: len(values)] = torch.tensor(values, dtype=torch.int32)
            out += list(rope.decode_host_inputs(t))
        return out

    def _idle_inputs(self):
        """Host tensors of the program's inputs (``_body``), for an idle run: their device copies are the trace's
        inputs, rewritten from the packed upload on every replay (the RoPE cos / sin by ``enqueue``)."""
        m = self.model
        rep = self.plan.replicate()

        def host(t, dtype):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep)

        tokens = host(torch.zeros(1, m.max_batch, dtype=torch.int32), ttnn.uint32)
        idle = [host(torch.full((m.max_batch,), -1, dtype=torch.int32), ttnn.int32) for _ in range(SPEC_SLOTS + 3)]
        if m.rope_tables is None:
            rope = self._rope_host()
        else:
            rope = [host(torch.zeros(1, m.max_batch, dtype=torch.int32), ttnn.uint32) for _ in range(3)]
        return [tokens, *idle, *rope]

    def capture_trace(self):
        """Compile with an idle run (every position -1: no K / V written), then capture the draft program."""
        assert self.trace is None and self.heads is not None
        m = self.model
        inputs = [ttnn.to_device(t, m.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in self._idle_inputs()]
        # The packed upload still holds the last served draft (a recapture): idle it, so the runs below write no K / V.
        m.spec_inputs.reset()
        self._body(inputs)
        ttnn.synchronize_device(m.mesh_device)
        trace_id = ttnn.begin_trace_capture(m.mesh_device, cq_id=0)
        self._body(inputs)
        ttnn.end_trace_capture(m.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(m.mesh_device)
        self.trace = {"id": trace_id, "inputs": inputs}

    def release_trace(self):
        if self.trace is not None:
            ttnn.release_trace(self.model.mesh_device, self.trace["id"])
            self.trace = None

    def enqueue(self, tokens, plan):
        """Draft after the replica's last decode or verify step (non-blocking): it reads that step's final hidden.
        ``tokens`` per slot (the tokens committed there), ``plan`` the true positions (reference.spec_decode.MtpInputs).
        Copies the packed upload, and the cos / sin without the decode RoPE tables."""
        m = self.model
        # The RoPE indices read rows of the device tables, which end at max_seq_len.
        assert max(plan.step3) < m.max_seq_len, f"draft position {max(plan.step3)} >= {m.max_seq_len}"
        t = time.perf_counter()
        packed = m.spec_inputs.mtp_host(tokens, plan)
        rope = self._rope_host(plan) if m.rope_tables is None else []
        self.last_inputs_s = time.perf_counter() - t
        ttnn.copy_host_to_device_tensor(packed, m.spec_inputs.mtp_packed)
        inputs = self.trace["inputs"]
        for h, d in zip(rope, inputs[len(inputs) - len(rope) :]):
            ttnn.copy_host_to_device_tensor(h, d)
        ttnn.execute_trace(m.mesh_device, self.trace["id"], cq_id=0, blocking=False)

    def read_drafts(self, anchor):
        """The drafts of the anchor slot (blocks until the draft program finished)."""
        return [int(self.plan.to_torch_replicated(h.next_tokens).reshape(-1)[anchor]) for h in self.heads]
