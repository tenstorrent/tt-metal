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

The layer is a decoder block with SWA attention (sinks, window 128) and a dense MLP, so it is a TtMiMoDecoderLayer
at the MTP layer index; ``eh_proj`` (column-parallel) mixes cat(enorm(embedding), hnorm(hidden)) into it, and its
own final norm feeds the target's LM head. Its KV cache is a small ring addressed at mtp_cache_position(t) with RoPE
at the true position t (reference/spec_decode.py).
"""

import os

import torch

import ttnn

from ..reference.mtp import load_mtp_weights
from ..reference.spec_decode import MAX_DRAFTS, SPEC_SLOTS, mtp_cache_position, mtp_table_row, serial_positions
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

    def _body(self, inputs):
        tokens, fill, *rest = inputs
        serial = list(rest[:SPEC_SLOTS])
        step2, step3, cos1, sin1, cos2, sin2, cos3, sin3 = rest[SPEC_SLOTS:]
        hidden = self.model.norm_buffers[0]
        h1 = self._pass(tokens, hidden, fill, serial, cos1, sin1, self.heads[0])
        h2 = self._pass(self.heads[0].next_tokens, h1, step2, None, cos2, sin2, self.heads[1])
        h1.deallocate(True)
        h3 = self._pass(self.heads[1].next_tokens, h2, step3, None, cos3, sin3, self.heads[2])
        h2.deallocate(True)
        h3.deallocate(True)

    # ------------------------------------------------------------------ host
    def host_inputs(self, tokens=None, plan=None):
        """Host tensors of one draft: ``tokens`` per slot (the tokens committed there), ``plan`` the true positions
        (reference.spec_decode.MtpInputs); None for an idle run that writes nothing."""
        m = self.model
        rep = self.plan.replicate()
        idle = [-1] * SPEC_SLOTS
        fill, step2, step3 = (plan.fill, plan.step2, plan.step3) if plan is not None else (idle, idle, idle)

        def padded(values, fill_value):
            t = torch.full((m.max_batch,), fill_value, dtype=torch.int32)
            t[: len(values)] = torch.tensor(values, dtype=torch.int32)
            return t

        def ids(values):
            return ttnn.from_torch(padded(values, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep)

        def cache(values):
            return ids([mtp_cache_position(p) for p in values])

        tok = padded(tokens if tokens is not None else [0] * SPEC_SLOTS, 0).reshape(1, -1)
        out = [ttnn.from_torch(tok, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep), cache(fill)]
        out += [ids(s) for s in serial_positions([mtp_cache_position(p) for p in fill])]
        out += [cache(step2), cache(step3)]
        rope = m.rope_setups["swa"]
        for values in (fill, step2, step3):
            out += list(rope.decode_host_inputs(padded(values, -1)))
        return out

    def capture_trace(self):
        """Compile with an idle run (every position -1: no K / V written), then capture the draft program."""
        assert self.trace is None and self.heads is not None
        m = self.model
        inputs = [ttnn.to_device(t, m.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in self.host_inputs()]
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
        """Draft after the replica's last decode or verify step (non-blocking): it reads that step's final hidden."""
        for h, d in zip(self.host_inputs(tokens, plan), self.trace["inputs"]):
            ttnn.copy_host_to_device_tensor(h, d)
        ttnn.execute_trace(self.model.mesh_device, self.trace["id"], cq_id=0, blocking=False)

    def read_drafts(self, anchor):
        """The drafts of the anchor slot (blocks until the draft program finished)."""
        return [int(self.plan.to_torch_replicated(h.next_tokens).reshape(-1)[anchor]) for h in self.heads]
