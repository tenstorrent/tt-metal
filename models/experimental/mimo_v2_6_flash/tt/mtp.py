# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo's MTP layers as a speculative drafter, on one TP=8 replica (reference/mtp.py, reference/spec_decode.py).

One traced program per replica drafts MAX_DRAFTS tokens for the one request of the lane that speculates, in one of two
ways (MIMO_MTP_LAYERS, tt/generator_vllm.py):

One layer (vLLM's MTP proposer, upstream vLLM for MiMo):

* fill: the MTP layer over the request's committed tokens of the last step. Slot ``s`` takes the target's final
  normed hidden row ``s`` of that step (the persistent buffer its last layer's reduction writes, TtMiMoModel
  ``norm_buffers[0]``) and the embedding of the token committed there (uploaded), and writes the layer's K / V at that
  position, slot by slot; the anchor slot's greedy token is draft 1;
* two steps on the anchor slot alone, each taking the layer's own normed output and the previous draft (on device),
  for drafts 2 and 3.

The program starts by writing its token and position inputs from one packed int32 tensor (tt/spec_inputs.py), and
ends with the drafts step (tt/spec_chain.py), which writes the anchor slot's three drafts into the result page the
host reads.

Every layer (vLLM 0.31's MultiModuleMTPSpeculator with the num_nextn_predict_layers patch): draft i + 1 from MTP layer
i, each layer over the same rows (plan_mtp_layers: the committed rows after up to two re-prefill rows from the stash,
right-aligned, the anchor in the last slot), each taking the target's hidden, layer i the layer-0 tokens shifted by i:

* the plan step (tt/spec_chain.py) writes the passes' inputs (``LayerInputs``) from the commit's description in the
  packed upload and the stash the last draft left: ring cache positions, serial positions, RoPE indices, the hidden
  table's rows and the layer-0 tokens, and the new stash;
* one gather builds the passes' hidden rows from the table [the step's final norm rows, the stash rows]
  (``ttnn.embedding``: an exact copy) and keeps them as the next stash;
* the three passes, one per layer, each writing its own ring; before passes 1 and 2 the tokens step writes that pass's
  tokens (the drafts so far appended). One RoPE gather serves the three passes (the same positions).

Behind a verify whose acceptance ran on device the accept trace's chain step wrote the packed upload on device, from
the committed tokens and count the accept step left there (``enqueue_chained``: nothing is copied). After a plain
decode step, a logits verify (the plugin samples on the host), or when the plugin committed other tokens than the
device did, the host copies it (``enqueue``: the plan with one layer, the commit's description with every layer).
With the decode RoPE tables (TtMiMoModel.setup_decode_rope_tables, at every context length) the cos / sin are rows
of the tables at the derived positions, as in decode; without them (MIMO_DEVICE_ROPE=0) a one-layer draft also copies
its six cos / sin, and every-layer drafting is not available (its positions are known on device only).

With the backfill (tt/mtp_backfill.py, MIMO_MTP_BACKFILL) a second traced program fills every layer's ring from the
history of the target's hidden before a draft that does not continue the rings (``enqueue_backfill``): after a
prompt, after the lane served several requests.

With draft groups (MIMO_SPEC_GROUPS=2, every layer) one program drafts for a lane's two requests: group g's rows run in
slots 4 g .. 4 g + 3 with its own ring (the page table maps those slots to ring g's blocks) and its own stash slots, from
its own description (row g of the packed upload). A group's first draft for a request backfills its ring and stash
(the history is per request slot), so a prompt resets nothing.

An MTP layer is a decoder block with SWA attention (sinks, window 128) and a dense MLP, so it is a TtMiMoDecoderLayer
at the MTP layer index; ``eh_proj`` (column-parallel) mixes cat(enorm(embedding), hnorm(hidden)) into it, and its own
final norm feeds the target's LM head. Each layer's KV cache is a small ring addressed at mtp_cache_position(t) with
RoPE at the true position t (reference/spec_decode.py).
"""

import time
from dataclasses import dataclass

import torch

import ttnn

from ..reference.mtp import load_mtp_weights
from ..reference.mtp_backfill import MTP_WINDOW
from ..reference.spec_chain import stash_page
from ..reference.spec_decode import (
    DRAFT_GROUPS,
    GROUP_SLOTS,
    MAX_DRAFTS,
    SERIAL_WRITES,
    SPEC_SLOTS,
    mtp_table_row,
    planned_drafts,
)
from ..reference.spec_inputs import pack_mtp_commits
from .common import TILE, as_weight, cache_fill, cache_name, compute_config
from .decoder import TtMiMoDecoderLayer, TtRMSNorm
from .greedy_head import GreedyHead
from .mtp_backfill import MtpHistory
from .spec_chain import SpecChain


class TtMtpLayer:
    """One MTP layer's weights and ring KV cache."""

    def __init__(self, model, ckpt, cache_dir, layer, weight_dtype):
        cfg, plan = model.cfg, model.plan
        self.index = layer
        tag = f"mtp{layer}_tp{plan.tp}"
        with cache_fill(cache_dir, tag) as need:
            assert ckpt is not None or not need, "cold cache needs a checkpoint"
            weights = load_mtp_weights(ckpt, cfg, layer) if need else None
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
            self.final_norm = TtRMSNorm(
                plan, head.get("final_layernorm"), eps, cache_name(cache_dir, tag, "final_norm")
            )
            eh = head["eh_proj"].T.contiguous() if weights is not None else None  # [2 * hidden, hidden]
            self.eh_proj = as_weight(eh, plan, plan.shard(-1), weight_dtype, cache_name(cache_dir, tag, "eh_proj"))
            del weights, head, eh
        self.kv_cache = None


@dataclass
class LayerInputs:
    """The every-layer draft program's persistent inputs, written by the plan and tokens steps on every replay
    (reference/spec_chain.py ``plan_step``): ring cache positions ``pos`` int32 [B], ``serial`` SERIAL_WRITES int32 [B]
    (slot j's alone), ``rope`` RoPE indices uint32 [1, B], ``hidden`` the hidden table's rows uint32 [1, B], ``tokens``
    each pass's tokens uint32 [1, B]; and the stash: its page int32 [B] (positions, layer-0 tokens) and its hidden rows
    ``stash_hidden`` bf16 row-major [1, 1, table_rows, hidden] (slot k's at row k), the table's second part."""

    pos: object
    serial: list
    rope: object
    hidden: object
    tokens: list
    stash: object
    stash_hidden: object
    table_rows: int


def make_layer_inputs(plan, max_batch, hidden_size):
    """The every-layer draft program's LayerInputs on the replica ``plan``: idle positions, an empty stash."""
    rep = plan.replicate()

    def tensor(values, dtype):
        return ttnn.from_torch(
            values,
            device=plan.mesh_device,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )

    def rows(value, dtype, shape):
        return tensor(torch.full(shape, value, dtype=torch.int32), dtype)

    b = max_batch
    return LayerInputs(
        pos=rows(-1, ttnn.int32, (b,)),
        serial=[rows(-1, ttnn.int32, (b,)) for _ in range(SERIAL_WRITES)],
        rope=rows(0, ttnn.uint32, (1, b)),
        hidden=rows(0, ttnn.uint32, (1, b)),
        tokens=[rows(0, ttnn.uint32, (1, b)) for _ in range(MAX_DRAFTS)],
        stash=tensor(torch.tensor(stash_page(b), dtype=torch.int32), ttnn.int32),
        stash_hidden=tensor(torch.zeros(1, 1, b, hidden_size), ttnn.bfloat16),
        table_rows=b,
    )


class TtMiMoMTP:
    def __init__(self, model, ckpt=None, cache_dir=None, layers=(0,), weight_dtype=ttnn.bfloat8_b, backfill=False):
        """``model``: the replica's TtMiMoModel (embedding, LM head, RoPE setups, greedy head, final hidden buffer).
        ``layers``: the MTP layers to load, one (it drafts every token) or MAX_DRAFTS (0, 1, 2: draft i + 1 from layer
        i). Every layer, and one layer drafting one token (groups of 2 slots, MIMO_SPEC_GROUPS=4), run the planned
        program (``multi``: reference.spec_decode.planned_drafts), which drafts for every draft group. ``ckpt`` may be None when ``cache_dir`` already holds their tensors (a later replica). ``backfill``: keep
        the target-hidden history and the backfill program (with the decode RoPE tables)."""
        layers = (layers,) if isinstance(layers, int) else tuple(layers)
        planned = planned_drafts(len(layers))
        assert planned or GROUP_SLOTS == 4, "the one-layer program drafts 3 tokens (MIMO_SPEC_SLOTS / GROUPS = 4)"
        assert len(layers) == 1 or layers == tuple(range(MAX_DRAFTS)), f"MTP layers {layers}: one, or one per draft"
        assert planned or DRAFT_GROUPS == 1, "the one-layer program drafts for one request per lane"
        self.model, self.cfg, self.plan = model, model.cfg, model.plan
        self.layers = [TtMtpLayer(model, ckpt, cache_dir, i, weight_dtype) for i in layers]
        self.multi = planned  # the planned program (plan step, hidden gather, a pass per layer)
        self.mm_config = compute_config(ttnn.MathFidelity.HiFi2)
        self.page_table = None
        self.heads = None  # one GreedyHead per draft: their persistent token rows are the drafts
        self.inputs = None  # every layer: LayerInputs
        self.backfill = backfill
        self.history = None  # MtpHistory (setup_cache, with ``backfill`` and the decode RoPE tables)
        self.trace = None
        self.backfill_trace = None
        self._inputs = None  # the one-layer draft program's inputs (``_trace_inputs``)

    def setup_cache(self):
        """The ring KV caches, their page table, the greedy heads, with every layer the program's LayerInputs, and the
        replica's SpecChain (its result page): persistent, made before any trace capture, after
        TtMiMoModel.setup_vllm_kv_cache (the verify's GreedyHead and VerifySampler, which the chain reads)."""
        m = self.model
        assert self.page_table is None and m.decode_users is not None and m.decode_users >= SPEC_SLOTS
        row, blocks = mtp_table_row(m.block_size)
        for layer in self.layers:
            layer.kv_cache = layer.block.attn.allocate_kv_cache(DRAFT_GROUPS * blocks, m.block_size, m.kv_dtype)
        table = torch.zeros(m.max_batch, len(row), dtype=torch.int32)
        for g in range(DRAFT_GROUPS):  # draft group g's slots read and write its own ring
            group_row = mtp_table_row(m.block_size, group=g)[0]
            table[g * GROUP_SLOTS : (g + 1) * GROUP_SLOTS] = torch.tensor(group_row, dtype=torch.int32)
        self.page_table = ttnn.from_torch(
            table,
            device=m.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.plan.replicate(),
        )
        self.heads = [GreedyHead(self.plan, m.vocab_local, m.decode_users, m.max_batch) for _ in range(MAX_DRAFTS)]
        if self.multi:
            # The hidden table's first part is the final norm buffer (TtMiMoModel.norm_buffers, one tile of rows).
            assert m.norm_buffers is not None and m.norm_buffers[0].shape[-2] == TILE == m.max_batch
            self.inputs = make_layer_inputs(self.plan, m.max_batch, self.cfg.hidden_size)
        if self.backfill and m.rope_tables is not None:
            # The history's ring and the backfill's window are sized for the MTP layers' window.
            assert self.cfg.sliding_window == MTP_WINDOW, f"MTP window {self.cfg.sliding_window} != {MTP_WINDOW}"
            self.history = MtpHistory(
                self.plan, m.max_batch, self.cfg.hidden_size, len(self.layers), m.rope_table_tails
            )
        # A draft group's ring and stash are refilled from the history (a prompt cannot zero one group's ring alone).
        assert DRAFT_GROUPS == 1 or self.history is not None, "draft groups need the MTP history (MIMO_MTP_BACKFILL)"
        assert m.verify_sampler is not None and m.greedy_head is not None, "the chain reads the verify's buffers"
        heads = [h.next_tokens for h in self.heads]
        m.spec_chain = SpecChain(
            self.plan, m.spec_inputs, m.verify_sampler.out, m.greedy_head.next_tokens, heads, self.inputs
        )

    def reset(self):
        """Zero the rings and empty the stash (a new request's prompt): its attention reads no K / V of an earlier
        request, and its first draft re-runs no row of one. One draft group only (with groups the first draft of a
        request backfills its group's ring and stash)."""
        assert DRAFT_GROUPS == 1, "draft groups refill a group's ring and stash by the backfill"
        for layer in self.layers:
            for t in layer.kv_cache:
                ttnn.fill(t, 0.0, output_tensor=t)
        if self.inputs is not None:
            host = torch.tensor(stash_page(self.model.max_batch), dtype=torch.int32)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.plan.replicate()
                ),
                self.inputs.stash,
            )

    # ------------------------------------------------------------------ device
    def _pass(self, tokens_tt, hidden, pos_tt, serial, cos, sin, head, layer=None):
        """One MTP layer step over the slots (``layer``, default the only one): tokens [1, B] u32, hidden [1, 1, B,
        hidden] (final normed), cache positions [B] i32 (-1 skips a slot), cos / sin [1, 1, B, D] of the true
        positions. Writes the greedy tokens into ``head.next_tokens`` and returns the layer's final normed output."""
        m = self.model
        layer = layer or self.layers[0]
        e = ttnn.embedding(tokens_tt, m.embedding, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        e = ttnn.reshape(e, (1, 1, m.max_batch, self.cfg.hidden_size))
        en = layer.enorm(e)
        e.deallocate(True)
        hn = layer.hnorm(hidden)
        cat = ttnn.concat([en, hn], dim=-1)
        en.deallocate(True)
        hn.deallocate(True)
        xs = ttnn.linear(
            cat,
            layer.eh_proj,
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
        y = layer.block.forward_decode(x, {True: rot}, positions, self.page_table, layer.kv_cache)
        x.deallocate(True)
        for t in rot:
            t.deallocate(True)
        h = layer.final_norm(y)
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
            pair = m.rope_table_rows(idx, m.rope_tables[:2])
            pairs.append(tuple(pair))
            made += pair
        return pairs, made

    def _body(self, inputs):
        """The one-layer draft program: its inputs written from the packed upload, the three passes, then the drafts
        step (the anchor's drafts into the result page). ``inputs``: tokens, fill, SERIAL_WRITES serial fill positions,
        step2, step3, then the RoPE inputs (three table indices, or the three passes' host cos / sin)."""
        tokens, fill, *rest = inputs
        serial = list(rest[:SERIAL_WRITES])
        step2, step3, *rope = rest[SERIAL_WRITES:]
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
        self.model.spec_chain.drafts()

    def _gather_hidden(self):
        """The passes' hidden rows [1, 1, B, hidden] (tile): row j of the table [final norm rows, stash rows] at the
        plan step's ``hidden`` index j, an exact copy; they also become the stash's hidden rows."""
        m, li = self.model, self.inputs
        step = ttnn.to_layout(m.norm_buffers[0], ttnn.ROW_MAJOR_LAYOUT)
        table = ttnn.concat([step, li.stash_hidden], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        step.deallocate(True)
        flat = ttnn.reshape(table, (2 * li.table_rows, self.cfg.hidden_size))
        rows = ttnn.embedding(li.hidden, flat, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
        table.deallocate(True)
        rows = ttnn.reshape(rows, (1, 1, m.max_batch, self.cfg.hidden_size))
        ttnn.copy(rows, li.stash_hidden)
        hidden = ttnn.to_layout(rows, ttnn.TILE_LAYOUT)
        rows.deallocate(True)
        return hidden

    def _body_layers(self):
        """The every-layer draft program: the plan step, the hidden gather, then pass i on MTP layer i with its tokens
        (written by the tokens step for i >= 1), and the drafts step. Reads the decode RoPE tables once."""
        m, li, chain = self.model, self.inputs, self.model.spec_chain
        chain.plan_layers()
        cos, sin = m.rope_table_rows(li.rope, m.rope_tables[:2])
        hidden = self._gather_hidden()
        for i, layer in enumerate(self.layers):
            if i:
                chain.tokens(i)
            h = self._pass(li.tokens[i], hidden, li.pos, li.serial, cos, sin, self.heads[i], layer)
            h.deallocate(True)
        hidden.deallocate(True)
        cos.deallocate(True)
        sin.deallocate(True)
        chain.drafts()

    def _body_backfill(self):
        """The backfill program: the plan step, the history rows' gather and their cos / sin, then per layer its input
        over the rows (row-invariant prefill norms) and its K / V filled into its whole ring."""
        m, hist = self.model, self.history
        hist.backfill_plan(self.inputs)
        hidden, (cos, sin) = hist.gather(m)
        rows = hidden.shape[-2]
        for i, layer in enumerate(self.layers):
            e = ttnn.embedding(hist.tokens[i], m.embedding, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            e = ttnn.reshape(e, (1, 1, rows, self.cfg.hidden_size))
            en = layer.enorm(e, prefill=True)
            e.deallocate(True)
            hn = layer.hnorm(hidden, prefill=True)
            cat = ttnn.concat([en, hn], dim=-1)
            en.deallocate(True)
            hn.deallocate(True)
            xs = ttnn.linear(
                cat,
                layer.eh_proj,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.mm_config,
            )
            cat.deallocate(True)
            x = self.plan.all_gather(xs, dim=-1)
            xs.deallocate(True)
            h = layer.block.input_norm(x, prefill=True)
            x.deallocate(True)
            layer.block.attn.fill_kv_rows(h, cos, sin, hist.table, layer.kv_cache)
            h.deallocate(True)
        for t in (hidden, cos, sin):
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
        """Host tensors of the one-layer program's inputs (``_body``), for an idle run: their device copies are the
        trace's inputs, rewritten from the packed upload on every replay (the RoPE cos / sin by ``enqueue``)."""
        m = self.model
        rep = self.plan.replicate()

        def host(t, dtype):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep)

        tokens = host(torch.zeros(1, m.max_batch, dtype=torch.int32), ttnn.uint32)
        idle = [host(torch.full((m.max_batch,), -1, dtype=torch.int32), ttnn.int32) for _ in range(SERIAL_WRITES + 3)]
        if m.rope_tables is None:
            rope = self._rope_host()
        else:
            rope = [host(torch.zeros(1, m.max_batch, dtype=torch.int32), ttnn.uint32) for _ in range(3)]
        return [tokens, *idle, *rope]

    def _trace_inputs(self):
        """The one-layer draft program's inputs (made once, before any trace capture); none with every layer."""
        m = self.model
        if self.multi:
            assert m.rope_tables is not None, "every-layer drafting reads its cos / sin from the decode RoPE tables"
            return []
        if self._inputs is None:
            self._inputs = [
                ttnn.to_device(t, m.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in self._idle_inputs()
            ]
        return self._inputs

    def _run_body(self, inputs):
        if self.multi:
            self._body_layers()
        else:
            self._body(inputs)

    def warm(self):
        """Compile the draft programs (the draft and, with the history, the backfill) with idle runs before any trace
        capture (TtMiMoModel.capture_decode_trace): an op that owns device buffers (the tiled reshape's mapping table)
        allocates them when its program compiles, and a buffer allocated after a capture may sit where that trace
        writes on replay. Every position -1 (no K / V written); with every layer an idle description, which leaves the
        stash as it was."""
        self.model.spec_inputs.reset()
        self._run_body(self._trace_inputs())
        if self.history is not None:
            self.history.upload_backfill()
            self._body_backfill()

    def capture_trace(self):
        """Compile with an idle run (warm: a hit by now), then capture the draft program."""
        assert self.trace is None and self.heads is not None
        m = self.model
        inputs = self._trace_inputs()

        def body():
            self._run_body(inputs)

        # The packed upload still holds the last served draft (a recapture): idle it, so the runs below write no K / V.
        m.spec_inputs.reset()
        body()
        ttnn.synchronize_device(m.mesh_device)
        trace_id = ttnn.begin_trace_capture(m.mesh_device, cq_id=0)
        body()
        ttnn.end_trace_capture(m.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(m.mesh_device)
        self.trace = {"id": trace_id, "inputs": inputs}
        if self.history is not None:
            # The backfill program: compiled by an idle run (an idle upload fills the rings' scratch blocks and leaves
            # the stash), then captured.
            self.history.upload_backfill()
            self._body_backfill()
            ttnn.synchronize_device(m.mesh_device)
            backfill_id = ttnn.begin_trace_capture(m.mesh_device, cq_id=0)
            self._body_backfill()
            ttnn.end_trace_capture(m.mesh_device, backfill_id, cq_id=0)
            ttnn.synchronize_device(m.mesh_device)
            self.backfill_trace = backfill_id

    def release_trace(self):
        if self.trace is not None:
            ttnn.release_trace(self.model.mesh_device, self.trace["id"])
            self.trace = None
        if self.backfill_trace is not None:
            ttnn.release_trace(self.model.mesh_device, self.backfill_trace)
            self.backfill_trace = None

    def _require_trace(self):
        """A draft replays the captured program (there is no eager draft)."""
        if self.trace is None:
            raise RuntimeError(
                "an MTP draft before warmup_model_decode captured the draft trace (it captures it with enable_trace: "
                "the plugin's trace_mode 'all' or 'decode_only', with enable_model_warmup)"
            )

    def enqueue_backfill(self, slot, a, group=0):
        """Fill every layer's ring of draft group ``group`` from the history of the request in lane ``slot`` before a
        draft whose first row is at ``a`` (non-blocking; enqueue the draft right after it): the window
        a - MTP_WINDOW + 1 .. a - 1, and with every layer the stash rows a - 2, a - 1 the draft then re-runs (those of
        them its commit leaves stale), in the group's stash slots."""
        self._require_trace()  # captured with the draft trace (capture_trace)
        assert self.backfill_trace is not None and 0 <= slot < SPEC_SLOTS and 0 <= group < DRAFT_GROUPS
        self.history.upload_backfill(slot, a, group)
        ttnn.execute_trace(self.model.mesh_device, self.backfill_trace, cq_id=0, blocking=False)

    def enqueue(self, tokens, plan):
        """One layer: draft after the replica's last decode or verify step (non-blocking): it reads that step's final
        hidden. ``tokens`` per slot (the tokens committed there), ``plan`` the true positions
        (reference.spec_decode.MtpInputs). Copies the packed upload, and the cos / sin without the decode RoPE
        tables."""
        self._require_trace()
        assert not self.multi, "every layer: enqueue_layers"
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

    def enqueue_layers(self, commits):
        """Every layer: draft after the replica's last decode or verify step (non-blocking) from ``commits``
        [(DraftSource, committed tokens, committed positions, accepted count)], one per draft group: copies their
        descriptions (reference.spec_inputs.pack_mtp_commits); the plan step plans the passes on device."""
        self._require_trace()
        assert self.multi, "one layer: enqueue"
        m = self.model
        assert all(int(positions[int(n) - 1]) < m.max_seq_len for _, _, positions, n in commits)
        t = time.perf_counter()
        packed = pack_mtp_commits(m.max_batch, commits)
        host = ttnn.from_torch(
            packed, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.plan.replicate()
        )
        self.last_inputs_s = time.perf_counter() - t
        ttnn.copy_host_to_device_tensor(host, m.spec_inputs.mtp_packed)
        ttnn.execute_trace(m.mesh_device, self.trace["id"], cq_id=0, blocking=False)

    def enqueue_chained(self):
        """Draft right behind a verify pass and its accept trace (non-blocking): the accept trace's chain step wrote
        the packed upload's buffer from the pass's accepted count and committed tokens, so nothing is copied. Needs the
        decode RoPE tables (without them the host copies the draft's cos / sin: ``enqueue``)."""
        self._require_trace()
        assert self.model.rope_tables is not None, "a chained draft reads its cos / sin from the decode RoPE tables"
        ttnn.execute_trace(self.model.mesh_device, self.trace["id"], cq_id=0, blocking=False)

    def read_drafts(self, anchor, group=0):
        """Draft group ``group``'s drafts of the anchor slot, from the result page the drafts step wrote (blocks until
        it finished)."""
        chain = self.model.spec_chain.read().groups[group]
        if chain.draft_anchor != anchor:
            raise RuntimeError(f"the draft ran at slot {chain.draft_anchor}, the host planned slot {anchor}")
        return chain.drafts
