# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter for the resident Qwen3.6 / 3.8-27B serving several users at once (rows mode).

The device side is a rows-mode ResidentModel (launches of ROWS independent users, paged KV pool, one trace per
launch) and the pipeline-parallel prefill; the contract is .omc/research/resident-multiuser-api.md. This module
holds the host bookkeeping the vLLM plugin's contract (docs/DECODE_RELOAD_CONTRACT.md) asks of a stateful model:

- Resident slots. A request lives in one resident slot (launch slot // ROWS, row slot % ROWS) from its prefill to
  its release; the device never moves state. The plugin addresses requests by its own state slots and permutes them
  (slot_remap); SlotMap keeps vLLM slot -> resident slot on the host and applies each remap there exactly once.
  A new request takes the lowest free resident slot, so 8 users or fewer replay one launch.
- Prefill into a slot: the PP prefill of all prompt tokens but the last, the handoff into the slot (GDN state,
  conv history row, KV into the request's block-table pages), then one masked first-token launch (only the new row
  active; preserve set when the launch has live neighbours) whose logits are the prefill's output.
- Chunked prefill (the plugin's TT chunk core with tt_prefill_chunk_tokens and tt_prefill_exclusive_chunks): the
  PP pipeline keeps one prompt's state across calls; every call feeds whole PP chunks, the final call drains the
  pipeline, hands off and runs the first-token launch. Decode steps of the live users run between the calls. Every
  call of a sequence, split or whole, runs the PP chunk the sequence's whole length picks (prefill_total_lens from
  the chunk core): a split prompt computes exactly what it would whole, whatever the other users were doing.
- Decode: on reload_inputs the host writes every replayed launch's token state (all live rows of a launch active:
  without preserve a launch rewrites every row's conv history, so a live row that sits out needs preserve) and the
  page tables; reload_page_table rewrites only the page tables; otherwise the launches replay on the token states
  the hubs fed (device sampling). Host sampling reads the launches' logits; device sampling reads each launch's
  per-row (token, position, valid) words.

RowsEngine binds this to ResidentModel / PPPrefill; tests/test_vllm_rows_fake.py drives the same adapter on CPU
with a fake model and prefill that check slots, block tables, conv-history preservation and outputs.
"""
import math

import torch
from loguru import logger

ROWS = 8  # users per launch: the rows-mode build is compiled at width 8 for every launch
BLOCK = 64  # KV block (positions) = vLLM block_size
CHUNK_UNIT = 1024  # the PP prefill's largest chunk: a split prompt resumes only at its multiples


def plugin_has_chunk_core():
    """whether the vllm-tt-plugin this process runs has the TT chunk core (branch resident-mu): it keeps one user's
    prompts whole and splits only at the model's unit. A plugin without it acts on supports_chunked_prefill with plain
    token-budget chunking (8192 / 2048-token chunks even at max_num_seqs 1), which the PP prefill cannot resume."""
    try:
        from vllm_tt_plugin import config
    except Exception:
        return False
    return hasattr(config, "get_tt_prefill_chunk_unit")


def resident_capabilities(top_k=32, chunk_core=None):
    """the resident vLLM class's model_capabilities, resolved once against the running plugin: the chunk keys only
    where the plugin has the chunk core, so max_num_seqs 1 resolves as before on any plugin (chunking off, the whole
    prompt in one step) and several users get aligned, exclusive chunks where the plugin can give them"""
    caps = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        # the hub samples from the top top_k logits; anything else samples on the host from the logits
        "supports_sample_on_device": True,
        "max_device_top_k": top_k,
        "supports_device_penalties": False,
    }
    if plugin_has_chunk_core() if chunk_core is None else chunk_core:
        caps.update(
            {
                "supports_chunked_prefill": True,
                "tt_prefill_chunk_tokens": CHUNK_UNIT,
                "tt_prefill_exclusive_chunks": True,
            }
        )
    return caps


class DecodeHandle:
    """what decode_forward returns when it does not read: the step's rows and their slots, nothing read yet. A plain
    class, not a tuple: the plugin takes a tuple of tensors for host output and skips the model's processing"""

    def __init__(self, device, rows, row_slot, n):
        self.device, self.rows, self.row_slot, self.n = device, rows, row_slot, n


class DecodeRead:
    """read_decode_output's host-side result: the rows and the engine's read (logits, or the out words' reads)"""

    def __init__(self, device, rows, n, read):
        self.device, self.rows, self.n, self.read = device, rows, n, read


class SlotMap:
    """vLLM state slot -> resident slot, and the free resident slots.

    The plugin's remap is a gather: remap[i] = j means vLLM slot i now holds what vLLM slot j held. Applied to this
    map only (resident state never moves): the new map sends i to the resident slot j was mapped to."""

    def __init__(self, n_slots):
        self.n = n_slots
        self.res_of = {}
        self.free = set(range(n_slots))

    def assign(self, vslot):
        """a fresh resident slot (the lowest free one) for a request prefilled into vLLM slot vslot"""
        old = self.res_of.pop(vslot, None)
        if old is not None:
            # the plugin hands a prefill only slots no live request holds: a mapping left here is a dead owner the
            # plugin released without telling us (or a resumed request); its resident slot is free again
            self.free.add(old)
        if not self.free:
            raise RuntimeError(f"no free resident slot for vLLM slot {vslot}: map {self.res_of}")
        r = min(self.free)
        self.free.remove(r)
        self.res_of[vslot] = r
        return r

    def release(self, vslot):
        r = self.res_of.pop(vslot, None)
        if r is not None:
            self.free.add(r)
        return r

    def remapped(self, remap):
        """the map after the gather remap (a full permutation of the vLLM slots)"""
        remap = [int(j) for j in remap]
        if sorted(remap) != list(range(len(remap))):
            raise RuntimeError(f"slot_remap is not a permutation: {remap}")
        return {i: self.res_of[j] for i, j in enumerate(remap) if j in self.res_of}


class ResidentRowsAdapter:
    """The plugin-facing multi-user model (decode_input_update_contract 1). `engine` provides the device calls (see
    RowsEngine); `emb` is not needed here: token states are built by the engine from token ids."""

    decode_input_update_contract = 1

    def __init__(self, engine, vocab, max_seq_len, max_batch, chunk_unit, mrope=False):
        self.engine, self.vocab, self.max_seq_len, self.mrope = engine, vocab, max_seq_len, mrope
        self.n_slots = -(-max_batch // ROWS) * ROWS
        self.slots = SlotMap(self.n_slots)
        self.chunk = chunk_unit  # PP resume unit: a split prompt's calls start / end on its multiples
        # resident slots whose state is decode-ready (handed off + first token); a partial prefill's slot is not
        self.live = set()
        # the partial prefill the PP pipeline holds: (resident slot, tokens fed so far) or None
        self.pp_partial = None
        self.tables = [[] for _ in range(self.n_slots)]  # each resident slot's block table (last written)
        self._expect_moves = None  # the moves note_state_slots_moved must report for the last applied remap
        self._sampling = None  # per resident slot (temperature, top_k, top_p, seed) of the device sampler
        self._fed_launches = None  # launches the last device-sampled step replayed (their token states are fed)

    # ------------------------------------------------------------------ helpers
    def _launch(self, r):
        return r // ROWS

    def _write_tables(self):
        self.engine.set_page_table(self.tables)

    def _blocks_for(self, page_table_row, n_pos):
        """the first ceil(n_pos / BLOCK) physical blocks of a page-table row (the rest is padding)"""
        need = math.ceil(n_pos / BLOCK)
        row = [int(b) for b in page_table_row[:need]]
        if len(row) < need:
            raise RuntimeError(f"block table of {len(row)} blocks does not cover {n_pos} positions")
        return row

    # ------------------------------------------------------------------ prefill
    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        start_pos=None,
        empty_slots=None,
        prefill_final_mask=None,
        prefill_total_lens=None,
        sampling_params=None,
        **kwargs,
    ):
        n = int(len(prompt_lens))
        ends = [int(x) for x in prompt_lens]
        starts = [0] * n if start_pos is None else [int(x) for x in start_pos[:n]]
        finals = [True] * n if prefill_final_mask is None else [bool(x) for x in prefill_final_mask[:n]]
        if prefill_final_mask is not None and prefill_total_lens is None:
            raise RuntimeError("the TT chunk core passes prefill_total_lens: vllm-tt-plugin resident-mu is out of date")
        # each row's whole sequence (without the chunk core every prompt runs whole: its end)
        totals = ends if prefill_total_lens is None else [int(x) for x in prefill_total_lens[:n]]
        if empty_slots is None:
            raise RuntimeError("a stateful multi-user model needs the plugin's empty_slots (destination slots)")
        if sampling_params is not None:
            raise RuntimeError("prefill logits are sampled on the host: serve with sample_on_device_mode decode_only")
        vslots = [int(s) for s in empty_slots[:n]]
        if prefill_final_mask is None and any(s > 0 for s in starts):
            raise RuntimeError("a resumed prefill chunk without prefill_final_mask: plugin without the TT chunk core")
        conts = [j for j in range(n) if starts[j] > 0]
        firsts = [j for j in range(n) if starts[j] == 0 and not finals[j]]
        wholes = [j for j in range(n) if starts[j] == 0 and finals[j]]
        if conts and n > 1:
            raise RuntimeError(
                "the PP prefill holds one prompt: a continuation chunk must be alone in its step "
                f"(rows {n}, starts {starts}); serve with the TT chunk core's exclusive chunks"
            )
        if len(firsts) + len(conts) > 1:
            raise RuntimeError(f"more than one split prompt in one step (starts {starts}, finals {finals})")
        out = torch.zeros(n, 1, self.vocab)
        # whole prompts first, the first chunk of a split prompt last: a whole prompt restarts the PP pipeline
        for j in wholes + conts + firsts:
            ids = tokens[j, : ends[j]]
            lg = self._prefill_row(ids, page_table[j], vslots[j], starts[j], ends[j], finals[j], totals[j])
            if lg is not None:
                out[j, 0] = lg
        return (out, torch.zeros(n, dtype=torch.long)) if self.mrope else out

    def _pp_chunk(self, total):
        """the PP chunk of a sequence of `total` tokens, the same whether the scheduler splits it or not (a prompt run
        at another chunk takes other rounding: its output would depend on the other users' timing)"""
        c = self.engine.pick_chunk(total - 1)
        if self.chunk % c:
            raise RuntimeError(f"PP chunk {c} does not divide the chunk unit {self.chunk}")
        return c

    def _prefill_row(self, ids, table_row, vslot, start, end, final, total):
        C = self.chunk
        if start == 0:
            if self.pp_partial is not None:
                # a new prompt restarts the pipeline: the plugin must not start one while a split prompt is in flight
                raise RuntimeError(
                    f"a new prompt while the split prompt in resident slot {self.pp_partial[0]} is partial"
                )
            r = self.slots.assign(vslot)
            self.live.discard(r)
            if final:
                T = end
                if T != total:
                    raise RuntimeError(f"a whole prompt of {T} tokens in a sequence of {total}")
                if T > 1:
                    self.engine.prefill_whole(ids[: T - 1], r, self._blocks_for(table_row, T), self._pp_chunk(T))
                else:
                    self.engine.clear_slot(r)
                return self._first_token(r, ids, T, table_row)
            if end % C:
                raise RuntimeError(f"first chunk ends at {end}, not a multiple of the PP chunk unit {C}")
            c = self._pp_chunk(total)
            self.engine.prefill_begin(r)
            self.engine.prefill_span(ids[:end], 0, end, c)
            self.pp_partial = (r, end, total, c, 1)
            return None
        r = self.slots.res_of.get(vslot)
        if self.pp_partial is None or r is None or self.pp_partial[0] != r:
            raise RuntimeError(f"continuation for vLLM slot {vslot} (resident {r}) but the PP holds {self.pp_partial}")
        _, fed, seq_total, c, calls = self.pp_partial
        if start != fed:
            raise RuntimeError(f"continuation starts at {start}, the PP has {fed} tokens")
        if total != seq_total:
            raise RuntimeError(f"continuation of a sequence of {total} tokens, the PP holds one of {seq_total}")
        if not final:
            if end % C:
                raise RuntimeError(f"chunk ends at {end}, not a multiple of the PP chunk unit {C}")
            self.engine.prefill_span(ids[start:end], start // c, end, c)
            self.pp_partial = (r, end, total, c, calls + 1)
            return None
        T = end
        if T != total:
            raise RuntimeError(f"a final chunk ending at {T} in a sequence of {total}")
        # the PP takes all prompt tokens but the last; the tail may be empty (the final chunk is the last token)
        self.engine.prefill_end(ids[start : T - 1], start // c, T - 1, r, self._blocks_for(table_row, T), c)
        self.pp_partial = None
        # the churn check counts these: split prompts are the ones whose numerics the per-sequence chunk keeps whole
        logger.info(f"split prefill: resident slot {r}, {T} tokens in {calls + 1} calls at PP chunk {c}")
        return self._first_token(r, ids, T, table_row)

    def _first_token(self, r, ids, T, table_row):
        """the masked first-token launch of resident slot r at position T - 1: [vocab] logits"""
        self.tables[r] = [int(b) for b in table_row]
        self._write_tables()
        k = self._launch(r)
        neighbours = any(self._launch(s) == k and s != r for s in self.live)
        lg = self.engine.first_token(r, int(ids[T - 1]), T - 1, preserve=neighbours)
        self.live.add(r)
        # the launch's token state now runs only row r: the next decode must rewrite it (the plugin reloads inputs
        # after a prefill), and the device sampler's fed state of this launch is stale
        self._fed_launches = None
        return lg

    # ------------------------------------------------------------------ decode
    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        *args,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        read_from_device=False,
        sampling_params=None,
        slot_remap=None,
        **kwargs,
    ):
        assert "reset_batch" not in kwargs, "contract 1 adapters take the explicit reload commands"
        if reload_inputs and reload_page_table:
            raise RuntimeError("reload_inputs and reload_page_table are exclusive")
        res_of = self.slots.remapped(slot_remap) if slot_remap is not None else dict(self.slots.res_of)
        pos = [int(p) for p in start_pos.reshape(-1)]
        rows = [i for i, p in enumerate(pos) if p >= 0]
        if rows != list(range(len(rows))):
            raise RuntimeError(f"decode rows must be a prefix of the wire batch: positions {pos}")
        row_slot = []
        for i in rows:
            r = res_of.get(i)
            if r is None or r not in self.live:
                raise RuntimeError(f"decode row {i} has no live resident slot (map {res_of}, live {sorted(self.live)})")
            row_slot.append(r)
        launches = sorted({self._launch(r) for r in row_slot})
        device = sampling_params is not None
        if device and not self.engine.device_sampling:
            raise RuntimeError(
                "device sampling needs the rows-mode hub feed (WP7); serve several users without sample_on_device_mode"
            )
        if reload_inputs or reload_page_table:
            if page_table is None:
                raise RuntimeError("a reload without the page table")
            for i, r in zip(rows, row_slot):
                self.tables[r] = [int(b) for b in page_table[i]]
            self._write_tables()
        if device and (reload_sampling_params or reset_sampling_state or self._sampling is None):
            self._set_sampling(sampling_params, rows, row_slot)
        if reload_inputs or not device:
            toks = [int(t) for t in tokens.reshape(len(pos), -1)[:, 0]]
            feed = {r: (toks[i], pos[i]) for i, r in zip(rows, row_slot)}
            # a live slot of a replayed launch that this step does not run keeps its conv history only with preserve
            sitting_out = {s for s in self.live if self._launch(s) in launches and s not in feed}
            self.engine.decode(feed, launches, preserve_launches={self._launch(s) for s in sitting_out})
        else:
            if self._fed_launches != launches:
                raise RuntimeError(
                    f"steady device decode over launches {launches}, but the fed token states are of "
                    f"{self._fed_launches}: the plugin should have reloaded the inputs"
                )
            self.engine.decode(None, launches)
        self._fed_launches = launches if device else None
        # accepted: commit the remap (exactly once) and expect the plugin's note of the same moves
        if slot_remap is not None:
            remap = [int(j) for j in slot_remap]
            self._expect_moves = {j: i for i, j in enumerate(remap) if i != j}
            self.slots.res_of = res_of
        handle = DecodeHandle(device, rows, row_slot, len(pos))
        if read_from_device:
            return self.process_decode_output_host(self.read_decode_output(handle))
        return handle

    def _set_sampling(self, sp, rows, row_slot):
        cur = dict(self._sampling or {})
        for i, r in zip(rows, row_slot):
            seed = sp.seed[i] if sp.seed is not None else None
            cur[r] = (float(sp.temperature[i]), int(sp.top_k[i]), float(sp.top_p[i]), seed)
        self._sampling = cur
        self.engine.set_sampling(cur)

    def read_decode_output(self, out, async_read=False, **kwargs):
        """async scheduling: the runner calls this right after decode_forward, then waits the events and processes"""
        if isinstance(out, torch.Tensor):
            return (out, []) if async_read else out
        if not isinstance(out, DecodeHandle):
            raise TypeError(f"read_decode_output takes decode_forward's handle, not {type(out).__name__}")
        read = (
            self.engine.read_tokens(out.row_slot, async_read) if out.device else self.engine.read_logits(out.row_slot)
        )
        host = DecodeRead(out.device, out.rows, out.n, read)
        # host-sampled logits are read synchronously (as the single-user adapter does); device-sampled tokens are
        # enqueued reads of the launches' out words, complete at the events
        return (host, self.engine.read_events(read) if out.device else []) if async_read else host

    def process_decode_output_host(self, out, is_tokens=True, **kwargs):
        """synchronous scheduling hands decode_forward's handle straight here (no read_decode_output): read it now"""
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, DecodeHandle):
            out = self.read_decode_output(out)
        if not isinstance(out, DecodeRead):
            raise TypeError(f"process_decode_output_host takes a decode handle or its read, not {type(out).__name__}")
        if out.device:
            got = self.engine.tokens_from(out.read)  # [len(rows)] int
            res = torch.zeros(out.n, 1, dtype=torch.int32)
            res[: len(out.rows), 0] = torch.as_tensor(got, dtype=torch.int32)
            return res
        lg = self.engine.logits_from(out.read)  # [len(rows), vocab]
        res = torch.zeros(out.n, 1, self.vocab)
        res[: len(out.rows), 0] = lg
        return res

    # ------------------------------------------------------------------ lifecycle
    def note_state_slots_moved(self, moves):
        """the plugin's after-acceptance note of the remap decode_forward already applied: check, do not re-apply"""
        moves = {int(a): int(b) for a, b in moves.items()}
        if self._expect_moves is None or moves != self._expect_moves:
            raise RuntimeError(f"slot moves {moves} do not match the applied remap {self._expect_moves}")
        self._expect_moves = None

    def release_request(self, vslot):
        r = self.slots.release(int(vslot))
        if r is None:
            return
        self.live.discard(r)
        if self.pp_partial is not None and self.pp_partial[0] == r:
            # a split prompt preempted or aborted mid-prefill: its resume starts from the first chunk
            self.pp_partial = None
        if self._sampling is not None:
            self._sampling.pop(r, None)

    def release_persistent_capture(self):
        self.pp_partial = None


class RowsEngine:
    """The device calls of ResidentRowsAdapter on a rows-mode ResidentModel and the PP prefill (the contract of
    .omc/research/resident-multiuser-api.md, as built on qwen38-res-mu-wp7). `model` / `pp` may be fakes with the same
    methods (tests): a fake's out_tensor(k) is already a host tensor."""

    def __init__(self, model, pp, emb, vocab, vocab_chip, n_slots, chunk_unit, mesh=None):
        self.model, self.pp, self.emb = model, pp, emb
        self.vocab, self.vocab_chip, self.n_slots = vocab, vocab_chip, n_slots
        self.chunk = chunk_unit
        self.mesh = mesh
        # the rows-mode hub draws and feeds each row's next token (WP7: a model built with embed)
        self.device_sampling = bool(getattr(model, "sample", False))

    def sync(self):
        if self.mesh is not None:
            import ttnn

            ttnn.synchronize_device(self.mesh)

    def _token_states(self, feed, launches, preserve_launches=()):
        """one host token state per launch (model.token_state_rows: {row: (x0, position)} active, the rest inactive),
        None for a launch left as it is"""
        by_launch = {k: {} for k in launches}
        for r, (tok, p) in feed.items():
            by_launch[r // ROWS][r % ROWS] = (self.emb[int(tok)].float(), int(p))
        states = {
            k: self.model.token_state_rows(rows, preserve=k in preserve_launches) for k, rows in by_launch.items()
        }
        return [states.get(k) for k in range(self.n_slots // ROWS)]

    def set_page_table(self, tables):
        self.model.set_page_table([t if t else [0] for t in tables])

    def pick_chunk(self, n_tokens):
        """the PP chunk for a prefill of n_tokens (its fastest; the adapter keeps it for every span of the sequence)"""
        return self.pp.pick_chunk(n_tokens)

    def prefill_whole(self, ids, r, blocks, chunk):
        self.pp.reset()
        self.pp.run(torch.as_tensor(ids), chunk=chunk)
        self.pp.handoff(self.model, len(ids), block_table=blocks, slot=r)

    def prefill_begin(self, r):
        self.pp.reset()

    def prefill_span(self, ids, first_chunk, n_seen, chunk):
        self.pp.run_span(torch.as_tensor(ids), first_chunk, n_seen, drain=False, chunk=chunk)

    def prefill_end(self, ids, first_chunk, n_tokens, r, blocks, chunk):
        self.pp.run_span(torch.as_tensor(ids), first_chunk, n_tokens, drain=True, chunk=chunk)
        self.pp.handoff(self.model, n_tokens, block_table=blocks, slot=r)

    def clear_slot(self, r):
        self.model.clear_slot(r)

    def first_token(self, r, tok, pos, preserve):
        k = r // ROWS
        self.model.step(self._token_states({r: (tok, pos)}, [k], {k} if preserve else ()), launches=[k])
        self.sync()
        return self._logits([r])[0]

    def decode(self, feed, launches, preserve_launches=()):
        if feed is None:
            self.model.step(None, launches=launches)
        else:
            self.model.step(self._token_states(feed, launches, preserve_launches), launches=launches)

    def _logits(self, slots):
        """[len(slots), vocab] logits of the given slots, read from their launches only"""
        ks = sorted({r // ROWS for r in slots})
        lg = self.model.logits(launches=ks)  # [8 * len(ks), n, vocab_chip] in the order of ks
        lg = lg[:, :, : self.vocab_chip].reshape(lg.shape[0], -1)[:, : self.vocab].float()
        idx = [ks.index(r // ROWS) * ROWS + r % ROWS for r in slots]
        return lg[torch.as_tensor(idx, dtype=torch.long)].clone()

    def read_logits(self, slots):
        self.sync()
        return self._logits(slots)

    def logits_from(self, read):
        return read

    def set_sampling(self, per_slot):
        default = (0.0, 1, 1.0, 0)
        self.model.set_sampling(rows=[per_slot.get(r, default) for r in range(self.n_slots)])

    def read_tokens(self, slots, async_read):
        """enqueue the reads of the out words (token, position, active, 0 per row) of the slots' launches: (slots,
        {launch: host read}, events)"""
        ks = sorted({r // ROWS for r in slots})
        reads, events = {}, []
        for k in ks:
            out = self.model.out_tensor(k)
            if isinstance(out, torch.Tensor):  # a fake model's host-side out words
                reads[k] = out.clone()
                continue
            import ttnn

            chip0 = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 0))
            reads[k] = ttnn.from_device(ttnn.get_device_tensors(out)[0], blocking=not async_read)
            if async_read:
                events.append(ttnn.record_event(self.mesh, 0, device_range=chip0))
        return slots, reads, events

    def read_events(self, read):
        return read[2]

    def tokens_from(self, read):
        slots, reads, _ = read
        rows = {}
        for k, host in reads.items():
            if not isinstance(host, torch.Tensor):
                import ttnn

                host = ttnn.to_torch(host)
            rows[k] = host.reshape(-1)[: ROWS * 4].reshape(ROWS, 4).to(torch.int64)
        out = []
        for r in slots:
            w = rows[r // ROWS][r % ROWS]
            if int(w[2]) != 1:
                raise RuntimeError(f"slot {r}: its launch's out word is not active ({w.tolist()})")
            out.append(int(w[0]))
        return out
