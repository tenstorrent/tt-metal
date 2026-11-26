# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Multi-user fast path: batched decode traces and chunked GDN prefill.

Enabled by QWEN_FAST=1 in the vLLM adapter. The model weights, MoE, collectives, sampler, output pool and
serving command layer are the existing ones. What changes:
  * decode carries all 32 physical rows as one tensor through every layer (no per-row op loops);
    GDN uses the native in-place recurrent op over 8-slot state segments, full attention uses one
    paged SDPA call for all rows. Traces exist per active extent in segments (8/16/24/32 rows).
  * prefill runs GDN layers over the whole chunk with the native chunked delta-rule kernel.
  * positions and page tables are host-shadowed and uploaded per submission (small H2D writes).
"""

import os
import time

import torch

import ttnn

from .fast_gdn import SEG, FastGDN, make_prefill_consts, write_row
from .generator import QwenGenerator
from .prefill import BoundedPrefill, attention_chunks

DRAM = ttnn.DRAM_MEMORY_CONFIG
ROWS = 32
ROUTE_EXTENT = os.environ.get("QWEN_FAST_ROUTE_EXTENT", "1") == "1"
PREFIX_CACHE = os.environ.get("QWEN_PREFIX_CACHE", "0") == "1"
PREFIX_SNAPSHOTS = int(os.environ.get("QWEN_PREFIX_SNAPSHOTS", "16"))
PREFIX_CHUNK = int(os.environ.get("QWEN_PREFIX_CHUNK", "2048"))
PREFIX_MARGIN = int(os.environ.get("QWEN_PREFIX_MARGIN", "8"))
PREFIX_ALIGN = int(
    os.environ.get("QWEN_PREFIX_ALIGN", "256")
)  # snapshot positions (multiple of the 32-token block)  # tokens of re-rendered prompt tail to stay before  # the server's prefill budget (max_num_batched_tokens)
SPEC_K = int(os.environ.get("QWEN_SPEC", "0"))  # speculative verify width K (0 = off), see fast_spec.py


SHARDED_NORM = os.environ.get("QWEN_FAST_SHARDED_NORM", "1") == "1"
_NORM_CFG = {}


def _norm_cfg(rows):
    """Width-sharded RMSNorm over 8 cores for a [1, rows<=32, 2048] decode activation (the interleaved op runs on one
    core per tile row, i.e. one core: 25 us -> 9 us incl. both layout moves)."""
    if rows not in _NORM_CFG:
        shard = ttnn.create_sharded_memory_config(
            (32, 256),
            core_grid=ttnn.CoreGrid(y=1, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        prog = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 1), subblock_w=1, block_h=1, block_w=8, inplace=False
        )
        _NORM_CFG[rows] = (shard, prog)
    return _NORM_CFG[rows]


def fast_norm(layer, x, key, sharded=False):
    if sharded and SHARDED_NORM and x.shape[-2] <= 32:
        shard, prog = _norm_cfg(x.shape[-2])
        y = ttnn.rms_norm(
            ttnn.to_memory_config(x, shard),
            weight=layer.w[key],
            epsilon=layer.config.rms_norm_eps,
            program_config=prog,
            memory_config=shard,
            compute_kernel_config=layer.fast_compute,
        )
        return ttnn.to_memory_config(y, DRAM)
    x = ttnn.to_memory_config(x, DRAM)
    return ttnn.rms_norm(
        x, weight=layer.w[key], epsilon=layer.config.rms_norm_eps, compute_kernel_config=layer.fast_compute
    )


class FastBoundedPrefill(BoundedPrefill):
    GDN_CHUNK = 2048

    def __init__(self, model, generator):
        super().__init__(model)
        self.generator = generator
        self.consts = make_prefill_consts(model.mesh)

    def _gdn(self, layer, index, source, destination, slot, start, length):
        fstate = self.generator.fstate[index]
        for offset in range(0, length, self.GDN_CHUNK):
            n = min(self.GDN_CHUNK, length - offset)
            t32 = (n + 31) // 32 * 32
            x = self._read(source, offset, t32, valid_rows=n)  # [1, t32, 2048], zero padded
            norm = fast_norm(layer, x, "input_layernorm.weight")
            partial = layer.fast.prefill(
                ttnn.reshape(norm, (1, 1, t32, 2048)),
                fstate,
                slot=slot,
                start=start + offset,
                length=n,
                consts=self.consts,
            )
            out = ttnn.add(x, layer._reduce(ttnn.reshape(partial, (1, t32, 2048))), fast_and_approximate_mode=False)
            self._write(out, destination, offset, n)
            del x, norm, partial, out

    def prefill(self, token_ids, *, cache, slot=0, start=0, on_logits=None, logit_positions=None):
        if not self.prepared:
            raise RuntimeError("Prepare prefill before capturing decode")
        if self._busy:
            raise RuntimeError("BoundedPrefill is a serial workspace")
        token_ids = (
            token_ids.reshape(-1).to(torch.int32)
            if isinstance(token_ids, torch.Tensor)
            else torch.tensor(token_ids, dtype=torch.int32).reshape(-1)
        )
        length = token_ids.numel()
        if not 0 <= start < start + length <= self.model.max_seq_len:
            raise ValueError("Logical prefill exceeds the configured context")
        if length > self.capacity:
            raise ValueError(
                f"Prefill chunk of {length} tokens exceeds the {self.capacity}-row workspace (QWEN_PREFILL_ROWS)"
            )
        self._busy = True
        try:
            source, attended, destination = self.residual
            for offset in range(0, length, self.TOKEN_CHUNK):
                n = min(self.TOKEN_CHUNK, length - offset)
                host = torch.zeros((1, self.TOKEN_CHUNK), dtype=torch.int32)
                host[0, :n] = token_ids[offset : offset + n]
                self._refresh(self.tokens, host, ttnn.uint32)
                embedded = ttnn.reshape(self.model.embed(self.tokens), (1, self.TOKEN_CHUNK, 2048))
                self._write(embedded, source, offset, n)
                del embedded
            for index, (layer, states) in enumerate(zip(self.model.layers, cache)):
                if layer.kind == "linear_attention":
                    self._gdn(layer, index, source, attended, slot, start, length)
                else:
                    state = states[slot]
                    for offset, n, absolute in attention_chunks(layer.kind, length, start):
                        result = self._attention(layer, state, source, offset, n, absolute, length)
                        self._write(result, attended, offset, n)
                        del result
                for offset in range(0, length, layer.policy.moe_chunk):
                    n = min(layer.policy.moe_chunk, length - offset)
                    physical = 1 if n == 1 else ((n + 31) // 32) * 32
                    x = self._read(attended, offset, physical, valid_rows=n)
                    result = self._moe(layer, x)
                    self._write(result, destination, offset, n)
                    del result, x
                source, destination = destination, source
            if getattr(self.generator, "mtp", None) is not None:
                self.generator.mtp.prefill(self, source, token_ids, slot, start, length)
                self.generator._mtp_pending[slot] = True
            if on_logits is not None:
                offsets = (
                    range(0, length, self.LOGIT_CHUNK)
                    if logit_positions is None
                    else sorted({p // self.LOGIT_CHUNK * self.LOGIT_CHUNK for p in logit_positions})
                )
                for offset in offsets:
                    n = min(self.LOGIT_CHUNK, length - offset)
                    hidden = self._read(source, offset, self.LOGIT_CHUNK, valid_rows=n)
                    logits = self.model.terminal(hidden)
                    on_logits(offset, n, logits)
                    del logits, hidden
            self.generator.f_host_pos[slot] = start + length
            return self.model.terminal(self._read(source, length - 1, 1))
        finally:
            self._busy = False

    def prepare(self, *, cache, reset_state, progress=None):
        if self.prepared:
            return
        for layer in self.model.layers:
            for n in (32, 64, 128, 256, 512, 1024, 2048):
                layer.prepare_moe(n)
        try:
            warm_rows = min(2048, self.capacity)
            zeros = self._upload(
                torch.zeros((1, warm_rows, 2048), dtype=torch.bfloat16), self.model.boundary_dtypes["residual_dtype"]
            )
            for buffer in self.residual:
                self.transport.rows(zeros, buffer, count=warm_rows)
            del zeros
            # Build every prefill program before any decode trace is captured. Some ops (tilize, tiled
            # reshape, ...) keep device buffers inside their program-cache entry; if such an entry is first
            # created after capture, its buffer can sit in trace scratch memory and is overwritten by every
            # replay (observed as a device hang on the next prefill). Lengths: 1 and every multiple of 32 up
            # to the GDN chunk (covers all GDN/MoE/attention chunk shapes), plus one continuation (start > 0).
            self.prepared = True
            limit = min(self.GDN_CHUNK, self.model.max_seq_len)
            lengths = [1] + list(range(32, limit + 1, 32))
            started = time.monotonic()
            for i, n in enumerate(lengths):
                self.prefill(torch.ones(n, dtype=torch.int32), cache=cache, slot=0, start=0)
                if progress is not None and i % 8 == 0:
                    progress({"phase": "fast_prefill_warmup", "rows": n})
            if self.model.max_seq_len > 300:
                self.prefill(torch.ones(100, dtype=torch.int32), cache=cache, slot=0, start=0)
                self.prefill(torch.ones(200, dtype=torch.int32), cache=cache, slot=0, start=100)
            ttnn.synchronize_device(self.mesh)
            print(f"FAST_PREFILL_WARMUP {len(lengths) + 2} shapes {time.monotonic() - started:.1f}s", flush=True)
        except BaseException:
            self.prepared = False
            raise
        finally:
            reset_state()
        self._prepared_cache_signature = self._cache_signature(cache)
        self.prepared = True


class FastQwenGenerator(QwenGenerator):
    def __init__(self, model, *, max_batch_size=32, kv_cache=None, decode_buckets=None, **kwargs):
        if max_batch_size > ROWS:
            raise ValueError("Fast path supports at most 32 slots")
        super().__init__(
            model,
            max_batch_size=max_batch_size,
            kv_cache=kv_cache,
            decode_buckets=tuple(sorted({min(2, max_batch_size), max_batch_size})),
            **kwargs,
        )
        mesh = self.mesh
        self.fast_segment_buckets = tuple(range(1, (max_batch_size + SEG - 1) // SEG + 1))
        self.fstate = {
            i: FastGDN.new_state(mesh) for i, layer in enumerate(model.layers) if layer.kind == "linear_attention"
        }
        up = lambda t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT: ttnn.from_torch(
            t, dtype=dtype, layout=layout, device=mesh, memory_config=DRAM, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
        )
        self._up = up
        self.f_host_pos = [0] * ROWS
        self.f_cur_pos = up(torch.full((ROWS,), -1, dtype=torch.int32), ttnn.int32)
        self.f_rope_idx = up(torch.zeros((1, ROWS), dtype=torch.int32), ttnn.uint32)
        full = [states for states in self.cache if "page_table" in states[0]]
        self.page_width = full[0][0]["page_table"].shape[-1] if full else 8
        self.f_host_pages = torch.zeros((ROWS, self.page_width), dtype=torch.int32)
        for slot, state in enumerate(full[0][:ROWS] if full else []):
            pages = state.get("_host_pages")
            if pages is not None:
                self.f_host_pages[slot] = pages.reshape(-1)[: self.page_width]
        self.f_page_table = up(self.f_host_pages, ttnn.int32)
        self.f_ring_idx = up(torch.arange(SEG * 8, dtype=torch.int32), ttnn.uint32)
        self.f_active_f = up(torch.zeros(1, 1, ROWS, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.f_inactive_f = up(torch.ones(1, 1, ROWS, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.f_active_f32 = up(torch.zeros(1, 1, ROWS, 1), ttnn.float32, ttnn.TILE_LAYOUT)
        self.f_attn_mask = up(torch.zeros(1, ROWS, 4, 256), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self._fast_active = None
        self.prefill_executor = None  # release the legacy executor's residual workspace first
        self.prefill_executor = FastBoundedPrefill(model, self)
        # The fast path owns GDN state; drop the legacy per-slot recurrent/conv groups from remaps.
        self.slot_state.groups = [
            g for g in self.slot_state.groups if not (g[0].endswith(".recurrent") or g[0].endswith(".conv"))
        ]
        legacy_apply = self.slot_state.apply

        def apply(remap):
            legacy_apply(remap)
            self._fast_remap(remap)

        self.slot_state.apply = apply
        self._fast_upload_masks(force=True)
        self._in_prefill = False
        self._mtp_pending = {}
        # Device penalties (presence/frequency/repetition) via the common sampler's traced penalties program.
        self.device_penalties = True
        for name, default in (("presence_penalty", 0.0), ("frequency_penalty", 0.0), ("repetition_penalty", 1.0)):
            self.sampling_values[name] = [default] * 32
            self.slot_state.register_host(
                "sampling_values." + name,
                lambda name=name: self.sampling_values[name],
                lambda value, name=name: self.sampling_values.__setitem__(name, value),
            )

    # ------------------------------------------------------------------ sampling (device penalties)
    def configure_sampling(
        self,
        *,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        seed=0,
        slots=None,
        reset_seed_state=True,
        presence_penalty=None,
        frequency_penalty=None,
        repetition_penalty=None,
    ):
        from models.common.sampling.generator import SamplingParams

        slots = tuple(range(self.max_batch_size)) if slots is None else tuple(slots)
        rows = len(slots)

        def values(value):
            if isinstance(value, torch.Tensor):
                value = value.flatten().tolist()
            if isinstance(value, (list, tuple)):
                if len(value) != rows:
                    raise ValueError("Sampling parameter rows must match slots")
                return list(value)
            return [value] * rows

        for name, given, default in (
            ("presence_penalty", presence_penalty, 0.0),
            ("frequency_penalty", frequency_penalty, 0.0),
            ("repetition_penalty", repetition_penalty, 1.0),
        ):
            entries = values(default if given is None else given)
            for slot, value in zip(slots, entries):
                self.sampling_values[name][slot] = default if value is None else float(value)
        super().configure_sampling(
            temperature=temperature, top_k=top_k, top_p=top_p, seed=seed, slots=slots, reset_seed_state=reset_seed_state
        )
        if (
            any(
                v != d
                for name, d in (("presence_penalty", 0.0), ("frequency_penalty", 0.0), ("repetition_penalty", 1.0))
                for v in self.sampling_values[name]
            )
            or self.sampler._penalties_active
        ):
            temps, ks = self.sampling_values["temperature"], self.sampling_values["top_k"]
            greedy = [t == 0 or k == 1 for t, k in zip(temps, ks)]
            limit = self.sampler.tt_sampling.max_top_k
            self.sampler.reset_sampling_params(
                SamplingParams(
                    temperature=torch.tensor([1.0 if g else 1.0 / t for g, t in zip(greedy, temps)]),
                    top_k=torch.tensor([1 if g else min(limit, max(1, k)) for g, k in zip(greedy, ks)]),
                    top_p=[0.0 if g else p for g, p in zip(greedy, self.sampling_values["top_p"])],
                    presence_penalty=list(self.sampling_values["presence_penalty"]),
                    frequency_penalty=list(self.sampling_values["frequency_penalty"]),
                    repetition_penalty=list(self.sampling_values["repetition_penalty"]),
                )
            )

    def reset_device_penalty_history(self, prompt_history, output_history, slots):
        pen = self.sampler.tt_penalties
        slots = list(slots)

        def usable(history):
            return history is not None and torch.as_tensor(history).ndim == 2 and torch.as_tensor(history).shape[-1] > 0

        if usable(prompt_history):
            pen.reset_prompt_tokens(torch.as_tensor(prompt_history), slots=slots)
        pen.reset_output_tokens(torch.as_tensor(output_history) if usable(output_history) else None, slots=slots)

    def _sampling_forward(self, penalties=False):
        from .slot_state import advance_seeds, merge_rows

        if not penalties:
            return super()._sampling_forward()
        ttnn.copy(self.tokens, self.previous_tokens)
        self.sampler._run_sampling(
            self.sampler_logits, penalties_on=True, grammar_on=False, tt_out_tok=self.tokens, count_tokens=False
        )
        merge_rows(self.tokens, self.previous_tokens, self.active_mask, self.mesh)
        ttnn.copy(self.previous_tokens, self.tokens)
        # Count the committed token. update_output_tokens deallocates its input, so hand it a trace-local copy.
        self.sampler.tt_penalties.update_output_tokens(ttnn.clone(self.tokens))
        advance_seeds(self.sampler.tt_sampling.seeds_tt_tensor, self.active_mask, self.mesh)

    def sampling_trace_key(self):
        key = super().sampling_trace_key()
        return key + "_pen" if self.sampler._penalties_active else key

    def replay_sampling(self):
        key = self.sampling_trace_key()
        if key not in self.traces:
            raise RuntimeError(f"Sampling execution signature {key} was not prepared")
        self.sampler.tt_sampling._sampling_draw_width = 64 if key.startswith("sampling_64") else 32
        ttnn.execute_trace(self.mesh, self.traces[key], cq_id=0, blocking=False)
        self.sampling_trace_replays[key] = self.sampling_trace_replays.get(key, 0) + 1
        self.counters["sampling_replays"] += 1

    # ------------------------------------------------------------------ host-shadowed inputs
    def _fast_upload_masks(self, force=False):
        active = tuple(self.active_slots)
        if not force and active == self._fast_active:
            return
        m = torch.zeros(ROWS)
        m[list(active)] = 1
        self._copy_host(m.reshape(1, 1, ROWS, 1).bfloat16(), self.f_active_f)
        self._copy_host((1 - m).reshape(1, 1, ROWS, 1).bfloat16(), self.f_inactive_f)
        self._copy_host(m.reshape(1, 1, ROWS, 1).float(), self.f_active_f32)
        self._copy_host(m.reshape(1, ROWS, 1, 1).expand(1, ROWS, 4, 256).bfloat16(), self.f_attn_mask)
        self._fast_active = active

    def _set_active_slots(self, slots):
        super()._set_active_slots(slots)
        if hasattr(self, "f_active_f"):
            self._fast_upload_masks()

    def _refresh(self, tokens, positions, slots):
        from .slot_state import merge_rows

        ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        for row, slot in enumerate(slots):
            ids[..., slot] = int(tokens[row])
            self.f_host_pos[slot] = int(positions[row])
        self._copy_host(ids, self.incoming_tokens)
        merge_rows(self.incoming_tokens, self.tokens, self.active_mask, self.mesh)
        self.counters["token_refreshes"] += 1

    def _bind_pages(self, page_table, cache, slots, page_tables_per_layer=None):
        if page_table is None and page_tables_per_layer is None:
            return
        # The legacy per-slot tables are read only by the prefill executor (full-attention fill/SDPA), and every
        # prefill rebinds its own slot. Decode reads the shared fast table, so skip the 10-layer per-slot copies.
        if getattr(self, "_in_prefill", True):
            super()._bind_pages(page_table, cache, slots, page_tables_per_layer)
        tables = [page_table] if page_tables_per_layer is None else list(page_tables_per_layer)
        changed = False
        for row, slot in enumerate(slots):
            row_pages = torch.as_tensor(tables[0][row], dtype=torch.int32).reshape(-1)
            for other in tables[1:]:
                if not torch.equal(torch.as_tensor(other[row], dtype=torch.int32).reshape(-1), row_pages):
                    raise ValueError("Fast decode requires identical page tables across attention layers")
            new = torch.zeros(self.page_width, dtype=torch.int32)
            new[: row_pages.numel()] = row_pages[: self.page_width]
            if row_pages.numel() < self.page_width:
                new[row_pages.numel() :] = row_pages[-1] if row_pages.numel() else 0
            if not torch.equal(new, self.f_host_pages[slot]):
                self.f_host_pages[slot] = new
                changed = True
        if changed:
            self._copy_host(self.f_host_pages, self.f_page_table)

    def fast_before_replay(self):
        if getattr(self, "spec", None) is not None:
            self.spec.ensure_ordinary(getattr(self, "_spec_pending_accepted", None))
            self._spec_pending_accepted = None
        cur = torch.full((ROWS,), -1, dtype=torch.int32)
        rope = torch.zeros((1, ROWS), dtype=torch.int32)
        for slot in self.active_slots:
            cur[slot] = self.f_host_pos[slot]
            rope[0, slot] = self.f_host_pos[slot]
        self._copy_host(cur, self.f_cur_pos)
        self._copy_host(rope, self.f_rope_idx)

    def fast_after_replay(self):
        for slot in self.active_slots:
            self.f_host_pos[slot] += 1

    def _fast_remap(self, remap):
        from .serving_state import normalize_remap

        order = normalize_remap(remap, self.max_batch_size)
        if order == tuple(range(self.max_batch_size)):
            return
        if getattr(self, "spec", None) is not None:
            self.spec.ensure_ordinary(getattr(self, "_spec_pending_accepted", None))
        moved = [(dst, src) for dst, src in enumerate(order) if dst != src]
        # host shadows
        pos = list(self.f_host_pos)
        pages = self.f_host_pages.clone()
        cursor = self.__dict__.setdefault("_prefix_cursor", {})
        old_cursor = dict(cursor)
        for dst, src in moved:
            self.f_host_pos[dst] = pos[src]
            self.f_host_pages[dst] = pages[src]
            if src in old_cursor:
                cursor[dst] = old_cursor[src]
            else:
                cursor.pop(dst, None)
        self._copy_host(self.f_host_pages, self.f_page_table)
        # device GDN state: snapshot sources, then write destinations
        for state in self.fstate.values():
            snaps = {}
            for _, src in moved:
                view, index = FastGDN.slot_view(state, src)
                snaps[src] = (
                    ttnn.clone(view[index : index + 1]),
                    [ttnn.clone(c[:, :, src : src + 1, :]) for c in state["conv"]],
                )
            for dst, src in moved:
                rec, conv = snaps[src]
                view, index = FastGDN.slot_view(state, dst)
                ttnn.fill_cache(view, rec, index)
                for c, row in zip(state["conv"], conv):
                    write_row(c, row, dst, self.prefill_executor.consts)

    # ------------------------------------------------------------------ decode graph
    def _fast_decode(self, segments):
        m = self.model
        tok = ttnn.reshape(self.tokens, (1, ROWS))
        if getattr(self, "mtp", None) is not None:
            ttnn.copy(tok, self.mtp.ord_in_tokens)  # the tokens this step consumes (MTP pending pairs)
        x = ttnn.reshape(m.embed(tok), (1, ROWS, 2048))
        cos = ttnn.reshape(ttnn.embedding(self.f_rope_idx, m.rope_cos, layout=ttnn.TILE_LAYOUT), (1, ROWS, 1, 64))
        sin = ttnn.reshape(ttnn.embedding(self.f_rope_idx, m.rope_sin, layout=ttnn.TILE_LAYOUT), (1, ROWS, 1, 64))
        for index, (layer, states) in enumerate(zip(m.layers, self.cache)):
            norm = fast_norm(layer, x, "input_layernorm.weight", sharded=True)
            if layer.kind == "linear_attention":
                partial = layer.fast.decode(
                    ttnn.reshape(norm, (1, 1, ROWS, 2048)),
                    self.fstate[index],
                    active_f=self.f_active_f,
                    inactive_f=self.f_inactive_f,
                    active_f32=self.f_active_f32,
                    segments=segments,
                    ring_idx=self.f_ring_idx,
                )
                partial = ttnn.reshape(partial, (1, ROWS, 2048))
            else:
                partial = layer.fast_attn.decode(
                    norm,
                    states[0],
                    cur_pos=self.f_cur_pos,
                    page_table=self.f_page_table,
                    cos=cos,
                    sin=sin,
                    attn_mask=self.f_attn_mask,
                )
            x = ttnn.add(x, layer._reduce(partial, batched=True), fast_and_approximate_mode=False)
            x = ttnn.to_memory_config(x, DRAM)
            norm = fast_norm(layer, x, "post_attention_layernorm.weight", sharded=True)
            moe = layer._moe(
                norm,
                batched=True,
                route_activity_mask=self.active_mask,
                # only the bucket's rows: post-combine reduce then skips the dead tail rows (valid_token_rows)
                route_slots=tuple(range(segments * SEG)) if ROUTE_EXTENT else tuple(range(ROWS)),
                fuse_moe_route_plan=m.fuse_moe_route_plan,
            )
            x = ttnn.add(x, layer._reduce(moe, batched=True), fast_and_approximate_mode=False)
            x = ttnn.to_memory_config(x, DRAM)
        if getattr(self, "mtp", None) is not None:
            ttnn.copy(x, self.mtp.ord_hidden)  # final pre-norm hidden per slot (MTP input after an ordinary step)
        logits = m.terminal(x)
        ttnn.copy(ttnn.reshape(logits, (1, 1, ROWS, logits.shape[-1])), self.sampler_logits)

    def _put(self, value, dest):
        """Host -> device copy skipped when `dest` already holds exactly `value` (spec-step inputs repeat a lot)."""
        cache = self.__dict__.setdefault("_put_cache", {})
        key = id(dest)
        prev = cache.get(key)
        if prev is not None and prev.shape == value.shape and torch.equal(prev, value):
            return
        self._copy_host(value, dest)
        cache[key] = value.clone()

    def spec_verify(self, tokens, positions, num_valid, accepted, return_logits=False):
        """One speculative verify step (fast_spec.py). tokens/positions [B, 1+K]; returns sampled ids [B, 1+K]
        (or logits [B, 1+K, V] with return_logits) on host; rows that are not live are 0."""
        spec = self.spec
        T = spec.T
        live = spec.prepare_step(tokens, positions, num_valid, accepted)
        ttnn.execute_trace(self.mesh, self.traces[f"spec_verify_{spec.bucket(live)}"], cq_id=0, blocking=False)
        self.counters["model_replays"] += 1
        spec.finish_step(live)
        B = len(positions)
        if return_logits:
            lg = self.model.host_logits(self.sampler_logits)[0, 0]
            out = torch.zeros(B, T, lg.shape[-1])
            for u in live:
                out[u] = lg[u * T : (u + 1) * T]
            return out
        self.replay_sampling()
        flat = torch.as_tensor(self._read_tokens(tuple(range(ROWS)))).reshape(-1)
        ids = torch.zeros(B, T, dtype=torch.int32)
        for u in live:
            ids[u] = flat[u * T : (u + 1) * T]
        return ids

    def spec_propose(self, after_verify, positions, accepted=None):
        """Draft K tokens per live user with the MTP layer; returns host drafts [32, K] (row = slot).
        after_verify: the step just run was spec_verify (positions [B, 1+K], accepted [B]) or an ordinary decode
        (positions [B] = the position each live slot's step processed, < 0 for idle rows)."""
        spec, mtp = self.spec, self.mtp
        T, K = spec.T, spec.K
        rows, users = [], {}
        if after_verify:
            positions = torch.as_tensor(positions).reshape(len(positions), -1)
            for u in range(min(spec.U, len(positions))):
                p = int(positions[u, 0])
                if p < 0:
                    continue
                a = int(accepted[u])
                for j in range(a):
                    r = u * T + j
                    rows.append((r, r, r, p + j + 1, u))
                users[u] = (u * T + a - 1, p + a + 1)
        else:
            positions = torch.as_tensor(positions).reshape(-1)
            for u in range(min(spec.U, len(positions))):
                p = int(positions[u])
                if p < 0:
                    continue
                if self._mtp_pending.get(u):
                    rows.append((2 * u, 64 + u, 32 + u, p, u))  # (h of the last prompt position, token it sampled)
                    self._mtp_pending[u] = False
                rows.append((2 * u + 1, 32 + u, u, p + 1, u))  # (this step's hidden, the token it sampled)
                users[u] = (2 * u + 1, p + 2)
        mtp.plan(rows, users)
        ttnn.execute_trace(self.mesh, self.traces["spec_propose"], cq_id=0, blocking=False)
        flat = ttnn.to_torch(ttnn.get_device_tensors(mtp.drafts_all)[0]).reshape(-1).to(torch.int32)  # one readback
        out = flat[: K * ROWS].reshape(K, ROWS).T.contiguous()
        return out, sorted(users)

    def _fast_key(self, segments):
        return f"fast_s{segments}"

    def model_trace_key(self, active_slots):
        extent = max(active_slots, default=0) + 1
        return self._fast_key((extent + SEG - 1) // SEG)

    def prepare_without_capture(self, *, prefill_cases=((32, 0),)):
        if self._variants_prepared:
            return self._preparation_seconds
        started = time.monotonic()
        if SPEC_K and getattr(self, "spec", None) is None:
            from .fast_mtp import FastMTP
            from .fast_spec import SpecVerify

            self.spec = SpecVerify(self, SPEC_K)
            target = next(states for states in self.cache if "page_table" in states[0])
            self.mtp = FastMTP(self.model, self, target[0]["k"].shape, target[0]["k"].dtype)
        self.prefill_executor.prepare(cache=self.cache, reset_state=self.reset)  # also warms the MTP prefill
        if getattr(self, "spec", None) is not None:
            self.spec.warm()
            # build the verify / propose programs eagerly once (all rows inactive) before any capture
            self._set_active_slots(())
            for rows in self.spec.buckets():
                self.spec.forward(rows)
            self.mtp.propose_forward()
        if PREFIX_CACHE and getattr(self, "prefix_store", None) is None:
            from .prefix_cache import GDNSnapshotStore

            # persistent pools + warmed save/load programs, both before any decode trace is captured
            self.prefix_store = GDNSnapshotStore(self, PREFIX_SNAPSHOTS)
            self.prefix_store.warm(slot=0)
        for segments in self.fast_segment_buckets:
            slots = tuple(range(min(segments * SEG, self.max_batch_size)))
            self._set_active_slots(slots)
            self._refresh([1] * len(slots), [1] * len(slots), slots)
            self.fast_before_replay()
            self._fast_decode(segments)
        for width in (32, 64):
            self.sampler.tt_sampling._sampling_draw_width = width
            self.sampler.precompile(self.sampler_logits, tt_out_tok=self.tokens)
            self._sampling_forward()
            self._sampling_forward(penalties=True)
        self.sampler.tt_penalties.reset_output_tokens()
        from .slot_state import merge_rows

        merge_rows(
            self.sampling_seed_updates, self.sampler.tt_sampling.seeds_tt_tensor, self.sampling_update_mask, self.mesh
        )
        self.prefill_logit_rows.prepare(source=self.zero_logits_row, destination=self.sampler_logits)
        self.slot_state.prepare()
        self.output_pool.prepare(token_template=self.tokens, logits_template=self.sampler_logits)
        self.reset()
        ttnn.synchronize_device(self.mesh)
        self._preparation_seconds = time.monotonic() - started
        self._prepared_decode_buckets = self.decode_buckets
        self._variants_prepared = True
        view = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
        print(
            "FAST_GENERATOR_PREPARED",
            round(self._preparation_seconds, 1),
            f"DRAM free/bank {int(view.total_bytes_free_per_bank) / 2**30:.2f} GiB " f"(x{int(view.num_banks)} banks)",
            flush=True,
        )
        return self._preparation_seconds

    def prepare(self, *, prefill_cases=((32, 0),), slots=(0,)):
        if self.prepared:
            return
        seconds = self.prepare_without_capture(prefill_cases=prefill_cases)
        for segments in self.fast_segment_buckets:
            started = time.monotonic()
            key = self._fast_key(segments)
            trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            try:
                self._fast_decode(segments)
            finally:
                ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
                self.traces[key] = trace
            print("MODEL_CAPTURED", key, flush=True)
            self.counters["captures"] += 1
            self._record_capture(key, trace, time.monotonic() - started, seconds, min(segments * SEG, ROWS))
        if getattr(self, "spec", None) is not None:
            for rows in self.spec.buckets():
                started = time.monotonic()
                trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                try:
                    self.spec.forward(rows)
                finally:
                    ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
                    self.traces[f"spec_verify_{rows}"] = trace
                print("MODEL_CAPTURED", f"spec_verify_{rows}", flush=True)
                self.counters["captures"] += 1
                self._record_capture(f"spec_verify_{rows}", trace, time.monotonic() - started, seconds)
            trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            try:
                self.mtp.propose_forward()
            finally:
                ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
                self.traces["spec_propose"] = trace
            print("MODEL_CAPTURED spec_propose", flush=True)
            self.counters["captures"] += 1
        for width, key, pen in (
            (32, "sampling", False),
            (32, "sampling_pen", True),
        ):
            self.sampler.tt_sampling._sampling_draw_width = width
            started = time.monotonic()
            trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            try:
                self._sampling_forward(penalties=pen)
            finally:
                ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
                self.traces[key] = trace
            print("SAMPLER_CAPTURED", key, flush=True)
            self.counters["captures"] += 1
            self._record_capture(key, trace, time.monotonic() - started, seconds)
        self.traces.setdefault("model_fixed", self.traces[self._fast_key(self.fast_segment_buckets[-1])])
        self.prepared = True
        self.reset()

    def _record_capture(self, key, trace, seconds, preparation_seconds, physical_rows=32):
        print(f"QWEN_TRACE_EVENT capture key={key} seconds={seconds:.1f}", flush=True)

    # ------------------------------------------------------------------ state lifecycle
    def reset(self):
        super().reset()
        if getattr(self, "spec", None) is not None:
            for u in range(self.spec.U):
                self.spec.drop(u)
            self.spec.restore_params()
        self.__dict__.setdefault("_mtp_pending", {}).clear()
        if hasattr(self, "fstate"):
            for state in self.fstate.values():
                FastGDN.reset_state(state)
            self.f_host_pos = [0] * ROWS
            self._fast_upload_masks(force=True)

    def _prefill_forward(self, tokens, **kwargs):
        if getattr(self, "prefix_store", None) is not None and not kwargs.get("return_all_logits"):
            return self._prefix_prefill(tokens, **kwargs)
        return self._plain_prefill_forward(tokens, **kwargs)

    def _prefix_prefill(self, tokens, **kwargs):
        """Prefix-cache aware prefill (see prefix_cache.py): per row, restore a GDN snapshot on a vLLM prefix hit,
        and snapshot the state at the last 32-aligned position of the prompt (splitting the chunk there)."""
        from .prefix_cache import block_hashes

        tokens = torch.as_tensor(tokens)
        ends = torch.as_tensor(kwargs["prompt_lens"]).reshape(-1).tolist()
        rows = len(ends)
        starts = (
            [0] * rows if kwargs.get("start_pos") is None else torch.as_tensor(kwargs["start_pos"]).reshape(-1).tolist()
        )
        sel = kwargs.get("slots") if kwargs.get("slots") is not None else kwargs.get("empty_slots")
        slots = list(range(rows)) if sel is None else torch.as_tensor(sel).reshape(-1).tolist()
        slot_key = "slots" if kwargs.get("slots") is not None else "empty_slots"
        cursor = self.__dict__.setdefault("_prefix_cursor", {})
        store = self.prefix_store
        outputs = []
        for row in range(rows):
            slot, start, end = slots[row], starts[row], ends[row]
            sub = dict(kwargs)
            sub["prompt_lens"] = [end]
            sub.pop("slots", None)
            sub.pop("empty_slots", None)
            sub[slot_key] = [slot]
            if kwargs.get("page_table") is not None:
                sub["page_table"] = torch.as_tensor(kwargs["page_table"])[row : row + 1]
            if kwargs.get("page_tables_per_layer") is not None:
                sub["page_tables_per_layer"] = [
                    torch.as_tensor(t)[row : row + 1] for t in kwargs["page_tables_per_layer"]
                ]
            row_tokens = tokens[row : row + 1]
            begin = start
            if start > 0 and cursor.get(slot) != start:
                # a new request whose first `start` tokens hit vLLM's prefix cache (not a chunk continuation)
                t_restore = time.perf_counter()
                begin = store.restore(slot, row_tokens[0], start)
                t_restore = time.perf_counter() - t_restore
                print(
                    f"QWEN_PREFIX_CACHE slot={slot} vllm_hit={start} restored={begin} end={end} "
                    f"restore {t_restore * 1e3:.0f} ms {store.stats}",
                    flush=True,
                )
            # One snapshot per prompt, at the last block boundary at least PREFIX_MARGIN tokens before its end: the
            # next turn of a chat re-renders this prompt's tail (the generation prompt, e.g.
            # "<|im_start|>assistant\n<think>\n") differently, so vLLM's hit for it ends at that boundary.
            # Aligned to 256 (not just the 32-token KV block): the attention prefill runs 32..256-token chunks only from
            # 256-aligned positions and single tokens before that, so a restore resumes at an aligned position and
            # re-prefills the < 256-token gap up to vLLM's hit (cheap) instead of crawling token by token.
            snap = ((end - PREFIX_MARGIN) // PREFIX_ALIGN) * PREFIX_ALIGN
            points = [snap] if begin < snap <= end else []
            if end - start == PREFIX_CHUNK and end % PREFIX_CHUNK == 0:
                points = []  # a full-budget chunk of a longer prompt: no snapshot (the prompt continues)
            cuts = [begin] + points + ([end] if not points or points[-1] != end else [])
            hashes = block_hashes(row_tokens[0], snap) if points else {}
            result = None
            cap = self.prefill_executor.capacity
            for a0, b in zip(cuts[:-1], cuts[1:]):
                # a restore can land well before vLLM's hit: keep every call within the prefill workspace, cutting
                # at 256-aligned positions so the attention prefill stays on its chunked path
                a = a0
                while a < b:
                    e = b if b - a <= cap else max(a + 32, ((a + cap) // PREFIX_ALIGN) * PREFIX_ALIGN)
                    sub["start_pos"] = [a]
                    sub["prompt_lens"] = [e]
                    result = self._plain_prefill_forward(row_tokens, **sub)
                    a = e
                if b in points:
                    t_save = time.perf_counter()
                    store.save(slot, hashes[b])
                    print(f"QWEN_PREFIX_CACHE save at {b}: {(time.perf_counter() - t_save) * 1e3:.0f} ms", flush=True)
            cursor[slot] = end
            outputs.append(result)
        if rows == 1:
            return outputs[0]
        if all(isinstance(o, list) for o in outputs):
            return [x for o in outputs for x in o]
        return torch.cat(outputs, dim=0)

    def _plain_prefill_forward(self, tokens, **kwargs):
        start = kwargs.get("start_pos")
        slots = kwargs.get("slots") if kwargs.get("slots") is not None else kwargs.get("empty_slots")
        if getattr(self, "spec", None) is not None:
            self.spec.restore_params()
            for slot in torch.as_tensor(slots).reshape(-1).tolist() if slots is not None else []:
                self.spec.drop(slot)
        # zero fresh slots' GDN state before the chunked kernel reads it
        if start is not None and slots is not None:
            for slot, s in zip(
                torch.as_tensor(slots).reshape(-1).tolist(), torch.as_tensor(start).reshape(-1).tolist()
            ):
                if s == 0:
                    self._fast_zero_slot(slot)
        self._in_prefill = True
        try:
            return super()._prefill_forward(tokens, **kwargs)
        finally:
            self._in_prefill = False

    def _fast_zero_slot(self, slot):
        # prefill with start == 0 ignores the recurrent slot and overwrites it; conv rows are rewritten too.
        pass

    def close(self):
        if self.closed:
            return
        for handle in list(self.output_pool.leases):
            if handle is not None:
                self.output_pool.discard(handle)
        seen = set()
        for trace in self.traces.values():
            if id(trace) in seen:
                continue
            seen.add(id(trace))
            ttnn.release_trace(self.mesh, trace)
            self.counters["releases"] += 1
        self.traces.clear()
        self.closed = True

    def release_request(self, slot):
        # The base release resets this slot's sampling params to greedy defaults; with device penalties that
        # also resets its penalty values (configure_sampling below) and clears its device output counts.
        result = super().release_request(slot)
        if self.sampler._penalties_active or self.sampling_values["presence_penalty"][slot] != 0.0:
            self.configure_sampling(
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                seed=0,
                slots=[slot],
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
            )
            if self.sampler._penalties_active:
                self.sampler.tt_penalties.reset_output_tokens(None, slots=[slot])
        self.f_host_pos[slot] = 0
        self.__dict__.setdefault("_prefix_cursor", {}).pop(slot, None)
        if getattr(self, "spec", None) is not None:
            self.spec.drop(slot)
        return result

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table=None,
        kv_cache=None,
        slots=None,
        refresh_inputs=True,
        read_tokens=True,
        return_logits=False,
        **kwargs,
    ):
        """Legacy direct-call decode (tests/benchmarks): same contract as the base class."""
        slots = tuple(range(len(start_pos))) if slots is None else tuple(slots)
        if refresh_inputs:
            active_rows = [r for r, p in enumerate(start_pos) if int(p) >= 0]
            active = tuple(slots[r] for r in active_rows)
            self._set_active_slots(active)
            if page_table is not None:
                from .serving import _validated_pages

                pages = _validated_pages(self, page_table, None, slots, active, [int(p) for p in start_pos])
                self._bind_pages(None, self.cache, slots, page_tables_per_layer=pages)
            flat = torch.as_tensor(tokens).reshape(-1)
            self._refresh([int(flat[r]) for r in active_rows], [int(start_pos[r]) for r in active_rows], active)
        key = self.model_trace_key(self.active_slots)
        self.fast_before_replay()
        ttnn.execute_trace(self.mesh, self.traces[key], cq_id=0, blocking=False)
        self.fast_after_replay()
        self.counters["model_replays"] += 1
        if return_logits:
            return self.model.host_logits(self.sampler_logits)[0, 0, list(slots), :]
        self._sample()
        return torch.tensor(self._read_tokens(slots)) if read_tokens else self.tokens


def attach_fast_layers(model, layer, sd):
    """Called from QwenModel construction for every layer while its state dict is in memory."""
    from .fast_attn import FastAttention

    layer.fast_compute = ttnn.init_device_compute_kernel_config(
        model.mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    if layer.kind == "linear_attention":
        layer.fast = FastGDN(layer, sd, model.mesh)
    else:
        layer.fast_attn = FastAttention(layer, sd, model.mesh)
