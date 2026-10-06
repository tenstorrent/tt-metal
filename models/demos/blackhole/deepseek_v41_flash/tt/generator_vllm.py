# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vLLM (tt-vllm-plugin) adapter of DeepSeek-V4.1-Flash on the 4x8 Blackhole galaxy.

Registered with the plugin as ``TTDeepseekV41ForCausalLM`` through ``../vllm_metadata.json`` (``arch`` DeepseekV41ForCausalLM -> ``TT`` prefix added by the plugin;
``EXTRA_MODELS_DIR`` must point at ``models/demos/blackhole`` so that the plugin finds the bundle folder ``deepseek_v41_flash``). Contract implemented
(vllm-tt-plugin ``docs/MODEL_CAPABILITIES.md``, ``docs/DECODE_RELOAD_CONTRACT.md``): ``initialize_vllm_model``, ``get_max_tokens_all_users``, ``allocate_kv_cache``,
``prefill_forward``, ``decode_forward``, ``warmup_model_prefill`` / ``warmup_model_decode``, ``release_request`` and the class level ``model_capabilities``.

How the vLLM concepts map onto the model
  * KV pool. The model owns ONE paged pool per mesh row (``tt/paged_ops.PagedKVPool``: 128-token pages = vLLM ``block_size 128``, bf16 or fp8 rows, plus per-user
    window rings, compressor state and index keys) and its own page allocator / page table, user b living on mesh row b // U. vLLM's block ids therefore do NOT address
    device memory: ``allocate_kv_cache`` returns inert placeholders and the ``page_table`` argument of prefill / decode is not read. The vLLM blocks only do the admission
    accounting (``get_max_tokens_all_users`` = max_model_len x max_num_seqs tokens, the capacity the model pool is built for, so vLLM never admits more than the
    pool holds); a request keeps its model user (physical slot) for its whole life, whatever blocks vLLM gave it.
  * Request -> user. The plugin's state-slot protocol is used: ``empty_slots`` of ``prefill_forward`` is the free logical slot of every new request, ``slot_remap`` of
    ``decode_forward`` is the permutation the plugin applies to the logical slots between steps, ``release_request(slot)`` frees a finished request. Device state never
    moves; ``tt/vllm_state.SlotTable`` keeps logical -> physical (model user) on the host.
  * Batch. The model decodes ``B = 4 x U`` users per step (U = ceil(max_num_seqs / 4), at most 32; max_num_seqs 128 -> U 32): the plugin pads its decode batch to
    max_num_seqs, the adapter pads further to a multiple of 4 and feeds unused users token 0 at position 0 (a decode step costs the same for any occupancy).
  * Prefill. ``Model.prefill_forward`` prefills the whole B-user batch in traced chunks. For continuous batching the new requests are prefilled with ``active``
    masking (``dsv41_model.Model.prefill_forward(active=...)``): the users that are decoding keep their pages / rings / compressor state / Engram history. The cost of
    a prefill call is that of ALL B users at the longest new prompt (the model computes filler rows for the others). While the decode indexer is on (contexts beyond
    512 / 1024 tokens) the index-key slabs are shared by the users of a mesh row, so a prefill while other requests are live re-prefills those from their token history
    (prompt + generated tokens): exact in the model's own terms but costly; when nothing is live (a burst of requests) it is a plain prefill.
  * Sampling. The model samples greedily on the device (mesh-wide argmax of the head, ``sample_on_device_mode all``). ``max_device_top_k = 1`` makes the plugin choose
    HOST sampling for any request with temperature > 0 and top_k != 1 (and for penalties, logit_bias, ...): prefill / decode then return the fp32 logits
    ([N, 1, vocab]) and the plugin samples them. This works but reads B x 129280 fp32 logits back each decode step (slow at large batch). Logprobs requests are not
    supported (a clear error). Prefix caching, chunked prefill, async scheduling and speculative decoding (``DSV41_SPEC``) are off.
  * Traces. Decode and prefill run traced. The decode trace is released before every prefill (prefill and decode traces share DRAM and the page table / state are
    rewritten) and recaptured by the first decode afterwards. One ISL range per process: a prefill with a longer S_pad than the captured one tears the prefill trace
    down and re-captures it (see ``vllm_state.s_pad_bucket``; a second, much longer ISL can run out of DRAM).
"""

import os
import time

import torch
from loguru import logger

from models.demos.blackhole.deepseek_v41_flash.tt import vllm_state as VS

_PAD_TOKEN = 0


def _resolve_weights_dir(hf_config):
    """Checkpoint directory: DSV41_CKPT, else the config's _name_or_path / MODEL_WEIGHTS_DIR / HF_MODEL when it is a directory."""
    if os.environ.get("DSV41_CKPT"):
        return os.environ["DSV41_CKPT"]
    for cand in (
        os.environ.get("MODEL_WEIGHTS_DIR"),
        os.environ.get("HF_MODEL"),
        getattr(hf_config, "_name_or_path", None),
    ):
        if (
            cand
            and os.path.isdir(os.path.expanduser(cand))
            and os.path.exists(os.path.join(os.path.expanduser(cand), "config.json"))
        ):
            return os.path.expanduser(cand)
    return None


class DeepseekV41ForCausalLM:
    """tt-vllm-plugin model class (see the module docstring)."""

    model_capabilities = {
        "supports_prefix_caching": False,
        # chunked prefill / interleaving of prefill with decode (INTERLEAVE_NOTES.md): only with DSV41_VLLM_INTERLEAVE=1
        "supports_chunked_prefill": os.environ.get("DSV41_VLLM_INTERLEAVE", "0") == "1",
        "supports_async_decode": False,
        "supports_sample_on_device": True,
        # device sampling = greedy only: temperature > 0 with top_k != 1 falls back to host sampling on the logits this class returns
        "max_device_top_k": 1,
        "supports_device_penalties": False,
    }

    # ---- construction ------------------------------------------------------------------------------------------------------------------------
    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None, **kwargs
    ):
        """Build the 40-layer (``DSV41_LAYERS``) model on the 4x8 mesh for ``max_batch_size`` (= max_num_seqs) requests of up to ``max_seq_len`` tokens."""
        mesh = tuple(mesh_device.shape)
        if mesh != (VS.MESH_ROWS, 8):
            raise ValueError(f"DeepSeek-V4.1-Flash needs the (4, 8) Blackhole galaxy mesh, got {mesh}")
        if tt_data_parallel != 1:
            raise ValueError("DeepSeek-V4.1-Flash does not support data parallelism (tt_data_parallel must be 1)")
        B = VS.padded_batch(max_batch_size)
        ckpt = _resolve_weights_dir(hf_config)
        if ckpt is not None:
            os.environ.setdefault(
                "DSV41_CKPT", ckpt
            )  # tt/model_args.py reads it at import time: set before the model modules are imported
        if os.environ.get("DSV41_VLLM_INTERLEAVE", "0") == "1":
            os.environ.setdefault(
                "DSV41_PF_UMASK", "1"
            )  # carried prefill state of users outside a replay stays intact (read when the prefill trace is built)
            if os.environ.get("DSV41_PREFILL_DYN", "1") == "0":
                raise ValueError("DSV41_VLLM_INTERLEAVE needs the traced chunk prefill (DSV41_PREFILL_DYN != 0)")
        if int(os.environ.get("DSV41_SPEC", "0")) > 0:
            raise ValueError("DSV41_SPEC (speculative decoding) is not part of the vLLM interface; unset it")
        from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
        from models.demos.blackhole.deepseek_v41_flash.tt.generator import Generator
        from models.tt_transformers.tt.common import PagedAttentionConfig

        a, _, b = os.environ.get("DSV41_LAYERS", "0-39").partition("-")
        layer_ids = list(range(int(a), int(b or a) + 1))
        U = B // VS.MESH_ROWS
        page_params = default_page_params(max_seq_len, U)
        paged = PagedAttentionConfig(
            block_size=page_params["page_block_size"], max_num_blocks=page_params["page_max_num_blocks_per_dp"]
        )
        logger.info(
            f"DSV4.1 vLLM: building {len(layer_ids)} layers, max_num_seqs {max_batch_size} -> batch {B} (U={U}), max_seq_len {max_seq_len}"
        )
        args, model, _pool, _ = create_tt_model(
            mesh_device, B, max_seq_len, paged, layer_ids=layer_ids, log=lambda m: logger.info(m)
        )
        return cls(Generator([model], [args], mesh_device, tokenizer=args.tokenizer), max_batch_size, max_seq_len)

    def __init__(self, generator, max_num_seqs, max_seq_len):
        self.generator = generator  # tt/generator.Generator (auto_chunk); the model is generator.m
        self.m = generator.m
        self.max_num_seqs, self.max_seq_len = int(max_num_seqs), int(max_seq_len)
        self.B = int(self.m.B)
        self.slots = VS.SlotTable(self.max_num_seqs, self.B)
        self.book = VS.TokenBook(self.B, self.max_seq_len + 1024)
        self.interleave = os.environ.get("DSV41_VLLM_INTERLEAVE", "0") == "1"
        self.inprog = VS.PrefillTracker()  # users prefilled in chunks and not decoding yet (interleave mode)
        self.s_pad_cur = 0
        self.s_pad_policy = os.environ.get("DSV41_VLLM_S_PAD", "bucket")
        self.kv_cache = None
        self.timing = {}
        self._warm = False
        self._calls = {"prefill": 0, "decode": 0}

    @classmethod
    def get_max_tokens_all_users(
        cls, model_name="", num_devices=1, tt_data_parallel=1, max_model_len=None, max_num_seqs=None, **kwargs
    ):
        """Token capacity of the model pool: every user may reach max_model_len (the pool is built for exactly that, ``tt.common.default_page_params``). The plugin adds one
        block per user on top. DSV41_VLLM_MAX_TOKENS_ALL_USERS lowers it for admission tests (the model pool is NOT shrunk).
        """
        env = os.environ.get("DSV41_VLLM_MAX_TOKENS_ALL_USERS")
        if env:
            return int(env)
        return int(max_model_len or 65536) * int(max_num_seqs or 1)

    # ---- KV cache ----------------------------------------------------------------------------------------------------------------------------
    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """vLLM asks for [num_blocks, kv_heads, 128, head] per layer. The pool is already allocated by the model (``PagedKVPool``, 128-token pages): hand back inert
        placeholders and keep the shape for validation. The blocks are not addressed (module docstring)."""
        num_blocks, heads, block, head = kv_cache_shape
        if block != VS.PAGE_TOKENS:
            raise ValueError(
                f"the DSV4.1 paged pool uses {VS.PAGE_TOKENS}-token pages; vLLM block_size is {block} (set block_size=128)"
            )
        need = -(-(self.max_seq_len * self.max_num_seqs) // VS.PAGE_TOKENS)
        if num_blocks < min(need, -(-self.max_seq_len // VS.PAGE_TOKENS)):
            raise ValueError(f"vLLM allocated only {num_blocks} blocks, fewer than one request of max_model_len needs")
        logger.info(
            f"DSV4.1 vLLM: kv_cache_shape {tuple(kv_cache_shape)} x {num_layers} layers is accounting only; the model pool "
            f"({self.m.num_pages} pages/mesh row, {self.m.pool.dtype}) was built at load time"
        )
        self.kv_cache = [(torch.empty(0), torch.empty(0)) for _ in range(num_layers)]
        return self.kv_cache

    # ---- prefill -----------------------------------------------------------------------------------------------------------------------------
    def _chunk(self, max_len):
        return self.generator.prefill_chunk or self.generator.auto_chunk(max_len)

    def prefill_forward(
        self,
        tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        enable_trace=True,
        sampling_params=None,
        start_pos=None,
        **kwargs,
    ):
        """Prefill N new requests. tokens [N, L] right padded, prompt_lens [N], empty_slots [N] (logical slots; default: the lowest free ones).
        Returns the first generated tokens [N] (int32, greedy on device) when ``sampling_params`` is given, else fp32 logits [N, 1, vocab] for host sampling.
        """
        if kwargs.get("page_tables_per_layer") is not None:
            raise ValueError("hybrid per-layer page tables are not used by DeepSeek-V4.1-Flash")
        N = int(tokens.shape[0])
        lens = (
            [int(x) for x in (prompt_lens.tolist() if hasattr(prompt_lens, "tolist") else prompt_lens)]
            if prompt_lens is not None
            else [int(tokens.shape[1])] * N
        )
        device_sampling = sampling_params is not None
        if device_sampling:
            if VS.wants_logprobs(sampling_params):
                raise ValueError(
                    "logprobs are not supported by the DSV4.1 adapter (no on-device logprobs); request without logprobs"
                )
            if not VS.sampling_wants_greedy(sampling_params):
                raise ValueError(
                    "DSV4.1 samples greedily on the device (temperature 0); temperature > 0 must use host sampling (max_device_top_k=1 selects it)"
                )
        if max(lens) + 1 > self.max_seq_len:
            raise ValueError(f"prompt of {max(lens)} tokens does not fit max_model_len {self.max_seq_len}")
        if self.interleave:
            return self._prefill_interleaved(tokens, lens, empty_slots, start_pos, device_sampling, enable_trace)
        if start_pos is not None and any(
            int(x) != 0 for x in (start_pos.tolist() if hasattr(start_pos, "tolist") else start_pos)
        ):
            raise ValueError(
                "prefill with a start position (chunked prefill / prefix caching) needs DSV41_VLLM_INTERLEAVE=1"
            )
        if empty_slots is None:
            used = {self.slots.phys[s] for s in range(self.max_num_seqs) if self.slots.phys[s] in self.slots.live}
            free = [s for s in range(self.max_num_seqs) if self.slots.phys[s] not in used]
            if len(free) < N:
                raise RuntimeError(f"{N} prefills but only {len(free)} free slots")
            empty_slots = free[:N]
        empty_slots = [int(s) for s in empty_slots]
        t0 = time.perf_counter()
        phys_new = self.slots.claim(empty_slots)
        live_other = sorted(self.slots.live - set(phys_new))
        live_other = [p for p in live_other if int(self.book.n[p]) > 0]
        refill = (
            bool(live_other) and self.m.use_indexer
        )  # shared index-key slabs: re-prefill the live users too (module docstring)
        ctx = {p: self.book.context(p) for p in live_other} if refill else None
        toks_B, lens_B, active_B, index_of = VS.build_prefill_batch(self.B, phys_new, tokens.reshape(N, -1), lens, ctx)
        S = int(lens_B.max())
        chunk = self._chunk(S)
        s_pad = VS.s_pad_bucket(S, chunk, self.s_pad_policy)
        want_logits = not device_sampling
        self.m.release_trace()  # decode trace: recaptured by the next decode step
        logger.info(
            f"DSV4.1 prefill: {N} new request(s) in slots {empty_slots} (users {phys_new}), lens max {S}, chunk {chunk}, S_pad {s_pad}, "
            f"{'re-prefill of ' + str(len(live_other)) + ' live users, ' if refill else ''}{'host sampling' if want_logits else 'device sampling'}"
        )
        first, logits = self.m.prefill_forward(
            toks_B,
            lens_B,
            chunk=chunk,
            max_new_tokens=0,
            want_logits=want_logits,
            enable_trace=bool(enable_trace),
            s_pad_max=s_pad,
            active=active_B,
        )
        for i, p in enumerate(phys_new):
            self.book.set_prompt(p, tokens[i, : lens[i]].reshape(-1))
        rows = torch.tensor(phys_new, dtype=torch.long)
        self.timing["prefill"] = time.perf_counter() - t0
        self._calls["prefill"] += 1
        logger.info(f"DSV4.1 prefill done in {self.timing['prefill']:.2f} s (model: {self.m.timing})")
        if device_sampling:
            return first[rows].to(torch.int32).reshape(N)
        return logits[rows].float().reshape(N, 1, -1)

    def _interleave_chunk(self):
        c = (
            int(os.environ.get("DSV41_VLLM_CHUNK", "0"))
            or self.generator.prefill_chunk
            or self.generator.auto_chunk(self.max_seq_len)
        )
        c = int(c or 1024)
        if c % 128:
            raise ValueError(f"prefill chunk {c} must be a multiple of 128")
        return c

    def _prefill_interleaved(self, tokens, ends, empty_slots, start_pos, device_sampling, enable_trace):
        """Prefill of the new requests / prompt chunks while other requests keep decoding (``Model.prefill_interleaved``): tokens [N, >= max end] (every prompt from
        position 0), ``ends`` the end position of this chunk of every row (= the prompt length for a whole prompt), ``start_pos`` the start (0 for a new request).
        A row with start > 0 is a continuation chunk of a request that is prefilled over several steps; the first token returned for a chunk that is not the last one
        of its prompt is meaningless (the plugin discards it). Positions [start, end) are computed in windows of one chunk (a model-wide constant).
        """
        N = len(ends)
        starts = (
            [int(x) for x in (start_pos.tolist() if hasattr(start_pos, "tolist") else start_pos)]
            if start_pos is not None
            else [0] * N
        )
        toks = tokens.reshape(N, -1)
        if empty_slots is None:
            used = {self.slots.phys[s] for s in range(self.max_num_seqs) if self.slots.phys[s] in self.slots.live}
            free = [s for s in range(self.max_num_seqs) if self.slots.phys[s] not in used]
            if len(free) < N:
                raise RuntimeError(f"{N} prefills but only {len(free)} free slots")
            empty_slots = free[:N]
        empty_slots = [int(s) for s in empty_slots]
        t0 = time.perf_counter()
        items, phys = [], []
        for i in range(N):
            st, e, slot = starts[i], ends[i], empty_slots[i]
            if not 0 <= st < e:
                raise ValueError(f"chunk [{st}, {e}) of row {i} is empty or reversed")
            p = None
            if st > 0:
                p = self.inprog.find(st, toks[i, :st])
                if p is not None and self.m.__dict__.get("pf_resume", {}).get(p) != st:
                    p = None  # its carried prefill state was lost (re-captured trace / unmasked trace): recompute from position 0
                if p is not None:
                    self.slots.bind(slot, p)
                    self.slots.live.add(p)
                else:
                    logger.warning(
                        f"DSV4.1 chunk continuation at {st} without carried state: recomputing from position 0"
                    )
                    st = 0
            if p is None:
                p = self.slots.claim([slot])[0]
            items.append((p, toks[i], st, e))
            phys.append(p)
        if len(set(phys)) != N:
            raise ValueError(f"duplicate prefill slots {empty_slots}")
        S = max(ends)
        chunk = self._interleave_chunk()
        self.s_pad_cur = max(self.s_pad_cur, VS.s_pad_bucket(S, chunk, self.s_pad_policy))
        want_logits = not device_sampling
        logger.info(
            f"DSV4.1 prefill (interleave): chunks {[(it[2], it[3]) for it in items]} in slots {empty_slots} (users {phys}), chunk {chunk}, "
            f"S_pad {self.s_pad_cur}, {'host sampling' if want_logits else 'device sampling'}"
        )
        res = self.m.prefill_interleaved(items, chunk, s_pad_max=self.s_pad_cur, want_logits=want_logits)
        for i, p in enumerate(phys):
            self.book.set_prompt(p, toks[i, : ends[i]].reshape(-1))
            self.inprog.update(p, ends[i], toks[i])
        self.timing["prefill"] = time.perf_counter() - t0
        self._calls["prefill"] += 1
        logger.info(f"DSV4.1 prefill (interleave) done in {self.timing['prefill']:.2f} s (model: {self.m.timing})")
        if device_sampling:
            return torch.tensor([int(res[p][0]) for p in phys], dtype=torch.int32).reshape(N)
        return torch.stack([res[p][1] for p in phys]).float().reshape(N, 1, -1)

    # ---- decode ------------------------------------------------------------------------------------------------------------------------------
    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        slot_remap=None,
        **kwargs,
    ):
        """One decode step of the running requests. tokens [W, 1] (W = the plugin's padded width), start_pos [W] (position of the fed token, -1 = padding row),
        slot_remap [max_num_seqs] optional permutation of the logical slots. Returns the next tokens [W, 1] (int32, greedy on device) when ``sampling_params`` is
        given, else the fp32 logits [W, 1, vocab] of the step (host sampling)."""
        device_sampling = sampling_params is not None
        if kwargs.get("num_valid_drafts") is not None or kwargs.get("spec_mode") is not None:
            raise ValueError("speculative decoding is not part of the DSV4.1 vLLM interface")
        if slot_remap is not None:
            self.slots.apply_remap(slot_remap)
        W = int(tokens.shape[0])
        tok_B, pos_B, rows = VS.build_decode_inputs(
            self.B, self.slots, tokens, start_pos, parked=self.inprog.parked() if self.interleave else None
        )
        for _, p, _ in rows:
            self.inprog.drop(p)  # decoding now
        if device_sampling:
            if VS.wants_logprobs(sampling_params, [i for i, _, _ in rows]):
                raise ValueError(
                    "logprobs are not supported by the DSV4.1 adapter (no on-device logprobs); request without logprobs"
                )
            if not VS.sampling_wants_greedy(sampling_params, [i for i, _, _ in rows]):
                raise ValueError(
                    "DSV4.1 samples greedily on the device (temperature 0); temperature > 0 must use host sampling (max_device_top_k=1 selects it)"
                )
        if not rows:
            return (
                torch.zeros(W, 1, dtype=torch.int32) if device_sampling else torch.zeros(W, 1, self.m.args.vocab_size)
            )
        for i, p, pos in rows:
            if pos >= self.m.max_ctx:
                raise ValueError(f"position {pos} reaches the model context {self.m.max_ctx}")
            if (
                p not in self.slots.live
            ):  # a request decoded without a prefill in this process (plugin restarted its slots): nothing to read
                raise RuntimeError(f"decode of logical slot {i} (user {p}) that was never prefilled")
        t0 = time.perf_counter()
        out = self.m.decode_forward(tok_B, pos_B, enable_trace=bool(enable_trace), reload_inputs=True)
        for i, p, pos in rows:
            self.book.note_fed(p, pos, int(tok_B[p]))
        self.timing["decode"] = time.perf_counter() - t0
        self._calls["decode"] += 1
        if device_sampling:
            return VS.scatter_rows(out.reshape(-1, 1).to(torch.int32), rows, W)
        logits = self.m.read_logits().float()  # [B, vocab] host
        return VS.scatter_rows(logits.reshape(self.B, 1, -1), rows, W)

    # ---- request lifecycle / warmup ---------------------------------------------------------------------------------------------------------
    def release_request(self, slot):
        """The plugin finished / preempted the request of logical ``slot``: free its pages (the model user is reused by the next prefill into that slot)."""
        p = self.slots.release(int(slot))
        if p is not None:
            if hasattr(self.m, "release_user"):
                self.m.release_user(
                    p
                )  # frees the pages, keeps the user admitted with one page (the decode loop grows the pages of every user)
            else:
                self.m.pool.release(p)
            self.book.clear(p)
            self.inprog.drop(p)

    def warmup_model_prefill(self, kv_cache=None, enable_trace=True, *args, **kwargs):
        """No-op: the prefill trace is keyed by the (chunk, S_pad) of the first prompt and cannot be captured without one. The first request compiles and captures
        it (``DSV41_VLLM_WARMUP_ISL=<tokens>`` runs one synthetic prefill + decode at load time instead, see ``warmup_model_decode``).
        """
        return

    def warmup_model_decode(self, *args, **kwargs):
        """Optional synthetic warm-up (DSV41_VLLM_WARMUP_ISL tokens, one request): compiles the prefill chunk programs and the decode trace before the first request.
        Skipped by default: with the ISL unknown it would capture a prefill trace for the wrong S_pad."""
        isl = int(os.environ.get("DSV41_VLLM_WARMUP_ISL", "0"))
        if self._warm or isl <= 0 or not kwargs.get("enable_trace", True):
            return
        self._warm = True
        t0 = time.perf_counter()
        sp = type("SP", (), {"temperature": [0.0], "top_k": [1], "enable_log_probs": [False]})()
        toks = torch.randint(1000, 100000, (1, isl), dtype=torch.int32)
        first = self.prefill_forward(toks, prompt_lens=[isl], empty_slots=[0], enable_trace=True, sampling_params=sp)
        self.decode_forward(first.reshape(1, 1), torch.tensor([isl]), sampling_params=sp, enable_trace=True)
        self.release_request(0)
        logger.info(f"DSV4.1 vLLM warm-up (ISL {isl}) took {time.perf_counter() - t0:.1f} s")
