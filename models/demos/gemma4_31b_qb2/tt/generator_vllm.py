# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Translate vLLM scheduler updates into the Gemma4 TP4 generator's inputs."""

import time

import torch
from loguru import logger

import ttnn
from models.demos.gemma4_31b_qb2.tt.decode_rows import decode_row_sizes, rows_covering, verify_shapes, verify_users
from models.demos.gemma4_31b_qb2.tt.decoder import Decoder
from models.demos.gemma4_31b_qb2.tt.generator import CacheState, build_generator


class Gemma4ForCausalLM:
    _router = ttnn.FabricRouterConfig()
    _router.max_packet_payload_size_bytes = 8192
    model_capabilities = {
        "supports_prefix_caching": True,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "supports_device_penalties": False,
        "supports_chunked_prefill": True,
        "supports_spec_decode": True,
        "spec_ids_are_sampled": True,
        "max_device_top_k": 32,
        "fabric_config": {
            "config": ttnn.FabricConfig.FABRIC_1D_RING,
            "router_config": _router,
        },
    }
    decode_input_update_contract = 1
    MAX_PREFILL_TOKENS = 8192
    # Draft length the launch settled on in spec_plan; 0 without speculation.
    spec_k = 0

    def __init__(self, generator, max_batch_size, max_seq_len, *, vllm_config=None):
        # vLLM inspects this keyword before the TT loader supplies a live mesh.
        if vllm_config is not None:
            raise RuntimeError("Initialize the TT model through initialize_vllm_model")
        self.generator = generator
        self.mesh = generator.mesh
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.rows = max_batch_size
        # Full-attention layers keep their KV in 256-token scheduler blocks packed into the shared wide rows (see
        # allocate_kv_cache_per_layer and _tables).
        self.full_layers = frozenset(i for i, layer in enumerate(generator.model.layers) if not layer.sliding)
        self.entry = None
        self.cache = None
        # Decode traces of fewer rows over the same KV pool; a step runs the smallest one that covers its live rows.
        self.row_sizes = decode_row_sizes(max_batch_size)
        # Speculative verify traces: draft columns ride in extra decode rows, column j of user u is row j * users + u.
        # One trace per (users, draft length) of decode_rows.VERIFY_K_BY_USERS; a step with more live requests is a plain decode.
        self.verify_shapes = verify_shapes(max_batch_size, self.spec_k) if self.spec_k else ()
        self.verify_caches = {}
        self.sub_caches = {}
        self.cur_state = None
        self.prefill_slot_keys = {}
        self.next_prefill_key = 0

    def embed_input_ids(self, input_ids):
        raise RuntimeError("Use the TT prefill_forward/decode_forward interface")

    def forward(self, input_ids, positions):
        raise RuntimeError("Use the TT prefill_forward/decode_forward interface")

    def compute_logits(self, hidden_states):
        raise RuntimeError("The TT generator owns the terminal norm and LM head")

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, *, max_seq_len=262144, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1 or not 1 <= max_batch_size <= 32:
            raise ValueError("Gemma4 QB2 requires TP4, DP1, and batch 1..32")
        if not 128 <= max_seq_len <= 262144 or max_seq_len % 128:
            raise ValueError("Serving context must be 128..262144, aligned to 128-token pages")
        if max_seq_len % 1024:
            # Chunked prefill attention reads its start from a tensor (flexible chunked SDPA), which needs the page-table row
            # (max_seq_len / 128 int32 entries) to be a multiple of 32 bytes: a multiple of 8 pages.
            raise ValueError(
                "Serving context must be a multiple of 1024 tokens (a page-table row of whole 32-byte groups)"
            )
        if hf_config.text_config.hidden_size != 5376 or hf_config.text_config.num_hidden_layers != 60:
            raise ValueError("Expected the Gemma4 31B checkpoint")
        torch.set_num_threads(8)
        return cls(build_generator(mesh_device, max_seq_len=max_seq_len), max_batch_size, max_seq_len)

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        from vllm_tt_plugin.kv_cache_anchor import AnchoredSlidingWindowSpec

        if vllm_config.max_in_flight_tokens > 2 * cls.MAX_PREFILL_TOKENS:
            raise ValueError("Gemma4 QB2 supports at most 8192 batched tokens with async chunked prefill")
        config = vllm_config.model_config.hf_text_config
        block = vllm_config.cache_config.block_size
        if block != 128:
            raise ValueError("Gemma4 QB2 uses 128-token cache pages")
        # Equal accounting page sizes: a sliding block is 128 tokens of 16 x 256 values and a full-attention block is 256
        # tokens of 4 x 512 values, the same 2 MiB (one wide row). The scheduler gives the full group 256-token blocks and
        # the sliding groups 128-token blocks over one global ID space, so a full block owns one whole wide row where it
        # used to leave half of a separate tensor idle; the hash block stays 128 tokens (cache_config.block_size).
        # The runner divides these global head counts by the TP degree.
        page_bytes = 2 * block * config.num_key_value_heads * config.head_dim * 2
        full_block = 2 * block
        if 2 * full_block * config.num_global_key_value_heads * config.global_head_dim * 2 != page_bytes:
            raise ValueError("A 256-token full-attention block must have the page size of a sliding block")
        result = {}
        for i, kind in enumerate(config.layer_types):
            sliding = kind == "sliding_attention"
            common = dict(
                block_size=block if sliding else full_block,
                num_kv_heads=config.num_key_value_heads if sliding else config.num_global_key_value_heads,
                head_size=config.head_dim if sliding else config.global_head_dim,
                dtype=torch.bfloat16,
                page_size_padded=page_bytes,
            )
            result[f"model.layers.{i}.self_attn"] = (
                AnchoredSlidingWindowSpec(**common, sliding_window=config.sliding_window)
                if sliding
                else FullAttentionSpec(**common)
            )
        return result

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        """Speculation follows the number of live requests (decode_rows.VERIFY_K_BY_USERS); beyond that a step is plain.

        The draft cap above the last verify trace is 0, which needs the plugin's zero cap (k_by_rows entries may be 0).
        """
        from vllm_tt_plugin.spec_decode import DRAFTER_STATE_INTERNAL, SpecPlan, SpecReject

        shapes = verify_shapes(max_num_seqs, requested_k)
        if not shapes:
            return SpecReject(reason=f"no verify trace fits {max_num_seqs} slots and K={requested_k}", supported_k=())
        cls.spec_k = shapes[0][1]
        return SpecPlan(
            effective_k=cls.spec_k,
            lanes_per_request=1 + cls.spec_k,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=("argmax_ids",),
            drafter_state=DRAFTER_STATE_INTERNAL,
            supports_narrow_decode=True,
            k_by_rows=tuple((users, k) for users, k in shapes)
            + (((max_num_seqs, 0),) if max_num_seqs > shapes[-1][0] else ()),
        )

    @classmethod
    def get_max_tokens_all_users(cls, *, max_num_seqs, max_model_len=262144, **kwargs):
        # The shared pool uses one full-attention group and five sliding groups.
        # Each window can straddle nine pages. Reserve both async batches of
        # in-flight prefill tokens globally, in addition to the per-request tails.
        # The pool is one request of --max-model-len plus those windows, so a shorter context frees DRAM.
        # vLLM checks that one request of the full context fits, and that request also keeps the eight pages of its
        # prompt's cache-hit window (AnchoredSlidingWindowSpec): reserve them once. Many requests holding their
        # windows at once wait for blocks in the scheduler instead of growing the pool.
        # The plugin separately adds one output page per request.
        # An ID is one wide row: 128 sliding tokens or 256 full-attention tokens (allocate_kv_cache_per_layer), so the
        # same number of IDs holds twice the full-attention tokens; the first term stays max_model_len / 128 IDs.
        tails = (
            Decoder.SLIDING_WINDOW_PAGES * Decoder.PAGE_SIZE * max_num_seqs
            + (Decoder.SLIDING_WINDOW_PAGES - 1) * Decoder.PAGE_SIZE
        )
        return max_model_len + 5 * (tails + 2 * cls.MAX_PREFILL_TOKENS)

    @staticmethod
    def _full_view(wide):
        """The full-attention view of a wide sliding-geometry backing: [P, 4, 128, 256] -> [2P, 1, 128, 512], zero-copy."""
        views = []
        for root in wide:
            shape = tuple(int(x) for x in root.shape)
            if (
                len(shape) != 4
                or shape[1:] != (4, 128, 256)
                or root.dtype != ttnn.bfloat8_b
                or root.layout != ttnn.TILE_LAYOUT
            ):
                raise ValueError(f"Unsupported wide KV backing: {shape} {root.dtype} {root.layout}")
            view = ttnn.experimental.view(root, [2 * shape[0], 1, 128, 512])
            if (
                view.buffer_unique_id() != root.buffer_unique_id()
                or view.buffer_address() != root.buffer_address()
                or view.buffer_aligned_page_size() != root.buffer_aligned_page_size()
            ):
                raise RuntimeError("The full-attention view must alias the wide backing")
            views.append(view)
        return tuple(views)

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """One wide backing per pool position; its full-attention layer reads it as 2P half-pages.

        A pool position holds one full-attention layer and its sliding layers. The scheduler hands every block ID to one
        group at a time, so they can share memory: the backing has the sliding layers' geometry [P, 4, 128, 256] (P = IDs +
        scratch rows) and the full layer sees it as [2P, 1, 128, 512] through a zero-copy view. A full-attention
        block (256 tokens) owns the whole wide row b, native half-pages 2b and 2b + 1 (_tables); null block 0 stays unused.
        """
        pools, kv, scratch = {}, [], {}
        scratch_rows = max([self.rows] + [users * (1 + k) for users, k in self.verify_shapes])
        layers = self.generator.model.layers
        if len(per_layer_specs) != len(layers):
            raise ValueError("The allocator must supply one spec per layer")
        members = {}
        for i, (layer, (shape, dtype, pool_id)) in enumerate(zip(layers, per_layer_specs)):
            expected = (layer.kv_heads, 128 if layer.sliding else 256, layer.head_dim)
            if tuple(shape[1:]) != expected or dtype != torch.bfloat16 or (i in self.full_layers) == layer.sliding:
                raise ValueError(f"Unexpected allocator geometry or accounting dtype: {shape}, {dtype}")
            members.setdefault(pool_id, []).append((i, layer, int(shape[0])))
        for pool_id, group in members.items():
            blocks = {n for _, _, n in group}
            sliding = [layer for _, layer, _ in group if layer.sliding]
            if len(blocks) != 1 or not sliding or sum(not layer.sliding for _, layer, _ in group) != 1:
                raise ValueError(
                    "A pool position needs one full-attention layer and sliding layers with equal block counts"
                )
            n = blocks.pop()
            if n <= 1:
                raise ValueError("A pool must contain the null block and live blocks")
            wide = sliding[0].allocate_cache(physical_pages=n + scratch_rows)
            pools[pool_id] = (wide, self._full_view(wide))
        for i, (layer, (shape, dtype, pool_id)) in enumerate(zip(layers, per_layer_specs)):
            wide, view = pools[pool_id]
            kv.append(wide if layer.sliding else view)
            first = shape[0] if layer.sliding else 2 * shape[0]
            scratch[i] = list(
                range(first, first + (1 if layer.sliding else 2) * scratch_rows, 1 if layer.sliding else 2)
            )
        tables = {i: torch.zeros(self.rows, self.max_seq_len // 128, dtype=torch.int32) for i in range(len(kv))}
        self.cache = CacheState(
            kv,
            tables,
            self.rows,
            self.max_seq_len,
            scratch_pages={i: scratch[i][: self.rows] for i in scratch},
            vllm_owned=True,
        )
        self.sub_caches = {
            n: CacheState(
                kv,
                {i: torch.zeros(n, self.max_seq_len // 128, dtype=torch.int32) for i in range(len(kv))},
                n,
                self.max_seq_len,
                scratch_pages={i: scratch[i][:n] for i in scratch},
                vllm_owned=True,
            )
            for n in self.row_sizes
        }
        self.verify_caches = {
            users: CacheState(
                kv,
                {i: torch.zeros(users * (1 + k), self.max_seq_len // 128, dtype=torch.int32) for i in range(len(kv))},
                users * (1 + k),
                self.max_seq_len,
                scratch_pages={i: scratch[i][: users * (1 + k)] for i in scratch},
                vllm_owned=True,
                columns=1 + k,
            )
            for users, k in self.verify_shapes
        }
        return self.cache

    def _tables(self, cache, page_tables_per_layer, *, slots=None):
        if cache is not self.cache and all(
            cache is not sub for sub in (*self.sub_caches.values(), *self.verify_caches.values())
        ):
            raise ValueError("Serving must use the cache allocated by vLLM")
        if page_tables_per_layer is None or len(page_tables_per_layer) != len(cache.kv):
            raise ValueError("Hybrid attention requires a page table for every layer")
        result, translated = {}, {}
        for i, source in enumerate(page_tables_per_layer):
            # Share immutable host rows only within this update: vLLM mutates
            # its source tensors in place between scheduler steps. Layers of one group may be separate views of
            # one table.
            key = (
                source.data_ptr(),
                source.storage_offset(),
                tuple(source.shape),
                source.stride(),
                i in self.full_layers,
            )
            if key not in translated:
                dest = torch.zeros_like(cache.page_tables[i])
                rows, cols = source.shape
                if rows > cache.batch or cols > dest.shape[1]:
                    raise ValueError("Scheduler page table exceeds the configured cache")
                indices = list(range(rows)) if slots is None else slots
                if len(indices) > rows:
                    raise ValueError("Page table has fewer rows than prefill requests")
                # Prefill rows are compact, then padded to the decode batch.
                # Scatter only the live prefix into its assigned state slots.
                live = source[: len(indices)].to(torch.int32).clamp_min(0)
                if i in self.full_layers:
                    live = self._half_pages(live, dest.shape[1])
                dest[indices, : live.shape[1]] = live
                if slots is None:
                    for column in range(1, cache.columns):
                        dest[column * rows : (column + 1) * rows] = dest[:rows]
                translated[key] = dest
            result[i] = translated[key]
        return result

    @staticmethod
    def _half_pages(blocks, width):
        """Full-attention scheduler blocks (256 tokens) as native 128-token half-pages: block b is pages 2b and 2b + 1.

        The null block 0 maps to page 0 in both halves. The scheduler's table is as wide as for 128-token blocks; the
        entries past width // 2 columns belong to blocks that cannot exist and must be empty.
        """
        columns = width // 2
        if blocks[:, columns:].any():
            raise ValueError("A full-attention block table has live entries past its 256-token capacity")
        blocks = blocks[:, :columns]
        pages = torch.stack((2 * blocks, 2 * blocks + 1), dim=2)
        return torch.where(blocks[:, :, None] > 0, pages, torch.zeros_like(pages)).reshape(blocks.shape[0], -1)

    def _sampling(self, params, positions, *, batch=None, reset_seed=True):
        """Per-row sampling state for `positions`, one per row; a slot's params repeat in its draft columns."""
        batch = self.max_batch_size if batch is None else batch

        def rows(name):
            value = getattr(params, name)
            value = value.tolist() if isinstance(value, torch.Tensor) else value
            slots = list(value)[:batch] if isinstance(value, (list, tuple)) else [value] * batch
            return slots * (len(positions) // batch)

        for name, default in (
            ("presence_penalty", 0),
            ("frequency_penalty", 0),
            ("repetition_penalty", 1),
            ("enable_log_probs", False),
        ):
            if any(x != default for x in rows(name)):
                raise ValueError(f"The shared host sampler must handle {name}")
        seeds = [0] * len(positions)
        if reset_seed:
            modulus = 2**31 - self.generator.max_seq_len
            seeds = [
                int(torch.randint(0, modulus, ()).item()) if seed is None else int(seed) % modulus
                for seed in rows("seed")
            ]
            seeds = [seed + max(0, int(pos)) for seed, pos in zip(seeds, positions)]
        return dict(
            top_k=rows("top_k"), top_p=rows("top_p"), temperature=rows("temperature"), seed=seeds, reset_seed=reset_seed
        )

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        *,
        sampling_params=None,
        page_tables_per_layer=None,
        start_pos=None,
        empty_slots=None,
        **kwargs,
    ):
        starts = [0] * len(prompt_lens) if start_pos is None else [int(x) for x in start_pos]
        lengths = [int(x) for x in prompt_lens]
        slots = list(range(len(lengths))) if empty_slots is None else list(empty_slots)
        tables = self._tables(kv_cache, page_tables_per_layer, slots=slots)
        self.entry = None
        # Prefix pages can be shared by several live requests. History belongs
        # to a request's stable state slot, never to a shared physical page.
        keys = []
        for slot in slots:
            if slot not in self.prefill_slot_keys:
                self.prefill_slot_keys[slot] = self.next_prefill_key
                self.next_prefill_key += 1
            keys.append(self.prefill_slot_keys[slot])
        outputs = self.generator.prefill_forward(
            tokens,
            page_table=tables,
            kv_cache=kv_cache,
            prompt_lens=lengths,
            slots=slots,
            return_device=sampling_params is not None,
            start_positions=starts,
            history_keys=keys,
        )
        if sampling_params is None:
            return outputs
        params = self._sampling(sampling_params, [n - 1 for n in lengths], batch=len(lengths))
        return self.generator.sample_prefill_outputs(outputs, kv_cache, params)

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        *,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        page_tables_per_layer=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=True,
        reset_sampling_state=True,
        slot_remap=None,
        spec_mode=None,
        num_valid_drafts=None,
        **kwargs,
    ):
        if "reset_batch" in kwargs:
            raise TypeError("decode_input_update_contract=1 requires explicit reload flags")
        if not enable_trace:
            raise ValueError("Gemma4 QB2 requires traced decode")
        if reset_sampling_state and not reload_inputs:
            raise ValueError("Resetting sampling state requires current tokens and positions")
        if kv_cache is not self.cache:
            raise ValueError("Serving must use the cache allocated by vLLM")
        if spec_mode is not None:
            return self._verify(
                tokens, start_pos, kv_cache, page_tables_per_layer, sampling_params, num_valid_drafts, slot_remap
            )
        sample = sampling_params is not None
        slot_positions = start_pos
        tokens = self._pad_rows(tokens.reshape(-1, 1), 0)
        start_pos = self._pad_rows(start_pos.reshape(-1), -1)
        # The batch is front-packed, so the live rows are 0..n-1 and the smallest trace covering row n-1 runs the step.
        # A change of n is a layout change, which always reloads the inputs, so a trace switch needs no state moved.
        live = int((start_pos >= 0).nonzero().max()) + 1 if bool((start_pos >= 0).any()) else 1
        rows = rows_covering(live, self.row_sizes, self.rows)
        state = self.sub_caches.get(rows, kv_cache)
        if reload_inputs:
            tables = self._tables(state, [table[:rows] for table in page_tables_per_layer])
            # A row entering decode no longer needs its private prefill tail.
            # The plugin notifies slot moves after submission succeeds, so
            # histories still use source slots when a decode row is gathered.
            self._release_entering_decode(slot_positions, slot_remap)
            self.entry = self.generator.prepare_decode(
                tokens[:rows], start_pos[:rows], page_table=tables, kv_cache=state
            )
            self.cur_state = state
        elif (
            self.entry is None or state is not self.cur_state or self.generator.traces.get(id(state)) is not self.entry
        ):
            # Steps that keep the loaded inputs rely on the plugin reloading them after anything that changes the
            # trace: a prefill (note_prefill_submitted; one that compiles programs retires every trace) and any
            # change of the live rows (a different covering trace). This check turns a broken contract into an error.
            raise ValueError("A retired decode trace requires current scheduler inputs")
        elif reload_page_table:
            tables = self._tables(state, [table[:rows] for table in page_tables_per_layer])
            self.generator._tables(state, tables)
        if sample and (reload_sampling_params or reset_sampling_state):
            params = self._sampling(sampling_params, start_pos, reset_seed=reset_sampling_state)
            self.generator.configure_sampling(batch=self.rows, **params)
        self.generator.replay(self.entry, sample=sample)
        if not sample:
            return self.generator._host_logits(self.entry["logits"])[0, 0, : self.max_batch_size].unsqueeze(1)
        output = self.entry["tokens"]
        return self.process_decode_output_host(output, is_tokens=True) if read_from_device else output

    def _release_entering_decode(self, positions, slot_remap):
        """A row entering decode no longer needs its private prefill tail (every decode step reloads, so this is idempotent)."""
        for row, position in enumerate(positions):
            if int(position) >= 0:
                self.release_request(row if slot_remap is None else int(slot_remap[row]))

    def _pad_rows(self, values, fill):
        """Extend a [slots, ...] decode input to the trace's rows; the draft-column rows stay inactive."""
        padded = torch.full((self.rows, *values.shape[1:]), fill, dtype=values.dtype)
        padded[: values.shape[0]] = values
        return padded

    def _verify(
        self,
        tokens,
        start_pos,
        kv_cache,
        page_tables_per_layer,
        sampling_params,
        num_valid_drafts=None,
        slot_remap=None,
    ):
        """Verify a [slots, 1 + K] candidate block as one decode step: column j of user u is row j * users + u.

        The block holds the live requests in its first rows; the trace of decode_rows.VERIFY_K_BY_USERS that covers
        them verifies the columns up to its draft length (the plugin caps a request's drafts to the same table). Every
        row writes its K/V at its own position before attention reads, so a draft attends to the drafts before it
        exactly as sequential decode steps would. Each row samples with its slot's parameters and seed + position, the
        draw a plain decode at that position makes, so accepted tokens are the plain decode's tokens. The plugin caps
        the drafts of a step with more live requests than the table covers at 0; such a block carries no draft.
        """
        from vllm_tt_plugin.spec_decode import ACCEPT_MODE_ARGMAX_IDS, VerifyOutput

        slots, width = tokens.shape
        if slots != self.max_batch_size or width != 1 + self.spec_k:
            raise ValueError(f"Expected a [{self.max_batch_size}, {1 + self.spec_k}] candidate block")
        live = int((start_pos[:, 0] >= 0).nonzero().max()) + 1 if bool((start_pos[:, 0] >= 0).any()) else 1
        ids = torch.full((slots, width), -1, dtype=torch.int32)
        self._release_entering_decode(start_pos[:, 0], slot_remap)
        users = verify_users(live, self.verify_shapes)
        if users is None:
            # More live requests than any verify trace covers: the plan's draft cap is 0, so no row carries a draft. The
            # plugin still sends a verify block when a row committed several tokens on its last step (a model with
            # candidate state resolves it then); this model has none, so the step is a plain decode.
            if num_valid_drafts is None or bool((num_valid_drafts > 0).any()):
                raise ValueError(f"No verify trace covers {live} live requests and the block carries drafts")
            ids[:, 0] = self._decode_plain_ids(
                tokens[:, 0], start_pos[:, 0], kv_cache, page_tables_per_layer, sampling_params
            )
            return VerifyOutput(spec_mode=ACCEPT_MODE_ARGMAX_IDS, argmax_ids=ids)
        state = self.verify_caches[users]
        columns = state.columns
        block_tokens, block_positions = tokens[:users, :columns], start_pos[:users, :columns]
        positions = block_positions.t().reshape(-1)
        flat_tokens = torch.where(positions >= 0, block_tokens.t().reshape(-1), 0).reshape(-1, 1)
        tables = self._tables(state, [table[:users] for table in page_tables_per_layer])
        self.entry = self.generator.prepare_decode(flat_tokens, positions, page_table=tables, kv_cache=state)
        self.cur_state = state
        if sampling_params is None:
            self.generator.configure_sampling(batch=state.batch, temperature=0)
        else:
            self.generator.configure_sampling(
                batch=state.batch, **self._sampling(sampling_params, positions, batch=users)
            )
        self.generator.replay(self.entry)
        out = self.read_decode_output(self.entry["tokens"])
        out = ttnn.to_torch(out).reshape(-1)[: state.batch].to(torch.int32)
        ids[:users, :columns] = out.reshape(columns, users).t()
        return VerifyOutput(spec_mode=ACCEPT_MODE_ARGMAX_IDS, argmax_ids=ids)

    def _decode_plain_ids(self, tokens, start_pos, kv_cache, page_tables_per_layer, sampling_params):
        """One plain decode step from host inputs (the column-0 tokens of a draftless block); returns each slot's next token."""
        live = int((start_pos >= 0).nonzero().max()) + 1 if bool((start_pos >= 0).any()) else 1
        rows = rows_covering(live, self.row_sizes, self.rows)
        state = self.sub_caches.get(rows, kv_cache)
        tokens = self._pad_rows(tokens.reshape(-1, 1), 0)[:rows]
        start_pos = self._pad_rows(start_pos.reshape(-1), -1)
        tables = self._tables(state, [table[:rows] for table in page_tables_per_layer])
        self.entry = self.generator.prepare_decode(tokens, start_pos[:rows], page_table=tables, kv_cache=state)
        self.cur_state = state
        if sampling_params is None:
            self.generator.configure_sampling(batch=self.rows, temperature=0)
        else:
            self.generator.configure_sampling(batch=self.rows, **self._sampling(sampling_params, start_pos))
        self.generator.replay(self.entry)
        out = self.read_decode_output(self.entry["tokens"])
        return ttnn.to_torch(out).reshape(-1)[: self.max_batch_size].to(torch.int32)

    def read_decode_output(self, output, async_read=False):
        if isinstance(output, torch.Tensor):
            return (output, []) if async_read else output
        # Tokens are replicated. Read one rank instead of gathering the vocabulary.
        host = ttnn.get_device_tensors(output)[0].cpu(blocking=not async_read)
        self.generator.counters["token_readbacks"] += 1
        return (host, [ttnn.record_event(self.mesh, 0)]) if async_read else host

    def process_decode_output_host(self, output, is_tokens=True):
        if not is_tokens:
            return output.reshape(self.max_batch_size, 1, -1)
        if output.storage_type() != ttnn.StorageType.HOST:
            output = self.read_decode_output(output)
        return ttnn.to_torch(output).reshape(-1)[: self.max_batch_size].long().reshape(-1, 1)

    def warmup_model_prefill(self, kv_cache, enable_trace, **kwargs):
        # Every 32-token prompt length from position 0 is compiled here: a first-time length costs up to tens of
        # seconds of program compilation in the middle of a request, and retires the decode trace.
        # The plugin's two-phase warm-up calls this twice for the same cache; the second pass would repeat the whole
        # lattice (minutes of prefill compute) for programs the cache already holds.
        # The lattice never exceeds the cache: a --max-model-len below the prefill limit has fewer page-table columns and
        # rejects longer prompts.
        limit = min(self.MAX_PREFILL_TOKENS, kv_cache.capacity)
        marker = (id(kv_cache), limit)
        if getattr(self, "_prefill_warmed", None) == marker:
            logger.info("Prefill length classes already warmed for this cache; skipping")
            return
        pages = limit // Decoder.PAGE_SIZE
        tables = {i: torch.zeros_like(table) for i, table in kv_cache.page_tables.items()}
        for table in tables.values():
            table[0, :pages] = torch.arange(1, pages + 1)
        self.generator.warm_prefill_shapes(kv_cache, tables, limit)
        self._prefill_warmed = marker

    def warmup_model_decode(self, kv_cache, enable_trace, **kwargs):
        # Every trace a step can pick exists before the first request: a missing one would be compiled and captured in
        # the middle of a request, retiring the others.
        states = [kv_cache, *self.sub_caches.values(), *self.verify_caches.values()]
        for state in states:
            self.generator._tables(state, state.page_tables)
        self.generator.configure_sampling(batch=self.rows, temperature=0)
        # The full trace first (the returned entry is the full batch's).
        start = time.perf_counter()
        self.entry = self.generator.prepare_entries(states)[0]
        self.cur_state = kv_cache
        used = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank
        logger.info(
            "Captured {} decode traces ({} rows) in {:.1f} s; DRAM allocated per bank {} MiB",
            len(states),
            [state.batch for state in states],
            time.perf_counter() - start,
            used >> 20,
        )

    def note_state_slots_moved(self, moves):
        self.prefill_slot_keys = {moves.get(slot, slot): key for slot, key in self.prefill_slot_keys.items()}

    def release_request(self, slot):
        key = self.prefill_slot_keys.pop(slot, None)
        if key is not None and self.cache is not None:
            self.cache.prefill_history.pop(key, None)

    def release_persistent_capture(self):
        if self.cache is not None:
            self.cache.prefill_history.clear()
        self.prefill_slot_keys.clear()
        self.generator.close()
        self._prefill_warmed = None
        self.entry = None
