# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Readiness generator with separate model and common-sampler decode traces."""

import os
import secrets
import time
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

import ttnn
from models.common.modules.tt_ccl import get_tt_ccl
from models.common.sampling.generator import (
    SamplingGenerator,
    SamplingParams,
    _hash_request_seed_to_device_seed,
    format_sampling_params,
)
from models.demos.gemma4_26b_a4b_qb2.tt.model import MODEL_ID, REVISION, Gemma4Model


class _Gemma4SamplingGenerator(SamplingGenerator):
    """Include the public-token copy in the canonical sampling trace."""

    _feedback_tokens = None
    _public_tokens = None

    def bind_public_tokens(self, tokens, public_tokens):
        self._feedback_tokens = tokens
        self._public_tokens = public_tokens

    def format_public_tokens(self):
        ttnn.slice(
            self._feedback_tokens,
            starts=(0, 0, 0, 0),
            ends=tuple(self._public_tokens.shape),
            steps=(1, 1, 1, 1),
            output_tensor=self._public_tokens,
        )

    def _run_sampling(self, logits, *, penalties_on, tt_out_tok, count_tokens=True):
        output = super()._run_sampling(
            logits, penalties_on=penalties_on, tt_out_tok=tt_out_tok, count_tokens=count_tokens
        )
        if tt_out_tok is not None and tt_out_tok is self._feedback_tokens:
            self.format_public_tokens()
        return output


class Gemma4Generator:
    def __init__(
        self,
        mesh_device,
        *,
        max_seq_len=None,
        layer_indices=None,
        host_sampling=False,
        trace_debug=False,
        precision_config=None,
    ):
        self.mesh = mesh_device
        self.trace_debug = trace_debug
        self.model = Gemma4Model(
            mesh_device, max_seq_len=max_seq_len, layer_indices=layer_indices, precision_config=precision_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        self.host_sampling = host_sampling
        self.sampled_mode = False
        args = SimpleNamespace(
            vocab_size=self.model.config.vocab_size,
            padded_vocab_size=self.model.config.vocab_size,
            cluster_shape=(1, 4),
            sampling_all_gather_axis=1,
            sampling_dp=1,
            num_devices=4,
            is_galaxy=False,
            max_batch_size=32,
            max_top_k=32,
            use_topk_logprobs=False,
            model_config={
                "SAMPLING_AG_CONFIG": {
                    "allow_force_argmax": False,
                    "num_links": 1,
                    "topology": ttnn.Topology.Linear,
                }
            },
        )
        self.sampler = _Gemma4SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=get_tt_ccl(mesh_device))
        self.sampler.reset_sampling_params(
            format_sampling_params(SamplingParams(temperature=0.0, top_k=1, top_p=1.0), 32)
        )
        self.trace_id = None
        self.prefill_trace_enabled = os.environ.get("GEMMA4_PREFILL_TRACE", "1") == "1"
        self.prefill_prepared = None
        self.prefill_trace_id = None
        self.prefill_sampling_trace_id = None
        print(f"PREFILL_TRACE_POLICY enabled={self.prefill_trace_enabled} max_length=1024 entries=1", flush=True)
        policies = {
            (
                layer.layer.moe.experts.prefill.short_prefill_batch_tokens,
                tuple(
                    (rows, configs[0].in0_block_w, configs[1].in0_block_w, configs[1].per_core_N)
                    for rows, configs in layer.layer.moe.experts.prefill.prefill_configs.items()
                ),
            )
            for layer in self.model.layers
        }
        print(
            f"PREFILL_EXPERT_POLICIES {sorted(policies)} fields=(rows,gateK,downK,downN) wider_rows=128..256",
            flush=True,
        )
        self._trace_returns_logits = False
        self.output_trace_id = None
        self.output_buffer = None
        self.cache = None
        self.table_host = None
        self.owned_cache = None
        self.owned_table = None
        self.counters = dict.fromkeys(
            (
                "model_replays",
                "sampling_replays",
                "token_refreshes",
                "position_refreshes",
                "cache_position_refreshes",
                "page_table_refreshes",
                "synchronizations",
                "token_readbacks",
                "full_logits_readbacks",
                "teacher_forcing_token_refreshes",
            ),
            0,
        )
        self.metrics = {}
        self.last_log_probs = None
        self._generation_signature = None

    def _copy(self, value, target, counter):
        host = ttnn.from_torch(
            value, dtype=target.dtype, layout=target.layout, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
        )
        ttnn.copy_host_to_device_tensor(host, target)
        self.counters[counter] = self.counters.get(counter, 0) + 1

    def _validate_sampling(self, params):
        formatted = format_sampling_params(params, 32)
        if any(formatted.enable_log_probs) or any(value > 0 for value in formatted.num_logprobs):
            raise ValueError("The common log-probability implementation does not support this TP4 mesh")
        if self.host_sampling and (
            any(k != 1 for k in formatted.top_k)
            or any(v != 0 for v in formatted.presence_penalty + formatted.frequency_penalty)
            or any(v != 1 for v in formatted.repetition_penalty)
        ):
            raise ValueError(
                "Host sampling compatibility supports greedy unpenalized tests; use device sampling for other parameters"
            )
        return formatted

    def configure_sampling(self, params, *, prompt_tokens=None, slots=None, seed_offsets=None, _reuse_trace=False):
        """Set request sampling state before binding/capturing a decode trace.

        Per-slot parameter lists use the common sampler's 32-lane padding.
        Device-owned seeds advance in the model trace, so the common sampler's
        host request-seed manager remains inactive. Optional seed offsets restore
        that device increment sequence when a scheduler rebuilds the batch.
        """
        formatted = self._validate_sampling(params)
        if not _reuse_trace:
            self._release_trace()
        temperatures = params.temperature if isinstance(params.temperature, list) else [params.temperature]
        self.sampled_mode = any(value != 0.0 for value in temperatures)
        if getattr(self, "prefill_prepared", None) is not None and (
            self.host_sampling
            or self.sampled_mode
            or any(value != 0 for value in formatted.presence_penalty + formatted.frequency_penalty)
            or any(value != 1 for value in formatted.repetition_penalty)
            or any(formatted.enable_log_probs)
        ):
            self._release_trace()
            self.prefill_prepared = None
        self._reset_sampling_seeds(formatted.seed, seed_offsets)
        self.sampler.reset_sampling_params(formatted)
        if prompt_tokens is not None:
            self.sampler.reset_prompt_tokens(prompt_tokens, slots=slots)
        self.sampler.reset_output_state()

    def _reset_sampling_seeds(self, request_seeds, seed_offsets=None):
        if seed_offsets is not None:
            seed_offsets = torch.as_tensor(seed_offsets)
            if (
                seed_offsets.ndim != 1
                or seed_offsets.numel() > len(request_seeds)
                or seed_offsets.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                or (seed_offsets < 0).any()
            ):
                raise ValueError("seed_offsets must be a nonnegative integer vector within the sampler batch")
        seeds = torch.tensor(
            [
                _hash_request_seed_to_device_seed(int(seed) if seed is not None else secrets.randbits(63), 0)
                for seed in request_seeds
            ],
            dtype=torch.int32,
        )
        if seed_offsets is not None:
            seeds[: seed_offsets.numel()] += seed_offsets.to(dtype=torch.int32)
        self._copy(seeds, self.sampler.tt_sampling.seeds_tt_tensor, "request_seed_refreshes")

    def restore_sampling_state(self, params, *, prompt_tokens=None, output_tokens=None, seed_offsets=None):
        """Restore scheduler row state with unchanged parameters and trace bindings."""
        if self.sampled_mode:
            self._reset_sampling_seeds(self._validate_sampling(params).seed, seed_offsets)
        if prompt_tokens is not None:
            self.sampler.reset_prompt_tokens(prompt_tokens)
        self.sampler.reset_output_state(output_tokens)

    def _read_tokens(self):
        self.counters["token_readbacks"] = self.counters.get("token_readbacks", 0) + 1
        return ttnn.to_torch(ttnn.get_device_tensors(self.tokens)[0]).flatten()[: self.batch].tolist()

    def _read_logits(self, logits):
        self.counters["full_logits_readbacks"] = self.counters.get("full_logits_readbacks", 0) + 1
        return ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1)).float()

    def _release_trace(self):
        for name in ("prefill_trace_id", "prefill_sampling_trace_id"):
            if (trace := getattr(self, name, None)) is not None:
                ttnn.release_trace(self.mesh, trace)
                setattr(self, name, None)
        if self.output_trace_id is not None:
            ttnn.release_trace(self.mesh, self.output_trace_id)
            self.output_trace_id = None
            self.output_buffer = None
        self.sampler.reset_trace()
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh, self.trace_id)
            self.trace_id = None
        self._trace_returns_logits = False

    def reset(self):
        # Only standalone-owned caches are cleared. External callers retain
        # ownership of their cache contents and allocation lifetime.
        if self.owned_cache is not None:
            for pair in self.owned_cache:
                for tensor in pair:
                    ttnn.mul(tensor, 0.0, output_tensor=tensor)
        self.sampler.reset_penalty_counts()
        self.table_host = None
        self.last_log_probs = None
        if hasattr(self, "tokens"):
            self._copy(torch.zeros(1, 1, 1, 32, dtype=torch.int32), self.tokens, "reset_refreshes")
            self._copy(torch.zeros(1, 32, dtype=torch.int32), self.positions, "reset_refreshes")
            self._copy(torch.zeros(self.batch, dtype=torch.int32), self.cache_positions, "reset_refreshes")
            self._copy(torch.zeros(1, 1, 1, self.batch, dtype=torch.int32), self.public_tokens, "reset_refreshes")
        self.counters = dict.fromkeys(
            (
                "model_replays",
                "sampling_replays",
                "token_refreshes",
                "position_refreshes",
                "cache_position_refreshes",
                "page_table_refreshes",
                "synchronizations",
                "token_readbacks",
                "full_logits_readbacks",
                "teacher_forcing_token_refreshes",
            ),
            0,
        )

    def teardown(self):
        self._release_trace()
        self.prefill_prepared = None

    def serving_prefill_eligible(self, params):
        """The bounded fast path currently covers canonical greedy sampling."""
        if not self.prefill_trace_enabled:
            return False
        formatted = self._validate_sampling(params)
        temperatures = params.temperature if isinstance(params.temperature, list) else [params.temperature]
        return (
            all(value == 0 for value in temperatures)
            and all(value == 0 for value in formatted.presence_penalty)
            and all(value == 0 for value in formatted.frequency_penalty)
            and all(value == 1 for value in formatted.repetition_penalty)
            and not any(formatted.enable_log_probs)
        )

    def _serving_prefill_key(self, tokens, page_table, kv_cache, prompt_lens):
        if not self.prefill_trace_enabled or len(prompt_lens) != 1 or tokens.shape[0] != 1:
            return None
        length = int(prompt_lens[0])
        if not 1 <= length <= min(1024, tokens.shape[1], self.model.max_seq_len):
            return None
        tables = self._tables(page_table)
        if any(not isinstance(table, torch.Tensor) or table.ndim != 2 or table.shape[0] != 1 for table in tables):
            return None
        return (length, id(kv_cache), tuple(id(t) for pair in kv_cache for t in pair), self._table_shapes(page_table))

    def can_reuse_serving_prefill(self, tokens, *, page_table, kv_cache, prompt_lens):
        key = self._serving_prefill_key(tokens, page_table, kv_cache, prompt_lens)
        return bool(
            key is not None
            and self.prefill_prepared is not None
            and self.prefill_prepared["key"] == key
            and self.prefill_trace_id is not None
            and self.prefill_sampling_trace_id is not None
        )

    def can_reuse_serving_decode(self, params, *, allow_eager_prefill=False):
        """Allow padded scheduler parameters to refresh a compatible graph.

        Prefill has compact parameter rows; decode pads them to max_num_seqs.
        Their repr differs even when the live request has unchanged sampling.
        The model graph also fixes whether device seeds advance, so that mode
        must match independently of the sampler's persistent parameter values.
        """
        if (
            (self.prefill_prepared is None and not allow_eager_prefill)
            or self.trace_id is None
            or self._trace_returns_logits
        ):
            return False
        formatted = self._validate_sampling(params)
        temperatures = params.temperature if isinstance(params.temperature, list) else [params.temperature]
        return (
            self._trace_sampled_mode == any(value != 0 for value in temperatures)
            and not self.sampler._penalties_active
            and not self.sampler._log_probs_active
            and all(value == 0 for value in formatted.presence_penalty)
            and all(value == 0 for value in formatted.frequency_penalty)
            and all(value == 1 for value in formatted.repetition_penalty)
            and not any(formatted.enable_log_probs)
        )

    def _prefill_trace_step(self):
        state = self.prefill_prepared
        logits = self.model.prefill_forward(
            state["tokens"], page_table=state["tables"], kv_cache=state["cache"], user_id=0
        )
        ttnn.copy(self._sampler_logits(logits), state["logits"])

    def _prefill_sampling_step(self):
        state = self.prefill_prepared
        self.sampler.sample(state["logits"], tt_out_tok=state["output"], enable_trace=False)

    def serving_prefill_tokens(self, tokens, *, page_table, kv_cache, prompt_lens):
        """Own one exact-length prefill graph, with an eager general fallback."""
        key = self._serving_prefill_key(tokens, page_table, kv_cache, prompt_lens)
        # Also enforce eligibility for direct generator callers: the adapter
        # check alone must not allow a sampled/penalized graph to be captured.
        if self.host_sampling or self.sampled_mode or self.sampler._penalties_active or self.sampler._log_probs_active:
            key = None
        if key is None:
            logits = self.prefill_forward(tokens, page_table=page_table, kv_cache=kv_cache, prompt_lens=prompt_lens)
            return self.sample_prefill(logits)
        if self.prefill_prepared is None or self.prefill_prepared["key"] != key:
            self._release_trace()
            self.prefill_prepared = None
            self.prefill_prepared = dict(
                key=key,
                cache=kv_cache,
                tokens=self.model.upload(
                    tokens[:, : key[0]].reshape(1, 1, 1, key[0]).int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
                ),
                tables=self._upload_page_tables(page_table),
                table_host=self._clone_tables(page_table),
                logits=self.model.upload(torch.zeros(1, 1, 32, self.model.config.vocab_size // 4)),
                output=self.model.upload(
                    torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
                ),
            )
        state = self.prefill_prepared
        self._copy(tokens[:, : key[0]].reshape(1, 1, 1, key[0]).int(), state["tokens"], "prefill_token_refreshes")
        changed = False
        for host, previous, device in zip(
            self._tables(page_table), self._tables(state["table_host"]), self._tables(state["tables"])
        ):
            if not torch.equal(host, previous):
                self._copy(host, device, "prefill_page_refreshes")
                changed = True
        if changed:
            state["table_host"] = self._clone_tables(page_table)
        # OSL1 workloads never enter decode warmup. On their second matching
        # request, the existing prefill programs and persistent outputs are
        # already warm, so they can own a prefill-only trace. A later decode
        # bind releases it before allocating decode state and recapturing.
        if self.prefill_trace_id is None and state.get("warmed", False):
            self._capture_prefill()
        if self.prefill_trace_id is None:
            self._prefill_trace_step()
            self._prefill_sampling_step()
            state["warmed"] = True
            self.counters["prefill_eager_calls"] = self.counters.get("prefill_eager_calls", 0) + 1
        else:
            ttnn.execute_trace(self.mesh, self.prefill_trace_id, cq_id=0, blocking=False)
            ttnn.execute_trace(self.mesh, self.prefill_sampling_trace_id, cq_id=0, blocking=False)
            self.counters["prefill_replays"] = self.counters.get("prefill_replays", 0) + 1
        self.last_log_probs = None
        return state["output"]

    def _capture_prefill(self):
        if self.prefill_prepared is None:
            return
        # Prepared inputs/output predate decode warmup. Capture records the
        # already-warmed prefill graph without mutating the live decode cache.
        try:
            trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self.prefill_trace_id = trace
            try:
                self._prefill_trace_step()
            finally:
                ttnn.end_trace_capture(self.mesh, trace, cq_id=0)

            trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self.prefill_sampling_trace_id = trace
            try:
                self._prefill_sampling_step()
            finally:
                ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
        except BaseException:
            self._release_trace()
            raise
        self.counters["prefill_captures"] = self.counters.get("prefill_captures", 0) + 1

    def _standalone_cache(self, context, *, reuse_trace=False):
        # New prompt signatures can compile prefill programs. Release decode
        # traces before preparing that request; reset itself retains traces.
        if not reuse_trace:
            self._release_trace()
        if self.owned_cache is None or self.owned_table.shape[1] * 32 < context:
            self.owned_cache, self.owned_table = self.model.allocate_cache(slots=1, context=context)
            for pair in self.owned_cache:
                for tensor in pair:
                    ttnn.mul(tensor, 0.0, output_tensor=tensor)
        self.cache = self.owned_cache
        return self.owned_table

    def prefill_forward(self, tokens, *, page_table, kv_cache, prompt_lens, return_all_logits=False, slots=None):
        if getattr(self, "prefill_prepared", None) is not None:
            self._release_trace()
            self.prefill_prepared = None
        slots = list(range(len(prompt_lens))) if slots is None else list(slots)
        if len(slots) != len(prompt_lens) or tokens.shape[0] != len(slots):
            raise ValueError("Prompt, slot and batch dimensions differ")
        table = self._upload_page_tables(page_table)
        outputs = []
        for row, (slot, length) in enumerate(zip(slots, prompt_lens)):
            if not 1 <= length <= min(tokens.shape[1], self.model.max_seq_len):
                raise ValueError("Invalid logical prompt length")
            ids = self.model.upload(
                tokens[row : row + 1, :length].reshape(1, 1, 1, length).int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
            )
            logits = self.model.prefill_forward(
                ids, page_table=table, kv_cache=kv_cache, user_id=slot, return_all_logits=return_all_logits
            )
            outputs.append(logits)
        if return_all_logits:
            # Logit compatibility boundary is explicit; token-out never uses it.
            max_length = max(prompt_lens)
            result = torch.zeros(len(outputs), max_length, self.model.config.vocab_size)
            for row, (logits, length) in enumerate(zip(outputs, prompt_lens)):
                result[row, :length] = self._read_logits(logits).reshape(length, -1)
            return result
        logits = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)
        return ttnn.reshape(logits, (len(outputs), 1, self.model.config.vocab_size // 4))

    def sample_prefill(self, logits, *, output_tokens=None):
        """Use the canonical common sampler for standalone and serving prefill."""
        logits = ttnn.reshape(logits, (1, 1, logits.shape[0], self.model.config.vocab_size // 4))
        sampled = self.sampler.sample(self._sampler_logits(logits), tt_out_tok=output_tokens, enable_trace=False)
        self.last_log_probs = sampled[1] if isinstance(sampled, tuple) else None
        return sampled[0] if isinstance(sampled, tuple) else sampled

    def prefill_logits(self, prompt_token_ids):
        self.reset()
        table = self._standalone_cache(len(prompt_token_ids))
        return self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=table,
            kv_cache=self.cache,
            prompt_lens=[len(prompt_token_ids)],
            return_all_logits=True,
        )

    @staticmethod
    def _sampler_logits(logits):
        # Common TP sampling owns 32 logical rows of parameters and offsets.
        # Implicit tile padding alone cannot broadcast B=2..31 against them.
        if logits.shape[-2] < 32:
            return ttnn.pad(logits, [(0, 0), (0, 0), (0, 32 - logits.shape[-2]), (0, 0)], 0.0)
        return logits

    def _forward(self):
        if self.trace_debug:
            self.consumed_tokens = ttnn.clone(self.tokens)
            self.consumed_positions = ttnn.clone(self.cache_positions)
        logits = self.model.decode_forward(
            self.tokens,
            current_pos=self.positions,
            cache_pos=self.cache_positions,
            page_table=self.table,
            kv_cache=self.cache,
            batch=self.batch,
            active_slots=self.active_slots,
        )
        ttnn.plus_one(self.positions, skip_negative_entries=True)
        ttnn.plus_one(self.cache_positions, skip_negative_entries=True)
        if self.sampled_mode and not self._trace_returns_logits:
            ttnn.plus_one(self.sampler.tt_sampling.seeds_tt_tensor)
        return self._sampler_logits(logits)

    @staticmethod
    def _tables(page_table):
        return tuple(page_table) if isinstance(page_table, (tuple, list)) else (page_table,)

    def _upload_page_tables(self, page_table):
        tables = [
            table if isinstance(table, ttnn.Tensor) else self.model.upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            for table in self._tables(page_table)
        ]
        return tuple(tables) if isinstance(page_table, (tuple, list)) else tables[0]

    @classmethod
    def _table_shapes(cls, page_table):
        return tuple(tuple(table.shape) for table in cls._tables(page_table))

    @classmethod
    def _clone_tables(cls, page_table):
        copies = [table.clone() for table in cls._tables(page_table)]
        return tuple(copies) if isinstance(page_table, (tuple, list)) else copies[0]

    def _bind(self, *, positions, page_table, kv_cache, batch):
        tables = self._tables(page_table)
        if len(tables) not in (1, len(self.model.layers)):
            raise ValueError("Page tables must be uniform or contain one table per layer")
        if not 1 <= batch <= 32 or any(t.ndim != 2 or t.shape[0] != batch for t in tables):
            raise ValueError("Decode requires 1..32 slots and one page-table row per slot")
        if any(value < -1 or any(value // 32 >= t.shape[1] for t in tables) for value in positions.tolist()):
            raise ValueError("Decode position outside the page table; only -1 denotes an inactive slot")
        self._release_trace()
        self.batch = batch
        if self.prefill_prepared is not None and (batch != 1 or self.prefill_prepared["cache"] is not kv_cache):
            self.prefill_prepared = None
        self.active_slots = tuple(i for i, value in enumerate(positions.tolist()) if value >= 0)
        if not self.active_slots or any(value >= self.model.max_seq_len for value in positions.tolist()):
            raise ValueError("Decode needs a valid active slot; -1 marks inactive positions")
        self.cache = kv_cache
        self.table_host = self._clone_tables(page_table)
        self.table = self._upload_page_tables(page_table)
        pos = torch.full((1, 32), -1, dtype=torch.int32)
        pos[0, :batch] = positions
        self.positions = self.model.upload(pos, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.cache_positions = self.model.upload(positions.int(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.tokens = self.model.upload(torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.public_tokens = self.model.upload(
            torch.zeros(1, 1, 1, batch, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
        )
        self.public_tokens_view = ttnn.reshape(self.public_tokens, (batch,))
        self.sampler.bind_public_tokens(self.tokens, self.public_tokens)

    def _format_tokens(self):
        self.sampler.format_public_tokens()

    def _capture(self, *, return_logits=False):
        self._trace_returns_logits = return_logits
        self._trace_sampled_mode = self.sampled_mode
        if not return_logits:
            self._format_tokens()
        initial_tokens = ttnn.clone(self.tokens)
        initial_positions = ttnn.clone(self.positions)
        initial_cache_positions = ttnn.clone(self.cache_positions)
        initial_seeds = None if return_logits else ttnn.clone(self.sampler.tt_sampling.seeds_tt_tensor)
        warmed = self._forward()
        if not return_logits:
            self.sampler.precompile(warmed, tt_out_tok=self.tokens)
        ttnn.copy(initial_tokens, self.tokens)
        ttnn.copy(initial_positions, self.positions)
        ttnn.copy(initial_cache_positions, self.cache_positions)
        if initial_seeds is not None:
            ttnn.copy(initial_seeds, self.sampler.tt_sampling.seeds_tt_tensor)
        ttnn.synchronize_device(self.mesh)
        self.trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.trace_logits = self._forward()
        ttnn.end_trace_capture(self.mesh, self.trace_id, cq_id=0)
        if not return_logits:
            self.sampler.capture_trace(self.trace_logits, tt_out_tok=self.tokens, skip_precompile=True)
            self._capture_prefill()
        ttnn.copy(initial_tokens, self.tokens)
        ttnn.copy(initial_positions, self.positions)
        ttnn.copy(initial_cache_positions, self.cache_positions)
        if initial_seeds is not None:
            ttnn.copy(initial_seeds, self.sampler.tt_sampling.seeds_tt_tensor)
        ttnn.synchronize_device(self.mesh)
        self.counters["decode_captures"] = self.counters.get("decode_captures", 0) + 1
        print("LOGITS_TRACE_READY" if return_logits else "SPLIT_TRACE_READY", flush=True)

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table,
        kv_cache,
        enable_trace=True,
        device_feedback=False,
        return_logits=False,
    ):
        """Replay the canonical model; explicit logits output leaves sampling to the caller."""
        if not enable_trace:
            raise ValueError("Decode requires tracing")
        if return_logits and (device_feedback or tokens is None):
            raise ValueError("Logits output requires explicit scheduler tokens and positions")
        batch = len(start_pos)
        binding_changed = (
            self.trace_id is None
            or self._trace_returns_logits != return_logits
            or kv_cache is not self.cache
            or batch != self.batch
            or self._table_shapes(page_table) != self._table_shapes(self.table)
            or tuple(i for i, v in enumerate(start_pos.tolist()) if v >= 0) != self.active_slots
        )
        if binding_changed:
            if device_feedback and tokens is None:
                raise ValueError("New cache/slot binding requires explicit input tokens")
            self._bind(positions=start_pos, page_table=page_table, kv_cache=kv_cache, batch=batch)
            ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
            ids.flatten()[:batch] = tokens.flatten().int()
            self._copy(ids, self.tokens, "token_refreshes")
            self._capture(return_logits=return_logits)
        else:
            old_tables = (
                self._tables(self.table_host)
                if self.table_host is not None
                else (None,) * len(self._tables(page_table))
            )
            refreshed = set()
            for incoming, previous, device in zip(self._tables(page_table), old_tables, self._tables(self.table)):
                if previous is None or not torch.equal(incoming, previous):
                    if id(device) not in refreshed:
                        self._copy(incoming, device, "page_table_refreshes")
                        refreshed.add(id(device))
            if refreshed:
                self.table_host = self._clone_tables(page_table)
            if not device_feedback:
                ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
                ids.flatten()[:batch] = tokens.flatten().int()
                self._copy(ids, self.tokens, "token_refreshes")
                pos = torch.full((1, 32), -1, dtype=torch.int32)
                pos[0, :batch] = start_pos
                self._copy(pos, self.positions, "position_refreshes")
                self._copy(start_pos.int(), self.cache_positions, "cache_position_refreshes")
        output = self._replay()
        if return_logits:
            return output
        if self.host_sampling:
            self._format_tokens()
        return self.public_tokens_view

    def _replay(self):
        ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
        self.counters["model_replays"] = self.counters.get("model_replays", 0) + 1
        if self._trace_returns_logits:
            self.last_log_probs = None
            return self.trace_logits
        if self.host_sampling:
            predicted = self._read_logits(self.trace_logits)[..., : self.batch, :].reshape(self.batch, -1).argmax(-1)
            ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
            ids.flatten()[: self.batch] = predicted
            self._copy(ids, self.tokens, "token_refreshes")
        else:
            sampled = self.sampler.sample(self.trace_logits, tt_out_tok=self.tokens, enable_trace=True)
            self.last_log_probs = sampled[1] if isinstance(sampled, tuple) else None
            self.counters["sampling_replays"] = self.counters.get("sampling_replays", 0) + 1
        return self.tokens

    def _record_token(self):
        # Both the destination row and token values are device-owned. Capture
        # owns the temporary indexed_fill output; replay allocates no tensors.
        updated = ttnn.indexed_fill(self.output_index, self.output_buffer, self.tokens, dim=0)
        ttnn.copy(updated, self.output_buffer)
        ttnn.plus_one(self.output_index)

    def _prepare_output_buffer(self, steps):
        if self.output_trace_id is not None and self.output_buffer.shape[0] < steps:
            ttnn.release_trace(self.mesh, self.output_trace_id)
            self.output_trace_id = None
            self.output_buffer = None
        if self.output_trace_id is None:
            self.output_buffer = self.model.upload(
                torch.zeros(steps, 1, 1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
            )
            self.output_index = self.model.upload(torch.zeros(1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            self._record_token()
        self._copy(torch.zeros(1, dtype=torch.int32), self.output_index, "request_output_refreshes")

    def _capture_output_buffer(self):
        if self.output_trace_id is None:
            ttnn.synchronize_device(self.mesh)
            self.output_trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._record_token()
            ttnn.end_trace_capture(self.mesh, self.output_trace_id, cq_id=0)
            self._copy(torch.zeros(1, dtype=torch.int32), self.output_index, "request_output_refreshes")

    def _generate_buffered(self, steps):
        for _ in range(steps):
            self._replay()
            ttnn.execute_trace(self.mesh, self.output_trace_id, cq_id=0, blocking=False)
            self.counters["output_replays"] = self.counters.get("output_replays", 0) + 1
        # Public generate returns the sequence, so exactly one final transfer is
        # required. There is no readback or host input refresh between replays.
        self.counters["token_readbacks"] += 1
        output = ttnn.to_torch(ttnn.get_device_tensors(self.output_buffer)[0])
        return output[:steps, 0, 0, 0].tolist()

    def generate(
        self,
        prompt_token_ids,
        max_new_tokens,
        *,
        next_input=None,
        enable_trace=True,
        sampling_params=None,
        stop_on_eos=True,
        buffer_tokens=True,
        **kwargs,
    ):
        if not enable_trace:
            raise ValueError("Decode requires tracing")
        if (
            max_new_tokens < 0
            or not prompt_token_ids
            or len(prompt_token_ids) + max_new_tokens > self.model.max_seq_len
        ):
            raise ValueError("Invalid prompt or generation length")
        if max_new_tokens == 0:
            return []
        buffered = buffer_tokens and not stop_on_eos and next_input is None and not self.host_sampling
        started = request_started = time.perf_counter()
        params = sampling_params or SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        self._validate_sampling(params)
        signature = (len(prompt_token_ids), repr(params), self.host_sampling)
        reuse_trace = (
            self.trace_id is not None
            and not self._trace_returns_logits
            and self._generation_signature == signature
            and self.cache is self.owned_cache
            and self.owned_table.shape[1] * 32 >= len(prompt_token_ids) + max_new_tokens
            and params == SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
            and (
                not buffered or (self.output_trace_id is not None and self.output_buffer.shape[0] >= max_new_tokens - 1)
            )
        )
        self.reset()
        table = self._standalone_cache(len(prompt_token_ids) + max_new_tokens, reuse_trace=reuse_trace)
        self.configure_sampling(params, prompt_tokens=torch.tensor([prompt_token_ids]), _reuse_trace=reuse_trace)
        position = torch.tensor([len(prompt_token_ids)], dtype=torch.int32)
        if reuse_trace:
            self._copy(table, self.table, "request_table_refreshes")
            self.table_host = table.clone()
            rope_positions = torch.full((1, 32), -1, dtype=torch.int32)
            rope_positions[0, 0] = len(prompt_token_ids)
            self._copy(rope_positions, self.positions, "request_position_refreshes")
            self._copy(position, self.cache_positions, "request_position_refreshes")
        else:
            self._bind(positions=position, page_table=table, kv_cache=self.cache, batch=1)
        logits = self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=self.table,
            kv_cache=self.cache,
            prompt_lens=[len(prompt_token_ids)],
        )
        logits = ttnn.reshape(logits, (1, 1, 1, self.model.config.vocab_size // 4))
        if self.host_sampling:
            ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
            ids.flatten()[0] = self._read_logits(logits).flatten().argmax()
            self._copy(ids, self.tokens, "token_refreshes")
        else:
            self.sample_prefill(
                ttnn.reshape(logits, (1, 1, self.model.config.vocab_size // 4)), output_tokens=self.tokens
            )
        del logits
        first = int(self._read_tokens()[0])
        ttft = time.perf_counter() - started
        result = [first]
        trace_setup_ms = 0.0
        eos = set([1, 106])
        if max_new_tokens > 1 and not (stop_on_eos and next_input is None and first in eos):
            capture_started = time.perf_counter()
            if buffered:
                self._prepare_output_buffer(max_new_tokens - 1)
            if not reuse_trace:
                self._capture()
            if buffered:
                self._capture_output_buffer()
            trace_setup_ms = (time.perf_counter() - capture_started) * 1000
            self._generation_signature = signature
            # The first token is ready for the caller only after the decode
            # state is prepared; count request trace setup in end-to-end TTFT.
            ttft = time.perf_counter() - request_started
            self.counters = dict.fromkeys(
                (
                    "model_replays",
                    "sampling_replays",
                    "token_refreshes",
                    "position_refreshes",
                    "cache_position_refreshes",
                    "page_table_refreshes",
                    "synchronizations",
                    "token_readbacks",
                    "full_logits_readbacks",
                    "teacher_forcing_token_refreshes",
                ),
                0,
            )
            started = time.perf_counter()
            if buffered:
                result.extend(self._generate_buffered(max_new_tokens - 1))
            else:
                for step in range(1, max_new_tokens):
                    if next_input is not None:
                        forced = next_input(step - 1, result[-1])
                        ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
                        ids.flatten()[0] = forced
                        self._copy(ids, self.tokens, "teacher_forcing_token_refreshes")
                    self._replay()
                    result.append(int(self._read_tokens()[0]))
                    if stop_on_eos and next_input is None and result[-1] in eos:
                        break
            seconds = time.perf_counter() - started
            if next_input is not None:
                next_input(max_new_tokens - 1, result[-1])
        else:
            seconds = 0
            if next_input is not None:
                next_input(0, first)
        self.metrics = {
            "ttft_ms": ttft * 1000,
            "trace_setup_ms": trace_setup_ms,
            "reused_request_trace": reuse_trace,
            "generation_wall_ms": (time.perf_counter() - request_started) * 1000,
            "decode_tps": (len(result) - 1) / seconds if seconds else None,
            "input_tokens": len(prompt_token_ids),
            "output_tokens": len(result),
            "batch": 1,
            "source": "teacher_forcing" if next_input else "autoregressive",
            "reduced_probe": self.model.reduced_probe,
            "host_sampling": self.host_sampling,
            "buffered_token_output": buffered,
            "counters": dict(self.counters),
        }
        return result


def build_generator(model_dir, mesh_device, **kwargs):
    return Gemma4Generator(mesh_device, **kwargs)
