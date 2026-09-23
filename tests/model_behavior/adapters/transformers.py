# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Real-model adapters using tt-metal's factories and Generator, without vLLM."""

import os
from contextlib import contextmanager
from unittest.mock import patch

import torch

from tests.model_behavior.adapters.paged import PagedAdapter
from tests.model_behavior.adapters.profiles import HARDWARE, PROFILES, select_sku
from tests.model_behavior.driver import Sample


def vocabulary_groups(shape, axis, data_parallel):
    """Device indices for each independent batch's complete vocabulary."""
    rows, columns = shape
    if min(shape) == 1:
        return [list(range(rows * columns))]
    if axis == 0:
        groups = [[row * columns + col for row in range(rows)] for col in range(columns)]
    else:
        groups = [[row * columns + col for col in range(columns)] for row in range(rows)]
    if data_parallel not in (1, len(groups)):
        raise ValueError(f"Unexpected sampling groups: {shape=}, {axis=}, {data_parallel=}")
    return groups[:data_parallel]


def sampled_logprob(logprobs, index, token):
    """Handle both sampled-token and production top-k logprob outputs."""
    if isinstance(logprobs, tuple):
        values, ids = logprobs
        ids = ids.reshape(-1, ids.shape[-1])
        values = values.reshape_as(ids)
        matches = (ids[index] == token).nonzero().reshape(-1)
        if matches.numel() != 1:
            raise ValueError(f"Sampled token {token} is absent or duplicated in top-k logprobs at row {index}")
        return float(values[index, int(matches[0])])
    if logprobs is None:
        raise ValueError("Missing device logprobs")
    return float(logprobs.reshape(-1)[index])


def allocate_pages(token_capacities, block_size, physical_blocks, domains=1):
    """Reserve padding page zero and prevent aliases within each cache domain."""
    capacity = len(token_capacities)
    if capacity % domains or any(length % block_size for length in token_capacities):
        raise ValueError("Cache domains and block size must divide the requested layout")
    page_table = torch.zeros((capacity, max(token_capacities) // block_size), dtype=torch.int32)
    group_size = capacity // domains
    for domain in range(domains):
        next_page = 1
        for slot in range(domain * group_size, (domain + 1) * group_size):
            count = token_capacities[slot] // block_size
            page_table[slot, :count] = torch.arange(next_page, next_page + count, dtype=torch.int32)
            next_page += count
        if next_page > physical_blocks:
            raise ValueError("The request page allocations exceed the model's cache")
    return page_table


class TransformersAdapter(PagedAdapter):
    def __init__(self, mesh, execution_mode, backend, sku, *, skip_model_load=False):
        import ttnn
        from models.common.sampling import SamplingParams
        from models.tt_transformers.tt.common import PagedAttentionConfig
        from models.tt_transformers.tt.generator import Generator

        self.backend, self.sku = backend, sku
        self.profile = PROFILES[backend]
        # GPT-OSS accepts chunk arguments at its model boundary but currently
        # does not forward them to attention. Do not count that as prefix reuse.
        self.supports_chunked_prefill = self.profile.family not in ("gpt_oss", "qwen")
        self.capacity = self.profile.capacity(sku)
        self.execution_mode = execution_mode
        self.enable_trace = execution_mode == "traced"
        self.sampling_params_type = SamplingParams
        self.prefill_events = []
        self.long_context_slots = tuple(sorted({0, (self.capacity - 1) // 2, self.capacity // 2, self.capacity - 1}))
        row_sharded = self.profile.family == "gpt_oss" and self.capacity > 1
        self.page_domains = int(mesh.shape[0]) if row_sharded else 1
        self.physical_blocks = 2048 if self.profile.family == "qwen" else 1024
        short_capacity = 2048 if self.profile.family == "qwen" else 1024
        self.slot_token_capacity = [
            self.max_seq_len if s in self.long_context_slots else short_capacity for s in range(self.capacity)
        ]
        self.page_table = allocate_pages(
            self.slot_token_capacity, self.block_size, self.physical_blocks, self.page_domains
        )
        paging = PagedAttentionConfig(block_size=self.block_size, max_num_blocks=self.physical_blocks)

        if self.profile.family == "gemma":
            from models.demos.gemma4.tt.generator import Gemma4Generator

            self.generator, self.kv_cache, self.tokenizer = Gemma4Generator.from_pretrained(
                mesh,
                os.environ["HF_MODEL"],
                max_batch_size=self.capacity,
                max_seq_len=self.max_seq_len,
                paged_attention_config=paging,
            )
        elif self.profile.family == "qwen":
            from transformers import AutoTokenizer

            from models.demos.blackhole.qwen36.tt.common import create_tt_model

            args, model, _ = create_tt_model(mesh, max_batch_size=self.capacity, max_seq_len=self.max_seq_len)
            cache = model.allocate_kv_caches(
                [self.physical_blocks, args.n_local_kv_heads, self.block_size, args.head_dim],
                ttnn.bfloat16,
                batch_size=self.capacity,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(args.CKPT_DIR, trust_remote_code=True)
            self.generator = Generator([model], [args], mesh, tokenizer=self.tokenizer)
            self.kv_cache = [cache]
        else:
            if self.profile.family == "llama":
                from models.tt_transformers.tt.common import create_tt_model
                from models.tt_transformers.tt.model_config import DecodersPrecision

                options = dict(
                    instruct=True,
                    optimizations=lambda args: DecodersPrecision.performance(args.n_layers, args.model_name),
                )
            else:
                from models.demos.gpt_oss.tt.common import create_tt_model

                options = dict(
                    users_row_sharded=row_sharded,
                    use_throughput_experts=row_sharded,
                    state_dict={} if skip_model_load else None,
                )
            args, model, cache, _ = create_tt_model(
                mesh,
                max_batch_size=self.capacity,
                max_seq_len=self.max_seq_len,
                paged_attention_config=paging,
                **options,
            )
            self.tokenizer = args.tokenizer
            self.generator = Generator([model], [args], mesh, tokenizer=self.tokenizer)
            self.kv_cache = [cache]
        self.vocab_size = self.generator.model_args[0].vocab_size
        if self.sampling_model.sampling is None:
            raise RuntimeError(f"{backend}/{sku} does not provide a device sampler")
        self.duplicate_seed_policy = (
            "salted" if self.sampling_model.sampling.seed_manager.salt_duplicate_seeds else "identical"
        )
        calculator = self.sampling_model.sampling.tt_sampling.log_probs_calculator
        self.logprob_skip_reason = None
        if not calculator._is_supported():
            self.logprob_phases = ()
            self.logprob_skip_reason = (
                f"The production logprob calculator does not support {sku} ({tuple(mesh.shape)}); "
                "it currently requires 8 or 32 devices and at least two vocabulary shards"
            )

    @property
    def sampling_model(self):
        return self.generator.model[0]

    def warmup(self):
        # Prefill is deliberately submitted one request at a time. This exercises
        # normal slot admission and gives the logprob oracle the exact logits
        # consumed by each sampler call, also when the model prefill is traced.
        self.generator.model_args[0].disable_batched_prefill = True
        self.generator.model_capabilities = dict(
            self.generator.model_capabilities, supports_chunked_prefill=self.supports_chunked_prefill
        )
        # Match serving's two-phase warmup: compile both paths and prepare
        # persistent decode inputs before recording any prefill traces.
        self.generator.warmup_model_prefill(self.kv_cache, False, can_sample_on_device=True)
        self.generator.warmup_model_decode(
            self.kv_cache,
            False,
            self.capacity,
            self.page_table.shape[1],
            can_sample_on_device=True,
        )
        if self.enable_trace:
            self.generator.already_warmed_up_prefill = False
            self.generator.warmup_model_prefill(self.kv_cache, True, can_sample_on_device=True)
            self.generator.warmup_model_decode(
                self.kv_cache,
                True,
                self.capacity,
                self.page_table.shape[1],
                can_sample_on_device=True,
            )

    def encode(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        options = dict(add_generation_prompt=True, enable_thinking=False)
        if self.profile.family == "gpt_oss":
            # The default GPT-OSS prompt ends at "assistant", so its first
            # sampled token is the channel delimiter even for diverse answers.
            # Continue a final response to exercise content sampling during
            # prefill, using the tokenizer's own channel/header formatting.
            messages.append({"role": "assistant", "content": ""})
            options = dict(add_generation_prompt=False, continue_final_message=True)
        tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            **options,
        )
        if isinstance(tokens, dict):
            tokens = tokens["input_ids"]
        return tuple(int(t) for t in tokens)

    def decode_tokens(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    @contextmanager
    def _observe_sampling(self, states):
        captured = []
        if not any(s.request.sampling.enable_log_probs for s in states):
            yield captured
            return
        import ttnn

        sampling = self.sampling_model.sampling
        sampler = sampling.tt_sampling
        groups = vocabulary_groups(
            tuple(self.generator.mesh_device.shape), sampler.sampling_all_gather_axis, sampler._sampling_dp
        )
        original = sampling.sample

        def observe(logits, **kwargs):
            if getattr(self.generator, "warming_up_prefill", False):
                return original(logits, **kwargs)
            shards = ttnn.get_device_tensors(logits)
            hosts = [torch.cat([ttnn.to_torch(shards[i]).float() for i in group], dim=-1) for group in groups]
            if any(h.shape[-1] != sampler.padded_vocab_size for h in hosts):
                raise ValueError("Oracle did not gather the complete padded vocabulary")
            reference = torch.cat([torch.log_softmax(h.reshape(-1, h.shape[-1]), dim=-1) for h in hosts])
            result = original(logits, **kwargs)
            ids = result[0]
            while isinstance(ids, (list, tuple)):
                ids = ids[0]
            id_shards = ttnn.get_device_tensors(ids)
            host_ids = torch.cat([ttnn.to_torch(id_shards[g[0]]).reshape(-1).long() for g in groups])
            if host_ids.numel() != reference.shape[0]:
                raise ValueError("Oracle token rows do not match model logits")
            captured.append((host_ids, reference.gather(1, host_ids[:, None]).reshape(-1)))
            return result

        with patch.object(sampling, "sample", observe):
            yield captured

    def _prefill_call(self, admitted, lengths, offsets, params):
        assert len(admitted) == 1
        self._prefill_reference_row = (lengths[0] - offsets[0] - 1) % 32
        return self.generator.prefill_forward_text(
            torch.tensor([admitted[0].prompt_tokens[: lengths[0]]], dtype=torch.long),
            page_table=self.page_table[[admitted[0].slot]],
            kv_cache=self.kv_cache,
            prompt_lens=lengths,
            empty_slots=[admitted[0].slot],
            sampling_params=params,
            enable_trace=self.enable_trace,
            start_pos=offsets,
        )

    @contextmanager
    def _observe_prefill_offsets(self):
        # Shared Generator uses user_id=0 with a sliced one-request page table.
        # Retain both that local row and the physical admission slot in reports.
        with super()._observe_prefill_offsets() as calls:
            yield calls
            for call in calls:
                call["local_slot"] = call["slot"]
                call["slot"] = self._observed_slot

    def _checked_prefill_call(self, admitted, lengths, offsets, params):
        self._observed_slot = admitted[0].slot
        return super()._checked_prefill_call(admitted, lengths, offsets, params)

    def prefill(self, admitted):
        if not self.supports_chunked_prefill and any(state.request.prefill_chunk_ends for state in admitted):
            raise ValueError(f"{self.backend} does not support external prefix resumption")
        output = {}
        for state in admitted:
            output.update(super().prefill([state]))
            if self.profile.family == "gemma":
                # The driver has completed every prefix chunk and sampled the
                # first token. Decode uses paged KV, so this request's separate
                # cross-chunk sliding tails are no longer needed. Keep other
                # requests' tails and the boot-owned trace buffers intact.
                req_key = int(self.page_table[state.slot, 0]) + 1
                for model in self.generator.model:
                    for layer in model.layers:
                        layer.self_attn._release_sliding_prefill_tail(req_key=req_key)
        return output

    def _map_samples(self, states, tokens, logprobs, references, *, prefill):
        tokens = tokens.reshape(-1)
        expected = len(states) if prefill else self.capacity
        if tokens.numel() != expected:
            raise ValueError(f"Returned {tokens.numel()} tokens; expected {expected}")
        if references:
            if isinstance(logprobs, tuple):
                values, ids = logprobs
                rows = ids.reshape(-1, ids.shape[-1]).shape[0]
                if values.shape != ids.shape:
                    raise ValueError("Top-k logprob values and IDs have different shapes")
            else:
                rows = 0 if logprobs is None else logprobs.numel()
            if rows != expected:
                raise ValueError(f"Logprob rows do not align with {expected} returned tokens")
        result = {}
        for row, state in enumerate(states):
            index = row if prefill else state.slot
            token = int(tokens[index])
            result[state.slot] = token
            if state.request.sampling.enable_log_probs:
                if len(references) != 1:
                    raise ValueError(f"Expected one real sampler observation, received {len(references)}")
                ids, reference = references[0]
                ref_index = self._prefill_reference_row if prefill else state.slot
                if int(ids[ref_index]) != token:
                    raise ValueError(f"Returned token differs from the observed sampled token at slot {state.slot}")
                result[state.slot] = Sample(token, sampled_logprob(logprobs, index, token), float(reference[ref_index]))
        return result

    def describe(self):
        return dict(
            backend=self.backend,
            hf_model=os.environ["HF_MODEL"],
            sku=self.sku,
            mesh_shape=tuple(self.generator.mesh_device.shape),
            execution=self.execution_mode,
            prefill_execution=self.execution_mode,
            prefill_batching="sequential",
            stochastic_prefill=self.stochastic_prefill,
            duplicate_seed_policy=self.duplicate_seed_policy,
            logprob_phases=self.logprob_phases,
            logprob_skip_reason=self.logprob_skip_reason,
            supports_chunked_prefill=self.supports_chunked_prefill,
            capacity=self.capacity,
            page_domains=self.page_domains,
            page_table=self.page_table.tolist(),
            slot_token_capacity=self.slot_token_capacity,
            prefill_chunks=self.prefill_events,
        )

    def close(self):
        try:
            self.sampling_model.sampling.reset_trace()
        finally:
            self.generator = None
            self.kv_cache = None


class QwenAdapter(TransformersAdapter):
    stochastic_prefill = False
    logprob_phases = ("decode",)
    supports_chunked_prefill = False

    def warmup(self):
        model = self.sampling_model
        previous = model._bind_gdn_prefill_scratch()
        try:
            model.capture_prefill_trace_chunked(
                self.generator.mesh_device,
                self.page_table[:1],
                chunk_size=2048,
                capture_chunk_trace=True,
            )
        finally:
            model._unbind_gdn_prefill_scratch(previous)
        self.generator.warmup_model_decode(
            self.kv_cache,
            self.enable_trace,
            self.capacity,
            self.page_table.shape[1],
            can_sample_on_device=True,
        )

    def prefill(self, admitted):
        if any(s.request.prefill_chunk_ends for s in admitted):
            raise ValueError("Qwen's GDN state does not support external prefix resumption")
        for state in admitted:
            if len(state.prompt_tokens) + state.request.max_tokens > self.slot_token_capacity[state.slot]:
                raise ValueError(f"Request exceeds allocated KV pages at slot {state.slot}")
        logits = self.sampling_model.prefill_paged_slots(
            [torch.tensor([s.prompt_tokens], dtype=torch.long) for s in admitted],
            self.page_table[[s.slot for s in admitted]],
            [s.slot for s in admitted],
            valid_lens=[len(s.prompt_tokens) for s in admitted],
        )
        # This is the demo's greedy host bootstrap. All tested sampling parameters
        # are exercised by the actual TT decode sampler; no substitute host
        # stochastic sampler is claimed as TT prefill sampling coverage.
        return {
            state.slot: int(logit.reshape(-1, self.vocab_size)[-1].argmax()) for state, logit in zip(admitted, logits)
        }

    def describe(self):
        return dict(super().describe(), prefill_execution="model-owned traced chunks", prefill_sampling="host argmax")


@contextmanager
def open_adapter(execution_mode, *, backend, sku=None, skip_model_load=False):
    import ttnn
    from models.demos.utils.trace_region_sizes import resolve_trace_region_size

    sku = select_sku(backend, ttnn.get_arch_name(), ttnn.get_num_devices(), sku)
    profile, hardware = PROFILES[backend], HARDWARE[sku]
    if skip_model_load and profile.family != "gpt_oss":
        raise ValueError("--model-behavior-skip-model-load is supported only by the GPT-OSS cache factory")
    env = {"HF_MODEL": os.environ.get("HF_MODEL", profile.hf_model), "MESH_DEVICE": hardware.mesh_name}
    if profile.family == "gpt_oss":
        env["TT_MM_THROTTLE_PERF"] = "2"
    mesh = parent_mesh = adapter = None
    with patch.dict(os.environ, env):
        if hardware.shape == (1, 1):
            fabric = ttnn.FabricConfig.DISABLED
        elif profile.family == "gemma":
            # Match Gemma's production test factory and linear CCL topology.
            fabric = ttnn.FabricConfig.FABRIC_1D
        else:
            fabric = ttnn.FabricConfig.FABRIC_1D_RING
        ttnn.set_fabric_config(fabric)
        try:
            # Galaxy fabric startup needs its neighboring chips initialized.
            # Keep the model on its CI-sized submesh when testing Gemma here.
            galaxy_submesh = profile.family == "gemma" and sku == "wh_llmbox_perf" and ttnn.get_num_devices() == 32
            parent_mesh = ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(4, 8) if galaxy_submesh else ttnn.MeshShape(*hardware.shape),
                trace_region_size=resolve_trace_region_size(backend, sku),
            )
            mesh = parent_mesh.create_submesh(ttnn.MeshShape(*hardware.shape)) if galaxy_submesh else parent_mesh
            mesh.enable_program_cache()
            cls = QwenAdapter if profile.family == "qwen" else TransformersAdapter
            adapter = cls(mesh, execution_mode, backend, sku, skip_model_load=skip_model_load)
            adapter.warmup()
            yield adapter
        finally:
            try:
                if adapter is not None:
                    adapter.close()
            finally:
                try:
                    if mesh is not None and mesh is not parent_mesh:
                        ttnn.close_mesh_device(mesh)
                finally:
                    try:
                        if parent_mesh is not None:
                            ttnn.close_mesh_device(parent_mesh)
                    finally:
                        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
