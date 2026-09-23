# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Galaxy-specific tensor packing; request scenarios live outside this module."""

import os
from contextlib import contextmanager
from unittest.mock import patch

import torch

from tests.model_behavior.adapters.paged import PagedAdapter
from tests.model_behavior.driver import Sample


class GalaxyLlamaAdapter(PagedAdapter):
    duplicate_seed_policy = "identical"
    capacity = 32
    max_seq_len = 8192
    block_size = 64
    physical_blocks = 1024

    def __init__(self, mesh_device, execution_mode):
        from models.common.sampling import SamplingParams
        from models.demos.llama3_70b_galaxy.demo.text_demo import create_tt_model
        from models.demos.llama3_70b_galaxy.tt.generator import Generator
        from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations

        self.execution_mode = execution_mode
        self.enable_trace = execution_mode == "traced"
        self.sampling_params_type = SamplingParams
        args, model, _, kv_cache = create_tt_model(
            mesh_device=mesh_device,
            instruct=True,
            max_batch_size=self.capacity,
            optimizations=LlamaOptimizations.performance,
            max_seq_len=self.max_seq_len,
            num_layers=80,
            dummy_weights=False,
            page_params={"page_block_size": self.block_size, "page_max_num_blocks": self.physical_blocks},
            use_paged_kv_cache=True,
        )
        self.generator = Generator(model, args, mesh_device)
        self.kv_cache = kv_cache
        self.vocab_size = args.vocab_size
        self.prefill_events = []
        # Block zero is the generator's sink for prefill padding. Give each slot
        # disjoint, stable pages and keep the same tensor shape throughout the run.
        # Four boundary slots support long prompts while the other 28 retain
        # 1024-token allocations. All 960 owned pages fit in the 1024-block cache.
        self.long_context_slots = (0, 15, 16, 31)
        self.slot_token_capacity = [8192 if slot in self.long_context_slots else 1024 for slot in range(self.capacity)]
        self.page_table = torch.zeros((self.capacity, self.max_seq_len // self.block_size), dtype=torch.int32)
        next_page = 1
        for slot, length in enumerate(self.slot_token_capacity):
            count = length // self.block_size
            self.page_table[slot, :count] = torch.arange(next_page, next_page + count, dtype=torch.int32)
            next_page += count
        assert next_page <= self.physical_blocks

    def encode(self, prompt):
        return tuple(int(token) for token in self.generator.model_args.encode_prompt(prompt, instruct=True))

    def decode_tokens(self, tokens):
        return self.generator.tokenizer.decode(tokens, skip_special_tokens=True)

    @contextmanager
    def _observe_sampling(self, states):
        """Read the same logits consumed by the device sampler for a CPU oracle.

        These scenarios use no penalties: logprobs are defined on the original
        model logits, before temperature/top-k/top-p filtering. Reading them is
        diagnostic only; the real sampler still produces every returned token.
        """
        captured = []
        if not any(state.request.sampling.enable_log_probs for state in states):
            yield captured
            return
        import ttnn

        sampling = self.generator.model.sampling
        original = sampling.sample

        def observe(logits, **kwargs):
            if getattr(self.generator, "warming_up_prefill", False):
                return original(logits, **kwargs)
            shards = ttnn.get_device_tensors(logits)
            # Galaxy replicates along columns and shards the vocabulary along
            # rows. Read one column, in row order, including vocabulary padding.
            rows, columns = tuple(self.generator.mesh_device.shape)
            host = torch.cat([ttnn.to_torch(shards[row * columns]).float() for row in range(rows)], dim=-1)
            if host.shape[-1] != self.generator.model_args.padded_vocab_size:
                raise ValueError(f"Unexpected full-vocabulary logits shape: {host.shape}")
            reference = torch.log_softmax(host.reshape(self.capacity, -1), dim=-1)
            result = original(logits, **kwargs)
            ids = result[0]
            while isinstance(ids, (list, tuple)):
                ids = ids[0]
            ids = ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(-1).long()
            captured.append((ids, reference.gather(1, ids[:, None]).reshape(-1)))
            return result

        with patch.object(sampling, "sample", observe):
            yield captured

    def _map_samples(self, states, tokens, logprobs, references, *, prefill):
        tokens = tokens.reshape(-1)
        expected = len(states) if prefill else self.capacity
        if tokens.numel() != expected:
            raise ValueError(f"Returned {tokens.numel()} tokens; expected {expected} ({prefill=})")
        if references and (logprobs is None or logprobs.numel() != expected):
            raise ValueError(f"Logprob rows do not align with {expected} returned tokens ({prefill=})")
        if references and len(references) not in (1, len(states) if prefill else 1):
            raise ValueError(f"Unexpected number of sampler invocations: {len(references)}")
        output = {}
        for row, state in enumerate(states):
            index = row if prefill else state.slot
            token = int(tokens[index])
            output[state.slot] = token
            if state.request.sampling.enable_log_probs:
                if not references:
                    raise ValueError("No model logits captured for logprob validation")
                # Seeded prefill samples each request from canonical row zero;
                # unseeded prefill and decode sample a physical-slot batch.
                canonical = prefill and any(s.request.sampling.seed is not None for s in states)
                ids, ref = references[row if canonical else 0]
                ref_index = 0 if canonical else state.slot
                if int(ids[ref_index]) != token:
                    raise ValueError(f"Returned token is not the sampled token at slot {state.slot}")
                output[state.slot] = Sample(token, float(logprobs.reshape(-1)[index]), float(ref[ref_index]))
        return output

    def _prefill_call(self, admitted, lengths, offsets, params):
        slots = [state.slot for state in admitted]
        tokens = torch.zeros((len(admitted), max(lengths)), dtype=torch.long)
        for row, state in enumerate(admitted):
            tokens[row, : lengths[row]] = torch.tensor(state.prompt_tokens[: lengths[row]], dtype=torch.long)
        return self.generator.prefill_forward_text(
            tokens,
            page_table=self.page_table[slots],
            kv_cache=self.kv_cache,
            prompt_lens=lengths,
            empty_slots=slots,
            sampling_params=params,
            enable_trace=self.enable_trace,
            start_pos=offsets,
        )

    def describe(self):
        return {
            "backend": "galaxy-llama70b",
            "model": self.generator.model_args.model_name,
            "execution": self.execution_mode,
            "capacity": self.capacity,
            "duplicate_seed_policy": self.duplicate_seed_policy,
            "block_size": self.block_size,
            "page_table": self.page_table.tolist(),
            "slot_token_capacity": self.slot_token_capacity,
            "prefill_chunks": self.prefill_events,
            "decode_trace_count": sum(t is not None for t in self.generator.trace_ids_decode.values()),
            "sampling_trace_count": sum(
                state["id"] is not None for state in self.generator.model.sampling._trace_states.values()
            ),
        }

    def close(self):
        # Sampling owns separate traces. Generator destruction releases model
        # traces and closes CCL before the context manager closes the mesh.
        try:
            self.generator.model.sampling.reset_trace()
        finally:
            self.generator = None
            self.kv_cache = None


@contextmanager
def open_adapter(execution_mode):
    import ttnn
    from models.demos.llama3_70b_galaxy.tt import prefetcher_common

    if not os.environ.get("HF_MODEL"):
        raise ValueError("Set HF_MODEL to the Llama-3.3-70B-Instruct checkpoint before running this adapter")
    if ttnn.get_arch_name() != "wormhole_b0" or ttnn.get_num_devices() != 32:
        raise RuntimeError("The galaxy-llama70b adapter requires a 32-device Wormhole Galaxy")
    mesh = None
    adapter = None
    # Match the Galaxy demo's model-startup reset: this module-level tensor is
    # tied to one mesh and must not survive the eager -> traced model rebuild.
    prefetcher_common.global_tt_tensor_address = None
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    try:
        mesh = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(8, 4),
            dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER, ttnn.DispatchCoreAxis.COL),
            worker_l1_size=1344544,
            trace_region_size=220000000 if execution_mode == "traced" else 0,
        )
        mesh.enable_program_cache()
        adapter = GalaxyLlamaAdapter(mesh, execution_mode)
        yield adapter
    finally:
        try:
            if adapter is not None:
                adapter.close()
        finally:
            try:
                prefetcher_common.global_tt_tensor_address = None
                if mesh is not None:
                    ttnn.close_mesh_device(mesh)
            finally:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
