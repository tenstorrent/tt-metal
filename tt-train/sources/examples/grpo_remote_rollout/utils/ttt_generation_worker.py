# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Data-parallel tt-transformers generation worker: owns a [1, N] mesh, splits it into
N [1, 1] submeshes, and runs prefill/decode concurrently across one model per submesh
via a single Generator (data_parallel == N).

On-device sampling: temperature/top_k/top_p/seed are baked into each submesh's decode
trace at construction, so generate()'s per-call temperature/top_p/seed are IGNORED.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, List, Optional, Sequence

import ttnn
from models.common.sampling import SamplingParams

OptimizationsFn = Callable[[int, str], Any]


class TttGenerationWorker:
    """Owns the caller's already-open [1, N] mesh (never closes it) and its submeshes."""

    def __init__(
        self,
        *,
        mesh_device: Any,
        model_source: str,
        max_batch_size: int,
        max_seq_len: int,
        instruct: bool,
        optimizations: OptimizationsFn,
        stop_token_ids: Sequence[int],
        pad_token_id: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        paged_block_size: int = 32,
        min_num_blocks: int = 1024,
        dummy_weights: bool = True,
    ) -> None:
        import torch

        from models.tt_transformers.tt.common import PagedAttentionConfig
        from models.tt_transformers.tt.generator import Generator, create_submeshes
        from models.tt_transformers.tt.model import Transformer
        from models.tt_transformers.tt.model_config import ModelArgs

        self.parent_mesh: Any = mesh_device
        self._dtype: Any = ttnn.bfloat16
        self._stop_token_ids: frozenset[int] = frozenset(int(t) for t in stop_token_ids)
        self._pad_token_id: int = int(pad_token_id)

        # one [1,1] submesh per device of the parent mesh
        self._data_parallel: int = mesh_device.get_num_devices()
        self.submeshes: List[Any] = create_submeshes(mesh_device, self._data_parallel)
        assert (
            len(self.submeshes) == self._data_parallel
        ), f"expected {self._data_parallel} submeshes, got {len(self.submeshes)}"

        # max_batch_size is per-submesh; the global batch spans all submeshes
        self._max_batch_size_per_dp: int = int(max_batch_size)
        self._global_batch_size: int = self._max_batch_size_per_dp * self._data_parallel

        os.environ["HF_MODEL"] = model_source  # ModelArgs reads HF_MODEL from env

        # paged block-table sizing (per submesh), sized for worst-case prompt+decode
        required_blocks_per_user = (max_seq_len + paged_block_size - 1) // paged_block_size
        max_num_blocks = max(min_num_blocks, self._max_batch_size_per_dp * required_blocks_per_user)
        blocks_per_user = max_num_blocks // self._max_batch_size_per_dp
        max_num_blocks = blocks_per_user * self._max_batch_size_per_dp
        self._paged_attention_config = PagedAttentionConfig(block_size=paged_block_size, max_num_blocks=max_num_blocks)
        self._paged_cache_max_seq_len = paged_block_size * blocks_per_user

        # global page table; decode_forward chunks it per submesh
        base = torch.arange(max_num_blocks, dtype=torch.int32).repeat(self._data_parallel)
        self.page_table = base.reshape(self._global_batch_size, blocks_per_user)

        # one model per submesh, reusing one host state_dict (DP copies, not shards)
        self.model_args: List[Any] = []
        self.models: List[Any] = []
        self.tt_kv_cache: List[Any] = []
        state_dict = None
        for submesh in self.submeshes:
            model_args = ModelArgs(
                submesh,
                instruct=instruct,
                max_batch_size=self._max_batch_size_per_dp,
                optimizations=lambda ma: optimizations(ma.n_layers, ma.model_name),
                max_seq_len=max_seq_len,
                cache_hf=True,
                dummy_weights=dummy_weights,  # fast boot; first update_weights() installs real weights
            )
            model_args.lm_head_dtype = ttnn.bfloat16
            model_args.ccl_dtype = ttnn.bfloat16
            if state_dict is None:
                state_dict = model_args.load_state_dict()
            weight_cache_path = model_args.weight_cache_path(self._dtype)
            model = Transformer(
                args=model_args,
                mesh_device=submesh,
                dtype=self._dtype,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                paged_attention_config=self._paged_attention_config,
            )
            self.model_args.append(model_args)
            self.models.append(model)
            self.tt_kv_cache.append([layer.attention.layer_past for layer in model.layers])

        # tokenizer=None: unused here (stop/pad IDs live on this class)
        self.generator = Generator(
            model=self.models,
            model_args=self.model_args,
            mesh_device=self.parent_mesh,
            tokenizer=None,
        )

        # baked into each submesh's decode trace at first capture, so pin once
        for model in self.models:
            assert model.sampling is not None, (
                "TttGenerationWorker requires on-device sampling support, but model.sampling "
                "is None for this configuration (vocab_size / mesh shape combination unsupported)."
            )
        self._sampling_params = SamplingParams(
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            seed=seed,
        )

    def generate(
        self,
        prompts: List[List[int]],
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        enable_trace: bool = True,
        stop_at_eos: bool = True,
    ) -> List[List[int]]:
        """Prefill + decode a token-ID prompt batch, data-parallel across submeshes. The
        batch is padded to the global size; temperature/top_p/seed are ignored (baked in)."""
        import torch

        del temperature, top_p, seed  # baked into self._sampling_params

        if max_new_tokens == 0:
            return [[] for _ in prompts]

        _t_total = time.perf_counter()

        prompts, prompt_lens, active_batch_size = self._prepare_prompt_batch(prompts, max_new_tokens)
        batch_size = len(prompts)  # == self._global_batch_size
        max_prompt_len = max(prompt_lens)
        print(
            f"[TttGenerationWorker] generate() start: data_parallel={self._data_parallel}, "
            f"active_batch_size={active_batch_size}, global_batch_size={batch_size}, "
            f"max_prompt_len={max_prompt_len}, max_new_tokens={max_new_tokens}, enable_trace={enable_trace}",
        )

        pad_id = self._pad_token_id
        input_tokens_prefill_pt = torch.full((batch_size, max_prompt_len), pad_id, dtype=torch.int32)
        for i, p in enumerate(prompts):
            input_tokens_prefill_pt[i, : len(p)] = torch.tensor(p, dtype=torch.int32)

        self._reset_kv_cache()

        # On-device sampling -> prefill returns (tokens, log_probs); tokens is [global_batch, 1].
        _t_prefill = time.perf_counter()
        prefill_out = self.generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=self.page_table,
            kv_cache=self.tt_kv_cache,
            prompt_lens=prompt_lens,
            sampling_params=self._sampling_params,
            warmup_prefill=False,
            enable_trace=enable_trace,
        )
        prefilled_token = (prefill_out[0] if isinstance(prefill_out, tuple) else prefill_out).reshape(-1)
        _prefill_s = time.perf_counter() - _t_prefill
        prefill_real_tokens = sum(prompt_lens[:active_batch_size])
        print(
            f"[TttGenerationWorker] generate(): prefill done in {_prefill_s:.2f}s "
            f"({batch_size} users, {prefill_real_tokens} real prompt tokens)",
        )

        completions: List[List[int]] = [[] for _ in range(batch_size)]
        user_done = [False] * batch_size
        for u in range(active_batch_size, batch_size):
            user_done[u] = True
        stop_ids = self._stop_token_ids if stop_at_eos else frozenset()

        def _collect_step(step_tokens: List[int]) -> None:
            for u in range(batch_size):
                if user_done[u]:
                    continue
                tok = step_tokens[u]
                if stop_at_eos and tok in stop_ids:
                    user_done[u] = True
                else:
                    completions[u].append(tok)

        _collect_step([int(t) for t in prefilled_token.tolist()])  # first token came from prefill

        if all(user_done) or max_new_tokens <= 1:
            print(
                f"[TttGenerationWorker] generate() done (no decode loop): "
                f"total={time.perf_counter() - _t_total:.2f}s",
            )
            return completions[:active_batch_size]

        current_pos = torch.tensor(prompt_lens, dtype=torch.int32)
        out_tok = prefilled_token.unsqueeze(1)  # stays on device; decoding continues on-device

        READ_EVERY = 4
        buffered_reads: List[Any] = []
        read_events: Any = None

        def _drain() -> None:
            for ev in read_events:
                ttnn.event_synchronize(mesh_event=ev)
            for step_reads in buffered_reads:
                gathered = self.generator.process_decode_output_host(step_reads, is_tokens=True)
                tokens = gathered[0] if isinstance(gathered, tuple) else gathered
                _collect_step([int(t) for t in tokens.reshape(-1).tolist()])

        _t_decode = time.perf_counter()
        steps_executed = 0
        for step in range(max_new_tokens - 1):
            decoded = self.generator.decode_forward(
                out_tok,
                current_pos,
                page_table=self.page_table,
                kv_cache=self.tt_kv_cache,
                enable_trace=enable_trace,
                sampling_params=self._sampling_params,
                reset_batch=(step == 0),
                prompt_tokens=input_tokens_prefill_pt,
                output_tokens=out_tok,
                read_from_device=False,
            )
            step_reads, read_events = self.generator.read_decode_output(decoded, async_read=True)
            buffered_reads.append(step_reads)
            current_pos = current_pos + 1
            steps_executed += 1
            if (step + 1) % READ_EVERY == 0:
                _drain()
                buffered_reads = []
                if stop_at_eos and all(user_done):
                    break

        if buffered_reads:
            _drain()

        _decode_s = time.perf_counter() - _t_decode
        total_s = time.perf_counter() - _t_total
        decode_active_tokens = sum(len(c) for c in completions[:active_batch_size])
        overall_tok_s = (decode_active_tokens / total_s) if total_s > 0 else 0.0
        print(
            f"[TttGenerationWorker] generate() done: total={total_s:.2f}s "
            f"(prefill={_prefill_s:.2f}s, decode={_decode_s:.2f}s over {steps_executed} steps), "
            f"completion_tokens={decode_active_tokens} -> {overall_tok_s:.1f} tok/s overall",
        )
        return completions[:active_batch_size]

    def update_weights(self, per_submesh: List[dict]) -> None:
        """Apply one received HF-keyed weight dict per submesh (order matches
        ``self.submeshes`` / the bridge's replication targets)."""
        assert len(per_submesh) == len(
            self.models
        ), f"update_weights got {len(per_submesh)} dicts but worker has {len(self.models)} submeshes"
        for model, hf_dict in zip(self.models, per_submesh):
            model.update_weights(hf_dict)

    def _reset_kv_cache(self) -> None:
        for model in self.models:
            for layer in model.layers:
                k_cache, v_cache = layer.attention.layer_past
                ttnn.mul(k_cache, 0, output_tensor=k_cache)
                ttnn.mul(v_cache, 0, output_tensor=v_cache)
        self.generator.prev_page_table = None

    def _prepare_prompt_batch(
        self, prompts: List[List[int]], max_new_tokens: int
    ) -> tuple[List[List[int]], List[int], int]:
        assert max_new_tokens >= 0, "max_new_tokens must be non-negative"

        active_batch_size = len(prompts)
        assert 0 < active_batch_size <= self._global_batch_size, (
            f"generate() got {active_batch_size} prompts but worker global batch is "
            f"{self._global_batch_size} (data_parallel={self._data_parallel} x "
            f"max_batch_size={self._max_batch_size_per_dp})"
        )

        normalized_prompts = [[int(tok) for tok in prompt] for prompt in prompts]
        prompt_lens = [len(p) for p in normalized_prompts]
        assert min(prompt_lens) > 0, "empty prompts are not supported"

        max_prefill_len = self.model_args[0].max_seq_len - max_new_tokens
        assert (
            max_prefill_len > 0
        ), f"max_new_tokens ({max_new_tokens}) must be smaller than max_seq_len ({self.model_args[0].max_seq_len})"

        if max(prompt_lens) > max_prefill_len:
            normalized_prompts = [p[-max_prefill_len:] for p in normalized_prompts]
            prompt_lens = [len(p) for p in normalized_prompts]

        max_prompt_len = max(prompt_lens)
        assert max_prompt_len + max_new_tokens <= self._paged_cache_max_seq_len, (
            f"prompt prefill tokens ({max_prompt_len}) + decode tokens ({max_new_tokens}) "
            f"must be <= paged-cache capacity ({self._paged_cache_max_seq_len})"
        )

        # Pad to the global batch so every submesh's slots are filled.
        if active_batch_size < self._global_batch_size:
            filler_prompt = [int(self._pad_token_id)]
            pad_slots = self._global_batch_size - active_batch_size
            normalized_prompts.extend([filler_prompt] * pad_slots)
            prompt_lens.extend([1] * pad_slots)

        return normalized_prompts, prompt_lens, active_batch_size
