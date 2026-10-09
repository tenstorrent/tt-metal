# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Data-parallel tt-transformers rollout sampler: owns a [1, N] mesh, splits it into
N [1, 1] submeshes, and runs prefill/decode concurrently across one model per submesh
via a single Generator (data_parallel == N).

On-device sampling: temperature/top_k/top_p/seed are baked into each submesh's decode
trace at construction and cannot vary per generate() call.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import ttnn
from huggingface_hub import snapshot_download
from models.common.sampling import SamplingParams
from models.tt_transformers.tt.common import PagedAttentionConfig
from models.tt_transformers.tt.generator import Generator, create_submeshes
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import ModelArgs

from ..grpo_trainer import RolloutBatch, RolloutSampler, check_new_weight_version
from ..rollout_batch_source import validate_rollout_batch

OptimizationsFn = Callable[[int, str], Any]


class TTTRolloutSampler(RolloutSampler):
    """tt-transformers rollout sampler; its log-probs come from tt-transformers' on-device sampling.

    Uses the caller's already-open [1, N] mesh (never closes it) and owns its submeshes.
    Boots with the HF weights of ``model_source`` at weight_version 0.
    """

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
        completions_per_prompt: int,
        max_completion_length: int,
        temperature: float = 1.0,
        top_k: int = 32,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        paged_block_size: int = 32,
        min_num_blocks: int = 1024,
        dummy_weights: bool = False,
    ) -> None:
        self.parent_mesh: Any = mesh_device
        self._dtype: Any = ttnn.bfloat16
        self._stop_token_ids: frozenset[int] = frozenset(int(t) for t in stop_token_ids)
        self._pad_token_id: int = int(pad_token_id)
        self._completions_per_prompt: int = int(completions_per_prompt)
        self._max_completion_length: int = int(max_completion_length)
        self._weight_version: int = 0

        # one [1,1] submesh per device of the parent mesh
        self._data_parallel: int = mesh_device.get_num_devices()
        self.submeshes: List[Any] = create_submeshes(mesh_device, self._data_parallel)
        if len(self.submeshes) != self._data_parallel:
            raise RuntimeError(f"expected {self._data_parallel} submeshes, got {len(self.submeshes)}")

        # max_batch_size is per-submesh; one call serves max_batch_size slots on every submesh
        self._max_batch_size_per_dp: int = int(max_batch_size)
        self._slots_per_call: int = self._max_batch_size_per_dp * self._data_parallel

        os.environ["HF_MODEL"] = model_source  # ModelArgs reads HF_MODEL from env
        if not dummy_weights and not os.path.isdir(model_source):
            # ModelArgs reads the HF config with local_files_only=True under CI. CI does not have
            # every model available locally, so download it as needed.
            print(f"[TTTRolloutSampler] Downloading HuggingFace snapshot for {model_source}")
            snapshot_download(
                repo_id=model_source,
                allow_patterns=["*.safetensors", "*.bin", "*.json", "*.model", "*.txt"],
            )

        # paged block-table sizing (per submesh), sized for worst-case prompt+decode
        required_blocks_per_user = (max_seq_len + paged_block_size - 1) // paged_block_size
        max_num_blocks = max(min_num_blocks, self._max_batch_size_per_dp * required_blocks_per_user)
        blocks_per_user = max_num_blocks // self._max_batch_size_per_dp
        max_num_blocks = blocks_per_user * self._max_batch_size_per_dp
        self._paged_attention_config = PagedAttentionConfig(block_size=paged_block_size, max_num_blocks=max_num_blocks)
        self._paged_cache_max_seq_len = paged_block_size * blocks_per_user

        # global page table; decode_forward chunks it per submesh
        base = torch.arange(max_num_blocks, dtype=torch.int32).repeat(self._data_parallel)
        self.page_table = base.reshape(self._slots_per_call, blocks_per_user)

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
                dummy_weights=dummy_weights,
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
            if model.sampling is None:
                raise RuntimeError(
                    "TTTRolloutSampler requires on-device sampling support, but model.sampling "
                    "is None for this configuration (vocab_size / mesh shape combination unsupported)."
                )
        # Per-slot lists (not scalars): scalar SamplingParams fields are padded
        # to the greedy default on non-slot-0 requests by scatter_sampling_params_to_slots.
        n = self._slots_per_call
        self._sampling_params = SamplingParams(
            temperature=[float(temperature)] * n,
            top_k=[int(top_k)] * n,
            top_p=[float(top_p)] * n,
            seed=[seed] * n if seed is not None else None,
            # Turn on the on-device log-probs kernel. Without this, tt_transformers'
            # process_decode_output_host substitutes torch.ones(...) as a sentinel
            # for the missing log-probs tensor, which makes _generate_impl return
            # 1.0 for every sampled-token log-prob (so np.exp(lp) == e everywhere
            # downstream, including the rollout log-probs pushed on the
            # RolloutQueue). Matches the on-device single-device log-probs path
            # enabled in models/common/sampling/tt_log_probs.py.
            enable_log_probs=[True] * n,
            # 0 → old path (single scalar sampled-token log-prob) rather than the
            # new top-K LogProbsResult path. _generate_impl still expects a plain
            # torch.Tensor of scalars.
            num_logprobs=[0] * n,
        )
        self._sampling_warmed_up = False

    @property
    def weight_version(self) -> int:
        return self._weight_version

    def generate(self, prompts: List[List[int]]) -> RolloutBatch:
        """Generate completions_per_prompt completions per prompt, in calls of at most slots_per_call."""
        version = self._weight_version
        g, W = self._completions_per_prompt, self._max_completion_length
        prompts_x = [list(p) for p in prompts for _ in range(g)]
        completions: List[List[int]] = []
        token_logprobs: List[List[float]] = []
        for start in range(0, len(prompts_x), self._slots_per_call):
            c, lp = self.generate_and_get_log_probs(prompts_x[start : start + self._slots_per_call], max_new_tokens=W)
            completions.extend(c)
            token_logprobs.extend(lp)

        logprobs = np.zeros((len(prompts_x), W), dtype=np.float32)
        for r, (c, lp) in enumerate(zip(completions, token_logprobs)):
            if len(lp) != len(c):
                raise RuntimeError(f"row {r}: {len(lp)} log-probs for {len(c)} tokens")
            logprobs[r, : len(c)] = lp
        batch = RolloutBatch(weight_version=version, prompts=prompts_x, completions=completions, logprobs=logprobs)
        validate_rollout_batch(batch, num_prompts=len(prompts), num_generations=g, max_completion_length=W)
        return batch

    def generate_tokens(
        self,
        prompts: List[List[int]],
        *,
        max_new_tokens: int = 128,
        enable_trace: bool = True,
        stop_at_eos: bool = True,
    ) -> List[List[int]]:
        """Prefill + decode a token-ID prompt batch, data-parallel across submeshes. The
        batch is padded to slots_per_call; sampling params were baked into
        ``self._sampling_params`` at construction and cannot vary per call."""
        completions, _ = self._generate_impl(
            prompts,
            max_new_tokens=max_new_tokens,
            enable_trace=enable_trace,
            stop_at_eos=stop_at_eos,
            collect_logprobs=True,
        )
        return completions

    def generate_and_get_log_probs(
        self,
        prompts: List[List[int]],
        *,
        max_new_tokens: int = 128,
        enable_trace: bool = True,
        stop_at_eos: bool = True,
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """Same as :meth:`generate_tokens`, but also returns per-token sampled log-probs
        (post temperature/top_k/top_p) aligned 1:1 with completion tokens.

        Returns:
            ``(completions, logprobs)`` where ``completions[u][t]`` is the ``t``-th
            emitted token for user ``u`` and ``logprobs[u][t]`` is the log-probability
            of that token under the (post-processed) sampling distribution.
        """
        completions, logprobs = self._generate_impl(
            prompts,
            max_new_tokens=max_new_tokens,
            enable_trace=enable_trace,
            stop_at_eos=stop_at_eos,
            collect_logprobs=True,
        )
        if logprobs is None:
            raise RuntimeError("collect_logprobs=True must return log-probs")
        return completions, logprobs

    def _warm_up_sampling(self, prompts: List[List[int]], *, max_new_tokens: int, stop_at_eos: bool) -> None:
        """Run one untraced generate with the sampler's sampling params before the first trace capture.

        Generator.warmup_model_prefill only warms log-prob sampling at batch size 1, so capturing a
        trace for the full batch with log-probs enabled would otherwise hit uncompiled programs.
        """
        self._sampling_warmed_up = True
        self._generate_impl(
            prompts,
            max_new_tokens=min(max_new_tokens, 2),
            enable_trace=False,
            stop_at_eos=stop_at_eos,
            collect_logprobs=True,
        )

    def _generate_impl(
        self,
        prompts: List[List[int]],
        *,
        max_new_tokens: int,
        enable_trace: bool,
        stop_at_eos: bool,
        collect_logprobs: bool,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """Shared prefill + decode driver behind :meth:`generate_tokens` and
        :meth:`generate_and_get_log_probs`. When ``collect_logprobs`` is True, also
        collects per-token sampled log-probs into a parallel list-of-lists sized to
        match ``completions`` slot-for-slot; when False, returns ``None`` for the
        log-probs.
        """
        if max_new_tokens == 0:
            empty_completions: List[List[int]] = [[] for _ in prompts]
            empty_logprobs: Optional[List[List[float]]] = [[] for _ in prompts] if collect_logprobs else None
            return empty_completions, empty_logprobs

        if enable_trace and not self._sampling_warmed_up:
            self._warm_up_sampling(prompts, max_new_tokens=max_new_tokens, stop_at_eos=stop_at_eos)

        _t_total = time.perf_counter()

        prompts, prompt_lens, active_batch_size = self._prepare_prompt_batch(prompts, max_new_tokens)
        batch_size = len(prompts)  # == self._slots_per_call
        max_prompt_len = max(prompt_lens)
        print(
            f"[TTTRolloutSampler] generate() start: data_parallel={self._data_parallel}, "
            f"active_batch_size={active_batch_size}, slots_per_call={batch_size}, "
            f"max_prompt_len={max_prompt_len}, max_new_tokens={max_new_tokens}, "
            f"enable_trace={enable_trace}, collect_logprobs={collect_logprobs}",
        )

        pad_id = self._pad_token_id
        input_tokens_prefill_pt = torch.full((batch_size, max_prompt_len), pad_id, dtype=torch.int32)
        for i, p in enumerate(prompts):
            input_tokens_prefill_pt[i, : len(p)] = torch.tensor(p, dtype=torch.int32)

        self._reset_kv_cache()

        # On-device sampling -> prefill returns (tokens, log_probs); tokens is [slots_per_call, 1].
        _t_prefill = time.perf_counter()
        prefill_out = self.generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=self.page_table,
            kv_cache=self.tt_kv_cache,
            prompt_lens=prompt_lens,
            sampling_params=self._sampling_params,
            warmup_prefill=True,
            enable_trace=enable_trace,
        )
        prefilled_token = (prefill_out[0] if isinstance(prefill_out, tuple) else prefill_out).reshape(-1)
        prefill_logprobs: Optional[List[float]] = None
        if collect_logprobs:
            if not isinstance(prefill_out, tuple) or len(prefill_out) < 2 or prefill_out[1] is None:
                raise RuntimeError(
                    "TTTRolloutSampler: collect_logprobs=True requires on-device sampling to emit log-probs; "
                    "prefill_forward_text did not return a log-probs tensor. Check the model's sampling config."
                )
            lp_prefill = prefill_out[1]
            if not isinstance(lp_prefill, torch.Tensor):
                raise RuntimeError(
                    "TTTRolloutSampler: prefill log-probs are not a torch.Tensor "
                    f"(got {type(lp_prefill).__name__}); the top-k LogProbsResult path is not supported here. "
                    "Configure sampling with top_k=0 and top_p=1.0 to get scalar per-token log-probs."
                )
            prefill_logprobs = [float(x) for x in lp_prefill.reshape(-1).tolist()]
        _prefill_s = time.perf_counter() - _t_prefill
        prefill_real_tokens = sum(prompt_lens[:active_batch_size])
        print(
            f"[TTTRolloutSampler] generate(): prefill done in {_prefill_s:.2f}s "
            f"({batch_size} users, {prefill_real_tokens} real prompt tokens)",
        )

        completions: List[List[int]] = [[] for _ in range(batch_size)]
        logprobs: Optional[List[List[float]]] = [[] for _ in range(batch_size)] if collect_logprobs else None
        user_done = [False] * batch_size
        for u in range(active_batch_size, batch_size):
            user_done[u] = True
        stop_ids = self._stop_token_ids if stop_at_eos else frozenset()

        def _collect_step(step_tokens: List[int], step_logprobs: Optional[List[float]] = None) -> None:
            for u in range(batch_size):
                if user_done[u]:
                    continue
                tok = step_tokens[u]
                if stop_at_eos and tok in stop_ids:
                    user_done[u] = True
                else:
                    completions[u].append(tok)
                    if logprobs is not None:
                        # step_logprobs may be None on paths that don't have them, in which
                        # case we align the token with 0.0 -- callers of
                        # generate_and_get_log_probs always request them, so in practice
                        # this branch is never taken there.
                        logprobs[u].append(float(step_logprobs[u]) if step_logprobs is not None else 0.0)

        _collect_step(
            [int(t) for t in prefilled_token.tolist()],
            prefill_logprobs,
        )  # first token came from prefill

        if all(user_done) or max_new_tokens <= 1:
            print(
                f"[TTTRolloutSampler] generate() done (no decode loop): "
                f"total={time.perf_counter() - _t_total:.2f}s",
            )
            return (
                completions[:active_batch_size],
                logprobs[:active_batch_size] if logprobs is not None else None,
            )

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
                if isinstance(gathered, tuple):
                    tokens_t = gathered[0]
                    lp_t = gathered[1]
                else:
                    tokens_t = gathered
                    lp_t = None
                step_tokens = [int(t) for t in tokens_t.reshape(-1).tolist()]
                step_lp: Optional[List[float]] = None
                if collect_logprobs:
                    if not isinstance(lp_t, torch.Tensor):
                        raise RuntimeError(
                            "TTTRolloutSampler: decode log-probs are not a torch.Tensor "
                            f"(got {type(lp_t).__name__ if lp_t is not None else 'None'}); "
                            "the top-k LogProbsResult path is not supported here."
                        )
                    step_lp = [float(x) for x in lp_t.reshape(-1).tolist()]
                _collect_step(step_tokens, step_lp)

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
            f"[TTTRolloutSampler] generate() done: total={total_s:.2f}s "
            f"(prefill={_prefill_s:.2f}s, decode={_decode_s:.2f}s over {steps_executed} steps), "
            f"completion_tokens={decode_active_tokens} -> {overall_tok_s:.1f} tok/s overall",
        )
        return (
            completions[:active_batch_size],
            logprobs[:active_batch_size] if logprobs is not None else None,
        )

    def update_weights(self, weights: List[dict], version: int) -> None:
        """Apply one HF-keyed weight dict per submesh (order matches ``self.submeshes`` /
        the bridge's replication targets) and set ``weight_version`` to ``version``."""
        version = check_new_weight_version(self._weight_version, version)
        if len(weights) != len(self.models):
            raise RuntimeError(f"update_weights got {len(weights)} dicts but sampler has {len(self.models)} submeshes")
        for model, hf_dict in zip(self.models, weights):
            model.update_weights(hf_dict)
        self._weight_version = version

    def close(self) -> None:
        """Drop the generator, models and KV caches (and their traces); the mesh stays open."""
        self.generator = None
        self.models = []
        self.tt_kv_cache = []

    def _reset_kv_cache(self) -> None:
        for model in self.models:
            for layer in model.layers:
                k_cache, v_cache = layer.attention.layer_past
                ttnn.fill(k_cache, 0, output_tensor=k_cache)
                ttnn.fill(v_cache, 0, output_tensor=v_cache)
        self.generator.prev_page_table = None

    def _prepare_prompt_batch(
        self, prompts: List[List[int]], max_new_tokens: int
    ) -> tuple[List[List[int]], List[int], int]:
        """Check one call's prompts and pad the batch to slots_per_call with 1-token pad prompts.

        Raises ValueError if a prompt doesn't fit: prompts are never truncated. Returns the
        padded prompts, their lengths, and the number of real prompts.
        """
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")

        active_batch_size = len(prompts)
        if not 0 < active_batch_size <= self._slots_per_call:
            raise ValueError(
                f"got {active_batch_size} prompts; one call serves 1..{self._slots_per_call} "
                f"(data_parallel={self._data_parallel} x max_batch_size={self._max_batch_size_per_dp})"
            )

        normalized_prompts = [[int(tok) for tok in prompt] for prompt in prompts]
        prompt_lens = [len(p) for p in normalized_prompts]
        if min(prompt_lens) == 0:
            raise ValueError("empty prompts are not supported")

        max_seq_len = self.model_args[0].max_seq_len
        max_prompt_len = max_seq_len - max_new_tokens
        if max_prompt_len <= 0:
            raise ValueError(f"max_new_tokens ({max_new_tokens}) must be smaller than max_seq_len ({max_seq_len})")
        too_long = [i for i, n in enumerate(prompt_lens) if n > max_prompt_len]
        if too_long:
            raise ValueError(
                f"{len(too_long)} prompt(s) longer than {max_prompt_len} tokens (max_seq_len - max_new_tokens), "
                f"first at index {too_long[0]} with {prompt_lens[too_long[0]]} tokens"
            )

        if max(prompt_lens) + max_new_tokens > self._paged_cache_max_seq_len:
            raise ValueError(
                f"prompt prefill tokens ({max(prompt_lens)}) + decode tokens ({max_new_tokens}) "
                f"must be <= paged-cache capacity ({self._paged_cache_max_seq_len})"
            )

        pad_slots = self._slots_per_call - active_batch_size
        normalized_prompts.extend([[int(self._pad_token_id)]] * pad_slots)
        prompt_lens.extend([1] * pad_slots)
        return normalized_prompts, prompt_lens, active_batch_size
