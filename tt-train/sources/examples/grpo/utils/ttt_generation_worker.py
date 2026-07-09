# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Family-agnostic tt-transformers generation worker.

Hosts a ``Transformer`` on one ``ttnn.MeshDevice``. ``generate`` is the
``generate_fn`` and ``update_weights`` the ``on_weights_received`` callback of
:class:`MPIRolloutServer`.

Sampling is on-device: ``temperature`` / ``top_k`` / ``top_p`` / ``seed`` are
baked into the captured decode trace at construction. The per-call
``temperature`` / ``top_p`` / ``seed`` kwargs of :meth:`generate` are accepted
for RPC-signature stability but IGNORED -- mutating them would silently diverge
from the trace.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

import ttnn
from models.common.sampling import SamplingParams

OptimizationsFn = Callable[[int, str], Any]


class TttGenerationWorker:
    """Generic tt-transformers generation worker.

    Does NOT open/close the device: the caller passes an already-open
    ``mesh_device`` and owns its lifetime.
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
        from models.tt_transformers.tt.generator import Generator
        from models.tt_transformers.tt.model import Transformer
        from models.tt_transformers.tt.model_config import ModelArgs

        self.mesh_device: Any = mesh_device
        self._model_source: str = model_source
        self._dtype: Any = ttnn.bfloat16
        self._max_batch_size: int = int(max_batch_size)
        self._stop_token_ids: frozenset[int] = frozenset(int(t) for t in stop_token_ids)
        self._pad_token_id: int = int(pad_token_id)

        # ModelArgs reads HF_MODEL from the env for checkpoint + tokenizer paths.
        os.environ["HF_MODEL"] = model_source

        # dummy_weights=True boots fast (skips HF tokenizer load + Hub download);
        # first update_weights() overwrites the dummy state with real weights.
        self.model_args = ModelArgs(
            self.mesh_device,
            instruct=instruct,
            max_batch_size=max_batch_size,
            optimizations=lambda ma: optimizations(ma.n_layers, ma.model_name),
            max_seq_len=max_seq_len,
            cache_hf=True,
            # Hard-coded: short-circuits the dump_tensor_flatbuffer collective that
            # would otherwise deadlock with an asymmetric peer.
            disable_disk_cache=True,
            dummy_weights=dummy_weights,
        )
        self.model_args.lm_head_dtype = ttnn.bfloat16
        self.model_args.ccl_dtype = ttnn.bfloat16

        # Paged-attention block-table sizing: big enough that worst-case
        # prompt+decode never overflows max_seq_len.
        required_blocks_per_user = (max_seq_len + paged_block_size - 1) // paged_block_size
        max_num_blocks = max(min_num_blocks, max_batch_size * required_blocks_per_user)
        blocks_per_user = max_num_blocks // max_batch_size
        max_num_blocks = blocks_per_user * max_batch_size
        self._paged_attention_config = PagedAttentionConfig(
            block_size=paged_block_size,
            max_num_blocks=max_num_blocks,
        )
        self._paged_cache_max_seq_len = paged_block_size * blocks_per_user
        self.page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(max_batch_size, blocks_per_user)

        state_dict = self.model_args.load_state_dict()

        self.model = Transformer(
            args=self.model_args,
            mesh_device=self.mesh_device,
            dtype=self._dtype,
            state_dict=state_dict,
            weight_cache_path=None,
            paged_attention_config=self._paged_attention_config,
        )
        self.kv_cache = [layer.attention.layer_past for layer in self.model.layers]

        # tokenizer=None: Generator only reads its tokenizer from multimodal
        # helpers we don't call; stop/pad IDs live on this class instead.
        self.generator = Generator(
            model=[self.model],
            model_args=[self.model_args],
            mesh_device=self.mesh_device,
            tokenizer=None,
        )

        # Pin one SamplingParams: decode_forward fuses the sampling kernel into
        # the captured trace and its return slot then carries sampled token IDs
        # instead of raw logits. Values are baked in at first capture, so pin
        # once here -- runtime changes would be ignored or force a re-capture.
        assert self.model.sampling is not None, (
            "TttGenerationWorker requires on-device sampling support, but "
            "model.sampling is None for this configuration (vocab_size / "
            "mesh shape combination unsupported)."
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
        """Prefill + decode a batch of token-ID prompts.

        ``temperature`` / ``top_p`` / ``seed`` are ignored (baked into the trace
        at construction); see the module docstring.
        """
        import torch

        del temperature, top_p, seed  # baked into self._sampling_params

        if max_new_tokens == 0:
            return [[] for _ in prompts]

        _t_total = time.perf_counter()

        prompts, prompt_lens, active_batch_size = self._prepare_prompt_batch(prompts, max_new_tokens)
        batch_size = len(prompts)
        max_prompt_len = max(prompt_lens)
        print(
            f"[TttGenerationWorker] generate() start: active_batch_size={active_batch_size}, "
            f"batch_size={batch_size}, max_prompt_len={max_prompt_len}, "
            f"max_new_tokens={max_new_tokens}, enable_trace={enable_trace}",
        )

        pad_id = self._pad_token_id
        input_tokens_prefill_pt = torch.full((batch_size, max_prompt_len), pad_id, dtype=torch.int32)
        for i, p in enumerate(prompts):
            input_tokens_prefill_pt[i, : len(p)] = torch.tensor(p, dtype=torch.int32)

        kv_cache = [self.kv_cache]
        self._reset_kv_cache()

        # With on-device sampling, prefill_forward_text returns
        # (output_tokens, log_probs); output_tokens is [batch_size, 1] int64.
        _t_prefill = time.perf_counter()
        output_tokens, _log_probs = self.generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=self.page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            sampling_params=self._sampling_params,
            warmup_prefill=False,
            enable_trace=enable_trace,
        )
        prefilled_token = output_tokens.reshape(-1)
        _prefill_s = time.perf_counter() - _t_prefill
        # Un-padded prompt lengths over active users only (filler slots excluded).
        prefill_real_tokens = sum(prompt_lens[:active_batch_size])
        prefill_tok_s = (prefill_real_tokens / _prefill_s) if _prefill_s > 0 else 0.0
        print(
            f"[TttGenerationWorker] generate(): prefill done in {_prefill_s:.2f}s "
            f"({batch_size} users, {prefill_real_tokens} real prompt tokens, "
            f"{prefill_tok_s:.1f} prompt tok/s)",
        )

        completions: List[List[int]] = [[] for _ in range(batch_size)]
        user_done = [False] * batch_size
        for u in range(active_batch_size, batch_size):
            user_done[u] = True
        stop_ids = self._stop_token_ids if stop_at_eos else frozenset()

        active_completion_tokens = 0
        for u in range(batch_size):
            if user_done[u]:
                continue
            tok = int(prefilled_token[u].item())
            if stop_at_eos and tok in stop_ids:
                user_done[u] = True
            else:
                completions[u].append(tok)
                active_completion_tokens += 1

        if all(user_done) or max_new_tokens <= 1:
            print(
                f"[TttGenerationWorker] generate() done (no decode loop needed): "
                f"total={time.perf_counter() - _t_total:.2f}s, "
                f"completion_tokens={active_completion_tokens}",
            )
            return completions[:active_batch_size]

        current_pos = torch.tensor(prompt_lens, dtype=torch.int32)
        # decode_forward expects `tokens` shape [batch_size, 1].
        out_tok = prefilled_token.unsqueeze(1)

        ASYNC_READ_CHUNK = 4

        _t_decode = time.perf_counter()
        steps_executed = 0
        decode_active_tokens = 0

        # ``pending`` is the just-finished chunk drained at the next boundary;
        # ``current`` is the chunk being filled this step.
        pending_hosts: List[Any] = []
        pending_event: Any = None
        current_hosts: List[Any] = []
        current_event: Any = None

        def _drain_chunk(host_chunks: List[Any]) -> None:
            """Fold an event-synced chunk's host tokens into completions /
            user_done / decode_active_tokens ('first stop wins' per user)."""
            nonlocal decode_active_tokens
            if not host_chunks:
                return
            chunk_len = len(host_chunks)
            chunk_arr = np.empty((batch_size, chunk_len), dtype=np.int64)
            for j, host_outs in enumerate(host_chunks):
                # data_parallel == 1: host_outs has one entry, either
                # (tokens_host, log_probs_host) or a bare tokens host tensor.
                h0 = host_outs[0]
                token_host = h0[0] if isinstance(h0, tuple) else h0
                tok_torch = self.model.process_output_decode(token_host, self._max_batch_size, S=1, is_tokens=True)
                chunk_arr[:, j] = tok_torch.to(torch.int64).numpy()

            for j in range(chunk_len):
                for u in range(batch_size):
                    if user_done[u]:
                        continue
                    tok = int(chunk_arr[u, j])
                    if stop_at_eos and tok in stop_ids:
                        user_done[u] = True
                    else:
                        completions[u].append(tok)
                        decode_active_tokens += 1

        broke_on_stop = False
        for step in range(max_new_tokens - 1):
            tt_decode_output = self.generator.decode_forward(
                out_tok,
                current_pos,
                page_table=self.page_table,
                kv_cache=kv_cache,
                enable_trace=enable_trace,
                sampling_params=self._sampling_params,
                reset_batch=(step == 0),
                prompt_tokens=input_tokens_prefill_pt,
                output_tokens=out_tok,
                read_from_device=False,
            )
            host_outs, event_list = self.generator.read_decode_output(tt_decode_output, async_read=True)
            current_hosts.append(host_outs)
            # cq=0 is in-order, so the latest event covers every d2h enqueued
            # earlier in the chunk -- sync only on this one at the boundary.
            current_event = event_list[-1]
            current_pos = current_pos + 1
            steps_executed += 1

            if (step + 1) % ASYNC_READ_CHUNK == 0:
                if pending_event is not None:
                    ttnn.event_synchronize(mesh_event=pending_event)
                    _drain_chunk(pending_hosts)
                    if stop_at_eos and all(user_done):
                        broke_on_stop = True
                        break
                pending_hosts, pending_event = current_hosts, current_event
                current_hosts, current_event = [], None

        # Final flush: even on early EOS break, event_synchronize in-flight
        # events so d2h DMAs finish before the host tensors fall out of scope.
        if pending_event is not None:
            ttnn.event_synchronize(mesh_event=pending_event)
            if not broke_on_stop:
                _drain_chunk(pending_hosts)
        if current_event is not None:
            ttnn.event_synchronize(mesh_event=current_event)
            if not broke_on_stop:
                _drain_chunk(current_hosts)

        active_completion_tokens += decode_active_tokens
        _decode_s = time.perf_counter() - _t_decode
        _per_step_ms = (_decode_s / steps_executed * 1000.0) if steps_executed > 0 else 0.0
        # "active" tok/s = useful tokens for active users; "device" tok/s =
        # everything computed incl. filler slots (matches benchmark figures).
        decode_active_tok_s = (decode_active_tokens / _decode_s) if _decode_s > 0 else 0.0
        decode_device_tokens = steps_executed * batch_size
        decode_device_tok_s = (decode_device_tokens / _decode_s) if _decode_s > 0 else 0.0
        print(
            f"[TttGenerationWorker] generate(): decode loop done in {_decode_s:.2f}s "
            f"({steps_executed} steps, {_per_step_ms:.1f} ms/step, "
            f"active_tokens={decode_active_tokens} -> {decode_active_tok_s:.1f} tok/s, "
            f"device_tokens={decode_device_tokens} (B={batch_size}) -> {decode_device_tok_s:.1f} tok/s)",
        )
        total_s = time.perf_counter() - _t_total
        overall_active_tok_s = (active_completion_tokens / total_s) if total_s > 0 else 0.0
        print(
            f"[TttGenerationWorker] generate() done: total={total_s:.2f}s "
            f"(prefill={_prefill_s:.2f}s, decode={_decode_s:.2f}s), "
            f"completion_tokens={active_completion_tokens} -> {overall_active_tok_s:.1f} tok/s overall",
        )
        return completions[:active_batch_size]

    def update_weights(self, hf_dict: dict) -> None:
        """Apply a received HF-keyed weight dict to the underlying model."""
        self.model.update_weights(hf_dict)

    def _reset_kv_cache(self) -> None:
        for layer in self.model.layers:
            k_cache, v_cache = layer.attention.layer_past
            ttnn.mul(k_cache, 0, output_tensor=k_cache)
            ttnn.mul(v_cache, 0, output_tensor=v_cache)
        self.generator.prev_page_table = None

    def _prepare_prompt_batch(
        self, prompts: List[List[int]], max_new_tokens: int
    ) -> tuple[List[List[int]], List[int], int]:
        assert max_new_tokens >= 0, "max_new_tokens must be non-negative"

        active_batch_size = len(prompts)
        assert 0 < active_batch_size <= self._max_batch_size, (
            f"generate() got {active_batch_size} prompts but worker was built with "
            f"max_batch_size={self._max_batch_size}"
        )

        normalized_prompts = [[int(tok) for tok in prompt] for prompt in prompts]
        prompt_lens = [len(p) for p in normalized_prompts]
        assert min(prompt_lens) > 0, "empty prompts are not supported"

        max_prefill_len = self.model_args.max_seq_len - max_new_tokens
        assert (
            max_prefill_len > 0
        ), f"max_new_tokens ({max_new_tokens}) must be smaller than max_seq_len ({self.model_args.max_seq_len})"

        if max(prompt_lens) > max_prefill_len:
            normalized_prompts = [p[-max_prefill_len:] for p in normalized_prompts]
            prompt_lens = [len(p) for p in normalized_prompts]

        max_prompt_len = max(prompt_lens)
        assert max_prompt_len + max_new_tokens <= self.model_args.max_seq_len, (
            f"prompt prefill tokens ({max_prompt_len}) + decode tokens ({max_new_tokens}) "
            f"must be <= max_seq_len ({self.model_args.max_seq_len})"
        )
        assert max_prompt_len + max_new_tokens <= self._paged_cache_max_seq_len, (
            f"prompt prefill tokens ({max_prompt_len}) + decode tokens ({max_new_tokens}) "
            f"must be <= paged-cache capacity ({self._paged_cache_max_seq_len})"
        )

        if active_batch_size < self._max_batch_size:
            filler_prompt = [int(self._pad_token_id)]
            pad_slots = self._max_batch_size - active_batch_size
            normalized_prompts.extend([filler_prompt] * pad_slots)
            prompt_lens.extend([1] * pad_slots)

        return normalized_prompts, prompt_lens, active_batch_size
