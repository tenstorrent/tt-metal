# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side generation driver for Muse-Glimmer-30B: data-parallel replicas, paged KV, traced decode."""

import time

import torch
from loguru import logger

import ttnn


class MuseGlimmerGenerator:
    def __init__(self, models: list, enable_trace: bool = True):
        self.models = models
        self.dp = len(models)
        self.enable_trace = enable_trace
        args = models[0].args
        self.B = args.max_batch_size
        self.n_blocks = args.max_num_blocks_per_user
        self.vocab_size = args.vocab_size
        self.page_table = torch.arange(self.B * self.n_blocks, dtype=torch.int32).reshape(self.B, self.n_blocks)
        self.page_table_tt = []
        self.decode_inputs = []
        for m in models:
            self.page_table_tt.append(
                ttnn.from_torch(
                    self.page_table,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=m.mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(m.mesh_device),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            self.decode_inputs.append(m.allocate_decode_inputs(self.B, self.n_blocks))
        self.trace_ids = [None] * self.dp
        self.device_sampling = all(getattr(m, "sampling", None) is not None for m in models)
        self.page_table_dirty = True  # device page-table copy is stale (first step / after prefill)
        self.last_step_breakdown = {}
        self.trace_outputs = [None] * self.dp

    @property
    def global_batch_size(self) -> int:
        return self.dp * self.B

    def prefill(self, prompts: list[list[int]]) -> torch.Tensor:
        assert len(prompts) <= self.global_batch_size
        self.page_table_dirty = True
        out = torch.zeros(len(prompts), self.vocab_size, dtype=torch.float32)
        for u, p in enumerate(prompts):
            r, slot = u // self.B, u % self.B
            t0 = time.time()
            out[u] = self.models[r].prefill_forward(torch.tensor(p, dtype=torch.int64), slot, self.page_table_tt[r])
            logger.info(f"Prefill user {u} (replica {r}, slot {slot}, {len(p)} tokens): {time.time() - t0:.2f}s")
        return out

    def prefill_tokens(self, prompts: list[list[int]], sampling_params) -> torch.Tensor:
        """Prefill and sample each prompt's first token on device -> LongTensor [n]."""
        assert len(prompts) <= self.global_batch_size
        self.page_table_dirty = True
        out = torch.zeros(len(prompts), dtype=torch.int64)
        loaded = set()
        for u, p in enumerate(prompts):
            r, slot = u // self.B, u % self.B
            m = self.models[r]
            t0 = time.time()
            logits_tt = m.prefill_forward_logits_tt(torch.tensor(p, dtype=torch.int64), slot, self.page_table_tt[r])
            tok_tt = m.sample_on_device(logits_tt, None if r in loaded else sampling_params)
            loaded.add(r)
            out[u] = m.tokens_to_torch(tok_tt, 1)[0]
            ttnn.deallocate(logits_tt)
            ttnn.deallocate(tok_tt)
            logger.info(f"Prefill user {u} (replica {r}, slot {slot}, {len(p)} tokens): {time.time() - t0:.2f}s")
        # Replicas without users still run in every decode step and need the request's parameters.
        for r, m in enumerate(self.models):
            if r not in loaded:
                m.sampling.reset_sampling_params(sampling_params)
        return out

    @staticmethod
    def _to_sampling_params(sampling: dict | None):
        """Our sampling dict (None => greedy) -> SamplingParams (greedy: temp 0 / top_k 1 / top_p 1, as gemma4)."""
        from models.common.sampling.generator import SamplingParams, format_sampling_params

        sp = sampling or {}
        t = float(sp.get("temperature", 1.0) or 0.0)
        if sampling is None or t <= 0:
            return format_sampling_params(SamplingParams(temperature=0.0, top_k=1, top_p=1.0), 32)
        # len(temperature) sets the active lane count; a scalar would leave lanes 1..31 greedy.
        # top_k <= 0 means "no filter": the sampler clamps k to max_top_k (32).
        return format_sampling_params(
            SamplingParams(
                temperature=[t] * 32,
                top_k=int(sp.get("top_k", 32)) or 32,
                top_p=float(sp.get("top_p", 1.0)),
            ),
            32,
        )

    def decode_step(self, tokens: torch.Tensor, positions: torch.Tensor, sampling=None, sampling_params=None):
        """One decode step for all replicas.

        sampling is None -> returns host logits [global_batch, vocab] (fallback path).
        sampling is a `SamplingParams` (device sampling) -> returns next tokens, LongTensor [global_batch].
        sampling_params is passed to the device sampler only when it must be (re)loaded (once per generate()).
        """
        device_sampling = sampling is not None
        tokens = tokens.reshape(-1)
        positions = positions.reshape(-1)
        assert tokens.shape[0] == self.global_batch_size and positions.shape[0] == self.global_batch_size
        bd = {}
        outs = []
        t_a = time.perf_counter()
        copy_pt = self.page_table_dirty
        # Issue all replicas before reading any result so they run concurrently.
        for r, m in enumerate(self.models):
            sl = slice(r * self.B, (r + 1) * self.B)
            host = m.prepare_decode_inputs_host(tokens[sl], positions[sl], self.page_table)
            m.copy_decode_inputs(host, self.decode_inputs[r], copy_page_table=copy_pt)
        self.page_table_dirty = False
        t_b = time.perf_counter()
        for r, m in enumerate(self.models):
            if not self.enable_trace:
                outs.append(m.decode_forward(*self.decode_inputs[r]))
            elif self.trace_ids[r] is None:
                # eager compile run, then capture; the captured run's output buffer is the trace output
                ttnn.deallocate(m.decode_forward(*self.decode_inputs[r]))
                ttnn.synchronize_device(m.mesh_device)
                self.trace_ids[r] = ttnn.begin_trace_capture(m.mesh_device, cq_id=0)
                self.trace_outputs[r] = m.decode_forward(*self.decode_inputs[r])
                ttnn.end_trace_capture(m.mesh_device, self.trace_ids[r], cq_id=0)
                # Capture does not execute; run the trace once for this step's real result.
                ttnn.execute_trace(m.mesh_device, self.trace_ids[r], cq_id=0, blocking=False)
                outs.append(self.trace_outputs[r])
            else:
                ttnn.execute_trace(m.mesh_device, self.trace_ids[r], cq_id=0, blocking=False)
                outs.append(self.trace_outputs[r])
        t_c = time.perf_counter()
        bd["host_input_copy_s"] = t_b - t_a
        bd["device_issue_s"] = t_c - t_b
        if device_sampling:
            # Sampling runs eagerly (its force-argmax all_gather semaphore must not be frozen into a trace).
            toks_tt = [m.sample_on_device(outs[r], sampling_params) for r, m in enumerate(self.models)]
            t_d = time.perf_counter()
            res = [m.tokens_to_torch(toks_tt[r], self.B) for r, m in enumerate(self.models)]
            t_e = time.perf_counter()
            for r in range(self.dp):
                ttnn.deallocate(toks_tt[r])
        else:
            t_d = time.perf_counter()
            res = [m.logits_to_torch(outs[r], self.B) for r, m in enumerate(self.models)]
            t_e = time.perf_counter()
        if not self.enable_trace:
            for o in outs:
                ttnn.deallocate(o)
        bd["sample_s"] = t_d - t_c
        bd["readback_s"] = t_e - t_d
        self.last_step_breakdown = bd
        return torch.cat(res, dim=0)

    def decode_step_tokens(self, tokens: torch.Tensor, positions: torch.Tensor, sampling, sampling_params=None):
        """Device-sampled next tokens, LongTensor [global_batch]. `sampling` is a SamplingParams."""
        return self.decode_step(tokens, positions, sampling=sampling, sampling_params=sampling_params)

    @staticmethod
    def _sample(logits: torch.Tensor, sampling: dict | None, gen: torch.Generator) -> torch.Tensor:
        logits = torch.nan_to_num(logits.float(), nan=float("-inf"))
        if sampling is None:
            return logits.argmax(dim=-1)
        temperature = float(sampling.get("temperature", 1.0))
        top_k = int(sampling.get("top_k", 0))
        top_p = float(sampling.get("top_p", 1.0))
        x = logits.float() / max(temperature, 1e-5)
        if top_k > 0:
            kth = torch.topk(x, min(top_k, x.shape[-1]), dim=-1).values[..., -1:]
            x = x.masked_fill(x < kth, float("-inf"))
        if top_p < 1.0:
            sv, si = torch.sort(x, descending=True, dim=-1)
            probs = torch.softmax(sv, dim=-1)
            cum = probs.cumsum(dim=-1)
            remove = (cum - probs) > top_p  # keep the first token that crosses top_p
            sv = sv.masked_fill(remove, float("-inf"))
            x = torch.full_like(x, float("-inf")).scatter(-1, si, sv)
        probs = torch.softmax(x, dim=-1)
        return torch.multinomial(probs, 1, generator=gen).squeeze(-1)

    def generate(
        self,
        prompts: list[list[int]],
        max_new_tokens: int,
        sampling: dict | None = None,
        stop_token_ids: list[int] | None = None,
        seed: int = 0,
    ) -> dict:
        n = len(prompts)
        assert 0 < n <= self.global_batch_size
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        gb = self.global_batch_size
        stop = set(stop_token_ids or [])
        gen = torch.Generator().manual_seed(seed)

        dev_sp = self._to_sampling_params(sampling) if self.device_sampling else None
        t0 = time.time()
        if dev_sp is not None:
            cur = self.prefill_tokens(prompts, dev_sp)
        else:
            cur = self._sample(self.prefill(prompts), sampling, gen)
        ttft = time.time() - t0
        logger.info(f"Prefill of {n} prompts took {ttft:.2f}s")

        generated: list[list[int]] = [[] for _ in range(n)]
        active = [True] * n
        pos = torch.tensor([len(p) for p in prompts], dtype=torch.int64)
        for u in range(n):
            t = int(cur[u])
            generated[u].append(t)
            if t in stop or pos[u] >= self.models[0].args.max_seq_len:
                active[u] = False

        decode_times = []
        step_users = []
        breakdowns = []
        for _ in range(max_new_tokens - 1):
            if not any(active):
                break
            tokens = torch.zeros(gb, dtype=torch.int64)
            positions = torch.full((gb,), -1, dtype=torch.int64)
            for u in range(n):
                if active[u]:
                    tokens[u] = cur[u]
                    positions[u] = pos[u]
            t1 = time.time()
            idx = torch.tensor([u for u in range(n) if active[u]], dtype=torch.long)
            cur = torch.zeros(n, dtype=torch.int64)
            if dev_sp is not None:
                step_tokens = self.decode_step_tokens(tokens, positions, dev_sp)
                cur[idx] = step_tokens[:n][idx]  # inactive users' tokens are ignored
            else:
                step_logits = self.decode_step(tokens, positions)[:n]
                # sample only active users: inactive rows may be stale/NaN (SDPA skips position -1)
                if idx.numel() > 0:
                    cur[idx] = self._sample(step_logits[idx], sampling, gen)
            decode_times.append(time.time() - t1)
            breakdowns.append(dict(self.last_step_breakdown))
            step_users.append(sum(active))
            for u in range(n):
                if not active[u]:
                    continue
                pos[u] += 1
                t = int(cur[u])
                generated[u].append(t)
                if t in stop or pos[u] >= self.models[0].args.max_seq_len:
                    active[u] = False

        steady = decode_times[1:] if len(decode_times) > 1 else decode_times
        mean_t = sum(steady) / len(steady) if steady else float("nan")
        steady_users = step_users[1:] if len(step_users) > 1 else step_users
        return {
            "tokens": generated,
            "ttft_s": ttft,
            "decode_times_s": decode_times,
            "step_breakdown_s": breakdowns,
            "device_sampling": dev_sp is not None,
            "tok_s_per_user": (1.0 / mean_t) if steady else float("nan"),
            "throughput_tok_s": (sum(steady_users) / sum(steady)) if steady else float("nan"),
        }

    def release_traces(self):
        for r, m in enumerate(self.models):
            if self.trace_ids[r] is not None:
                ttnn.release_trace(m.mesh_device, self.trace_ids[r])
                self.trace_ids[r] = None
                self.trace_outputs[r] = None
