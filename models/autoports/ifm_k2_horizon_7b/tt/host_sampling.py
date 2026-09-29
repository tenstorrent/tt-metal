"""Model-owned CPU sampler with unchanged per-request random streams.

Audited against vLLM 5c5a3cdacb7ad84f733c679845967e1f4172fccd, PyTorch
2.11.0+cpu, and NumPy 2.3.5. ``forward_native`` mirrors the pinned upstream
method except for its final random-sampling call. Finite FP32 full-vocabulary
filtering sorts values with NumPy but retains upstream softmax/cumsum exactly.
A cutoff splitting equal values falls back to upstream's index tie-breaking.
Revalidate the host tests when these dependencies change.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, apply_top_k_top_p, random_sample
from vllm.v1.sample.sampler import Sampler


class _SeededCPUParallelTopKTopPSampler(TopKTopPSampler):
    def __init__(self, logprobs_mode="raw_logprobs", *, max_workers=8):
        super().__init__(logprobs_mode)
        if max_workers < 1:
            raise ValueError("Host sampling requires at least one worker")
        self.max_workers = max_workers
        self._pool = None
        self._pool_lock = threading.Lock()
        self._closed = False
        self._stats = dict(
            numpy_sort_calls=0,
            numpy_sort_rows=0,
            stock_sort_calls=0,
            tie_fallback_rows=0,
            parallel_rng_calls=0,
            parallel_rng_rows=0,
            stock_rng_calls=0,
        )
        self.apply_top_k_top_p = self._filter_logits
        # TT uses the upstream native implementation on CPU logits. Keep that
        # choice explicit even in a host-only test with another platform class.
        self.forward = self.forward_native

    def _count(self, **counts):
        with self._pool_lock:
            for key, value in counts.items():
                self._stats[key] += value

    def stats_snapshot(self):
        with self._pool_lock:
            return dict(
                self._stats,
                max_workers=self.max_workers,
                pool_created=self._pool is not None,
                closed=self._closed,
                numpy_version=np.__version__,
            )

    @staticmethod
    def _sort_row(row):
        row.sort(kind="quicksort")

    def _filter_logits(self, logits, k, p):
        supported = (
            logits.device.type == "cpu"
            and logits.dtype == torch.float32
            and logits.ndim == 2
            and logits.shape[0] > 0
            and logits.shape[1] > 0
            and logits.is_contiguous()
            and not logits.requires_grad
            and not logits.is_neg()
            and not logits.is_conj()
            and p is not None
            and p.device.type == "cpu"
            and p.ndim == 1
            and p.numel() == logits.shape[0]
            and bool(torch.isfinite(p).all())
            and (k is None or bool((k == logits.shape[1]).all()))
            and bool(torch.isfinite(logits).all())
        )
        if not supported:
            self._count(stock_sort_calls=1)
            return apply_top_k_top_p(logits, k, p)

        # Own the NumPy buffer: sorting must never mutate caller logits, which
        # upstream may also retain for raw-logit/logprob reporting.
        data = logits.numpy().copy()
        list(self._executor().map(self._sort_row, data))
        values = torch.from_numpy(data)
        probs = values.softmax(dim=-1)
        torch.cumsum(probs, dim=-1, out=probs)
        # The upstream cumulative probabilities are monotone. Their right
        # insertion point counts the same <= cutoff entries without an entire
        # vocabulary-sized Boolean mask and int64 reduction. Preserve the
        # upstream rule that the final entry is always retained.
        first_keep = torch.searchsorted(probs, (1 - p).unsqueeze(1), right=True).squeeze(1)
        first_keep.clamp_max_(logits.shape[1] - 1)
        rows = torch.arange(logits.shape[0])
        threshold = values[rows, first_keep]
        tie = (first_keep > 0) & (threshold == values[rows, (first_keep - 1).clamp_min(0)])
        # Without a split tie, retaining values >= the exact upstream cutoff
        # reconstructs the same mask without allocating a vocabulary index map.
        result = logits.masked_fill(logits < threshold.unsqueeze(1), -float("inf"))
        tie_count = int(tie.sum())
        if tie_count:
            result[tie] = apply_top_k_top_p(logits[tie].clone(), None, p[tie])
        self._count(numpy_sort_calls=1, numpy_sort_rows=logits.shape[0], tie_fallback_rows=tie_count)
        return result

    def _executor(self):
        with self._pool_lock:
            if self._closed:
                raise RuntimeError("K2 host sampler is closed")
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="k2-host-sampling")
            return self._pool

    @staticmethod
    def _fill_row(item):
        row, generator = item
        # Inference mode is thread-local. The serving caller may have created
        # q as an inference tensor; worker threads must allow its in-place fill.
        with torch.inference_mode():
            row.exponential_(generator=generator)

    def _random_sample(self, probs, generators):
        rows = probs.shape[0]
        supported = (
            probs.device.type == "cpu"
            and probs.dtype == torch.float32
            and probs.ndim == 2
            and rows > 1
            and probs.shape[1] > 0
            and set(generators) == set(range(rows))
            and all(type(index) is int for index in generators)
            and all(isinstance(generator, torch.Generator) for generator in generators.values())
            and len({id(generator) for generator in generators.values()}) == rows
            and all(generator.device.type == "cpu" for generator in generators.values())
        )
        if not supported:
            self._count(stock_rng_calls=1)
            return random_sample(probs, generators)
        q = torch.empty_like(probs)
        # Every job owns one output row and one independent generator. Finish
        # the same draws as upstream before its unchanged division and argmax.
        jobs = ((q[index], generator) for index, generator in generators.items())
        list(self._executor().map(self._fill_row, jobs))
        self._count(parallel_rng_calls=1, parallel_rng_rows=rows)
        return probs.div_(q).argmax(dim=-1).view(-1)

    def forward_native(self, logits, generators, k, p):
        logits = self.apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return self._random_sample(probs, generators), logits_to_return

    def close(self):
        with self._pool_lock:
            self._closed = True
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)


class K2HostSampler(Sampler):
    """Optional K2-only callable matching the upstream Sampler interface."""

    def __init__(self, logprobs_mode="raw_logprobs", *, max_workers=8):
        super().__init__(logprobs_mode)
        self.topk_topp_sampler = _SeededCPUParallelTopKTopPSampler(logprobs_mode, max_workers=max_workers)

    def close(self):
        self.topk_topp_sampler.close()

    def stats_snapshot(self):
        return self.topk_topp_sampler.stats_snapshot()
