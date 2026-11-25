# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stable penalty state owned by the canonical generator's sampling traces."""

import torch

import ttnn


class PenaltyHistory:
    """Update canonical TTPenalties buffers without replacing their storage.

    Construct and warm before any capture. Reset histories are CPU tensors in
    fixed-slot order, padded with -1; selected rows replace previous history.
    The caller owns dirty-slot gating and decides when authoritative histories
    are available. Steady sampling calls only ``sample_logits`` inside a trace.
    """

    def __init__(self, generator):
        if getattr(generator, "traces", {}):
            raise RuntimeError("Allocate penalty history before capturing generator traces")
        self.generator = generator
        self.penalties = generator.sampling.tt_penalties
        p = self.penalties
        self.rows = p._total_batch
        if self.rows != 32 or p._sampling_dp != 1:
            raise ValueError("Qwen penalty history requires one 32-row TP sampler")
        self.logits = ttnn.clone(generator.logits)
        self.keep_rows = generator.model.io._upload(torch.ones(self.rows, 1, dtype=torch.int32), ttnn.int32)
        self.prompt_update = p._alloc_int_buffer(shard_dims=p._shard_dims_mask)
        self.counts_update = p._alloc_int_buffer(shard_dims=p._shard_dims_gathered)
        self._remap_tensors = [
            p.prompt_mask,
            p.output_counts_gathered,
            p.presence_penalties,
            p.frequency_penalties,
            p.repetition_penalties,
            p.inverse_repetition_penalties,
        ]
        self.indices = {}
        for tensor in self._remap_tensors:
            key = (tuple(tensor.shape), tensor.layout)
            if key not in self.indices:
                host = torch.arange(self.rows, dtype=torch.int32).reshape(self.rows, 1).expand(tuple(tensor.shape))
                self.indices[key] = generator.model.io._upload(host, ttnn.uint32, tensor.layout)
        p._prompt_tokens_host = torch.full((self.rows, 1), -1, dtype=torch.int64)

    def _history(self, tokens):
        if tokens is None:
            return torch.full((self.rows, 1), -1, dtype=torch.int64)
        tokens = torch.as_tensor(tokens, dtype=torch.int64)
        if tokens.ndim != 2 or not 1 <= tokens.shape[0] <= self.rows:
            raise ValueError("Penalty histories must be fixed-slot tensors with 1..32 rows")
        if bool(((tokens < -1) | (tokens >= self.penalties.vocab_size)).any()):
            raise ValueError("Penalty history contains a token outside the vocabulary or -1 padding")
        # Empty output histories still need a nonempty physical host width.
        result = torch.full((self.rows, max(1, tokens.shape[1])), -1, dtype=torch.int64)
        result[: tokens.shape[0], : tokens.shape[1]] = tokens
        return result

    def _refresh_output_views(self):
        p = self.penalties
        ttnn.slice(
            p.output_counts_gathered,
            p.slice_start,
            p.slice_end,
            output_tensor=p.output_counts,
            slice_dim=1,
            num_devices=p.num_devices,
            **p._op_kwargs,
        )
        ttnn.gt(p.output_counts, 0, output_tensor=p.output_mask, **p._op_kwargs)

    def reset(self, slots=None, prompt_tokens=None, output_tokens=None):
        """Replace selected histories; omitted histories clear those rows.

        Inputs are fixed-slot indexed, not compact rows corresponding to
        ``slots``. Unselected rows retain device counts and host prompt shadow.
        """
        slots = list(range(self.rows)) if slots is None else list(map(int, slots))
        if len(set(slots)) != len(slots) or any(not 0 <= slot < self.rows for slot in slots):
            raise ValueError("Penalty reset slots must be distinct indices in 0..31")
        if not slots:
            return
        p = self.penalties
        prompt, output = self._history(prompt_tokens), self._history(output_tokens)
        # Validate before mutating either host ownership or device state.
        shadow = p._prompt_tokens_host
        width = max(prompt.shape[1], shadow.shape[1])
        merged = torch.full((self.rows, width), -1, dtype=torch.int64)
        merged[:, : shadow.shape[1]] = shadow
        merged[slots] = -1
        merged[slots, : prompt.shape[1]] = prompt[slots]
        selected = torch.zeros((self.rows, 1), dtype=torch.int32)
        selected[slots] = 1
        prompt_mask = (p._token_counts_host(prompt) > 0).to(torch.int32) * selected
        output_counts = p._token_counts_host(output) * selected
        self.generator._write(1 - selected, self.keep_rows)
        p._copy_int_host_to_device(self.prompt_update, prompt_mask, p._shard_dims_mask)
        p._copy_int_host_to_device(self.counts_update, output_counts, p._shard_dims_gathered)
        for target, update in ((p.prompt_mask, self.prompt_update), (p.output_counts_gathered, self.counts_update)):
            ttnn.mul(target, self.keep_rows, output_tensor=target, **p._op_kwargs)
            ttnn.add(target, update, output_tensor=target, **p._op_kwargs)
        self._refresh_output_views()
        p._prompt_tokens_host = merged

    def remap_graph(self):
        """Apply already-uploaded row permutation once; no host state changes."""
        for target in self._remap_tensors:
            indices = self.indices[(tuple(target.shape), target.layout)]
            moved = ttnn.gather(target, 0, indices)
            ttnn.copy(moved, target)
        self._refresh_output_views()

    def remap(self, padded_permutation):
        """Upload and apply one full 32-row permutation, including host shadow."""
        permutation = torch.as_tensor(padded_permutation, dtype=torch.int32).flatten()
        if permutation.numel() != self.rows or sorted(permutation.tolist()) != list(range(self.rows)):
            raise ValueError("Penalty remap must be a complete permutation of rows 0..31")
        for (shape, _), target in self.indices.items():
            self.generator._write(permutation.reshape(self.rows, 1).expand(shape), target)
        self.remap_graph()
        self.penalties._prompt_tokens_host = self.penalties._prompt_tokens_host[permutation.to(torch.int64)].clone()

    def sample_logits(self):
        """Stage raw logits and exact row-count mask inside the penalties trace."""
        ttnn.copy(self.generator.logits, self.logits)
        mask = ttnn.reshape(self.generator.sample_active_rows, (self.rows, 1))
        mask = ttnn.typecast(mask, ttnn.int32)
        mask = ttnn.to_layout(mask, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy(mask, self.penalties.decode_src)
        return self.logits

    def warm(self):
        """Compile reset/remap/staging before capture; destroys dummy history.

        The generator must separately warm its canonical counted sampling graph
        with nonneutral penalties, then clear those synthetic output counts.
        """
        if self.generator.traces:
            raise RuntimeError("Penalty history warmup must precede every capture")
        self.reset()
        self.remap(torch.arange(self.rows, dtype=torch.int32))
        self.sample_logits()
