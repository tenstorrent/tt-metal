# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Chunked-stream writer + reader for long-decode Group A debug traces.

Replaces the per-step ``step_{s}/`` directory layout for long traces. Each
``(kind, layer)`` pair gets its own immutable safetensors chunks of up to
``chunk_rows`` rows. A 128K-decode Group A trace produces ~1984 files instead
of ~16M; chunks are friendly to mmap + ``safe_open(...).get_slice(...)``.

Scope: Group A (decoder I/O + KV cache) + top-K capture (always-on when
Group A is enabled; see §6.5.11 of ``docs/long_context_mla.md``). Group B
internals + ``attn_internals`` stay on the per-step path.

Layout::

    {run_dir}/
        index.json                          # authoritative chunk manifest
        metadata.json                       # run metadata (prompt, generated tokens, …)
        kv_cache/
            layer_{i}/
                rows_{start:08d}_{end:08d}.safetensors    # key: kv_post_transform_layer_{i}
        decoder_io/
            decoder_input_layer_0/
                rows_*.safetensors                        # key: decoder_input_layer_0
            decoder_output_layer_{i}/
                rows_*.safetensors                        # key: decoder_output_layer_{i}
        topk/                               # only if any top-K rows were captured
            indices/rows_*.safetensors      # key: topk_indices,   int32 (N, K)
            logprobs/rows_*.safetensors     # key: topk_logprobs,  bf16  (N, K)
            logsumexp/rows_*.safetensors    # key: topk_logsumexp, fp32  (N, 1)
        kv_cache_layer_{i}.safetensors      # optional merged single-file form,
                                            # produced by scripts/merge_kv_cache.py;
                                            # reader fast-path uses these when
                                            # index.json["merged_streams"] declares them

``decoder_input_layer_{i}`` for ``i >= 1`` is **not** materialized; the alias
rule in ``index.json`` says ``decoder_input_layer_{i}(row) :=
decoder_output_layer_{i-1}(row)``. Readers apply this transparently.

The ``rows_{start}_{end}.safetensors`` filenames are an **implementation
detail** — the reader never parses them. ``index.json`` carries each
chunk's ``{row_start, row_end, path}`` so the reader resolves paths
verbatim. Cross-language consumers (JAX / Rust / web) only need to read
``index.json`` and ``safetensors.load_file`` each ``chunks[*].path``.

Filenames use ``[start, end)`` row indexing (Python slice convention). The
chunk for any given row ``r`` is ``r // chunk_rows`` (uniform sizing within a
stream); the final chunk may be partial.

Two row axes
------------

Streams come in two indexing flavors:

* **Group A streams** (``decoder_input_layer_*``, ``decoder_output_layer_*``,
  ``kv_post_transform_layer_*``) — row index = **token position in the
  full sequence**. Sequence length = ``T_prompt + (n_decode_steps - 1)``;
  prefill writes ``T_prompt`` rows, each decode forward writes 1 row. The
  step → row-range mapping lives in ``index.json["step_rows"]`` and is
  exposed by ``reader.step_rows(step_idx)``.

* **Top-K streams** (``topk_indices``, ``topk_logprobs``, ``topk_logsumexp``)
  — row index = **sampling event** (one row per forward pass; exactly one
  sampling event per forward under batch=1 greedy). The step → row-range
  mapping lives in ``index.json["step_rows_topk"]`` and is exposed by
  ``reader.step_rows_topk(step_idx)``. Top-K row ``r`` is "the prediction
  made by forward pass ``r``" — for batch=1 greedy that means the model's
  prediction of ``completion_token_ids[r]`` given
  ``prompt + completion_token_ids[:r]``.

How to read a trace
-------------------

The reader + ``index.json`` are the canonical entry point. ::

    import json
    from analysis.debug_trace_chunked_io import ChunkedTraceReader

    r = ChunkedTraceReader("/path/to/trace_dir")
    meta = json.load(open("/path/to/trace_dir/metadata.json"))

    # ---- Group A (per-token-position) ----
    out_l30 = r.get_decoder_output(layer=30)             # (T, hidden_dim)
    in_l1   = r.get_decoder_input(layer=1)               # alias rule applied
    kv_l30  = r.get_kv(layer=30)                         # (T, 576)
    one_row = r.get_decoder_output(layer=30, rows=8000)  # single row, no full-chunk load
    step42  = r.get_step(step_idx=42)                    # legacy step-shaped dict

    # ---- Top-K (per-sampling-event) ----
    # Always present when capture_group_a=True. K from `r.topk_K`.
    K        = r.topk_K                                  # 100 by default
    indices  = r.get_topk_indices()                      # (N_steps, K) int32
    logprobs = r.get_topk_logprobs()                     # (N_steps, K) bf16
    log_z    = r.get_topk_logsumexp()                    # (N_steps, 1) fp32

    # Teacher-forcing test against the consumer's own implementation:
    gen_ids = meta.get("completion_token_ids") or meta["generated_token_ids"]
    for i in range(len(gen_ids)):
        # Reference top-1 is by construction the sampled token (greedy):
        assert int(indices[i, 0]) == gen_ids[i]
        # Consumer's own prediction at the same prefix:
        my_top1 = my_model.predict_top1(prompt, gen_ids[:i])
        topk_acc = {
            "top-1":   my_top1 in indices[i, :1].tolist(),
            "top-5":   my_top1 in indices[i, :5].tolist(),
            "top-100": my_top1 in indices[i, :100].tolist(),
        }

    # Recover the raw logit for a captured top-K token:
    #     logit_k = logprob_k + log_z
    raw_logit_top1 = logprobs[i, 0].float() + log_z[i, 0]
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from models.demos.deepseek_v3_b1.model_dimensions import LogicalModelDimensions as D

# Top-K capture (part of Group A; always-on when Group A is enabled).
# K=100 is a module-level constant; edit here for a one-off K change. See
# docs/long_context_mla.md §6.5.11 for the rationale.
TOPK_K = 100

_TOPK_KEYS = ("topk_indices", "topk_logprobs", "topk_logsumexp")

_MODEL_TRACE_IDS = {
    "DeepSeek_R1_0528": "ds_r1_0528",
}


# --------------------------------------------------------------------------- helpers


def _chunk_filename(row_start: int, row_end: int) -> str:
    return f"rows_{row_start:08d}_{row_end:08d}.safetensors"


def _atomic_save(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Write via temp file + rename so partial writes never appear as full chunks."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(tmp))
    tmp.rename(path)


def _dir_for_key(run_dir: Path, key: str) -> Path:
    if key == "decoder_input_layer_0":
        return run_dir / "decoder_io" / "decoder_input_layer_0"
    if key.startswith("decoder_output_layer_"):
        return run_dir / "decoder_io" / key
    if key.startswith("kv_post_transform_layer_"):
        layer = key[len("kv_post_transform_layer_") :]
        return run_dir / "kv_cache" / f"layer_{layer}"
    # Top-K streams (one row per forward / sampling event).
    if key == "topk_indices":
        return run_dir / "topk" / "indices"
    if key == "topk_logprobs":
        return run_dir / "topk" / "logprobs"
    if key == "topk_logsumexp":
        return run_dir / "topk" / "logsumexp"
    raise ValueError(f"unsupported key for chunked layout: {key!r}")


# --------------------------------------------------------------------------- writer


class ChunkedGroupAWriter:
    """Streaming chunked writer for Group A.

    Usage::

        w = ChunkedGroupAWriter(run_dir, n_layers=61, chunk_rows=4096)
        for step_idx, buffers in enumerate(all_steps):
            w.add_step(step_idx, buffers)
        index = w.finalize()
        # caller writes index.json + metadata.json

    The writer accepts the same per-step buffer dicts that
    ``DebugTracer._save_step`` consumes today, and silently ignores keys
    outside Group A (so it's safe to pass full buffers including Group B /
    attn_internals — they're handled by the legacy per-step writer in
    parallel if needed).
    """

    def __init__(
        self,
        run_dir: Path | str,
        n_layers: int,
        chunk_rows: int = 4096,
        async_io: bool = True,
    ) -> None:
        if chunk_rows <= 0:
            raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
        self.run_dir = Path(run_dir)
        self.n_layers = n_layers
        self.chunk_rows = chunk_rows
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Per-key state: accumulated unflushed rows (concat-on-flush) + total flushed
        self._unflushed: dict[str, list[torch.Tensor]] = {}
        self._unflushed_rows: dict[str, int] = {}
        self._flushed_rows: dict[str, int] = {}
        # Per-key chunk index: list of (row_start, row_end, path-relative-to-run_dir)
        self._chunks: dict[str, list[tuple[int, int, str]]] = {}
        # Per-key tail-of-shape (everything past dim 0). Set on first append.
        self._shape_tail: dict[str, tuple[int, ...]] = {}
        # Per-key dtype. Set on first append.
        self._dtype: dict[str, str] = {}

        # Per-step row range, keyed by step_idx (for the extractor to rebuild
        # legacy step_{s}/ dirs). Records the global row range that step
        # contributed for the canonical reference stream (decoder_output_layer_0).
        # Group A streams (decoder_io, kv_cache) all share this row indexing
        # — each step contributes T_new rows (T_prompt at prefill, 1 per decode).
        self._step_rows: dict[int, tuple[int, int]] = {}
        # Top-K streams use their own row indexing: exactly 1 row per step
        # (one sampling event per forward in batch=1 greedy). Tracked
        # separately so the consumer can distinguish "per-token row" from
        # "per-sampling-event row".
        self._step_rows_topk: dict[int, tuple[int, int]] = {}

        # Async I/O (default): _flush submits the safetensors save to a
        # ThreadPoolExecutor instead of running inline. Lets the GPU keep
        # decoding while CPU threads write chunks to disk. finalize()
        # drains pending futures before returning. Pass async_io=False to
        # force inline writes (useful for tight bytewise determinism in
        # tests, where you want flush order to match append order). See
        # docs/long_context_mla.md §6.5.12.
        self._io_pool: concurrent.futures.ThreadPoolExecutor | None = None
        self._pending: list[concurrent.futures.Future] = []
        if async_io:
            self._io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="trace-writer")

    # ------------------------------------------------------------------ public

    def add_step(self, step_idx: int, buffers: dict[str, torch.Tensor]) -> int:
        """Append one step's Group A rows. Returns the number of rows added.

        Buffers must already be in CPU + save_dtype (the tracer's ``_finalize_buffers``
        path produces these). Keys outside Group A are silently skipped.

        Top-K keys (``topk_indices``, ``topk_logprobs``, ``topk_logsumexp``) are
        appended on a parallel row axis: exactly 1 row per step, tracked in
        ``self._step_rows_topk``.
        """
        added: int | None = None
        canonical_key = "decoder_output_layer_0"
        canonical_start: int | None = None

        # Sort keys deterministically so chunk write order is stable across runs.
        for key in sorted(buffers.keys()):
            if not self._is_group_a_key(key):
                continue
            tensor = buffers[key]
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            n_rows = tensor.shape[0]
            if added is None:
                added = n_rows
            elif n_rows != added:
                raise ValueError(
                    f"row-count mismatch in step {step_idx}: {key} has {n_rows} rows, " f"expected {added}"
                )
            if key == canonical_key:
                canonical_start = self._flushed_rows.get(key, 0) + self._unflushed_rows.get(key, 0)
            self._append(key, tensor)

        if canonical_start is not None and added is not None:
            self._step_rows[step_idx] = (canonical_start, canonical_start + added)

        # Top-K: separate row axis (1 row per step). Each topk_* tensor must be
        # exactly 1 row (the last position's prediction).
        topk_start: int | None = None
        topk_added: int | None = None
        for key in _TOPK_KEYS:
            if key not in buffers:
                continue
            tensor = buffers[key]
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            n_rows = tensor.shape[0]
            if n_rows != 1:
                raise ValueError(
                    f"top-K stream {key} must have exactly 1 row per step, got {n_rows} " f"in step {step_idx}"
                )
            if topk_start is None:
                topk_start = self._flushed_rows.get(key, 0) + self._unflushed_rows.get(key, 0)
                topk_added = 1
            self._append(key, tensor)

        if topk_start is not None and topk_added is not None:
            self._step_rows_topk[step_idx] = (topk_start, topk_start + topk_added)

        return added or 0

    def finalize(self) -> dict:
        """Flush partial chunks and return the index payload (caller writes it).

        After ``finalize`` the writer is no longer usable. When async_io is
        enabled, drains the executor pool before returning so that the
        returned index matches the on-disk state.
        """
        for key in sorted(self._unflushed.keys()):
            if self._unflushed_rows[key] == 0:
                continue
            self._flush(key, partial=True)
        if self._io_pool is not None:
            # Block until every pending save lands on disk. .result()
            # re-raises any exception so a write failure surfaces here
            # rather than getting silently swallowed.
            for f in self._pending:
                f.result()
            self._io_pool.shutdown(wait=True)
            self._pending = []
            self._io_pool = None
        return self._build_index()

    # ------------------------------------------------------------------ internal

    def _is_group_a_key(self, key: str) -> bool:
        return (
            key == "decoder_input_layer_0"
            or key.startswith("decoder_output_layer_")
            or key.startswith("kv_post_transform_layer_")
        )

    def _append(self, key: str, tensor: torch.Tensor) -> None:
        # Initialize per-key state on first sighting.
        if key not in self._unflushed:
            self._unflushed[key] = []
            self._unflushed_rows[key] = 0
            self._flushed_rows[key] = 0
            self._chunks[key] = []
            self._shape_tail[key] = tuple(tensor.shape[1:])
            self._dtype[key] = str(tensor.dtype).removeprefix("torch.")
        else:
            if tuple(tensor.shape[1:]) != self._shape_tail[key]:
                raise ValueError(
                    f"shape-tail mismatch for {key}: appending {tuple(tensor.shape[1:])} "
                    f"but stream expects {self._shape_tail[key]}"
                )

        self._unflushed[key].append(tensor)
        self._unflushed_rows[key] += tensor.shape[0]

        while self._unflushed_rows[key] >= self.chunk_rows:
            self._flush(key, partial=False)

    def _flush(self, key: str, partial: bool) -> None:
        """Write the next chunk_rows rows (or remainder if partial) for ``key``."""
        target = self._unflushed_rows[key] if partial else self.chunk_rows
        concat = torch.cat(self._unflushed[key], dim=0)
        chunk = concat[:target].contiguous()
        remainder = concat[target:]

        row_start = self._flushed_rows[key]
        row_end = row_start + target
        out_dir = _dir_for_key(self.run_dir, key)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / _chunk_filename(row_start, row_end)
        # Submit save: async via the pool when configured, else inline.
        # The chunk tensor is kept alive by the executor's reference (via
        # the args tuple); refcount drops to 0 when the save completes and
        # the storage is freed. Meanwhile self._unflushed[key] points at
        # remainder + future appends, which live in independent allocations.
        self._submit_save(key, path, {key: chunk}, row_start, row_end)

        rel = path.relative_to(self.run_dir).as_posix()
        self._chunks[key].append((row_start, row_end, rel))
        self._flushed_rows[key] = row_end

        # Stash remainder back into the unflushed list.
        if remainder.shape[0] > 0:
            self._unflushed[key] = [remainder]
            self._unflushed_rows[key] = remainder.shape[0]
        else:
            self._unflushed[key] = []
            self._unflushed_rows[key] = 0

    def _submit_save(
        self,
        key: str,
        path: Path,
        tensors: dict[str, torch.Tensor],
        row_start: int,
        row_end: int,
    ) -> None:
        """Save a chunk; async via the pool if configured, else inline."""
        if self._io_pool is None:
            _atomic_save(path, tensors)
            return
        chunk = tensors[key]
        size_bytes = chunk.element_size() * chunk.numel()
        t_submit = time.monotonic()
        fut = self._io_pool.submit(_atomic_save, path, tensors)
        fut.add_done_callback(self._make_chunk_done_callback(key, row_start, row_end, path, size_bytes, t_submit))
        self._pending.append(fut)
        # Reap completed futures so the list doesn't grow unbounded. Errors
        # are surfaced by the done-callback; finalize() also re-raises via
        # f.result() to guarantee no silent failure.
        self._pending = [f for f in self._pending if not f.done()]

    def _make_chunk_done_callback(
        self,
        key: str,
        row_start: int,
        row_end: int,
        path: Path,
        size_bytes: int,
        t_submit: float,
    ):
        """Build a done-callback that prints a progress line per chunk write."""

        def cb(future: concurrent.futures.Future) -> None:
            try:
                future.result()
            except Exception as e:
                print(
                    f"[trace] FAILED flush of {key} rows [{row_start}, {row_end}) " f"-> {path.name}: {e}",
                    flush=True,
                )
                return
            wall = time.monotonic() - t_submit
            size_mb = size_bytes / (1024 * 1024)
            try:
                rel = path.relative_to(self.run_dir).as_posix()
            except ValueError:
                rel = str(path)
            print(
                f"[trace] flushed {key} rows [{row_start}, {row_end}) -> {rel} " f"({size_mb:.1f} MB, {wall:.2f}s)",
                flush=True,
            )

        return cb

    def _build_index(self) -> dict:
        tensor_streams: dict[str, dict] = {}
        for key in sorted(self._chunks.keys()):
            chunks_payload = [{"row_start": s, "row_end": e, "path": p} for s, e, p in self._chunks[key]]
            tensor_streams[key] = {
                "row_count": self._flushed_rows[key],
                "dtype": self._dtype[key],
                "shape_tail": list(self._shape_tail[key]),
                "chunks": chunks_payload,
            }
        index = {
            "format_version": 1,
            "layout": "chunked_group_a_v1",
            "chunk_rows": self.chunk_rows,
            "n_layers": self.n_layers,
            "decoder_input_alias_rule": (
                "decoder_input_layer_{i}(row) := decoder_output_layer_{i-1}(row) "
                "for i in [1, n_layers); decoder_input_layer_0 has its own stream"
            ),
            "tensor_streams": tensor_streams,
            "step_rows": {str(k): list(v) for k, v in sorted(self._step_rows.items())},
        }
        # Top-K is part of Group A (always-on); only emit fields if any top-K
        # rows were written (preserves backwards-compat for old traces).
        if self._step_rows_topk:
            index["topk_K"] = TOPK_K
            index["step_rows_topk"] = {str(k): list(v) for k, v in sorted(self._step_rows_topk.items())}
        return index


# --------------------------------------------------------------------------- reader


class ChunkedTraceReader:
    """Random-access reader for the chunked Group A layout.

    Uses ``safe_open(...).get_slice(...)`` so per-row reads don't materialize
    whole chunks. Applies the decoder-input alias rule transparently:
    ``get_decoder_input(layer=i)`` for ``i >= 1`` returns
    ``get_decoder_output(layer=i-1)``.
    """

    def __init__(self, trace_dir: Path | str) -> None:
        self.trace_dir = Path(trace_dir)
        index_path = self.trace_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"chunked-layout index.json not found under {self.trace_dir}; " "is this a chunked trace?"
            )
        with open(index_path) as f:
            self.index = json.load(f)
        if self.index.get("layout") != "chunked_group_a_v1":
            raise ValueError(f"unknown layout {self.index.get('layout')!r} in {index_path}")
        # Optional fast-path: streams that have been merged into a single file
        # by `scripts/merge_kv_cache.py` (or similar). The dict is
        # ``{stream_key: Path_to_merged_file}``; _read prefers it over the
        # chunked path when present.
        self._merged_paths: dict[str, Path] = {}
        for key, info in self.index.get("merged_streams", {}).items():
            self._merged_paths[key] = self.trace_dir / info["path"]

    # ------------------------------------------------------------------ properties

    @property
    def n_layers(self) -> int:
        return int(self.index["n_layers"])

    @property
    def chunk_rows(self) -> int:
        return int(self.index["chunk_rows"])

    def streams(self) -> list[str]:
        return list(self.index["tensor_streams"].keys())

    def row_count(self, key: str) -> int:
        return int(self.index["tensor_streams"][key]["row_count"])

    def step_rows(self, step_idx: int) -> tuple[int, int]:
        """Return [row_start, row_end) for ``step_idx`` (canonical reference stream)."""
        s = self.index.get("step_rows", {}).get(str(step_idx))
        if s is None:
            raise KeyError(f"step_idx {step_idx} not present in index.step_rows")
        return (int(s[0]), int(s[1]))

    @property
    def topk_K(self) -> int | None:
        """Top-K value used at capture time (None if no top-K data present)."""
        k = self.index.get("topk_K")
        return None if k is None else int(k)

    def step_rows_topk(self, step_idx: int) -> tuple[int, int]:
        """Return [row_start, row_end) for ``step_idx`` in the top-K row axis (1 row/step)."""
        s = self.index.get("step_rows_topk", {}).get(str(step_idx))
        if s is None:
            raise KeyError(f"step_idx {step_idx} not present in index.step_rows_topk")
        return (int(s[0]), int(s[1]))

    # ------------------------------------------------------------------ getters

    def get_decoder_input(self, layer: int, rows: slice | int | None = None) -> torch.Tensor:
        if layer == 0:
            return self._read("decoder_input_layer_0", rows)
        return self._read(f"decoder_output_layer_{layer - 1}", rows)

    def get_decoder_output(self, layer: int, rows: slice | int | None = None) -> torch.Tensor:
        return self._read(f"decoder_output_layer_{layer}", rows)

    def get_kv(self, layer: int, rows: slice | int | None = None) -> torch.Tensor:
        return self._read(f"kv_post_transform_layer_{layer}", rows)

    def get_topk_indices(self, rows: slice | int | None = None) -> torch.Tensor:
        """Top-K token indices per step. Shape: (n_rows, topk_K), int32."""
        return self._read("topk_indices", rows)

    def get_topk_logprobs(self, rows: slice | int | None = None) -> torch.Tensor:
        """Top-K log-probabilities per step. Shape: (n_rows, topk_K)."""
        return self._read("topk_logprobs", rows)

    def get_topk_logsumexp(self, rows: slice | int | None = None) -> torch.Tensor:
        """Per-step logsumexp Z (log-normalizer). Shape: (n_rows,) or (n_rows, 1)."""
        return self._read("topk_logsumexp", rows)

    def get_step(
        self,
        step_idx: int,
        layers: Sequence[int] | None = None,
        kinds: Sequence[str] = ("decoder_input", "decoder_output", "kv"),
    ) -> dict[str, torch.Tensor]:
        """Return a dict mirroring the per-step buffer shape for a single step.

        Keys are the same as the legacy step-dir kinds:
            decoder_input_layer_{i}, decoder_output_layer_{i}, kv_post_transform_layer_{i}.
        ``decoder_input_layer_{i}`` is resolved via the alias rule for ``i >= 1``.
        """
        row_start, row_end = self.step_rows(step_idx)
        rows = slice(row_start, row_end)
        layers = list(range(self.n_layers)) if layers is None else list(layers)
        out: dict[str, torch.Tensor] = {}
        for i in layers:
            if "decoder_input" in kinds:
                out[f"decoder_input_layer_{i}"] = self.get_decoder_input(i, rows)
            if "decoder_output" in kinds:
                out[f"decoder_output_layer_{i}"] = self.get_decoder_output(i, rows)
            if "kv" in kinds:
                out[f"kv_post_transform_layer_{i}"] = self.get_kv(i, rows)
        return out

    # ------------------------------------------------------------------ internal

    def _read(self, key: str, rows: slice | int | None) -> torch.Tensor:
        stream = self.index["tensor_streams"].get(key)
        if stream is None:
            raise KeyError(f"tensor stream {key!r} not found in index")
        n = int(stream["row_count"])
        if rows is None:
            start, stop = 0, n
        elif isinstance(rows, int):
            start, stop = rows, rows + 1
        else:
            start = 0 if rows.start is None else rows.start
            stop = n if rows.stop is None else rows.stop
        if start < 0 or stop > n or start > stop:
            raise IndexError(f"row range [{start}, {stop}) out of bounds for {key} (row_count={n})")
        if start == stop:
            return torch.empty((0, *stream["shape_tail"]), dtype=_dtype_from_str(stream["dtype"]))

        # Fast path: merged single-file artifact (e.g. produced by
        # scripts/merge_kv_cache.py). One file open, one mmap'd slice.
        merged_path = self._merged_paths.get(key)
        if merged_path is not None and merged_path.exists():
            with safe_open(str(merged_path), framework="pt", device="cpu") as f:
                return f.get_slice(key)[start:stop]

        # Chunked path: find chunks that overlap [start, stop) and slice each.
        parts: list[torch.Tensor] = []
        for chunk in stream["chunks"]:
            cs, ce = int(chunk["row_start"]), int(chunk["row_end"])
            if ce <= start or cs >= stop:
                continue
            chunk_path = self.trace_dir / chunk["path"]
            with safe_open(str(chunk_path), framework="pt", device="cpu") as f:
                ts = f.get_slice(key)
                local_start = max(0, start - cs)
                local_stop = min(ce - cs, stop - cs)
                parts.append(ts[local_start:local_stop])
        if not parts:
            return torch.empty((0, *stream["shape_tail"]), dtype=_dtype_from_str(stream["dtype"]))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)


def _dtype_from_str(s: str) -> torch.dtype:
    return getattr(torch, s)


# --------------------------------------------------------------------------- host-IO adapter


def model_trace_id(model_id: str) -> str:
    """Return the trace-registry slug for a public model id."""
    try:
        return _MODEL_TRACE_IDS[model_id]
    except KeyError as e:
        raise ValueError(f"unknown trace model_id {model_id!r}") from e


def _load_metadata(trace_dir: Path) -> dict:
    metadata_path = trace_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"chunked trace metadata.json not found under {trace_dir}")
    with open(metadata_path) as f:
        return json.load(f)


def _rows_for_decode_steps(
    *,
    trace_dir: Path,
    reader: ChunkedTraceReader,
    num_decode_steps: int | None,
) -> slice | None:
    if num_decode_steps is None:
        return None
    if num_decode_steps <= 0:
        raise ValueError(f"num_decode_steps must be > 0 when provided, got {num_decode_steps}")

    metadata = _load_metadata(trace_dir)
    try:
        n_prompt_tokens = int(metadata["n_prompt_tokens"])
    except KeyError as e:
        raise ValueError(f"{trace_dir / 'metadata.json'}: missing required key 'n_prompt_tokens'") from e

    row_end = n_prompt_tokens + num_decode_steps - 1
    max_rows = reader.row_count("decoder_output_layer_0")
    if row_end > max_rows:
        raise ValueError(
            f"num_decode_steps={num_decode_steps} resolves to rows [0, {row_end}), "
            f"but trace only has {max_rows} Group A rows"
        )
    return slice(0, row_end)


def _validate_reference_trace(trace: dict[str, torch.Tensor], *, source: Path) -> dict[str, torch.Tensor]:
    if not isinstance(trace, dict):
        raise ValueError(f"{source}: expected dict, got {type(trace).__name__}")
    if not set(trace.keys()) >= {"input", "output"}:
        raise ValueError(f"{source}: missing required keys 'input'/'output'; got {sorted(trace.keys())}")
    inp, out = trace["input"], trace["output"]
    if not (isinstance(inp, torch.Tensor) and isinstance(out, torch.Tensor)):
        raise ValueError(f"{source}: 'input' and 'output' must be torch.Tensor")
    if inp.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError(f"{source}: expected bfloat16 tensors, got input={inp.dtype}, output={out.dtype}")
    if inp.ndim != 2 or out.ndim != 2:
        raise ValueError(
            f"{source}: expected 2D (seq_len, HIDDEN_SIZE) tensors, got input.shape={tuple(inp.shape)}, "
            f"output.shape={tuple(out.shape)}"
        )
    if inp.shape[-1] != D.HIDDEN_SIZE or out.shape[-1] != D.HIDDEN_SIZE:
        raise ValueError(
            f"{source}: last dim must equal D.HIDDEN_SIZE ({D.HIDDEN_SIZE}), got "
            f"input.shape={tuple(inp.shape)}, output.shape={tuple(out.shape)}"
        )
    if inp.shape[0] != out.shape[0]:
        raise ValueError(f"{source}: input and output seq_len mismatch: input={inp.shape[0]}, output={out.shape[0]}")
    return trace


def _trace_kv_to_tt_device_layout(kv: torch.Tensor) -> torch.Tensor:
    """Convert trace/eager k_pe split-halves order to TT-device interleaved order."""
    if kv.shape[-1] != D.KV_A_DIM:
        raise ValueError(f"expected KV last dim {D.KV_A_DIM}, got {kv.shape[-1]}")
    kpe_start = D.KV_B_LORA_RANK
    kpe_dim = D.KV_A_DIM - D.KV_B_LORA_RANK
    if kpe_dim % 2 != 0:
        raise ValueError(f"k_pe dim must be even for interleave conversion, got {kpe_dim}")

    out = kv.clone()
    kpe = kv[..., kpe_start:]
    half = kpe_dim // 2
    out[..., kpe_start::2] = kpe[..., :half]
    out[..., kpe_start + 1 :: 2] = kpe[..., half:]
    return out


def _validate_reference_kv(kv: torch.Tensor, *, source: Path) -> torch.Tensor:
    if not isinstance(kv, torch.Tensor):
        raise ValueError(f"{source}: expected KV torch.Tensor, got {type(kv).__name__}")
    if kv.dtype != torch.bfloat16:
        raise ValueError(f"{source}: expected KV bfloat16 tensor, got {kv.dtype}")
    if kv.ndim != 3:
        raise ValueError(f"{source}: expected KV shape (1, seq_len, {D.KV_A_DIM}), got {tuple(kv.shape)}")
    if kv.shape[0] != 1 or kv.shape[-1] != D.KV_A_DIM:
        raise ValueError(f"{source}: expected KV shape (1, seq_len, {D.KV_A_DIM}), got {tuple(kv.shape)}")
    return kv


def load_reference_trace(
    trace_root: Path | str,
    *,
    model_id: str,
    prompt_id: str,
    layer: int,
    num_decode_steps: int | None = None,
) -> dict[str, torch.Tensor]:
    """Load one chunked reference trace as the host-IO harness input/output contract."""
    trace_dir = Path(trace_root) / model_trace_id(model_id) / prompt_id
    if not trace_dir.is_dir():
        raise FileNotFoundError(
            f"chunked trace directory not found: {trace_dir} "
            f"(from trace_root={trace_root!r}, model_id={model_id!r}, prompt_id={prompt_id!r})"
        )

    reader = ChunkedTraceReader(trace_dir)
    if layer < 0 or layer >= reader.n_layers:
        raise ValueError(f"layer must be in [0, {reader.n_layers}), got {layer}")

    rows = _rows_for_decode_steps(trace_dir=trace_dir, reader=reader, num_decode_steps=num_decode_steps)
    trace = {
        "input": reader.get_decoder_input(layer, rows=rows),
        "output": reader.get_decoder_output(layer, rows=rows),
    }
    return _validate_reference_trace(trace, source=trace_dir)


def load_reference_kv(
    trace_root: Path | str,
    *,
    model_id: str,
    prompt_id: str,
    layer: int,
    num_decode_steps: int | None = None,
    target_layout: Literal["trace", "tt_device"] = "trace",
) -> torch.Tensor:
    """Load one chunked KV reference, optionally converted to TT device layout."""
    trace_dir = Path(trace_root) / model_trace_id(model_id) / prompt_id
    if not trace_dir.is_dir():
        raise FileNotFoundError(
            f"chunked trace directory not found: {trace_dir} "
            f"(from trace_root={trace_root!r}, model_id={model_id!r}, prompt_id={prompt_id!r})"
        )

    reader = ChunkedTraceReader(trace_dir)
    if layer < 0 or layer >= reader.n_layers:
        raise ValueError(f"layer must be in [0, {reader.n_layers}), got {layer}")

    rows = _rows_for_decode_steps(trace_dir=trace_dir, reader=reader, num_decode_steps=num_decode_steps)
    kv = reader.get_kv(layer, rows=rows).unsqueeze(0)

    if target_layout == "trace":
        pass
    elif target_layout == "tt_device":
        kv = _trace_kv_to_tt_device_layout(kv)
    else:
        raise ValueError(f"unknown KV target_layout {target_layout!r}; expected 'trace' or 'tt_device'")

    return _validate_reference_kv(kv, source=trace_dir)
