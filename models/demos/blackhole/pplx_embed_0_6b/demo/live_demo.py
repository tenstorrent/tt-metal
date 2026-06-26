# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Live / on-demand embedding inference for pplx-embed-v1-0.6B on Blackhole (P150).

Unlike the ``demo_bs*_isl*.py`` perf scripts (which run a fixed synthetic input
to benchmark latency), this script loads the model **once** onto a single P150
and keeps it **resident**, then serves embedding requests on demand.  Use it to
call ``model.forward`` manually on custom inputs before any serving-stack
integration:

  * Interactive prompt (default): type/paste text, press Enter, get an embedding.
    The process stays alive between requests — the device model is loaded once.
    With ``--fast`` every prompt replays the captured prefill trace and prints
    the **live trace-replay latency (device + D2H + pool)** alongside the
    accurate post-processed (RMSNorm + mean-pool + L2-norm) embedding.
  * ``--input FILE``:  encode one text per non-empty line of the file.
  * ``--input DIR``:   encode each file in the directory as a single document.

Each request runs a real forward pass on device:
  tokenize -> bidirectional prefill -> final RMSNorm -> mean-token pooling
  -> (optional) L2 normalization.

This matches pplx-embed's mean-pooling embedding recipe (see ``eval_accuracy.py``
for the CPU reference used to validate parity).

Examples
--------
    # Interactive live demo (model stays up; --fast prints replay latency/prompt):
    TT_VISIBLE_DEVICES=0 python models/demos/blackhole/pplx_embed_0_6b/demo/live_demo.py --fast

    # Interactive, accurate-for-any-length (SDPA padding mask + real-token pool):
    TT_VISIBLE_DEVICES=0 python models/demos/blackhole/pplx_embed_0_6b/demo/live_demo.py --fast --mask

    # Encode a file (one text per line) and save embeddings to .npy:
    TT_VISIBLE_DEVICES=0 python models/demos/blackhole/pplx_embed_0_6b/demo/live_demo.py \\
        --input my_texts.txt --output embeddings.npy

    # Encode every file in a folder (one document per file) to JSONL:
    TT_VISIBLE_DEVICES=0 python models/demos/blackhole/pplx_embed_0_6b/demo/live_demo.py \\
        --input ./docs/ --output embeddings.jsonl
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.pplx_embed_0_6b.demo._common import apply_workload_env, build_single_device_model
from models.tt_transformers.tt.common import (
    Mode,
    copy_host_to_device,
    get_block_size,
    get_padded_prefill_len,
    num_blocks_in_seq,
)

# The TT prefill path requires the (padded) sequence length to be a multiple of
# 128 (see attention.forward_prefill).  We pad up to the nearest multiple and
# mean-pool only over the real tokens.
PREFILL_ALIGN = 128


def _extract_final_norm(model):
    """Pull the final RMSNorm weight + eps off the device so we can apply the
    norm on host over the *full* token sequence (the on-device path only ever
    norms the last-token tile).  ``add_unit_offset`` is already baked into the
    stored weight."""
    norm_weight = ttnn.to_torch(ttnn.get_device_tensors(model.norm.norm.weight)[0]).reshape(-1).float()
    eps = float(model.norm.norm.eps)
    return norm_weight, eps


@torch.no_grad()
def encode_one(generator, model, kv_cache, page_table, tokenizer, norm_weight, eps, text, max_length, normalize=True):
    """Run one real forward pass on device and return a mean-pooled embedding.

    Pads the tokenized text up to the nearest tile (32) — keeping padding
    minimal — runs a bidirectional prefill, applies the final RMSNorm on host
    over the real tokens, and mean-pools.
    """
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    ids = enc["input_ids"]
    n = int(ids.shape[1])
    if n == 0:
        return None

    # Pad up to the nearest 128 multiple (prefill requirement), keeping padding
    # as small as possible.
    seq_len = ((n + PREFILL_ALIGN - 1) // PREFILL_ALIGN) * PREFILL_ALIGN
    seq_len = min(seq_len, model.args.max_seq_len)
    n = min(n, seq_len)

    ids_padded = torch.zeros(1, seq_len, dtype=torch.long)
    ids_padded[0, :n] = ids[0, :n]

    block_size = get_block_size(kv_cache)
    num_blocks = num_blocks_in_seq(seq_len, block_size)
    page_table_user = page_table[0:1, :num_blocks]

    generator.mode = Mode.PREFILL
    tt_out = generator.prefill_forward_single_user_text(
        ids_padded,
        page_table=page_table_user,
        user_id=0,
        last_token_idx=n - 1,
        kv_cache=kv_cache,
        model_id=0,
        return_hidden_states=True,
    )

    # Full pre-norm hidden states [1, 1, seq_len, H] -> host
    hidden = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])[0, 0, :n, :].float()
    ttnn.deallocate(tt_out)

    # Final RMSNorm (on host, over real tokens) then mean-token pooling
    variance = hidden.pow(2).mean(dim=-1, keepdim=True)
    hidden = hidden * torch.rsqrt(variance + eps) * norm_weight
    emb = hidden.mean(dim=0)

    if normalize:
        emb = emb / emb.norm().clamp_min(1e-12)
    return emb


def maybe_patch_fastokens():
    """Best-effort global patch of HF ``AutoTokenizer`` with crusoe/atero ``fastokens``.

    Enabled when ``PPLX_FASTOKENS=1`` is set in the environment (the server's
    ``main`` propagates it to spawned workers). A no-op (with a warning) if the
    package is not installed, so the import is never a hard dependency. Must run
    *before* any ``from_pretrained`` so the fast Rust engine is adopted.
    Returns ``True`` if the patch was applied.
    """
    if os.environ.get("PPLX_FASTOKENS", "0") != "1":
        return False
    try:
        import fastokens

        fastokens.patch_transformers()
        return True
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning(f"PPLX_FASTOKENS=1 but fastokens unavailable ({exc}); using HF tokenizer.")
        return False


def _fast_encode(tokenizer, text, max_len):
    """Tokenize ``text`` to a ``list[int]`` (special tokens included), truncated
    to ``max_len``, using the raw Rust ``backend_tokenizer`` when available.

    The HF Python wrappers (``__call__``/``encode``) add several milliseconds of
    BatchEncoding/truncation bookkeeping per call; the backend tokenizer's
    ``encode(text).ids`` is the same result with far less host overhead. Falls
    back to ``tokenizer.encode`` if no fast backend is present.
    """
    raw = getattr(tokenizer, "_tt_raw_backend", "unset")
    if raw == "unset":
        raw = getattr(tokenizer, "backend_tokenizer", None)
        try:
            tokenizer._tt_raw_backend = raw  # cache on the tokenizer instance
        except Exception:
            pass
    if raw is not None:
        ids = raw.encode(text).ids
        if len(ids) > max_len:
            ids = ids[:max_len]
        return ids
    return tokenizer.encode(text, truncation=True, max_length=max_len)


class TracedEncoder:
    """Fixed-ISL encoder that replays a captured prefill trace per request.

    This is the low-latency serving path: it captures the prefill forward as a
    hardware trace once (at ``seq_len``), then each request only copies fresh
    tokens to device and replays the trace — matching the benchmark's direct
    trace latency (~7.1-7.2ms at ISL=512 on a P150).  Every input is padded /
    truncated to the fixed ``seq_len`` and mean-pooled over the real tokens.
    """

    def __init__(
        self,
        generator,
        model,
        kv_cache,
        page_table,
        tokenizer,
        norm_weight,
        eps,
        seq_len,
        device,
        pool="fast",
        use_mask=False,
        use_2cq=False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.norm_weight = norm_weight
        self.eps = eps
        self.seq_len = seq_len
        self.device = device
        self.kv_cache = kv_cache
        # 2-CQ overlapped-input serving (standard structure, see
        # models/tt_cnn/tt/executor.py:MultiCQTracedModelOverlappedInputExecutor):
        # per-request tokens are written to a DRAM staging buffer on CQ1, copied
        # into the trace's input on CQ0, so the next request's H2D overlaps the
        # current trace's compute. Requires the device opened with 2 CQs.
        self.use_2cq = use_2cq
        self.op_event = None
        # "fast"   -> device mean-pool over the full ISL (lowest latency).
        # "masked" -> device norm only; host mean over real tokens.
        self.pool = pool
        # When True, inject an additive SDPA padding mask so padded tokens are
        # excluded from bidirectional attention (accuracy becomes independent of
        # padding fraction). Forces real-token pooling.
        self.use_mask = use_mask

        # Release warmup traces captured by the generator so the capture below
        # doesn't trip "unsafe allocation during active trace" warnings.
        for key, tid in list(generator.trace_id_prefill.items()):
            if tid is not None:
                ttnn.release_trace(device, tid)
                generator.trace_id_prefill[key] = None

        block_size = get_block_size(kv_cache)
        num_blocks = num_blocks_in_seq(seq_len, block_size)
        self.page_table_user = page_table[0:1, :num_blocks]

        # Persistent padding-mask device buffer (shared across all layers' SDPA).
        self.mask_buf = None
        self._mask_torch = None
        self._last_n = -1
        if use_mask:
            from models.demos.blackhole.pplx_embed_0_6b.tt.attention import set_pad_attn_mask

            (self.mask_buf,) = copy_host_to_device((self._make_mask_host(seq_len),), mesh_device=device)
            set_pad_attn_mask(self.mask_buf)

        # Persistent pooling selector [1,1,1,S] (1/n on real cols) for device-side
        # masked mean: matmul(pool_vec, hidden) == mean over real tokens.
        self.pool_vec = None
        if self.pool == "masked":
            (self.pool_vec,) = copy_host_to_device((self._make_pool_vec_host(seq_len),), mesh_device=device)

        dummy = torch.zeros(1, seq_len, dtype=torch.long)
        host_inputs = model.prepare_prefill_inputs_trace(dummy, page_table=self.page_table_user)
        self._rot_g, self._rot_l = host_inputs[1], host_inputs[2]
        ext_host = (host_inputs[0], host_inputs[3], host_inputs[4])

        # Warm-run (compile), then capture the forward as a trace.
        device_inputs = copy_host_to_device(ext_host, mesh_device=device)
        _ = self._forward(device_inputs)
        ttnn.synchronize_device(device)

        device_inputs = copy_host_to_device(ext_host, mesh_device=device)
        self.trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        self.trace_out = self._forward(device_inputs)
        ttnn.end_trace_capture(device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(device)
        self.device_inputs = device_inputs

        if use_2cq:
            # DRAM staging buffers written on CQ1; consumed into device_inputs on CQ0.
            self.stage_inputs = copy_host_to_device(ext_host, mesh_device=device)
            self.op_event = ttnn.record_event(device, 0)

    def _forward(self, device_inputs):
        tr = self.model.transform_and_embed_prefill_inputs_device(*device_inputs)
        out = self.model.ttnn_prefill_forward(
            x=tr[0],
            rot_mats_global=self._rot_g,
            rot_mats_local=self._rot_l,
            page_table=tr[1],
            chunk_page_table=tr[2],
            kv_cache=self.kv_cache,
        )
        # Fold the final RMSNorm + mean-pool onto the device so only the pooled
        # [1, H] vector is copied back (keeps replay at the benchmark's ~7.1-7.2ms
        # instead of a 512xH D2H + host norm).
        h = self.model.norm(out, mode="prefill")
        if self.pool == "fast":
            # Mean over the full (padded) ISL.
            h = ttnn.mean(h, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            # Masked mean over real tokens via a [1,1,1,S] selector matmul.
            h = ttnn.matmul(self.pool_vec, h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return h

    def _make_mask_host(self, n):
        """Additive padding mask [1, 1, S, S] (bf16): real key columns 0, padded -1e9.

        SDPA requires the mask query dim to match Q (``mask_shape[2] == S``), so the
        full SxS mask is built. A persistent host buffer is reused to avoid realloc;
        only the boundary changes per request.
        """
        if self._mask_torch is None:
            self._mask_torch = torch.zeros(1, 1, self.seq_len, self.seq_len, dtype=torch.bfloat16)
        self._mask_torch[:, :, :, :n] = 0.0
        self._mask_torch[:, :, :, n:] = -1e9
        return ttnn.from_torch(self._mask_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def _make_pool_vec_host(self, n):
        """Pooling selector [1, 1, 1, S] (bf16): 1/n on real cols, 0 on padded.

        ``matmul(pool_vec, hidden)`` then yields the mean over the n real tokens.
        """
        v = torch.zeros(1, 1, 1, self.seq_len, dtype=torch.bfloat16)
        n = max(1, min(n, self.seq_len))
        v[:, :, :, :n] = 1.0 / n
        return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def _tokenize(self, text):
        """Tokenize ``text`` to a list of ids, capped at this encoder's ISL.

        Prefers the raw Rust ``backend_tokenizer`` (much faster than the HF Python
        ``__call__``/``encode`` wrappers, which carry BatchEncoding + truncation
        bookkeeping overhead), falling back to ``.encode`` if unavailable.
        """
        ids = _fast_encode(self.tokenizer, text, self.seq_len)
        return ids, min(len(ids), self.seq_len)

    def _build_inputs(self, ids, n):
        """Build host input tensors from pre-tokenized ids (host-side, not replay).

        ``ids`` is a ``list[int]``. Only the token tensor is request-dependent.
        The page table is static, so it is built once and cached. The rotary
        matrices that ``prepare_prefill_inputs_trace`` also computes are baked into
        the captured trace and unused here, so we skip recomputing them every
        request (this was the bulk of the old per-request host overhead).
        """
        if n == 0:
            return None
        ids_padded = torch.zeros(1, 1, 1, self.seq_len, dtype=torch.long)
        ids_padded[0, 0, 0, :n] = torch.as_tensor(ids[:n], dtype=torch.long)
        tokens_host = ttnn.from_torch(
            ids_padded,
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.model.mesh_device),
        )
        if getattr(self, "_static_host_inputs", None) is None:
            full = self.model.prepare_prefill_inputs_trace(
                torch.zeros(1, self.seq_len, dtype=torch.long), page_table=self.page_table_user
            )
            self._static_host_inputs = (full[3], full[4])
        return (tokens_host, self._static_host_inputs[0], self._static_host_inputs[1]), n

    def _prepare(self, text):
        """Tokenize + build host input tensors (host-side, not timed as replay)."""
        ids, n = self._tokenize(text)
        if n == 0:
            return None
        return self._build_inputs(ids, n)

    def _prepare_from_ids(self, ids):
        """Build host input tensors from ids tokenized elsewhere (e.g. the server).

        Skips the tokenizer entirely — used by the server-side tokenization path
        where the HTTP layer tokenizes on its dedicated cores and ships token ids
        to the worker, so the worker never contends for CPU on the tokenizer.
        """
        n = min(len(ids), self.seq_len)
        if n == 0:
            return None
        return self._build_inputs(ids, n)

    def _write_inputs(self, ext_host, n):
        """1-CQ input update: H2D tokens (+ mask/pool on length change) on CQ0."""
        copy_host_to_device(ext_host, device_tensors=self.device_inputs, mesh_device=self.device)
        # Pooling selector + padding mask only change when the token count changes;
        # cache by ``n`` so repeated-length requests skip the host rebuild + H2D.
        if n != self._last_n:
            if self.pool == "masked":
                copy_host_to_device(
                    (self._make_pool_vec_host(n),), device_tensors=(self.pool_vec,), mesh_device=self.device
                )
            if self.use_mask:
                copy_host_to_device(
                    (self._make_mask_host(n),), device_tensors=(self.mask_buf,), mesh_device=self.device
                )
            self._last_n = n

    def _write_inputs_2cq(self, ext_host, n):
        """2-CQ input update (overlapped-input executor structure).

        CQ1 writes the request tokens into DRAM staging (overlapping the previous
        trace's compute on CQ0); CQ0 then copies staging into the trace inputs and
        records ``op_event`` so CQ1 may safely overwrite staging during the next
        compute.  Mask/pool (rare, length-change only) are written on CQ0.
        """
        ttnn.wait_for_event(1, self.op_event)
        for h, s in zip(ext_host, self.stage_inputs):
            if h is not None:
                ttnn.copy_host_to_device_tensor(h, s, cq_id=1)
        write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, write_event)
        for s, d in zip(self.stage_inputs, self.device_inputs):
            if s is not None:
                ttnn.copy(s, d)
        if n != self._last_n:
            if self.pool == "masked":
                ttnn.copy_host_to_device_tensor(self._make_pool_vec_host(n), self.pool_vec)
            if self.use_mask:
                ttnn.copy_host_to_device_tensor(self._make_mask_host(n), self.mask_buf)
            self._last_n = n
        self.op_event = ttnn.record_event(self.device, 0)

    @torch.no_grad()
    def _replay(self, ext_host, n, normalize=True):
        """Copy fresh tokens, replay the prefill trace, pull the pooled embedding.

        Norm + mean-pool happen on-device inside the trace, so only the pooled
        [H] vector is returned.  The device pools over the full (padded) ISL; for
        full-length inputs this equals the real-token mean.
        """
        if self.use_2cq:
            self._write_inputs_2cq(ext_host, n)
        else:
            self._write_inputs(ext_host, n)
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=False)
        out_host = self.trace_out.cpu(blocking=False)
        ttnn.synchronize_device(self.device)

        emb = ttnn.to_torch(ttnn.get_device_tensors(out_host)[0]).reshape(-1).float()
        if normalize:
            emb = emb / emb.norm().clamp_min(1e-12)
        return emb

    @torch.no_grad()
    def _replay_timed(self, ext_host, n, normalize=True):
        """Like ``_replay`` but returns ``(emb, timings)`` with each step timed.

        ``timings`` (ms): ``h2d`` (token + mask/pool copy), ``device`` (trace
        execute = prefill + RMSNorm + pool, all folded into the trace), ``d2h``
        (copy the pooled [H] vector back), ``host`` (to_torch + L2-norm), and
        ``total``.  Steps use blocking boundaries so they don't overlap, so the
        sum is marginally higher than the production overlapped ``_replay``.
        """
        timings = {}
        t = time.perf_counter()
        if self.use_2cq:
            self._write_inputs_2cq(ext_host, n)
        else:
            self._write_inputs(ext_host, n)
        timings["h2d"] = (time.perf_counter() - t) * 1000

        # Device compute: prefill + RMSNorm + pool (all inside the trace).
        t = time.perf_counter()
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.device)
        timings["device"] = (time.perf_counter() - t) * 1000

        # D2H: copy the pooled [1,H] vector back to host.
        t = time.perf_counter()
        out_host = self.trace_out.cpu(blocking=True)
        timings["d2h"] = (time.perf_counter() - t) * 1000

        # Host finalize: to_torch + (optional) L2 normalization.
        t = time.perf_counter()
        emb = ttnn.to_torch(ttnn.get_device_tensors(out_host)[0]).reshape(-1).float()
        if normalize:
            emb = emb / emb.norm().clamp_min(1e-12)
        timings["host"] = (time.perf_counter() - t) * 1000

        timings["total"] = timings["h2d"] + timings["device"] + timings["d2h"] + timings["host"]
        return emb, timings

    @torch.no_grad()
    def encode(self, text, normalize=True):
        prepared = self._prepare(text)
        if prepared is None:
            return None
        ext_host, n = prepared
        return self._replay(ext_host, n, normalize=normalize)


def bucket_lengths(max_length):
    """Canonical prefill buckets for ``max_length`` using the same tiers as
    ``get_padded_prefill_len`` ({128, 256, 512, 1024, ...}), capped at the
    padded ``max_length``.  Mirrors the standard text/VL prefill bucketing in
    ``tt_transformers.generator`` (one trace per padded length)."""
    cap = get_padded_prefill_len(max_length)
    tiers = [128, 256, 512, 1024]
    out = [t for t in tiers if t <= cap]
    if not out or out[-1] != cap:
        out.append(cap)
    return out


class BucketedEncoder:
    """Sequence-length-bucketed encoder: holds one ``TracedEncoder`` per padded
    length tier and routes each request to the smallest bucket that fits.

    This is the standard prefill pattern used across the text/VL demos (see
    ``Generator._easy_trace_prefill``: traces are keyed by ``get_padded_prefill_len``
    and captured per length).  A short input is padded to e.g. 128 instead of 512,
    so it runs a smaller/faster trace (≈5.4ms@128 vs ≈7.6ms@512) *and* carries far
    less padding — keeping the cheap maskless ``fast`` path accurate without the
    512×512 SDPA mask.

    Exposes the same ``_prepare`` / ``_replay`` / ``_replay_timed`` / ``encode``
    interface as ``TracedEncoder`` so REPL / bench / DP call sites are unchanged.
    The chosen per-request encoder is stashed in ``_active`` between ``_prepare``
    and the matching ``_replay*`` (single-threaded per process, as in every
    call site here).
    """

    def __init__(
        self,
        generator,
        model,
        kv_cache,
        page_table,
        tokenizer,
        norm_weight,
        eps,
        buckets,
        device,
        pool="fast",
        use_mask=False,
        use_2cq=False,
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.buckets = sorted(buckets)
        self.max_seq = self.buckets[-1]
        self.use_mask = use_mask
        self.pool = pool
        self.encoders = {}
        for b in self.buckets:
            self.encoders[b] = TracedEncoder(
                generator,
                model,
                kv_cache,
                page_table,
                tokenizer,
                norm_weight,
                eps,
                b,
                device,
                pool=pool,
                use_mask=use_mask,
                use_2cq=use_2cq,
            )
        self._active = None

    def _pick(self, n):
        """Smallest bucket >= n (canonical ``get_padded_prefill_len`` tiers),
        clamped to the largest available bucket."""
        return get_padded_prefill_len(min(max(1, n), self.max_seq))

    def _bucket_for_text(self, text):
        enc = self.tokenizer(text, truncation=True, max_length=self.max_seq, return_tensors="pt")
        n = int(enc["input_ids"].shape[1])
        return self.encoders[self._pick(n)], n

    def _prepare(self, text):
        # Tokenize once (at the largest bucket's limit), pick the smallest bucket
        # that fits, then build inputs from the already-tokenized ids — avoids the
        # double tokenization of the old _bucket_for_text + te._prepare path.
        t0 = time.perf_counter()
        ids = _fast_encode(self.tokenizer, text, self.max_seq)
        n = len(ids)
        if n == 0:
            self._active = None
            return None
        te = self.encoders[self._pick(n)]
        self._active = te
        t1 = time.perf_counter()
        out = te._build_inputs(ids, min(n, te.seq_len))
        self._prep_tok_ms = (t1 - t0) * 1000.0
        self._prep_build_ms = (time.perf_counter() - t1) * 1000.0
        return out

    def _prepare_from_ids(self, ids):
        """Pick a bucket + build host tensors from ids tokenized elsewhere.

        The server-side tokenization counterpart of ``_prepare``: token ids are
        produced on the HTTP layer's cores, so here we only select the smallest
        bucket that fits and build the (cheap) per-request token tensor.
        """
        n = len(ids)
        if n == 0:
            self._active = None
            return None
        te = self.encoders[self._pick(n)]
        self._active = te
        t1 = time.perf_counter()
        out = te._build_inputs(ids, min(n, te.seq_len))
        self._prep_tok_ms = 0.0  # tokenized off-worker (server threadpool)
        self._prep_build_ms = (time.perf_counter() - t1) * 1000.0
        return out

    @torch.no_grad()
    def _replay(self, ext_host, n, normalize=True):
        return self._active._replay(ext_host, n, normalize=normalize)

    @torch.no_grad()
    def _replay_timed(self, ext_host, n, normalize=True):
        emb, steps = self._active._replay_timed(ext_host, n, normalize=normalize)
        steps["bucket"] = self._active.seq_len
        return emb, steps

    @torch.no_grad()
    def encode(self, text, normalize=True):
        te, _ = self._bucket_for_text(text)
        return te.encode(text, normalize=normalize)


def _read_inputs(path: Path):
    """Return list of (name, text). File -> one text per non-empty line;
    directory -> one document per file."""
    if path.is_dir():
        items = []
        for fp in sorted(path.iterdir()):
            if fp.is_file():
                items.append((fp.name, fp.read_text(encoding="utf-8", errors="replace").strip()))
        return [it for it in items if it[1]]
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [(f"line_{i}", ln.strip()) for i, ln in enumerate(lines) if ln.strip()]


def _write_outputs(out_path: Path, names, embeddings):
    import numpy as np

    arr = torch.stack(embeddings).cpu().numpy()
    if out_path.suffix == ".npy":
        np.save(out_path, arr)
    elif out_path.suffix in (".jsonl", ".json"):
        import json

        with out_path.open("w") as f:
            for name, vec in zip(names, arr):
                f.write(json.dumps({"name": name, "embedding": vec.tolist()}) + "\n")
    else:  # default to .npy alongside a names sidecar
        np.save(out_path.with_suffix(".npy"), arr)
        out_path.with_suffix(".names.txt").write_text("\n".join(names))
    logger.info(f"Wrote {len(embeddings)} embeddings (dim={arr.shape[1]}) to {out_path}")


def _run_repl(encode_fn, traced=None, normalize=True, metrics=False):
    """Interactive prompt. With a ``traced`` encoder (``--fast``), each prompt is
    split into host prep (tokenize) and the live trace replay (device + D2H +
    pool) so the replay latency is reported per request next to the accurate,
    post-processed embedding.

    Per-request timing (the device/D2H/host/H2D breakdown and total latency) is
    only printed when ``metrics=True`` (``--metrics``); otherwise just the
    embedding is shown."""
    print()
    print("=" * 64)
    print("  pplx-embed-v1-0.6B is loaded and resident on device.")
    print("  Type text and press Enter to get an embedding.")
    if traced is not None and metrics:
        print("  Each prompt reports live trace-replay latency (device+D2H+pool).")
    print("  Commands: /quit or /exit to stop, Ctrl-D also exits.")
    print("=" * 64)
    prev = None
    while True:
        try:
            line = input("text> ").strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break

        if traced is not None:
            # Split timing: host tokenize/prep vs. the per-step device replay.
            t0 = time.perf_counter()
            prepared = traced._prepare(line)
            t_prep = (time.perf_counter() - t0) * 1000
            if prepared is None:
                print("  (empty input — nothing to encode)")
                continue
            ext_host, n = prepared
            emb, steps = traced._replay_timed(ext_host, n, normalize=normalize)
            t_replay = steps["total"]
            total = t_prep + t_replay
        else:
            t0 = time.perf_counter()
            emb = encode_fn(line)
            total = (time.perf_counter() - t0) * 1000
            t_prep = t_replay = None
            steps = None
            n = None

        if emb is None:
            print("  (empty input — nothing to encode)")
            continue
        preview = ", ".join(f"{v:+.4f}" for v in emb[:8].tolist())
        print(f"  dim={emb.shape[0]}  |emb|={emb.norm().item():.4f}  tokens={n if n is not None else '?'}")
        print(f"  first 8 dims: [{preview}, ...]")
        if metrics:
            if steps is not None:
                print(
                    f"  replay steps: device(prefill+norm+pool)={steps['device']:.1f}ms"
                    f"  D2H={steps['d2h']:.2f}ms  host(to_torch+norm)={steps['host']:.2f}ms"
                    f"  H2D={steps['h2d']:.2f}ms"
                )
                print(f"  latency: replay={t_replay:.1f}ms  | tokenize+prep={t_prep:.1f}ms  | total={total:.1f}ms")
            else:
                print(f"  latency: end-to-end={total:.1f}ms")
        if prev is not None:
            cos = torch.dot(prev, emb).item()
            print(f"  cosine similarity to previous input: {cos:.4f}")
        prev = emb


# ──────────────────────────────────────────────────────────────────────────────
# Data-parallel (DP=N) live serving across a Blackhole Galaxy (32 chips)
#
# Each chip runs in its own subprocess (TT_VISIBLE_DEVICES isolation, CPU-pinned),
# builds a resident TracedEncoder, and serves embedding requests from a queue.
# The parent process reads user text / files and dispatches work across all chips,
# matching the proven multi-process DP model in ``dp32_multiprocess.py`` but for
# real, on-demand inputs instead of a synthetic benchmark.
# ──────────────────────────────────────────────────────────────────────────────


def _smt_cores(phys_ids, n_phys):
    """Expand physical core ids to all their SMT-sibling logical cpu ids.

    On a 2-threads/core host, physical core ``p`` owns logical cpus ``p`` and
    ``p + n_phys`` (verified via /sys thread_siblings).
    """
    cores = set()
    for p in phys_ids:
        cores.add(int(p))
        cores.add(int(p) + n_phys)
    return cores


def compute_worker_cores(chip_id, n_workers, cpw, server_phys=0, total_logical=None):
    """Logical-cpu affinity set for worker ``chip_id`` (or ``None`` => OS-float).

    SMT-aware: physical core ``p`` -> logical ``{p, p+n_phys}``.
      * ``cpw > 0``  : dedicate ``cpw`` physical cores per worker (round-robin over
                       the worker pool) -> clean per-worker isolation (each worker
                       owns whole physical cores incl. both hyperthreads).
      * ``cpw == 0`` : workers share the pool. Returns ``None`` (unset affinity)
                       when the pool is every core; otherwise the shared set.
    ``server_phys`` physical cores at the top are reserved for the server.
    """
    total_logical = total_logical or os.cpu_count() or 64
    n_phys = max(1, total_logical // 2)
    pool_phys = max(1, n_phys - max(0, int(server_phys)))  # worker pool = phys [0, pool_phys)
    if cpw and cpw > 0:
        phys = [(chip_id * cpw + i) % pool_phys for i in range(cpw)]
        return _smt_cores(phys, n_phys)
    if server_phys and server_phys > 0:
        return _smt_cores(range(pool_phys), n_phys)
    return None  # everything floats (OS-scheduled)


def _dp_worker(chip_id, args, normalize, task_q, result_q, ready_q, worker_cores):
    """Single chip: build a resident TracedEncoder, then serve from ``task_q``.

    ``worker_cores`` is a precomputed set of logical cpu ids to pin to, or ``None``
    to leave affinity unset (OS-scheduled). Pinning a worker to a *single
    hyperthread* starves the device dispatch/completion thread and adds large tail
    latency; whole physical cores (both siblings) or OS-float are preferred.

    Protocol:
      * On ready (after warmup) puts ``(chip_id, build_sec)`` on ``ready_q``.
      * For each ``(idx, text)`` task, puts ``(idx, chip_id, emb_or_None, replay_ms)``
        on ``result_q`` (emb is a float32 numpy array).
      * A ``None`` task terminates the worker.
    """
    os.environ["TT_VISIBLE_DEVICES"] = str(chip_id)
    # Keep each worker's tokenizer single-threaded: with 32 workers, the HF Rust
    # tokenizer's intra-op thread pool would otherwise fan out to 32xN threads and
    # oversubscribe the host. (Also silences the post-fork parallelism warning.)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    maybe_patch_fastokens()
    if worker_cores:
        try:
            os.sched_setaffinity(0, set(worker_cores))
        except (OSError, AttributeError):
            pass

    try:
        import numpy as np

        import ttnn
        from models.demos.blackhole.pplx_embed_0_6b.demo._common import apply_workload_env, build_single_device_model
        from models.tt_transformers.tt.common import get_padded_prefill_len

        apply_workload_env(1, args.max_length)
        t_build0 = time.perf_counter()
        device = ttnn.open_device(
            device_id=0,
            l1_small_size=32768,
            trace_region_size=200_000_000,
            num_command_queues=2 if getattr(args, "cq2", False) else 1,
        )
        generator, model_args, kv_caches, page_table = build_single_device_model(
            device, batch_size=1, seq_len=args.max_length
        )
        model = generator.model[0]
        norm_weight, eps = _extract_final_norm(model)
        pool_mode = "masked" if args.mask else "fast"
        cq2 = getattr(args, "cq2", False)
        if args.no_bucket:
            fixed_isl = get_padded_prefill_len(args.max_length)
            enc = TracedEncoder(
                generator,
                model,
                kv_caches[0],
                page_table,
                model_args.tokenizer,
                norm_weight,
                eps,
                fixed_isl,
                device,
                pool=pool_mode,
                use_mask=args.mask,
                use_2cq=cq2,
            )
        else:
            enc = BucketedEncoder(
                generator,
                model,
                kv_caches[0],
                page_table,
                model_args.tokenizer,
                norm_weight,
                eps,
                bucket_lengths(args.max_length),
                device,
                pool=pool_mode,
                use_mask=args.mask,
                use_2cq=cq2,
            )
        _ = enc.encode("warm up the model", normalize=normalize)  # compile/warm
        ready_q.put((chip_id, time.perf_counter() - t_build0))

        while True:
            task = task_q.get()
            t_recv = time.perf_counter()
            if task is None:
                break
            idx, payload = task
            # ``payload`` is either the raw text (worker tokenizes) or a list of
            # pre-tokenized ids (server-side tokenization path).
            if isinstance(payload, str):
                prepared = enc._prepare(payload)
            else:
                prepared = enc._prepare_from_ids(payload)
            t_prep = time.perf_counter()
            if prepared is None:
                result_q.put((idx, chip_id, None, None, 0))
                continue
            ext_host, n = prepared
            emb, steps = enc._replay_timed(ext_host, n, normalize=normalize)
            # Cross-process timestamps (CLOCK_MONOTONIC, comparable to the server)
            # so the server can localize host-side overhead per hop.
            steps["t_recv"] = t_recv
            steps["t_prep"] = t_prep
            steps["t_done"] = time.perf_counter()
            steps["tok"] = getattr(enc, "_prep_tok_ms", 0.0)
            steps["build"] = getattr(enc, "_prep_build_ms", 0.0)
            result_q.put((idx, chip_id, emb.float().numpy().astype(np.float32), steps, n))
    except Exception as exc:  # surface build/serve failures to the parent
        try:
            ready_q.put((chip_id, None, f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
    finally:
        try:
            ttnn.close_device(device)  # noqa: F821
        except Exception:
            pass


class _DPServer:
    """Spawns N chip workers and dispatches embedding requests across them."""

    def __init__(self, args, normalize, cores_per_worker=0):
        self.n = args.dp
        self.normalize = normalize
        ctx = mp.get_context("spawn")
        self.task_qs = [ctx.Queue() for _ in range(self.n)]
        self.result_q = ctx.Queue()
        ready_q = ctx.Queue()
        self.procs = []
        for i in range(self.n):
            worker_cores = compute_worker_cores(i, self.n, cores_per_worker, server_phys=0)
            p = ctx.Process(
                target=_dp_worker,
                args=(i, args, normalize, self.task_qs[i], self.result_q, ready_q, worker_cores),
                name=f"chip-{i}",
                daemon=True,
            )
            p.start()
            self.procs.append(p)

        logger.info(f"Building {self.n} resident encoders (one per chip) — this takes ~1-2 min...")
        self.ready, self.failed = set(), {}
        for _ in range(self.n):
            msg = ready_q.get()
            if len(msg) == 3 and msg[1] is None:
                self.failed[msg[0]] = msg[2]
            else:
                self.ready.add(msg[0])
                logger.info(f"  chip {msg[0]:>2} ready (build {msg[1]:.0f}s) [{len(self.ready)}/{self.n}]")
        if self.failed:
            logger.error(f"{len(self.failed)} chip(s) failed to build:")
            for cid, err in sorted(self.failed.items()):
                logger.error(f"  chip {cid}: {err[:200]}")
        if not self.ready:
            raise RuntimeError("All DP workers failed to build.")
        logger.info(f"DP server ready: {len(self.ready)}/{self.n} chips live.")

    def _alive_chips(self):
        return sorted(self.ready)

    def broadcast(self, text):
        """Send the same text to every live chip; return (results, wall_ms).

        Demonstrates DP scaling: all chips embed in parallel in one round.
        results: list of (chip_id, emb_np_or_None, steps_or_None) where ``steps``
        is the per-step timing dict from ``_replay_timed``.
        """
        chips = self._alive_chips()
        t0 = time.perf_counter()
        for c in chips:
            self.task_qs[c].put((0, text))
        got = [self.result_q.get() for _ in chips]
        wall_ms = (time.perf_counter() - t0) * 1000
        return [(c, e, s) for (_, c, e, s, _n) in got], wall_ms

    def map(self, texts):
        """Distribute a list of texts round-robin across chips; return embeddings
        in input order plus (wall_ms, steps_list)."""
        chips = self._alive_chips()
        t0 = time.perf_counter()
        for k, text in enumerate(texts):
            self.task_qs[chips[k % len(chips)]].put((k, text))
        out = [None] * len(texts)
        steps_list = []
        for _ in range(len(texts)):
            idx, _chip, emb, steps, _n = self.result_q.get()
            out[idx] = emb
            if steps is not None:
                steps_list.append(steps)
        wall_ms = (time.perf_counter() - t0) * 1000
        return out, wall_ms, steps_list

    def close(self):
        for c in self._alive_chips():
            try:
                self.task_qs[c].put(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()


def _run_dp(args, normalize):
    """Interactive / file-driven live serving across DP=N chips on the galaxy."""
    server = _DPServer(args, normalize)
    n_live = len(server.ready)
    try:
        if args.input:
            in_path = Path(args.input)
            if not in_path.exists():
                logger.error(f"Input path does not exist: {in_path}")
                return
            items = _read_inputs(in_path)
            names = [nm for nm, _ in items]
            texts = [tx for _, tx in items]
            logger.info(f"Encoding {len(texts)} input(s) across {n_live} chips...")
            embs, wall_ms, steps_list = server.map(texts)
            done = [e for e in embs if e is not None]
            eps = len(done) / (wall_ms / 1000) if wall_ms > 0 else 0.0

            def _med(key):
                return statistics.median([s[key] for s in steps_list]) if steps_list else 0.0

            logger.info(f"Encoded {len(done)} texts in {wall_ms/1000:.2f}s  | aggregate {eps:.0f} emb/s")
            logger.info(
                f"  per-request medians: device(prefill+norm+pool)={_med('device'):.1f}ms  "
                f"D2H={_med('d2h'):.2f}ms  host(to_torch+norm)={_med('host'):.2f}ms  "
                f"H2D={_med('h2d'):.2f}ms  total={_med('total'):.1f}ms"
            )
            if args.output:
                embt = [torch.from_numpy(e) for e in embs if e is not None]
                _write_outputs(Path(args.output), [nm for nm, e in zip(names, embs) if e is not None], embt)
            return
        _run_dp_repl(server, n_live, metrics=args.metrics)
    finally:
        logger.info("Shutting down DP workers...")
        server.close()


def _run_dp_repl(server, n_live, metrics=False):
    print()
    print("=" * 64)
    print(f"  pplx-embed-v1-0.6B is resident on {n_live} chips (DP={n_live}).")
    print("  Type text and press Enter: it is embedded on ALL chips in parallel.")
    if metrics:
        print("  Reporting per-chip replay latency + aggregate throughput.")
    print("  Commands: /quit or /exit to stop, Ctrl-D also exits.")
    print("=" * 64)
    prev = None
    while True:
        try:
            line = input("text> ").strip()
        except EOFError:
            print()
            break
        if not line or line in ("/quit", "/exit"):
            if line in ("/quit", "/exit"):
                break
            continue
        results, wall_ms = server.broadcast(line)
        valid = [(c, e, s) for (c, e, s) in results if e is not None and s is not None]
        if not valid:
            print("  (empty input — nothing to encode)")
            continue
        steps_list = [s for _, _, s in valid]
        emb0 = torch.from_numpy(valid[0][1]).float()
        # All chips run the identical model+input → embeddings should match.
        max_dev = max(float(torch.from_numpy(e).float().sub(emb0).abs().max()) for _, e, _ in valid)
        eps = len(valid) / (wall_ms / 1000) if wall_ms > 0 else 0.0

        def _med(key):
            return statistics.median([s[key] for s in steps_list])

        def _rng(key):
            return min(s[key] for s in steps_list), max(s[key] for s in steps_list)

        preview = ", ".join(f"{v:+.4f}" for v in emb0[:8].tolist())
        print(f"  dim={emb0.shape[0]}  |emb|={emb0.norm().item():.4f}  chips={len(valid)}/{n_live}")
        print(f"  first 8 dims: [{preview}, ...]")
        if metrics:
            dev_med, (dev_lo, dev_hi) = _med("device"), _rng("device")
            print("  per-chip step medians (across chips):")
            print(f"    device (prefill+norm+pool): {dev_med:.1f}ms  [min {dev_lo:.1f}, max {dev_hi:.1f}]")
            print(f"    D2H  (pooled vector copy):  {_med('d2h'):.2f}ms")
            print(f"    host (to_torch + L2-norm):  {_med('host'):.2f}ms")
            print(f"    H2D  (tokens + mask/pool):  {_med('h2d'):.2f}ms")
            print(f"    total replay:               {_med('total'):.1f}ms")
            print(f"  DP round: {len(valid)} embeddings in {wall_ms:.1f}ms  → {eps:.0f} emb/s aggregate")
        print(f"  cross-chip max |Δ| (parity): {max_dev:.2e}")
        if prev is not None:
            print(f"  cosine similarity to previous input: {torch.dot(prev, emb0).item():.4f}")
        prev = emb0


def _bench_throughput(te, n_iters, ext_host, n):
    """Pipelined issue throughput (emb/s): back-to-back input-write + trace-execute
    with a single sync at the end (outputs not read back). With ``use_2cq`` the next
    request's H2D (CQ1) overlaps the current trace compute (CQ0); with 1 CQ the
    writes serialize behind compute on CQ0 — so this isolates the overlap benefit."""
    # Warm one iteration and settle.
    if te.use_2cq:
        te._write_inputs_2cq(ext_host, n)
    else:
        te._write_inputs(ext_host, n)
    ttnn.execute_trace(te.device, te.trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(te.device)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        if te.use_2cq:
            te._write_inputs_2cq(ext_host, n)
        else:
            te._write_inputs(ext_host, n)
        ttnn.execute_trace(te.device, te.trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(te.device)
    wall = time.perf_counter() - t0
    return n_iters / wall, wall / n_iters * 1000  # (emb/s, ms/iter)


def main():
    parser = argparse.ArgumentParser(
        description="Live / on-demand embedding inference for pplx-embed-v1-0.6B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=None, help="File (one text/line) or directory (one doc/file)")
    parser.add_argument("--output", type=str, default=None, help="Output path (.npy or .jsonl)")
    parser.add_argument("--device-id", type=int, default=0, help="Device id to open (default: 0)")
    parser.add_argument(
        "--dp",
        type=int,
        default=1,
        help="Data-parallel chip count (e.g. 32 on a Blackhole Galaxy). >1 spawns one "
        "resident encoder per chip and serves requests across all of them (fast/traced path).",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Max tokens per text (default: 512)")
    parser.add_argument("--no-normalize", action="store_true", help="Do not L2-normalize embeddings")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Low-latency serving path: capture a fixed-ISL prefill trace and replay it per "
        "request (~7.1-7.2ms at ISL=512). Inputs are padded/truncated to the fixed ISL.",
    )
    parser.add_argument(
        "--bench",
        type=int,
        default=0,
        help="Benchmark mode: encode the warmup text N times and report avg/best per-request latency.",
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        help="With --fast: inject the SDPA padding mask + real-token pooling (near-reference "
        "accuracy regardless of padding). Slightly higher latency than plain --fast.",
    )
    parser.add_argument(
        "--no-bucket",
        action="store_true",
        help="Disable sequence-length bucketing (--fast). By default --fast captures one trace "
        "per padded-length tier (128/256/512...) and routes each request to the smallest bucket "
        "that fits, so short inputs run a faster trace with less padding. Use this to force a "
        "single fixed-ISL trace at --max-length.",
    )
    parser.add_argument(
        "--cq2",
        action="store_true",
        help="With --fast: use the 2-command-queue overlapped-input serving structure (input H2D "
        "on CQ1 overlaps trace compute on CQ0). Improves pipelined/batch throughput, not "
        "single-request latency. Opens the device with 2 command queues.",
    )
    parser.add_argument(
        "--pool",
        choices=["fast", "masked", "masked-attn", "mask-fastpool"],
        default=None,
        help="Override pooling path for benchmarking: fast | masked (real-token, no mask) | "
        "masked-attn (real-token + SDPA padding mask) | mask-fastpool (SDPA mask + full-ISL mean).",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Print per-request performance metrics in the interactive prompt: the device / D2H / "
        "host / H2D step breakdown, total replay latency, and (with --dp) aggregate throughput. "
        "Off by default — only the embedding is shown.",
    )
    args = parser.parse_args()
    if args.pool in ("masked-attn", "mask-fastpool"):
        args.mask = True
    _pool_override = args.pool

    normalize = not args.no_normalize

    # Data-parallel (DP=N) serving across multiple chips (e.g. galaxy DP=32).
    # Each chip runs in its own process, so the parent must not open a device.
    if args.dp > 1:
        _run_dp(args, normalize)
        return

    apply_workload_env(1, args.max_length)

    logger.info(f"Opening device {args.device_id}...")
    device = ttnn.open_device(
        device_id=args.device_id,
        l1_small_size=32768,
        trace_region_size=200_000_000,
        num_command_queues=2 if (args.fast and args.cq2) else 1,
    )

    try:
        logger.info("Building pplx-embed-v1-0.6B (one-time load, then resident)...")
        generator, model_args, kv_caches, page_table = build_single_device_model(
            device, batch_size=1, seq_len=args.max_length
        )
        model = generator.model[0]
        tokenizer = model_args.tokenizer
        norm_weight, eps = _extract_final_norm(model)

        if args.fast:
            pool_mode = (
                "fast"
                if _pool_override in ("fast", "mask-fastpool")
                else "masked"
                if (args.mask or _pool_override == "masked")
                else "fast"
            )
            if args.no_bucket:
                fixed_isl = get_padded_prefill_len(args.max_length)
                logger.info(
                    f"Capturing fixed-ISL prefill trace (ISL={fixed_isl}, "
                    f"{'2-CQ' if args.cq2 else '1-CQ'}) for low-latency serving..."
                )
                traced = TracedEncoder(
                    generator,
                    model,
                    kv_caches[0],
                    page_table,
                    tokenizer,
                    norm_weight,
                    eps,
                    fixed_isl,
                    device,
                    pool=pool_mode,
                    use_mask=args.mask,
                    use_2cq=args.cq2,
                )
            else:
                buckets = bucket_lengths(args.max_length)
                logger.info(
                    f"Capturing bucketed prefill traces (ISL tiers={buckets}, "
                    f"{'2-CQ' if args.cq2 else '1-CQ'}) for low-latency serving..."
                )
                traced = BucketedEncoder(
                    generator,
                    model,
                    kv_caches[0],
                    page_table,
                    tokenizer,
                    norm_weight,
                    eps,
                    buckets,
                    device,
                    pool=pool_mode,
                    use_mask=args.mask,
                    use_2cq=args.cq2,
                )

            def encode_fn(text):
                return traced.encode(text, normalize=normalize)

        else:

            def encode_fn(text):
                return encode_one(
                    generator,
                    model,
                    kv_caches[0],
                    page_table,
                    tokenizer,
                    norm_weight,
                    eps,
                    text,
                    args.max_length,
                    normalize=normalize,
                )

        # Warm the program cache so the first real request isn't a compile.
        logger.info("Warming up...")
        _ = encode_fn("warm up the model")
        logger.info("Model is resident and ready.")

        if args.bench:
            if args.fast and isinstance(traced, BucketedEncoder):
                # Bucketed: report trace-replay latency per padded-length tier so
                # the per-bucket speedup (e.g. ~5.4ms@128 vs ~7.6ms@512) is visible.
                logger.info(f"Benchmarking {args.bench} iterations per bucket (tiers={traced.buckets})...")
                for b in traced.buckets:
                    te = traced.encoders[b]
                    ext_host, n = te._prepare("benchmark request")
                    replays = []
                    for _ in range(args.bench):
                        t0 = time.perf_counter()
                        _ = te._replay(ext_host, n, normalize=normalize)
                        replays.append((time.perf_counter() - t0) * 1000)
                    avg_r = sum(replays) / len(replays)
                    eps, ms_iter = _bench_throughput(te, args.bench, ext_host, n)
                    logger.info(
                        f"  ISL={b:>4}  Trace replay (device+D2H+pool): "
                        f"avg={avg_r:.1f}ms  best={min(replays):.1f}ms  worst={max(replays):.1f}ms  | "
                        f"pipelined {'2CQ' if traced.encoders[b].use_2cq else '1CQ'}: "
                        f"{eps:.0f} emb/s ({ms_iter:.1f}ms/iter)"
                    )
            else:
                fixed_isl = get_padded_prefill_len(args.max_length)
                logger.info(f"Benchmarking {args.bench} iterations (ISL={fixed_isl})...")
                totals = []
                for _ in range(args.bench):
                    t0 = time.perf_counter()
                    _ = encode_fn("benchmark request")
                    totals.append((time.perf_counter() - t0) * 1000)
                avg = sum(totals) / len(totals)
                logger.info(
                    f"  End-to-end (tokenize+prep+replay+pool): avg={avg:.1f}ms  best={min(totals):.1f}ms  worst={max(totals):.1f}ms"
                )
                if args.fast:
                    # Isolate the trace-replay latency (device + D2H + pool) that the
                    # benchmark suite reports — input prep/tokenization is pre-built.
                    ext_host, n = traced._prepare("benchmark request")
                    replays = []
                    for _ in range(args.bench):
                        t0 = time.perf_counter()
                        _ = traced._replay(ext_host, n, normalize=normalize)
                        replays.append((time.perf_counter() - t0) * 1000)
                    avg_r = sum(replays) / len(replays)
                    eps, ms_iter = _bench_throughput(traced, args.bench, ext_host, n)
                    logger.info(
                        f"  Trace replay (device+D2H+pool):        avg={avg_r:.1f}ms  best={min(replays):.1f}ms  worst={max(replays):.1f}ms"
                    )
                    logger.info(
                        f"  Pipelined throughput ({'2CQ' if traced.use_2cq else '1CQ'}): "
                        f"{eps:.0f} emb/s ({ms_iter:.1f}ms/iter)"
                    )
        elif args.input:
            in_path = Path(args.input)
            if not in_path.exists():
                logger.error(f"Input path does not exist: {in_path}")
                sys.exit(1)
            items = _read_inputs(in_path)
            logger.info(f"Encoding {len(items)} input(s) from {in_path}...")
            names, embeddings = [], []
            t0 = time.perf_counter()
            for name, text in items:
                emb = encode_fn(text)
                if emb is None:
                    continue
                names.append(name)
                embeddings.append(emb)
            dt = time.perf_counter() - t0
            logger.info(
                f"Encoded {len(embeddings)} texts in {dt:.1f}s ({dt / max(1, len(embeddings)) * 1000:.1f}ms/text)"
            )
            if args.output:
                _write_outputs(Path(args.output), names, embeddings)
            else:
                for name, emb in zip(names, embeddings):
                    logger.info(f"  {name}: dim={emb.shape[0]} |emb|={emb.norm().item():.4f}")
        else:
            _run_repl(encode_fn, traced=traced if args.fast else None, normalize=normalize, metrics=args.metrics)
    finally:
        logger.info("Closing device...")
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
