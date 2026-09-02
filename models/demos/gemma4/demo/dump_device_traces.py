# SPDX-License-Identifier: Apache-2.0
"""SCRIPT 1 — dump device-side tensors in the bit_sculpt GPU-trace format.

Produces, for a 262144-token / chunk-8192 prefill on an 8x4 mesh (CP=8, TP=4):

    <out>/decoder_io/decoder_input_layer_0/rows_{s:08d}_{e:08d}.safetensors
    <out>/decoder_io/decoder_output_layer_{i}/rows_{s:08d}_{e:08d}.safetensors   i in 0..59
    <out>/kv_cache/layer_{i}/rows_{s:08d}_{e:08d}.safetensors                    i in 0..59

matching the reference layout byte-for-byte in path, tensor key, shape and dtype, so
script 2 can compare them without any special-casing.

WHY THIS RUNS EAGER. Intermediate per-layer activations cannot be extracted from a
replayed trace -- a trace is a recorded graph and its interior tensors are not addressable
from the host. So this dumps with eager dispatch, one forward per chunk. It is much slower
than the traced test and is a diagnostic, not a perf measurement.

KV is different: the ring cache is a persistent tensor that accumulates and is never
cleared, so ALL chunks are read once at the end rather than 32 times.

Env:
    GEMMA4_TRACE_DUMP_DIR      required, output root
    GEMMA4_DUMP_LAYERS         "all" (default) or "0,1,5"
    GEMMA4_DUMP_CHUNKS         "all" (default) or "0,1,2"
    GEMMA4_DUMP_SKIP_KV        "1" to skip the KV pass
    GEMMA4_DUMP_SKIP_DECODER   "1" to skip decoder in/out
"""
from __future__ import annotations

import os
import pathlib
import time

import pytest
import torch
from loguru import logger
from safetensors.torch import save_file

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric

from .text_demo_prefill import (
    _SLIDING_WINDOW_TOKENS,
    L1_SMALL_SIZE,
    TRACE_REGION_SIZE,
    _build_prefill_model,
    _cp_gather_torch,
    _host_tensor,
    _lm_head_deferred,
    _mesh_config,
    _model_path,
    _prefill_tokens,
)


def _sel(env, n):
    """Parse an "all" / "0,1,5" selector into a sorted index list."""
    raw = os.environ.get(env, "all").strip()
    if raw in ("all", ""):
        return list(range(n))
    return sorted({int(x) for x in raw.split(",") if x.strip() != ""})


def _rows_name(chunk_idx, chunk):
    lo, hi = chunk_idx * chunk, (chunk_idx + 1) * chunk
    return f"rows_{lo:08d}_{hi:08d}.safetensors"


def _save(out_dir, fname, key, tensor):
    out_dir.mkdir(parents=True, exist_ok=True)
    # bf16 to match the reference's save_dtype, and contiguous because safetensors
    # refuses views.
    save_file({key: tensor.to(torch.bfloat16).contiguous()}, str(out_dir / fname))


class _CaptureLayer:
    """Transparent wrapper that reads a layer's output back IMMEDIATELY.

    ``model.layers`` is iterated as ``for i, layer in enumerate(self.layers)`` and called
    as ``layer(...)``, so replacing the list entries is enough -- no monkeypatching of
    classes, and the wrapper forwards attribute access so ``layer.self_attn`` still works.

    The readback must happen inside ``__call__``, not afterwards from a stashed handle:
    tt-metal frees each layer's ``hidden_states`` as soon as the next layer consumes it,
    so a handle kept until after the forward points at deallocated device memory and
    segfaults in ``to_torch``.
    """

    def __init__(self, inner, idx, on_output):
        self._inner, self._idx, self._on_output = inner, idx, on_output

    def __call__(self, *a, **kw):
        out = self._inner(*a, **kw)
        self._on_output(self._idx, out)
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


@torch.no_grad()
@parametrize_mesh_with_fabric(
    [(8, 4), (4, 8)],
    device_params_extra={"trace_region_size": TRACE_REGION_SIZE, "l1_small_size": L1_SMALL_SIZE},
)
@pytest.mark.parametrize("token_source", ["text"], ids=lambda t: t)
@pytest.mark.parametrize("chunk_size", [8192], ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize("context_len", [262144], ids=lambda c: f"ctx_{c // 1024}k")
@pytest.mark.timeout(21600)
def test_dump_device_traces(mesh_device, context_len, chunk_size, token_source, reset_seeds, request):
    """Run the prefill eagerly and dump every tensor the GPU trace set contains."""
    from models.demos.gemma4.tt.ccl import cp_degree

    out_root = os.environ.get("GEMMA4_TRACE_DUMP_DIR")
    if not out_root:
        pytest.skip("set GEMMA4_TRACE_DUMP_DIR to enable the dump")
    out_root = pathlib.Path(out_root)

    chunk = chunk_size
    mesh_config = _mesh_config(mesh_device)
    cp = cp_degree(mesh_config)
    if cp <= 1:
        pytest.skip(f"targets CP>1; mesh {tuple(mesh_device.shape)} gives CP={cp}")
    if chunk < _SLIDING_WINDOW_TOKENS * cp:
        pytest.skip(f"chunk={chunk} under window*cp={_SLIDING_WINDOW_TOKENS * cp}")

    model_path = _model_path()
    n_chunks = context_len // chunk
    model_args, model, kv_cache, page_table_tt = _build_prefill_model(
        mesh_device, model_path, chunk, context_len=context_len
    )
    tokens_all = _prefill_tokens(model_path, context_len, model_args.vocab_size, token_source)

    n_layers = len(model.layers)
    want_layers = _sel("GEMMA4_DUMP_LAYERS", n_layers)
    want_chunks = _sel("GEMMA4_DUMP_CHUNKS", n_chunks)
    do_kv = os.environ.get("GEMMA4_DUMP_SKIP_KV", "0") != "1"
    do_dec = os.environ.get("GEMMA4_DUMP_SKIP_DECODER", "0") != "1"
    logger.info(
        f"[dump] out={out_root} layers={len(want_layers)}/{n_layers} chunks={len(want_chunks)}/{n_chunks} "
        f"decoder={do_dec} kv={do_kv} cp={cp} tp={mesh_device.shape[1]}"
    )

    host_input = _host_tensor(
        mesh_device,
        tokens_all[:, :chunk].contiguous(),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh_config=mesh_config,
        seq_dim=-1,
    )
    device_input = ttnn.to_device(host_input, device=mesh_device)

    dec_dir = out_root / "decoder_io"
    want_layer_set = set(want_layers)
    ctx = {"chunk": -1, "saved": 0}

    def _on_layer_output(idx, tensor):
        """Gather + persist this layer's output while it is still allocated."""
        if not do_dec or ctx["chunk"] not in want_chunks or idx not in want_layer_set:
            return
        t = _cp_gather_torch(tensor, mesh_device, mesh_config).reshape(-1, model_args.dim)
        _save(
            dec_dir / f"decoder_output_layer_{idx}",
            _rows_name(ctx["chunk"], chunk),
            f"decoder_output_layer_{idx}",
            t,
        )
        ctx["saved"] += 1
        del t

    model.layers = [_CaptureLayer(l, i, _on_layer_output) for i, l in enumerate(model.layers)]
    t0 = time.time()
    for c in range(n_chunks):
        chunk_start = c * chunk
        staged = _host_tensor(
            mesh_device,
            tokens_all[:, chunk_start : chunk_start + chunk].contiguous(),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh_config=mesh_config,
            seq_dim=-1,
        )
        ttnn.copy_host_to_device_tensor(staged, device_input)
        model.ccl_manager.set_ring_metadata(slot_idx=0, kv_actual_global=chunk_start)

        ctx["chunk"] = c
        tc = time.time()
        with _lm_head_deferred(model):
            embeds, page_table, chunk_page_table, _ = model.transform_and_embed_prefill_inputs_device(
                device_input, page_table_tt, None, None
            )
            # decoder_input_layer_0 is the embedding output -- the real stream, not an alias.
            if do_dec and c in want_chunks:
                emb = _cp_gather_torch(embeds, mesh_device, mesh_config).reshape(-1, model_args.dim)
                _save(
                    dec_dir / "decoder_input_layer_0",
                    _rows_name(c, chunk),
                    "decoder_input_layer_0",
                    emb,
                )
                del emb
            out = model.ttnn_prefill_forward(
                x=embeds,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start,
                kv_cache=kv_cache,
                get_last_token=-1,
                user_id=0,
            )
        ttnn.synchronize_device(mesh_device)
        fwd_s = time.time() - tc

        out.deallocate(True)
        logger.info(f"[dump] chunk {c + 1}/{n_chunks} fwd={fwd_s:.1f}s elapsed={time.time() - t0:.0f}s")

    # ── KV: one read per layer, covering every chunk ──────────────────────────────
    if do_kv:
        composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 1))
        slab = chunk // cp
        kv_dir = out_root / "kv_cache"
        for i in want_layers:
            attn = model.layers[i].self_attn
            if getattr(attn, "ring_kv_cache", None) is None:
                logger.warning(f"[dump] layer {i} has no ring_kv_cache, skipped")
                continue
            parts = []
            for tt in (attn.ring_kv_cache[0], attn.ring_kv_cache[1]):
                # [1, H, cp*seq_local, D] with the token axis in RANK-major order
                full = ttnn.to_torch(tt, mesh_composer=composer).float()
                _, heads, rows_all, hd = full.shape
                parts.append(full.reshape(1, heads, cp, rows_all // cp, hd))
                del full
            k_all, v_all = parts
            for c in want_chunks:
                blocks = []
                for a in (k_all, v_all):
                    b = a[:, :, :, c * slab : (c + 1) * slab, :]  # [1,H,cp,slab,D]
                    # rank-major -> global token order, then flatten heads: [T, H*D]
                    b = b.permute(0, 2, 3, 1, 4).reshape(cp * slab, -1)
                    blocks.append(b)
                kv = torch.cat(blocks, dim=-1)  # K||V on the feature axis, head-major
                _save(kv_dir / f"layer_{i}", _rows_name(c, chunk), f"kv_post_transform_layer_{i}", kv)
                del kv, blocks
            del k_all, v_all, parts
            logger.info(f"[dump] kv layer {i}: wrote {len(want_chunks)} chunks")

    logger.info(f"[dump] DONE in {time.time() - t0:.0f}s -> {out_root}")
