# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared driver of the serving-contract part tests: the bring-up model interface (hooks.device_model) run at the
chunk starts tt-d-gen sends, with its KV read back and compared to the golden.

The contract adds two keyword arguments to the framework's model interface (serving_contract.md "Attention and cache
writes"); the ladder's positional calls stay valid:

    model.embed(tokens, start=0)                 tokens: one chunk [CHUNK] int64 in natural order, PAD_ID past the
                                                 real tokens; laid out on the SP rows as the server lays them out for
                                                 a chunk whose first token is at absolute position `start`
    model.layer(i, h, start, state, end=None)    end = actual_end: rows at or past it are pad
    state.load_prefix(i, {"kv_latent": t}, n) / state.to_torch(i, n)["kv_latent"]   natural order [n, 576]
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from models.demos.xing40_a4b_d_p.tests.bringup.contract import server_rules as R

LAYERS = [0, 1]  # dense layer 0 (KV from the embedding only) and layer 1 (KV from a real attention output)


def build_model(mesh_device, spec):
    """hooks.device_model for LAYERS, failing cleanly ("not built") when the serving keywords are missing."""
    R.self_check()
    hooks = spec.hooks()
    if not hasattr(hooks, "device_model"):
        pytest.fail("not built: hooks.device_model is missing", pytrace=False)
    model = hooks.device_model(mesh_device, spec, LAYERS, lm_head=False)
    missing = []
    if "start" not in inspect.signature(model.embed).parameters:
        missing.append("embed(tokens, start=...)")
    if "end" not in inspect.signature(model.layer).parameters:
        missing.append("layer(i, h, start, state, end=...)")
    if missing:
        pytest.fail(
            "not built: the device model has no serving entry point " + ", ".join(missing) + " (serving_contract.md, "
            "'Attention and cache writes': any tile-aligned start, pad rows past actual_end)",
            pytrace=False,
        )
    return model


def chunk_tokens(tokens: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """The chunk's tokens in natural order, PAD_ID past actual_end (prefill_writer.cpp:93-97)."""
    t = torch.full((R.CHUNK,), R.PAD_ID, dtype=torch.int64)
    t[: end - start] = tokens[start:end]
    return t


def run_chunks(model, state, tokens: torch.Tensor, plan: list[tuple]) -> None:
    """Every chunk of `plan` through LAYERS; a model that rejects a start fails as "not built"."""
    for s, e in plan:
        try:
            h = model.embed(chunk_tokens(tokens, s, e), start=s)
            for i in LAYERS:
                h2 = model.layer(i, h, s, state, end=e)
                model.free(h)
                h = h2
            model.free(h)
            model.sync()
        except Exception as err:  # an assert / TT_FATAL on an unaligned start is "not built", not a test error
            pytest.fail(f"not built: chunk [{s}, {e}) was rejected: {type(err).__name__}: {err}", pytrace=False)


def read_kv(state, layer: int, n: int) -> torch.Tensor:
    return state.to_torch(layer, n)["kv_latent"].float()[:n]


def load_prefix(state, g, n: int) -> dict:
    """Golden kv_latent [0, n) into every layer's state; returns what the state then holds (for the untouched check)."""
    held = {}
    for i in LAYERS:
        if n:
            state.load_prefix(i, {"kv_latent": R.golden_kv(g, i)[:n]}, n)
            held[i] = read_kv(state, i, n)
    return held


def check_kv(state, g, held: dict, prefix: int, end: int, thr: float) -> list[str]:
    """Rows [0, prefix) untouched, [prefix, end) vs golden (kv_dump_compare's per-channel PCC), pad rows of the last
    32-row record zero."""
    _, kvc = R.harness()
    fails = []
    pad_end = R.ceil_to(end, R.RECORD_TOKENS)
    for i in LAYERS:
        got = read_kv(state, i, pad_end)
        if prefix and not torch.equal(got[:prefix], held[i]):
            bad = (got[:prefix] != held[i]).any(dim=1).nonzero().flatten()
            fails.append(
                f"layer {i}: rows below actual_start changed ({bad.numel()} rows, first {bad[:4].tolist()}); a chunk "
                "may only write [actual_start, actual_end)"
            )
        fails += kv_vs_golden(kvc, got[prefix:end], g, i, prefix, end, thr)
        pad = got[end:pad_end]
        if pad.numel() and not torch.all(pad == 0):
            fails.append(
                f"layer {i}: pad rows [{end}, {pad_end}) not zero (max |x| {pad.abs().max().item():.4g}); the last "
                "32-token record ships whole, its pad rows must be zero before the ack"
            )
    return fails


def kv_vs_golden(kvc, got: torch.Tensor, g, layer: int, lo: int, hi: int, thr: float) -> list[str]:
    want = R.golden_kv(g, layer)[lo:hi]
    return R.kv_pcc_failures(kvc, got.numpy(), want.numpy(), layer, thr, f"kv [{lo}, {hi})")


def as_np(t: torch.Tensor) -> np.ndarray:
    return t.float().numpy()
