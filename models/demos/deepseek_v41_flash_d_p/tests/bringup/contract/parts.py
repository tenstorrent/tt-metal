# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared driver of the serving-contract part tests: the bring-up model interface (hooks.device_model) run at the
chunk starts tt-d-gen sends, its served caches read back and compared with the golden.

The contract adds keyword arguments and two state methods to the framework's model interface (serving_contract.md
"Input", "Attention and cache writes"); the ladder's positional calls stay valid:

    model.embed(tokens, start=0)              tokens: one chunk [CHUNK] int64, natural order, PAD_ID (0xFFFFFFFF) past
                                              the real tokens; laid out on the SP rows as the server lays out a chunk
                                              whose first token is at absolute position `start`
    model.from_host(h, start=0)               h: [CHUNK, 4 * 5120 + 4] block input rows, natural order, same layout rule
    model.layer(i, h, start, state, end=None) end = actual_end: rows at or past it are pad
    model.begin_chunk(tokens, start)          (F65) the chunk's ids, natural order, PAD_ID past actual_end
    state.served(layer, length) -> {config name: tensor}
                                              the layer's served caches (caches.served_configs), natural order, the rows
                                              of every record covering token positions [0, ceil32(length)): window_kv
                                              [ceil32(length), 512]; compress_kv_rR / index_k_rR [ceil32(length) / R, w]
    state.load_served(layer, {config name: tensor}, length), state.load_tokens(ids [length])
                                              the inverse (a migrated prefix, for the runner smoke's stand-in decode)
"""

from __future__ import annotations

import inspect
import os
import re

import numpy as np
import pytest
import torch

from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import caches as C
from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import server_rules as R


def run_layers(spec) -> tuple[str, list[int]]:
    """The layers the part test runs: at a component gate C.<block type>.attention (BRINGUP_TASK) the block type's
    representative and its state sources (hooks.state_sources: a reader layer reads its source's caches as of the end
    of the chunk); by hand, at K.1 or Z.1 every spec layer."""
    m = re.match(r"^C\.(\w+)\.attention$", os.environ.get("BRINGUP_TASK", ""))
    if m and m.group(1) in spec.data["block_types"]:
        rep = spec.representative_layer(m.group(1))
        return m.group(1), sorted(set(spec.state_sources(rep)) | {rep})
    return "all", list(spec.layers())


def build_model(mesh_device, spec, layers):
    """hooks.device_model for `layers`, failing cleanly ("not built") when a serving entry point is missing."""
    R.self_check()
    hooks = spec.hooks()
    try:
        model = hooks.device_model(mesh_device, spec, layers, lm_head=False)
    except NotImplementedError as e:
        pytest.fail(f"not built: hooks.device_model: {e}", pytrace=False)
    missing = []
    for fn, kw in (("embed", "start"), ("from_host", "start"), ("layer", "end")):
        f = getattr(model, fn, None)
        if f is None or kw not in inspect.signature(f).parameters:
            missing.append(f"{fn}(..., {kw}=...)")
    if missing:
        pytest.fail(
            "not built: the device model has no serving entry point " + ", ".join(missing) + " (serving_contract.md "
            "'Input' / 'Attention and cache writes': any 32-aligned start in the server's SP layout, pad rows past "
            "actual_end)",
            pytrace=False,
        )
    return model


def new_state(model):
    state = model.new_state(R.MAX_SEQ)
    if not callable(getattr(state, "served", None)):
        pytest.fail(
            "not built: state.served(layer, length) -> {config name: tensor} is missing (serving_contract.md "
            "'KV cache': the served caches, read back in natural order)",
            pytrace=False,
        )
    return state


def chunk_tokens(tokens: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """The chunk's ids in natural order, PAD_ID past actual_end (prefill_writer.cpp:93-97)."""
    t = torch.full((R.CHUNK,), R.PAD_ID, dtype=torch.int64)
    t[: end - start] = tokens[start:end]
    return t


def block_rows(gk: C.GoldenKV, layer: int, start: int) -> torch.Tensor:
    """Golden block input rows [start, start + CHUNK): real rows past actual_end too, so a model that lets pad rows
    into the cache writes non-zero rows there (the pad check catches it)."""
    x = gk.block_input(layer)[start : start + R.CHUNK]
    assert x.shape[0] == R.CHUNK, (start, x.shape)
    return x.float()


def run_chunk(model, state, gk: C.GoldenKV, layers: list[int], start: int, end: int) -> None:
    """One chunk through `layers` (layer 0 from the embedding, the others from their golden block input); a model that
    rejects the start fails as "not built"."""
    toks = chunk_tokens(gk.tokens(), start, end)
    try:
        if hasattr(model, "begin_chunk"):
            model.begin_chunk(toks, start)
        for i in layers:
            h = model.embed(toks, start=start) if i == 0 else model.from_host(block_rows(gk, i, start), start=start)
            h2 = model.layer(i, h, start, state, end=end)
            model.free(h)
            model.free(h2)
        model.sync()
    except Exception as err:  # an assert / TT_FATAL on a start the server sends is "not built", not a test error
        pytest.fail(f"not built: chunk [{start}, {end}) was rejected: {type(err).__name__}: {err}", pytrace=False)


def served(state, layer: int, length: int) -> dict:
    """state.served as {config name: bf16 torch tensor}, the shapes checked."""
    out = {}
    got = state.served(layer, length)
    for cfg in C.layer_configs(layer):
        if cfg["name"] not in got:
            pytest.fail(
                f"not built: state.served({layer}, ...) has no {cfg['name']!r} (has {list(got)})", pytrace=False
            )
        t = got[cfg["name"]].detach().cpu().to(torch.bfloat16)
        rows = R.ceil_to(length, 32) // cfg["ratio"]
        if t.ndim != 2 or t.shape[1] != cfg["width"] or t.shape[0] < rows:
            pytest.fail(
                f"state.served({layer}, {length})[{cfg['name']!r}] is {tuple(t.shape)}, expected at least "
                f"[{rows}, {cfg['width']}]",
                pytrace=False,
            )
        out[cfg["name"]] = t[:rows]
    return out


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def check_turn(kvc, gk, layer: int, got: dict, held: dict | None, start: int, end: int, thr: float) -> list[str]:
    """After a turn [start, end): rows of [0, start) bit-identical to what the state held, rows of [start, end) vs the
    golden (server per-channel PCC), the pad rows of the last record (and of a group still filling) exactly zero."""
    fails = []
    for cfg in C.layer_configs(layer):
        t = got[cfg["name"]]
        a, b = C.rows_of(cfg, start, end)
        pad_end = R.ceil_to(end, 32) // cfg["ratio"]
        if held is not None and a:
            h = held[cfg["name"]][:a]
            if not torch.equal(bits(t[:a]), bits(h)):
                bad = (bits(t[:a]) != bits(h)).any(dim=1).nonzero().flatten()
                fails.append(
                    f"{cfg['name']} layer {layer}: {bad.numel()} rows below actual_start {start} changed (first rows "
                    f"{bad[:4].tolist()}); a chunk may only write its own positions"
                )
        want = gk.rows(cfg, layer, start, end)
        fails += C.pcc_failures(kvc, cfg, t[a:b].float().numpy(), want, layer, thr, f"[{start}, {end})")
        pad = t[b:pad_end]
        if pad.numel() and not torch.all(pad == 0):
            fails.append(
                f"{cfg['name']} layer {layer}: pad rows {b}..{pad_end - 1} (tokens [{end}, {R.ceil_to(end, 32)})) not "
                f"zero (max |x| {pad.float().abs().max().item():.4g}): the last 32-token record ships whole"
            )
    return fails


def as_np(t: torch.Tensor) -> np.ndarray:
    return t.float().numpy()
