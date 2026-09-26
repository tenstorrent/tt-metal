# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Performance: warm time of one chunk at several start positions, to show how a chunk's cost grows with the prefix.

Positions come from spec ``perf.positions`` (token offsets), default 0 and 1..4 x the start of the target's last chunk
(Gemma-4 at 55k in 5k chunks: 0, 50k, 100k, 150k, 200k; k = 1024). Positions may lie past the target: the device model
is built with ``target.seq`` raised in memory to the longest position, so it sizes its position tables for it. Timing
only: the KV prefix is zeros and the tokens are random ids (the work does not depend on the values). Each position runs
once to compile, then once timed, with one sync and nothing read back.

Records pos_chunk, pos_ms_<start> per position, device_model_hybrid.
"""

from __future__ import annotations

import gc
import time

import torch

from models.demos.common.bringup.core import metrics


def positions(s) -> list[int]:
    got = s.get("perf.positions")
    if got:
        return [int(p) for p in got]
    last = int(s.get("target.seq")) - int(s.get("target.chunk"))
    return [k * last for k in range(5)]


def _free(state) -> None:
    """Release a state's device buffers: its own free(), else free() on each cache it holds."""
    if hasattr(state, "free"):
        state.free()
        return
    for v in vars(state).values():
        for c in v.values() if isinstance(v, dict) else [v]:
            if hasattr(c, "free"):
                c.free()


def run_positions(s, mesh) -> list[tuple[int, float]]:
    chunk, starts = int(s.get("target.chunk")), positions(s)
    s.data["target"]["seq"] = max(int(s.get("target.seq")), max(starts) + chunk)  # in memory only
    layers = s.layers()
    model = s.hooks().device_model(mesh, s, layers, lm_head=False)
    metrics.record("device_model_hybrid", int("Hybrid" in type(model).__name__))
    metrics.record("pos_chunk", chunk)
    vocab = int(s.get("checkpoint.config.vocab_size", 32000) or 32000)
    rows = []
    for start in starts:
        state = model.new_state(start + chunk)
        tokens = torch.randint(0, vocab, (chunk,))

        def once():
            h = model.embed(tokens)
            for i in layers:
                h2 = model.layer(i, h, start, state)
                model.free(h)
                h = h2
            if layers[-1] == s.num_layers - 1:
                h2 = model.final_norm(h)
                model.free(h)
                h = h2
            model.sync()
            model.free(h)

        once()  # compile this position's programs
        t0 = time.time()
        once()
        ms = (time.time() - t0) * 1e3
        metrics.record(f"pos_ms_{start}", round(ms, 1))
        rows.append((start, ms))
        print(f"position {start}->{start + chunk}: {ms:.1f} ms ({chunk / ms * 1e3:.0f} tok/s)", flush=True)
        _free(state)
        del state
        gc.collect()
    return rows
