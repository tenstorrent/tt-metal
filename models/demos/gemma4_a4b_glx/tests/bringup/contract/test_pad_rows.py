# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Pad rows: the server pads a chunk past actual_end with 0xFFFFFFFF (prefill_writer.cpp:93-97) and the layer ack
for the chunk tells the KV Manager every record up to actual_end is final. The device model's serving entry points
(embed(tokens, start=), layer(i, h, start, state, end=)) must take the pad id and leave the cache rows at and past
actual_end zero (the record holding actual_end is shipped whole), on a sliding and a global layer, for a chunk
ending mid-record (1017) and one starting at a tile-aligned, non-chunk-aligned position (1312 -> 3017).
"""

import inspect

import pytest
import torch
from bringup_fw.testing.contract import PAD, engine_env
from bringup_fw.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
engine_env(S)
pytestmark = device_timeout(S)
LAYERS = [0, 1, 2, 3, 4, 5]  # through the first global layer (5): sliding window 1024 and K = V
CASES = ((0, 1017), (1312, 3017))  # (actual_start, actual_end); 1312 = 41 * 32, not a multiple of 64 or the chunk


def _not_built(why):
    pytest.fail(f"not built: {why}", pytrace=False)


@mesh_parametrize
def test_pad_rows(mesh_device):
    if mesh_device is None:
        return
    rung = S.rung(S.data["ladder"][0]["name"])
    seq, chunk = rung["seq"], rung["chunk"]
    model = S.hooks().device_model(mesh_device, S, LAYERS, lm_head=False)
    if "start" not in inspect.signature(model.embed).parameters:
        _not_built("device model embed(tokens, start=) (rotary/position offset of a chunk the server sends)")
    if "end" not in inspect.signature(model.layer).parameters:
        _not_built("device model layer(i, h, start, state, end=) (actual_end: rows past it are pad)")
    gen = torch.Generator().manual_seed(0)
    vocab = int(getattr(model.cfg, "vocab_size", 262144))
    failed = []
    for start, end in CASES:
        state = model.new_state(seq)
        if start:
            # A resident prefix (random bytes are fine: only the rows past `end` are checked).
            for layer in LAYERS:
                cur = state.to_torch(layer, start)
                state.load_prefix(layer, {k: torch.randn_like(v) for k, v in cur.items()}, start)
        real = torch.randint(0, vocab, (end - start,), generator=gen)
        tokens = torch.cat([real, torch.full((chunk - (end - start),), PAD, dtype=torch.long)])
        h = model.embed(tokens, start=start)
        for layer in LAYERS:
            h2 = model.layer(layer, h, start, state, end=end)
            model.free(h)
            h = h2
        model.free(h)
        model.sync()
        for layer in LAYERS:
            kv = state.to_torch(layer, min(start + chunk, seq))
            for name, t in kv.items():
                t = t.float()
                past = t[..., end:, :]
                if not torch.isfinite(t[..., start:end, :]).all():
                    failed.append(f"[{start}, {end}) layer {layer} {name}: non-finite rows below actual_end")
                if torch.count_nonzero(past):
                    rows = sorted({int(r) + end for r in torch.nonzero(past.abs().sum(-1))[..., -1].unique()})
                    failed.append(
                        f"[{start}, {end}) layer {layer} {name}: {len(rows)} pad rows past actual_end not "
                        f"zero, e.g. {rows[:4]}"
                    )
        print(f"chunk [{start}, {end}) + pad to {start + chunk}: checked layers {LAYERS}")
    assert not failed, "\n".join(failed)
