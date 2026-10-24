# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, "Attention and cache writes" (serving_contract.md): every served layer at tt-d-gen's starts.

The framework's chunk-start tests (L.starts) run only the first two layers. MiMo's other attention kinds sit deeper:
layer 5 is a full-attention layer (4 KV heads, 1 per chip) after four sliding-window layers (window 128, 8 KV heads,
attention sink), and a sliding layer at a mid-chunk start must read the 128 keys before actual_start from the cache
rows an earlier chunk wrote, wherever the SP reshuffle put them. This test runs all served layers (0-5) through the
device model's serving entry points, from the target rung's golden KV prefix:

    follow_up_tile   a follow-up over a 32-aligned (not 64-aligned) resident prefix, its first chunk starting mid SP
                     slab (1312 = 2 x 640 + 32): tt-d-gen's plan for it (server_rules.scenario, KNOWN 5120 / 56320)
    pulled_back      the chunk before a pulled-back last chunk, then that chunk at max_seq - chunk (tt-d-gen #430)

Pass, at every layer: rows below actual_start unchanged, rows [start, end) vs the golden KV with PCC >= max(0.93,
thresholds.state), the pad rows of the last 32-token record exactly zero (chunk_starts.check_kv). Gates M.1
(assemble).
"""

import inspect

import pytest
import torch
from bringup_fw.reference.golden import Golden
from bringup_fw.testing import chunk_starts as CS
from bringup_fw.testing import server_rules as R
from bringup_fw.testing.harness import device_timeout, mesh_parametrize, spec


class _Model:
    """hooks.device_model driven the way the runtime drives it for one slot: embed(tokens, start=), layer(..., end=)."""

    def __init__(self, s, mesh, layers, max_seq, chunk):
        try:
            self.model = s.hooks().device_model(mesh, s, layers, lm_head=False)
        except NotImplementedError as e:
            raise CS.NotBuilt(f"hooks.device_model: {e}") from e
        missing = []
        if "start" not in inspect.signature(self.model.embed).parameters:
            missing.append("embed(tokens, start=...)")
        if "end" not in inspect.signature(self.model.layer).parameters:
            missing.append("layer(i, h, start, state, end=...)")
        if missing:
            raise CS.NotBuilt("the device model has no serving entry point " + ", ".join(missing))
        self.layers, self.chunk = layers, chunk
        self.state = self.model.new_state(max_seq)

    def load_prefix(self, layer, tensors, n):
        self.state.load_prefix(layer, tensors, n)

    def run(self, tokens, plan):
        for lo, hi in plan:
            t = torch.full((self.chunk,), R.PAD_ID, dtype=torch.int64)
            t[: hi - lo] = tokens[lo:hi]
            try:
                h = self.model.embed(t, start=lo)
                for i in self.layers:
                    h2 = self.model.layer(i, h, lo, self.state, end=hi)
                    self.model.free(h)
                    h = h2
                self.model.free(h)
                self.model.sync()
            except Exception as e:  # an assert / TT_FATAL on a start the model does not take is "not built"
                raise CS.NotBuilt(f"chunk [{lo}, {hi}) was rejected: {type(e).__name__}: {e}") from e

    def read(self, layer, n):
        return self.state.to_torch(layer, n)

    def free(self):
        if hasattr(self.state, "free"):
            self.state.free()


def _case(s, mesh, g, layers, tokens, chunk, max_seq, name, prefix, plan, end) -> list[str]:
    print(f"case {name}: layers {layers}, golden prefix {prefix}, chunks {plan}", flush=True)
    mod = _Model(s, mesh, layers, max_seq, chunk)
    try:
        held = CS.load_prefix(s, mod, g, layers, prefix)
        mod.run(tokens, plan)
        return CS.check_kv(s, mod, g, layers, held, prefix, end, f"all_layers_{name}")
    finally:
        mod.free()


@mesh_parametrize
@device_timeout(spec())
def test_cache_writes_all_layers(mesh_device):
    if mesh_device is None:
        pytest.skip("no mesh (CPU proof mode): the device model's cache writes")
    s = spec()
    rung = CS.target_rung(s)
    if rung is None:
        pytest.fail("not built: no all-chunks rung with more than one chunk to take the golden KV from", pytrace=False)
    chunk, max_seq = rung["chunk"], rung["seq"]
    try:
        g = Golden.for_rung(s, rung["name"])
    except FileNotFoundError as e:
        pytest.fail(f"not built: {e}", pytrace=False)
    layers, tokens, sc = list(s.layers()), g.tokens(), R.scenario(chunk, max_seq)
    fails = []
    try:
        res, n = sc["follow_up_tile"]
        fails += _case(
            s,
            mesh_device,
            g,
            layers,
            tokens,
            chunk,
            max_seq,
            "follow_up_tile",
            res,
            R.chunk_plan(n, res, chunk, max_seq),
            n,
        )
        res, n = sc["pulled_back"]
        plan = R.chunk_plan(n, res, chunk, max_seq)
        assert R.pulled_back(plan) and plan[-1] == (max_seq - chunk, n), plan  # the server's plan, not the model's
        fails += _case(s, mesh_device, g, layers, tokens, chunk, max_seq, "pulled_back", plan[-2][0], plan[-2:], n)
    except CS.NotBuilt as e:
        pytest.fail(f"not built: {e}", pytrace=False)
    for f in fails:
        print(f"FAIL {f}")
    assert not fails, f"{len(fails)} cache-write checks failed, first: {fails[0]}"
