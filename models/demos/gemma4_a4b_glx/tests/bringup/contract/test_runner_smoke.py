# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""K.4: the runner smoke. The real prefill runner (tt-d-gen's engine feeding it) prefills the intake smoke prompt;
the KV of every layer is read through the exported table (hooks.contract_read_slot_kv) while the runner is up; the
runner stops, the device model loads that KV below the last prompt token (state.load_prefix), recomputes the tail
from there (embed(start=), layer(start, end=)) and decodes greedily. The answer must contain the expected text.

Records smoke_runner_ok (K.1's gate) and runner_smoke.json.
"""

import tempfile
import time
from pathlib import Path

import pytest
import torch
from bringup_fw.core import metrics
from bringup_fw.intake.check_hf_sanity import smoke_ids
from bringup_fw.reference.golden import tokenizer
from bringup_fw.reference.prompt import template_kwargs
from bringup_fw.testing.contract import contract_rung, engine_env
from bringup_fw.testing.harness import device_params, device_timeout, impl_mode, spec

from models.demos.gemma4_a4b_glx.tests.bringup.contract import _runner as R

S = spec()
engine_env(S)
pytestmark = device_timeout(S)
MAX_NEW = 8


def _open_mesh(s):
    import ttnn

    fabric = device_params(s).get("fabric_config")
    if fabric is not None:
        ttnn.set_fabric_config(fabric)
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*s.mesh)), fabric


def _close_mesh(mesh, fabric):
    import ttnn

    ttnn.close_mesh_device(mesh)
    if fabric is not None:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# The runner opens the mesh itself: this test opens it only after the runner has exited.
@pytest.mark.parametrize("mesh_device", [None] if impl_mode() != "device" else ["runner"], ids=["runner"])
def test_runner_smoke(mesh_device):
    if mesh_device is None:
        return
    metrics.record("smoke_runner_ok", 0)
    t_all = time.time()
    kind, build = R.preflight(S)
    if kind != "tt-d-gen engine":
        pytest.skip(f"feeder is {kind}: the runner smoke drives tt-d-gen's engine")
    sm = S.get("intake.smoke")
    tok = tokenizer(S)
    ids = smoke_ids(tok, sm["prompt"], template_kwargs(S))
    rung = S.rung(contract_rung(S))
    seq, chunk = rung["seq"], rung["chunk"]
    assert len(ids) + MAX_NEW <= chunk, f"smoke prompt of {len(ids)} tokens does not fit one {chunk}-token chunk"
    try:
        trace = str(R.golden(S).dir)
    except BaseException:  # noqa: BLE001 -- the smoke needs no golden; the runner falls back to its default
        trace = None

    with tempfile.TemporaryDirectory(prefix="g4glx_smoke_") as td:
        runner = R.Runner(S, Path(td), trace_dir=trace)
        try:
            runner.wait_ready()
            out = runner.drive(build, [{"name": "smoke", "token_ids": ids}], "smoke")["requests"]["smoke"]
            kv = S.hooks().contract_read_slot_kv(S, runner.table(), runner.dmap(), int(out["slot"]), len(ids))
        finally:
            runner.stop()

    keep = len(ids) - 1  # the server's KV below the last prompt token; decode recomputes from there
    mesh, fabric = _open_mesh(S)
    try:
        model = S.hooks().device_model(mesh, S, S.layers(), lm_head=True)
        eos = {t for t in (tok.eos_token_id,) if t is not None}
        new = []
        for step in range(MAX_NEW):
            tail = ids[keep:] + new
            state = model.new_state(seq)
            for layer in S.layers():
                state.load_prefix(layer, kv[layer], keep)
            tokens = torch.tensor(tail + [0] * (chunk - len(tail)), dtype=torch.long)
            h = model.embed(tokens, start=keep)
            for layer in S.layers():
                h2 = model.layer(layer, h, keep, state, end=keep + len(tail))
                model.free(h)
                h = h2
            hidden = model.final_norm(h)
            model.free(h)
            nxt = int(model.logits(hidden, [len(tail) - 1]).float()[0].argmax())
            model.free(hidden)
            print(f"runner smoke step {step}: token {nxt} {tok.decode([nxt])!r}", flush=True)
            if nxt in eos:
                break
            new.append(nxt)
            if step >= 1 and sm["expect"].lower() in tok.decode(new).lower():
                break
    finally:
        _close_mesh(mesh, fabric)

    text = tok.decode(new)
    ok = sm["expect"].lower() in text.lower()
    print(f"runner smoke: {sm['prompt']!r} -> {text!r} (expect {sm['expect']!r}): {'ok' if ok else 'FAIL'}")
    metrics.record("smoke_runner_ok", int(ok))
    metrics.record_json(
        "runner_smoke",
        {
            "prompt": sm["prompt"],
            "expected": sm["expect"],
            "answer": text,
            "token_ids": [int(t) for t in new],
            "ok": bool(ok),
            "slot": out["slot"],
            "prompt_len": len(ids),
            "kv_from_server": keep,
            "seconds": round(time.time() - t_all, 1),
        },
    )
    assert ok, f"runner smoke answered {text!r}, expected {sm['expect']!r}"
