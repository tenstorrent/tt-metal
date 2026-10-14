# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, runner smoke: disaggregated serving end to end, with the bring-up device model standing in for
decode (serving_contract.md, "Adapter and table"). The contract's last test.

1. Prefill: tt-metal's real prefill runner (runner_case.py, as test_runner_contract runs it: the spec's 2D fabric, 1
   slot, D2H layer acks, mock migration on the migration-enabled path), fed by tt-d-gen's engine when a build is found,
   else by the producer fallback (the output and the recorded answer's `feeder` say which), prefills one request: the
   spec's intake.smoke prompt (smoke.prompt_ids: chat template with text.template_kwargs) after the fixed neutral
   system message PAD_SYSTEM, which puts it past 128 tokens so the migration boundary is past the first two blocks.
2. Migration: the KV the runner dumped through the exported table over UMD (what a KV Manager ships) is decoded for
   every served cache up to the boundary decode asks for, cap = (prompt_len - 1) // 64 * 64 (prefix_indexer.hpp:75-78,
   validation.py:55-61: decode receives [0, cap) and recomputes the last block).
3. Stand-in decode: a fresh mesh (spec device params), hooks.device_model(..., lm_head=True) for every layer,
   state.load_served(layer, {config: migrated rows}, cap) and state.load_tokens(ids[:cap]); the tail [cap, prompt_len)
   recomputed as one chunk at start=cap (model.embed(tokens, start=cap), layer(..., end=len)), final_norm + logits at
   the last position, greedy; each new token appended and the tail recomputed, until EOS, the expected answer, or 24
   tokens.
Pass: the decoded answer contains intake.smoke.expect ("Paris"). Records smoke_runner_ok and <results>/<task>_runner_smoke
.json (prompt, expected, answer, ids, ok, seconds, mode "runner", prompt_len, boundary, records, feeder).

It needs every layer of the model: in a partial run (the spec's layers stop short of the last layer, or skip one) it
skips with the reason; the owner decides where it runs (serving_contract.md, "Questions for the owner").
Fail fast: test_runner_contract.run_child. The precompile pass (UP_FRONT_COLLECT=1) skips the body.
"""

import json
import os
import time

import pytest
import torch

from models.demos.common.bringup.testing.harness import device_params, device_timeout, impl_mode, spec
from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import caches as C
from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import server_rules as R

S = spec()
pytestmark = device_timeout(S)
MAX_NEW = 24
PAD_SYSTEM = (
    "You are a helpful, knowledgeable assistant. You answer questions about geography, history, science and everyday "
    "life. Read each question carefully before you answer. Keep your answers short and accurate, and follow any format "
    "the user asks for, such as a single word, a number or a short list. If a question has one clear answer, give that "
    "answer directly without extra explanation. If the user asks for one word, reply with exactly one word. Do not "
    "repeat the question, do not add greetings, and do not describe your reasoning. Use plain language and common "
    "spellings of names and places."
)


def log(msg: str) -> None:
    print(f"[runner smoke] {msg}", flush=True)


def smoke_prompt(tok) -> tuple[list[int], str]:
    from models.demos.common.bringup.reference.prompt import template_kwargs
    from models.demos.common.bringup.testing.smoke import prompt_ids

    ids = prompt_ids(S, tok)
    if len(ids) > 2 * R.KV_BLOCK:
        return ids, "smoke prompt"
    msgs = [{"role": "system", "content": PAD_SYSTEM}, {"role": "user", "content": S.data["intake"]["smoke"]["prompt"]}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **template_kwargs(S))
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert len(ids) > 2 * R.KV_BLOCK, f"padded smoke prompt is {len(ids)} tokens, need > {2 * R.KV_BLOCK}"
    return ids, "padded smoke prompt (PAD_SYSTEM + smoke prompt)"


def open_mesh():
    import ttnn

    p = device_params(S)
    if p.get("fabric_config"):
        ttnn.set_fabric_config(p["fabric_config"])
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*S.mesh), l1_small_size=int(p.get("l1_small_size", 0)))
    return mesh, p


def close_mesh(mesh, p) -> None:
    import ttnn

    ttnn.close_mesh_device(mesh)
    if p.get("fabric_config"):
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def tail_next(model, state, layers, seq: list[int], cap: int) -> int:
    """The tail [cap, len(seq)) as one chunk at start=cap over the loaded prefix; greedy token at the last position.
    The chunk's rows sit in the server's layout for start=cap (server_rules.placement, device-major)."""
    n = len(seq)
    assert n - cap <= R.CHUNK, f"tail of {n - cap} tokens does not fit one {R.CHUNK}-token chunk"
    toks = torch.full((R.CHUNK,), R.PAD_ID, dtype=torch.int64)
    toks[: n - cap] = torch.tensor(seq[cap:], dtype=torch.int64)
    if hasattr(model, "begin_chunk"):
        model.begin_chunk(toks, cap)
    h = model.embed(toks, start=cap)
    for i in layers:
        h2 = model.layer(i, h, cap, state, end=n)
        model.free(h)
        h = h2
    hidden = model.final_norm(h)
    model.free(h)
    flat = [p for row in R.placement(cap) for p in row]
    nxt = int(model.logits(hidden, [flat.index(n - 1)]).float()[0].argmax())
    model.free(hidden)
    return nxt


def test_runner_smoke(tmp_path):
    if impl_mode() != "device":
        pytest.skip("device test")
    if os.environ.get("UP_FRONT_COLLECT") == "1":
        pytest.skip("precompile pass: the runner subprocess needs the chips; runs in the real pass only")
    layers = C.served_prefix()
    if layers != list(range(S.num_layers)):
        pytest.skip(
            f"partial run: the runner serves layers {layers[0]}-{layers[-1]} of {S.num_layers}; the smoke's answer needs "
            "every layer (run it where the whole model fits: serving_contract.md, 'Questions for the owner')"
        )
    from models.demos.common.bringup.core import metrics
    from models.demos.common.bringup.testing.smoke import record_answer, tokenizer
    from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract.test_runner_contract import (
        child_outcome,
        run_child,
        runner_env,
        use_engine,
    )

    t_all = time.time()
    R.self_check()
    R.harness()
    tok = tokenizer(S)
    ids, what = smoke_prompt(tok)
    n = len(ids)
    cap = R.reusable_prefix_cap(R.KV_BLOCK, n)
    log(f"{what}: prompt_len {n}, migration boundary {cap}, tail [{cap}, {n}); chunks {R.chunk_plan(n)}")

    # ---- 1. prefill through the real runner
    out = tmp_path
    trace = out / "trace"
    trace.mkdir()
    (trace / "metadata.json").write_text(json.dumps({"token_ids": ids}))
    env = runner_env(out, trace, len(layers), 1)
    env.update(
        PREFILL_PRODUCER_MAX_REQUESTS="1",
        PREFILL_PRODUCER_MULTI_TURN_PROB="0.0",
        DS41_CONTRACT_TURNS=json.dumps({"0": [n]}),
    )
    smoke_req = [{"name": "smoke", "token_ids": [int(t) for t in ids], "after": None}]
    feeder = use_engine(out, env, smoke_req, {}, 1, len(layers))
    log(f"feeder: {feeder}")
    t0 = time.time()
    rc, why = run_child(out, env, out / "runner.log")
    res, _, _ = child_outcome(out, env, rc, why, feeder)
    chunks = res.get("chunks", [])
    slot = chunks[0][0] if chunks else -1
    if [tuple(c[1:]) for c in chunks] != R.chunk_plan(n) or int(res.get("ends", {}).get(str(slot), -1)) != n:
        pytest.fail(f"runner chunks {chunks} / ends {res.get('ends')}, server plan {R.chunk_plan(n)}", pytrace=False)
    log(f"prefill done in {time.time() - t0:.0f} s on slot {slot}")

    # ---- 2. the KV that would migrate, through the exported table
    tfails = C.table_rules(str(out / "table.pb"), layers, 1)
    if tfails:
        pytest.fail("table: " + "; ".join(tfails[:10]), pytrace=False)
    migrated, records = {i: {} for i in layers}, 0
    for cfg in C.served_configs(layers):
        d = C.load_dump(out / "final", cfg, slot)
        for layer in cfg["layers"]:
            rows, _ = C.valid_and_pad(d, cfg, layer, 0, cap)
            if not torch.isfinite(torch.from_numpy(rows)).all():
                pytest.fail(f"{cfg['name']} layer {layer}: migrated rows [0, {cap}) not finite", pytrace=False)
            migrated[layer][cfg["name"]] = torch.from_numpy(rows.copy()).to(torch.bfloat16)
            records += cap // R.RECORD_TOKENS
    log(f"KV read: {records} records = [0, {cap}) of every served cache of {len(layers)} layers")

    # ---- 3. stand-in decode on the device
    mesh, p = open_mesh()
    try:
        model = S.hooks().device_model(mesh, S, layers, lm_head=True)
        state = model.new_state(R.MAX_SEQ)
        if not callable(getattr(state, "load_served", None)) or not callable(getattr(state, "load_tokens", None)):
            pytest.fail("not built: state.load_served(layer, tensors, length) / state.load_tokens(ids)", pytrace=False)
        for layer in layers:
            state.load_served(layer, migrated[layer], cap)
        state.load_tokens(torch.tensor(ids[:cap], dtype=torch.int64))
        eos = {t for t in (tok.eos_token_id,) if t is not None}
        expect = S.data["intake"]["smoke"]["expect"]
        out_ids = []
        for step in range(MAX_NEW):
            t0 = time.time()
            nxt = tail_next(model, state, layers, ids + out_ids, cap)
            log(f"tail [{cap}, {n + len(out_ids)}) -> {nxt} {tok.decode([nxt])!r} ({time.time() - t0:.1f} s)")
            if nxt in eos:
                break
            out_ids.append(nxt)
            if expect.lower() in tok.decode(out_ids).lower():
                break
        if hasattr(state, "free"):
            state.free()
    finally:
        close_mesh(mesh, p)
    text = tok.decode(out_ids)
    ok = expect.lower() in text.lower()
    log(f"answer {text!r} (expect {expect!r}): {'ok' if ok else 'FAIL'}; feeder {feeder}")
    metrics.record("smoke_runner_ok", int(ok))
    record_answer(
        "runner_smoke",
        "runner",
        S,
        text,
        out_ids,
        ok,
        time.time() - t_all,
        prompt_len=n,
        boundary=cap,
        records=records,
        prompt_variant=what,
        feeder=feeder,
    )
    assert ok, f"decoded answer {text!r} does not contain {expect!r} (prompt_len {n}, migrated [0, {cap}))"
