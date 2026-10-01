# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving contract, runner smoke: disaggregated serving end to end, with the bring-up device model standing in for
decode (serving_contract.md, "Adapter and table").

1. Prefill: tt-metal's real prefill runner (runner_case.py, as test_runner_contract.py runs it: runner + producer
   subprocess, FABRIC_2D, 1 slot, D2H layer acks, mock migration on the migration-enabled path) prefills one request,
   the spec's `intake.smoke.prompt` wrapped by the chat template with `text.template_kwargs` (smoke.py prompt_ids).
   That prompt is 16 tokens: below 64 the migration boundary is 0 and nothing would migrate, so the test uses the
   padded variant: the fixed neutral system message PAD_SYSTEM before the same user turn (135 tokens).
2. Migration: after the runner exits, the KV it dumped through the exported table over UMD (engine.TableReader, what
   a KV Manager ships) is decoded for every layer up to the boundary decode asks for,
   cap = (prompt_len - 1) // 64 * 64 (prefix_indexer.hpp:75-78 reusable_prefix_cap, backend_runtime.cpp Inbound).
3. Stand-in decode: a fresh mesh (spec box.device_params), hooks.device_model(..., lm_head=True), every layer's
   state.load_prefix(layer, {"kv_latent": migrated}, cap) (the ladder's prefix_from_golden path), the tail
   [cap, prompt_len) recomputed on top as one chunk at start=cap (as decode recomputes its last block), final_norm +
   logits at the last position, greedy. Each new token is appended and the tail recomputed, until EOS or 8 tokens.
Pass: the table has the geometry tables.py knows, and the decoded answer contains `intake.smoke.expect` ("Paris").
Fail fast: test_runner_contract.run_child (bounded waits, process-group kill). The precompile pass of
run_safe_pytest (UP_FRONT_COLLECT=1) skips the body: the parent opens the mesh for step 3 and must not hold the chips
while a runner subprocess wants them.
"""

import json
import os
import time

import numpy as np
import pytest
import torch

from models.demos.common.bringup.testing.harness import device_params, device_timeout, impl_mode, spec
from models.demos.xing40_a4b_d_p.tests.bringup.contract import engine as E
from models.demos.xing40_a4b_d_p.tests.bringup.contract import server_rules as R

S = spec()
pytestmark = device_timeout(S)
MAX_NEW = 8
# Prepended as a system turn when the smoke prompt alone is too short to migrate anything (boundary 0). Neutral: it
# names no place and no answer; it only asks for short, direct replies.
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
    """smoke.prompt_ids; the padded variant (PAD_SYSTEM first) when it would migrate nothing."""
    from models.demos.common.bringup.reference.prompt import template_kwargs
    from models.demos.common.bringup.testing.smoke import prompt_ids

    ids = prompt_ids(S, tok)
    if R.reusable_prefix_cap(R.KV_BLOCK, len(ids)) > 0:
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
    """The tail [cap, len(seq)) as one chunk at start=cap over the loaded prefix; greedy token at the last position."""
    from models.demos.xing40_a4b_d_p.tt.layout import server_order

    n = len(seq)
    assert n - cap <= R.CHUNK, f"tail of {n - cap} tokens does not fit one {R.CHUNK}-token chunk"
    toks = torch.full((R.CHUNK,), R.PAD_ID, dtype=torch.int64)
    toks[: n - cap] = torch.tensor(seq[cap:], dtype=torch.int64)
    h = model.embed(toks, start=cap)
    for i in layers:
        h2 = model.layer(i, h, cap, state, end=n)
        model.free(h)
        h = h2
    hidden = model.final_norm(h)
    model.free(h)
    # embed laid the chunk out as the server does for start=cap: host row k holds natural index order[k]
    order = server_order(cap, R.CHUNK, R.SP) if cap % R.CHUNK else torch.arange(R.CHUNK)
    row = int((order == n - 1 - cap).nonzero().flatten()[0])
    nxt = int(model.logits(hidden, [row]).float()[0].argmax())
    model.free(hidden)
    return nxt


def test_runner_smoke(tmp_path):
    if impl_mode() != "device":
        pytest.skip("device test")
    if os.environ.get("UP_FRONT_COLLECT") == "1":
        # run_safe_pytest's precompile pass runs each body with the device ops neutered; a mesh this process opened
        # there keeps the chips locked, and the real pass's runner subprocess then waits forever for them.
        pytest.skip("precompile pass: the runner subprocess needs the chips; runs in the real pass only")
    from models.demos.common.bringup.testing.smoke import tokenizer
    from models.demos.xing40_a4b_d_p.tests.bringup.contract.test_runner_contract import run_child, runner_env, tail

    t_all = time.time()
    R.self_check()
    tables, kvc = R.harness()
    tok = tokenizer(S)
    ids, what = smoke_prompt(tok)
    n = len(ids)
    cap = R.reusable_prefix_cap(R.KV_BLOCK, n)
    log(
        f"{what}: prompt_len {n}, migration boundary {cap} (= ({n} - 1) // {R.KV_BLOCK} * {R.KV_BLOCK}), tail [{cap}, {n})"
    )
    log(f"chunks the server sends: {R.chunk_plan(n)}")

    # ---- 1. prefill through the real runner
    out = tmp_path
    trace = out / "trace"
    trace.mkdir()
    (trace / "metadata.json").write_text(json.dumps({"token_ids": ids}))
    env = runner_env(out, trace)
    env.update(
        PREFILL_NUM_USERS="1",
        PREFILL_PRODUCER_MAX_REQUESTS="1",
        PREFILL_PRODUCER_MULTI_TURN_PROB="0.0",
        XING_CONTRACT_TURNS=json.dumps({"0": [n]}),
    )
    runner_log = out / "runner.log"
    log(f"runner up: prefill_runner + producer for 1 request ({runner_log})")
    t0 = time.time()
    rc, why = run_child(out, env, runner_log)
    if why:
        pytest.fail(
            f"{why}; see {runner_log} and {out / 'producer.log'}\n{tail(out / 'producer.log', 20)}\n{tail(runner_log)}",
            pytrace=False,
        )
    if (out / "not_built.txt").exists():
        pytest.fail(f"not built: {(out / 'not_built.txt').read_text()}", pytrace=False)
    res = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
    pj = json.loads((out / "producer.json").read_text()) if (out / "producer.json").exists() else {}
    if rc != 0 or res.get("errors") or pj.get("errors"):
        pytest.fail(
            f"runner exit {rc}; runner errors {res.get('errors')}; producer errors {pj.get('errors')}\n"
            f"see {runner_log} and {out / 'producer.log'}\n{res.get('traceback', '')}\n{tail(runner_log)}",
            pytrace=False,
        )
    log(
        f"prefill done in {time.time() - t0:.0f} s: ends {res.get('ends')}, acks {pj.get('acks_drained')}/"
        f"{pj.get('acks_expected')}"
    )
    if int(res.get("ends", {}).get("0", -1)) != n:
        pytest.fail(f"slot 0 ended at {res.get('ends')}, expected {n}", pytrace=False)

    # ---- 2. the KV that would migrate, through the exported table
    tfails, geom = E.table_rules(str(out / "table.pb"), R.NUM_LAYERS, 1)
    if not geom:
        pytest.fail("table: " + "; ".join(tfails), pytrace=False)
    gk = {k: geom[k] for k in ("dtype", "width", "storage")}
    cbytes = geom["width"] // 32 * (2048 if geom["dtype"] == "bf16" else 1088)
    d = kvc.load_dump(str(out / "final"), slot=0, chunk_bytes=cbytes)
    layers = list(range(R.NUM_LAYERS))
    migrated = {}
    for layer in layers:
        kv = kvc.reassemble(d, layer, 0, cap, **gk)
        assert kv.shape == (cap, R.MLA_WIDTH), f"layer {layer}: migrated KV {kv.shape}, expected ({cap}, {R.MLA_WIDTH})"
        if not np.isfinite(kv).all():
            pytest.fail(f"layer {layer}: migrated KV [0, {cap}) has non-finite values", pytrace=False)
        migrated[layer] = torch.from_numpy(np.ascontiguousarray(kv, dtype=np.float32))
    log(
        f"KV read: {len(layers) * cap // R.RECORD_TOKENS} records ({cap // R.RECORD_TOKENS} per layer x {len(layers)} "
        f"layers, {geom['dtype']} {geom['storage']}, {cbytes} B each) = [0, {cap}) of every layer"
    )
    if tfails:
        log("table rule notes: " + "; ".join(tfails))

    # ---- 3. stand-in decode on the device
    mesh, p = open_mesh()
    try:
        t0 = time.time()
        model = S.hooks().device_model(mesh, S, layers, lm_head=True)
        log(f"device model up: {len(layers)} layers + LM head in {time.time() - t0:.0f} s")
        state = model.new_state(R.MAX_SEQ)
        for layer in layers:
            state.load_prefix(layer, {"kv_latent": migrated[layer]}, cap)
        log(f"prefix loaded: migrated KV [0, {cap}) into every layer's state")
        eos = {t for t in (tok.eos_token_id,) if t is not None}
        out_ids = []
        for step in range(MAX_NEW):
            seq = ids + out_ids
            t0 = time.time()
            nxt = tail_next(model, state, layers, seq, cap)
            log(
                f"tail recompute [{cap}, {len(seq)}) -> token {step}: {nxt} {tok.decode([nxt])!r} "
                f"({time.time() - t0:.1f} s)"
            )
            if nxt in eos:
                break
            out_ids.append(nxt)
        if hasattr(state, "free"):
            state.free()
    finally:
        close_mesh(mesh, p)
    text = tok.decode(out_ids)
    expect = S.data["intake"]["smoke"]["expect"]
    ok = expect.lower() in text.lower()
    log(
        f"answer {text!r} (expect {expect!r}): {'ok' if ok else 'FAIL'}; prompt_len {n}, boundary {cap}, "
        f"total {time.time() - t_all:.0f} s"
    )
    assert ok, f"decoded answer {text!r} does not contain {expect!r} (prompt_len {n}, migrated [0, {cap}))"
