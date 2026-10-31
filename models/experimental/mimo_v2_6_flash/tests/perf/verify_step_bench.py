# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode step time of one row engine through the vLLM adapter (generator_vllm.MiMoV2ForCausalLM), without vLLM:
the steps a row engine runs at long context with 1..4 requests decoding, MTP speculation (MIMO_SPEC_SLOTS /
MIMO_SPEC_GROUPS as in serving) or plain decode (``--modes``).

Run on one Galaxy row as a row engine does: TT_VISIBLE_DEVICES=<the row's chips>, a (1, 8) mesh, FABRIC_1D_RING. Each
request gets its own prompt of ``--ctx`` tokens of real text (source files of this checkout: varied routing and
natural acceptance) and its own blocks, prefilled through prefill_forward. Then for each ``--users`` count the first
n requests decode ``--steps`` steps: a verify (fused_sample, greedy) and the chained propose per step, synchronously,
or a traced plain decode step. Every step's wall time is measured; the first ``--warm`` are dropped.

    VSB users=<n> mode=<spec K | plain> step_ms=<median> mean=<mean> tokens_per_step=<per request> tok_s=<per request>

Every line to grep starts with ``VSB``.
"""

import argparse
import glob
import os
import statistics
import time
from types import SimpleNamespace

os.environ.setdefault("MIMO_PREFILL_BATCH", "0")

import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.reference.checkpoint import default_model_path
from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.reference.spec_decode import MAX_DRAFTS
from models.experimental.mimo_v2_6_flash.tests.perf.vllm_adapter_prefill import SWA_RING, vllm_kv_layout
from models.experimental.mimo_v2_6_flash.tt import generator_vllm
from models.experimental.mimo_v2_6_flash.tt.generator_vllm import BLOCK_SIZE, MiMoV2ForCausalLM

CORPUS = "ttnn/cpp/ttnn/operations/**/*.[hc]pp"
# Under tracy (TT_METAL_DEVICE_PROFILER=1) the device's profiler buffer is read after every step: a verify replays
# thousands of programs, more than the buffer holds.
PROFILING = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"


def say(msg):
    logger.info(msg)
    print(f"VSB {msg}", flush=True)


def prompts(path, n, users):
    """``users`` prompts of ``n`` tokens of this checkout's source text, each from its own stretch of files."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    files = sorted(glob.glob(CORPUS, recursive=True))
    out, i = [], 0
    for u in range(users):
        ids = []
        while len(ids) < n:
            ids += tok.encode(open(files[i % len(files)], errors="ignore").read(), add_special_tokens=False)
            i += 1
        out.append(torch.tensor(ids[:n], dtype=torch.int32))
    return out


def breakdown(m, n=20):
    """Each captured trace replayed ``n`` times on idle inputs (every position -1: no K / V written, no request's
    work): the fixed cost of each program of a step."""
    mesh = m.mesh_device  # the row's submesh, which owns the traces
    m.spec_inputs.reset()
    traces = []
    if m.decode_trace is not None:
        traces.append(("decode", m.decode_trace["id"]))
    for key, tr in m.verify_traces.items():
        traces.append((f"verify {key}", tr["id"]))
        if tr["accept"] is not None:
            traces.append((f"accept {key}", tr["accept"]))
    mtp = m.mtp
    if mtp is not None:
        traces.append(("mtp draft", mtp.trace["id"]))
        if getattr(mtp, "trace_one", None) is not None:
            traces.append(("mtp draft layer 0", mtp.trace_one["id"]))
        if mtp.backfill_trace is not None:
            mtp.history.upload_backfill()
            traces.append(("mtp backfill", mtp.backfill_trace))
    for name, tid in traces:
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        say(f"breakdown {name}: {1e3 * (time.perf_counter() - t0) / n:.2f} ms idle")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, default=150_000)
    p.add_argument("--users", default="1,2,3,4")
    p.add_argument("--steps", type=int, default=96)
    p.add_argument("--warm", type=int, default=16)
    p.add_argument("--per-lane", type=int, default=8, help="--max-num-seqs")
    p.add_argument("--budget", type=int, default=4096, help="prefill tokens per step")
    p.add_argument("--modes", default="spec,plain", help="spec (verify + propose) and / or plain decode steps")
    p.add_argument("--k", type=int, default=MAX_DRAFTS, help="drafts per request (at most MAX_DRAFTS)")
    p.add_argument("--breakdown", action="store_true", help="also time each captured trace alone on idle inputs")
    a = p.parse_args()
    users_list = [int(x) for x in a.users.split(",")]
    users = max(users_list)
    assert users <= a.per_lane and 1 <= a.k <= MAX_DRAFTS
    path = default_model_path()
    cfg = MiMoConfig.from_model_path(path)
    pair = generator_vllm.KV_PAIR_SPLIT
    groups, buffers = vllm_kv_layout(cfg)
    MiMoV2ForCausalLM._layer_groups = groups
    MiMoV2ForCausalLM.prefill_token_budget = a.budget
    max_seq_len = -(-(a.ctx + 8192) // 128) * 128  # whole blocks of the pair-split global group

    global_span = 2 * BLOCK_SIZE if pair else BLOCK_SIZE
    global_blocks = -(-max_seq_len // global_span)
    num_groups = max(groups) + 1
    swa_groups = sorted({g for i, g in enumerate(groups) if cfg.is_swa(i)})
    per_request = global_blocks + len(swa_groups) * SWA_RING
    num_blocks = max(1 + users * per_request, -(-max_seq_len // BLOCK_SIZE) + 1)
    rows = a.per_lane
    width = -(-max_seq_len // BLOCK_SIZE)
    tables = {g: torch.zeros(rows, width, dtype=torch.int32) for g in range(num_groups)}
    for u in range(users):
        for g in range(num_groups):
            if g in swa_groups:
                base = 1 + u * per_request + global_blocks + swa_groups.index(g) * SWA_RING
                tables[g][u] = base + torch.arange(width, dtype=torch.int32) % SWA_RING
            else:
                tables[g][u, :global_blocks] = 1 + u * per_request + torch.arange(global_blocks, dtype=torch.int32)
    layer_tables = [tables[g] for g in groups]

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 8),
        l1_small_size=65536,
        trace_region_size=1_200_000_000,
        dispatch_core_config=ttnn.DispatchCoreConfig(axis=ttnn.DispatchCoreAxis.COL),
    )
    try:
        t0 = time.time()
        hf = SimpleNamespace(num_hidden_layers=cfg.num_hidden_layers, _name_or_path=path)
        model = MiMoV2ForCausalLM.initialize_vllm_model(hf, mesh, rows, max_seq_len, tt_data_parallel=1)
        specs = []
        for i in range(cfg.num_hidden_layers):
            dims = cfg.attention_dims(i)
            block = 2 * BLOCK_SIZE if pair and not cfg.is_swa(i) else BLOCK_SIZE
            specs.append(((num_blocks, 1, block, dims.head_dim), torch.bfloat16, buffers[i]))
        kv = model.allocate_kv_cache_per_layer(specs)
        model.warmup_model_prefill(kv)
        model.warmup_model_decode(kv, enable_trace=False)
        model.warmup_model_decode(kv, enable_trace=True)
        say(f"built and warmed up in {time.time() - t0:.0f} s, {num_blocks} blocks")

        texts = prompts(path, a.ctx, users)
        last = []
        t0 = time.time()
        for u in range(users):
            pos = 0
            while pos < a.ctx:
                end = min(pos + a.budget, a.ctx)
                logits = model.prefill_forward(
                    texts[u][None],
                    torch.tensor([pos], dtype=torch.int32),
                    torch.tensor([end], dtype=torch.int32),
                    tables[0][u : u + 1],
                    kv,
                    empty_slots=[u],
                    page_tables_per_layer=[t[u : u + 1] for t in layer_tables],
                )
                pos = end
            last.append(int(logits[0, -1].argmax()))
        say(f"prefilled {users} x {a.ctx} tokens in {time.time() - t0:.0f} s")
        if PROFILING:
            ttnn.ReadDeviceProfiler(mesh)

        k = a.k
        for plain, n in [(m == "plain", n) for m in a.modes.split(",") for n in users_list]:
            # Each run continues every request from where the last run left it (its own KV, its own positions).
            # pos[r]: the position of request r's last committed token tok[r] (not yet in the cache).
            pos = getattr(main, "_pos", [a.ctx] * users)
            tok = getattr(main, "_tok", list(last))
            times, per_step_tokens = [], []
            if plain:
                for s in range(a.steps):
                    tokens = torch.zeros(rows, 1, dtype=torch.int32)
                    positions = torch.full((rows,), -1, dtype=torch.int32)
                    for r in range(n):
                        tokens[r, 0], positions[r] = tok[r], pos[r]
                    t1 = time.perf_counter()
                    logits = model.decode_forward(
                        tokens, positions, tables[0], kv, enable_trace=True, page_tables_per_layer=layer_tables
                    )
                    times.append(time.perf_counter() - t1)
                    if PROFILING:
                        ttnn.ReadDeviceProfiler(mesh)
                    for r in range(n):
                        tok[r], pos[r] = int(logits[r, 0].argmax()), pos[r] + 1
                    per_step_tokens.append(1.0)
                mode = "plain"
            else:
                # A plain step (the drafter reads its resident hidden), the first propose, then verify + propose.
                tokens = torch.zeros(rows, 1, dtype=torch.int32)
                positions = torch.full((rows,), -1, dtype=torch.int32)
                for r in range(n):
                    tokens[r, 0], positions[r] = tok[r], pos[r]
                logits = model.decode_forward(
                    tokens, positions, tables[0], kv, enable_trace=True, page_tables_per_layer=layer_tables
                )
                committed = torch.zeros(rows, 1 + MAX_DRAFTS, dtype=torch.int32)
                cpos = torch.full((rows, 1 + MAX_DRAFTS), -1, dtype=torch.int32)
                ones = torch.ones(rows, dtype=torch.int32)
                for r in range(n):
                    tok[r], pos[r] = int(logits[r, 0].argmax()), pos[r] + 1
                    committed[r, 0], cpos[r, 0] = tok[r], pos[r]
                drafts = model.propose_draft_tokens(k, committed, cpos, ones)
                for s in range(a.steps):
                    tokens = torch.zeros(rows, 1 + k, dtype=torch.int32)
                    positions = torch.full((rows, 1 + k), -1, dtype=torch.int32)
                    valid = torch.zeros(rows, dtype=torch.int32)
                    for r in range(n):
                        nv = int(drafts.num_valid[r]) if drafts.num_valid is not None else k
                        tokens[r, 0] = tok[r]
                        tokens[r, 1 : 1 + nv] = drafts.draft_token_ids[r, :nv]
                        positions[r, : 1 + nv] = torch.arange(pos[r], pos[r] + 1 + nv, dtype=torch.int32)
                        valid[r] = nv
                    t1 = time.perf_counter()
                    out = model.decode_forward(
                        tokens,
                        positions,
                        tables[0],
                        kv,
                        page_tables_per_layer=layer_tables,
                        spec_mode="fused_sample",
                        num_valid_drafts=valid,
                        accepted_counts=ones,
                    )
                    counts = out.accepted_counts.to(torch.int32).clone()
                    ids = out.accepted_token_ids
                    committed = torch.zeros(rows, 1 + MAX_DRAFTS, dtype=torch.int32)
                    cpos = torch.full((rows, 1 + MAX_DRAFTS), -1, dtype=torch.int32)
                    for r in range(n):
                        c = int(counts[r])
                        committed[r, :c] = ids[r, :c]
                        cpos[r, :c] = torch.arange(pos[r] + 1, pos[r] + 1 + c, dtype=torch.int32)
                    drafts = model.propose_draft_tokens(k, committed, cpos, counts)
                    times.append(time.perf_counter() - t1)
                    if PROFILING:
                        ttnn.ReadDeviceProfiler(mesh)
                    got = []
                    for r in range(n):
                        c = int(counts[r])
                        tok[r], pos[r] = int(ids[r, c - 1]), pos[r] + c
                        got.append(c)
                    counts[n:] = 1
                    per_step_tokens.append(sum(got) / n)
                mode = f"spec{k}"
            main._pos, main._tok = pos, tok
            t = times[a.warm :]
            tps = statistics.mean(per_step_tokens[a.warm :])
            med = statistics.median(t)
            say(
                f"users={n} mode={mode} step_ms={1e3 * med:.2f} mean={1e3 * statistics.mean(t):.2f} "
                f"tokens_per_step={tps:.2f} tok_s={tps / statistics.mean(t):.1f}"
            )
        if a.breakdown:
            breakdown(model.models[0])
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
