# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""1M-token prompts on the Galaxy rows, standalone model (#61419).

``--mode capacity``: build at ``--context`` + DECODE_ROOM for ``--users`` users per row and log DRAM per chip after
the weights and with the KV caches. Then run the last ``--chunk`` of a ``--context`` prompt in slot 0 of every row
(the earlier chunks only take their SWA blocks, the cache keeps zeros, as profile_prefill.py ``--fill skip``), twice,
and one decode step of every user, logging DRAM and the chunk time: the tightest point of a prefill and of decode.

``--mode passkey``: one passkey prompt of ``--context`` tokens per row in slot 0 of ``--rows`` rows (key at a different
depth on each row). The rows' chunks are enqueued round-robin, chunk by chunk, so every row has work in flight
(``prefill_enqueue`` of a whole prompt blocks once the row's command queue is full). Logs each row's TTFT, then
greedy-decodes ``--decode`` tokens from that cache and checks the key is in the text.

Every line to grep starts with ``MIMO_1M``.
"""

import argparse
import os
import time

import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.experimental.mimo_v2_6_flash.reference.checkpoint import MiMoCheckpoint
from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.tests.perf import moe_decode_common as common
from models.experimental.mimo_v2_6_flash.tests.test_model_long_tp8_dp4 import dram_gb, passkey_prompt
from models.experimental.mimo_v2_6_flash.tt.generator import MiMoGalaxyGenerator
from models.experimental.mimo_v2_6_flash.tt.model import TtMiMoModel, prefill_padded_len

DECODE_ROOM = 512
DEPTHS = (0.10, 0.35, 0.65, 0.90)


def say(msg):
    logger.info(msg)
    print(f"MIMO_1M {msg}", flush=True)


def build(mesh, args):
    cfg = MiMoConfig.from_model_path(common.MODEL_PATH)
    after_weights = {}
    setup_kv = TtMiMoModel.setup_kv_cache

    def measured(model, num_users):
        after_weights[id(model)] = dram_gb(model.mesh_device)[0]
        return setup_kv(model, num_users)

    TtMiMoModel.setup_kv_cache = measured
    try:
        gen = MiMoGalaxyGenerator(
            mesh,
            cfg,
            MiMoCheckpoint(common.MODEL_PATH, cfg),
            os.path.join(common.CACHE_ROOT, "model_bfp4experts"),
            dp=4,
            users_per_replica=args.users,
            max_seq_len=args.context + DECODE_ROOM,
            replicate_global_kv=args.replicate_global_kv,
        )
    finally:
        TtMiMoModel.setup_kv_cache = setup_kv
    for r, model in enumerate(gen.models):
        used, total = dram_gb(model.mesh_device)
        say(
            f"row {r} DRAM per chip: weights {after_weights[id(model)]:.2f} GB, with KV for {args.users} users x "
            f"{args.context + DECODE_ROOM} {used:.2f} GB, of {total:.2f} GB ({total - used:.2f} GB free)"
        )
    return cfg, gen


def capacity(gen, cfg, args):
    models = gen.models[: args.rows]
    start = args.context - args.chunk
    tokens = torch.randint(0, cfg.vocab_size, (args.chunk,), generator=torch.Generator().manual_seed(0))
    for model in models:
        for s in range(0, start, args.chunk):
            model.swa_table.prefill(0, s, s + args.chunk, s + prefill_padded_len(args.chunk))
    for attempt in range(2):
        t0 = time.time()
        pend = [m.prefill_enqueue(tokens, 0, start_pos=start, chunk_size=args.chunk) for m in models]
        for m, h in zip(models, pend):
            m.prefill_read(h)
        say(f"last chunk ({args.chunk} at {start}) on {len(models)} rows, run {attempt}: {time.time() - t0:.3f} s")
    for r, model in enumerate(models):
        say(f"row {r} DRAM per chip after the chunk: {dram_gb(model.mesh_device)[0]:.2f} GB")
    users = [r * args.users + s for r in range(len(models)) for s in range(args.users)]
    pos = args.context - 1
    for attempt in range(2):
        t0 = time.time()
        out = gen.decode({u: 1 for u in users}, {u: pos for u in users}, replicas=range(len(models)))
        say(f"decode step of {len(users)} users at position {pos}, run {attempt}: {time.time() - t0:.3f} s")
    assert all(torch.isfinite(out[u]).all() for u in users)
    for r, model in enumerate(models):
        say(f"row {r} DRAM per chip after decode: {dram_gb(model.mesh_device)[0]:.2f} GB")


def passkey(gen, cfg, args):
    tok = AutoTokenizer.from_pretrained(common.MODEL_PATH)
    models = gen.models[: args.rows]
    keys = [int(k) for k in torch.randint(10000, 100000, (4,), generator=torch.Generator().manual_seed(61419))]
    t0 = time.time()
    prompts = [passkey_prompt(tok, args.context, DEPTHS[r], keys[r]) for r in range(len(models))]
    say(f"{len(models)} passkey prompts of {args.context} tokens built in {time.time() - t0:.0f} s")

    n, chunk = args.context, args.chunk
    t0 = time.time()
    pending = {}
    for s in range(0, n, chunk):
        for r, model in enumerate(models):
            h = model.prefill_enqueue(prompts[r][s : s + chunk], 0, start_pos=s, chunk_size=chunk)
            if s + chunk < n:
                h["logits"].deallocate(True)
            else:
                pending[r] = h
        if s // chunk % 32 == 0:
            say(f"enqueued chunks to position {s + chunk} on {len(models)} rows at {time.time() - t0:.0f} s")
    say(f"host enqueue of every chunk done at {time.time() - t0:.1f} s")
    first, ttft = {}, {}
    for r, model in enumerate(models):
        first[r] = model.prefill_read(pending[r])[0]
        ttft[r] = time.time() - t0
        say(f"row {r} TTFT {ttft[r]:.1f} s ({n / ttft[r]:.0f} tokens/s), next token {first[r].argmax().item()}")
    for r, model in enumerate(models):
        say(f"row {r} DRAM per chip after prefill: {dram_gb(model.mesh_device)[0]:.2f} GB")

    users = [r * args.users for r in range(len(models))]
    gen_tokens = {u: [int(first[gen.locate(u)[0]].argmax())] for u in users}
    positions = {u: n for u in users}
    step_times = []
    for _ in range(args.decode - 1):
        t1 = time.time()
        out = gen.decode({u: t[-1] for u, t in gen_tokens.items()}, positions, replicas=range(len(models)))
        step_times.append(time.time() - t1)
        for u, lg in out.items():
            gen_tokens[u].append(int(lg.argmax()))
            positions[u] += 1
    found = {}
    for r in range(len(models)):
        text = tok.decode(gen_tokens[r * args.users])
        found[r] = str(keys[r]) in text
        say(f"row {r} key {keys[r]} at {DEPTHS[r]:.0%}: greedy {text!r} found {found[r]}")
    if step_times:
        say(f"decode steps (untraced, logits read): median {sorted(step_times)[len(step_times) // 2] * 1e3:.0f} ms")
    say(f"RESULT rows {len(models)} context {n} TTFT {[round(ttft[r], 1) for r in ttft]} found {found}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("capacity", "passkey"), required=True)
    parser.add_argument("--context", type=int, default=1 << 20)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--users", type=int, default=1, help="users per row the caches are sized for")
    parser.add_argument("--rows", type=int, default=1, help="rows that run (all 4 are built)")
    parser.add_argument("--replicate-global-kv", action="store_true")
    parser.add_argument("--decode", type=int, default=12, help="passkey: greedy tokens")
    args = parser.parse_args()
    assert args.context % args.chunk == 0

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=65536)
    try:
        cfg, gen = build(mesh, args)
        (capacity if args.mode == "capacity" else passkey)(gen, cfg, args)
    finally:
        for submesh in mesh.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
