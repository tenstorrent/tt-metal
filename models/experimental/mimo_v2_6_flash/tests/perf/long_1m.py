# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""1M-token prompts on the Galaxy rows, standalone model (#61419). Results: doc/tp8_dp4_plan.md, "1M prefill on all 4
rows".

``--mode capacity``: build at ``--context`` + DECODE_ROOM for ``--users`` users per row and log DRAM per chip after
the weights and with the KV caches. Then run the last ``--chunk`` of a ``--context`` prompt in slot 0 of ``--rows``
rows twice (the earlier chunks only take their SWA blocks and the cache keeps zeros, as profile_prefill.py
``--fill skip``), and one decode step of every user, logging DRAM and the times: the tightest point of a prefill and
of decode.

``--mode passkey``: one passkey prompt of ``--context`` tokens in each of the ``--users`` slots of ``--rows`` rows, each
with its own key and depth (``depth``). The slots of a row run one after the other. The rows' chunks are enqueued
round-robin, chunk by chunk, so every row has work in flight: ``prefill_enqueue`` of a whole prompt blocks once the
row's command queue is full. Logs each user's TTFT from the start of the run (the first chunk's compile included),
then greedy-decodes ``--decode`` tokens of every user from that cache (untraced) and checks each key is in the text.

``--replicate-global-kv`` keeps the global layers' K / V on both chips of a pair, as vLLM's caches do with
MIMO_KV_PAIR_SPLIT=0 (a later chunk splits the heads over the pair). ``--kv-pair-split`` splits every 128-position
global block over the pair instead, each chip keeping its halves, as vLLM's caches do by default (a later chunk runs
all 16 heads over the chip's half of the keys and merges by LSE: tt/pair_split.py).

    python models/experimental/mimo_v2_6_flash/tests/perf/long_1m.py --mode passkey --rows 4 --users 2 \\
        --replicate-global-kv

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

DECODE_ROOM = 512  # positions after the prompt, for decode
DEPTHS = (0.10, 0.35, 0.65, 0.90)  # where the pass keys sit (``depth``)
KEY_SEED = 61419


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
            kv_pair_split=args.kv_pair_split,
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


def depth(row, slot):
    """Where the key of ``slot`` on ``row`` sits: slot 0 at DEPTHS[row], slot 1 at DEPTHS[3 - row], so 4 rows of 2 users
    cover every depth twice; later slots shift round."""
    first = row if slot % 2 == 0 else len(DEPTHS) - 1 - row
    return DEPTHS[(first + slot // 2) % len(DEPTHS)]


def passkey(gen, cfg, args):
    tok = AutoTokenizer.from_pretrained(common.MODEL_PATH)
    models = gen.models[: args.rows]
    slots = range(args.users)
    # 4 users on each of 4 rows at most, slot-major: a user's key does not depend on the rows or users run.
    assert args.users <= 4
    draws = torch.randint(10000, 100000, (16,), generator=torch.Generator().manual_seed(KEY_SEED))
    key = {(r, s): int(draws[s * 4 + r]) for r in range(len(models)) for s in slots}
    t0 = time.time()
    prompts = {(r, s): passkey_prompt(tok, args.context, depth(r, s), key[(r, s)]) for (r, s) in key}
    say(f"{len(prompts)} passkey prompts of {args.context} tokens built in {time.time() - t0:.0f} s")

    n, chunk = args.context, args.chunk
    t0 = time.time()
    first, ttft = {}, {}
    for s in slots:
        pending = {}
        for c in range(0, n, chunk):
            for r, model in enumerate(models):
                h = model.prefill_enqueue(prompts[(r, s)][c : c + chunk], s, start_pos=c, chunk_size=chunk)
                if c + chunk < n:
                    if not h.get("persistent"):  # a replayed chunk's logits row is the slot's own
                        h["logits"].deallocate(True)
                else:
                    pending[r] = h
            if c // chunk % 64 == 0:
                say(
                    f"slot {s}: enqueued chunks to position {c + chunk} on {len(models)} rows at {time.time() - t0:.0f} s"
                )
        for r, model in enumerate(models):
            first[(r, s)] = model.prefill_read(pending[r])[0]
            ttft[(r, s)] = time.time() - t0
            say(
                f"row {r} slot {s} TTFT {ttft[(r, s)]:.1f} s from the start of the run, next token "
                f"{first[(r, s)].argmax().item()}"
            )
    for r, model in enumerate(models):
        say(f"row {r} DRAM per chip after prefill: {dram_gb(model.mesh_device)[0]:.2f} GB")

    users = {r * args.users + s: (r, s) for (r, s) in key}
    gen_tokens = {u: [int(first[rs].argmax())] for u, rs in users.items()}
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
    for u, (r, s) in sorted(users.items()):
        text = tok.decode(gen_tokens[u])
        found[(r, s)] = str(key[(r, s)]) in text
        say(f"row {r} slot {s} key {key[(r, s)]} at {depth(r, s):.0%}: greedy {text!r} found {found[(r, s)]}")
    if step_times:
        say(f"decode steps (untraced, logits read): median {sorted(step_times)[len(step_times) // 2] * 1e3:.0f} ms")
    say(
        f"RESULT rows {len(models)} users/row {args.users} context {n} "
        f"TTFT {[(r, s, round(t, 1)) for (r, s), t in sorted(ttft.items())]} found {sorted(found.items())}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("capacity", "passkey"), required=True)
    parser.add_argument("--context", type=int, default=1 << 20)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--users", type=int, default=1, help="users per row the caches are sized for")
    parser.add_argument("--rows", type=int, default=1, help="rows that run (all 4 are built)")
    kv = parser.add_mutually_exclusive_group()
    kv.add_argument(
        "--replicate-global-kv",
        action="store_true",
        help="both chips of a global layer's pair keep the K / V (vLLM with MIMO_KV_PAIR_SPLIT=0)",
    )
    kv.add_argument(
        "--kv-pair-split", action="store_true", help="each chip of the pair keeps half of every 128-token block (vLLM)"
    )
    parser.add_argument("--decode", type=int, default=12, help="passkey: greedy tokens")
    parser.add_argument(
        "--trace-chunks",
        action="store_true",
        help="later chunks replay a trace of the whole chunk (TtMiMoModel.setup_prefill_chunk_traces)",
    )
    parser.add_argument("--trace-region-mb", type=int, default=400)
    args = parser.parse_args()
    assert args.context % args.chunk == 0

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    trace = {"trace_region_size": args.trace_region_mb << 20} if args.trace_chunks else {}
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=65536, **trace)
    try:
        cfg, gen = build(mesh, args)
        if args.trace_chunks:
            padded = prefill_padded_len(args.chunk, args.chunk, args.kv_pair_split)
            t0 = time.time()
            for model in gen.models[: args.rows]:
                keys = model.setup_prefill_chunk_traces([padded])
                model.warm_prefill_chunk_traces()
                model.capture_prefill_chunk_traces()
            say(f"later-chunk traces {keys} on {args.rows} rows in {time.time() - t0:.0f} s")
        (capacity if args.mode == "capacity" else passkey)(gen, cfg, args)
    finally:
        for submesh in mesh.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
