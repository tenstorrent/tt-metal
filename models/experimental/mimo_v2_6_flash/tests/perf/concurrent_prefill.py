# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Concurrent long prefill from one host thread, eager vs traced later chunks (TtMiMoModel.setup_prefill_chunk_traces),
standalone model on the Galaxy rows (each a TP=8 replica; all 4 are built).

For each of ``--rows`` (e.g. "1,2,4") and each mode (eager, traced) every one of ``--users`` users on each of the
first R rows prefills its own ``--context``-token prompt (random ids) at the same time: one round enqueues one
``--chunk`` of every user, round-robin (as long_1m.py, but the users of a row also interleave). Each user's TTFT is
from the start of the run to its logits read. Per round: host enqueue time (the calls) and wall time, so the
bottleneck shows: eager rounds are bound by the host when the enqueue time is near the wall time. The traced run's last
logits are compared with the eager run's per user (the eager run hides the traces; both on the same model).

Every line to grep starts with ``MIMO_CC``.
"""

import argparse
import os
import time

import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.reference.checkpoint import MiMoCheckpoint
from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.tests.perf import moe_decode_common as common
from models.experimental.mimo_v2_6_flash.tt.generator import MiMoGalaxyGenerator
from models.experimental.mimo_v2_6_flash.tt.model import prefill_padded_len


def say(msg):
    logger.info(msg)
    print(f"MIMO_CC {msg}", flush=True)


def run(models, prompts, users, chunk, label):
    """Every (row, slot) user prefills its prompt, one chunk of each per round. Returns ({(r, s): logits}, ttft)."""
    n = prompts[(0, 0)].numel()
    keys = [(r, s) for r in range(len(models)) for s in range(users)]
    t0 = time.time()
    host, wall = [], []
    pending = {}
    for c in range(0, n, chunk):
        t1 = time.perf_counter()
        for r, s in keys:
            h = models[r].prefill_enqueue(prompts[(r, s)][c : c + chunk], s, start_pos=c, chunk_size=chunk)
            if c + chunk < n:
                if not h.get("persistent"):
                    h["logits"].deallocate(True)
            else:
                pending[(r, s)] = h
        t2 = time.perf_counter()
        host.append(t2 - t1)
        if wall:
            wall[-1] = t1 - wall[-1]
        wall.append(t1)
        if (c // chunk) % 32 == 31:
            k = min(32, len(wall) - 1)
            say(
                f"{label}: position {c + chunk}, last {k} rounds: host enqueue {1e3 * sum(host[-k - 1 : -1]) / k:.0f} ms, "
                f"wall {1e3 * sum(wall[-k - 1 : -1]) / k:.0f} ms per round"
            )
    out, ttft = {}, {}
    for r, s in keys:
        out[(r, s)] = models[r].prefill_read(pending[(r, s)])[0]
        ttft[(r, s)] = time.time() - t0
    wall = wall[:-1]
    # Rounds from the 1/4 point on: past the early, short chunks.
    q = len(wall) // 4
    tail_host = sum(host[q : len(wall)]) / max(1, len(wall) - q)
    tail_wall = sum(wall[q:]) / max(1, len(wall) - q)
    total = max(ttft.values())
    say(
        f"RESULT {label}: {len(keys)} users ({len(models)} rows x {users}) x {n} tokens: TTFT per user "
        f"min {min(ttft.values()):.1f} / max {total:.1f} s, {len(keys) * n / total:.0f} tok/s total, "
        f"{n / total:.0f} tok/s per user; rounds 1/4 .. end: host enqueue {1e3 * tail_host:.0f} ms, wall "
        f"{1e3 * tail_wall:.0f} ms per round ({len(keys)} chunks)"
    )
    return out, total


def skip_to(model, slot, start, chunk, pair):
    """The SWA blocks of ``slot``'s chunks before ``start`` from a new request (the cache keeps what it holds): a
    later chunk at ``start`` then runs as in a whole prompt (timing only: its keys are not the prompt's)."""
    for s in range(0, start, chunk):
        model.swa_table.prefill(slot, s, s + chunk, s + prefill_padded_len(chunk, s, pair))


def sampled(models, prompts, users, chunk, label, prefix, samples, rounds, pair):
    """Extrapolated time of a whole concurrent prefill: the rounds from 0 to ``prefix`` run as in ``run`` (measured),
    then ``rounds`` rounds at each start in ``samples`` (the earlier chunks skipped: SWA bookkeeping only), each
    giving a round time at its position; the rounds past ``prefix`` are integrated piecewise linearly between the
    prefix's last 8 rounds and the samples (flat past the last sample)."""
    n = prompts[(0, 0)].numel()
    keys = [(r, s) for r in range(len(models)) for s in range(users)]

    def rounds_from(start, count):
        times, host = [], []
        for c in range(start, start + count * chunk, chunk):
            t1 = time.perf_counter()
            hs = [
                models[r].prefill_enqueue(prompts[(r, s)][c : c + chunk], s, start_pos=c, chunk_size=chunk)
                for r, s in keys
            ]
            host.append(time.perf_counter() - t1)
            for h in hs:
                if not h.get("persistent"):
                    h["logits"].deallocate(True)
            times.append(t1)
        for m in models:
            ttnn.synchronize_device(m.mesh_device)
        times.append(time.perf_counter())
        return [b - a for a, b in zip(times, times[1:])], host

    t0 = time.time()
    walls, host = rounds_from(0, prefix // chunk)
    t_prefix = time.time() - t0
    say(
        f"{label}: measured {t_prefix:.1f} s for positions 0 .. {prefix} ({len(walls)} rounds; last 8 rounds wall "
        f"{1e3 * sum(walls[-8:]) / 8:.0f} ms, host {1e3 * sum(host[-8:]) / 8:.0f} ms)"
    )
    points = [(prefix - 4 * chunk, sum(walls[-8:]) / 8, sum(host[-8:]) / 8)]
    for start in samples:
        start = min(start, n - rounds * chunk)  # its rounds end inside the prompt
        for r, s in keys:
            skip_to(models[r], s, start, chunk, pair)
        w, h = rounds_from(start, rounds)
        w, h = w[1:], h[1:]  # the first round fills an empty queue
        points.append((start + rounds * chunk // 2, sum(w) / len(w), sum(h) / len(h)))
        say(
            f"{label}: {rounds} rounds at {start}: wall {1e3 * points[-1][1]:.0f} ms, host {1e3 * points[-1][2]:.0f} ms per round"
        )
    total = t_prefix
    for c in range(prefix, n, chunk):
        x = c + chunk // 2
        lo = max((p for p in points if p[0] <= x), key=lambda p: p[0])
        hi = [p for p in points if p[0] > x]
        if not hi:
            total += lo[1]
            continue
        hi = min(hi, key=lambda p: p[0])
        total += lo[1] + (hi[1] - lo[1]) * (x - lo[0]) / (hi[0] - lo[0])
    say(
        f"RESULT {label} EXTRAPOLATED: {len(keys)} users x {n} tokens: measured {t_prefix:.1f} s to {prefix}, "
        f"whole prompt ~{total:.1f} s, ~{len(keys) * n / total:.0f} tok/s total; round samples (pos, wall ms, host ms) "
        f"{[(p[0], round(1e3 * p[1]), round(1e3 * p[2])) for p in points]}"
    )
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=524288)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--rows", default="1,2,4")
    parser.add_argument("--users", default="1", help="users per row prefilling at once (a list: one run each)")
    parser.add_argument(
        "--prefix", type=int, default=0, help="eager only: extrapolate from a run to here (0: whole; traced runs whole)"
    )
    parser.add_argument("--samples", default="", help="--prefix: starts of the sampled late rounds")
    parser.add_argument("--sample-rounds", type=int, default=6)
    parser.add_argument("--save", default="prefill_logits", help="logits files prefix (its directory is created)")
    parser.add_argument("--modes", default="eager,traced")
    parser.add_argument("--trace-region-mb", type=int, default=400)
    kv = parser.add_mutually_exclusive_group()
    kv.add_argument("--replicate-global-kv", action="store_true")
    kv.add_argument("--kv-pair-split", action="store_true")
    args = parser.parse_args()
    row_counts = [int(r) for r in args.rows.split(",")]
    user_counts = [int(u) for u in args.users.split(",")]

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(4, 8), l1_small_size=65536, trace_region_size=args.trace_region_mb << 20
    )
    try:
        cfg = MiMoConfig.from_model_path(common.MODEL_PATH)
        gen = MiMoGalaxyGenerator(
            mesh,
            cfg,
            MiMoCheckpoint(common.MODEL_PATH, cfg),
            os.path.join(common.CACHE_ROOT, "model_bfp4experts"),
            dp=4,
            users_per_replica=max(user_counts),
            max_seq_len=args.context + 512,
            replicate_global_kv=args.replicate_global_kv,
            kv_pair_split=args.kv_pair_split,
        )
        models = gen.models[: max(row_counts)]
        prompts = {
            (r, s): torch.randint(
                0, cfg.vocab_size, (args.context,), generator=torch.Generator().manual_seed(1000 + 4 * s + r)
            )
            for r in range(len(models))
            for s in range(max(user_counts))
        }
        from models.experimental.mimo_v2_6_flash.tests.test_model_long_tp8_dp4 import dram_gb

        used, total = dram_gb(models[0].mesh_device)
        say(f"DRAM per chip with KV for {max(user_counts)} users x {args.context + 512}: {used:.2f} of {total:.2f} GB")
        # Compile every program eagerly first (one prompt through all rows), then the traces: nothing compiles after.
        t0 = time.time()
        warm = min(args.context, 65536 + 2 * args.chunk)
        for m in models:
            m.prefill_read(m.prefill_enqueue(prompts[(0, 0)][:warm], 0, chunk_size=args.chunk))
        padded = prefill_padded_len(args.chunk, args.chunk, args.kv_pair_split)
        for m in models:
            keys = m.setup_prefill_chunk_traces([padded], sdpa_chunks=(256,))
            m.warm_prefill_chunk_traces()
            m.capture_prefill_chunk_traces()
        saved = [m.chunk_traces for m in models]
        say(f"warm-up and traces {keys} in {time.time() - t0:.0f} s")
        results = {}
        for rows in row_counts:
            for users in user_counts:
                for mode in args.modes.split(","):
                    for m, traces in zip(models, saved):
                        m.chunk_traces = traces if mode == "traced" else {}
                    label = f"ISL {args.context} {rows} rows x {users} users (c={rows * users}) {mode}"
                    if args.prefix and mode == "eager":
                        samples = [int(x) for x in args.samples.split(",") if x]
                        sampled(
                            models[:rows],
                            prompts,
                            users,
                            args.chunk,
                            label,
                            args.prefix,
                            samples,
                            args.sample_rounds,
                            args.kv_pair_split,
                        )
                        continue
                    out, total = run(models[:rows], prompts, users, args.chunk, label)
                    path = f"{args.save}_{args.context}_{rows}x{users}_{mode}.pt"
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                    torch.save({k: v.float() for k, v in out.items()}, path)
                    other = f"{args.save}_{args.context}_{rows}x{users}_{'eager' if mode == 'traced' else 'traced'}.pt"
                    if os.path.exists(other):
                        ref = torch.load(other)
                        d = max(float((ref[k] - out[k].float()).abs().max()) for k in out)
                        top = all(int(ref[k].argmax()) == int(out[k].argmax()) for k in out)
                        say(f"CHECK {label} vs {other}: logits max|d| {d:.4g}, top1 equal {top}")
    finally:
        for submesh in mesh.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
