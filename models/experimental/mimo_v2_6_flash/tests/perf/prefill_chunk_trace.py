# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host time of a later prefill chunk, and the traced later chunks (TtMiMoModel.setup_prefill_chunk_traces), on one
Galaxy row (each row a TP=8 replica; all 4 are built, row 0 runs).

``--breakdown``: at each of ``--starts`` (the earlier chunks only take their SWA blocks and the cache keeps its zeros, as
profile_prefill.py ``--fill skip``) one eager chunk, timed by phase on the host (SWA bookkeeping and its table copy,
the block tables, RoPE, embedding, each layer kind's enqueue, the last-token logits), then the wait for the device.
Each chunk starts on an idle device. With ``--trace`` the same chunks then replay the trace: host time of the replay
(input copies) and the device time. The eager and traced logits of each chunk are compared.

``--ttft``: whole prompts of each of ``--contexts`` tokens (random ids, seed 0), eager then traced (``--trace``), each
the second of two runs unless ``--once``; the traced prompt's logits are compared with the eager one's.

``--profile``: cProfile of one eager chunk at the first start (top functions by own time).

Every line to grep starts with ``MIMO_CT``.
"""

import argparse
import cProfile
import io
import os
import pstats
import time
from collections import defaultdict

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
    print(f"MIMO_CT {msg}", flush=True)


def timed_chunk(model, tokens, slot, start):
    """TtMiMoModel._prefill_chunk + the last-token logits of prefill_enqueue, with host timers per phase. Returns
    (phases {name: s}, logits [vocab])."""
    clock = time.perf_counter
    ph = defaultdict(float)
    t = clock()

    def lap(name):
        nonlocal t
        now = clock()
        ph[name] += now - t
        t = now

    n = tokens.numel()
    padded = prefill_padded_len(n, start, model.pair is not None)
    layers = range(len(model.layers))
    if model.swa_table is not None:
        model.swa_table.prefill(slot, start, start + n, start + padded)
        lap("swa_bookkeeping")
        model._sync_swa_table()
        lap("swa_table_copy")
    tables, start_t = model._prefill_tables(slot, start, padded, layers)
    lap("block_tables")
    for setup in model.rope_setups.values():
        setup.begin_prefill_chunk()
        setup.get_prefill_rot_mats(start, padded)
    lap("rope")
    x = model._embed(torch.nn.functional.pad(tokens.to(torch.int32), (0, padded - n)))
    lap("embed")
    with model._prefill_reduce(padded):
        for i in layers:
            layer = model.layers[i]
            y = layer.forward_prefill(
                x,
                slot,
                model._layer_page_table(i),
                model.kv_caches[i],
                real_len=n,
                start_pos=start,
                tables=tables[model._table_key(i)],
            )
            x.deallocate(True)
            x = y
            lap(f"layers_{'swa' if layer.is_swa else 'global'}_{'moe' if layer.is_moe else 'dense'}")
    for setup in model.rope_setups.values():
        setup.end_prefill_chunk()
    for tb in tables.values():
        tb.release()
    if start_t is not None:
        start_t.deallocate(True)
    lap("release")
    tile_start = (n - 1) // 32 * 32
    from models.experimental.mimo_v2_6_flash.tt import kv as kv_lib

    at = kv_lib.replicated(model.plan, torch.tensor([0, 0, tile_start, 0], dtype=torch.int32), ttnn.uint32)
    last = ttnn.slice(x, at, at, slice_dim=2, num_devices=padded // 32)
    at.deallocate(True)
    x.deallocate(True)
    logits = model._logits(last, prefill=True)
    last.deallocate(True)
    lap("logits_enqueue")
    ttnn.synchronize_device(model.mesh_device)
    lap("device_wait")
    out, _ = model.prefill_read({"logits": logits, "row": n - 1 - tile_start, "hidden": None})
    lap("read")
    return dict(ph), out


def compare(a, b):
    a, b = a.float(), b.float()
    return f"max|d| {float((a - b).abs().max()):.4g}, top1 {int(a.argmax())}/{int(b.argmax())}"


def skip_to(model, start, chunk):
    """Slot 0's SWA blocks of the chunks before ``start``, from a new request (the cache keeps what it holds)."""
    for s in range(0, start, chunk):
        model.swa_table.prefill(0, s, s + chunk, s + prefill_padded_len(chunk))


def breakdown(model, tokens, args):
    chunk = args.chunk
    starts = [int(s) for s in args.starts.split(",")]
    for start in starts:  # compile every program first: nothing compiles after the capture
        skip_to(model, start, chunk)
        timed_chunk(model, tokens[start : start + chunk], 0, start)
    if args.trace:
        padded = prefill_padded_len(chunk, chunk, model.pair is not None)
        t0 = time.time()
        keys = model.setup_prefill_chunk_traces([padded])
        model.warm_prefill_chunk_traces()
        t1 = time.time()
        model.capture_prefill_chunk_traces()
        say(f"traces {keys}: set up and warmed {t1 - t0:.1f} s, captured {time.time() - t1:.1f} s")
    for start in starts:
        skip_to(model, start, chunk)
        piece = tokens[start : start + chunk]
        for run in range(2):
            ttnn.synchronize_device(model.mesh_device)
            prof = cProfile.Profile() if args.profile and run == 1 and start == starts[0] else None
            if prof:
                prof.enable()
            ph, eager = timed_chunk(model, piece, 0, start)
            if prof:
                prof.disable()
                for key, count in (("tottime", 40), ("cumulative", 60)):
                    out = io.StringIO()
                    pstats.Stats(prof, stream=out).sort_stats(key).print_stats(count)
                    print(out.getvalue(), flush=True)
        enq = sum(v for k, v in ph.items() if k not in ("device_wait", "read"))
        say(
            f"eager chunk {chunk} at {start}: enqueue {enq * 1e3:.1f} ms, device wait {ph['device_wait'] * 1e3:.1f} ms, "
            f"wall {(enq + ph['device_wait']) * 1e3:.1f} ms | " + ", ".join(f"{k} {v * 1e3:.1f}" for k, v in ph.items())
        )
        if not args.trace:
            continue
        for run in range(3):
            ttnn.synchronize_device(model.mesh_device)
            t0 = time.perf_counter()
            pend = model.prefill_enqueue(piece, 0, start_pos=start, chunk_size=chunk)
            t1 = time.perf_counter()
            ttnn.synchronize_device(model.mesh_device)
            t2 = time.perf_counter()
            logits, _ = model.prefill_read(pend)
        assert pend.get("persistent"), "the chunk did not replay a trace"
        say(
            f"traced chunk {chunk} at {start}: host {1e3 * (t1 - t0):.1f} ms, device wait {1e3 * (t2 - t1):.1f} ms, "
            f"wall {1e3 * (t2 - t0):.1f} ms; vs eager {compare(eager, logits)}"
        )


def ttft(model, tokens, args):
    contexts = [int(c) for c in args.contexts.split(",")]
    padded = prefill_padded_len(args.chunk, args.chunk, model.pair is not None)
    results = {}
    # Traces --breakdown captured stay with the model; the eager runs hide them (prefill_enqueue would replay them).
    captured = model.chunk_traces
    for traced in (False, True) if args.trace else (False,):
        model.chunk_traces = captured if traced else {}
        if traced and not model.chunk_traces:
            t0 = time.time()
            keys = model.setup_prefill_chunk_traces([padded])
            model.warm_prefill_chunk_traces()
            model.capture_prefill_chunk_traces()
            say(f"traces {keys} set up, warmed and captured in {time.time() - t0:.1f} s")
            captured = model.chunk_traces
        for n in contexts:
            times = []
            for _ in range(1 if args.once else 2):
                t0 = time.time()
                logits, _ = model.prefill_read(model.prefill_enqueue(tokens[:n], 0, chunk_size=args.chunk))
                times.append(time.time() - t0)
            results[(traced, n)] = logits
            extra = f"; vs eager {compare(results[(False, n)], logits)}" if traced else ""
            say(
                f"TTFT {'traced' if traced else 'eager'} {n} tokens in chunks of {args.chunk}: {times[-1]:.3f} s "
                f"({n / times[-1]:.0f} tokens/s){extra}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq-len", type=int, default=1 << 20)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--users", type=int, default=1)
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--starts", default="4096,61440,126976,520192,1044480")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--ttft", action="store_true")
    parser.add_argument("--contexts", default="4096,32768,131072")
    parser.add_argument("--once", action="store_true", help="--ttft: one run per context")
    parser.add_argument("--trace", action="store_true", help="also the traced later chunks")
    parser.add_argument("--trace-region-mb", type=int, default=400)
    kv = parser.add_mutually_exclusive_group()
    kv.add_argument("--replicate-global-kv", action="store_true")
    kv.add_argument("--kv-pair-split", action="store_true")
    args = parser.parse_args()

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
            users_per_replica=args.users,
            max_seq_len=args.max_seq_len,
            replicate_global_kv=args.replicate_global_kv,
            kv_pair_split=args.kv_pair_split,
        )
        model = gen.models[0]
        tokens = torch.randint(0, cfg.vocab_size, (args.max_seq_len,), generator=torch.Generator().manual_seed(0))
        if args.breakdown:
            breakdown(model, tokens, args)
        if args.ttft:
            ttft(model, tokens, args)
    finally:
        for submesh in mesh.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
