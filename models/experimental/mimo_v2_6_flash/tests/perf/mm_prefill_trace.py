# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Eager vs traced prefill chunks with multimodal rows on one 1x8 row (run under a row lock: TT_VISIBLE_DEVICES holds
the row's 8 chips).

For each of ``--lengths`` (padded first-chunk lengths) a prompt of that many tokens at start 0 whose rows
[8, 8 + ``--mm-fraction`` * n) are host rows (embedding rows of random tokens, as an image's rows would be: real
bf16 rows of the model's scale), prefilled:

* eagerly (``prefill_enqueue(..., embeds=, allow_trace=False)``: the merge and every op dispatched from the host),
* replaying the traced first chunk of that length with the rows merged from its persistent inputs
  (TtMiMoModel.enable_trace_embeds), and
* the same prompt as text, traced (the replay without host rows), for reference.

Host enqueue (until prefill_enqueue returns) and total (until the logits are on the host) per mode, the median of
``--repeats`` runs each started on an idle device; the eager and traced logits must be identical. Also the trace region
the traces take. Every line to grep starts with ``MIMO_MMT``.
"""

import argparse
import os
import statistics
import time

import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.reference.checkpoint import MiMoCheckpoint, default_model_path
from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.tt.common import fabric_config
from models.experimental.mimo_v2_6_flash.tt.generator import create_dp_submeshes
from models.experimental.mimo_v2_6_flash.tt.model import TtMiMoModel, prefill_padded_len


def say(msg):
    logger.info(msg)
    print(f"MIMO_MMT {msg}", flush=True)


def timed(model, tokens, embeds=None, rows=None, allow_trace=True):
    ttnn.synchronize_device(model.mesh_device)
    t0 = time.perf_counter()
    handle = model.prefill_enqueue(tokens, 0, embeds=embeds, embed_rows=rows, allow_trace=allow_trace)
    t1 = time.perf_counter()
    logits = model.prefill_read(handle)[0]
    t2 = time.perf_counter()
    return 1e3 * (t1 - t0), 1e3 * (t2 - t0), logits, bool(handle.get("persistent")) and allow_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="512,1024,2048,4096")
    parser.add_argument("--mm-fraction", type=float, default=0.8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trace-region-mb", type=int, default=512)
    parser.add_argument("--no-pair-split", action="store_true", help="global KV on both chips (default: pair split)")
    args = parser.parse_args()
    lengths = [int(n) for n in args.lengths.split(",")]
    assert all(prefill_padded_len(n) == n for n in lengths), lengths

    path = os.environ.get("MIMO_MODEL_PATH", default_model_path())
    cfg = MiMoConfig.from_model_path(path)
    ckpt = MiMoCheckpoint(path, cfg)
    cache = os.path.join(
        os.environ.get("MIMO_TT_CACHE", "/home/zni/models/mimo-tt-cache"), os.path.basename(path), "model_bfp4experts"
    )
    ttnn.set_fabric_config(**fabric_config())
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 8),
        l1_small_size=65536,
        trace_region_size=args.trace_region_mb << 20,
        dispatch_core_config=ttnn.DispatchCoreConfig(axis=ttnn.DispatchCoreAxis.COL),
    )
    try:
        row = create_dp_submeshes(mesh, 1)[0]
        t0 = time.time()
        model = TtMiMoModel(
            row,
            cfg,
            ckpt=ckpt,
            cache_dir=cache,
            max_seq_len=2 * max(lengths),
            max_prefill_chunk=max(lengths),
            decode_users=2,
            kv_pair_split=not args.no_pair_split,
        )
        model.setup_kv_cache(2)
        model.enable_trace_embeds()
        say(f"model ready in {time.time() - t0:.0f}s (pair split {not args.no_pair_split})")
        embedding = ckpt.embedding().to(torch.bfloat16)
        gen = torch.Generator().manual_seed(0)
        prompts = {}
        for n in lengths:
            tokens = torch.randint(0, cfg.vocab_size, (n,), generator=gen, dtype=torch.int32)
            rows = torch.zeros(n, dtype=torch.bool)
            rows[8 : 8 + int(args.mm_fraction * n)] = True
            embeds = embedding[torch.randint(0, cfg.vocab_size, (n,), generator=gen)]
            prompts[n] = (tokens, embeds, rows)
            timed(model, tokens, embeds, rows, allow_trace=False)  # compile the eager passes before any capture
            timed(model, tokens, allow_trace=False)

        t1 = time.time()
        keys = model.setup_prefill_chunk_traces(lengths, first=True)
        model.warm_prefill_chunk_traces()
        used0 = model.trace_region_bytes()[0]
        assert model.capture_prefill_chunk_traces(), "the traces did not fit"
        used, total = model.trace_region_bytes()
        segments = [int(t.shape[0]) for t in model.trace_rows]
        say(
            f"{len(keys)} traced first chunks {lengths} set up, warmed and captured in {time.time() - t1:.0f}s: "
            f"{(used - used0) / 2**20:.1f} MB of the trace region ({(used - used0) / 2**20 / len(keys):.1f} MB each); "
            f"row buffer segments {segments} ({sum(segments) * cfg.hidden_size * 2 / 2**20:.0f} MB per chip)"
        )
        programs = row.num_program_cache_entries()

        say("padded | tokens | MM rows | eager enqueue / total ms | traced enqueue / total ms | text traced total ms")
        for n in lengths:
            tokens, embeds, rows = prompts[n]
            res = {"eager": [], "traced": [], "text": []}
            ref = None
            for _ in range(args.repeats):
                for mode in res:
                    if mode == "text":
                        enq, tot, logits, replayed = timed(model, tokens)
                    else:
                        enq, tot, logits, replayed = timed(model, tokens, embeds, rows, allow_trace=mode == "traced")
                        if ref is None:
                            ref = logits
                        diff = float((logits - ref).abs().max())
                        assert torch.equal(logits, ref), f"{n}: {mode} logits differ from eager (max |diff| {diff})"
                    assert replayed == (mode != "eager"), (mode, replayed)
                    res[mode].append((enq, tot))
            med = {m: tuple(statistics.median(v[k] for v in r) for k in range(2)) for m, r in res.items()}
            say(
                f"{n} | {n} | {int(rows.sum())} | {med['eager'][0]:.1f} / {med['eager'][1]:.1f} | "
                f"{med['traced'][0]:.1f} / {med['traced'][1]:.1f} | {med['text'][1]:.1f} "
                f"(eager == traced logits, {args.repeats} runs)"
            )
        assert row.num_program_cache_entries() == programs, "compiled programs after the captures"
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
