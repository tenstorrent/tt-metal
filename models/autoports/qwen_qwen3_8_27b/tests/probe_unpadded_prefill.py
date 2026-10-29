# SPDX-License-Identifier: Apache-2.0
"""Strict all64-layer exactness gate: unpadded (compact live-row) vs padded prefill.

One process, one Model/Generator (B32, serving_state, external-style page map),
same scripted serving sequence run twice from reset: padded, then unpadded.
Every step records SHA-256 of logits (all four devices) and, at checkpoints,
every conv/recurrent/K/V byte. Sequence: cold multi-chunk prefill, ragged mixed
lengths, >8 equal rows (split groups), aligned cached continuation, non-128
bridge continuation, traced decode, slot cancellation/reset + reuse, and a
continuation after churn. Pass requires byte equality at every recorded point.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import ttnn


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/workspace/.cache/qwen38-full")
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--layers", type=int, nargs="*")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    for k in ("QWEN38_GROUPED_RECURRENCE", "QWEN38_MASKED_RECURRENCE", "QWEN38_FLAT_PREFILL"):
        os.environ.setdefault(k, "1")
    os.environ["QWEN38_UNPADDED_PREFILL"] = "1"
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024**2)
    from models.autoports.qwen_qwen3_8_27b.tt.model import Model
    from models.autoports.qwen_qwen3_8_27b.tt.generator import Generator

    out = dict(scope="strict exactness unpadded vs padded; generator level, B32 serving_state",
               flags={k: v for k, v in os.environ.items() if k.startswith("QWEN38_")},
               context=a.context, chunk=a.chunk, layers=a.layers or "all64", runs={})
    save = lambda: a.output.write_text(json.dumps(out, indent=1))
    try:
        model = Model(a.checkpoint, mesh, layer_indices=a.layers)
        B = 32
        gen = Generator(model, a.checkpoint, batch_size=B, context_length=a.context,
                        prefill_chunk_size=a.chunk, serving_state=True)
        t = time.perf_counter()
        gen.prepare_decode()
        out["prepare_s"] = time.perf_counter() - t
        out["prepared_compact"] = sorted(gen.prepared_compact)
        V = model.config.vocab_size
        g = torch.Generator().manual_seed(1234)
        rand = lambda *shape: torch.randint(1000, V // 2, shape, generator=g, dtype=torch.int64)
        script_tokens = {k: rand(B, n) for k, n in
                         dict(cold=1024, ragged=320, many=200, cont=512, bridge=300, decode=8, reuse=256).items()}
        table = gen.page_table_host.clone()

        def states(tag, rec):
            ttnn.synchronize_device(mesh)
            h = {}
            for i, st in enumerate(gen.kv_cache):
                for f, v in vars(st).items():
                    for d, part in enumerate(ttnn.get_device_tensors(v)):
                        h[f"{i}/{f}/{d}"] = sha(ttnn.to_torch(part))
            rec["states"][tag] = h

        def logits(tag, rec, value):
            ttnn.synchronize_device(mesh)
            rec["logits"][tag] = [sha(ttnn.to_torch(part)) for part in ttnn.get_device_tensors(value)]

        def prefill(lens, toks, start=0):
            full = torch.zeros(B, max(lens), dtype=torch.int64)
            for s, n in enumerate(lens):
                if n:
                    full[s, :n] = toks[s, :n]
            if start:
                return gen.prefill_continuation(full, page_table=table, kv_cache=gen.kv_cache,
                                                prompt_lens=lens, start_position=start)
            return gen.prefill_forward(full, page_table=table, kv_cache=gen.kv_cache,
                                       prompt_lens=lens, return_device=True)

        def scenario(unpadded):
            gen.unpadded_prefill = unpadded
            gen.reset()
            rec = dict(logits={}, states={}, tokens={}, timings={})
            pos = [0] * B
            T = script_tokens

            def step(tag, lens, toks, start=0):
                t0 = time.perf_counter()
                lg = prefill(lens, toks, start)
                ttnn.synchronize_device(mesh)
                rec["timings"][tag] = time.perf_counter() - t0
                logits(tag, rec, lg)
                del lg
            # 1 cold single row, two physical chunks
            lens = [0] * B; lens[3] = 1024
            step("cold_row3_1024", lens, T["cold"]); pos[3] = 1024
            # 2 ragged mixed lengths in one call (b=2 group + b=1 group)
            lens = [0] * B; lens[5] = 300; lens[9] = 300; lens[17] = 77
            step("ragged", lens, T["ragged"]); pos[5] = pos[9] = 300; pos[17] = 77
            # 3 ten equal rows -> compact groups 8 + 2
            lens = [0] * B
            for s in range(20, 30):
                lens[s] = 200; pos[s] = 200
            step("many10_200", lens, T["many"])
            states("after_cold_all", rec)
            # 4 aligned cached continuation of row 3 (start 1024, 512 tokens)
            lens = [0] * B; lens[3] = 512
            step("cont_row3_1024_512", lens, T["cont"], start=1024); pos[3] = 1536
            # 5 bridge continuation: row 5 from 300 (non-128), 300 tokens
            lens = [0] * B; lens[5] = 300
            step("bridge_row5_300_300", lens, T["bridge"], start=300); pos[5] = 600
            states("after_continuations", rec)
            # 6 traced decode, all live rows, 8 steps
            live = [s for s in range(B) if pos[s]]
            toks = []
            for i in range(8):
                sp = torch.tensor([pos[s] if pos[s] else -1 for s in range(B)])
                lg = gen.decode_forward(T["decode"][:, i], sp, page_table=table, kv_cache=gen.kv_cache,
                                        return_logits=True)
                toks.append(lg.argmax(-1)[live].tolist())
                rec["logits"][f"decode{i}"] = sha(lg.float())
                for s in live:
                    pos[s] += 1
            rec["tokens"]["decode_argmax"] = toks
            states("after_decode", rec)
            # 7 cancellation: reset row 9 mid-life, reuse slot with a fresh prompt
            gen.reset_state_slots([9])
            lens = [0] * B; lens[9] = 256
            step("reuse_row9_256", lens, T["reuse"]); pos[9] = 256
            # 8 continuation after churn on row 20 at an aligned start (200+8=208 -> bridge)
            lens = [0] * B; lens[20] = 256
            step("churn_row20", lens, T["reuse"], start=pos[20]); pos[20] += 256
            states("final", rec)
            rec["unpadded_counters"] = dict(gen.unpadded_counters)
            return rec

        for name, flag in (("padded", False), ("unpadded", True), ("padded_repeat", False)):
            t = time.perf_counter()
            out["runs"][name] = scenario(flag)
            out["runs"][name]["wall_s"] = time.perf_counter() - t
            save()
            print("SCENARIO", name, out["runs"][name]["wall_s"], json.dumps(out["runs"][name]["timings"]), flush=True)
        ref, cand, rep = (out["runs"][k] for k in ("padded", "unpadded", "padded_repeat"))
        diffs = []
        for kind in ("logits", "states", "tokens"):
            for tag in ref[kind]:
                if ref[kind][tag] != cand[kind].get(tag):
                    if kind == "states":
                        bad = [k for k in ref[kind][tag] if ref[kind][tag][k] != cand[kind][tag].get(k)]
                        diffs.append(dict(kind=kind, tag=tag, count=len(bad), first=bad[:12]))
                    else:
                        diffs.append(dict(kind=kind, tag=tag))
        out["repeat_padded_exact"] = all(ref[k] == rep[k] for k in ("logits", "states", "tokens"))
        out["diffs"] = diffs
        out["exact"] = not diffs
        out["state_fields_hashed"] = len(next(iter(ref["states"].values())))
        out["status"] = "pass" if out["exact"] else "mismatch"
        print("RESULT", out["status"], json.dumps(diffs)[:4000], flush=True)
    except BaseException as e:
        out.update(status="fail", error=repr(e))
        raise
    finally:
        save()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
