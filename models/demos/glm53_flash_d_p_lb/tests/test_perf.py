# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill performance of the whole model on the device (no golden needed): the canonical prompt (spec target.seq
tokens) in target.chunk chunks from position 0.

1. Full prefill twice: cold (compiles) and warm. Per-chunk wall time with one sync per chunk (the chunk's ops are
   dispatched asynchronously, so this is the end-to-end time including host dispatch), the total (TTFT of the whole
   prompt without the LM head) and tokens/s.
2. One warm chunk at the last position with a device sync after every layer: per-layer wall time, summed per block
   type (kda_dense / dsa_moe / kda_moe). The syncs add a little time per layer; the chunk total of (1) is the real one.
3. Long-context check: next-token top-1 / top-5 of the last chunk's rows vs the prompt text itself (host LM head).

    TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 PYTHONPATH=$PWD \\
    BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \\
    scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/tests/test_perf.py -s

GLM_PERF_SEQ / GLM_PERF_CHUNK override the length and chunk (the chunk must be one the model builds tables for:
a ladder chunk or target.chunk). Results: generated/glm53_flash_d_p_lb/perf.json.
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch

import ttnn
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)
OUT = Path(__file__).resolve().parents[4] / "generated" / "glm53_flash_d_p_lb" / "perf.json"


def _prompt_tokens(n: int) -> torch.Tensor:
    from models.demos.common.bringup.reference.prompt import tokens

    return tokens(S, n).to(torch.long)


def _block_type(cfg, i: int) -> str:
    return ("kda" if cfg.is_kda(i) else "dsa") + ("_moe" if cfg.is_moe(i) else "_dense")


@mesh_parametrize
def test_perf(mesh_device):
    seq = int(os.environ.get("GLM_PERF_SEQ", S.get("target.seq")))
    chunk = int(os.environ.get("GLM_PERF_CHUNK", S.get("target.chunk")))
    assert seq % chunk == 0, (seq, chunk)
    tokens = _prompt_tokens(seq)
    layers = S.layers()

    t0 = time.time()
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=True)
    load_s = time.time() - t0
    cfg = model.cfg
    print(f"loaded {len(layers)} layers in {load_s:.0f}s; seq {seq}, chunk {chunk}", flush=True)
    mv = ttnn.get_memory_view(
        mesh_device, ttnn.BufferType.DRAM
    )  # allocator view (per chip: every chip allocates alike)
    print(
        f"DRAM per chip after load: allocated {mv.total_bytes_allocated_per_bank * mv.num_banks / 2**30:.2f} GB of "
        f"{mv.total_bytes_per_bank * mv.num_banks / 2**30:.2f} GB",
        flush=True,
    )

    def run_chunk(start: int, per_layer: dict | None = None, keep_hidden: bool = False):
        h = model.embed(tokens[start : start + chunk])
        for i in layers:
            t = time.time()
            h2 = model.layer(i, h, start, None)
            model.free(h)
            h = h2
            if per_layer is not None:
                model.sync()
                per_layer[i] = time.time() - t
        if keep_hidden:
            hid = model.final_norm(h)
            model.free(h)
            return hid
        model.free(h)
        return None

    def full_prefill(tag: str) -> list[float]:
        times = []
        for start in range(0, seq, chunk):
            t = time.time()
            run_chunk(start)
            model.sync()
            times.append(time.time() - t)
            print(f"[{tag}] chunk [{start},{start + chunk}) {times[-1] * 1e3:8.1f} ms", flush=True)
        print(f"[{tag}] total {sum(times):.2f}s  {seq / sum(times):.0f} tok/s", flush=True)
        return times

    cold = full_prefill("cold")
    warm = full_prefill("warm")

    # per-layer breakdown of one warm chunk at the last position (state content does not change the work done)
    last = seq - chunk
    per_layer = {}
    t = time.time()
    hid = run_chunk(last, per_layer=per_layer, keep_hidden=True)
    model.sync()
    synced_s = time.time() - t
    by_type = defaultdict(lambda: [0, 0.0])
    for i, dt in per_layer.items():
        bt = by_type[_block_type(cfg, i)]
        bt[0] += 1
        bt[1] += dt
    print(f"\nper-layer (synced) chunk at {last}: {synced_s * 1e3:.0f} ms", flush=True)
    for name, (n, tot) in sorted(by_type.items()):
        print(f"  {name:10s} {n:2d} layers  {tot * 1e3:8.1f} ms total  {tot / n * 1e3:6.1f} ms/layer", flush=True)

    # long-context check: next-token accuracy over the last chunk vs the prompt itself
    rows = list(range(chunk - 1))
    want = tokens[last + 1 : last + chunk]
    hits1 = hits5 = 0
    for r0 in range(0, len(rows), 512):
        rr = rows[r0 : r0 + 512]
        lg = model.logits(hid, rr).float()
        top5 = lg.topk(5, dim=-1).indices
        w = want[r0 : r0 + len(rr)]
        hits1 += int((top5[:, 0] == w).sum())
        hits5 += int((top5 == w[:, None]).any(-1).sum())
    model.free(hid)
    top1, top5 = hits1 / len(rows), hits5 / len(rows)
    print(f"last chunk next-token vs text: top1 {top1:.4f} top5 {top5:.4f}", flush=True)

    res = {
        "mesh": list(mesh_device.shape),
        "layers": len(layers),
        "experts_dtype": str(S.get("device.experts_dtype")),
        "seq": seq,
        "chunk": chunk,
        "load_s": round(load_s, 1),
        "cold_chunk_ms": [round(x * 1e3, 1) for x in cold],
        "warm_chunk_ms": [round(x * 1e3, 1) for x in warm],
        "warm_total_s": round(sum(warm), 3),
        "warm_tok_s": round(seq / sum(warm), 1),
        "last_chunk_synced_ms": round(synced_s * 1e3, 1),
        "per_layer_ms": {i: round(v * 1e3, 2) for i, v in per_layer.items()},
        "by_block_type_ms": {k: round(v[1] * 1e3, 1) for k, v in by_type.items()},
        "last_chunk_top1_vs_text": round(top1, 4),
        "last_chunk_top5_vs_text": round(top5, 4),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT}", flush=True)
