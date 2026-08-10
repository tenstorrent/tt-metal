# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Like-for-like GPT decode profile on a 1xN mesh at a realistic utterance length.

Answers two things the short (64-step) profile could not:

1. **Does per-step cost grow with KV-cache depth?** Decode is timed in buckets across the whole
   run, so growth from position P to P+steps is visible instead of assumed linear.
2. **What does the host sampling head actually cost at length?** The repetition penalty loops
   over the `seen` set, which grows by one token per step — so its cost grows *with the sequence*,
   and a figure measured over 67 steps under-predicts a 586-step utterance.

Both loops run back-to-back on the same decoder (caches reset between), so the only difference is
whether the host sampling head is in the loop.

    XTTS_CKPT=/var/tmp/xtts_ref/model.pth python profile_mesh_decode.py --replicas 32 --steps 586
"""

import argparse
import time

import torch
import ttnn

from models.experimental.xtts_v2.reference.xtts_gpt_ref import load_gen_head
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt import preprocess_gpt_parameters
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt_decode import TTNNGPTTracedDecoder

TEMPERATURE, TOP_K, TOP_P, PENALTY = 0.75, 50, 0.85, 10.0


def sample(latent, seen, mh_w, mh_b, gen, vectorized_penalty=True):
    """Byte-for-byte the sampling head phase_tt_mesh.py runs per slot per step.

    `vectorized_penalty=False` reproduces the original Python loop over `seen`, kept so the two
    can be timed against each other at the same sequence length (they are bit-identical)."""
    logits = (latent @ mh_w.t() + mh_b)[0, 0].clone().float()
    if vectorized_penalty:
        if seen:
            idx = torch.tensor(sorted(seen))
            v = logits[idx]
            logits[idx] = torch.where(v > 0, v / PENALTY, v * PENALTY)
    else:
        for tok in seen:  # O(len(seen)) per slot per step -> grows with the utterance
            logits[tok] = logits[tok] / PENALTY if logits[tok] > 0 else logits[tok] * PENALTY
    logits = logits / TEMPERATURE
    if TOP_K and TOP_K < logits.numel():
        kth = torch.topk(logits, TOP_K).values[-1]
        logits[logits < kth] = float("-inf")
    if TOP_P < 1.0:
        sl, si = torch.sort(logits, descending=True)
        drop = torch.softmax(sl, dim=-1).cumsum(dim=-1) > TOP_P
        drop[1:] = drop[:-1].clone()
        drop[0] = False
        logits[si[drop]] = float("-inf")
    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=gen))


def run_loop(dec, mesh, shard, comp, N, P, steps, heads, with_sampling, bucket, vec=True):
    dec.reset_caches()
    emb = torch.randn(N, 1, 1024) * 0.1
    pos = [P] * N
    seen = [{1024} for _ in range(N)]
    gens = [torch.Generator().manual_seed(i) for i in range(N)]
    mh_w, mh_b = heads["mel_head_w"], heads["mel_head_b"]
    mel_emb, mel_pos = heads["mel_emb"], heads["mel_pos"]
    buckets = []
    t_bucket = time.time()
    t0 = time.time()
    for s in range(steps):
        e = ttnn.from_torch(emb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=shard)
        lat_dev = dec.step_device(e, pos)
        lat = ttnn.to_torch(lat_dev, mesh_composer=comp) if comp else ttnn.to_torch(lat_dev)
        lat = lat.float()
        pos = [p + 1 for p in pos]
        if with_sampling:
            rows = []
            for i in range(N):
                code = sample(lat[i : i + 1], seen[i], mh_w, mh_b, gens[i], vec)
                seen[i].add(code)
                rows.append((mel_emb[code] + mel_pos[min(s + 1, mel_pos.shape[0] - 1)]).view(1, 1, -1))
            emb = torch.cat(rows, 0)
        if (s + 1) % bucket == 0:
            dt = time.time() - t_bucket
            buckets.append((s + 1 - bucket, s + 1, 1000 * dt / bucket))
            t_bucket = time.time()
    total = time.time() - t0
    return total, buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas", type=int, default=32)
    ap.add_argument("--steps", type=int, default=586)
    ap.add_argument("--prompt-len", type=int, default=75)
    ap.add_argument("--bucket", type=int, default=100)
    args = ap.parse_args()
    N, S, P = args.replicas, args.steps, args.prompt_len

    heads = load_gen_head()
    print(f"opening 1x{N} mesh", flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), l1_small_size=65536, trace_region_size=60_000_000)
    try:
        mesh.enable_program_cache()
        shard = ttnn.shard_tensor_to_mesh_mapper(mesh, 0) if N > 1 else None
        comp = ttnn.concat_mesh_to_tensor_composer(mesh, 0) if N > 1 else None

        t = time.time()
        params = preprocess_gpt_parameters(mesh, dtype=ttnn.bfloat16)
        dec = TTNNGPTTracedDecoder(mesh, params, max_seq=P + 1 + S, batch=N, data_mapper=shard)
        dec.reset_caches()
        dec.prefill((torch.randn(N, P, 1024) * 0.1).contiguous())
        dec.capture()
        print(f"setup (weights + prefill + trace): {time.time() - t:.1f}s  max_seq={dec.max_seq}", flush=True)

        for with_sampling, vec, tag in (
            (False, True, "device-only (no sampling)"),
            (True, False, "WITH sampling head, ORIGINAL python penalty loop"),
            (True, True, "WITH sampling head, VECTORIZED penalty"),
        ):
            print(f"\n=== {S} decode steps, {N} requests, {tag} ===", flush=True)
            total, buckets = run_loop(dec, mesh, shard, comp, N, P, S, heads, with_sampling, args.bucket, vec)
            for a, b, ms in buckets:
                print(f"  steps {a:4d}-{b:4d} (cache pos {P+a:4d}-{P+b:4d}): {ms:7.2f} ms/step", flush=True)
            print(
                f"  TOTAL {total:7.2f}s   mean {1000*total/S:6.2f} ms/step   "
                f"{N*S/total:7.0f} tok/s aggregate   {1000*total/(S*N):5.3f} ms/token"
            )
            print(f"  -> per-request latency for a {S}-token utterance: {total:.1f}s")
    finally:
        mesh.quiesce_devices()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
