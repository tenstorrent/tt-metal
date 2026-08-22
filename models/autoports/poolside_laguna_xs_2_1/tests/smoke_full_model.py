# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full-model profile smoke: build, allocate qualified KV, prefill, and replay traced decode."""
from __future__ import annotations

import argparse
import json
import time

import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests.laguna_test_utils import (
    add_profile_args,
    assert_memory_margin,
    close_mesh,
    open_mesh,
    print_memory_snapshot,
    profile_from_args,
    profile_summary,
)
from models.autoports.poolside_laguna_xs_2_1.tt.generator import LagunaGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--prompt", type=int, default=100, help="non-aligned prompt length")
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--enforce-memory-margin", action="store_true")
    ap.add_argument(
        "--acceptance",
        action="store_true",
        help="require the selected profile's exact context cap and all sampler-equivalence gates",
    )
    add_profile_args(ap)
    args = ap.parse_args()
    profile = profile_from_args(args)
    max_seq_len = args.max_seq_len or profile.max_context
    if args.acceptance and max_seq_len != profile.max_context:
        raise ValueError(
            f"acceptance requires exact {profile.name} context {profile.max_context}, got {max_seq_len}"
        )
    if max_seq_len < args.prompt + args.gen:
        raise ValueError("max-seq-len must cover prompt + generation")
    print("PROFILE", json.dumps(profile_summary(profile), sort_keys=True))

    mesh = open_mesh(ttnn, profile)
    gen = None

    def snapshot(label):
        value = print_memory_snapshot(ttnn, mesh, label)
        if args.enforce_memory_margin:
            assert_memory_margin(value)

    try:
        t0 = time.perf_counter()
        gen = LagunaGenerator.from_pretrained(mesh, max_seq_len=max_seq_len)  # all 40 layers
        print(f"BUILD 40 layers: {time.perf_counter()-t0:.1f}s")
        snapshot("weights")

        torch.manual_seed(1)
        P = args.prompt
        prompt = torch.randint(0, gen.vocab, (P,), dtype=torch.int64).tolist()
        N = args.gen
        gen._ensure_cache(1, max_seq_len)
        snapshot("uniform_kv")

        # device traced free-run
        gen.reset()
        t1 = time.perf_counter()
        seq = gen.generate(prompt, N, next_input=None, enable_trace=True)
        dt = time.perf_counter() - t1
        snapshot("trace_warmup")
        print("tokens:", seq[:16], "...")
        print("counters:", dict(gen.counters))
        print(f"generate {N} tok wall: {dt*1000:.1f} ms  (~{N/dt:.1f} tok/s incl prefill+readback)")

        # host-argmax control on the first few tokens (correctness of device greedy)
        gen.host_sampling = True
        gen.reset()
        seq_h = gen.generate(prompt, 8, next_input=None, enable_trace=False)
        gen.host_sampling = False
        match = sum(int(a == b) for a, b in zip(seq, seq_h))
        print(f"device-vs-host greedy first-8 match: {match}/8  host={seq_h}")
        assert match == 8, f"device sampler disagrees with host argmax for {8 - match}/8 tokens"
        print("PROBE_SMOKE_DONE")
    finally:
        try:
            if gen is not None:
                gen.teardown()
        except Exception:
            pass
        close_mesh(ttnn, mesh)


if __name__ == "__main__":
    main()
