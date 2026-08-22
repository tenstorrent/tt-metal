# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Mechanism validation on the reduced (one-of-each-kind) model: traced on-device split-sampling
generate vs a host-argmax control, teacher-forcing override, and host-work counters. NOT an
accuracy gate (reduced layer stack != HF)."""
from __future__ import annotations

import argparse
import json

import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests.laguna_test_utils import (
    add_profile_args,
    close_mesh,
    open_mesh,
    profile_from_args,
    profile_summary,
)
from models.autoports.poolside_laguna_xs_2_1.tt.generator import LagunaGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq-len", type=int, default=2048)
    add_profile_args(parser, default_trace_region_size=200_000_000)
    args = parser.parse_args()
    profile = profile_from_args(args)
    print("PROFILE", json.dumps(profile_summary(profile), sort_keys=True))
    mesh = open_mesh(ttnn, profile)
    gen = None
    try:
        gen = LagunaGenerator.from_pretrained(mesh, max_seq_len=args.max_seq_len, num_layers=[0, 1, 4])
        torch.manual_seed(0)
        prompt = torch.randint(0, gen.vocab, (24,), dtype=torch.int64).tolist()
        N = 12

        # 1) traced device split-sampling, free-running
        gen.reset()
        seq_dev = gen.generate(prompt, N, next_input=None, enable_trace=True)
        c = dict(gen.counters)
        print("DEV traced free-run tokens:", seq_dev)
        print("counters:", c)

        # 2) host-argmax control (eager)
        gen.host_sampling = True
        gen.reset()
        seq_host = gen.generate(prompt, N, next_input=None, enable_trace=False)
        gen.host_sampling = False
        print("HOST eager tokens:      ", seq_host)
        match = sum(int(a == b) for a, b in zip(seq_dev, seq_host))
        print(f"MATCH device-vs-host greedy: {match}/{N}")

        # 3) teacher-forcing override + prediction bookkeeping
        gen.reset()
        forced = []

        def nxt(step, pred):
            forced.append(pred)
            return int(prompt[step % len(prompt)])  # arbitrary forced next input

        preds = gen.generate(prompt, N, next_input=nxt, enable_trace=True)
        print("TF preds:", preds)
        print("TF callback count:", len(forced), "expected", N)
        print("TF token_refresh counter:", gen.counters["token_refresh"], "(should be ~N: 1 init + N-1 forced)")
        print("TF trace_replay:", gen.counters["trace_replay"], "expected", N - 1)
        print("TF pos_refresh:", gen.counters["pos_refresh"], "(should be 1: positions advance on device)")

        print("PROBE_GEN_DONE")
    finally:
        try:
            if gen is not None:
                gen.teardown()
        except Exception:
            pass
        close_mesh(ttnn, mesh)


if __name__ == "__main__":
    main()
