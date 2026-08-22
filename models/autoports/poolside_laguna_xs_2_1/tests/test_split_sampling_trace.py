# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Focused split-sampling trace test (reduced one-of-each-kind model — mechanism is layer-count
independent). Proves the delivered token-out decode path with ONE captured trace over persistent
device inputs:

  * device feedback — the sampled token from step N is the decode input for step N+1 with NO host
    reconstruction (verified against a host-argmax control);
  * position coherence — cur_pos/rope advance on device inside the trace (read back between replays),
    no per-token host position refresh;
  * unchanged page-table — the persistent page-table tensor is never rewritten across free-run steps;
  * changed page-table — copying a DIFFERENT physical block range into the SAME persistent page-table
    tensor (one host copy, no reallocation, no re-capture) and replaying the SAME trace yields the
    identical greedy token (both ranges hold the same prompt KV) — correct changed/unchanged handling.

Run: cd /tmp && TT_METAL_HOME=<tree> PYTHONPATH=<repo> python -u -m \
     models.autoports.poolside_laguna_xs_2_1.tests.test_split_sampling_trace
"""
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

FAIL = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq-len", type=int, default=2048)
    add_profile_args(parser, default_trace_region_size=400_000_000)
    args = parser.parse_args()
    profile = profile_from_args(args)
    print("PROFILE", json.dumps(profile_summary(profile), sort_keys=True))
    mesh = open_mesh(ttnn, profile)
    gen = None

    def read(t):
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0].flatten().tolist()

    def host_pt(vals):
        return ttnn.from_torch(
            torch.tensor([vals], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    try:
        gen = LagunaGenerator.from_pretrained(mesh, max_seq_len=args.max_seq_len, num_layers=[0, 1, 4])
        torch.manual_seed(3)
        P, K = 40, 6
        prompt = torch.randint(0, gen.vocab, (P,), dtype=torch.int64).tolist()

        # cache with 2 users' worth of blocks; user0 blocks = PT_a, user1 blocks = PT_b.
        kv = gen.model.alloc_kv_cache(max_users=2, max_seq_len=P + K + 8, block_size=32)
        gen._kv_cache, gen._kv_users, gen._kv_seq = kv, 2, P + K + 8
        bpu = kv[0]["blocks_per_user"]
        # the host-argmax control (gen.generate) needs an owned page table; give it PT_a's mapping.
        gen._page_table = ttnn.from_torch(
            torch.arange(0, bpu, dtype=torch.int32).reshape(1, bpu),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        pt_a_vals = list(range(0, bpu))
        pt_b_vals = list(range(bpu, 2 * bpu))
        pt_a = ttnn.from_torch(
            torch.tensor([pt_a_vals], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        pt_b = ttnn.from_torch(
            torch.tensor([pt_b_vals], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # prefill the SAME prompt into BOTH physical block ranges
        gen.model.prefill_layers(
            gen.model.embed_prefill(gen._tokens_to_device(torch.tensor(prompt))), kv, pt_a, user_id=0, start_pos=0
        )
        gen.model.prefill_layers(
            gen.model.embed_prefill(gen._tokens_to_device(torch.tensor(prompt))), kv, pt_b, user_id=0, start_pos=0
        )
        # p0 from prefill last position (device greedy sample)
        h = gen.model.prefill_layers(
            gen.model.embed_prefill(gen._tokens_to_device(torch.tensor(prompt))), kv, pt_a, user_id=0, start_pos=0
        )
        last = ttnn.slice(h, [0, P - 1, 0], [1, P, gen.hidden])
        shards = gen.model.lm_head_shards_decode(ttnn.reshape(last, (1, 1, 1, gen.hidden)))
        tb0 = gen._rep(torch.zeros([1, 1, 1, 1], dtype=torch.int32), ttnn.uint32)
        gen._greedy_sample(shards, 1, tb0)
        p0 = gen._read_token(tb0, 1)[0]

        # persistent page-table tensor (starts = PT_a); capture ONE trace over it (all allocs done).
        pt_persist = ttnn.from_torch(
            torch.tensor([pt_a_vals], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        st = gen._decode_trace_state(1, pt_persist, P, p0)
        tok, cur, ridx, tid = st["tok"], st["cur"], st["ridx"], st["tid"]

        def stage():
            ttnn.copy_host_to_device_tensor(gen._host_rank4_tok(p0), tok)
            ttnn.copy_host_to_device_tensor(gen._host_pos(P), cur)
            ttnn.copy_host_to_device_tensor(gen._host_ridx(P), ridx)

        # ---- Part A: device feedback + position coherence + unchanged page table ----
        ttnn.copy_host_to_device_tensor(host_pt(pt_a_vals), pt_persist)  # ensure PT_a
        stage()
        pt_before = read(pt_persist)
        dev_seq = [p0]
        fed = [p0]
        for i in range(K):
            pos_before = read(cur)[0]
            fed.append(read(tok)[0])
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            pos_after = read(cur)[0]
            dev_seq.append(gen._read_token(tok, 1)[0])
            check(pos_after == pos_before + 1, f"pos advances on device {pos_before}->{pos_after} (step {i})")
            if i > 0:
                check(fed[i] == dev_seq[i], f"device feedback: step{i} input == step{i-1} sampled")
        check(read(pt_persist) == pt_before, "unchanged page-table tensor never rewritten across replays")

        # ---- Part B: changed page table (same trace, one host copy, no realloc/recapture) ----
        # Run BEFORE the host control (which resets/zeros the cache) so both prefilled block ranges
        # are still intact.
        ttnn.copy_host_to_device_tensor(host_pt(pt_a_vals), pt_persist)
        stage()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        tok_a = gen._read_token(tok, 1)[0]
        pt_a_read = read(pt_persist)
        ttnn.copy_host_to_device_tensor(host_pt(pt_b_vals), pt_persist)  # ONE copy: page table CHANGED
        pt_b_read = read(pt_persist)
        stage()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        tok_b = gen._read_token(tok, 1)[0]
        check(pt_a_read != pt_b_read, "changed page-table: persistent tensor updated by the one copy")
        check(tok_a == tok_b, f"changed page-table gives identical greedy token ({tok_a} == {tok_b})")

        # ---- host-argmax control (resets the cache last) ----
        gen.host_sampling = True
        gen.reset()
        host_seq = gen.generate(prompt, K + 1, next_input=None, enable_trace=False)
        gen.host_sampling = False
        m = sum(int(a == b) for a, b in zip(dev_seq, host_seq))
        check(m == len(host_seq), f"device-feedback greedy == host-argmax control ({m}/{len(host_seq)})")

        print("SPLIT_SAMPLING_TRACE:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}", flush=True)
    finally:
        try:
            if gen is not None:
                gen.teardown()
        except Exception:
            pass
        close_mesh(ttnn, mesh)


if __name__ == "__main__":
    main()
    import os as _os
    import sys as _sys

    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(1 if FAIL else 0)
