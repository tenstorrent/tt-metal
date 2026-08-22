# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""De-risking probe: build a reduced (one-of-each-kind) full model, run prefill, then exercise
the split-sampling terminal path (Sampling1D greedy top-k(k=1) + force-argmax) and the on-device
token->embedding feedback. Prints tensor shapes so the generator can be written against the real
contract. NOT a correctness gate — just shape/behaviour discovery.

Run with ``--profile p150|p150x2|p150x4`` (or ``LAGUNA_PROFILE``).
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
from models.autoports.poolside_laguna_xs_2_1.tt.model import LagunaModel
from models.common.modules.sampling.sampling_1d import Sampling1D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq-len", type=int, default=2048)
    add_profile_args(parser, default_trace_region_size=200_000_000)
    args = parser.parse_args()
    profile = profile_from_args(args)
    print("PROFILE", json.dumps(profile_summary(profile), sort_keys=True))
    mesh = open_mesh(ttnn, profile)
    try:
        model = LagunaModel.from_pretrained(mesh, max_seq_len=args.max_seq_len, num_layers=[0, 1, 4])
        print(
            "MODEL built. layers:",
            model.meta["layer_indices"],
            "vocab",
            model.cfg.vocab,
            "per_dev_vocab",
            model.per_device_vocab,
        )

        # ---- prefill a short prompt ----
        P = 16
        torch.manual_seed(0)
        toks = torch.randint(0, model.cfg.vocab, (1, P), dtype=torch.int64)
        tok_tt = ttnn.from_torch(
            toks.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        x = model.embed_prefill(tok_tt)
        print("embed_prefill:", x.shape, x.dtype)
        kv = model.alloc_kv_cache(max_users=1, max_seq_len=P + 64, block_size=32)
        pt = model.make_page_table(1, kv[0]["blocks_per_user"])
        h = model.prefill_layers(x, kv, pt, user_id=0, start_pos=0)
        print("prefill_layers out:", h.shape, h.dtype)
        # last position logits (shards)
        last = ttnn.slice(h, [0, P - 1, 0], [1, P, model.cfg.hidden])
        shards = model.lm_head_shards_prefill(last)
        print("lm_head shard (prefill):", shards.shape, shards.dtype)
        full = model.logits_to_host(shards)
        print("gathered logits host:", full.shape)
        host_argmax = int(full.reshape(-1, model.cfg.vocab)[-1].argmax())
        print("HOST argmax token:", host_argmax)

        # ---- decode-shaped logits [1,1,B,V/D] for sampling ----
        B = 1
        hd = model.embed_decode(
            ttnn.from_torch(
                toks[:, -1:].to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
        )
        print("embed_decode:", hd.shape)
        dshards = model.lm_head_shards_decode(hd)
        print("lm_head shard (decode):", dshards.shape, dshards.dtype, dshards.memory_config())

        # ---- Sampling1D greedy top-k(k=1) ----
        sampler = Sampling1D(
            vocab_size=model.cfg.vocab,
            mesh_device=mesh,
            max_batch_size=B,
            max_top_k=32,
            allow_force_argmax=True,
            pad_to_power_of_2=True,
        )
        sampler.load_device_buffers()

        def rep(t, dt, lay=ttnn.ROW_MAJOR_LAYOUT):
            return ttnn.from_torch(t, dtype=dt, layout=lay, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        k = rep(torch.ones([B], dtype=torch.int32), ttnn.uint32)
        p = rep(torch.ones([B], dtype=torch.float32), ttnn.bfloat16)
        t = rep(torch.ones([B], dtype=torch.float32), ttnn.bfloat16)
        tok_out, lp = sampler.decode_forward(dshards, k=k, p=p, temp=t)
        print("topk-k1 token shape:", tok_out.shape, tok_out.dtype)
        th = ttnn.to_torch(tok_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        print("  token host (per-device):", th.flatten().tolist(), "vs host_argmax", host_argmax)

        # ---- feedback: preallocate tok buffer, sample into it, embed the same buffer ----
        try:
            tok_buf = rep(torch.zeros([1, 1, 1, B], dtype=torch.int32), ttnn.uint32)
            out_tok, _ = sampler.decode_forward(dshards, k=k, p=p, temp=t, tt_out_tok=tok_buf)
            print("tok_out via tt_out_tok:", out_tok.shape, out_tok.dtype, "same obj:", out_tok is tok_buf)
            emb2 = model.embed_decode(ttnn.reshape(tok_buf, (1, B)))
            print("re-embed from sampled token buffer OK:", emb2.shape)
        except Exception as e:
            print("feedback embed FAILED:", repr(e)[:300])

        print("PROBE_DONE")
    finally:
        close_mesh(ttnn, mesh)


if __name__ == "__main__":
    main()
