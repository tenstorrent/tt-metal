# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fast layer-level smoke: build the optimized decoder, prefill + decode, PCC vs HF.

The iteration loop for the optimized stage.  Not the acceptance gate -- that is
``tests/test_optimized_decoder.py`` -- but small enough to run after every
layout or precision change.

    python .../bench/smoke.py [--kinds sliding,full] [--seq-lens 100,4097]
"""

from __future__ import annotations

import argparse

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import reference as R
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import reference_layer_indices
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import OptimizedDecoder

PAGE_BLOCK = 64
MAX_SEQ = 16384


def page_table(mesh, batch, max_seq_len, seed=7):
    blocks = (max_seq_len + PAGE_BLOCK - 1) // PAGE_BLOCK
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(batch * blocks, generator=gen).reshape(batch, blocks).to(torch.int32)
    return ttnn.from_torch(
        perm, device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def to_dev(mesh, hidden):
    flat = hidden.reshape(1, 1, hidden.shape[0] * hidden.shape[1], hidden.shape[2])
    return ttnn.from_torch(
        flat, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def pos_tensors(mesh, positions):
    cur = ttnn.from_torch(
        positions.to(torch.int32),
        device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    rope = ttnn.from_torch(
        positions.reshape(1, -1).to(torch.int32),
        device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return cur, rope


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().to(torch.float64), b.flatten().to(torch.float64)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="sliding,full")
    ap.add_argument("--seq-lens", default="100,4097")
    ap.add_argument("--decode-steps", type=int, default=2)
    args = ap.parse_args()

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    ttnn.SetDefaultDevice(mesh)
    try:
        idxs = reference_layer_indices(R.hf_config())
        worst = 1.0
        for kind in args.kinds.split(","):
            layer_idx = idxs[kind]
            state_dict = R.synthetic_state_dict(layer_idx)
            ref_layer = R.reference_layer(layer_idx, state_dict)
            dec = OptimizedDecoder.from_state_dict(
                state_dict,
                hf_config=R.hf_config(),
                layer_idx=layer_idx,
                mesh_device=mesh,
                max_batch_size=1,
                max_seq_len=MAX_SEQ,
                page_block_size=PAGE_BLOCK,
                prefill_chunk_size=8192,
            )
            for seq_len in [int(s) for s in args.seq_lens.split(",")]:
                pt = page_table(mesh, 1, MAX_SEQ, seed=3)
                hidden = R.synthetic_hidden_states(1, seq_len, seed=42)
                ref_out, ref_cache = R.reference_prefill(ref_layer, layer_idx, hidden)
                tt_out = dec.prefill_forward(to_dev(mesh, hidden), page_table=pt, user_id=0)
                got = ttnn.to_torch(tt_out).reshape(1, seq_len, -1)
                p = pcc(got, ref_out)
                worst = min(worst, p)
                print(f"SMOKE prefill kind={kind:8s} seq_len={seq_len:6d} pcc={p:.6f}", flush=True)
                ttnn.deallocate(tt_out)

                for step in range(args.decode_steps):
                    pos = seq_len + step
                    token = R.synthetic_hidden_states(1, 1, seed=100 + step)
                    ref_dec = R.reference_decode(
                        ref_layer, layer_idx, token, past_key_values=ref_cache, positions=torch.tensor([pos])
                    )
                    cur, rope = pos_tensors(mesh, torch.tensor([pos]))
                    tt_dec = dec.decode_forward(to_dev(mesh, token), current_pos=cur, page_table=pt, rope_pos_ids=rope)
                    got = ttnn.to_torch(tt_dec).reshape(1, 1, -1)
                    p = pcc(got, ref_dec)
                    worst = min(worst, p)
                    print(f"SMOKE decode  kind={kind:8s} pos={pos:6d} pcc={p:.6f}", flush=True)
                    ttnn.deallocate(tt_dec)
                ttnn.deallocate(pt)
            del dec
        print(f"SMOKE_WORST_PCC {worst:.6f}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
