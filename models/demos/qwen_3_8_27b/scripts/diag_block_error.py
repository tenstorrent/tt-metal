# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DIAGNOSTIC (reduced sequence — never a graded result): per-block relative error of the device vs a CPU
bf16 run of the same block, both against the fp32 reference, all fed the fp32 reference's own input.
A block whose device error is much larger than its CPU-bf16 error is where the device loses accuracy
beyond the bf16 floor.

    python models/demos/qwen_3_8_27b/scripts/diag_block_error.py N LAYER [LAYER ...]
"""

import sys

import torch

from models.demos.qwen_3_8_27b.config import QWEN38, PrefillSpec
from models.demos.qwen_3_8_27b.reference import qwen3_8_ref as ref
from models.demos.qwen_3_8_27b.reference.checkpoint import CheckpointReader
from models.demos.qwen_3_8_27b.tests.common import from_sp, to_sp
from models.demos.qwen_3_8_27b.tt.context import PrefillCtx
from models.demos.qwen_3_8_27b.tt.kv_cache import allocate_caches, cache_capacity
from models.demos.qwen_3_8_27b.tt.layer import TtDecoderLayer
from models.demos.qwen_3_8_27b.tt.mesh import MeshConfig, close_mesh, make_ccl_manager, open_mesh
from models.demos.qwen_3_8_27b.tt.rope import TtRope


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main():
    N = int(sys.argv[1])
    layers = [int(x) for x in sys.argv[2:]]
    spec = PrefillSpec.load()
    ref_h = torch.load(f"/tmp/qwen38_diag/ref_hidden_{N}.pt")
    reader = CheckpointReader()
    mesh = open_mesh(spec.mesh_shape)
    try:
        mc = MeshConfig(mesh, spec.sp, spec.tp)
        ccl = make_ccl_manager(mesh)
        rope = TtRope(mc, QWEN38)
        print(f"N={N} REDUCED. rel-err vs fp32 ref:  mixer(device / cpu-bf16)   mlp(device / cpu-bf16)")
        for L in layers:
            sd = reader.layer(L)
            R = ref.DecoderLayer(QWEN38, L)
            R.load_state_dict(sd)
            x = ref_h[L - 1].float()
            cos, sin = ref.rope_cos_sin(QWEN38, torch.arange(N), dtype=torch.float32)
            with torch.no_grad():
                Rf = R.float()
                h = Rf.input_layernorm(x)
                m = Rf.self_attn(h, cos, sin)[0] if R.is_full else Rf.linear_attn(h)[0]
                x2 = x + m
                h2 = Rf.post_attention_layernorm(x2)
                f = Rf.mlp(h2)
                Rb = R.to(torch.bfloat16)
                cb, sb = cos.bfloat16(), sin.bfloat16()
                hb = h.bfloat16()
                mb = Rb.self_attn(hb, cb, sb)[0] if R.is_full else Rb.linear_attn(hb)[0]
                fb = Rb.mlp(h2.bfloat16())
            tt = TtDecoderLayer(mc, ccl, QWEN38, sd, L, spec)
            caches = allocate_caches(
                mesh, num_attn_layers=16, gdn_layers=[L], max_seq_len=cache_capacity(N, [N]), head_dim=QWEN38.head_dim
            )
            ctx = PrefillCtx(caches, 0, 0, N, N)
            ctx.cos, ctx.sin = rope.tables(0, N // mc.sp)
            hd = to_sp(h[None].bfloat16(), mc)
            md = tt.mixer(hd, ctx, rope) if tt.is_full else tt.mixer(hd, caches.gdn_state(0, L))
            fd = tt.mlp(to_sp(h2[None].bfloat16(), mc))
            md, fd = from_sp(md, mc)[0], from_sp(fd, mc)[0]
            print(
                f"  layer {L:2d} {QWEN38.layer_types[L]:17s}  {rel(md, m):.5f} / {rel(mb, m):.5f}      {rel(fd, f):.5f} / {rel(fb, f):.5f}"
            )
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
