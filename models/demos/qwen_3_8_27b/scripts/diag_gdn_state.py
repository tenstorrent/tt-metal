# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DIAGNOSTIC (reduced sequence, single layer — never a graded result): which GDN weight's dtype drives the
recurrent-state error on the real checkpoint. One GDN layer is fed the fp32 reference's own input (from
``diag_layer_drift.py``'s cache) and its recurrent / conv state and output are compared with the fp32
reference layer, for several per-part weight dtypes.

    python models/demos/qwen_3_8_27b/scripts/diag_gdn_state.py LAYER [N]
"""

import sys

import torch

import ttnn
from models.demos.qwen_3_8_27b.config import QWEN38, PrefillSpec
from models.demos.qwen_3_8_27b.reference import qwen3_8_ref as ref
from models.demos.qwen_3_8_27b.reference.checkpoint import CheckpointReader
from models.demos.qwen_3_8_27b.tests.common import from_sp, pcc, to_sp
from models.demos.qwen_3_8_27b.tt.gdn import TtGatedDeltaNet
from models.demos.qwen_3_8_27b.tt.mesh import MeshConfig, close_mesh, open_mesh


def main():
    L = int(sys.argv[1])
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    spec = PrefillSpec.load()
    ref_h = torch.load(f"/tmp/qwen38_diag/ref_hidden_{N}.pt")
    sd = CheckpointReader().layer(L)
    layer = ref.DecoderLayer(QWEN38, L)
    layer.load_state_dict(sd)
    layer = layer.float()
    with torch.no_grad():
        h = layer.input_layernorm(ref_h[L - 1].float())
        want, want_conv, want_rec = layer.linear_attn(h)
    gsd = {k[len("linear_attn.") :]: v for k, v in sd.items() if k.startswith("linear_attn.")}
    mesh = open_mesh(spec.mesh_shape)
    try:
        mc = MeshConfig(mesh, spec.sp, spec.tp)
        bf8, bf16 = ttnn.bfloat8_b, ttnn.bfloat16
        variants = {
            "all bf8 (spec)": {},
            "ab bf16": {"w_ab": bf16},
            "qkvz bf16": {"w_qkvz": bf16},
            "out bf16": {"w_out": bf16},
            "all bf16": {"w_ab": bf16, "w_qkvz": bf16, "w_out": bf16},
        }
        print(f"GDN layer {L}, N={N} (REDUCED, reference input):  out / recurrent / conv PCC")
        for name, pdt in variants.items():
            tt = TtGatedDeltaNet(mc, QWEN38, gsd, weight_dtype=bf8, part_dtypes=pdt)
            st = {}
            out = from_sp(tt(to_sp(h[None].to(torch.bfloat16), mc), st), mc)[0]
            rec, conv = tt.read_state(st)
            print(f"  {name:16s} {pcc(out, want):.6f} / {pcc(rec, want_rec):.6f} / {pcc(conv, want_conv):.6f}")
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
