# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced top-1 / top-5 of the ttnn TP model (bespoke prefill_tp / decode_tp) against a Hugging Face
reference made by qwen36_resident's hf_teacher_ref.py: prefill PREFIX tokens, then decode feeding the true tokens.

usage: python -m models.demos.blackhole.qwen36.tests.teacher_forced_hf_scratch <ref.pt> <out.json> [prefix]
"""
import json
import sys

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model


@torch.no_grad()
def main():
    ref_path, out_path = sys.argv[1], sys.argv[2]
    P = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    assert P % 128 == 0, "prefill_tp pads to the GDN chunk; keep the prefix a multiple of 128"
    ref = torch.load(ref_path)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, num_command_queues=2)
    try:
        model = Qwen36Model.from_pretrained(mesh, max_batch_size=1, max_seq_len=1024)
        vocab = model.args.vocab_size
        out = dict(prefix=P, teacher=[], mismatch=[])
        top1 = top5 = total = 0
        for i, s in enumerate(ref["seqs"]):
            ids = s["ids"]
            last = len(ids) - 2
            model.reset_tp()
            lg = model.prefill_tp(ids[None, :P].long(), valid_len=P)
            lg = (ttnn.to_torch(lg) if not isinstance(lg, torch.Tensor) else lg).reshape(-1)[:vocab].float()
            h1 = h5 = 0
            for p in range(P - 1, last + 1):
                if p >= P:
                    lg = model.decode_tp(int(ids[p]), p)
                    lg = (ttnn.to_torch(lg) if not isinstance(lg, torch.Tensor) else lg).reshape(-1)[:vocab].float()
                want = int(s["top_i"][p, 0])
                dev1 = int(lg.argmax())
                h1 += dev1 == want
                h5 += want in lg.topk(5).indices.tolist()
                if dev1 != want:
                    hit = (s["top_i"][p] == dev1).nonzero()
                    hf_gap = float(s["top_v"][p, 0] - s["top_v"][p, hit[0, 0]]) if len(hit) else float("inf")
                    out["mismatch"].append(dict(seq=i, pos=p, dev=dev1, hf=want, hf_gap=hf_gap))
            n = last - P + 2
            top1, top5, total = top1 + h1, top5 + h5, total + n
            out["teacher"].append(dict(seq=i, top1=h1 / n, top5=h5 / n, scored=n))
            print(f"ttnn teacher seq {i}: top-1 {h1 / n:.4f} top-5 {h5 / n:.4f} ({n} positions)", flush=True)
        out.update(top1=top1 / total, top5=top5 / total, scored=total)
        print(f"ttnn teacher all {total}: top-1 {top1 / total:.4f} top-5 {top5 / total:.4f}", flush=True)
        json.dump(out, open(out_path, "w"), indent=1)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
