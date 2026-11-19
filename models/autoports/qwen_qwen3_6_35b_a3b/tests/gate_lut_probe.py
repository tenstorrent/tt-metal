# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic for setup-folded BF16-domain recurrent gate tables."""
import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import weights


def run(args):
    record(args.output)
    torch.set_num_threads(8)
    sd = weights(0)
    values = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16).float()
    bias = sd["linear_attn.dt_bias"].float().reshape(32, 1)
    scale = -sd["linear_attn.A_log"].float().exp().reshape(32, 1)
    gate = scale * torch.nn.functional.softplus(values.reshape(1, -1) + bias)
    decay = gate.exp()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    rows = []
    try:
        tg, te = [ttnn.from_torch(v, device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT) for v in (gate, decay)]
        for pos in (128, 129, 130, 1024, 3979, 5011, 7551, 7703, 8157):
            capture = torch.load(f"{args.input}.pos{pos}.pt", weights_only=False)
            a = capture["hf_values"]["projection_a"].reshape(32).bfloat16()
            expected_g = scale.reshape(32) * torch.nn.functional.softplus(a.float() + bias.reshape(32))
            ta = ttnn.from_torch(a.reshape(1, 1, 32), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            indices = ttnn.reshape(ttnn.bitcast(ta, ttnn.uint16), (32, 1))
            got_g = ttnn.to_torch(ttnn.gather(tg, 1, indices)).reshape(32)
            got_e = ttnn.to_torch(ttnn.gather(te, 1, indices)).reshape(32)
            rows.append(
                dict(
                    position=pos,
                    gate_different=int((got_g != expected_g).sum()),
                    decay_different=int((got_e != expected_g.exp()).sum()),
                )
            )
        Path(args.output).write_text(json.dumps(dict(table_bytes=gate.numel() * 8, rows=rows), indent=2) + "\n")
        print("GATE_LUT", rows, flush=True)
        assert all(r["gate_different"] == r["decay_different"] == 0 for r in rows)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance_precise_probe.json"
    )
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
