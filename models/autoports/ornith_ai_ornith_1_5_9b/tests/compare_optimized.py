# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight, same-input fused versus optimized layer/state equivalence."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import ROOT, config, pcc, weights
from models.autoports.ornith_ai_ornith_1_5_9b.tests.validate_optimized import buffers, clear, device_only, upload
from models.autoports.ornith_ai_ornith_1_5_9b.tt.fused_decoder import FusedDecoder as BaselineDecoder
from models.autoports.ornith_ai_ornith_1_5_9b.tt.optimized_decoder import OptimizedDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lengths", default="1,2,3,4,31,32,33,1023,1024,1025,2051")
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(73)
    c = config()
    sd = weights(args.layer)
    records = []
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        layers = [
            cls.from_state_dict(sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh)
            for cls in [BaselineDecoder, OptimizedDecoder]
        ]
        states = [layer.allocate_state(context=2080) for layer in layers]
        table = upload(torch.randperm(65, dtype=torch.int32)[None], mesh, True)

        def check(label, values, residual=None):
            a, b = [ttnn.to_torch(v) for v in values]
            row = {"label": label, "pcc": pcc(a, b)}
            if residual is not None:
                row["contribution_pcc"] = pcc(a - residual, b - residual)
            assert all(v >= 0.995 for k, v in row.items() if "pcc" in k), row
            records.append(row)

        for n in map(int, args.lengths.split(",")):
            x = (torch.randn(1, 1, n, 4096) * 0.1).bfloat16()
            tx = upload(x, mesh)
            for state in states:
                clear(state)
            with device_only():
                outputs = [layer.prefill_forward(tx, state=st, page_table=table) for layer, st in zip(layers, states)]
            check(f"prefill_{n}", outputs, x)
            for index, pair in enumerate(zip(*[buffers(s) for s in states])):
                check(f"prefill_state_{n}_{index}", pair)
            # Advance through distinct inputs and page positions without restoring state.
            # This catches accumulated recurrent/cache errors hidden by restored replay tests.
            for step in range(args.steps):
                x = (torch.randn(1, 1, 1, 4096) * 0.1).bfloat16()
                dx = upload(x, mesh)
                pos = upload(torch.tensor([n + step], dtype=torch.int32), mesh, True)
                with device_only():
                    outputs = [
                        layer.decode_forward(dx, state=st, page_table=table, current_position=pos)
                        for layer, st in zip(layers, states)
                    ]
                check(f"decode_{n}_{step}", outputs, x)
            for index, pair in enumerate(zip(*[buffers(s) for s in states])):
                check(f"decode_state_{n}_{index}", pair)
        Path(args.output).write_text(
            json.dumps(
                {
                    "layer": args.layer,
                    "weights": "real",
                    "records": records,
                    "implementations": {
                        name: hashlib.sha256((ROOT / f"tt/{name}.py").read_bytes()).hexdigest()
                        for name in ["fused_decoder", "optimized_decoder"]
                    },
                },
                indent=2,
            )
            + "\n"
        )
        print("PAIRED_EQUIVALENCE_PASS", args.output)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
