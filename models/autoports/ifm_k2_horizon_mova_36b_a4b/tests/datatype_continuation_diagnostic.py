# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Discriminate trace state from quantized prefill/decode segmentation numerics."""

import argparse
import json
from pathlib import Path

import torch

import ttnn

from ..tt.generator import build_generator

ROOT = Path(__file__).resolve().parents[1]


def compare(a, b):
    return dict(
        pcc=float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1]),
        bitwise_equal=torch.equal(a, b),
        top1_equal=torch.equal(a.argmax(-1), b.argmax(-1)),
        top1_a=a.argmax(-1).tolist(),
        top1_b=b.argmax(-1).tolist(),
        max_abs=float((a - b).abs().max()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--precision-config")
    ap.add_argument("--context", type=int, default=524288)
    args = ap.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    try:
        kw = {} if args.precision_config is None else dict(precision_config=args.precision_config)
        gen = build_generator(ROOT, mesh, override_num_layers=2, max_seq_len=args.context, **kw)
        gen.capture()
        prompt = (torch.arange(129) * 11 + 3001)[None]
        pages = torch.roll(gen.page_table_host, gen.pages_per_slot // 2, dims=-1)
        values = {}
        for name, traced, start in [
            ("traced_segmented", True, 33),
            ("eager_segmented", False, 33),
            ("eager_monolithic", False, 0),
            ("traced_monolithic", True, 0),
            ("traced_segmented_repeated", True, 33),
        ]:
            if start:
                gen.prefill_forward(
                    prompt, page_table=pages, kv_cache=gen.kv_cache, prompt_lens=[start], enable_trace=traced
                )
            values[name] = gen.prefill_forward(
                prompt,
                page_table=pages,
                kv_cache=gen.kv_cache,
                start_pos=[start],
                prompt_lens=[129],
                enable_trace=traced,
            )
        result = dict(
            config_id=gen.model.precision["config_id"],
            context=args.context,
            layers=2,
            comparisons={
                f"{a}_vs_{b}": compare(values[a], values[b])
                for a, b in [
                    ("traced_segmented", "eager_segmented"),
                    ("traced_segmented", "eager_monolithic"),
                    ("eager_segmented", "eager_monolithic"),
                    ("traced_monolithic", "eager_monolithic"),
                    ("traced_segmented", "traced_segmented_repeated"),
                ]
            },
            counters=dict(gen.counters),
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        gen.close()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
