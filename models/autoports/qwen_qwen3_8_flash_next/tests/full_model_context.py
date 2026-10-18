"""Public non-aligned and exact-context execution with trace/input-state evidence."""

import argparse
import json
import time
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator
from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_capacity_probe import view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--layers", default="0,1,3")
    ap.add_argument("--context", type=int, default=262144)
    ap.add_argument("--tag", default="context_contract_probe")
    ap.add_argument("--stage", default="full_model")
    ap.add_argument("--prefill-trace-max", type=int, default=None)
    a = ap.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=768000000 if a.layers == "all" else 160000000)
    gen = None
    try:
        indices = None if a.layers == "all" else [int(s) for s in a.layers.split(",")]
        gen = build_generator(
            "models/autoports/qwen_qwen3_8_flash_next",
            mesh,
            checkpoint=a.checkpoint,
            prefill_trace_max=a.prefill_trace_max,
            layer_indices=indices,
            max_seq_len=a.context,
        )
        gen.prepare()
        ids = dict(gen.traces)
        rows = []
        for length, count in ((a.context - 3, 4), (a.context, 1)):
            tick = time.perf_counter()
            tokens = gen.generate([100 + i % 300 for i in range(length)], count)
            pos = ttnn.to_torch(ttnn.get_device_tensors(gen.tt_pos)[0])[:, 0].tolist()
            assert len(tokens) == count and pos == [a.context], (length, tokens, pos)
            assert gen.traces == ids and gen.counters["captures"] == len(ids) and gen.counters["releases"] == 0
            rows.append(
                dict(
                    length=length,
                    generated_tokens=tokens,
                    final_position=pos,
                    seconds=time.perf_counter() - tick,
                    perf=dict(gen.last_perf),
                )
            )
            print("CONTEXT_BOUNDARY_PASS", rows[-1], flush=True)
        result = dict(args=vars(a), rows=rows, memory=view(mesh), counters=dict(gen.counters))
        Path(f"models/autoports/qwen_qwen3_8_flash_next/doc/{a.stage}/{a.tag}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
