"""Reduced real-weight model and terminal-path probe; no acceptance substitution."""

import argparse, json, time
from pathlib import Path
import torch, ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_components import CHECKPOINT
from models.autoports.qwen_qwen3_8_flash_next.tt.model import QwenFlashNextModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", default="0,1,3")
    p.add_argument("--generator", action="store_true")
    p.add_argument("--length", type=int, default=33)
    p.add_argument("--context", type=int, default=262144)
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    a = p.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=160000000)
    try:
        start = time.perf_counter()
        model = QwenFlashNextModel(
            a.checkpoint,
            mesh,
            layer_indices=[int(x) for x in a.layers.split(",") if x],
            max_seq_len=a.context,
            cache_dir="/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/de4b8e4_ple",
        )
        if a.generator:
            from transformers import AutoTokenizer
            from models.autoports.qwen_qwen3_8_flash_next.tt.generator import QwenFlashNextGenerator

            gen = QwenFlashNextGenerator(model, AutoTokenizer.from_pretrained(a.checkpoint, local_files_only=True))
            print("GENERATOR_SETUP", flush=True)
            output = gen.generate(list(range(100, 100 + a.length)), 4)
            print("GENERATOR_OUTPUT", output, dict(gen.counters), gen.last_perf, flush=True)
            gen.close()
            return
        model.prepare([a.length])
        ids = torch.arange(a.length).reshape(1, -1).int() + 100
        tokens = model.ops.upload(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        pos = model.ops.upload(torch.zeros(1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        slots = model.ops.upload(torch.zeros(1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        page = model.ops.upload(torch.arange(a.context // 32).reshape(1, -1).int(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        print("PREFILL_START", flush=True)
        logits = model.forward(tokens, positions=pos, slots=slots, page_table=page, kv_cache=model.cache)
        ttnn.synchronize_device(mesh)
        print("PREFILL_OK", logits.shape, flush=True)
        ids = ids[:, -1:]
        tokens = model.ops.upload(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        ph = torch.zeros(1, 32, dtype=torch.int32)
        ph[:, 0] = a.length
        pos = model.ops.upload(ph, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        print("DECODE_START", flush=True)
        out = model.forward(tokens, positions=pos, slots=slots, page_table=page, kv_cache=model.cache, decode=True)
        ttnn.synchronize_device(mesh)
        print("DECODE_OK", out.shape, flush=True)
        print("PROBE_SECONDS", time.perf_counter() - start, flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
