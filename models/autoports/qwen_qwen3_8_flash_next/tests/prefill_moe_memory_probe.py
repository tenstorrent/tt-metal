#!/usr/bin/env python3
"""DRAM per bank right after the 1024-row prefill MoE group allocates its capacity buffer (eager warm-up).

The capacity buffer is transient (allocated inside group1024), so the post-build DRAM view does not show
it; this probe wraps ttnn.empty and reports allocated/free MB per bank at that moment.
Usage: prefill_moe_memory_probe.py --layers 0,1,2,3
"""
import argparse
import os

p = argparse.ArgumentParser()
p.add_argument("--layers", default="0,1,2,3")
args = p.parse_args()
os.environ["QWEN_FLASH_SERVING_LAYERS"] = args.layers
os.environ.setdefault("MODEL_WEIGHTS_DIR", os.environ["QWEN_WEIGHTS"])

import torch  # noqa: E402
from transformers import AutoConfig  # noqa: E402

import ttnn  # noqa: E402
from models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm import QwenFlashNextForCausalLM  # noqa: E402

original_empty = ttnn.empty
peaks = []


def empty(shape, *a, **k):
    out = original_empty(shape, *a, **k)
    shape = tuple(shape)
    if len(shape) == 4 and shape[-1] == 2560 and shape[2] >= 10 * 1024:
        view = ttnn.get_memory_view(out.device(), ttnn.BufferType.DRAM)
        peaks.append(
            (shape[2], view.total_bytes_allocated_per_bank // 2**20, view.total_bytes_free_per_bank // 2**20)
        )
    return out


ttnn.empty = empty


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1073741824)
    adapter = None
    try:
        cfg = AutoConfig.from_pretrained(os.environ["QWEN_WEIGHTS"], local_files_only=True)
        adapter = QwenFlashNextForCausalLM.initialize_vllm_model(cfg, mesh, 32, 262144)
        cache = adapter.allocate_kv_cache((262144 // 32 + 32, 1, 32, 256), torch.float8_e4m3fn, 1)
        adapter.warmup_model_prefill(cache, enable_trace=True, can_sample_on_device=True)
        for rows, allocated, free in sorted(set(peaks)):
            print(f"MOE_CAPACITY rows={rows} DRAM per bank MB allocated={allocated} free={free}", flush=True)
    finally:
        if adapter is not None:
            adapter.release_persistent_capture()
        ttnn.close_mesh_device(mesh)


main()
