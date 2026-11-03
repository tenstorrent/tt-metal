#!/usr/bin/env python3
"""Census of ttnn.reshape calls in one eager serial-prefill bucket (default 1024), by model call site.

Wraps ttnn.reshape while SerialPrefillView.warm() runs the chosen bucket and prints, per call site,
the count, layout and shapes, so the device-copying reshapes in a prefill profile can be located.
Usage: prefill_reshape_census.py --layers 0,1,2,3 --bucket 1024
"""
import argparse
import collections
import os
import traceback

p = argparse.ArgumentParser()
p.add_argument("--layers", default="0,1,2,3")
p.add_argument("--bucket", type=int, default=1024)
args = p.parse_args()
os.environ["QWEN_FLASH_SERVING_LAYERS"] = args.layers
os.environ.setdefault("MODEL_WEIGHTS_DIR", os.environ["QWEN_WEIGHTS"])

import torch  # noqa: E402
from transformers import AutoConfig  # noqa: E402

import ttnn  # noqa: E402
from models.autoports.qwen_qwen3_8_flash_next.tt import prefill_view  # noqa: E402
from models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm import QwenFlashNextForCausalLM  # noqa: E402

state = {"on": False}
sites = collections.Counter()
original_reshape = ttnn.reshape


def reshape(x, shape, *a, **k):
    if state["on"]:
        frame = next(
            (f for f in reversed(traceback.extract_stack()[:-1]) if "qwen_qwen3_8_flash_next/tt/" in f.filename), None
        )
        where = f"{os.path.basename(frame.filename)}:{frame.lineno}" if frame else "?"
        layout = "TILE" if x.layout == ttnn.TILE_LAYOUT else "RM"
        sites[(where, layout, str(x.dtype).split(".")[-1], tuple(x.shape), tuple(x.padded_shape), tuple(shape))] += 1
    return original_reshape(x, shape, *a, **k)


ttnn.reshape = reshape
original_forward = prefill_view.SerialPrefillView._forward


def forward(self, n):
    state["on"] = n == args.bucket
    try:
        return original_forward(self, n)
    finally:
        state["on"] = False


prefill_view.SerialPrefillView._forward = forward


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1073741824)
    adapter = None
    try:
        cfg = AutoConfig.from_pretrained(os.environ["QWEN_WEIGHTS"], local_files_only=True)
        adapter = QwenFlashNextForCausalLM.initialize_vllm_model(cfg, mesh, 32, 262144)
        cache = adapter.allocate_kv_cache((262144 // 32 + 32, 1, 32, 256), torch.float8_e4m3fn, 1)
        adapter.warmup_model_prefill(cache, enable_trace=True, can_sample_on_device=True)
        for (where, layout, dtype, shape, padded, target), count in sorted(sites.items()):
            print(
                f"RESHAPE {where:28s} x{count:<3d} {layout:4s} {dtype:9s} {shape} pad{padded} -> {target}", flush=True
            )
    finally:
        if adapter is not None:
            adapter.release_persistent_capture()
        ttnn.close_mesh_device(mesh)


main()
