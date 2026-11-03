#!/usr/bin/env python3
"""Single-user prefill timing (and optional device profile) of the B32 serving model.

Builds the vLLM adapter like tools/perf_probe.py, warms one prefill of --isl tokens, then times
--repeat prefills; under the device profiler (tools/perf_profile.sh) the last prefill alone is
recorded. Usage: prefill_profile_probe.py --layers 0,1,2,3 --isl 512 [--repeat 2]
"""
import argparse
import json
import os
import time

p = argparse.ArgumentParser()
p.add_argument("--layers", default="0,1,2,3", help="comma list, or 'all'")
p.add_argument("--isl", type=int, default=512)
p.add_argument("--repeat", type=int, default=2)
args = p.parse_args()

CKPT = os.environ["QWEN_WEIGHTS"]
if args.layers != "all":
    os.environ["QWEN_FLASH_SERVING_LAYERS"] = args.layers
os.environ.setdefault("MODEL_WEIGHTS_DIR", CKPT)

import torch  # noqa: E402
from transformers import AutoConfig  # noqa: E402

import ttnn  # noqa: E402
from models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm import QwenFlashNextForCausalLM  # noqa: E402

BATCH, CONTEXT = 32, 262144
profiling = os.environ.get("TT_METAL_DEVICE_PROFILER", "0") not in ("", "0")


def main():
    torch.manual_seed(0)
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1073741824)
    adapter = None
    try:
        cfg = AutoConfig.from_pretrained(CKPT, local_files_only=True)
        t0 = time.time()
        adapter = QwenFlashNextForCausalLM.initialize_vllm_model(cfg, mesh, BATCH, CONTEXT)
        cache = adapter.allocate_kv_cache((CONTEXT // 32 + BATCH, 1, 32, 256), torch.float8_e4m3fn, 1)
        adapter.warmup_model_prefill(cache, enable_trace=True, can_sample_on_device=True)
        g = adapter.generator
        g.sampling_mode = "host"
        print("BUILD", round(time.time() - t0, 1), "s", flush=True)
        isl = args.isl
        tokens = torch.zeros(BATCH, isl, dtype=torch.int32)
        tokens[0] = torch.randint(100, g.model.cfg.vocab_size - 100, (isl,), dtype=torch.int32)
        pages = torch.zeros(BATCH, CONTEXT // 32, dtype=torch.int32)
        pages[0, : (isl + 31) // 32] = torch.arange((isl + 31) // 32)
        lens = [isl] + [0] * (BATCH - 1)
        one = [True] + [False] * (BATCH - 1)
        times = []
        for i in range(args.repeat):
            g.reset()
            ttnn.synchronize_device(mesh)
            if profiling and i == args.repeat - 1:
                ttnn.ReadDeviceProfiler(mesh)  # drop build/warm records; keep only the last prefill
            t = time.perf_counter()
            g.prefill_forward(
                tokens,
                page_table=pages,
                kv_cache=g.kv_cache,
                prompt_lens=lens,
                slots=list(range(BATCH)),
                active_rows=one,
            )
            ttnn.synchronize_device(mesh)
            times.append(time.perf_counter() - t)
        if profiling:
            ttnn.ReadDeviceProfiler(mesh)
        print(
            "PREFILL", json.dumps({"layers": args.layers, "isl": isl, "s": times, "tok_s": isl / times[-1]}), flush=True
        )
    finally:
        if adapter is not None:
            adapter.release_persistent_capture()
        ttnn.close_mesh_device(mesh)


main()
