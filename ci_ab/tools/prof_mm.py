# ttnn.linear at model matmul shapes on the Metal 2.0 bmm fork (scratch, not for merge), under
# `python -m tracy -r -p --no-web-server -o <dir> prof_mm.py <case>...`: per case 2 warm-up calls, a signpost "<case>",
# SM_NMEAS measured calls; DRAM interleaved inputs, auto program config. Prints BITS <case> <md5 of the output>.
import gc
import hashlib
import os
import sys

import torch
import ttnn
from tracy import signpost

N_WARM, N_MEAS = 2, int(os.environ.get("SM_NMEAS", "12"))
DT = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b, "bf4": ttnn.bfloat4_b}
FID = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}


def tens(d, shape, dt, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g) * 0.1
    return ttnn.from_torch(x, dtype=DT[dt], layout=ttnn.TILE_LAYOUT, device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def lin(m, k, n, wdt, fid, fp32=False, bias=False, batch=1):
    def setup(d):
        a = tens(d, (1, batch, m, k), "bf16", 1)
        w = tens(d, (k, n), wdt, 2)
        b = tens(d, (1, n), "bf16", 3) if bias else None
        c = ttnn.init_device_compute_kernel_config(d.arch(), math_fidelity=FID[fid], math_approx_mode=False,
                                                   fp32_dest_acc_en=fp32, packer_l1_acc=True)
        return (a, w, b, c)

    def run(d, a, w, b, c):
        return ttnn.linear(a, w, bias=b, compute_kernel_config=c, dtype=ttnn.bfloat16,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return setup, run


CASES = {
    # Llama 3.1-8B
    "llama8b_prefill_qkv_hifi2": lin(1024, 4096, 6144, "bf8", "hifi2"),
    "llama8b_prefill_ff1_lofi": lin(1024, 4096, 14336, "bf4", "lofi"),
    "llama8b_prefill_ff1_hifi2": lin(1024, 4096, 14336, "bf8", "hifi2"),
    "llama8b_prefill_ff2_hifi2": lin(1024, 14336, 4096, "bf8", "hifi2"),
    "llama8b_prefill_wo_hifi2": lin(1024, 4096, 4096, "bf8", "hifi2"),
    "llama8b_decode_qkv_hifi2": lin(32, 4096, 6144, "bf8", "hifi2"),
    "llama8b_decode_ff1_lofi": lin(32, 4096, 14336, "bf4", "lofi"),
    "llama8b_decode_ff2_hifi2": lin(32, 14336, 4096, "bf8", "hifi2"),
    # Qwen3-32B (4-way TP slices)
    "qwen32b_prefill_ff1_hifi2": lin(1024, 5120, 6400, "bf8", "hifi2"),
    "qwen32b_prefill_ff2_hifi2": lin(1024, 6400, 5120, "bf8", "hifi2"),
    "qwen32b_prefill_qkv_hifi4": lin(1024, 5120, 2560, "bf16", "hifi4", fp32=True),
    # DeepSeek V3
    "dsv3_q_a_hifi2": lin(1024, 7168, 1536, "bf8", "hifi2"),
    "dsv3_q_b_hifi2": lin(1024, 1536, 3072, "bf8", "hifi2"),
    "dsv3_expert_w1_lofi": lin(256, 7168, 2048, "bf4", "lofi"),
    "dsv3_expert_w2_hifi2": lin(256, 2048, 7168, "bf8", "hifi2"),
    # BERT-large, batch 8 x 384, with bias
    "bert_qkv_hifi4_bias": lin(384, 1024, 3072, "bf8", "hifi4", bias=True, batch=8),
    "bert_ff1_hifi4_bias": lin(384, 1024, 4096, "bf8", "hifi4", bias=True, batch=8),
    "bert_ff2_hifi4_bias": lin(384, 4096, 1024, "bf8", "hifi4", bias=True, batch=8),
    # more LoFi (bf4 and bf8 weights)
    "qwen32b_prefill_ff1_lofi": lin(1024, 5120, 6400, "bf4", "lofi"),
    "qwen32b_decode_ff1_lofi": lin(32, 5120, 6400, "bf4", "lofi"),
    "llama70b_prefill_ff1_lofi": lin(1024, 8192, 3584, "bf4", "lofi"),
    "llama8b_prefill_ff1_lofi_bf8": lin(1024, 4096, 14336, "bf8", "lofi"),
    "bert_ff1_lofi_bias": lin(384, 1024, 4096, "bf8", "lofi", bias=True, batch=8),
    "sq2048_lofi_bf8": lin(2048, 2048, 2048, "bf8", "lofi"),
    # the body's square shape
    "sq2048_lofi": lin(2048, 2048, 2048, "bf16", "lofi"),
    "sq2048_hifi2": lin(2048, 2048, 2048, "bf16", "hifi2"),
    "sq2048_hifi4": lin(2048, 2048, 2048, "bf16", "hifi4"),
}


def main():
    names = sys.argv[1:] or list(CASES)
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        for name in names:
            setup, run = CASES[name]
            gc.collect()
            try:
                args = setup(dev)
                for _ in range(N_WARM):
                    run(dev, *args)
                ttnn.synchronize_device(dev)
                signpost(header=name)
                for _ in range(N_MEAS):
                    y = run(dev, *args)
                ttnn.synchronize_device(dev)
                ttnn.ReadDeviceProfiler(dev)
                t = ttnn.to_torch(y).contiguous()
                h = hashlib.md5(t.view(torch.int16).numpy().tobytes()).hexdigest()[:12]
                print(f"BITS {name} {h}", flush=True)
                print(f"CASE {name} ok", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"CASE {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
