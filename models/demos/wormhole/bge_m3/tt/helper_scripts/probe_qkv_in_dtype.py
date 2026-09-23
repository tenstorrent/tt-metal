"""Does the QKV matmul get faster when its activation arrives bfloat8_b?

The B8 Tracy capture reports the QKV projection as "LoFi BF16 x BFP8 => BFP8":
the weight is bf8 and the output is bf8, but the activation is still bf16,
because it comes from a LayerNorm that emits the model dtype it was given.
Halving that read is item 2.

Measured the same way as sdpa_sweep.py: device kernel duration per launch from
the profiler's per-op list, read after the device closes.
"""

import os

import torch

import ttnn

BATCH, SEQ = 8, 512
HIDDEN, QKV_WIDTH = 1024, 3072
WARMUP, MEASURED = 2, 5


def launch_durations_us():
    from tracy.device_post_proc_config import default_setup
    from tracy.process_device_log import import_log_run_stats

    setup = default_setup()
    if not os.path.exists(setup.deviceInputLog):
        return []
    data = import_log_run_stats(setup)
    devices = data.get("devices") or {}
    if 0 not in devices:
        return []
    freq = data["deviceInfo"]["freq"]
    risc = devices[0]["cores"]["DEVICE"]["riscs"]["TENSIX"]
    return [
        op["analysis"]["device_kernel_duration"]["stats"]["Max"] / freq
        for op in risc.get("ops", [])
        if op.get("analysis", {}).get("device_kernel_duration")
    ]


log = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated", "profiler", ".logs", "profile_log_device.csv")
if os.path.exists(log):
    os.remove(log)

device = ttnn.open_device(device_id=0)
from models.demos.wormhole.bge_m3.tt.optimizations import Optimizations

opts = Optimizations.build(device, max_batch_size=BATCH, max_seq_len=SEQ, dtype=ttnn.bfloat8_b)
attn = opts.attention

torch.manual_seed(0)
weight_host = torch.randn((HIDDEN, QKV_WIDTH), dtype=torch.bfloat16)
weight = ttnn.from_torch(
    weight_host.reshape(1, 1, HIDDEN, QKV_WIDTH),
    device=device,
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

order = []
for label, in_dtype in (("bf16 activation (today)", ttnn.bfloat16), ("bf8 activation (item 2)", ttnn.bfloat8_b)):
    act = ttnn.from_torch(
        torch.randn((1, 1, BATCH * SEQ, HIDDEN), dtype=torch.bfloat16),
        device=device,
        dtype=in_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for _ in range(WARMUP + MEASURED):
        out = ttnn.linear(
            act,
            weight,
            memory_config=attn.qkv_memcfg,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=attn.qkv_compute_kernel_cfg,
            core_grid=attn.core_grid if attn.qkv_prg_config is None else None,
            program_config=attn.qkv_prg_config,
        )
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    ttnn.deallocate(act)
    order.append(label)

ttnn.close_device(device)

per = WARMUP + MEASURED
durations = launch_durations_us()
print("  launches recorded: %d for %d expected" % (len(durations), per * len(order)))
results = {}
for index, label in enumerate(order):
    window = durations[index * per : (index + 1) * per][WARMUP:]
    if window:
        results[label] = min(window)
        print("  %-26s %7.1f us" % (label, min(window)))
if len(results) == 2:
    a, b = (results[k] for k in order)
    print("  saving %.1f us/call (%.1f%%)  ->  x24 layers = %.2f ms" % (a - b, 100 * (a - b) / a, 24 * (a - b) / 1000))
