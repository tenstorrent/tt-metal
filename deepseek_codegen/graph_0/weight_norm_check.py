# SPDX-License-Identifier: Apache-2.0
"""Localize the ~1.276x routed-output scale: compare the norm of the source ce_cache weights vs the
bf4-prepped moe_compute weights. Norm is invariant to reshape/transpose/zero-pad, so a ratio != 1.0
means the prepare/quantize step systematically rescales the weights."""
import os
import sys

sys.path.insert(0, "/home/ubuntu/tt-metal/deepseek_codegen/graph_0")
os.chdir("/home/ubuntu/tt-metal/deepseek_codegen/graph_0")
import torch  # noqa: E402
import ttnn  # noqa: E402
import utils  # noqa: E402

dev = utils.DeviceGetter.get_device((4, 8))


def total_norm(path):
    t = ttnn.load_tensor(path, device=dev)
    shards = ttnn.get_device_tensors(t)
    n2 = 0.0
    for s in shards:
        x = ttnn.to_torch(s).float()
        n2 += float((x * x).sum())
    return n2**0.5, tuple(shards[0].shape), len(shards)


g_n, g_sh, g_nd = total_norm("moe_io/ce_cache/main_const_eval_gate_up.tensorbin")
w2_n, w2_sh, w2_nd = total_norm("moe_io/ce_cache/main_const_eval_39.tensorbin")
p01_n, p01_sh, p01_nd = total_norm("moe_io/wcache/tt_w0w1.tensorbin")
p2_n, p2_sh, p2_nd = total_norm("moe_io/wcache/tt_w2.tensorbin")
print(f"[wnorm] source gate_up : norm={g_n:.3f} shard{g_sh} nd={g_nd}")
print(f"[wnorm] source w2(_39) : norm={w2_n:.3f} shard{w2_sh} nd={w2_nd}")
print(f"[wnorm] prepped w0w1bf4: norm={p01_n:.3f} shard{p01_sh} nd={p01_nd}")
print(f"[wnorm] prepped w2 bf4 : norm={p2_n:.3f} shard{p2_sh} nd={p2_nd}")
print(f"[wnorm] RATIO prepped_w0w1 / source_gate_up = {p01_n / (g_n + 1e-9):.4f}")
print(f"[wnorm] RATIO prepped_w2   / source_w2      = {p2_n / (w2_n + 1e-9):.4f}")
print("[wnorm] (~1.276 on w1/w2 => prep/quantize rescales weights; ~1.0 => scale is in the matmul/activation)")
ttnn.close_mesh_device(dev)
print("WNORM_DONE")
