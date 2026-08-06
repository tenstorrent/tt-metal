# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""A/B GLMOProjResidual against the five ttnn dispatches it replaces.

This is the best remaining candidate for an end-to-end win. q_kv_a fails because a 47 us ttnn op
cannot absorb a boundary pair; this cluster is the layer's LARGEST matmul (K=5120, N=2048) plus a
head flatten and a residual add, so one boundary pair amortises over far more work.

ttnn side is what the model actually runs at the shipping operating point (TP pinned off, so
o_proj is a plain replicated linear and the next op is a local add):

    permute -> reshape -> to_memory_config -> linear(5120x2048) -> add(residual) -> DRAM

Gated on PCC against that same ttnn path, not a torch golden: the claim being tested is
substitutability for the op in the model.
"""

import sys

sys.path.insert(0, "/home/ttuser/sdawle/skills/blaze-vs-ttnn-bench/scripts")
import ab_harness

ab_harness.set_profiler_env()  # must precede ttnn; _measure returns None on no samples

import torch  # noqa: E402
import ttnn  # noqa: E402
from blaze.models.blaze_tests_namespace import register_blaze_tests_namespace  # noqa: E402

register_blaze_tests_namespace()
from blaze.fused_program import FusedProgram  # noqa: E402
from blaze.ops.glm_oproj_residual import GLMOProjResidual  # noqa: E402
from blaze_tests.micro_ops.common.test_dram_streaming_matmul import _make_weights_tensor  # noqa: E402
from models.demos.deepseek_v3.utils.config_helpers import (  # noqa: E402
    dram_sharded_weight_config,
    get_activation_sharding_core_counts_for_dram_matmul,
    get_dram_sharded_matmul_config,
)

H, DV, HIDDEN = 20, 256, 2048
K = H * DV  # 5120

d = ttnn.open_device(device_id=0)
banks = d.dram_grid_size().x
torch.manual_seed(47)

v_t = torch.randn(1, H, 1, DV).bfloat16().float()
w_o = torch.randn(1, 1, K, HIDDEN).bfloat16().float()
res_t = torch.randn(1, 1, 1, HIDDEN).bfloat16().float()

mk = lambda t: ttnn.from_torch(
    t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
v = mk(v_t)
residual = mk(res_t)
out = mk(torch.zeros(1, 1, 1, HIDDEN))
b_w = _make_weights_tensor(d, w_o, k=K, n_padded=HIDDEN, tile_w=32, num_banks=banks, weight_dtype=ttnn.bfloat8_b)


def build():
    f = FusedProgram(
        kernel=None,
        device=d,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        name="oproj_ab",
    )
    GLMOProjResidual.emit(f, v, b_w, residual, out, prefix="oproj", num_heads=H, v_head_dim=DV, fp32_dest_acc_en=True)
    return f


# ---- ttnn: the five dispatches the model runs today
g = d.compute_with_storage_grid_size()
mx = g.x * g.y
in_c = max(get_activation_sharding_core_counts_for_dram_matmul(K, mx))
out_c = max(get_activation_sharding_core_counts_for_dram_matmul(HIDDEN, mx))
t_w = ttnn.from_torch(w_o, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT).to(
    d, dram_sharded_weight_config(K, HIDDEN, d.dram_grid_size())
)
t_pc = get_dram_sharded_matmul_config(m=32, k=K, n=HIDDEN, input_num_shards=in_c, output_num_shards=out_c)
t_ckc = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
)
act_mc = ttnn.create_sharded_memory_config_(
    shape=(32, K // in_c),
    core_grid=ttnn.num_cores_to_corerangeset(in_c, ttnn.CoreCoord(g.x, g.y), row_wise=True),
    strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    tile_layout=True,
    use_height_and_width_as_shard_shape=True,
)


def ttnn_fn():
    flat = ttnn.reshape(ttnn.permute(v, (0, 2, 1, 3)), (1, 1, 1, K))
    a = ttnn.to_memory_config(flat, act_mc)
    o = ttnn.linear(
        a, t_w, program_config=t_pc, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, compute_kernel_config=t_ckc
    )
    return ttnn.add(ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG), residual)


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


try:
    prog = build()
    print("RESULT built OK", flush=True)
    ref = ttnn.to_torch(ttnn_fn())[..., :1, :]
    prog.run()
    got = ttnn.to_torch(out)[..., :1, :]
    pcc = _pcc(ref, got)
    print(f"RESULT PCC vs ttnn 5-dispatch path = {pcc:.6f}", flush=True)
    gate = pcc >= 0.99
    if not gate:
        print("RESULT GATE FAILED -- timings meaningless", flush=True)
    t_us, _ = ab_harness._measure(d, ttnn_fn, warmup=2, iters=5)
    b_us, _ = ab_harness._measure(d, lambda: prog.run(), warmup=2, iters=5)
    print(f"RESULT ttnn  o_proj+residual (5 dispatches) = {t_us} us", flush=True)
    print(f"RESULT blaze GLMOProjResidual (1 dispatch)  = {b_us} us", flush=True)
    if t_us and b_us:
        print(
            f"RESULT speedup {t_us/b_us:.2f}x | x47 layers {(t_us-b_us)*47/1000:+.2f} ms/token"
            f" | gate={'PASS' if gate else 'FAIL'}",
            flush=True,
        )
except Exception as e:
    print("RESULT FAIL:", str(e).split(chr(10))[0][:200], flush=True)

ttnn.close_device(d)
