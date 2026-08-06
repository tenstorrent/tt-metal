# SPDX-License-Identifier: Apache-2.0
"""A/B: GLM's q_a + kv_a projections -- two ttnn matmuls vs one fused blaze dispatch.

This is the first FUSED-op comparison in this evaluation. Everything before it measured single
MicroOps, which is blaze with its main advantage switched off.

ttnn side runs what the model runs: two independent dram_sharded_linear calls, each resharding
the activation across up to 64 cores (mirrors glm4_moe_lite/tt/linear_helpers.py).
blaze side runs GLMQKVAProjection: ONE dispatch, one activation on the 8 DRAM-bank workers,
both matmuls consuming it from the same CB.

Same host values, one golden per output. The same two asymmetries as every other comparison here
apply and are reported: blaze computes the 1 real decode row where ttnn pads to 32, and uses 8
cores where ttnn uses up to 64.
"""

import importlib.util
import sys
import types
from pathlib import Path

sys.path.insert(0, "/home/ttuser/sdawle/skills/blaze-vs-ttnn-bench/scripts")
sys.path.insert(0, str(Path("/home/ttuser/sdawle/tt-blaze/tests/blaze/micro_ops/common")))


def _register_torch_golden() -> None:
    root = "/home/ttuser/sdawle/tt-blaze"
    for name, path in (
        ("tests", f"{root}/tests"),
        ("tests.blaze", f"{root}/tests/blaze"),
        ("tests.blaze.utils", f"{root}/tests/blaze/utils"),
    ):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [path]
            sys.modules[name] = m
    n = "tests.blaze.utils.torch_golden"
    if n not in sys.modules:
        spec = importlib.util.spec_from_file_location(n, f"{root}/tests/blaze/utils/torch_golden.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[n] = m
        spec.loader.exec_module(m)


_register_torch_golden()

import torch

import ab_harness
import ttnn
from blaze.fused_program import FusedProgram
from blaze.ops.dram_streaming_matmul import DRAMStreamingMatmul
from blaze.ops.glm_qkv_a_projection import GLMQKVAProjection
from blaze.utils import get_pinned_optimal_dram_bank_to_logical_worker_assignment
from models.demos.deepseek_v3.utils.config_helpers import (
    dram_sharded_weight_config,
    get_activation_sharding_core_counts_for_dram_matmul,
    get_dram_sharded_matmul_config,
)
from tests.blaze.utils.torch_golden import comp_pcc
from test_dram_streaming_matmul import (
    _make_act_tensor,
    _make_output_tensor,
    _make_weights_tensor,
    _pad_to_dram_banks,
    _roundtrip_weights,
)

K, N_Q, N_KV, TILE_W = 2048, 768, 576, 32  # GLM-4.7-Flash hidden, q_lora, kv_lora+qk_rope
DS_BATCH, WEIGHT_DTYPE, N_LAYERS = 32, ttnn.bfloat8_b, 47


def test_ab_glm_qkv_a(device):
    grid = device.compute_with_storage_grid_size()
    max_cores, banks = grid.x * grid.y, device.dram_grid_size().x
    nq, nkv = _pad_to_dram_banks(N_Q, TILE_W, TILE_W * banks), _pad_to_dram_banks(N_KV, TILE_W, TILE_W * banks)

    torch.manual_seed(42)
    row = torch.randn(1, 1, 1, K).bfloat16().float()
    wq = torch.randn(1, 1, K, nq).bfloat16().float()
    wkv = torch.randn(1, 1, K, nkv).bfloat16().float()
    gq1 = DRAMStreamingMatmul.golden(row, _roundtrip_weights(wq, WEIGHT_DTYPE))
    gkv1 = DRAMStreamingMatmul.golden(row, _roundtrip_weights(wkv, WEIGHT_DTYPE))
    rows32 = row.repeat(1, 1, DS_BATCH, 1)
    gq32 = DRAMStreamingMatmul.golden(rows32, _roundtrip_weights(wq, WEIGHT_DTYPE))
    gkv32 = DRAMStreamingMatmul.golden(rows32, _roundtrip_weights(wkv, WEIGHT_DTYPE))

    # ---- ttnn: two independent dram_sharded_linear calls, as the model does
    in_cores = max(get_activation_sharding_core_counts_for_dram_matmul(K, max_cores))
    t_act = ttnn.from_torch(
        rows32,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.create_sharded_memory_config_(
            shape=(DS_BATCH, K // in_cores),
            core_grid=ttnn.num_cores_to_corerangeset(in_cores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True),
            strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            tile_layout=True,
            use_height_and_width_as_shard_shape=True,
        ),
    )
    t_ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )

    def mk_ttnn(w, n):
        oc = max(get_activation_sharding_core_counts_for_dram_matmul(n, max_cores))
        tw = ttnn.from_torch(w, dtype=WEIGHT_DTYPE, layout=ttnn.TILE_LAYOUT).to(
            device, dram_sharded_weight_config(K, n, device.dram_grid_size())
        )
        pc = get_dram_sharded_matmul_config(m=DS_BATCH, k=K, n=n, input_num_shards=in_cores, output_num_shards=oc)
        return tw, pc

    tw_q, pc_q = mk_ttnn(wq, nq)
    tw_kv, pc_kv = mk_ttnn(wkv, nkv)

    def ttnn_fn():
        a = ttnn.linear(
            t_act,
            tw_q,
            program_config=pc_q,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=t_ckc,
        )
        b = ttnn.linear(
            t_act,
            tw_kv,
            program_config=pc_kv,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=t_ckc,
        )
        return a, b

    # THE BASELINE THE MODEL ACTUALLY RUNS. GLM4_MOE_LITE_FUSE_QKV_A=1 is in the winning
    # defaults, so the shipping path is ONE concatenated 2048 -> (768+576) matmul, sliced
    # afterwards -- not two independent ones. Comparing against the two-matmul form overstates
    # blaze by ~2x, so measure the fused ttnn form too.
    n_cat = _pad_to_dram_banks(N_Q + N_KV, TILE_W, TILE_W * banks)
    w_cat = torch.cat([wq, wkv], dim=-1)[:, :, :, :n_cat].contiguous()
    tw_cat, pc_cat = mk_ttnn(w_cat, n_cat)

    def ttnn_fused_fn():
        return ttnn.linear(
            t_act,
            tw_cat,
            program_config=pc_cat,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=t_ckc,
        )

    # ---- blaze: ONE fused dispatch
    cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)
    bg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores])
    b_act = _make_act_tensor(device, row, m=1, k=K, tile_h=1, tile_w=TILE_W, compute_core_grid=bg, num_cores=banks)
    b_wq = _make_weights_tensor(device, wq, k=K, n_padded=nq, tile_w=TILE_W, num_banks=banks, weight_dtype=WEIGHT_DTYPE)
    b_wkv = _make_weights_tensor(
        device, wkv, k=K, n_padded=nkv, tile_w=TILE_W, num_banks=banks, weight_dtype=WEIGHT_DTYPE
    )
    b_oq = _make_output_tensor(
        device, m=1, n_padded=nq, tile_h=1, tile_w=TILE_W, compute_core_grid=bg, per_core_N=nq // banks
    )
    b_okv = _make_output_tensor(
        device, m=1, n_padded=nkv, tile_h=1, tile_w=TILE_W, compute_core_grid=bg, per_core_N=nkv // banks
    )
    sub_k = max(1, K // TILE_W // 2)

    def blaze_fn():
        f = FusedProgram(
            kernel=None,
            device=device,
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            name="glm_qkv_a",
        )
        GLMQKVAProjection.emit(
            f,
            b_act,
            b_wq,
            b_wkv,
            q_a_out=b_oq,
            kv_a_out=b_okv,
            prefix="glm_qkv_a",
            fp32_dest_acc_en=True,
            q_a_subblock_k=sub_k,
            kv_a_subblock_k=sub_k,
        )
        f.run()
        return b_oq, b_okv

    t_us, t_res = ab_harness._measure(device, ttnn_fn, warmup=2, iters=5)
    tf_us, _ = ab_harness._measure(device, ttnn_fused_fn, warmup=2, iters=5)
    b_us, b_res = ab_harness._measure(device, blaze_fn, warmup=2, iters=5)
    _, tq = comp_pcc(gq32, ttnn.to_torch(t_res[0]))
    _, tkv = comp_pcc(gkv32, ttnn.to_torch(t_res[1]))
    _, bq = comp_pcc(gq1, ttnn.to_torch(b_res[0]))
    _, bkv = comp_pcc(gkv1, ttnn.to_torch(b_res[1]))

    sp = t_us / b_us if (t_us and b_us) else float("nan")
    text = "\n".join(
        [
            "| impl | dispatches | cores | us | q_a PCC | kv_a PCC |",
            "|---|---:|---:|---:|---:|---:|",
            f"| ttnn: 2x dram_sharded_linear | 2 | {in_cores} | {t_us:.1f} | {tq:.4f} | {tkv:.4f} |",
            f"| ttnn: 1x FUSED q_kv_a (what the model runs) | 1 | {in_cores} | {tf_us:.1f} | - | - |",
            f"| blaze: GLMQKVAProjection | 1 | {banks} | {b_us:.1f} | {bq:.4f} | {bkv:.4f} |",
            "",
            f"vs the REAL baseline (fused ttnn): {tf_us/b_us:.2f}x",
            f"vs two separate matmuls: {sp:.2f}x | saved {(t_us-b_us)*N_LAYERS/1000.0:+.2f} ms/token over {N_LAYERS} layers (UPPER bound)",
        ]
    )
    print(text)
    with open(
        "/tmp/claude-1000/-home-ttuser-sdawle-tt-metal/0247ce80-749c-4947-8139-2d31330dccb7/scratchpad/ab_qkv_a.md", "w"
    ) as fh:
        fh.write(text + "\n")
    assert tq > 0.99 and tkv > 0.99, f"ttnn PCC {tq:.4f}/{tkv:.4f} -- rig wrong"
    assert bq > 0.99 and bkv > 0.99, f"blaze PCC {bq:.4f}/{bkv:.4f}"
