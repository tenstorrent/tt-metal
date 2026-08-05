# SPDX-License-Identifier: Apache-2.0
"""A/B every GLM-4.7-Flash decode matmul: ttnn DRAM-sharded vs blaze DRAMStreamingMatmul.

Generalises ab_oproj_v2_bench.py from one shape to the whole decode step. All six shapes are
numerically correct in blaze at m=1 (glm47_all_shapes_check.py), so the open question is how
much each is worth.

ttnn side mirrors glm4_moe_lite/tt/linear_helpers.py::dram_sharded_linear exactly -- the
model's own path, not a reconstruction:
  input/output_num_shards = max(get_activation_sharding_core_counts_for_dram_matmul(dim, cores))
  activation width-sharded over num_cores_to_corerangeset(..., row_wise=True)
  output L1_WIDTH_SHARDED_MEMORY_CONFIG, program cfg from get_dram_sharded_matmul_config
  m = _DS_BATCH = 32, LoFi, no approx, fp32_dest_acc_en=False, packer_l1_acc=True

blaze side runs m=1 (its native 1x32 decode tile; m=32 is numerically broken -- F1) on the 8
DRAM-bank workers, with the column-major tile shuffle its weights require. So the comparison
carries the same two asymmetries as the o_proj bench, reported not hidden: blaze computes the
1 real row where ttnn must pad to 32, and uses 8 cores where ttnn spreads over up to 80.

Per-call deltas are multiplied by 47 layers to bound the step-level prize. That is an UPPER
bound: it assumes every call is on the critical path and translates perfectly, and this model
has already measured a 23%-op reduction landing as 0.0 ms under trace.
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
            mod = types.ModuleType(name)
            mod.__path__ = [path]
            sys.modules[name] = mod
    name = "tests.blaze.utils.torch_golden"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, f"{root}/tests/blaze/utils/torch_golden.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)


_register_torch_golden()

import pytest
import torch

import ab_harness
import ttnn
from blaze.ops.dram_streaming_matmul import DRAMStreamingMatmul
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
    _run_dram_streaming_matmul,
)

TILE_W = 32
DS_BATCH = 32  # linear_helpers._DS_BATCH
WEIGHT_DTYPE = ttnn.bfloat8_b
N_LAYERS = 47

# (label, K, N, calls_per_layer)
SHAPES = [
    ("q_a_proj", 2048, 768, 1),
    ("kv_a_proj", 2048, 576, 1),
    ("q_b_proj", 768, 5120, 1),
    ("o_proj", 5120, 2048, 1),
    ("mlp_gate_up", 2048, 1536, 2),  # gate and up are separate matmuls
    ("mlp_down", 1536, 2048, 1),
]

_RESULTS: list[tuple] = []


@pytest.mark.parametrize("label, K, N", [pytest.param(s[0], s[1], s[2], id=s[0]) for s in SHAPES])
def test_ab_shape(device, label, K, N):
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    banks = device.dram_grid_size().x
    n_pad = _pad_to_dram_banks(N, TILE_W, TILE_W * banks)

    torch.manual_seed(42)
    row = torch.randn(1, 1, 1, K).bfloat16().float()
    w = torch.randn(1, 1, K, n_pad).bfloat16().float()
    w_q = _roundtrip_weights(w, WEIGHT_DTYPE)
    golden_1 = DRAMStreamingMatmul.golden(row, w_q)
    golden_32 = DRAMStreamingMatmul.golden(row.repeat(1, 1, DS_BATCH, 1), w_q)

    # ---- ttnn: exactly dram_sharded_linear's configuration
    in_cores = max(get_activation_sharding_core_counts_for_dram_matmul(K, max_cores))
    out_cores = max(get_activation_sharding_core_counts_for_dram_matmul(n_pad, max_cores))
    t_act = ttnn.from_torch(
        row.repeat(1, 1, DS_BATCH, 1),
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
    t_w = ttnn.from_torch(w, dtype=WEIGHT_DTYPE, layout=ttnn.TILE_LAYOUT).to(
        device, dram_sharded_weight_config(K, n_pad, device.dram_grid_size())
    )
    t_prog = get_dram_sharded_matmul_config(
        m=DS_BATCH, k=K, n=n_pad, input_num_shards=in_cores, output_num_shards=out_cores
    )
    t_ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    def ttnn_fn():
        return ttnn.linear(
            t_act,
            t_w,
            program_config=t_prog,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=t_ckc,
        )

    # ---- blaze: 8 bank workers, 1x32 tile, shuffled weights
    cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)
    b_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores])
    per_core_N_t = n_pad // TILE_W // banks
    b_act = _make_act_tensor(device, row, m=1, k=K, tile_h=1, tile_w=TILE_W, compute_core_grid=b_grid, num_cores=banks)
    b_w = _make_weights_tensor(
        device, w, k=K, n_padded=n_pad, tile_w=TILE_W, num_banks=banks, weight_dtype=WEIGHT_DTYPE
    )
    b_out = _make_output_tensor(
        device,
        m=1,
        n_padded=n_pad,
        tile_h=1,
        tile_w=TILE_W,
        compute_core_grid=b_grid,
        per_core_N=per_core_N_t * TILE_W,
    )
    subblock_k = max(1, K // TILE_W // 2)

    def blaze_fn():
        return _run_dram_streaming_matmul(
            device, in0_t=b_act, in1_t=b_w, out_t=b_out, fp32_dest_acc_en=True, subblock_k=subblock_k
        )

    t_us, t_res = ab_harness._measure(device, ttnn_fn, warmup=2, iters=5)
    b_us, b_res = ab_harness._measure(device, blaze_fn, warmup=2, iters=5)
    _, t_pcc = comp_pcc(golden_32, ttnn.to_torch(t_res))
    _, b_pcc = comp_pcc(golden_1, ttnn.to_torch(b_res))

    calls = dict((s[0], s[3]) for s in SHAPES)[label]
    speedup = t_us / b_us if (t_us and b_us) else float("nan")
    saved_ms = (t_us - b_us) * calls * N_LAYERS / 1000.0
    line = (
        f"| {label} {K}x{N} | {in_cores}/{out_cores} | {t_us:.1f} | {b_us:.1f} | {speedup:.2f}x "
        f"| {t_pcc:.4f} | {b_pcc:.4f} | {calls} | {saved_ms:+.2f} |"
    )
    print("ROW " + line)
    with open(
        "/tmp/claude-1000/-home-ttuser-sdawle-tt-metal/0247ce80-749c-4947-8139-2d31330dccb7/scratchpad/ab_all_shapes_rows.md",
        "a",
    ) as fh:
        fh.write(line + "\n")

    assert t_pcc > 0.99, f"ttnn PCC {t_pcc:.4f} -- rig wrong for {label}, do not read timings"
    assert b_pcc > 0.99, f"blaze PCC {b_pcc:.4f} for {label}"
