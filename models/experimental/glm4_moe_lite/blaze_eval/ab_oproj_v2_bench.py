# SPDX-License-Identifier: Apache-2.0
"""A/B: ttnn DRAM-sharded matmul vs blaze DRAMStreamingMatmul, GLM o_proj, bs=1 decode.

v2. The v1 ttnn baseline was wrong (PCC -0.014) because I reconstructed the config instead
of mirroring the model. Every parameter was off:

  wrong in v1                              what the model actually does
  ------------------------------------     --------------------------------------------
  input/output_num_shards = 8 (DRAM banks) max(get_activation_sharding_core_counts_...)
                                           -> K=5120 (160 tiles): up to 80 cores
                                           -> N=2048  (64 tiles): up to 64 cores
  act sharded over 8 SCATTERED pinned      ttnn.num_cores_to_corerangeset(..., row_wise)
    DRAM-bank worker cores                   -- contiguous
  hand-built output memory config          ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
  in0_block_w hand-guessed (4, then 5)     derived from the real shard count
  fp32_dest_acc_en=True, approx=True       fp32_dest_acc_en=False, approx=False

The 8-vs-80 mistake was the fatal one: the program config described a layout the tensors
did not have. Source of truth is
glm4_moe_lite/tt/linear_helpers.py::dram_sharded_linear (+ _ds_act_mc, _DS_BATCH, _DS_CKC).

READ THE RESULT WITH THIS IN MIND: the two implementations parallelise differently. ttnn
spreads the activation over ~80 compute cores; blaze uses exactly 8, one per DRAM bank. So
a raw speedup conflates "faster" with "on 10x fewer cores" -- the core cost is reported too.
blaze also runs m=1 (its native 1x32 decode tile) because it is numerically WRONG at m=32
for this shape (PCC 0.0074); ttnn must pad to m=32.
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

import torch

import ttnn
import ab_harness
from tests.blaze.utils.torch_golden import comp_pcc
from blaze.ops.dram_streaming_matmul import DRAMStreamingMatmul
from blaze.utils import get_pinned_optimal_dram_bank_to_logical_worker_assignment
from test_dram_streaming_matmul import (
    _make_act_tensor,
    _make_output_tensor,
    _make_weights_tensor,
    _pad_to_dram_banks,
    _roundtrip_weights,
    _run_dram_streaming_matmul,
)

# Imported from the SAME tt-metal tree blaze builds against, so ttnn types match.
from models.demos.deepseek_v3.utils.config_helpers import (
    dram_sharded_weight_config,
    get_activation_sharding_core_counts_for_dram_matmul,
    get_dram_sharded_matmul_config,
)

K, N, TILE_W = 5120, 2048, 32
WEIGHT_DTYPE = ttnn.bfloat8_b
DS_BATCH = 32  # linear_helpers._DS_BATCH


def test_ab_oproj_v2(device):
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    banks = device.dram_grid_size().x

    torch.manual_seed(42)
    row = torch.randn(1, 1, 1, K).bfloat16().float()
    n_pad = _pad_to_dram_banks(N, TILE_W, TILE_W * banks)
    assert n_pad == N, f"o_proj N={N} should need no bank padding, got {n_pad}"
    w = torch.randn(1, 1, K, N).bfloat16().float()
    w_q = _roundtrip_weights(w, WEIGHT_DTYPE)
    golden_1 = DRAMStreamingMatmul.golden(row, w_q)
    golden_32 = DRAMStreamingMatmul.golden(row.repeat(1, 1, DS_BATCH, 1), w_q)

    # ---------- ttnn side: exactly what dram_sharded_linear does ----------
    in_cores = max(get_activation_sharding_core_counts_for_dram_matmul(K, max_cores))
    out_cores = max(get_activation_sharding_core_counts_for_dram_matmul(N, max_cores))

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
    # Weights: natural tile order, DRAM width-sharded over the banks. Built via host->device
    # to_memory_config, the same route layer_weights uses.
    t_w = ttnn.from_torch(w, dtype=WEIGHT_DTYPE, layout=ttnn.TILE_LAYOUT).to(
        device, dram_sharded_weight_config(K, N, device.dram_grid_size())
    )
    t_prog = get_dram_sharded_matmul_config(
        m=DS_BATCH, k=K, n=N, input_num_shards=in_cores, output_num_shards=out_cores
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

    # ---------- blaze side: 8 bank-pinned workers, shuffled weights, 1x32 tile ----------
    cores = get_pinned_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)
    b_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores])
    per_core_N_t = N // TILE_W // banks
    b_act = _make_act_tensor(device, row, m=1, k=K, tile_h=1, tile_w=TILE_W, compute_core_grid=b_grid, num_cores=banks)
    b_w = _make_weights_tensor(device, w, k=K, n_padded=N, tile_w=TILE_W, num_banks=banks, weight_dtype=WEIGHT_DTYPE)
    b_out = _make_output_tensor(
        device, m=1, n_padded=N, tile_h=1, tile_w=TILE_W, compute_core_grid=b_grid, per_core_N=per_core_N_t * TILE_W
    )

    def blaze_fn():
        return _run_dram_streaming_matmul(
            device, in0_t=b_act, in1_t=b_w, out_t=b_out, fp32_dest_acc_en=True, subblock_k=K // TILE_W // 2
        )

    t_us, t_res = ab_harness._measure(device, ttnn_fn, warmup=2, iters=5)
    b_us, b_res = ab_harness._measure(device, blaze_fn, warmup=2, iters=5)
    _, t_pcc = comp_pcc(golden_32, ttnn.to_torch(t_res))
    _, b_pcc = comp_pcc(golden_1, ttnn.to_torch(b_res))

    speedup = (t_us / b_us) if (t_us and b_us) else float("nan")
    text = "\n".join(
        [
            "| impl | cores | rows | us | PCC |",
            "|---|---:|---:|---:|---:|",
            f"| ttnn dram-sharded matmul | {in_cores} act / {out_cores} out | {DS_BATCH} | {t_us:.1f} | {t_pcc:.4f} |",
            f"| blaze DRAMStreamingMatmul | {banks} | 1 | {b_us:.1f} | {b_pcc:.4f} |",
            "",
            f"raw speedup {speedup:.2f}x -- but blaze uses {banks} cores vs ttnn's {in_cores}, "
            f"and 1 row vs {DS_BATCH}.",
        ]
    )
    print(text)
    out = "/tmp/claude-1000/-home-ttuser-sdawle-tt-metal/0247ce80-749c-4947-8139-2d31330dccb7/scratchpad/ab_oproj_v2_result.md"
    with open(out, "w") as fh:
        fh.write(text + "\n")
    assert t_pcc > 0.99, f"ttnn PCC {t_pcc:.4f} -- baseline still wrong, do not read timings"
    assert b_pcc > 0.99, f"blaze PCC {b_pcc:.4f}"
