# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 17: decode weight streaming (regime R4 `dram_sharded`) at the DRAM roofline. Every Qwen3.8 decode
case of the golden suite at its own config: bf16 A L1 width-sharded on `cores` cores, bfp4 B DRAM width-sharded
over the banks (64-column granule under swiglu), LoFi, fp32 DEST, output in the op-built L1 width shard.

Skipped unless GENIFY_MATMUL_PERF=1. GENIFY_MATMUL_KNOBS="NAME=value,..." overrides program-descriptor knobs
(the perf conftest); GENIFY_MATMUL_ONLY="id1,id2" selects cases by exact id; GENIFY_MATMUL_PLAN=1 prints the R4
topology and blocking; GENIFY_MATMUL_CHECK=1 also checks the output against torch (PCC, relative RMS).
"""
import math
import os
import pytest
import torch
import ttnn
from eval.golden_tests.genify_matmul.axes import observed as genify_matmul
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor

pytestmark = [
    pytest.mark.use_module_device,
    pytest.mark.skipif(os.environ.get("GENIFY_MATMUL_PERF") != "1", reason="perf guard: set GENIFY_MATMUL_PERF=1"),
]

# (id, m, k, n, A cores, epilogue, roofline_ns = bfp4 B bytes at 460.8 GB/s)
R17_CASES = [
    ("qkvg", 1, 5120, 3584, 10, "none", 22400),
    ("lm_last", 1, 5120, 12928, 10, "none", 80800),
    ("in_proj", 1, 5120, 4608, 10, "none", 28800),
    ("out_proj", 1, 1536, 5120, 8, "none", 9600),
    ("down", 1, 4352, 5120, 8, "none", 27200),
    ("gate_up", 1, 5120, 9216, 10, "swiglu", 57600),
    ("lm_head", 1, 5120, 16384, 10, "none", 102400),
    ("gate_up_b32", 32, 5120, 9216, 10, "swiglu", 57600),
    ("down_b32", 32, 4352, 5120, 8, "none", 27200),
    ("out_proj_b16", 16, 1536, 5120, 8, "none", 9600),
]


# GENIFY_MATMUL_CSWEEP="1,2,3,...": run every case once per forced cores_per_bank (invalid splits skip).
_CSWEEP = [int(c) for c in os.environ.get("GENIFY_MATMUL_CSWEEP", "").split(",") if c] or [None]


@pytest.mark.parametrize("cpb", _CSWEEP, ids=[f"c{c}" for c in _CSWEEP])
@pytest.mark.parametrize("case", R17_CASES, ids=[c[0] for c in R17_CASES])
def test_perf_r17(case, cpb, device, request, monkeypatch):
    cid, m, k, n, cores, epilogue, _ = case
    only = [t for t in os.environ.get("GENIFY_MATMUL_ONLY", "").split(",") if t]
    if only and cid not in only:
        pytest.skip("not selected by GENIFY_MATMUL_ONLY")
    if cpb is not None:
        from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

        sw = -(-n // ((64 if epilogue == "swiglu" else 32) * device.dram_grid_size().x)) * (
            2 if epilogue == "swiglu" else 1
        )
        try:
            monkeypatch.setattr(pd, "DS_CORES_PER_BANK", cpb)
            pd.ds_cores_per_bank(sw, device.dram_grid_size().x, 110, epilogue == "swiglu")
        except ValueError:
            pytest.skip("invalid split")
    torch.manual_seed(0)
    A = torch.randn(1, 1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    grid = ttnn.num_cores_to_corerangeset(cores, device.compute_with_storage_grid_size(), True)
    a_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [32, k // cores], ttnn.ShardOrientation.ROW_MAJOR),
    )
    banks = device.dram_grid_size().x
    granule = 64 if epilogue == "swiglu" else 32
    width = -(-n // (granule * banks)) * granule
    b_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
            [k, width],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=a_mc)
    tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, memory_config=b_mc)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
    )
    kwargs = {"activation": "swiglu"} if epilogue == "swiglu" else {}
    out = genify_matmul(ta, tb, compute_kernel_config=cfg, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, **kwargs)
    ttnn.synchronize_device(device)
    if os.environ.get("GENIFY_MATMUL_PLAN") == "1":
        from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

        geom = pd._geometry(ta, tb, out, True, epilogue, math_fidelity=ttnn.MathFidelity.LoFi)
        topo = pd.dram_sharded_topology(ta, tb, epilogue == "swiglu", geom)
        print(
            f"PLAN {cid} c{cpb} shards={topo.num_shards} kw={topo.shard_k_tiles} sw={topo.bank_n_tiles} "
            f"C={topo.cores_per_bank} cw={topo.core_n_tiles} workers={len(topo.workers)} rect={topo.rect}"
        )
    if os.environ.get("GENIFY_MATMUL_CHECK") == "1":
        ref = ttnn.to_torch(ta).float() @ ttnn.to_torch(tb).float()
        if epilogue == "swiglu":
            t = ref.reshape(*ref.shape[:-1], n // 64, 2, 32)
            ref = (torch.nn.functional.silu(t[..., 0, :]) * t[..., 1, :]).reshape(*ref.shape[:-1], n // 2)
        got = ttnn.to_torch(out).float()[..., : ref.shape[-1]]
        pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
        rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        print(f"CHECK {cid} pcc={pcc:.6f} rel_rms={rms:.5f}")
        assert pcc > 0.999, pcc
