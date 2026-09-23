"""bank_paired_writes bake-off harness (perf-part-optimizer; artifacts in
ttnn/ttnn/operations/tilize/perf_experiments/bank_paired_writes/).

BPW_VARS="W_base,W_s2,..." BPW_SHAPES="1x1x16384x64,..."  (run with --profile for ns)
Each variant is a kernel dir kernels_<name> there; `pd.KERNEL_DIR` is monkeypatched to it.
Variants whose name starts with "C" (correct candidates) and "F_base" are checked bit-exact.
BPW_MEM: "dram" (default), "l1" (L1-interleaved in and out), "hs" (height-sharded in -> DRAM out).
"""
import os
from pathlib import Path
import pytest, torch, ttnn
from ttnn.operations.tilize import tilize
import ttnn.operations.tilize.tilize_program_descriptor as pd

HERE = Path(__file__).resolve().parents[5] / "ttnn/ttnn/operations/tilize/perf_experiments/bank_paired_writes"
VARS = os.environ.get("BPW_VARS", "F_base").split(",")
SHAPES = [tuple(int(d) for d in s.split("x")) for s in os.environ.get("BPW_SHAPES", "1x1x16384x64").split(",")]
MEM = os.environ.get("BPW_MEM", "dram")


def _input(device, x):
    if MEM == "dram":
        mc = ttnn.DRAM_MEMORY_CONFIG
    elif MEM == "l1":
        mc = ttnn.L1_MEMORY_CONFIG
    else:  # height-sharded, 64 cores
        H = x.shape[-2] * x.shape[0] * x.shape[1]
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
        mc = ttnn.create_sharded_memory_config(
            (H // 64, x.shape[-1]), grid, ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
        )
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)


@pytest.mark.parametrize("variant", VARS)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_bpw(device, monkeypatch, shape, variant):
    monkeypatch.setattr(pd, "KERNEL_DIR", HERE / f"kernels_{variant}")
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    t = _input(device, x)
    kw = {}
    if MEM == "l1":
        kw["memory_config"] = ttnn.L1_MEMORY_CONFIG
    elif MEM == "hs":
        kw["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
    out = tilize(t, **kw)
    ttnn.synchronize_device(device)
    if variant == "F_base" or variant.startswith("C"):
        assert torch.equal(ttnn.to_torch(out), x)
    print(f"BPW {variant} {shape} {MEM} done")
