"""Ablation harness: TILIZE_ABL="full,S,..." TILIZE_ABL_SHAPES="1x1x16384x64,..." (run with --profile)."""
import os
from pathlib import Path
import pytest, torch, ttnn
from ttnn.operations.tilize import tilize
import ttnn.operations.tilize.tilize_program_descriptor as pd

HERE = Path(__file__).resolve().parents[5] / "ttnn/ttnn/operations/tilize/perf_experiments/breakdown"
VARS = os.environ.get("TILIZE_ABL", "full").split(",")
SHAPES = [tuple(int(d) for d in s.split("x")) for s in os.environ.get("TILIZE_ABL_SHAPES", "1x1x16384x64").split(",")]


@pytest.mark.parametrize("variant", VARS)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_ablate(device, monkeypatch, shape, variant):
    monkeypatch.setattr(pd, "KERNEL_DIR", HERE / f"kernels_{variant}")
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = tilize(t)
    ttnn.synchronize_device(device)
    if variant == "full":
        assert torch.equal(ttnn.to_torch(out), x)
    print(f"ABL {variant} {shape} done")
