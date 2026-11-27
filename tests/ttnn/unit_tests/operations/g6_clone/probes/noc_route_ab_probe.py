# Refinement 3 perf probe: NoC route knobs of bank_affine_runs, device kernel time.
# Run explicitly under the profiler:
#   scripts/run_safe_pytest.sh --profile --no-py-zones \
#       tests/ttnn/unit_tests/operations/g6_clone/probes/noc_route_ab_probe.py
# Each case runs 1 + REPEATS g6_clone calls, in parametrize order; read DEVICE KERNEL DURATION [ns]
# of the generic ops in CSV order (probes/perf_csv_cases.py groups them). Variants (host knobs patched
# per case, cache isolated):
#   base        - shipped knobs (one VC, reads NoC0 / writes NoC1);
#   vcr / vcw / vcrw - BANK_AFFINE_PER_CORE_VC = ("read",) / ("write",) / ("read", "write");
#   alt / half  - BANK_AFFINE_NOC_SPLIT = "alternate" / "half";
#   u<k>        - every bank-affine read and write on VC k (DEFAULT_UNICAST_VC = k);
#   r/w/b<a><b>.. - per-core VC on reads / writes / both, BANK_AFFINE_VC_CYCLE = (a, b, ...);
#   *_rdonly    - production reader with ablate_kernels/noop_writer.cpp (no writes);
#   *_wronly    - ablate_kernels/noop_reader.cpp (no reads) with the production writer.
# Ablated variants skip the bit check: their output is wrong by design.

import importlib
import os
from pathlib import Path

import pytest
import torch
import ttnn

op = importlib.import_module("ttnn.operations.g6_clone.g6_clone")
pd = importlib.import_module("ttnn.operations.g6_clone.g6_clone_program_descriptor")

REPEATS = 2
ABLATE_DIR = Path(__file__).parent / "ablate_kernels"
VC_DIRECTIONS = {"vcr": ("read",), "vcw": ("write",), "vcrw": ("read", "write")}
SPLITS = {"alt": "alternate", "half": "half"}

SHAPES = {
    "p1": ((16, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT),
    "p2": ((8, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT),
    "p3": ((1, 512, 5120), ttnn.bfloat16, ttnn.TILE_LAYOUT),
    "p4": ((1, 1, 32, 62080), ttnn.bfloat16, ttnn.TILE_LAYOUT),
    "q8": ((1, 1, 1, 62080), ttnn.bfloat16, ttnn.TILE_LAYOUT),
    "q3_rm": ((16, 3, 2560), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    "int32_rm_1page": ((1, 1, 1, 32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
}
# "name:variant,variant;name:variant" -> cases in that order.
CASE_SPEC = os.environ.get("G6C_CASES", "p1:base,vcr,vcw,vcrw,alt,half")
CASES = [
    pytest.param(*SHAPES[name], variant, id=f"{name}-{variant}")
    for group in CASE_SPEC.split(";")
    for name, _, variants in [group.partition(":")]
    for variant in variants.split(",")
    if variant
]


@pytest.mark.parametrize("shape,dtype,layout,variant", CASES)
def test_noc_route_ab(device, shape, dtype, layout, variant, monkeypatch):
    monkeypatch.setattr(op, "_DESCRIPTOR_CACHE", {})
    base, _, ablation = variant.partition("_")
    if base in VC_DIRECTIONS:
        monkeypatch.setattr(pd, "BANK_AFFINE_PER_CORE_VC", VC_DIRECTIONS[base])
    elif base in SPLITS:
        monkeypatch.setattr(pd, "BANK_AFFINE_NOC_SPLIT", SPLITS[base])
    elif base[0] == "u" and base[1:].isdigit():
        monkeypatch.setattr(pd, "DEFAULT_UNICAST_VC", int(base[1:]))
    elif base[0] in "rwb" and base[1:].isdigit():
        directions = {"r": ("read",), "w": ("write",), "b": ("read", "write")}[base[0]]
        monkeypatch.setattr(pd, "BANK_AFFINE_PER_CORE_VC", directions)
        monkeypatch.setattr(pd, "BANK_AFFINE_VC_CYCLE", tuple(int(c) for c in base[1:]))
    elif base != "base":
        raise ValueError(f"unknown variant {variant}")
    if ablation in ("rdonly", "wronly"):
        create = pd.create_program_descriptor

        def create_ablated(input_tensor, output_tensor, plan):
            descriptor = create(input_tensor, output_tensor, plan)
            idx, stub = (
                (pd.WRITER_KERNEL_IDX, "noop_writer.cpp")
                if ablation == "rdonly"
                else (pd.READER_KERNEL_IDX, "noop_reader.cpp")
            )
            kernels = descriptor.kernels
            kernels[idx].kernel_source = str(ABLATE_DIR / stub)
            descriptor.kernels = kernels
            return descriptor

        monkeypatch.setattr(op, "create_program_descriptor", create_ablated)

    torch.manual_seed(0)
    if dtype in (ttnn.int32, ttnn.uint32):
        torch_input = torch.randint(-(2**31), 2**31 - 1, shape, dtype=torch.int32)
    else:
        torch_input = torch.randn(shape, dtype=torch.float32 if dtype == ttnn.float32 else torch.bfloat16)
    x = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = op.g6_clone(x)
    for _ in range(REPEATS):
        out = op.g6_clone(x)
    ttnn.synchronize_device(device)
    plan = op.g6_clone_plan(x)
    print(
        f"PLAN {variant} {shape} regime={plan['regime']} cores={plan['num_cores']} "
        f"vc={plan['bank_vc_directions']} split={plan['bank_noc_split']} "
        f"route_args={sorted(set(plan['route_args']))}"
    )
    if ablation not in ("rdonly", "wronly"):
        bits = torch.int16 if dtype == ttnn.bfloat16 else torch.int32
        assert torch.equal(ttnn.to_torch(out).view(bits), ttnn.to_torch(x).view(bits))
