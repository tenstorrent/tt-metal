# Refinement 2 perf probe: bank_affine_runs vs flat_page_range, device kernel time.
# Run explicitly under the profiler:
#   scripts/run_safe_pytest.sh --profile --no-py-zones \
#       tests/ttnn/unit_tests/operations/g6_clone/probes/bank_affinity_ab_probe.py
# Each case runs 1 + REPEATS g6_clone calls, in parametrize order; read DEVICE KERNEL DURATION [ns]
# of the generic ops in CSV order. Variants (host knobs patched per case, cache isolated):
#   flat        - flat_page_range forced (bank-affine gate disabled);
#   side        - bank_affine_runs, "dram_side" placement (the shipped default);
#   interleave  - bank_affine_runs, core k serves bank k mod num_banks;
#   rows        - bank_affine_runs, "dram_rows" placement (each bank's anchor row only);
#   side_bN     - "dram_side" with BLOCKS_PER_CORE_KNOB = N (depth-2 ring);
#   *_wep1      - the writer addresses its own NoC1 DRAM endpoints, not the reader's;
#   *_stag      - FLAT_BANK_STAGGER on (consecutive-page walks start on bank core_idx mod num_banks);
#   *_rdonly    - the production reader with ablate_kernels/noop_writer.cpp (no writes);
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
# Variant base name -> BANK_AFFINE_PLACEMENT.
PLACEMENTS = {"side": "dram_side", "interleave": "interleave", "rows": "dram_rows"}
ABLATE_DIR = Path(__file__).parent / "ablate_kernels"
BANK_ENDPOINT_CT_IDX = 4

P1 = ((16, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT, "p1")
VARIANTS = os.environ.get(
    "G6C_VARIANTS", "flat,side,interleave,side_b2,side_b4,flat_rdonly,flat_wronly,side_rdonly,side_wronly"
).split(",")
EXTRA = [
    ((8, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT, "p2"),
    ((1, 512, 5120), ttnn.bfloat16, ttnn.TILE_LAYOUT, "p3"),
    ((1, 1, 32, 62080), ttnn.bfloat16, ttnn.TILE_LAYOUT, "p4"),
    ((1, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT, "q13"),
    ((1, 2048, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT, "q_256p"),
    ((1, 1, 1, 62080), ttnn.bfloat16, ttnn.TILE_LAYOUT, "q8"),
    ((16, 3, 2560), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "q3_rm"),
    ((1, 1, 1, 32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, "int32_rm_1page"),
    ((1, 5120), ttnn.bfloat16, ttnn.TILE_LAYOUT, "f13"),
]
EXTRA_NAMES = os.environ.get("G6C_EXTRA_NAMES")
if EXTRA_NAMES:
    EXTRA = [case for case in EXTRA if case[3] in EXTRA_NAMES.split(",")]
EXTRA_VARIANTS = os.environ.get("G6C_EXTRA_VARIANTS", "flat,side").split(",")

CASES = [pytest.param(*P1[:3], v, id=f"{P1[3]}-{v}") for v in VARIANTS if v] + [
    pytest.param(shape, dtype, layout, v, id=f"{name}-{v}")
    for shape, dtype, layout, name in EXTRA
    for v in EXTRA_VARIANTS
    if v
]


@pytest.mark.parametrize("shape,dtype,layout,variant", CASES)
def test_bank_affinity_ab(device, shape, dtype, layout, variant, monkeypatch):
    monkeypatch.setattr(op, "_DESCRIPTOR_CACHE", {})
    base, _, ablation = variant.partition("_")
    if base == "flat":
        monkeypatch.setattr(pd, "BANK_AFFINE_MIN_CORE_PAGES", 1 << 30)
    elif base in PLACEMENTS:
        monkeypatch.setattr(pd, "BANK_AFFINE_PLACEMENT", PLACEMENTS[base])
    if ablation.startswith("b"):
        monkeypatch.setattr(pd, "BLOCKS_PER_CORE_KNOB", int(ablation[1:]))
    if ablation == "stag":
        monkeypatch.setattr(pd, "FLAT_BANK_STAGGER", True)
    if ablation == "wep1":
        # The writer addresses its own NoC1 endpoint table instead of the reader's endpoints.
        create_ep = op.create_program_descriptor

        def create_wep1(input_tensor, output_tensor, plan):
            descriptor = create_ep(input_tensor, output_tensor, plan)
            kernels = descriptor.kernels
            ct_args = list(kernels[pd.WRITER_KERNEL_IDX].compile_time_args)
            ct_args[BANK_ENDPOINT_CT_IDX] = 1
            kernels[pd.WRITER_KERNEL_IDX].compile_time_args = ct_args
            descriptor.kernels = kernels
            return descriptor

        monkeypatch.setattr(op, "create_program_descriptor", create_wep1)
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
    x = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = op.g6_clone(x)
    for _ in range(REPEATS):
        out = op.g6_clone(x)
    ttnn.synchronize_device(device)
    plan = op.g6_clone_plan(x)
    print(
        f"PLAN {variant} {shape} regime={plan['regime']} pages={plan['tensor_pages']} cores={plan['num_cores']} "
        f"core_pages={plan['core_pages_min']}..{plan['core_pages_max']} block_pages={plan['block_pages']} "
        f"depth={plan['cb_depth_blocks']} stagger={plan['bank_stagger']}"
    )
    if ablation not in ("rdonly", "wronly"):
        bits = torch.int16 if dtype == ttnn.bfloat16 else torch.int32
        assert torch.equal(ttnn.to_torch(out).view(bits), ttnn.to_torch(x).view(bits))
