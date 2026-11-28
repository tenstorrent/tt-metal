# Perf 1 ablation of nhwc_rm_transpose on Panoptic p44 ([1,128,256,32] bf16 ROW_MAJOR L1, dims (0,3,1,2)).
# Each arm stubs payloads of the op's kernels (copies in perf_experiments/ablation/kernels, guarded by
# ABL_* defines), keeping every CB handshake. 12 launches per arm; read DEVICE KERNEL DURATION per op id
# from the CSV. The copies are the RISC-V-transpose design Perf 1 replaced: this measured the breakdown
# that ranked the round, and runs only on the commit before the Tensix transpose (skipped here).
from pathlib import Path

import pytest
import torch
import ttnn

import importlib

op_mod = importlib.import_module("ttnn.operations.g6_permute.g6_permute")
from ttnn.operations.g6_permute import g6_permute_nhwc_rm_transpose as nhwc

KDIR = Path(__file__).resolve().parents[6] / "ttnn/ttnn/operations/g6_permute/perf_experiments/ablation/kernels"
ARMS = {
    "full": [],
    "nogather": ["ABL_NO_GATHER"],
    "nogather_nowrites": ["ABL_NO_GATHER", "ABL_NO_WRITES"],
    "nogather_nowrites_noreads": ["ABL_NO_GATHER", "ABL_NO_WRITES", "ABL_NO_READS"],
    "noreads": ["ABL_NO_READS"],
    "nowrites": ["ABL_NO_WRITES"],
}


@pytest.mark.skip(reason="ablation of the pre-Perf-1 nhwc kernels; run it on the commit before the Tensix transpose")
@pytest.mark.parametrize("arm", list(ARMS))
def test_arm(arm, device, monkeypatch):
    real = nhwc.create_program_descriptor

    def patched(plan, a, b):
        desc = real(plan, a, b)
        for k in desc.kernels:
            k.kernel_source = str(KDIR / Path(k.kernel_source).name)
            k.defines = [(d, "1") for d in ARMS[arm]]
        return desc

    monkeypatch.setattr(nhwc, "create_program_descriptor", patched)
    op_mod._PROGRAM_CACHE.clear()
    torch.manual_seed(0)
    x = torch.randn(1, 128, 256, 32, dtype=torch.bfloat16)
    xt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    first = ttnn._ttnn.get_device_operation_id()
    out = None
    for i in range(12):
        y = op_mod.g6_permute(xt, (0, 3, 1, 2))
        ttnn.synchronize_device(device)
        if i == 0:
            out = ttnn.to_torch(y)
        ttnn.deallocate(y)
    print(f"\nABL_ARM {arm} first_op={first} last_op={ttnn._ttnn.get_device_operation_id() - 1}")
    if arm == "full":
        assert torch.equal(out, x.permute(0, 3, 1, 2))
    op_mod._PROGRAM_CACHE.clear()
