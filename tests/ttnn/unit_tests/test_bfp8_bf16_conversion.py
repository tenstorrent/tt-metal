import pytest
import torch

import ttnn

TILE_HEIGHT = 32
TILE_WIDTH = 32

cpu_layout = ttnn.Layout.ROW_MAJOR
npu_layout = ttnn.Layout.TILE


def test_typecast_bf16_to_bfp8_b(device):
    torch.manual_seed(0)
    shape = [32, 32]

    # bf16 --> bfp8_b by cpu.
    torch_bf16 = torch.randn(shape, dtype=torch.bfloat16)
    bfp8_b_by_cpu = ttnn.Tensor(torch_bf16, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b_by_cpu.to(cpu_layout).to_torch()

    # bf16 --> bfp8_b by npu
    tt_bf16 = ttnn.Tensor(torch_bf16, ttnn.bfloat16).to(npu_layout).to(device)
    bfp8_b_by_npu = ttnn.typecast(tt_bf16, ttnn.bfloat8_b)
    npu_version = bfp8_b_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    # print(cpu_version[0, 0:16])
    # print(npu_version[0, 0:16])
    assert passed


def print_mismatches(cpu, npu, num_max_print):
    different_indices = (cpu != npu).nonzero(as_tuple=True)
    count = 0
    for idx in zip(*different_indices):
        count = count + 1
        print(f"idx={idx} cpu={cpu[idx]} npu={npu[idx]}")
        if count > num_max_print:
            break


@pytest.mark.parametrize("seed", [0, 2, 4, 6, 8])
@pytest.mark.parametrize("scale", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("bias", [0, 1, 2, 4, 8, 16, 32, 64, 128])
def test_typecast_bf16_to_bfp8_b_various_input(seed, scale, bias, device):
    torch.manual_seed(seed)
    shape = [1024, 1024]

    bias = bias
    low = bias - scale
    high = bias + scale
    torch_bf16 = random_tensor = torch.empty(shape).uniform_(low, high).to(torch.bfloat16)

    random_signs = torch.randint(0, 2, shape) * 2 - 1
    torch_bf16 = torch_bf16 * random_signs

    # bf16 --> bfp8_b by cpu.
    bfp8_b_by_cpu = ttnn.Tensor(torch_bf16, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b_by_cpu.to(cpu_layout).to_torch()

    # bf16 --> bfp8_b by npu
    tt_bf16 = ttnn.Tensor(torch_bf16, ttnn.bfloat16).to(npu_layout).to(device)
    bfp8_b_by_npu = ttnn.typecast(tt_bf16, ttnn.bfloat8_b)
    npu_version = bfp8_b_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    if not passed:
        print_mismatches(cpu_version, npu_version, 16)
    assert passed


@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("scale", [4])
@pytest.mark.parametrize("bias", [2])
# NaN becomes -Inf when converted to bfloat8_b format, skip testing
@pytest.mark.parametrize("insert_inf, insert_nan", [[True, False]])  # , [False, True], [True, True]])
def test_typecast_bf16_to_bfp8_b_with_inf_nan(seed, scale, bias, insert_inf, insert_nan, device):
    torch.manual_seed(seed)
    shape = [1024, 1024]

    bias = bias
    low = bias - scale
    high = bias + scale

    torch_bf16 = random_tensor = torch.empty(shape).uniform_(low, high).to(torch.bfloat16)
    if insert_inf:
        num_inf = torch_bf16.numel() // 8  # 16 elements are pcked into
        inf_indices = torch.randint(0, torch_bf16.numel(), (num_inf,))
        torch_bf16.view(-1)[inf_indices] = float("inf")
    if insert_nan:
        num_nan = torch_bf16.numel() // 8
        nan_indices = torch.randint(0, torch_bf16.numel(), (num_nan,))
        torch_bf16.view(-1)[nan_indices] = float("nan")
    random_signs = torch.randint(0, 2, shape) * 2 - 1
    torch_bf16 = torch_bf16 * random_signs

    # bf16 --> bfp8_b by cpu.
    bfp8_b_by_cpu = ttnn.Tensor(torch_bf16, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b_by_cpu.to(cpu_layout).to_torch()

    # bf16 --> bfp8_b by npu
    tt_bf16 = ttnn.Tensor(torch_bf16, ttnn.bfloat16).to(npu_layout).to(device)
    bfp8_b_by_npu = ttnn.typecast(tt_bf16, ttnn.bfloat8_b)
    npu_version = bfp8_b_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    if not passed:
        print_mismatches(cpu_version, npu_version, 16)
    assert passed


def test_typecast_bfp8_b_to_bf16(device):
    torch.manual_seed(0)
    shape = [1024, 1024]

    # bfp8_b --> bf16 by cpu.
    torch_bf16 = torch.randn(shape, dtype=torch.bfloat16)
    bfp8_b = ttnn.Tensor(torch_bf16, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b.to(cpu_layout).to_torch()

    # bfp8_b --> bf16 by npu.
    bf16_by_npu = ttnn.typecast(bfp8_b.to(device), ttnn.bfloat16)
    npu_version = bf16_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    # print(cpu_version[0, 0:16])
    # print(npu_version[0, 0:16])
    assert passed


def test_typecast_fp32_to_bfp8_b(device):
    torch.manual_seed(0)
    shape = [32, 32]

    # fp32 --> bfp8_b by cpu.
    torch_fp32 = torch.randn(shape, dtype=torch.float32)
    bfp8_b_by_cpu = ttnn.Tensor(torch_fp32, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b_by_cpu.to(cpu_layout).to_torch()

    # fp32 --> bfp8_b by npu
    tt_fp32 = ttnn.Tensor(torch_fp32, ttnn.float32).to(npu_layout).to(device)
    bfp8_b_by_npu = ttnn.typecast(tt_fp32, ttnn.bfloat8_b)
    npu_version = bfp8_b_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    # print(cpu_version[0, 0:16])
    # print(npu_version[0, 0:16])
    assert passed


def test_typecast_bfp8_b_to_fp32(device):
    torch.manual_seed(0)
    shape = [1024, 1024]

    # bfp8_b --> fp32 by cpu.
    torch_fp32 = torch.randn(shape, dtype=torch.float32)
    bfp8_b = ttnn.Tensor(torch_fp32, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b.to(cpu_layout).to_torch()

    # bfp8_b --> fp32 by npu.
    fp32_by_npu = ttnn.typecast(bfp8_b.to(device), ttnn.float32)
    npu_version = fp32_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    # print(cpu_version[0, 0:16])
    # print(npu_version[0, 0:16])
    assert passed
