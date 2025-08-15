# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import compare_equal
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 1, 2])),),
)
@pytest.mark.parametrize(
    "testing_dtype",
    ["bfloat16", "float32"],
)
def test_fmod(input_shapes, testing_dtype, device):
    torch_dtype = getattr(torch, testing_dtype)
    ttnn_dtype = getattr(ttnn, testing_dtype)
    torch_input_a = torch.tensor([1.0, 0.0, -1.0], dtype=torch_dtype)
    torch_input_b = torch.tensor([0.0, 0.0, 0.0], dtype=torch_dtype)

    golden_function = ttnn.get_golden_function(ttnn.fmod)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.fmod(tt_in_a, tt_in_b)
    output_tensor = ttnn.to_torch(tt_result)
    print()
    print("dtype : ", testing_dtype)
    print("Expected : ", golden)
    print("TTNN.    : ", output_tensor)

    print()
    print("Torch analysis : ")
    if testing_dtype == "float32":
        A = torch_input_a
        B = torch_input_b
        print("A:", torch_input_a)
        print("B:", torch_input_b)

        # 1. Divide a, b
        div_result = A / B
        print("A / B --> Divide a,b : ", div_result)

        # 2. Round off using trunc (towards zero)
        trunc_result = torch.trunc(div_result)
        print("Trunc(A / B) --> Round off using trunc : ", trunc_result)

        # 3. Multiply trunc_result with B
        mul_result = trunc_result * B
        print("Trunc(A / B) * B --> Mutiply division result with B : ", mul_result)

        # 4. Subtract from A
        sub_result = A - mul_result
        print("A - (Trunc(A / B) * B) --> Subtract with A : ", sub_result)

        # 5. Replace with 0.0 if A == B
        final_result = torch.where(A == B, torch.tensor(0.0), sub_result)
        print("Final Result (0 if A == B):", final_result)
    else:
        a = torch_input_a
        b = torch_input_b
        print("A:", a)
        print("B:", b)

        # Step 1: Divide a, b
        division_result = a / b
        print("Division (a / b):", division_result)

        # Step 2: Round off using trunc
        trunc_result = torch.trunc(division_result)
        print("Truncated result:", trunc_result)

        # Step 3: Typecast Input A to FP32
        a_fp32 = a.to(torch.float32)
        print("A (FP32):", a_fp32)

        # Step 4: Typecast Input B to FP32
        b_fp32 = b.to(torch.float32)
        print("B (FP32):", b_fp32)

        # Step 5: Typecast trunc result to FP32
        trunc_fp32 = trunc_result.to(torch.float32)
        print("Truncated (FP32):", trunc_fp32)

        # Step 6: Multiply division result with B
        mul_result = division_result * b
        print("Mutiply division result with B :", mul_result)

        # Step 7: Subtract with A
        sub_result = mul_result - a
        print("Subtraction with A :", sub_result)

        # Step 8: Replace with 0.0 if A == B
        zeroed_result = torch.where(a == b, torch.tensor(0.0, dtype=sub_result.dtype), sub_result)
        print("Replace with 0.0 if A==B :", zeroed_result)

        # Step 9: Typecast result to FP32 → result
        final_result = zeroed_result.to(torch.float32)
        print("Typecast result to FP32 → result:", final_result)

    assert torch.equal(golden, output_tensor)
