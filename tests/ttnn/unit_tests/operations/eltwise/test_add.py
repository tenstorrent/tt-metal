# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import skip_for_slow_dispatch


@pytest.mark.parametrize(
    "shapes",
    [
        [[63, 1, 4], [1, 9, 4]],
        [[13600, 1, 4], [1, 9, 4]],
        [[1, 16, 6, 64, 64], [1, 16, 1, 64, 64]],
        [[63, 1, 4], [1, 1, 1]],
    ],
)
def test_non_4D_channel_bcast(device, shapes):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shapes[0], dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shapes[1], dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("size", [64, 1, 0])
def test_add_1D_tensor_and_scalar(device, scalar, size):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((size,), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor + scalar

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = input_tensor + scalar
    output_tensor = ttnn.to_torch(output_tensor, torch_rank=1)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
    assert output_tensor.shape == (size,)


@pytest.mark.parametrize("hw", [(1024, 1024)])
def test_add_2D_tensors(device, hw):
    import numpy as np
    import os

    save = False
    check = True

    # Configuration: How many operand_a values to test (must be a multiple of 16)
    total_operand_a_values_to_test = 65536
    start_a_value = 0

    # tensor shape: 1024x1024 = 1,048,576 elements
    # We can test 16 different operand_a values in a 1024x1024 elwadd, each tested against all 65,536 operand_b values
    total_elements_per_tensor = hw[0] * hw[1]
    values_per_operand_b_sweep = 65536  # (full 16-bit range)
    operand_a_values_per_batch = total_elements_per_tensor // values_per_operand_b_sweep  # 16 values
    num_batches = total_operand_a_values_to_test // operand_a_values_per_batch

    # File splitting configuration
    max_batches_per_file = 128  # Maximum 128 batches (2048 operand_a values) per file
    num_files = (num_batches + max_batches_per_file - 1) // max_batches_per_file  # ceil division

    assert (
        total_operand_a_values_to_test % operand_a_values_per_batch == 0
    ), f"total_operand_a_values_to_test ({total_operand_a_values_to_test}) must be a multiple of {operand_a_values_per_batch}"

    print(
        f"Testing {total_operand_a_values_to_test} operand_a values in {num_batches} batches of {operand_a_values_per_batch}"
    )
    print(f"Each batch uses SINGLE 1024x1024 ttnn.add() operation")
    print(f"Each operand_a tested against all {values_per_operand_b_sweep} operand_b values")
    print(f"Splitting output into {num_files} file(s) (max {max_batches_per_file} batches per file)")

    # File handles (will be opened/closed as needed per file)
    output_file_handle = None
    reference_file_handle = None
    current_file_idx = -1
    current_file_batch_count = 0

    total_mismatches = 0
    mismatch_records = []  # List to collect all mismatches: (input_a, input_b, result, expected)

    # Exp_diff_9 streaming output configuration
    exp_diff_9_file_handle = None
    exp_diff_9_file_idx = -1
    exp_diff_9_current_file_records = 0
    exp_diff_9_total_records = 0
    max_bytes_per_exp_diff_file = 128 * 1024 * 1024  # 128MB
    bytes_per_exp_diff_record = 4 * 2  # 4 uint16 values = 8 bytes
    header_bytes = 8
    max_records_per_exp_diff_file = (max_bytes_per_exp_diff_file - header_bytes) // bytes_per_exp_diff_record

    def get_exponent_bf16(val):
        """Get exponent of bfloat16 value (biased)."""
        return (val >> 7) & 0xFF

    def calc_exp_diff(a_val, b_val):
        """Calculate absolute exponent difference between two bfloat16 values."""
        exp_a = get_exponent_bf16(a_val)
        exp_b = get_exponent_bf16(b_val)
        return abs(exp_a - exp_b)

    # Pre-allocate input tensors
    torch_input_a_uint16 = torch.zeros(hw, dtype=torch.uint16)
    torch_input_b_uint16 = torch.zeros(hw, dtype=torch.uint16)

    batch_outputs_buffer = torch.zeros((operand_a_values_per_batch * values_per_operand_b_sweep,), dtype=torch.int16)

    # Build input_b tensor once (it's the same for all batches)
    # 16 complete sets of all 65536 values (0x0000 to 0xFFFF)
    print(f"Initializing input_b tensor (same for all batches)...")
    for local_idx in range(operand_a_values_per_batch):
        for b_value in range(values_per_operand_b_sweep):
            linear_idx = local_idx * values_per_operand_b_sweep + b_value
            row = linear_idx // hw[1]
            col = linear_idx % hw[1]
            torch_input_b_uint16[row, col] = b_value

    # Convert to bfloat16 once
    torch_input_b_bf16 = torch_input_b_uint16.view(torch.bfloat16)

    # Initialize input_a tensor for the first batch
    print(f"Initializing input_a tensor for first batch...")
    first_operand_a_start = start_a_value
    for local_idx in range(operand_a_values_per_batch):
        operand_a_value = first_operand_a_start + local_idx
        for i in range(values_per_operand_b_sweep):
            linear_idx = local_idx * values_per_operand_b_sweep + i
            row = linear_idx // hw[1]
            col = linear_idx % hw[1]
            torch_input_a_uint16[row, col] = operand_a_value

    # Process each batch
    for batch_idx in range(num_batches):
        # Check if we need to open a new file
        if current_file_batch_count == 0 or current_file_batch_count >= max_batches_per_file:
            # Close previous file if open
            if output_file_handle is not None:
                output_file_handle.close()
                batches_in_file = min(max_batches_per_file, num_batches - (current_file_idx * max_batches_per_file))
                operand_a_values_in_file = batches_in_file * operand_a_values_per_batch
                print(
                    f"\nClosed file {current_file_idx} ({batches_in_file} batches, {operand_a_values_in_file} operand_a values)"
                )

            if reference_file_handle is not None:
                reference_file_handle.close()

            # Open new file
            current_file_idx += 1
            current_file_batch_count = 0

            if num_files > 1:
                output_file = f"ttnn_add_output_part_{current_file_idx:03d}.bin"
            else:
                output_file = "ttnn_add_output_all.bin"

            if save:
                output_file_handle = open(output_file, "wb")
                print(f"\n{'='*60}")
                print(f"Opened {output_file} for writing (file {current_file_idx + 1}/{num_files})")
                print(f"{'='*60}")

            if check:
                if not os.path.exists(output_file):
                    raise FileNotFoundError(f"Reference file {output_file} not found. Run with save=True first.")
                reference_file_handle = open(output_file, "rb")
                print(f"\n{'='*60}")
                print(f"Opened {output_file} for checking (file {current_file_idx + 1}/{num_files})")
                print(f"{'='*60}")

        operand_a_start = start_a_value + (batch_idx * operand_a_values_per_batch)
        operand_a_end = operand_a_start + operand_a_values_per_batch - 1

        if batch_idx % 10 == 0:
            print(
                f"\nBatch {batch_idx + 1}/{num_batches}: operand_a range 0x{operand_a_start:04x} - 0x{operand_a_end:04x}"
            )

        # Update input_a tensor: for batch > 0, increment all values by 16
        # Convert to int32 for addition since uint16 doesn't support arithmetic ops
        if batch_idx > 0:
            torch_input_a_uint16 = (torch_input_a_uint16.to(torch.int32) + 16).to(torch.uint16)

        # Convert input_a to bfloat16
        torch_input_a_bf16 = torch_input_a_uint16.view(torch.bfloat16)

        # Convert to ttnn tensors (input_b_bf16 is already initialized outside the loop)
        input_tensor_a = ttnn.from_torch(torch_input_a_bf16, layout=ttnn.TILE_LAYOUT, device=device)
        input_tensor_b = ttnn.from_torch(torch_input_b_bf16, layout=ttnn.TILE_LAYOUT, device=device)

        # Perform addition
        # print(f"Executing ttnn.add()...")
        output = ttnn.add(input_tensor_a, input_tensor_b)

        # Convert output back to torch
        output_torch = ttnn.to_torch(output)
        output_int16 = output_torch.view(torch.int16)

        # Extract all results for this batch using efficient memory copy
        # print(f"Extracting results...")
        batch_outputs_buffer.copy_(output_int16.flatten())

        if save:
            # Append this batch's results to file
            output_file_handle.write(batch_outputs_buffer.numpy().tobytes())
            # print(f"  Saved batch results ({batch_outputs_buffer.numel()} values, {batch_outputs_buffer.numel() * 2} bytes)")
            # if batch_idx == 0:
            #    print(f"  First few values: {['0x{:04x}'.format(v.item() & 0xFFFF) for v in batch_outputs_buffer[:10]]}")

        if check:
            # Read this batch's reference data
            reference_bytes = reference_file_handle.read(batch_outputs_buffer.numel() * 2)
            reference_data = np.frombuffer(reference_bytes, dtype=np.int16)
            reference_batch = torch.from_numpy(reference_data)

            # Compare
            matches = torch.eq(batch_outputs_buffer, reference_batch)
            num_matches = torch.sum(matches).item()
            total_elements = batch_outputs_buffer.numel()
            match_percentage = (num_matches / total_elements) * 100

            print(f"  Comparison: {num_matches}/{total_elements} match ({match_percentage:.2f}%)")

            # Stream exp_diff == 9 cases to disk as we find them
            batch_exp_diff_9_count = 0
            batch_exp_diff_9_records = []  # Temporary buffer for this batch

            for idx in range(total_elements):
                local_idx = idx // values_per_operand_b_sweep
                operand_b_idx = idx % values_per_operand_b_sweep
                operand_a_value = operand_a_start + local_idx

                if calc_exp_diff(operand_a_value, operand_b_idx) == 9:
                    batch_exp_diff_9_count += 1
                    batch_exp_diff_9_records.append(
                        [
                            operand_a_value,
                            operand_b_idx,
                            batch_outputs_buffer[idx].item() & 0xFFFF,
                            reference_batch[idx].item() & 0xFFFF,
                        ]
                    )

            # Write batch exp_diff_9 records to file
            if batch_exp_diff_9_count > 0:
                print(f"    Found {batch_exp_diff_9_count} cases with exp_diff == 9")

                for record in batch_exp_diff_9_records:
                    # Check if we need to open a new exp_diff_9 file
                    if (
                        exp_diff_9_file_handle is None
                        or exp_diff_9_current_file_records >= max_records_per_exp_diff_file
                    ):
                        # Close previous file and update its header
                        if exp_diff_9_file_handle is not None:
                            # Update header with actual record count
                            current_pos = exp_diff_9_file_handle.tell()
                            exp_diff_9_file_handle.seek(0)
                            exp_diff_9_file_handle.write(
                                exp_diff_9_current_file_records.to_bytes(8, byteorder="little")
                            )
                            exp_diff_9_file_handle.close()
                            file_size = header_bytes + exp_diff_9_current_file_records * bytes_per_exp_diff_record
                            print(
                                f"    Closed exp_diff_9 file {exp_diff_9_file_idx} ({exp_diff_9_current_file_records} records, {file_size / (1024*1024):.2f} MB)"
                            )

                        # Open new file
                        exp_diff_9_file_idx += 1
                        exp_diff_9_current_file_records = 0

                        exp_diff_9_filename = f"ttnn_add_exp_diff_9_part_{exp_diff_9_file_idx:03d}.bin"
                        exp_diff_9_file_handle = open(exp_diff_9_filename, "wb")
                        # Write placeholder header (will be updated when file is closed)
                        exp_diff_9_file_handle.write((0).to_bytes(8, byteorder="little"))
                        print(f"    Opened {exp_diff_9_filename} for writing exp_diff_9 records")

                    # Write record
                    record_tensor = torch.tensor([record], dtype=torch.int64)
                    record_uint16 = record_tensor.to(torch.uint16)
                    exp_diff_9_file_handle.write(record_uint16.numpy().tobytes())
                    exp_diff_9_current_file_records += 1
                    exp_diff_9_total_records += 1

            if num_matches != total_elements:
                batch_mismatches = total_elements - num_matches
                total_mismatches += batch_mismatches

                # Find all mismatches and collect them
                mismatch_indices = torch.where(~matches)[0]

                for mismatch_idx in mismatch_indices:
                    idx = mismatch_idx.item()
                    mismatch_local_idx = idx // values_per_operand_b_sweep
                    mismatch_operand_b_idx = idx % values_per_operand_b_sweep
                    mismatch_operand_a_value = operand_a_start + mismatch_local_idx

                    # Store: (input_a, input_b, result, expected)
                    mismatch_records.append(
                        [
                            mismatch_operand_a_value,
                            mismatch_operand_b_idx,
                            batch_outputs_buffer[idx].item() & 0xFFFF,
                            reference_batch[idx].item() & 0xFFFF,
                        ]
                    )

                # Report first mismatch for this batch
                first_mismatch_idx = mismatch_indices[0].item()
                mismatch_local_idx = first_mismatch_idx // values_per_operand_b_sweep
                mismatch_operand_b_idx = first_mismatch_idx % values_per_operand_b_sweep
                mismatch_operand_a_value = operand_a_start + mismatch_local_idx

                print(f"    Found {batch_mismatches} mismatches in this batch")
                print(f"    First mismatch:")
                print(f"      operand_a=0x{mismatch_operand_a_value:04x}, operand_b=0x{mismatch_operand_b_idx:04x}")
                print(f"      Current: 0x{batch_outputs_buffer[first_mismatch_idx].item() & 0xFFFF:04x}")
                print(f"      Reference: 0x{reference_batch[first_mismatch_idx].item() & 0xFFFF:04x}")
            else:
                print(f"    Batch matches reference! ✓")

        current_file_batch_count += 1

    # Close final files
    if output_file_handle is not None:
        output_file_handle.close()
        batches_in_final_file = num_batches - (current_file_idx * max_batches_per_file)
        operand_a_values_in_final_file = batches_in_final_file * operand_a_values_per_batch
        print(
            f"\nClosed final file {current_file_idx} ({batches_in_final_file} batches, {operand_a_values_in_final_file} operand_a values)"
        )

    if reference_file_handle is not None:
        reference_file_handle.close()

    # Close final exp_diff_9 file
    if exp_diff_9_file_handle is not None:
        # Update header with actual record count
        exp_diff_9_file_handle.seek(0)
        exp_diff_9_file_handle.write(exp_diff_9_current_file_records.to_bytes(8, byteorder="little"))
        exp_diff_9_file_handle.close()
        file_size = header_bytes + exp_diff_9_current_file_records * bytes_per_exp_diff_record
        print(
            f"\nClosed final exp_diff_9 file {exp_diff_9_file_idx} ({exp_diff_9_current_file_records} records, {file_size / (1024*1024):.2f} MB)"
        )

    # Summary
    total_size = total_operand_a_values_to_test * values_per_operand_b_sweep
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total batches processed: {num_batches}")
    print(f"Total operand_a values tested: {total_operand_a_values_to_test}")
    print(f"Total size: {total_size} values ({total_size * 2} bytes)")
    print(f"Number of output files: {num_files}")

    if num_files > 1:
        print(f"Output files:")
        for file_idx in range(num_files):
            batches_in_this_file = min(max_batches_per_file, num_batches - (file_idx * max_batches_per_file))
            operand_a_in_this_file = batches_in_this_file * operand_a_values_per_batch
            file_name = f"ttnn_add_output_part_{file_idx:03d}.bin"
            print(f"  {file_name}: {batches_in_this_file} batches, {operand_a_in_this_file} operand_a values")

    if check:
        if total_mismatches == 0:
            print(f"\nAll {num_batches} batches matched reference! ✓")
        else:
            print(f"\nTotal mismatches across all batches: {total_mismatches}")

            # Save all mismatches to a file

            if len(mismatch_records) > 0:
                mismatch_tensor = torch.tensor(mismatch_records, dtype=torch.int64)
                print(f"Mismatch tensor shape: {mismatch_tensor.shape}")

                mismatch_file = "ttnn_add_mismatches.bin"
                with open(mismatch_file, "wb") as f:
                    # Write header: number of mismatches
                    f.write(total_mismatches.to_bytes(8, byteorder="little"))
                    # Write all mismatch records as uint16 values
                    mismatch_uint16 = mismatch_tensor.to(torch.uint16)
                    f.write(mismatch_uint16.numpy().tobytes())

                print(f"Saved {total_mismatches} mismatches to {mismatch_file}")
                print(f"Format: 8-byte header (num_mismatches), then {total_mismatches} records of 4 uint16 values")
                print(f"Each record: [input_a, input_b, result, expected]")
                print(f"File size: {8 + total_mismatches * 4 * 2} bytes")

                # Show first few mismatches
                print(f"\nFirst 5 mismatches:")
                for i in range(min(5, len(mismatch_records))):
                    rec = mismatch_records[i]
                    print(
                        f"  {i+1}. a=0x{rec[0]:04x}, b=0x{rec[1]:04x}, result=0x{rec[2]:04x}, expected=0x{rec[3]:04x}"
                    )

        # Summary of exp_diff == 9 records (written incrementally to disk)
        if exp_diff_9_total_records > 0:
            print(f"\nExp_diff == 9 Summary:")
            print(f"Total cases with exp_diff == 9: {exp_diff_9_total_records}")
            print(f"Number of exp_diff_9 files: {exp_diff_9_file_idx + 1}")
            print(f"Format: 8-byte header (num_cases in file), then records of 4 uint16 values")
            print(f"Each record: [input_a, input_b, result, expected]")
            if exp_diff_9_file_idx == 0:
                print(f"File written: ttnn_add_exp_diff_9_part_000.bin")
            else:
                print(
                    f"Files written: ttnn_add_exp_diff_9_part_000.bin through ttnn_add_exp_diff_9_part_{exp_diff_9_file_idx:03d}.bin"
                )

            assert False, f"Found {total_mismatches} mismatches"


@pytest.mark.parametrize("hw", [(32, 64), (1, 1), (0, 0)])
def test_add_2D_tensors_with_program_cache(device, hw):
    torch_input_tensor_a = torch.rand(hw, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(hw, dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.add(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("hw", [(32, 64), (1, 1), (0, 0)])
@pytest.mark.parametrize("scalar", [0.42])
def test_add_scalar(device, hw, scalar):
    torch_input_tensor_a = torch.rand(hw, dtype=torch.bfloat16)
    torch_output_tensor = scalar + torch_input_tensor_a

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    output = input_tensor_a + scalar
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("hw", [(32, 64), (1, 1), (0, 0)])
@pytest.mark.parametrize("scalar", [0.42])
def test_reverse_add_scalar(device, hw, scalar):
    torch_input_tensor_a = torch.rand(hw, dtype=torch.bfloat16)
    torch_output_tensor = scalar + torch_input_tensor_a

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    output = scalar + input_tensor_a
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("hw", [(32, 64), (1, 1), (0, 0)])
def test_add_4D_tensors(device, hw):
    torch_input_tensor_a = torch.rand((5, 64, hw[0], hw[1]), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((5, 64, hw[0], hw[1]), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.add(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_add_with_broadcast(device, h, w):
    # See #4005, we basically are using ttnn.repeat to get this to pass.
    torch_input_tensor_a = torch.rand((2, 16, 1, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((2, 16, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("h", [500])
@pytest.mark.parametrize("w", [512])
def test_expand_and_broadcast(device, h, w):
    torch_input_tensor_a = torch.rand((1, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_add_with_broadcast_on_batch(device, h, w):
    torch_input_tensor_a = torch.rand((1, 16, 1, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((2, 16, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("shape", [(8, 16, 384, 384)])
@pytest.mark.parametrize("scalar", [0.125])
def test_add_attention_scores_to_scalar(device, shape, scalar):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor + scalar

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    output_tensor = ttnn.add(input_tensor, scalar, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
    assert output_tensor.shape == shape


@pytest.mark.parametrize("shape_a", [(8, 16, 128, 128)])
@pytest.mark.parametrize("shape_b", [(1, 16, 128, 128)])
def test_add_with_batch_broadcast(device, shape_a, shape_b):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape_b, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
    assert output_tensor.shape == shape_a


@pytest.mark.parametrize("shape_a", [(4096, 4096)])
@pytest.mark.parametrize("shape_b", [(1, 4096)])
def test_add_dram_and_l1_tensor(device, shape_a, shape_b):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape_b, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
    assert output_tensor.shape == shape_a


@pytest.mark.parametrize("shape", [(1, 1, 32, 32)])
@pytest.mark.parametrize("activations", [[], [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]])
def test_add_and_apply_activations(device, shape, activations):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    for activation in activations:
        if activation == "relu":
            torch_output_tensor = torch.relu(torch_output_tensor)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, activations=activations)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.99988)
    assert output_tensor.shape == shape


@pytest.mark.parametrize("shape", [(1, 1, 32, 32)])
@pytest.mark.parametrize("activations", [[], [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]])
def test_in_place_add_and_apply_activations(device, shape, activations):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    for activation in activations:
        if activation == "relu":
            torch_output_tensor = torch.relu(torch_output_tensor)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.add_(input_tensor_a, input_tensor_b, activations=activations)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.99988)
    assert output_tensor.shape == shape


@pytest.mark.skip(reason="#11002/#4005: Bcast does not appear to be doing what we expect.  Leaving test for reference.")
@pytest.mark.parametrize("shape_a", [(1, 1, 8192, 320)])
@pytest.mark.parametrize("shape_b", [(2, 1, 1, 320)])
def test_add_with_different_batch(device, shape_a, shape_b):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape_b, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_with_storage_grid_size = device.compute_with_storage_grid_size()
    device_grid_size = ttnn.CoreGrid(y=compute_with_storage_grid_size.y, x=compute_with_storage_grid_size.x)

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(1024, 64),
        core_grid=device_grid_size,  # ttnn.CoreGrid(y=8, x=5),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_tensor_a = ttnn.to_memory_config(input_tensor_a, block_sharded_mem_config)

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    # Intended to swap code below with: output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # print("here!!!!!!!!!!!!!!!!!!!!!!!")
    # output_tensor = ttnn.bcast(
    #     input_tensor_a,
    #     input_tensor_b,
    #     ttnn.BcastOpMath.ADD,
    #     ttnn.BcastOpDim.H,
    #     memory_config=input_tensor_a.memory_config(),
    # )
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    output_tensor = ttnn.to_torch(output_tensor)

    # We do not support broadcasting as one would expect,
    # our bcast will return a tensor without the batch 2
    # we also get incorrect pcc as well
    torch_output_tensor = torch_output_tensor[:1]

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
    assert output_tensor.shape == shape_a


@pytest.mark.parametrize("input_a_sharded", [True, False])
@pytest.mark.parametrize("input_b_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_add_with_height_sharding(device, input_a_sharded, input_b_sharded, out_sharded, shard_orientation):
    torch.manual_seed(0)
    shape = (1, 1, 1024, 1024)
    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (1024 // 8, 1024)
    else:
        shard_shape = (1024, 1024 // 8)

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=2, x=4),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input_a_sharded:
        input_tensor_a = ttnn.to_memory_config(input_tensor_a, height_sharded_mem_config)

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input_b_sharded:
        input_tensor_b = ttnn.to_memory_config(input_tensor_b, height_sharded_mem_config)

    if out_sharded:
        out_mem_config = height_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
    assert output_tensor.shape == shape


@pytest.mark.parametrize("input_a_sharded", [True, False])
@pytest.mark.parametrize("input_b_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_add_with_width_sharding(device, input_a_sharded, input_b_sharded, out_sharded, shard_orientation):
    torch.manual_seed(0)
    shape = (1, 1, 1024, 1024)
    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (1024, 1024 // 8)
    else:
        shard_shape = (1024 // 8, 1024)

    width_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=2, x=4),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input_a_sharded:
        input_tensor_a = ttnn.to_memory_config(input_tensor_a, width_sharded_mem_config)

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input_b_sharded:
        input_tensor_b = ttnn.to_memory_config(input_tensor_b, width_sharded_mem_config)

    if out_sharded:
        out_mem_config = width_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
    assert output_tensor.shape == shape


@pytest.mark.parametrize("input_a_sharded", [True, False])
@pytest.mark.parametrize("input_b_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_add_with_block_sharding(device, input_a_sharded, input_b_sharded, out_sharded, shard_orientation):
    torch.manual_seed(0)
    shape = (1, 1, 1024, 1024)
    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)

    shard_shape = (1024 // 2, 1024 // 4)

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=2, x=4),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input_a_sharded:
        input_tensor_a = ttnn.to_memory_config(input_tensor_a, block_sharded_mem_config)

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input_b_sharded:
        input_tensor_b = ttnn.to_memory_config(input_tensor_b, block_sharded_mem_config)

    if out_sharded:
        out_mem_config = block_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
    assert output_tensor.shape == shape


@pytest.mark.parametrize(
    "data",
    [
        ([], [], []),
        ([1], [2], [3]),
        ([1], [], []),
        ([], [1], []),
        ([1, 2], [3], [4, 5]),
        ([1], [2, 3], [3, 4]),
        ([1, 2], [3, 4], [4, 6]),
    ],
)
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_01_volume_tensors(device, data, memory_config):
    (a, b, c_golden) = data
    a = torch.BFloat16Tensor(a)
    b = torch.BFloat16Tensor(b)
    assert torch.add(a, b).tolist() == c_golden

    ttnn_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_c = ttnn.add(ttnn_a, ttnn_b)
    c = ttnn.to_torch(ttnn_c).reshape((-1))

    assert c.tolist() == c_golden


@skip_for_slow_dispatch()
@pytest.mark.parametrize("input_a_sharded", [True, False])
@pytest.mark.parametrize("input_b_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_add_with_sub_devices(device, input_a_sharded, input_b_sharded, out_sharded, shard_orientation):
    torch.manual_seed(0)
    shape = (1, 1, 1024, 1024)
    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (1024 // 8, 1024)
    else:
        shard_shape = (1024, 1024 // 8)

    core_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(2, 2), ttnn.CoreCoord(3, 3)),
            ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(1, 1)),
            ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 2)),
        ]
    )

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=core_range_set,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input_a_sharded:
        input_tensor_a = ttnn.to_memory_config(input_tensor_a, height_sharded_mem_config)

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input_b_sharded:
        input_tensor_b = ttnn.to_memory_config(input_tensor_b, height_sharded_mem_config)

    if out_sharded:
        out_mem_config = height_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    sub_device = ttnn.SubDevice(
        [
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(4, 4)),
                    ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(5, 0)),
                ]
            )
        ]
    )
    sub_device_manager_id = device.create_sub_device_manager([sub_device], 0)
    device.load_sub_device_manager(sub_device_manager_id)
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
    assert output_tensor.shape == shape


@pytest.mark.parametrize("hw", [(1024, 1024)])
def test_add_2D_tensors_exp_diff_9(device, hw):
    """
    Test addition of bfloat16 tensors with specific constraints:
    - operand_a: arbitrary normalized numbers (configurable start value)
    - operand_b: normalized numbers with same sign as operand_a and exponent difference of exactly 9
    """
    import numpy as np
    import os
    import struct

    save = True
    check = False

    # Configuration: How many operand_a values to test
    total_operand_a_values_to_test = 128 * 16
    # Start value for operand_a (must be normalized bfloat16)
    start_a_value = 0x00E0  # Example: 1.0 (0x3f80) in bfloat16
    # Number of operand_b values to test per operand_a
    num_operand_b_per_a = 128

    def is_normalized_bf16(val):
        """Check if a bfloat16 value (as uint16) is normalized."""
        exp = (val >> 7) & 0xFF
        return exp != 0 and exp != 0xFF

    def get_sign_bf16(val):
        """Get sign bit of bfloat16 value."""
        return (val >> 15) & 1

    def get_exponent_bf16(val):
        """Get exponent of bfloat16 value (biased)."""
        return (val >> 7) & 0xFF

    def set_sign_bf16(val, sign):
        """Set sign bit of bfloat16 value."""
        return (val & 0x7FFF) | (sign << 15)

    def create_bf16_with_exp_diff(base_val, exp_diff):
        """
        Create a bfloat16 value with the same sign as base_val and exponent difference of exp_diff.
        Returns None if not possible to create a valid normalized number.
        """
        base_sign = get_sign_bf16(base_val)
        base_exp = get_exponent_bf16(base_val)

        # Calculate target exponent
        if base_exp <= exp_diff:
            target_exp = base_exp + exp_diff
        else:
            target_exp = base_exp - exp_diff

        # Check if target exponent is valid for normalized number (1-254)
        if target_exp < 1 or target_exp > 254:
            return None

        # Create new value with target exponent, base sign, and mantissa of 0
        new_val = (base_sign << 15) | (target_exp << 7) | 0

        return new_val

    def generate_operand_b_values(operand_a_val, count, exp_diff=9):
        """
        Generate 'count' normalized bfloat16 operand_b values with:
        - Same sign as operand_a_val
        - Exponent difference of exp_diff from operand_a_val
        - Different mantissa values to get variety
        """
        operand_b_list = []
        base_sign = get_sign_bf16(operand_a_val)
        base_exp = get_exponent_bf16(operand_a_val)

        target_exp = base_exp + exp_diff

        if target_exp < 1 or target_exp > 254:
            # Cannot create valid normalized number with this exp_diff
            return None

        # Generate values with different mantissa bits (0-127)
        # Mantissa in bfloat16 is 7 bits
        mantissa_step = max(1, 128 // count)

        for i in range(count):
            mantissa = (i * mantissa_step) % 128
            val = (base_sign << 15) | (target_exp << 7) | mantissa
            operand_b_list.append(val)

        return operand_b_list

    # Validate start_a_value
    if not is_normalized_bf16(start_a_value):
        raise ValueError(f"start_a_value 0x{start_a_value:04x} is not a normalized bfloat16 number")

    # Calculate tensor dimensions
    total_elements_per_tensor = hw[0] * hw[1]
    total_elements_needed = total_operand_a_values_to_test * num_operand_b_per_a

    # Determine how many batches we need
    elements_per_batch = total_elements_per_tensor
    operand_a_per_batch = elements_per_batch // num_operand_b_per_a

    if operand_a_per_batch == 0:
        raise ValueError(
            f"Tensor size {total_elements_per_tensor} is too small for {num_operand_b_per_a} operand_b values"
        )

    num_batches = (total_operand_a_values_to_test + operand_a_per_batch - 1) // operand_a_per_batch

    print(f"Testing {total_operand_a_values_to_test} operand_a values in {num_batches} batches")
    print(f"Each operand_a tested against {num_operand_b_per_a} operand_b values")
    print(f"Operand_a values per batch: {operand_a_per_batch}")
    print(f"Start operand_a value: 0x{start_a_value:04x}")
    print(f"Exponent difference: 9")

    # File configuration
    max_batches_per_file = 128
    num_files = (num_batches + max_batches_per_file - 1) // max_batches_per_file

    output_file_handle = None
    reference_file_handle = None
    current_file_idx = -1
    current_file_batch_count = 0
    total_mismatches = 0
    mismatch_records = []

    # Pre-allocate tensors
    torch_input_a_uint16 = torch.zeros(hw, dtype=torch.uint16)
    torch_input_b_uint16 = torch.zeros(hw, dtype=torch.uint16)

    # Track current operand_a value
    current_a_value = start_a_value

    # Process each batch
    for batch_idx in range(num_batches):
        # Check if we need to open a new file
        if current_file_batch_count == 0 or current_file_batch_count >= max_batches_per_file:
            if output_file_handle is not None:
                output_file_handle.close()
                batches_in_file = min(max_batches_per_file, num_batches - (current_file_idx * max_batches_per_file))
                print(f"\nClosed file {current_file_idx} ({batches_in_file} batches)")

            if reference_file_handle is not None:
                reference_file_handle.close()

            current_file_idx += 1
            current_file_batch_count = 0

            if num_files > 1:
                output_file = f"ttnn_add_exp9_output_part_{current_file_idx:03d}.bin"
            else:
                output_file = "ttnn_add_exp9_output_all.bin"

            if save:
                output_file_handle = open(output_file, "wb")
                print(f"\n{'='*60}")
                print(f"Opened {output_file} for writing (file {current_file_idx + 1}/{num_files})")
                print(f"{'='*60}")

            if check:
                if not os.path.exists(output_file):
                    raise FileNotFoundError(f"Reference file {output_file} not found. Run with save=True first.")
                reference_file_handle = open(output_file, "rb")
                print(f"\n{'='*60}")
                print(f"Opened {output_file} for checking (file {current_file_idx + 1}/{num_files})")
                print(f"{'='*60}")

        # Determine how many operand_a values in this batch
        remaining_a_values = total_operand_a_values_to_test - (batch_idx * operand_a_per_batch)
        batch_operand_a_count = min(operand_a_per_batch, remaining_a_values)

        if batch_idx % 10 == 0:
            print(
                f"\nBatch {batch_idx + 1}/{num_batches}: Testing {batch_operand_a_count} operand_a values starting from 0x{current_a_value:04x}"
            )

        # Build input tensors for this batch
        linear_idx = 0
        batch_a_values = []

        for a_idx in range(batch_operand_a_count):
            # Get current operand_a value and ensure it's normalized
            a_val = current_a_value

            # Find next normalized value if current is not normalized
            max_search = 1000
            search_count = 0
            while not is_normalized_bf16(a_val) and search_count < max_search:
                a_val = (a_val + 1) & 0xFFFF
                search_count += 1

            if search_count >= max_search:
                raise ValueError(f"Could not find normalized value starting from 0x{current_a_value:04x}")

            batch_a_values.append(a_val)

            # Generate operand_b values for this operand_a
            b_values = generate_operand_b_values(a_val, num_operand_b_per_a, exp_diff=9)

            if b_values is None:
                # Skip this operand_a if we can't generate valid operand_b values
                print(f"Warning: Skipping operand_a 0x{a_val:04x} - cannot create operand_b with exp_diff=9")
                current_a_value = (a_val + 1) & 0xFFFF
                continue

            # Fill tensor with operand_a and operand_b pairs
            for b_idx, b_val in enumerate(b_values):
                if linear_idx >= total_elements_per_tensor:
                    break

                row = linear_idx // hw[1]
                col = linear_idx % hw[1]
                torch_input_a_uint16[row, col] = a_val
                torch_input_b_uint16[row, col] = b_val
                linear_idx += 1

            # Move to next operand_a value
            current_a_value = (a_val + 1) & 0xFFFF

        # Convert to bfloat16
        torch_input_a_bf16 = torch_input_a_uint16.view(torch.bfloat16)
        torch_input_b_bf16 = torch_input_b_uint16.view(torch.bfloat16)

        # Convert to ttnn tensors
        input_tensor_a = ttnn.from_torch(torch_input_a_bf16, layout=ttnn.TILE_LAYOUT, device=device)
        input_tensor_b = ttnn.from_torch(torch_input_b_bf16, layout=ttnn.TILE_LAYOUT, device=device)

        # Perform addition
        output = ttnn.add(input_tensor_a, input_tensor_b)

        # Convert output back to torch
        output_torch = ttnn.to_torch(output)
        output_int16 = output_torch.view(torch.int16)

        # Extract results for this batch
        batch_outputs_buffer = output_int16.flatten()

        if save:
            output_file_handle.write(batch_outputs_buffer.numpy().tobytes())

        if check:
            reference_bytes = reference_file_handle.read(batch_outputs_buffer.numel() * 2)
            reference_data = np.frombuffer(reference_bytes, dtype=np.int16)
            reference_batch = torch.from_numpy(reference_data)

            matches = torch.eq(batch_outputs_buffer, reference_batch)
            num_matches = torch.sum(matches).item()
            total_elements = batch_outputs_buffer.numel()
            match_percentage = (num_matches / total_elements) * 100

            print(f"  Comparison: {num_matches}/{total_elements} match ({match_percentage:.2f}%)")

            if num_matches != total_elements:
                batch_mismatches = total_elements - num_matches
                total_mismatches += batch_mismatches

                mismatch_indices = torch.where(~matches)[0]

                for mismatch_idx in mismatch_indices:
                    idx = mismatch_idx.item()
                    row = idx // hw[1]
                    col = idx % hw[1]

                    mismatch_records.append(
                        [
                            torch_input_a_uint16[row, col].item(),
                            torch_input_b_uint16[row, col].item(),
                            batch_outputs_buffer[idx].item() & 0xFFFF,
                            reference_batch[idx].item() & 0xFFFF,
                        ]
                    )

                first_mismatch_idx = mismatch_indices[0].item()
                row = first_mismatch_idx // hw[1]
                col = first_mismatch_idx % hw[1]

                print(f"    Found {batch_mismatches} mismatches in this batch")
                print(f"    First mismatch:")
                print(
                    f"      operand_a=0x{torch_input_a_uint16[row, col].item():04x}, operand_b=0x{torch_input_b_uint16[row, col].item():04x}"
                )
                print(f"      Current: 0x{batch_outputs_buffer[first_mismatch_idx].item() & 0xFFFF:04x}")
                print(f"      Reference: 0x{reference_batch[first_mismatch_idx].item() & 0xFFFF:04x}")
            else:
                print(f"    Batch matches reference!")

        current_file_batch_count += 1

    # Close final files
    if output_file_handle is not None:
        output_file_handle.close()

    if reference_file_handle is not None:
        reference_file_handle.close()

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total batches processed: {num_batches}")
    print(f"Total operand_a values tested: {total_operand_a_values_to_test}")
    print(f"Number of operand_b per operand_a: {num_operand_b_per_a}")

    if check:
        if total_mismatches == 0:
            print(f"\nAll {num_batches} batches matched reference!")
        else:
            print(f"\nTotal mismatches: {total_mismatches}")

            if len(mismatch_records) > 0:
                mismatch_tensor = torch.tensor(mismatch_records, dtype=torch.int64)
                mismatch_file = "ttnn_add_exp9_mismatches.bin"

                with open(mismatch_file, "wb") as f:
                    f.write(total_mismatches.to_bytes(8, byteorder="little"))
                    mismatch_uint16 = mismatch_tensor.to(torch.uint16)
                    f.write(mismatch_uint16.numpy().tobytes())

                print(f"Saved {total_mismatches} mismatches to {mismatch_file}")

                print(f"\nFirst 5 mismatches:")
                for i in range(min(5, len(mismatch_records))):
                    rec = mismatch_records[i]
                    print(
                        f"  {i+1}. a=0x{rec[0]:04x}, b=0x{rec[1]:04x}, result=0x{rec[2]:04x}, expected=0x{rec[3]:04x}"
                    )
