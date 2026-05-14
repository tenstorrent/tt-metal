# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Capture ResNet50 trace and export to .ttb file.

Uses the existing ResNet50 test infrastructure and TracedModelExecutor pattern
to compile the model, capture a trace, and serialize it for lightweight replay.

Usage:
    python capture_resnet50.py [--output resnet50.ttb] [--batch-size 16]

Requirements:
    - Tenstorrent device available
    - torchvision installed (for ResNet50 weights)
    - TT_METAL_HOME and PYTHONPATH set to tt-metal root
"""

import argparse
import sys
import os
import torch

import ttnn
from trace_binary import export_trace


def collect_device_tensors_from_model(model, visited=None):
    """Recursively collect all allocated device tensors from a model object."""
    if visited is None:
        visited = set()
    obj_id = id(model)
    if obj_id in visited:
        return []
    visited.add(obj_id)

    tensors = []
    if isinstance(model, ttnn.Tensor):
        try:
            if model.is_allocated():
                model.buffer_address()
                tensors.append(model)
        except Exception:
            pass
        return tensors

    if isinstance(model, dict):
        for v in model.values():
            tensors.extend(collect_device_tensors_from_model(v, visited))
        return tensors

    if isinstance(model, (list, tuple)):
        for v in model:
            tensors.extend(collect_device_tensors_from_model(v, visited))
        return tensors

    if callable(model) and hasattr(model, "__closure__") and model.__closure__:
        for cell in model.__closure__:
            try:
                tensors.extend(collect_device_tensors_from_model(cell.cell_contents, visited))
            except ValueError:
                pass
        return tensors

    if hasattr(model, "__dict__"):
        for attr_name, attr_val in model.__dict__.items():
            if attr_name.startswith("_"):
                continue
            tensors.extend(collect_device_tensors_from_model(attr_val, visited))

    return tensors


def main():
    parser = argparse.ArgumentParser(description="Capture ResNet50 trace to .ttb")
    parser.add_argument("--output", default="resnet50.ttb", help="Output .ttb file path")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (16 or 32)")
    parser.add_argument("--save-ref", default=None, help="Directory to save reference input/output for evaluation")
    args = parser.parse_args()

    ref_dir = args.save_ref

    # Import model infrastructure
    from models.demos.vision.classification.resnet50.ttnn_resnet.tt.ttnn_functional_resnet50 import resnet50
    from models.demos.vision.classification.resnet50.ttnn_resnet.tt.custom_preprocessing import (
        create_custom_mesh_preprocessor,
    )
    from ttnn.model_preprocessing import preprocess_model_parameters
    from models.common.utility_functions import divup

    batch_size = args.batch_size
    act_dtype = ttnn.bfloat8_b
    weight_dtype = ttnn.bfloat8_b
    math_fidelity = ttnn.MathFidelity.LoFi

    print(f"Opening device (trace_region_size=5554176, l1_small_size=24576)...")
    device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
        l1_small_size=24576,
        trace_region_size=5554176,
    )

    print(f"Loading ResNet50 model (batch_size={batch_size})...")
    import torchvision

    torch_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).eval()

    model_config = {
        "MATH_FIDELITY": math_fidelity,
        "WEIGHTS_DTYPE": weight_dtype,
        "ACTIVATIONS_DTYPE": act_dtype,
    }

    input_shape = (batch_size, 3, 224, 224)
    torch.manual_seed(42)
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

    # Compute PyTorch reference output before loading to device (to save memory)
    pytorch_output = None
    if ref_dir:
        print("Computing PyTorch reference output...")
        torch_model.to(torch.bfloat16)
        with torch.no_grad():
            pytorch_output = torch_model(torch_input).float()
        torch_model.float()

    print("Preprocessing model parameters...")
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=None,
    )
    del torch_model
    import gc; gc.collect()

    resnet50_first_conv_kernel_size = 3
    resnet50_first_conv_stride = 2

    ttnn_model = resnet50(
        device=device,
        parameters=parameters,
        batch_size=batch_size,
        model_config=model_config,
        input_shape=input_shape,
        kernel_size=resnet50_first_conv_kernel_size,
        stride=resnet50_first_conv_stride,
        dealloc_input=True,
        final_output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Setup input memory config (L1 sharded)
    if batch_size == 16:
        core_grid = ttnn.CoreGrid(y=8, x=6)
    elif batch_size == 32:
        core_grid = ttnn.CoreGrid(y=10, x=13)
    else:
        raise ValueError(f"Unsupported batch size: {batch_size}")

    n, c, h, w = input_shape
    num_cores = core_grid.x * core_grid.y
    shard_h = (n * c * h + num_cores - 1) // num_cores
    grid_coord = ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, w), ttnn.ShardOrientation.ROW_MAJOR)
    l1_input_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )

    # DRAM interleaved input config (interleaved so C++ replay can write with matching layout)
    tt_input_host = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    dram_input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM
    )

    # Allocate persistent DRAM input tensor
    dram_input_tensor = ttnn.allocate_tensor_on_device(
        tt_input_host.shape, tt_input_host.dtype, tt_input_host.layout, device, dram_input_mem_config
    )

    # Compile run
    print("Running compile pass...")
    ttnn.copy_host_to_device_tensor(tt_input_host, dram_input_tensor)
    l1_input = ttnn.to_memory_config(dram_input_tensor, l1_input_mem_config)
    compile_output = ttnn_model(l1_input, device, {})
    ttnn.deallocate(l1_input)
    ttnn.deallocate(compile_output, force=True)
    print("Compile pass complete.")

    # Prepare for trace capture
    ttnn.copy_host_to_device_tensor(tt_input_host, dram_input_tensor)

    # Capture trace (includes DRAM→L1 transfer so C++ replay populates L1 automatically)
    print("Capturing trace...")
    trace_id = ttnn.begin_trace_capture(device)
    l1_input_for_trace = ttnn.to_memory_config(dram_input_tensor, l1_input_mem_config)
    output_tensor = ttnn_model(l1_input_for_trace, device, {})
    ttnn.end_trace_capture(device, trace_id)
    print("Trace captured.")

    # Verify trace replay
    print("Verifying trace replay...")
    ttnn.copy_host_to_device_tensor(tt_input_host, dram_input_tensor)
    ttnn.execute_trace(device, trace_id, blocking=True)

    trace_output = ttnn.to_torch(output_tensor).squeeze().float()
    print(f"Output shape: {trace_output.shape}")

    if ref_dir:
        os.makedirs(ref_dir, exist_ok=True)
        # Save input as raw bfloat16
        input_raw = tt_input_host.to_torch()
        input_bytes = bytes(input_raw.contiguous().untyped_storage())
        with open(os.path.join(ref_dir, "input.bin"), "wb") as f:
            f.write(input_bytes)
        # Save trace replay output
        torch.save(trace_output, os.path.join(ref_dir, "ttnn_trace_output.pt"))
        # Save pytorch reference output
        torch.save(pytorch_output, os.path.join(ref_dir, "pytorch_reference_output.pt"))
        del pytorch_output
        print(f"Reference data saved to {ref_dir}/")
    print()

    # Collect model weight tensors for serialization
    print("Collecting model weight tensors...")
    weight_tensors = collect_device_tensors_from_model(ttnn_model)

    # Sort by address so C++ replay allocates in the same order
    weight_tensors.sort(key=lambda t: t.buffer_address())

    # Deduplicate by address
    seen_addrs = set()
    unique_tensors = []
    for t in weight_tensors:
        addr = t.buffer_address()
        if addr not in seen_addrs:
            seen_addrs.add(addr)
            unique_tensors.append(t)
    weight_tensors = unique_tensors

    print(f"Found {len(weight_tensors)} unique weight tensors")

    # Export to .ttb
    io_tensors = {
        "dram_input": dram_input_tensor,
        "output": output_tensor,
    }
    export_trace(device, trace_id, args.output, io_tensors, persistent_tensors=weight_tensors)

    ttnn.release_trace(device, trace_id)
    ttnn.close_mesh_device(device)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
