import os
import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc, comp_allclose


from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup


def test_ff1_per_device_from_saved():
    """
    Test that loads saved concatenated FF1 inputs/weights/outputs from llama_mlp.py,
    splits them into per-device chunks, and compares torch matmul with ttnn outputs.
    """
    save_path = "models/demos/llama3_70b_galaxy/tests/"

    # Check if saved files exist (concatenated format)
    x_path = f"{save_path}ff1_x_concat.pt"
    w1_path = f"{save_path}ff1_w1_concat.pt"
    out_path = f"{save_path}ff1_out_concat.pt"

    if not os.path.exists(x_path):
        pytest.skip(f"Saved tensors not found at {x_path}. Run llama_mlp with PCC < 0.90 to generate.")

    # Load saved concatenated tensors
    x_concat = torch.load(x_path)
    w1_concat = torch.load(w1_path)
    out_concat = torch.load(out_path)

    print(f"Loaded concatenated tensors: x={x_concat.shape}, w1={w1_concat.shape}, out={out_concat.shape}")

    # Split concatenated tensors into per-device chunks
    # ConcatMesh2dToTensor with dims=(0, 1) and mesh_shape=[8, 4] means:
    # - dim 0 concatenates 8 rows
    # - dim 1 concatenates 4 columns
    num_rows, num_cols = 8, 4
    x_chunk_size_0 = x_concat.shape[0] // num_rows
    x_chunk_size_1 = x_concat.shape[1] // num_cols
    w1_chunk_size_0 = w1_concat.shape[0] // num_rows
    w1_chunk_size_1 = w1_concat.shape[1] // num_cols
    out_chunk_size_0 = out_concat.shape[0] // num_rows
    out_chunk_size_1 = out_concat.shape[1] // num_cols

    print("=" * 80)
    print("Per-device FF1 matmul PCC comparison (from saved concatenated tensors):")
    print("=" * 80)

    failing_devices = []
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            dev_idx = row_idx * num_cols + col_idx
            # Extract per-device chunks
            x_dev = x_concat[
                row_idx * x_chunk_size_0 : (row_idx + 1) * x_chunk_size_0,
                col_idx * x_chunk_size_1 : (col_idx + 1) * x_chunk_size_1,
                :,
                :,
            ].to(torch.bfloat16)
            w1_dev = w1_concat[
                row_idx * w1_chunk_size_0 : (row_idx + 1) * w1_chunk_size_0,
                col_idx * w1_chunk_size_1 : (col_idx + 1) * w1_chunk_size_1,
                :,
                :,
            ].to(torch.bfloat16)
            ttnn_out = out_concat[
                row_idx * out_chunk_size_0 : (row_idx + 1) * out_chunk_size_0,
                col_idx * out_chunk_size_1 : (col_idx + 1) * out_chunk_size_1,
                :,
                :,
            ].to(torch.bfloat16)

            # Run torch matmul
            torch_out = x_dev @ w1_dev
            # Compare
            passing, pcc_message = comp_pcc(torch_out, ttnn_out)
            # Extract PCC value
            try:
                pcc_val = float(pcc_message.split()[-1])
            except (ValueError, IndexError):
                pcc_val = 0.0

            status = "PASS" if pcc_val >= 0.90 else "FAIL"
            print(
                f"  Device [{row_idx}, {col_idx}] (idx={dev_idx}): [{status}] {pcc_message} | "
                f"x_shape={x_dev.shape}, w1_shape={w1_dev.shape}, out_shape={ttnn_out.shape}"
            )

            if pcc_val < 0.90:
                failing_devices.append((dev_idx, row_idx, col_idx, pcc_val))

    print("=" * 80)

    if failing_devices:
        print(f"\nFailing devices ({len(failing_devices)}):")
        for dev_idx, row_idx, col_idx, pcc_val in failing_devices:
            print(f"  Device [{row_idx}, {col_idx}] (idx={dev_idx}): PCC = {pcc_val}")
        print()

    assert len(failing_devices) == 0, f"{len(failing_devices)} devices have PCC < 0.90"


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_matmul(mesh_device):
    w_in = torch.load("models/demos/llama3_70b_galaxy/tests/w1_weight.pt")
    x_in = torch.load("models/demos/llama3_70b_galaxy/tests/w1_in.pt")
    ref_after_w1 = torch.load("models/demos/llama3_70b_galaxy/tests/ref_after_w1.pt")
    comp_out = torch.load("models/demos/llama3_70b_galaxy/tests/comp_out.pt")

    w_in = ttnn.from_torch(
        w_in,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=[8, 4]),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    x_in = ttnn.from_torch(
        x_in,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=[8, 4]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out = ttnn.matmul(x_in, w_in, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 0), mesh_shape=[8, 4])
    out = ttnn.to_torch(out, mesh_composer=mesh_composer).sum(0)
    out = torch.permute(out, (1, 0, 2))
    passing, pcc_message = comp_pcc(ref_after_w1, out)
    print(f"Non-Prefetch Matmul PCC with reference: {pcc_message}")
    print(comp_allclose(ref_after_w1, out))

    passing, pcc_message = comp_pcc(comp_out, out)
    print(f"Non-Prefetch Matmul PCC with ring matmul output: {pcc_message}")
    print(comp_allclose(comp_out, out))


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_ring_matmul(mesh_device):
    """
    Load concatenated tensors saved from llama_mlp.py (when PCC < 0.90),
    split them across devices using ShardTensor2dMesh, run ring matmul,
    and compare per-device outputs with torch matmul.
    """
    save_path = "models/demos/llama3_70b_galaxy/tests/"

    # Load concatenated tensors (saved from llama_mlp.py)
    x_concat_path = f"{save_path}ff1_x_concat.pt"
    w1_concat_path = f"{save_path}ff1_w1_concat.pt"
    out_concat_path = f"{save_path}ff1_out_concat.pt"

    if not os.path.exists(x_concat_path):
        pytest.skip(f"Concatenated tensors not found at {x_concat_path}. Run llama_mlp with PCC < 0.90 to generate.")

    x_in = torch.load(x_concat_path)
    w_in = torch.load(w1_concat_path)
    original_ttnn_out_concat = torch.load(out_concat_path)

    print(f"Loaded concatenated tensors: x_in={x_in.shape}, w_in={w_in.shape}, out={original_ttnn_out_concat.shape}")

    RING_SIZE = 24
    model_args = TtQwenModelArgs(mesh_device, max_batch_size=32, dummy_weights=False, max_seq_len=128)

    pc_1_3 = model_args.matmul_1d_ring_config(
        1,  # B
        32,  # M
        5120 // 4,  # K = 1280
        3840,  # Use padded N
        RING_SIZE,
        prefetch=False,
    )

    compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    out_memory_config = model_args.model_config["SHARDED_FF12_OUT_RING_MEMCFG"]
    in_memory_config = model_args.model_config["SHARDED_FF12_RING_MEMCFG"]
    weight_memory_config = model_args.create_dram_sharded_mem_config(
        k=1280,
        n=3840,
    )

    # Use ShardTensor2dMesh to split concatenated tensors across devices
    w_in_tt = ttnn.from_torch(
        w_in,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=[8, 4]),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=weight_memory_config,
    )

    x_in_tt = ttnn.from_torch(
        x_in,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=[8, 4]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=in_memory_config,
    )

    # Save inputs and weights for all devices before matmul
    x_device_tensors = ttnn.get_device_tensors(x_in_tt)
    w_device_tensors = ttnn.get_device_tensors(w_in_tt)
    x_torch_per_device = [ttnn.to_torch(t) for t in x_device_tensors]
    w_torch_per_device = [ttnn.to_torch(t) for t in w_device_tensors]

    # Run ring matmul
    out_ring_mm = ttnn.linear(
        x_in_tt,
        w_in_tt,
        compute_kernel_config=compute_kernel_config_hifi2,
        dtype=ttnn.bfloat8_b,
        program_config=pc_1_3,
        memory_config=out_memory_config,
        core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_1_3 else None,
    )

    # Get ttnn outputs per device
    out_device_tensors = ttnn.get_device_tensors(out_ring_mm)
    out_torch_per_device = [ttnn.to_torch(t) for t in out_device_tensors]

    # Compare per-device: torch matmul vs ttnn output
    print("=" * 80)
    print("Per-device Ring Matmul PCC comparison (torch matmul vs ttnn output):")
    print("=" * 80)
    pcc_values = []
    all_devices_data = []
    for dev_idx in range(len(x_torch_per_device)):
        row_idx = dev_idx // 4  # 8 rows
        col_idx = dev_idx % 4  # 4 columns
        # Get input and weight for this device
        x_dev = x_torch_per_device[dev_idx].to(torch.bfloat16)
        w_dev = w_torch_per_device[dev_idx].to(torch.bfloat16)
        # Run torch matmul
        torch_out = x_dev @ w_dev
        # Get ttnn output for this device
        ttnn_out = out_torch_per_device[dev_idx].to(torch.bfloat16)
        # Compare
        passing, pcc_message = comp_pcc(torch_out, ttnn_out)
        try:
            pcc_val = float(pcc_message)
        except (ValueError, IndexError):
            pcc_val = 0.0
        pcc_values.append(pcc_val)
        status = "PASS" if pcc_val >= 0.90 else "FAIL"
        print(
            f"  Device [{row_idx}, {col_idx}] (idx={dev_idx}): [{status}] {pcc_message} | "
            f"x_shape={x_dev.shape}, w_shape={w_dev.shape}, out_shape={ttnn_out.shape}"
        )
        all_devices_data.append((dev_idx, row_idx, col_idx, x_dev, w_dev, torch_out, ttnn_out, pcc_val))
    print("=" * 80)
    print()

    # Print tensor subsections for ALL devices
    print("=" * 80)
    print("TENSOR SUBSECTIONS FOR ALL DEVICES (test_ring_matmul):")
    print("=" * 80)
    for dev_idx, row_idx, col_idx, x_dev, w_dev, torch_out, ttnn_out, pcc_val in all_devices_data:
        status = "PASS" if pcc_val >= 0.90 else "FAIL"
        print(f"\n--- Device [{row_idx}, {col_idx}] (idx={dev_idx}) [{status}] PCC={pcc_val:.4f} ---")
        print(f"Input x[0,0,:8,:8]:\n{x_dev[0,0,:8,:8].float()}")
        print(f"Weight w1[0,0,:8,:8]:\n{w_dev[0,0,:8,:8].float()}")
        print(f"Torch output[0,0,:8,:8]:\n{torch_out[0,0,:8,:8].float()}")
        print(f"TTNN output[0,0,:8,:8]:\n{ttnn_out[0,0,:8,:8].float()}")
        print(f"Diff (torch - ttnn)[0,0,:8,:8]:\n{(torch_out[0,0,:8,:8] - ttnn_out[0,0,:8,:8]).float()}")
    print("=" * 80)
    print()

    # Save all inputs/outputs if any device has PCC < 0.90
    # min_pcc = min(pcc_values)
    # if min_pcc < 0.90:
    #     print(f"WARNING: Min PCC {min_pcc} < 0.90, saving test inputs/outputs...")
    #     torch.save(x_torch_per_device, f"{save_path}test_x_per_device.pt")
    #     torch.save(w_torch_per_device, f"{save_path}test_w1_per_device.pt")
    #     torch.save(out_torch_per_device, f"{save_path}test_out_per_device.pt")
    #     print(f"Saved: test_x_per_device.pt, test_w1_per_device.pt, test_out_per_device.pt")

    # Also compare concatenated output
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=[8, 4])
    out_concat = ttnn.to_torch(out_ring_mm, mesh_composer=mesh_composer)
    passing, pcc_message = comp_pcc(original_ttnn_out_concat, out_concat)
    print(f"Concatenated output PCC (origi-nal vs test): {pcc_message}")
    print(comp_allclose(original_ttnn_out_concat, out_concat))


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_prefetcher_ring_matmul(mesh_device):
    """
    Load concatenated tensors saved from llama_mlp.py (when PCC < 0.90),
    split them across devices using ShardTensor2dMesh, run prefetcher ring matmul,
    and compare per-device outputs with torch matmul.
    """
    save_path = "models/demos/llama3_70b_galaxy/tests/"

    # Load concatenated tensors (saved from llama_mlp.py)
    x_concat_path = f"{save_path}ff1_x_concat.pt"
    w1_concat_path = f"{save_path}ff1_w1_concat.pt"
    out_concat_path = f"{save_path}ff1_out_concat.pt"

    if not os.path.exists(x_concat_path):
        pytest.skip(f"Concatenated tensors not found at {x_concat_path}. Run llama_mlp with PCC < 0.90 to generate.")

    x_in = torch.load(x_concat_path)
    w_in = torch.load(w1_concat_path)
    original_ttnn_out_concat = torch.load(out_concat_path)

    print(f"Loaded concatenated tensors: x_in={x_in.shape}, w_in={w_in.shape}, out={original_ttnn_out_concat.shape}")

    # prefetcher setup
    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=1,
        n_layers=1,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    RING_SIZE = 24
    model_args = TtQwenModelArgs(mesh_device, max_batch_size=32, dummy_weights=False, max_seq_len=128)

    pc_1_3 = model_args.matmul_1d_ring_config(
        1,  # B
        32,  # M
        5120 // 4,  # K = 1280
        3840,  # Use padded N
        RING_SIZE,
    )

    compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )

    out_memory_config = model_args.model_config["SHARDED_FF12_OUT_RING_MEMCFG"]
    in_memory_config = model_args.model_config["SHARDED_FF12_RING_MEMCFG"]
    weight_memory_config = model_args.create_dram_sharded_mem_config(
        k=1280,
        n=3840,
    )

    # Use ShardTensor2dMesh to split concatenated tensors across devices
    w_in_tt = ttnn.from_torch(
        w_in,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=[8, 4]),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=weight_memory_config,
    )

    x_in_tt = ttnn.from_torch(
        x_in,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=[8, 4]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=in_memory_config,
    )

    # Save inputs and weights for all devices before matmul
    x_device_tensors = ttnn.get_device_tensors(x_in_tt)
    w_device_tensors = ttnn.get_device_tensors(w_in_tt)
    x_torch_per_device = [ttnn.to_torch(t) for t in x_device_tensors]
    w_torch_per_device = [ttnn.to_torch(t) for t in w_device_tensors]

    prefetcher_setup.insert_tensor(w_in_tt)
    prefetcher_setup.create_global_cb()

    ttnn.dram_prefetcher(
        prefetcher_setup.get_input_tensors(),
        num_layers=1,
        global_cb=prefetcher_setup.global_circular_buffer,
    )
    mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])

    # Run prefetcher ring matmul
    out_ring_mm = ttnn.linear(
        x_in_tt,
        w_in_tt,
        compute_kernel_config=compute_kernel_config_hifi2,
        dtype=ttnn.bfloat8_b,
        program_config=pc_1_3,
        memory_config=out_memory_config,
        core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_1_3 else None,
        global_cb=prefetcher_setup.global_circular_buffer,
        sub_device_id=prefetcher_setup.worker_sub_device_id,
    )

    # Get ttnn outputs per device
    out_device_tensors = ttnn.get_device_tensors(out_ring_mm)
    out_torch_per_device = [ttnn.to_torch(t) for t in out_device_tensors]

    # Compare per-device: torch matmul vs ttnn output
    print("=" * 80)
    print("Per-device Prefetcher Ring Matmul PCC comparison (torch matmul vs ttnn output):")
    print("=" * 80)
    pcc_values = []
    all_devices_data = []
    for dev_idx in range(len(x_torch_per_device)):
        row_idx = dev_idx // 4  # 8 rows
        col_idx = dev_idx % 4  # 4 columns
        # Get input and weight for this device
        x_dev = x_torch_per_device[dev_idx].to(torch.bfloat16)
        w_dev = w_torch_per_device[dev_idx].to(torch.bfloat16)
        # Run torch matmul
        torch_out = x_dev @ w_dev
        # Get ttnn output for this device
        ttnn_out = out_torch_per_device[dev_idx].to(torch.bfloat16)
        # Compare
        passing, pcc_message = comp_pcc(torch_out, ttnn_out)
        try:
            pcc_val = float(pcc_message)
        except (ValueError, IndexError):
            pcc_val = 0.0
        pcc_values.append(pcc_val)
        status = "PASS" if pcc_val >= 0.90 else "FAIL"
        print(
            f"  Device [{row_idx}, {col_idx}] (idx={dev_idx}): [{status}] {pcc_message} | "
            f"x_shape={x_dev.shape}, w_shape={w_dev.shape}, out_shape={ttnn_out.shape}"
        )
        all_devices_data.append((dev_idx, row_idx, col_idx, x_dev, w_dev, torch_out, ttnn_out, pcc_val))
    print("=" * 80)
    print()

    # Print tensor subsections for ALL devices
    print("=" * 80)
    print("TENSOR SUBSECTIONS FOR ALL DEVICES (test_prefetcher_ring_matmul):")
    print("=" * 80)
    for dev_idx, row_idx, col_idx, x_dev, w_dev, torch_out, ttnn_out, pcc_val in all_devices_data:
        status = "PASS" if pcc_val >= 0.90 else "FAIL"
        print(f"\n--- Device [{row_idx}, {col_idx}] (idx={dev_idx}) [{status}] PCC={pcc_val:.4f} ---")
        print(f"Input x[0,0,:8,:8]:\n{x_dev[0,0,:8,:8].float()}")
        print(f"Weight w1[0,0,:8,:8]:\n{w_dev[0,0,:8,:8].float()}")
        print(f"Torch output[0,0,:8,:8]:\n{torch_out[0,0,:8,:8].float()}")
        print(f"TTNN output[0,0,:8,:8]:\n{ttnn_out[0,0,:8,:8].float()}")
        print(f"Diff (torch - ttnn)[0,0,:8,:8]:\n{(torch_out[0,0,:8,:8] - ttnn_out[0,0,:8,:8]).float()}")
    print("=" * 80)
    print()

    # Also compare concatenated output
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=[8, 4])
    out_concat = ttnn.to_torch(out_ring_mm, mesh_composer=mesh_composer)
    passing, pcc_message = comp_pcc(original_ttnn_out_concat, out_concat)
    print(f"Concatenated output PCC (original vs test): {pcc_message}")
    print(comp_allclose(original_ttnn_out_concat, out_concat))
