import pytest
import torch
import ttnn
import os
import sys

# Use local utils for device management
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils

@pytest.mark.nightly
@pytest.mark.single_device
def test_matmul_index_precision():
    print("=" * 70)
    print("MATMUL SANITY TEST: FLOAT32 vs Precise Integer Indexing")
    print("=" * 70)

    device = utils.DeviceGetter.get_device((1, 1))
    MC = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)

    # 1. Setup Shapes (Matching production model)
    N, C, H, W = 8, 32, 25, 34
    num_indices = 4096 
    
    print(f"\n--- Generating Random Data (Shape: {N}, {C}, {H}, {W}) ---")
    # Coordinates (Simulating the grid_sample input grid)
    ix = (torch.rand(num_indices, 1) * (W - 1)).float()
    iy = (torch.rand(num_indices, 1) * (H - 1)).float()
    
    # Indices for the base corner (Corner 0)
    batch_idx = torch.randint(0, N, (num_indices, 1)).float()
    chan_idx  = torch.randint(0, C, (num_indices, 1)).float()
    y_idx     = torch.floor(iy)
    x_idx     = torch.floor(ix)
    
    indices_f32 = torch.cat([batch_idx, chan_idx, y_idx, x_idx], dim=-1)
    
    # Feature map (High-entropy random noise)
    feat_map_cpu = torch.randn(N*C*H*W, 1).float() * 10.0
    
    # 2. Golden Reference (Perfect Integer Math on CPU)
    golden_ref = (batch_idx.long() * 27200 + chan_idx.long() * 850 + y_idx.long() * 34 + x_idx.long())
    
    # 3. Running on Device via ttnn.matmul
    tt_indices = ttnn.from_torch(indices_f32, device=device, layout=ttnn.Layout.TILE, dtype=ttnn.DataType.FLOAT32)
    
    # Stride tensor [27200, 850, 34, 1]
    stride_f32 = torch.tensor([[27200.0], [850.0], [34.0], [1.0]], dtype=torch.float32)
    tt_stride = ttnn.from_torch(stride_f32, device=device, layout=ttnn.Layout.TILE, dtype=ttnn.DataType.FLOAT32)
    
    print("\n--- Running ttnn.matmul (FLOAT32) on hardware ---")
    tt_output = ttnn.matmul(
        tt_indices, tt_stride,
        dtype=ttnn.DataType.FLOAT32,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True))
    
    # Convert back to CPU
    tt_output_cpu = ttnn.to_torch(tt_output)
    tt_output_long = tt_output_cpu.long()
    
    # 4. Compare Results
    mismatches = (tt_output_long != golden_ref).sum().item()
    total = golden_ref.numel()
    match_pct = (1.0 - mismatches / total) * 100
    pcc = torch.corrcoef(torch.stack([tt_output_cpu.flatten(), golden_ref.float().flatten()]))[0, 1].item()
    
    print(f"\nResults for {total} indices:")
    print(f"  PCC (Float Accuracy):     {pcc:.9f}")
    print(f"  Exact Integer Matches:    {match_pct:.2f}% ({mismatches} mismatches)")
    
    # =========================================================================
    # 5. DEMONSTRATING PCC DROP: HOW WRONG ADDRESSES RUIN THE OUTPUT (on HW)
    # =========================================================================
    tt_feat_map = ttnn.from_torch(feat_map_cpu, device=device, layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.BFLOAT16)
    tt_idx_matmul = ttnn.from_torch(tt_output_long.reshape(1, total).to(torch.int32), 
                                    device=device, layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.UINT32)

    # Run ttnn.embedding on hardware
    tt_out_matmul = ttnn.embedding(tt_idx_matmul, tt_feat_map, layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.BFLOAT16)
    out_matmul_cpu = ttnn.to_torch(tt_out_matmul).float()
    out_golden_cpu = feat_map_cpu[golden_ref.flatten()].reshape(1, total, 1)

    pcc_float_path = torch.corrcoef(torch.stack([out_matmul_cpu.flatten(), out_golden_cpu.flatten()]))[0, 1].item()

    print(f"\nFinal Hardware Value Comparison (At Output Level):")
    print(f"  1. FLOAT32 Matmul Output PCC:  {pcc_float_path:.6f}")
    
    # Standard Pytest assertions
    assert pcc_float_path > 0.99, f"PCC drop detected: {pcc_float_path:.6f} (Internal bug in matmul indexing!)"
    assert mismatches == 0, f"Index mismatches detected: {mismatches} errors out of {total} indices."

if __name__ == "__main__":
    test_matmul_index_precision()
