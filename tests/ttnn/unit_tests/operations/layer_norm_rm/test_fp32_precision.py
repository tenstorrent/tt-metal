# Compare precision: fp32_dest_acc_en=True (with workaround) vs False
import pytest
import torch
import ttnn
from .layer_norm_rm import layer_norm_rm


SHAPES = [
    (1, 1, 32, 32),
    (1, 1, 32, 128),
    (1, 1, 64, 128),
    (1, 1, 32, 256),
    (4, 2, 64, 64),
]


def run_layer_norm(device, shape, fp32_dest):
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    W = shape[-1]
    torch_gamma = torch.randn(1, 1, 1, W, dtype=torch.bfloat16).abs() + 0.5
    torch_beta = torch.randn(1, 1, 1, W, dtype=torch.bfloat16) * 0.1

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_beta = ttnn.from_torch(
        torch_beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm_rm(ttnn_input, ttnn_gamma, ttnn_beta, fp32_dest_acc_en=fp32_dest)
    result = ttnn.to_torch(ttnn_output)

    # Compute fp32 reference
    inp_f32 = torch_input.float()
    gamma_f32 = torch_gamma.float()
    beta_f32 = torch_beta.float()
    mean = inp_f32.mean(dim=-1, keepdim=True)
    var = ((inp_f32 - mean) ** 2).mean(dim=-1, keepdim=True)
    normed = (inp_f32 - mean) / torch.sqrt(var + 1e-5)
    expected = gamma_f32 * normed + beta_f32

    return result.float(), expected


@pytest.mark.parametrize("shape", SHAPES, ids=[f"{'x'.join(str(d) for d in s)}" for s in SHAPES])
def test_fp32_precision_comparison(device, shape):
    """Compare fp32_dest_acc_en=True vs False against fp32 reference."""
    result_fp32, expected = run_layer_norm(device, shape, fp32_dest=True)
    result_fp16, _ = run_layer_norm(device, shape, fp32_dest=False)

    diff_fp32 = (result_fp32 - expected).abs()
    diff_fp16 = (result_fp16 - expected).abs()

    max_fp32 = diff_fp32.max().item()
    max_fp16 = diff_fp16.max().item()
    mean_fp32 = diff_fp32.mean().item()
    mean_fp16 = diff_fp16.mean().item()

    print(f"\n{'='*60}")
    print(f"Shape: {shape}")
    print(f"{'':>20} {'fp32_dest=True':>16} {'fp32_dest=False':>16}")
    print(f"{'Max abs error':>20} {max_fp32:>16.6f} {max_fp16:>16.6f}")
    print(f"{'Mean abs error':>20} {mean_fp32:>16.6f} {mean_fp16:>16.6f}")
    print(f"{'Improvement':>20} {max_fp16/max_fp32 if max_fp32 > 0 else 0:>15.2f}x {'(max)':>15}")
    print(f"{'':>20} {mean_fp16/mean_fp32 if mean_fp32 > 0 else 0:>15.2f}x {'(mean)':>15}")
    print(f"{'='*60}")

    # Both should be reasonably close to reference
    assert max_fp32 < 1.0, f"fp32 dest max error too large: {max_fp32}"
    assert max_fp16 < 1.0, f"fp16 dest max error too large: {max_fp16}"
