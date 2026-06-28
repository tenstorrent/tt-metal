import torch
import ttnn
import math
import sys

sys.path.insert(0, "tests/ttnn/unit_tests/operations/scaled_dot_product_attention")
import reference
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

# Use reference.py's exact inputs (fp32 with seed 0), converted to bf16
Q_bf16 = reference._Q.to(torch.bfloat16)
K_bf16 = reference._K.to(torch.bfloat16)
V_bf16 = reference._V.to(torch.bfloat16)

device = ttnn.open_device(device_id=0)
ttnn_Q = ttnn.from_torch(
    Q_bf16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
ttnn_K = ttnn.from_torch(
    K_bf16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
ttnn_V = ttnn.from_torch(
    V_bf16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)

output = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V)
torch_output = ttnn.to_torch(output)
print("=== TTNN Output (first 4x4) ===")
print(torch_output[0, 0, :4, :4])

# Compare with reference.py's Stage 14 output
# Run the full reference pipeline
Q_blk, m_i, l_i, O_i = reference.reference_phase_init()
scores, m_i, l_i, O_i = reference.reference_phase_qkt(Q_blk, m_i, l_i, O_i)
scores, m_i, l_i, O_i = reference.reference_phase_scale(scores, m_i, l_i, O_i)
scores_masked, m_i, l_i, O_i = reference.reference_phase_mask(scores, m_i, l_i, O_i)
scores_masked, m_blk, m_i, l_i, O_i = reference.reference_phase_rowmax(scores_masked, m_i, l_i, O_i)
scores_masked, m_blk, alpha, m_i, l_i, O_i = reference.reference_phase_alpha(scores_masked, m_blk, m_i, l_i, O_i)
scores_masked, m_blk, alpha, m_i, l_i, O_i = reference.reference_phase_rescale_o(
    scores_masked, m_blk, alpha, m_i, l_i, O_i
)
scores_masked, m_blk, alpha, m_i, l_i, O_i = reference.reference_phase_rescale_l(
    scores_masked, m_blk, alpha, m_i, l_i, O_i
)
scores_masked, m_blk, alpha, m_i, l_i, O_i = reference.reference_phase_subtract_max(
    scores_masked, m_blk, alpha, m_i, l_i, O_i
)
exp_scores, m_blk, alpha, m_i, l_i, O_i = reference.reference_phase_exp(scores_masked, m_blk, alpha, m_i, l_i, O_i)
exp_scores, m_blk, alpha, l_blk, m_i, l_i, O_i = reference.reference_phase_rowsum(
    exp_scores, m_blk, alpha, m_i, l_i, O_i
)
exp_scores, m_blk, alpha, l_blk, m_i, l_i, O_i = reference.reference_phase_update_l(
    exp_scores, m_blk, alpha, l_blk, m_i, l_i, O_i
)
m_blk, alpha, l_blk, m_i, l_i, O_i = reference.reference_phase_pv(exp_scores, m_blk, alpha, l_blk, m_i, l_i, O_i)
m_i, l_i, O_i = reference.reference_phase_update_m(m_blk, alpha, l_blk, m_i, l_i, O_i)
ref_output = reference.reference_phase_output(m_i, l_i, O_i)
print("\n=== Reference Stage 14 Output (first 4x4) ===")
print(ref_output[:4, :4])

max_diff = (torch_output[0, 0].float() - ref_output).abs().max().item()
print(f"\nMax diff: {max_diff:.2e}")

ttnn.close_device(device)
