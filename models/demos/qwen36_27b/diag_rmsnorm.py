"""Pinpoint RMSNorm bug: compare HF vs TT weight values and computation."""
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from pathlib import Path

SNAP = "/home/yito/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"

# Load HF model (4 layers) and extract input_layernorm weight
from transformers import AutoModelForCausalLM, AutoConfig
config = AutoConfig.from_pretrained(SNAP, trust_remote_code=True)
config.text_config.num_hidden_layers = 4
hf_model = AutoModelForCausalLM.from_pretrained(
    SNAP, config=config, torch_dtype=torch.bfloat16,
    trust_remote_code=True, device_map="cpu"
)
hf_model.eval()

lm = hf_model.language_model if hasattr(hf_model, 'language_model') else hf_model.model

# Get HF weight
hf_ln_weight = lm.layers[0].input_layernorm.weight.data
print(f"HF input_layernorm weight: shape={hf_ln_weight.shape}, dtype={hf_ln_weight.dtype}")
print(f"  norm={hf_ln_weight.float().norm():.4f}, min={hf_ln_weight.float().min():.4f}, max={hf_ln_weight.float().max():.4f}, mean={hf_ln_weight.float().mean():.4f}")
print(f"  first10: {hf_ln_weight[:10]}")

# Load our state dict
from models.demos.qwen36_27b.tt.load_weights import load_state_dict
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig

tt_config = Qwen36ModelConfig()
tt_config.num_hidden_layers = 4
sd = load_state_dict(tt_config, max_layers=4, model_path=SNAP)

tt_ln_weight = sd["model.layers.0.input_layernorm.weight"]
print(f"\nTT input_layernorm weight: shape={tt_ln_weight.shape}, dtype={tt_ln_weight.dtype}")
print(f"  norm={tt_ln_weight.float().norm():.4f}, min={tt_ln_weight.float().min():.4f}, max={tt_ln_weight.float().max():.4f}, mean={tt_ln_weight.float().mean():.4f}")
print(f"  first10: {tt_ln_weight[:10]}")

# Compare weights
weight_cos = F.cosine_similarity(hf_ln_weight.float().flatten().unsqueeze(0),
                                  tt_ln_weight.float().flatten().unsqueeze(0))
weight_diff = (hf_ln_weight.float() - tt_ln_weight.float()).abs().max()
print(f"\nWeight comparison: cosine={weight_cos.item():.6f}, max_diff={weight_diff:.6f}")

# Run RMSNorm manually on same input
embed_w = sd["model.embed_tokens.weight"]
x = embed_w[151644].unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
print(f"\nInput: shape={x.shape}, norm={x.float().norm():.6f}")

# HF RMSNorm
x_f = x.float()
variance = x_f.pow(2).mean(-1, keepdim=True)
x_normed = x_f * torch.rsqrt(variance + 1e-6)
hf_result = (hf_ln_weight.float() * x_normed).to(torch.bfloat16)
print(f"HF RMSNorm result: norm={hf_result.float().norm():.4f}, first5: {hf_result.flatten()[:5]}")

# Our SimpleRMSNorm computation (simulated)
x_4d = x.unsqueeze(0)  # [1, 1, 1, hidden_size] - what ttnn would give
x_f_4d = x_4d.float()
variance_4d = x_f_4d.pow(2).mean(-1, keepdim=True)
x_normed_4d = x_f_4d * torch.rsqrt(variance_4d + 1e-6)
tt_result = (tt_ln_weight * x_normed_4d).to(torch.bfloat16)
print(f"TT RMSNorm result: norm={tt_result.float().norm():.4f}, first5: {tt_result.flatten()[:5]}")

result_cos = F.cosine_similarity(hf_result.float().flatten().unsqueeze(0),
                                  tt_result.float().flatten().unsqueeze(0))
print(f"Result cosine sim: {result_cos.item():.6f}")

# Now test with ttnn roundtrip to check if device transfer introduces errors
print("\n=== With ttnn roundtrip ===")
import ttnn
device = ttnn.open_device(device_id=0)
try:
    x_tt = ttnn.from_torch(x_4d.to(torch.bfloat16), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device)
    x_back = ttnn.to_torch(x_tt)
    print(f"ttnn roundtrip: shape={x_back.shape}, norm={x_back.float().norm():.6f}")
    roundtrip_cos = F.cosine_similarity(x_4d.float().flatten().unsqueeze(0),
                                         x_back.float().flatten().unsqueeze(0))
    print(f"Roundtrip cosine sim: {roundtrip_cos.item():.6f}")

    # Actual SimpleRMSNorm with ttnn
    from models.demos.qwen36_27b.tt.decoder import SimpleRMSNorm
    ln = SimpleRMSNorm(device, 5120, sd, "model.layers.0.input_layernorm")
    print(f"\nSimpleRMSNorm weight: shape={ln.weight.shape}, dtype={ln.weight.dtype}")
    print(f"  norm={ln.weight.float().norm():.4f}, first10: {ln.weight[:10]}")

    ln_out = ln(x_tt)
    ln_out_cpu = ttnn.to_torch(ln_out).float()
    print(f"SimpleRMSNorm output: shape={ln_out_cpu.shape}, norm={ln_out_cpu.norm():.4f}")
    print(f"  first5: {ln_out_cpu.flatten()[:5]}")

    actual_cos = F.cosine_similarity(hf_result.float().flatten().unsqueeze(0),
                                      ln_out_cpu.flatten()[:5120].unsqueeze(0))
    print(f"SimpleRMSNorm vs HF cosine sim: {actual_cos.item():.6f}")

finally:
    ttnn.close_device(device)
