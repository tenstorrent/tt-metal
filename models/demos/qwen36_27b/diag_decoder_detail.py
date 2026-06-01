"""
Decoder layer step-by-step: find where HF vs TT diverge.
Embedding → RMSNorm → DeltaNet → residual → RMSNorm → MLP → residual
"""
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from pathlib import Path

SNAP = "/home/yito/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"

# Load HF model (4 layers)
from transformers import AutoModelForCausalLM, AutoConfig
config = AutoConfig.from_pretrained(SNAP, trust_remote_code=True)
config.text_config.num_hidden_layers = 4
hf_model = AutoModelForCausalLM.from_pretrained(
    SNAP, config=config, torch_dtype=torch.bfloat16,
    trust_remote_code=True, device_map="cpu"
)
hf_model.eval()

lm = hf_model.language_model if hasattr(hf_model, 'language_model') else hf_model.model

# Hook all sub-modules of layer 0
hooks = {}
def make_hook(name):
    def fn(module, input, output):
        inp = input
        while isinstance(inp, tuple) and len(inp) > 0:
            inp = inp[0]
        if isinstance(inp, torch.Tensor):
            inp_t = inp.detach().float()
        else:
            inp_t = torch.tensor(0.0)
        out = output
        while isinstance(out, tuple) and len(out) > 0:
            out = out[0]
        if isinstance(out, torch.Tensor):
            out_t = out.detach().float()
        else:
            out_t = torch.tensor(0.0)
        hooks[name] = (inp_t, out_t)
    return fn

layer0 = lm.layers[0]
layer0.input_layernorm.register_forward_hook(make_hook("input_ln"))
if hasattr(layer0, 'linear_attn'):
    layer0.linear_attn.register_forward_hook(make_hook("deltanet"))
elif hasattr(layer0, 'token_mixer'):
    layer0.token_mixer.register_forward_hook(make_hook("deltanet"))
layer0.post_attention_layernorm.register_forward_hook(make_hook("post_attn_ln"))
layer0.mlp.register_forward_hook(make_hook("mlp"))
layer0.register_forward_hook(make_hook("layer0_full"))
lm.embed_tokens.register_forward_hook(make_hook("embed"))

input_ids = torch.tensor([[151644]])
with torch.no_grad():
    hf_out = hf_model(input_ids)

print("=== HF Intermediate States (Layer 0) ===")
for name in ["embed", "input_ln", "deltanet", "post_attn_ln", "mlp", "layer0_full"]:
    if name in hooks:
        inp, out = hooks[name]
        print(f"  {name:20s}: in_norm={inp.norm():10.4f}, out_norm={out.norm():10.4f}, out_first5={out.flatten()[:5]}")

hf_embed = hooks["embed"][1]
hf_input_ln_out = hooks["input_ln"][1]
hf_deltanet_out = hooks["deltanet"][1]
hf_post_ln_out = hooks["post_attn_ln"][1]
hf_mlp_out = hooks["mlp"][1]
hf_layer0_out = hooks["layer0_full"][1]

# Manually compute residual connections to verify
hf_after_attn_residual = hf_embed + hf_deltanet_out
print(f"\n  Manual after-attn residual norm: {hf_after_attn_residual.norm():.4f}")
print(f"  post_attn_ln input norm:        {hooks['post_attn_ln'][0].norm():.4f}")

# Now TT
print("\n=== TT Implementation (Layer 0) ===")
import ttnn
device = ttnn.open_device(device_id=0)
try:
    from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
    from models.demos.qwen36_27b.tt.load_weights import load_state_dict
    from models.demos.qwen36_27b.tt.model import TtQwen36Model
    from models.demos.qwen36_27b.tt.decoder import TtHybridDecoderLayer, SimpleRMSNorm
    from models.demos.qwen36_27b.tt.deltanet import TtDeltaNetState

    tt_config = Qwen36ModelConfig()
    tt_config.num_hidden_layers = 4
    sd = load_state_dict(tt_config, max_layers=4, model_path=SNAP)

    model = TtQwen36Model(device, sd, tt_config)
    del sd

    # Get embedding
    hidden_tt = model.embed(input_ids)
    h_cpu = ttnn.to_torch(hidden_tt).float()
    embed_cos = F.cosine_similarity(hf_embed.flatten().unsqueeze(0), h_cpu.flatten().unsqueeze(0))
    print(f"  Embedding cosine sim: {embed_cos.item():.6f}")

    # Step through layer 0 manually
    layer = model.layers[0]
    deltanet_state = model.create_deltanet_state()

    # input_layernorm
    normed = layer.input_layernorm(hidden_tt)
    normed_cpu = ttnn.to_torch(normed).float()
    ln_cos = F.cosine_similarity(hf_input_ln_out.flatten().unsqueeze(0), normed_cpu.flatten().unsqueeze(0))
    ln_diff = (hf_input_ln_out.flatten() - normed_cpu.flatten()).abs().max()
    print(f"  input_layernorm: cosine={ln_cos.item():.6f}, max_diff={ln_diff:.6f}, tt_norm={normed_cpu.norm():.4f}")

    # DeltaNet
    delta_out = layer.token_mixer(normed, deltanet_state, mode="decode")
    delta_cpu = ttnn.to_torch(delta_out).float()
    delta_cos = F.cosine_similarity(hf_deltanet_out.flatten().unsqueeze(0), delta_cpu.flatten().unsqueeze(0))
    delta_diff = (hf_deltanet_out.flatten() - delta_cpu.flatten()).abs().max()
    print(f"  DeltaNet output: cosine={delta_cos.item():.6f}, max_diff={delta_diff:.6f}, tt_norm={delta_cpu.norm():.4f}, hf_norm={hf_deltanet_out.norm():.4f}")

    # Residual
    after_attn = ttnn.add(hidden_tt, delta_out)
    after_attn_cpu = ttnn.to_torch(after_attn).float()
    residual_cos = F.cosine_similarity(hf_after_attn_residual.flatten().unsqueeze(0), after_attn_cpu.flatten().unsqueeze(0))
    print(f"  After-attn residual: cosine={residual_cos.item():.6f}, tt_norm={after_attn_cpu.norm():.4f}, hf_norm={hf_after_attn_residual.norm():.4f}")

    # post_attention_layernorm
    post_normed = layer.post_attention_layernorm(after_attn)
    post_normed_cpu = ttnn.to_torch(post_normed).float()
    post_cos = F.cosine_similarity(hf_post_ln_out.flatten().unsqueeze(0), post_normed_cpu.flatten().unsqueeze(0))
    print(f"  post_attn_layernorm: cosine={post_cos.item():.6f}, tt_norm={post_normed_cpu.norm():.4f}, hf_norm={hf_post_ln_out.norm():.4f}")

    # MLP
    mlp_out = layer.mlp(post_normed)
    mlp_cpu = ttnn.to_torch(mlp_out).float()
    mlp_cos = F.cosine_similarity(hf_mlp_out.flatten().unsqueeze(0), mlp_cpu.flatten().unsqueeze(0))
    mlp_diff = (hf_mlp_out.flatten() - mlp_cpu.flatten()).abs().max()
    print(f"  MLP output: cosine={mlp_cos.item():.6f}, max_diff={mlp_diff:.6f}, tt_norm={mlp_cpu.norm():.4f}, hf_norm={hf_mlp_out.norm():.4f}")

    # Final residual
    layer_out = ttnn.add(after_attn, mlp_out)
    layer_out_cpu = ttnn.to_torch(layer_out).float()
    layer_cos = F.cosine_similarity(hf_layer0_out.flatten().unsqueeze(0), layer_out_cpu.flatten().unsqueeze(0))
    print(f"  Layer 0 output: cosine={layer_cos.item():.6f}, tt_norm={layer_out_cpu.norm():.4f}, hf_norm={hf_layer0_out.norm():.4f}")

    # Test MLP with bfloat16 weights (not bfloat8_b)
    print("\n=== Testing MLP with bfloat16 weights ===")
    from models.demos.qwen36_27b.tt.mlp import TtMLP
    sd2 = load_state_dict(tt_config, max_layers=4, model_path=SNAP)
    mlp_bf16 = TtMLP(device, sd2, layer_idx=0, config=tt_config, weights_dtype=ttnn.bfloat16)
    mlp_bf16_out = mlp_bf16(post_normed)
    mlp_bf16_cpu = ttnn.to_torch(mlp_bf16_out).float()
    mlp_bf16_cos = F.cosine_similarity(hf_mlp_out.flatten().unsqueeze(0), mlp_bf16_cpu.flatten().unsqueeze(0))
    mlp_bf16_diff = (hf_mlp_out.flatten() - mlp_bf16_cpu.flatten()).abs().max()
    print(f"  MLP bf16: cosine={mlp_bf16_cos.item():.6f}, max_diff={mlp_bf16_diff:.6f}")
    print(f"  MLP bf8b: cosine={mlp_cos.item():.6f}")

finally:
    ttnn.close_device(device)
