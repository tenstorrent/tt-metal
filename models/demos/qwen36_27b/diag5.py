"""Layer-by-layer comparison of HF vs TT hidden states."""
import torch
import os, glob

SNAP = "/home/yito/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Load HF model with 4 layers
config = AutoConfig.from_pretrained(SNAP, trust_remote_code=True)
config.text_config.num_hidden_layers = 4
hf_model = AutoModelForCausalLM.from_pretrained(
    SNAP, config=config, torch_dtype=torch.bfloat16,
    trust_remote_code=True, device_map="cpu"
)
hf_model.eval()
tokenizer = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)

# Hook into HF model to capture intermediate states
hf_hidden = {}
def hook_fn(name):
    def fn(module, input, output):
        if isinstance(output, tuple):
            hf_hidden[name] = output[0].detach().float()
        else:
            hf_hidden[name] = output.detach().float()
    return fn

lm = hf_model.language_model if hasattr(hf_model, 'language_model') else hf_model.model
# Hook embedding
lm.embed_tokens.register_forward_hook(hook_fn("embed"))
# Hook each layer
for i, layer in enumerate(lm.layers):
    layer.register_forward_hook(hook_fn(f"layer_{i}"))
lm.norm.register_forward_hook(hook_fn("final_norm"))

# Run HF forward pass on a single token
input_ids = torch.tensor([[151644]])  # Single token
with torch.no_grad():
    hf_out = hf_model(input_ids)

print(f"HF embedding: shape={hf_hidden['embed'].shape}, norm={hf_hidden['embed'].norm():.6f}")
for i in range(4):
    h = hf_hidden[f'layer_{i}']
    print(f"HF after layer {i}: shape={h.shape}, norm={h.norm():.6f}, first5={h[0,0,:5]}")
print(f"HF final_norm: norm={hf_hidden['final_norm'].norm():.6f}, first5={hf_hidden['final_norm'][0,0,:5]}")

# Now TT model
import ttnn
device = ttnn.open_device(device_id=0)
try:
    from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
    from models.demos.qwen36_27b.tt.load_weights import load_state_dict
    from models.demos.qwen36_27b.tt.model import TtQwen36Model
    from models.demos.qwen36_27b.tt.generator import Qwen36Generator
    
    tt_config = Qwen36ModelConfig()
    tt_config.num_hidden_layers = 4
    state_dict = load_state_dict(tt_config, max_layers=4, model_path=SNAP)
    model = TtQwen36Model(device, state_dict, tt_config)
    del state_dict
    
    # Manually trace through TT model
    hidden_tt = model.embed(input_ids)
    h_cpu = ttnn.to_torch(hidden_tt).float()
    print(f"\nTT embedding: shape={h_cpu.shape}, norm={h_cpu.norm():.6f}")
    
    hf_e = hf_hidden['embed'].reshape_as(h_cpu)
    cos_sim = torch.nn.functional.cosine_similarity(hf_e.flatten().unsqueeze(0), h_cpu.flatten().unsqueeze(0))
    print(f"  Embed cosine sim: {cos_sim.item():.6f}")
    
    cos, sin = model.get_rope(0)
    deltanet_state = model.create_deltanet_state()
    kv_caches = {}
    
    for i, layer in enumerate(model.layers):
        layer_type = tt_config.layer_types[i]
        kv_cache = kv_caches.get(i) if layer_type == "full_attention" else None
        
        hidden_tt, new_kv = layer(
            hidden_tt, deltanet_state=deltanet_state,
            cos=cos, sin=sin, kv_cache=kv_cache, mode="decode",
        )
        if new_kv is not None:
            kv_caches[i] = new_kv
        
        h_cpu = ttnn.to_torch(hidden_tt).float()
        hf_h = hf_hidden[f'layer_{i}']
        hf_h = hf_h.reshape(1, 1, 1, -1) if hf_h.dim() == 3 else hf_h
        
        cos_sim = torch.nn.functional.cosine_similarity(
            hf_h.flatten().unsqueeze(0), h_cpu.flatten().unsqueeze(0)
        )
        max_err = (hf_h.flatten() - h_cpu.flatten()).abs().max()
        print(f"TT after layer {i}: norm={h_cpu.norm():.6f}, cosine_sim={cos_sim.item():.6f}, max_err={max_err:.6f}")
        print(f"  HF first5: {hf_h.flatten()[:5]}")
        print(f"  TT first5: {h_cpu.flatten()[:5]}")
    
    # Final norm
    hidden_normed = model.rms_norm(hidden_tt, model.final_norm_weight)
    h_normed_cpu = ttnn.to_torch(hidden_normed).float()
    hf_fn = hf_hidden['final_norm'].reshape_as(h_normed_cpu)
    cos_sim = torch.nn.functional.cosine_similarity(hf_fn.flatten().unsqueeze(0), h_normed_cpu.flatten().unsqueeze(0))
    print(f"\nFinal norm cosine sim: {cos_sim.item():.6f}")
    
finally:
    ttnn.close_device(device)
