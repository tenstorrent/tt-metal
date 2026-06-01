import torch
from transformers import AutoModelForCausalLM, AutoConfig

SNAP = "/home/yito/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
config = AutoConfig.from_pretrained(SNAP, trust_remote_code=True)
config.text_config.num_hidden_layers = 2
hf = AutoModelForCausalLM.from_pretrained(
    SNAP, config=config, torch_dtype=torch.bfloat16,
    trust_remote_code=True, device_map="cpu"
)

print("=== Model children ===")
for n, c in hf.model.named_children():
    print(f"  hf.model.{n}: {type(c).__name__}")

print(f"\nhf.model has layers: {hasattr(hf.model, 'layers')}")
print(f"hf.model has language_model: {hasattr(hf.model, 'language_model')}")

print("\n=== All modules containing 'layernorm' ===")
for n, mod in hf.named_modules():
    if "input_layernorm" in n:
        w = mod.weight
        print(f"  {n}: norm={w.float().norm():.4f} mean={w.float().mean():.4f} first3={w[:3].tolist()}")

print("\n=== State dict keys for layer 0 ===")
for k in sorted(hf.state_dict().keys()):
    if ".0." in k and "layernorm" in k:
        v = hf.state_dict()[k]
        print(f"  {k}: shape={v.shape} norm={v.float().norm():.4f}")

print("\n=== diag5.py approach ===")
lm = hf.language_model if hasattr(hf, "language_model") else hf.model
print(f"lm = hf.{'language_model' if hasattr(hf, 'language_model') else 'model'}")
print(f"type(lm) = {type(lm).__name__}")
print(f"lm has layers: {hasattr(lm, 'layers')}")
print(f"lm has embed_tokens: {hasattr(lm, 'embed_tokens')}")
if hasattr(lm, "layers"):
    l0 = lm.layers[0]
    print(f"lm.layers[0] type: {type(l0).__name__}")
    for a, c in l0.named_children():
        print(f"  l0.{a}: {type(c).__name__}")
    w = l0.input_layernorm.weight
    print(f"lm.layers[0].input_layernorm.weight: norm={w.float().norm():.4f} first3={w[:3].tolist()}")
