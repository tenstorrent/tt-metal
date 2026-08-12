"""Generate a Gemma-4 reference forward pass for test_gemma4_parity.

Gemma-4 landed in transformers 5, but tt-metal's env pins 4.53 and upgrading it in place
would disturb every other model, so this runs out-of-process in a throwaway venv and hands
the result over as safetensors — which, unlike a pickle, does not care that the two sides
are on different torch versions::

    python -m venv /tmp/g4ref
    /tmp/g4ref/bin/pip install transformers==5.10.1
    /tmp/g4ref/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    /tmp/g4ref/bin/python models/tt_dit/tests/encoders/gemma4/gen_gemma4_reference.py
"""

import torch
from safetensors.torch import save_file
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

OUT = "/tmp/g4ref/gemma4_reference.safetensors"
SEQ_LEN = 128

config = Gemma4TextConfig(
    vocab_size=1000,
    hidden_size=1024,
    intermediate_size=2048,
    num_hidden_layers=6,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=256,
    global_head_dim=512,
    num_global_key_value_heads=1,
    attention_k_eq_v=True,
    sliding_window=1024,
    max_position_embeddings=SEQ_LEN,
    layer_types=["sliding_attention"] * 5 + ["full_attention"],
    rope_parameters={
        "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
    },
    rms_norm_eps=1e-6,
    attention_bias=False,
    hidden_activation="gelu_pytorch_tanh",
    num_kv_shared_layers=0,
    enable_moe_block=False,
    use_double_wide_mlp=False,
    hidden_size_per_layer_input=0,
    attn_implementation="eager",
)

torch.manual_seed(0)
model = Gemma4TextModel(config).to(torch.float32).eval()

# Distinct per-layer scalars so a dropped or misplaced multiply cannot pass.
for idx, layer in enumerate(model.layers):
    layer.layer_scalar.fill_(1.0 + 0.05 * (idx + 1))

for idx, layer in enumerate(model.layers):
    attn = layer.self_attn
    print(
        f"layer {idx}: {attn.layer_type:18} head_dim={attn.head_dim:4} v_proj={'None' if attn.v_proj is None else 'yes'}"
    )

input_ids = torch.randint(0, config.vocab_size, (1, SEQ_LEN))

with torch.no_grad():
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

# The same forward in bf16 gives the device test a floor to measure against: bf16 error
# compounds with depth, so an fp32-only target is unreachable and says nothing about
# whether the port is correct.
with torch.no_grad():
    bf16_out = model.to(torch.bfloat16)(input_ids=input_ids, output_hidden_states=True, use_cache=False)
model = model.to(torch.float32)

tensors = {f"weight.{k}": v.contiguous().float() for k, v in model.state_dict().items()}
tensors["input_ids"] = input_ids.to(torch.int64)
for i, hs in enumerate(bf16_out.hidden_states):
    tensors[f"bf16.{i}"] = hs.contiguous().clone().float()
# hidden_states[i] for i < N is the *input* to layer i, and the last entry is already
# the post-norm output — the final layer's pre-norm activation is never exposed.
for i, hs in enumerate(out.hidden_states):
    tensors[f"hidden.{i}"] = hs.contiguous().clone().float()

save_file(tensors, OUT)
print(f"\nhidden_states: {len(out.hidden_states)} (last is post-norm)")
print(
    f"last_hidden_state {tuple(out.last_hidden_state.shape)} "
    f"mean={out.last_hidden_state.mean():.5f} std={out.last_hidden_state.std():.5f}"
)
print(f"wrote {OUT}")
