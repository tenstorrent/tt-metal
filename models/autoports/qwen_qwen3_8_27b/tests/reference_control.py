# SPDX-License-Identifier: Apache-2.0
"""CPU-only HF chunk-schedule control for the long linear-attention diagnosis."""
import json
import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
from safetensors.torch import load_file

torch.set_num_threads(8)
torch.manual_seed(982)
cfg = AutoConfig.from_pretrained('.cache/qwen38-weights').text_config
ref = Qwen3_5DecoderLayer(cfg, 0).to(torch.bfloat16).eval()
ref.load_state_dict({k.removeprefix('model.language_model.layers.0.'):v for k,v in load_file('.cache/qwen38-weights/layer_0.safetensors').items()})
x = torch.randn(1, 16385, 5120, dtype=torch.bfloat16) * .1

def run(boundaries):
    cache = DynamicCache(config=cfg)
    start = 0
    with torch.no_grad():
        for end in boundaries:
            y = ref.linear_attn(ref.input_layernorm(x[:, start:end]), cache_params=cache)
            start = end
    return y[:, -1:].clone(), cache.layers[0].recurrent_states.clone()

def pcc(a,b):
    return torch.corrcoef(torch.stack([a.float().flatten(),b.float().flatten()]))[0,1].item()

a,sa = run(list(range(1024,16257,1024)) + [16257,16385])
b,sb = run(list(range(128,16385,128)) + [16385])
print(json.dumps(dict(last_mixer_pcc=pcc(a,b),state_pcc=pcc(sa,sb),state_maxerr=(sa-sb).abs().max().item())))
