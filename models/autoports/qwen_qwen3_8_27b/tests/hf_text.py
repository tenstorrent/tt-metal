# SPDX-License-Identifier: Apache-2.0
"""HF text-only control loader for the exact multimodal checkpoint."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoConfig,AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM,Qwen3_5TextRotaryEmbedding

def load_model(checkpoint='.cache/qwen38-full'):
    root=Path(checkpoint)
    cfg=AutoConfig.from_pretrained(root).text_config
    cfg._attn_implementation='sdpa'
    with torch.device('meta'): model=Qwen3_5ForCausalLM(cfg)
    index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    keys={k:k.replace('model.language_model.','model.') for k in index if k.startswith('model.language_model.') or k=='lm_head.weight'}
    for filename in sorted({index[k] for k in keys}):
        with safe_open(root/filename,framework='pt') as f:
            model.load_state_dict({keys[k]:f.get_tensor(k) for k in keys if index[k]==filename},strict=False,assign=True)
    if any(v.is_meta for v in model.parameters()): raise RuntimeError('Missing HF text parameter')
    model.model.rotary_emb=Qwen3_5TextRotaryEmbedding(cfg)
    return model.eval(),AutoTokenizer.from_pretrained(root)
