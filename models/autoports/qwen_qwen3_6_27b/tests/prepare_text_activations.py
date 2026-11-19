# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only real prompt activations for decoder precision evidence (no generation)."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding
from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import MODEL, REVISION, load_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--length', type=int, default=512)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    root = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    texts = [
        'Explain how a computer stores a list of numbers and sorts it. Give an example and check each step.',
        'A train travels 120 kilometers in 90 minutes. Calculate its average speed and explain the units.',
        'Compare renewable energy sources, including solar, wind, and hydroelectric power, in plain language.',
        'Write a Python function that counts words in a string. Explain the treatment of punctuation.',
    ]
    ids = []
    for row in range(args.batch):
        tokens = tokenizer.encode(texts[row % len(texts)], add_special_tokens=True)
        ids.append((tokens * ((args.length + 2) // len(tokens) + 1))[:args.length + 2])
    ids = torch.tensor(ids)
    mapping = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    name = 'model.language_model.embed_tokens.weight'
    with safe_open(root / mapping[name], framework='pt', device='cpu') as f:
        slices = f.get_slice(name)
        unique, inverse = torch.unique(ids, return_inverse=True)
        selected = torch.cat([slices[int(i):int(i)+1] for i in unique])
        hidden = selected[inverse].reshape(args.batch,args.length+2,-1).to(torch.bfloat16)
    c = AutoConfig.from_pretrained(root, local_files_only=True).text_config
    c._attn_implementation = 'sdpa'
    rot = Qwen3_5TextRotaryEmbedding(c)
    cache = DynamicCache(config=c)
    args.output.mkdir(parents=True,exist_ok=True)
    metadata = {'model':MODEL,'revision':REVISION,'batch':args.batch,'length':args.length,'prompts':texts,'token_ids':ids.tolist(),'generation':False,'source':'checkpoint token embeddings and preceding HF decoder layers; CPU only'}
    with torch.no_grad():
        for layer_idx in range(4):
            if layer_idx in (0,3):
                path=args.output/f'text_l{layer_idx}_b{args.batch}_s{args.length}.pt'
                torch.save({'x':hidden[:,:args.length].contiguous(),'d':hidden[:,args.length:args.length+1].contiguous(),'d2':hidden[:,args.length+1:].contiguous()},path)
                (path.with_suffix('.json')).write_text(json.dumps(metadata|{'layer':layer_idx,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2)+'\n')
                print(path,flush=True)
            if layer_idx ==3:break
            ref=Qwen3_5DecoderLayer(c,layer_idx).eval().to(torch.bfloat16)
            ref.load_state_dict(load_layer(layer_idx))
            hidden=ref(hidden,position_embeddings=rot(hidden,torch.arange(args.length+2).expand(args.batch,-1)),past_key_values=cache)
            del ref

if __name__=='__main__': main()
