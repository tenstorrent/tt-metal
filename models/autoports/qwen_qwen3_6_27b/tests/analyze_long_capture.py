# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only selected long-sequence counterfactuals. Does not import TTNN.

Usage: python -m models.autoports.qwen_qwen3_6_27b.tests.analyze_long_capture \
    --capture-dir .../capture36 --output .../capture36_cpu.json
"""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open


def metrics(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return {"pcc": float(torch.corrcoef(torch.stack([a.double(), b.double()]))[0, 1]),
            "relative_rms": float((a-b).norm()/a.norm().clamp_min(1e-30)),
            "max_abs": float((a-b).abs().max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    checkpoint = Path(snapshot_download('Qwen/Qwen3.6-27B', revision='8f33b98937c6031bb49e713dee3e8e185aafd26f', local_files_only=True))
    index = json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    keys = [f'mlp.{name}_proj.weight' for name in ('gate', 'up', 'down')]
    keys += [f'linear_attn.in_proj_{name}.weight' for name in ('qkv', 'z', 'a', 'b')]
    keys += ['linear_attn.out_proj.weight', 'post_attention_layernorm.weight']
    weights = {}
    for key in keys:
        name = 'model.language_model.layers.0.' + key
        with safe_open(checkpoint/index[name], framework='pt', device='cpu') as f:
            weights[key] = f.get_tensor(name).to(torch.bfloat16)

    def mlp(x):
        gate = F.linear(x, weights['mlp.gate_proj.weight'])
        up = F.linear(x, weights['mlp.up_proj.weight'])
        return F.linear(F.silu(gate)*up, weights['mlp.down_proj.weight'])

    def norm(x):
        xf = x.float()
        return (xf*(xf.square().mean(-1, keepdim=True)+1e-6).rsqrt()
                *(1+weights['post_attention_layernorm.weight'].float())).to(x.dtype)

    reports = []
    for path in sorted(args.capture_dir.glob('layer0_step*.pt')):
        d = torch.load(path, weights_only=False, map_location='cpu')
        h, t = d['hf'], d['tt']
        x = d['input'].reshape(-1, 5120)
        def gated(core, z):
            core = core.to(torch.bfloat16)
            xf = core.float()
            unweighted = (xf*(xf.square().mean(-1, keepdim=True)+1e-6).rsqrt()).to(torch.bfloat16)
            return ((unweighted*d['hf_norm_weight'])*F.silu(z.float())).to(torch.bfloat16)
        def tail(core, z):
            g = gated(core, z)
            attn = F.linear(g.reshape(-1, 6144), weights['linear_attn.out_proj.weight'])
            residual = x+attn
            return {'gated':g, 'attention':attn, 'output':residual+mlp(norm(residual))}
        hr, tr = x+h['attention'], x+t['attention']
        exact_h_mlp = hr+mlp(h['post_attention_norm'])
        exact_t_mlp = tr+mlp(t['post_attention_norm'])
        row = {'step':d['step'], 'users':d['users'], 'artifact':str(path),
               'hf_full_tail_reconstruction': {k:metrics(h[k],v) for k,v in tail(h['core'],h['z']).items()},
               'hf_mlp_reconstruction': metrics(h['output'],exact_h_mlp),
               'hf_mlp_on_tt_input_vs_hf':metrics(h['output'],exact_t_mlp),
               'hf_mlp_on_tt_input_vs_tt':metrics(t['output'],exact_t_mlp),
               'cpu_hf_gated_on_tt_input_vs_tt':metrics(t['gated'],gated(t['core'],t['z'])),
               'projections':{}, 'counterfactuals':{}, 'top_heads':[]}
        for key in ('qkv','z','a','b'):
            w = weights[f'linear_attn.in_proj_{key}.weight']
            h_proj, t_proj = F.linear(h['input_norm'],w), F.linear(t['input_norm'],w)
            actual = t['z'].reshape(t_proj.shape) if key=='z' else t['projection_'+key]
            target = h['z'].reshape(h_proj.shape) if key=='z' else h['projection_'+key]
            row['projections'][key] = {'hf_reconstruction':metrics(target,h_proj),
                'hidden_norm_sensitivity':metrics(h_proj,t_proj), 'tt_vs_sameinput_exact_weight':metrics(t_proj,actual),
                'total_tt_vs_hf':metrics(target,actual)}
        cores = {'captured_tt_core':t['core']} | {name:value['core'] for name,value in d['oracles'].items()}
        for name, core in cores.items():
            row['counterfactuals'][name] = {'core':metrics(h['core'],core),
                'hf_tail_tt_z': {k:metrics(h[k],v) for k,v in tail(core,t['z']).items()},
                'hf_tail_hf_z': {k:metrics(h[k],v) for k,v in tail(core,h['z']).items()}}
        err=(h['gated'].float()-t['gated'].float()).square().sum(-1)
        state_err=(h['initial_state'].float()-t['initial_state'].float()).square().sum((-1,-2))
        for user in range(err.shape[0]):
            for head in err[user].argsort(descending=True)[:6].tolist():
                hs,ts=h['initial_state'][user,head].float(),t['initial_state'][user,head].float()
                row['top_heads'].append({'user':d['users'][user],'head':head,
                    'gated_squared_error_fraction':float(err[user,head]/err.sum()),
                    'state_squared_error_fraction':float(state_err[user,head]/state_err.sum()),
                    'state':metrics(hs,ts),'state_gain':float((hs*ts).sum()/hs.square().sum()),
                    'hf_core_rms':float(h['core'][user,head].float().square().mean().sqrt()),
                    'core':metrics(h['core'][user,head],t['core'][user,head])})
        reports.append(row)
        print('step', d['step'], 'top heads', [(r['head'],r['gated_squared_error_fraction']) for r in row['top_heads']], flush=True)
        print('counterfactual output PCC', {k:v['hf_tail_tt_z']['output']['pcc'] for k,v in row['counterfactuals'].items()}, flush=True)
    args.output.write_text(json.dumps({'cpu_only':True,'reports':reports},indent=2)+'\n')

if __name__ == '__main__':
    main()
