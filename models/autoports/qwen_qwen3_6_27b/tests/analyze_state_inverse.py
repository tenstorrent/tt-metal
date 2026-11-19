# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only real-input conditioning probe; TF32 emulation is not a device simulator."""
import torch,json
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
root=Path('models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder')
torch.set_num_threads(8)
cp=Path(snapshot_download('Qwen/Qwen3.6-27B',revision='8f33b98937c6031bb49e713dee3e8e185aafd26f',local_files_only=True))
index=json.load(open(cp/'model.safetensors.index.json'))['weight_map'];prefix='model.language_model.layers.0.';weights={}
for shard in sorted({v for k,v in index.items() if k.startswith(prefix+'linear_attn.') or k==prefix+'input_layernorm.weight'}):
 with safe_open(cp/shard,framework='pt',device='cpu') as f:
  for name in f.keys():
   if name.startswith(prefix+'linear_attn.') or name==prefix+'input_layernorm.weight':weights[name[len(prefix):]]=f.get_tensor(name)
c=AutoConfig.from_pretrained(cp).text_config
attn=Qwen3_5GatedDeltaNet(c,0).eval().to(torch.bfloat16)
attn.load_state_dict({k[len('linear_attn.'):]:v for k,v in weights.items() if k.startswith('linear_attn.')})
x=torch.load(root/'text_activations/text_l0_b32_s33.pt',weights_only=True)['x'][:4]
x=(x.float()*(x.float().square().mean(-1,keepdim=True)+c.rms_norm_eps).rsqrt()*(1+weights['input_layernorm.weight'].float())).to(torch.bfloat16)
captured={};original=attn.chunk_gated_delta_rule
def hook(q,k,v,**kw):
 captured.update(q=q.clone(),k=k.clone(),v=v.clone(),g=kw['g'].clone(),beta=kw['beta'].clone())
 return original(q,k,v,**kw)
attn.chunk_gated_delta_rule=hook
with torch.no_grad():attn(x)
k=captured['k'].float();k=k*(k.square().sum(-1,keepdim=True)+1e-6).rsqrt();k=k.transpose(1,2)[:,:,:32]
beta=captured['beta'].float().transpose(1,2)[:,:,:32];g=captured['g'].transpose(1,2)[:,:,:32].float().cumsum(-1)
decay=(g.unsqueeze(-1)-g.unsqueeze(-2)).tril().exp().tril();n=(-(k*beta.unsqueeze(-1))@k.transpose(-1,-2)*decay).tril(-1)
eye=torch.eye(32)
def tf32(x):return (x.contiguous().view(torch.int32)&-8192).view(torch.float32)
def invpoly(n,truncate):
 inv=eye+n;power=n.clone();peaks=[float(power.abs().max())]
 def mm(a,b):return (tf32(a)@tf32(b)) if truncate else a@b
 for _ in range(4):
  power=mm(power,power);inv=mm(inv,eye+power);peaks.append(float(power.abs().max()))
 return inv,peaks
rows=[]
for b in range(4):
 for h in [7,12,14,20,24,44]:
  a=n[b,h];exact=torch.linalg.solve_triangular(eye.double()-a.double(),eye.double(),upper=False)
  inv,peaks=invpoly(a,False);approx,tp=invpoly(a,True)
  row={'user':b,'head':h,'beta_max':float(beta[b,h].max()),'N_max':float(a.abs().max()),'N_powers_max':peaks,'exact_inverse_max':float(exact.abs().max()),'condition_number':float(torch.linalg.cond(eye.double()-a.double())), 'fp32_inverse_relative_error':float((inv-exact).norm()/exact.norm()), 'tf32_truncated_inverse_relative_error':float((approx-exact).norm()/exact.norm()),'tf32_inverse_max':float(approx.abs().max())}
  rows.append(row)
print(json.dumps(rows,indent=2))
(root/'text_state_inverse_cpu.json').write_text(json.dumps(rows,indent=2)+'\n')
