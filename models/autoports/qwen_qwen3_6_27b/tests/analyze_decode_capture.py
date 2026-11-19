# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only reproduction of capture_chunk MLP attribution; no TTNN import.

Requires capture_chunk/layer0_step{19,58,99}.pt and the pinned local checkpoint.
Writes doc/optimized_decoder/capture_chunk_cpu_attribution.json.
"""
import torch,json
import torch.nn.functional as F
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors import safe_open
root=Path('models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder')
torch.set_num_threads(8)
checkpoint=Path(snapshot_download('Qwen/Qwen3.6-27B',revision='8f33b98937c6031bb49e713dee3e8e185aafd26f',local_files_only=True))
index=json.load(open(checkpoint/'model.safetensors.index.json'))['weight_map']
weights={}
for key in ['gate','up','down']:
 name=f'model.language_model.layers.0.mlp.{key}_proj.weight'
 with safe_open(checkpoint/index[name],framework='pt',device='cpu') as f:weights[key]=f.get_tensor(name).to(torch.bfloat16)
def mlp(x,fuse=False):
 gate,up=F.linear(x,weights['gate']),F.linear(x,weights['up'])
 act=(F.silu(gate.float())*up.float()).to(torch.bfloat16) if fuse else F.silu(gate)*up
 return F.linear(act,weights['down'])
def met(a,b):
 a,b=a.float().flatten(),b.float().flatten()
 return {'pcc':float(torch.corrcoef(torch.stack([a.double(),b.double()]))[0,1]),'relative_rms':float((a-b).norm()/a.norm()),'maxabs':float((a-b).abs().max())}
report=[]
for step in [19,58,99]:
 d=torch.load(root/'capture_chunk'/f'layer0_step{step}.pt',weights_only=False);h,t=d['hf'],d['tt']
 hr=(d['input'].reshape(1,5120)+h['attention']);tr=(d['input'].reshape(1,5120)+t['attention'])
 hm,tm=mlp(h['post_attention_norm']),mlp(t['post_attention_norm'])
 hy,ty=hr+hm,tr+tm
 fty=tr+mlp(t['post_attention_norm'],True)
 row={'step':step,'cpuHF_reconstruction_vs_HF_output':met(h['output'],hy),'HF_MLP_input_sensitivity':met(hm,tm),'TT_norm_HF_MLP_vs_HF_output':met(h['output'],ty),'TT_norm_HF_MLP_vs_TT_output':met(t['output'],ty),'fused_silu_HF_MLP_vs_TT_output':met(t['output'],fty),'fused_silu_HF_MLP_vs_HF_output':met(h['output'],fty)}
 # Same captured core/z, CPU gated RMSNorm: isolates TT post-core arithmetic.
 def gdn(core,z):
  core=core.to(torch.bfloat16);nf=core.float()*(core.float().square().mean(-1,keepdim=True)+1e-6).rsqrt()
  return ((nf.to(torch.bfloat16)*d['hf_norm_weight'])*F.silu(z.float())).to(torch.bfloat16)
 row['CPU_HF_gdn_on_TT_inputs_vs_TT_gated']=met(t['gated'],gdn(t['norm_input'],t['z']))
 row['CPU_HF_gdn_on_TT_inputs_vs_HF_gated']=met(h['gated'],gdn(t['norm_input'],t['z']))
 report.append(row)
print(json.dumps(report,indent=2))
(root/'capture_chunk_cpu_attribution.json').write_text(json.dumps(report,indent=2)+'\n')
