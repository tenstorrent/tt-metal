import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM

tag, path = sys.argv[1], sys.argv[2]
torch.set_num_threads(12)
K = "/home/ttuser/atupe/qwen38_work/runs/T3/kl/"
toks = torch.load(K + "hf38_ref2048.refpt")["reference_tokens"][:, :1024]
cfg = AutoConfig.from_pretrained(path)
cfg.rope_scaling = {"factor": 4.0, "original_max_position_embeddings": 32768, "type": "yarn"}
model = AutoModelForCausalLM.from_pretrained(path, config=cfg, torch_dtype=torch.float32)
model.eval()
with torch.no_grad():
    lg = model(toks).logits[0, 511:1023].float()
P = torch.log_softmax(lg, -1)
ref = torch.load(K + f"hf{tag}_logp.pt")
Q = torch.log_softmax(ref["logp"].float(), -1)  # bf16 HF reference
kl = (P.exp() * (P - Q)).sum(-1)
am = P.argmax(-1)
print(
    f'HF{tag} fp32 vs HF{tag} bf16-ref: argmax agree {(am==ref["argmax"]).float().mean()*100:.2f}%  KL(fp32||bf16) mean {kl.mean():.4f} med {kl.median():.4f} p99 {kl.quantile(.99):.4f}'
)
for b in range(4):
    s = slice(b * 128, (b + 1) * 128)
    print(" block", b * 128, "dis", int((am[s] != ref["argmax"][s]).sum()), "meanKL", float(kl[s].mean()))
torch.save({"argmax": am, "logp": P.half()}, f"/home/ttuser/atupe/qwen38_work/runs/T3/bisect/hf{tag}_fp32_logp.pt")
