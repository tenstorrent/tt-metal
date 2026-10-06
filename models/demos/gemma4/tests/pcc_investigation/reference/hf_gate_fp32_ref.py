# Hugging Face Gemma 4 with fp32 arithmetic (CPU) on the accuracy gate's tokens (prompt + forced tokens of
# test_optimizer_gemma4_pcc.py): the fp32 reference chip/attn_sweep.py scores the gate positions against.
# Weights stay stored in bf16 (the checkpoint's own values) and are upcast at use, so every operation runs in fp32
# with half the memory of a float32 load (validated against the true fp32 model at PCC 0.99989 on the book text).
import os as _os
from pathlib import Path as _Path

DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import torch
import torch.nn as nn
import transformers.models.gemma4.modeling_gemma4 as G
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.demos.gemma4.tests import test_optimizer_gemma4_pcc as GATE

torch.set_num_threads(8)


def router_fwd(self, hidden_states):  # = HF Gemma4TextRouter.forward, fp32
    h = self.norm(hidden_states.float())
    h = h * self.scale.float() * self.scalar_root_size
    probs = torch.softmax(nn.functional.linear(h, self.proj.weight.float()), dim=-1)
    w, idx = torch.topk(probs, k=self.config.top_k_experts, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    w = w * self.per_expert_scale.float()[idx]
    return probs, w, idx


def experts_fwd(self, hidden_states, top_k_index, top_k_weights):  # = HF expert loop, fp32
    hidden_states = hidden_states.float()
    final = torch.zeros_like(hidden_states)
    mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
    for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
        e = e[0]
        pos, tok = torch.where(mask[e])
        gate, up = nn.functional.linear(hidden_states[tok], self.gate_up_proj[e].float()).chunk(2, dim=-1)
        h = nn.functional.linear(self.act_fn(gate) * up, self.down_proj[e].float())
        final.index_add_(0, tok, h * top_k_weights[tok, pos, None].float())
    return final


G.Gemma4TextRouter.forward = router_fwd
G.Gemma4TextExperts.forward = experts_fwd

P = f"{MODELS}/gemma-4-26B-A4B-it"
tokens = GATE.encode_text(AutoTokenizer.from_pretrained(P, local_files_only=True))[: GATE.PROMPT_TOKENS + GATE.FORCED_TOKENS]
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).eval()


def _lin(self, x):
    return nn.functional.linear(x.float(), self.weight.float(), None if self.bias is None else self.bias.float())


for mod in m.modules():
    if isinstance(mod, nn.Linear):
        mod.forward = _lin.__get__(mod)
emb = m.model.language_model.embed_tokens
emb.forward = (lambda self, x: nn.functional.embedding(x, self.weight).float() * float(self.scalar_embed_scale)).__get__(emb)
with torch.no_grad():
    logits = m(torch.tensor([tokens])).logits[0].float()
torch.save(logits, f"{DATA}/gate_hf_fp32_logits.pt")
print("GATE_FP32 saved", tuple(logits.shape), "tokens", len(tokens), flush=True)
