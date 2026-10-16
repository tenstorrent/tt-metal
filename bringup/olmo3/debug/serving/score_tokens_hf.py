"""Teacher-force the tokens the device generated through the HF CPU model; report how plausible they are per window."""
import json, glob, sys, torch, time, math
from transformers import AutoModelForCausalLM

f = sorted(glob.glob(sys.argv[1]))[-1]
d = json.load(open(f))
ids = d["prompts"][0] + d["generated"][0]
P = len(d["prompts"][0])
n = int(sys.argv[2]) if len(sys.argv) > 2 else len(ids)
ids = ids[:n]
m = AutoModelForCausalLM.from_pretrained(
    "/home/ubuntu/models/olmo31-32b-think", torch_dtype=torch.bfloat16, attn_implementation="sdpa"
)
m.eval()
t = time.time()
with torch.no_grad():
    lg = m(torch.tensor([ids]), use_cache=False).logits[0].float()
print("forward secs", round(time.time() - t))
lp = torch.log_softmax(lg[:-1], -1).gather(1, torch.tensor(ids[1:])[:, None])[:, 0]  # logprob of token i+1 given prefix
res = []
W = 256
for s in range(P, len(ids) - 1, W):
    w = lp[s - 1 : s - 1 + W]
    res.append((s - P, float(w.mean()), float((w < math.log(1e-4)).float().mean())))
    print(
        f"gen pos {s-P:5d}-{s-P+W:5d} (abs {s}-{s+W}): mean logprob {w.mean():.2f}  frac(p<1e-4) {float((w<math.log(1e-4)).float().mean()):.3f}",
        flush=True,
    )
json.dump(res, open("/home/ubuntu/ssinghal/olmo3-logs/score_tokens_hf.json", "w"))
