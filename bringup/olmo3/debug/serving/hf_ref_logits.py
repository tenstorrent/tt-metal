"""HF CPU reference logits for the positions decoded in the teacher-forced device run."""
import json, sys, torch, time
from transformers import AutoModelForCausalLM

cfg = json.load(open(sys.argv[1]))
ids = cfg["ids"]
N = cfg["prefill_len"]
steps = int(sys.argv[3])
m = AutoModelForCausalLM.from_pretrained(
    "/home/ubuntu/models/olmo31-32b-think", dtype=torch.bfloat16, attn_implementation="sdpa"
)
m.eval()
t = time.time()
with torch.no_grad():
    lg = m(torch.tensor([ids[: N + steps]]), use_cache=False).logits[0]
print("hf forward secs", round(time.time() - t))
# row 0 predicts ids[N] (prefill), row i predicts ids[N+i]
torch.save(lg[N - 1 : N + steps].to(torch.float16).clone(), sys.argv[2])
print("saved", lg[N - 1 : N + steps].shape)
