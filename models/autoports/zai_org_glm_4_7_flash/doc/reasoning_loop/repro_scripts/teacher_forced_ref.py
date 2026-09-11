"""Teacher-forced HF (CPU, bf16) check of the device's greedy trace for doc0.
Feeds prompt + the device's first ~N chars of reasoning to the reference model in one
forward pass and asks, at every generated position, whether the reference's argmax
equals the token the device actually produced. Then free-runs HF greedy from the
loop-entry point for a few hundred tokens to see whether the reference loops too."""
import json, sys, time, torch, collections
torch.set_num_threads(12)
from transformers import AutoTokenizer, AutoModelForCausalLM

SNAP = "/home/stisi/.cache/huggingface/hub/models--zai-org--GLM-4.7-Flash/snapshots/7dd20894a642a0aa287e9827cb1a1f7f91386b67"
dump = json.load(open(sys.argv[1]))
N_CHARS = int(sys.argv[2]) if len(sys.argv) > 2 else 3200
FREE_RUN = int(sys.argv[3]) if len(sys.argv) > 3 else 200

prompt = ("Write a 300+ word summary of the wikipedia page "
          "\"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". "
          "Do not use any commas and highlight at least 3 sections that has titles "
          "in markdown format, for example *highlighted section part 1*, "
          "*highlighted section part 2*, *highlighted section part 3*.")
reasoning = dump["choices"][0]["message"].get("reasoning") or ""
tok = AutoTokenizer.from_pretrained(SNAP, local_files_only=True)
p_ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True)
if hasattr(p_ids, "keys"): p_ids = p_ids["input_ids"]
p_ids = [int(i) for i in p_ids]
g_ids = tok(reasoning[:N_CHARS], add_special_tokens=False)["input_ids"]
ids = torch.tensor([p_ids + g_ids])
P, G = len(p_ids), len(g_ids)
print(f"prompt tokens={P} device-generated tokens fed={G} (first {N_CHARS} chars of {len(reasoning)})", flush=True)

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(SNAP, local_files_only=True, dtype=torch.bfloat16).eval()
print(f"HF model loaded on cpu bf16 in {time.time()-t0:.0f}s", flush=True)

t0 = time.time()
with torch.no_grad():
    logits = model(ids).logits[0]          # [P+G, vocab]
print(f"teacher-forced forward over {P+G} tokens: {time.time()-t0:.0f}s", flush=True)

pred = logits[P-1:P+G-1].float()          # predicts positions P..P+G-1
ref_arg = pred.argmax(-1)
dev = ids[0, P:P+G]
agree = (ref_arg == dev)
probs = torch.softmax(pred, -1)
p_dev = probs.gather(1, dev[:, None])[:, 0]
p_top = probs.max(-1).values
print(f"\nteacher-forced argmax agreement: {int(agree.sum())}/{G} = {agree.float().mean():.3f}")
for a, b in [(0,100),(100,250),(250,400),(400,550),(550,G)]:
    b = min(b, G)
    if a >= b: break
    print(f"  tokens {a:3d}-{b:3d}: agree {agree[a:b].float().mean():.3f}   "
          f"mean p_ref(device token)={p_dev[a:b].mean():.3f}  mean p_ref(top1)={p_top[a:b].mean():.3f}")
dis = (~agree).nonzero()[:, 0].tolist()
print(f"\nfirst disagreements (pos, device tok -> ref tok, p_ref(dev), p_ref(top)):")
for i in dis[:12]:
    print(f"  {i:4d}: {tok.decode([int(dev[i])])!r:>14} -> {tok.decode([int(ref_arg[i])])!r:<14} "
          f"p_dev={p_dev[i]:.3f} p_top={p_top[i]:.3f}")
# does the device token ever fall OUTSIDE the reference's top-p 0.95 nucleus?
srt = probs.sort(-1, descending=True)
cum = srt.values.cumsum(-1)
rank = (srt.indices == dev[:, None]).float().argmax(-1)
in_nucleus = torch.tensor([cum[i, max(0, int(rank[i])-1)] < 0.95 or rank[i] == 0 for i in range(G)])
print(f"\ndevice tokens outside the reference top-p=0.95 nucleus: {int((~in_nucleus).sum())}/{G}; "
      f"device tokens ranked >32 by reference: {int((rank > 32).sum())}/{G}")

# free-run the reference greedily from the loop-entry region
loop_char = reasoning.find("*He participated in the First Crusade.*")
if loop_char > 0 and FREE_RUN > 0:
    pre = tok(reasoning[:loop_char], add_special_tokens=False)["input_ids"]
    start = torch.tensor([p_ids + pre])
    print(f"\nfree-running HF greedy for {FREE_RUN} tokens from loop entry (char {loop_char}, {len(pre)} gen tokens in)...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(start, attention_mask=torch.ones_like(start), max_new_tokens=FREE_RUN,
                             do_sample=False, num_beams=1, pad_token_id=model.config.pad_token_id)
    gen = tok.decode(out[0, start.shape[1]:].tolist(), skip_special_tokens=False)
    print(f"  {time.time()-t0:.0f}s. HF continuation:\n{gen}")
    dev_cont = reasoning[loop_char:loop_char+len(gen)]
    print(f"\n  device continuation from the same point:\n{dev_cont}")
