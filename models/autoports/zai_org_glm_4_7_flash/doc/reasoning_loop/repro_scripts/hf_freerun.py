"""Reference (HF bf16 CPU) generation of doc0 from the prompt: greedy, then sampled trials.
Answers: does the reference model itself loop on this prompt?"""
import json, sys, time, torch, collections
torch.set_num_threads(14)
from transformers import AutoTokenizer, AutoModelForCausalLM
SNAP = "/home/stisi/.cache/huggingface/hub/models--zai-org--GLM-4.7-Flash/snapshots/7dd20894a642a0aa287e9827cb1a1f7f91386b67"
MAX_NEW = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
N_SAMPLED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/hf_freerun.json"
prompt = ("Write a 300+ word summary of the wikipedia page "
          "\"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". "
          "Do not use any commas and highlight at least 3 sections that has titles "
          "in markdown format, for example *highlighted section part 1*, "
          "*highlighted section part 2*, *highlighted section part 3*.")
tok = AutoTokenizer.from_pretrained(SNAP, local_files_only=True)
ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True)
if hasattr(ids, "keys"): ids = ids["input_ids"]
inp = torch.tensor([[int(i) for i in ids]])
model = AutoModelForCausalLM.from_pretrained(SNAP, local_files_only=True, dtype=torch.bfloat16).eval()
end_think = tok.convert_tokens_to_ids("</think>")
eos = model.config.eos_token_id if isinstance(model.config.eos_token_id, list) else [model.config.eos_token_id]
print(f"prompt tokens={inp.shape[1]}  </think> id={end_think}  eos={eos}", flush=True)

def loopiness(text, win=120):
    c = collections.Counter(text[i:i+win] for i in range(0, max(0, len(text)-win), 7))
    return c.most_common(1)[0][1] if c else 0

def run(tag, **gen):
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(inp, attention_mask=torch.ones_like(inp), max_new_tokens=MAX_NEW,
                             pad_token_id=model.config.pad_token_id, **gen)
    g = out[0, inp.shape[1]:].tolist()
    text = tok.decode(g, skip_special_tokens=False)
    closed = end_think in g
    ended = any(e in g for e in eos)
    rep = loopiness(text)
    print(f"[{tag}] {len(g)} tokens in {time.time()-t0:.0f}s ({len(g)/(time.time()-t0):.1f} tok/s) "
          f"</think>={'YES@'+str(g.index(end_think)) if closed else 'no'} eos={'yes' if ended else 'no'} "
          f"max-repeat-of-120char-window={rep}x", flush=True)
    if not closed:
        print(f"   tail: {text[-300:]!r}", flush=True)
    else:
        ans = text[text.find('</think>')+8:]
        print(f"   answer head: {ans[:200]!r}", flush=True)
    return {"tag": tag, "tokens": len(g), "closed_think": closed, "eos": ended, "max_repeat": rep, "text": text}

results = [run("greedy", do_sample=False, num_beams=1)]
for i in range(N_SAMPLED):
    torch.manual_seed(1000 + i)
    results.append(run(f"sampled_t1.0_p0.95_#{i}", do_sample=True, temperature=1.0, top_p=0.95))
json.dump(results, open(OUT, "w"))
print("done", flush=True)
