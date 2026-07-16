"""#48291 bf16-floor across 5 seeds (CPU-only, HF fp32 vs bf16, zero TT).

Same HF model in fp32 vs bf16, identical seeded 8-step injected noise, canonical prompt.
Canvas + noise are generated deterministically from the seed (matches replay_hf_tt main()),
so no saved artifact is needed. Reproduces seed0=0.863 / seed1=0.914 as a validation.
"""
import gc, os, sys, time
import torch

torch.set_num_threads(os.cpu_count() or 1)
sys.path.insert(0, "/home/zni/tt-metal")
CKPT = "/home/zni/dg_models/diffusiongemma-26B-A4B-it"
PROMPT = "Explain what a diffusion language model is in one sentence."
SEEDS = [0, 1, 2, 3, 4]
STEPS = 8  # matches the existing 8-step gate table

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.demo.replay_hf_tt import (
    _hf_text_vocab_size,
    _load_hf_model,
    _make_replay_noise,
    _run_hf_reference,
)

cfg = DiffusionConfig(canvas_length=256, max_denoise_steps=STEPS)


def run_all_seeds(dtype, tok, vocab):
    _, model = _load_hf_model(CKPT, local_files_only=True, dtype=dtype)
    out = {}
    for seed in SEEDS:
        g = torch.Generator(device="cpu").manual_seed(seed)
        canvas = torch.randint(0, vocab, (1, cfg.canvas_length), dtype=torch.long, generator=g)
        gumbel, renoise = _make_replay_noise(
            seed=seed, steps=STEPS, canvas_length=cfg.canvas_length, vocab_size=vocab, mode="seeded"
        )
        t0 = time.time()
        traj = _run_hf_reference(model, tok, PROMPT, canvas, cfg, gumbel_noise=gumbel, renoise_tokens=renoise)[1]
        out[seed] = traj
        print(f"[{dtype} seed {seed}] {time.time()-t0:.0f}s", flush=True)
    del model
    gc.collect()
    return out


from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(CKPT, local_files_only=True, trust_remote_code=True)
# vocab from a quick bf16 build reused for both dtypes' noise
_, m0 = _load_hf_model(CKPT, local_files_only=True, dtype=torch.bfloat16)
vocab = _hf_text_vocab_size(m0, tok)
del m0
gc.collect()

fp32 = run_all_seeds(torch.float32, tok, vocab)
bf16 = run_all_seeds(torch.bfloat16, tok, vocab)

print("\n===== HF-fp32 vs HF-bf16 committed_match (5 seeds, 8-step) =====")
print(f"{'seed':>4} {'committed':>10} {'ndiff':>6} {'argmax_step0':>12}")
vals = []
for seed in SEEDS:
    a, b = fp32[seed].committed.flatten(), bf16[seed].committed.flatten()
    cm = (a == b).float().mean().item()
    vals.append(cm)
    n = int((a != b).sum())
    arg0 = (fp32[seed].per_step[0].argmax == bf16[seed].per_step[0].argmax).float().mean().item()
    print(f"{seed:>4} {cm:>10.4f} {n:>6} {arg0:>12.4f}")
import statistics

print(f"\nmean={statistics.fmean(vals):.4f} min={min(vals):.4f} max={max(vals):.4f}  (bar=0.95)")
print("all below 0.95?", all(v < 0.95 for v in vals))
torch.save({"fp32": fp32, "bf16": bf16, "seeds": SEEDS, "steps": STEPS}, "/tmp/dg48291_floor_5seed.pt")
print("[done]", flush=True)
