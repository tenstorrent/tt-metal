"""Ground-truth check of the tt-metal residual taps: run the HF Qwen3.6-27B text model on CPU over the
same 128-token prompt that produced profiles/dflash2_real.npz (ids saved there) and compare
hidden_states[l+1] for l in [5,19,33,47,61] with the captured target_hidden_cat (PCC per tap layer).

Run: python_env/bin/python models/demos/blackhole/qwen36/tests/dflash2_hf_tap_check.py
"""
import os
import time

import numpy as np
import torch

CKPT = os.environ.get("HF_MODEL", "/home/ttuser/experiments/qwen36_27b/model_volume/weights/Qwen3.6-27B")
REAL = os.environ.get("DFLASH_REAL", "/home/ttuser/experiments/qwen36_27b/profiles/dflash2_real.npz")
TAPS = [int(x) for x in os.environ.get("DFLASH_TAPS", "5,19,33,47,61").split(",")]
OUT = os.environ.get("DFLASH_HF_OUT", "/home/ttuser/experiments/qwen36_27b/profiles/dflash2_hf_taps.npz")


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / (a.std() * b.std() * (a.numel() - 1)))


def main():
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    real = np.load(REAL)
    ids = torch.from_numpy(real["ids"]).long()  # (1,128)
    tt_taps = torch.from_numpy(real["target_hidden_cat"]).float()  # (1,128,25600)
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        print(f"[hf] AutoModelForCausalLM failed ({type(e).__name__}: {str(e)[:200]}); trying ImageTextToText")
        model = AutoModelForImageTextToText.from_pretrained(CKPT, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    print(f"[hf] loaded {type(model).__name__} in {time.time() - t0:.0f}s", flush=True)
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    print(f"[hf] prompt: {tok.decode(ids[0][:40])!r}...")
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    print(f"[hf] forward in {time.time() - t0:.0f}s; {len(out.hidden_states)} hidden states", flush=True)
    hs = out.hidden_states
    print(
        f"[hf] argmax at last position: {int(out.logits[0, -1].float().argmax())} (tt 'anchor' field = last prompt id {int(real['anchor'])})"
    )
    cmp = TAPS == [5, 19, 33, 47, 61]
    for i, l in enumerate(TAPS if cmp else []):
        h = hs[l + 1][0].float()  # (128,5120) output of layer l
        t = tt_taps[0, :, i * 5120 : (i + 1) * 5120]
        print(
            f"[hf] tap L{l}: pcc={pcc(h, t):.5f}  hf row-norm mean {h.norm(dim=-1).mean():.2f}  tt {t.norm(dim=-1).mean():.2f}  "
            f"max|hf-tt| {(h - t).abs().max():.3f}  rel {((h - t).norm() / h.norm()):.4f}",
            flush=True,
        )
    # Which HF index best matches each tt tap (catches an off-by-one in the layer offset)?
    for i, l in enumerate(TAPS if cmp else []):
        t = tt_taps[0, :, i * 5120 : (i + 1) * 5120]
        best = max(range(len(hs)), key=lambda j: pcc(hs[j][0].float(), t))
        print(
            f"[hf] tt tap L{l} best matches hidden_states[{best}] (expected {l + 1}) pcc={pcc(hs[best][0].float(), t):.5f}"
        )
    cat = torch.cat([hs[l + 1][0].float() for l in TAPS], dim=-1).unsqueeze(0)  # (1,S,ntaps*H) in tap order
    np.savez(
        OUT,
        **{f"hs{l + 1}": hs[l + 1][0].float().numpy() for l in TAPS},
        logits_last=out.logits[0, -1].float().numpy(),
        target_hidden_cat=cat.numpy(),
        ids=real["ids"],
        S=real["S"],
        anchor=np.int64(int(out.logits[0, -1].float().argmax())),
    )
    print(f"[hf] saved {OUT} (target_hidden_cat {tuple(cat.shape)}, anchor = HF argmax at the last prompt position)")


if __name__ == "__main__":
    main()
