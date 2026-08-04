# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Dump the MiniMax-H3 text-conditioning reference on CPU, for the TT encoder's PCC gate.

Runs the *full 64-layer* HF Qwen3-VL and takes `hidden_states[50]`, which is what H3 conditions
on -- a stack truncated to exactly 50 layers would give the post-norm output instead, which is a
different tensor, and the diffusers reference raises rather than let that happen. The TT encoder
is built at 50 layers with a raw tap, so it must match *this*.

Also captures the reference's own rotary cos/sin via a forward hook, so the claim that
text-only mRoPE collapses to plain 1-D rope (making `mrope_interleaved: true` indistinguishable
from the chunked split tt_dit implements) is gated against HF rather than assumed.

Writes one .pt holding everything. CPU-only, no ttnn import.
"""

import json
import os
import sys
import time

import torch

CKPT = os.environ.get("MINIMAX_H3_DIFFUSERS_DIR", "/data/cglagovich/MiniMax-H3-diffusers")
OUT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(os.environ.get("TT_DIT_CACHE_DIR", os.path.expanduser("~/.cache/tt-dit")), "h3_text_golden.pt")
)
TAP = 50  # MINIMAX_H3_TEXT_ENCODER_LAYER

# The prompt the pipeline actually generates from, plus a long one at the ~512-token length the
# 768P working point is costed at (`test_transformer_block_perf_minimax_h3.py::NUM_TEXT_TOKENS`).
# Short prompts are not gated: a 50-layer causal stack accumulates over the context, so a 20-token
# measurement does not stand in for a 500-token one.
E2E_PROMPT = (
    "A red fox trots across a snowy field at dawn, its breath visible in the cold air. "
    "The low sun throws long blue shadows behind it, and loose snow lifts from each footfall."
)

_SCENE = (
    "The camera drifts slowly along a rain-slick city street at night, neon signage reflecting in "
    "the standing water. A tram passes left to right, its windows warm and crowded, and the wires "
    "overhead sway briefly in its wake. Steam rises from a grate near the curb. A vendor under a "
    "striped awning turns skewers over a charcoal brazier, sparks lifting and dying in the damp "
    "air, while two people share an umbrella and step around a puddle without breaking their "
    "conversation. Somewhere off frame a saxophone plays, unhurried and slightly flat. The tram "
    "bell sounds once as it recedes, and the street noise settles back into rain on awnings, tyres "
    "on wet asphalt, and the low hum of a transformer on the corner pole. Reflections stretch and "
    "compress as the camera continues its slow lateral move, holding the vendor's brazier in the "
    "near field while the tram's tail lights shrink into the mist at the end of the block. "
)


def _long_prompt(tokenizer, target: int) -> str:
    """A natural-language prompt of about `target` tokens, built by extending one scene.

    Length is what is being controlled here, not content: the point is to gate the conditioner at
    the context length the working point is costed at rather than at a sentence.
    """
    text = _SCENE
    while len(tokenizer(text, add_special_tokens=False)["input_ids"]) < target:
        text += _SCENE
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:target]
    return tokenizer.decode(ids)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    cfg = json.loads(open(f"{CKPT}/text_encoder/config.json").read())
    n_layers = cfg["text_config"]["num_hidden_layers"]
    log(f"checkpoint {CKPT}, text layers={n_layers}, tap={TAP}")
    assert n_layers > TAP, f"tap {TAP} needs more than {TAP} layers, have {n_layers}"

    tokenizer = AutoTokenizer.from_pretrained(CKPT, subfolder="tokenizer")
    prompts = [E2E_PROMPT, _long_prompt(tokenizer, 512)]
    for prompt in prompts:
        log(f"prompt: {len(tokenizer(prompt, add_special_tokens=False)['input_ids'])} tokens")

    log("loading text_encoder on CPU in bfloat16 (~63 GB) ...")
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        CKPT, subfolder="text_encoder", dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    log(f"loaded in {time.time() - t0:.1f} s")

    # Capture the reference's rotary tables. Found by class name rather than by attribute path,
    # which moves between transformers versions.
    captured = {}

    def hook(module, args, output):
        if isinstance(output, tuple) and len(output) == 2 and "rope" not in captured:
            captured["rope"] = tuple(o.detach().float().clone() for o in output)

    handles = []
    for name, module in model.named_modules():
        if "rotary" in type(module).__name__.lower() or name.endswith("rotary_emb"):
            handles.append(module.register_forward_hook(hook))
            log(f"hooked {name} ({type(module).__name__})")

    records = []
    for prompt in prompts:
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        n = input_ids.shape[1]

        kwargs = dict(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            output_hidden_states=True,
        )
        # Text-only, so the token type ids are all zeros; pass them the way the reference does
        # if this transformers version takes them, and without if it does not.
        mm = torch.zeros_like(input_ids)
        captured.pop("rope", None)
        log(f"forward: {n} tokens -- {prompt[:44]!r}")
        t0 = time.time()
        with torch.no_grad():
            try:
                out = model.model(mm_token_type_ids=mm, **kwargs)
            except TypeError as e:
                log(f"  mm_token_type_ids rejected ({e}); retrying without")
                out = model.model(**kwargs)
        dt = time.time() - t0

        hs = out.hidden_states
        tap = hs[TAP].detach().float().squeeze(0)  # [n, 5120]
        final = hs[-1].detach().float().squeeze(0)
        rec = {
            "prompt": prompt,
            "token_ids": token_ids,
            "num_tokens": n,
            "tap": tap,
            "num_hidden_states": len(hs),
            # The post-norm final state, kept only to prove the tap is *not* it -- the
            # distinction the diffusers guard exists to protect.
            "final_norm_state": final,
            "rope": captured.get("rope"),
        }
        records.append(rec)
        log(
            f"  {dt:.1f} s, hidden_states={len(hs)}, tap {tuple(tap.shape)} "
            f"mean={tap.mean():.5f} std={tap.std():.5f}, "
            f"tap-vs-final maxdiff={(tap - final).abs().max():.4f}"
        )
        if rec["rope"] is not None:
            log(f"  rope cos {tuple(rec['rope'][0].shape)} sin {tuple(rec['rope'][1].shape)}")

    for h in handles:
        h.remove()

    payload = {
        "checkpoint": CKPT,
        "tap_index": TAP,
        "num_text_layers": n_layers,
        "text_config": cfg["text_config"],
        "transformers_version": __import__("transformers").__version__,
        "records": records,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(payload, OUT)
    log(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
