"""Pinned HF continuation from the actual TT branch before the disputed word."""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..tt.weights import HF_MODEL, HF_REVISION
from .generate_full_reference import load_reference


def main():
    root = Path("models/autoports/google_gemma_4_31b_it/doc/full_model")
    hf = next(r for r in json.loads((root / "qualitative_hf.json").read_text()) if r["id"] == "shared_3")
    tt = next(r for r in json.loads((root / "qualitative_tt.json").read_text()) if r["id"] == "shared_3")
    tok = AutoTokenizer.from_pretrained(HF_MODEL, revision=HF_REVISION)
    generated = tt["generated_tokens"]
    index = next(i for i in range(len(generated)) if "closed" in tok.decode(generated[: i + 1]))
    prefix = hf["prompt_tokens"] + generated[:index]
    model = load_reference()
    ids = torch.tensor([prefix])
    with torch.inference_mode():
        logits = model(ids, logits_to_keep=1).logits[0, -1].float()
        values, tokens = logits.topk(10)
        result = model.generate(
            ids, max_new_tokens=32, do_sample=False, attention_mask=torch.ones_like(ids), pad_token_id=tok.pad_token_id
        )
    output = result[0, len(prefix) :].tolist()
    report = dict(
        hf_model=HF_MODEL,
        hf_revision=HF_REVISION,
        prompt_id="shared_3",
        tt_prefix_tokens=generated[:index],
        tt_prefix=tok.decode(generated[:index]),
        disputed_token_index=index,
        tt_token=generated[index],
        tt_token_text=tok.decode([generated[index]]),
        hf_rank_of_tt_token=int((logits > logits[generated[index]]).sum()) + 1,
        top10=[dict(token=int(t), text=tok.decode([int(t)]), logit=float(v)) for t, v in zip(tokens, values)],
        hf_continuation_tokens=output,
        hf_continuation=tok.decode(output),
        scope="Exact TT-prefix conditioned full pinned HF CPU control; greedy,32-token cap",
    )
    (root / "thermodynamics_control.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
