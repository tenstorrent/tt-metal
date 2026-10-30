# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generate the HF golden reference for the e2e text-generation gate (Source A).

Greedy-decodes N tokens with NemotronHForCausalLM and saves, per step, the
next-token logits (fp32) and the chosen token ids, plus the tokenized prompt.
Cached to _captured/_e2e_golden/ so the on-device e2e test does not have to
re-run HF generation every iteration.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt._hf_compat import install_hf_compat

install_hf_compat()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

HF_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
PROMPT = os.environ.get("TT_E2E_PROMPT", "The capital of France is")
N = int(os.environ.get("TT_E2E_N", "5"))

OUT = Path(__file__).resolve().parents[2] / "_captured" / "_e2e_golden"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()

    enc = tok(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"]
    print(f"[golden] prompt={PROMPT!r} input_ids={input_ids.tolist()}", flush=True)

    with torch.no_grad():
        gen = model.generate(
            input_ids,
            max_new_tokens=N,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else 0,
        )

    seq = gen.sequences  # (1, prompt_len + N)
    new_ids = seq[0, input_ids.shape[1] :].tolist()
    # gen.scores: tuple length N, each (1, vocab) -- the pre-softmax logits used
    # to pick each new token (greedy argmax).
    per_step_logits = torch.stack([s[0].float() for s in gen.scores], dim=0)  # (N, vocab)

    torch.save(input_ids, OUT / "input_ids.pt")
    torch.save(torch.tensor(new_ids, dtype=torch.long), OUT / "golden_new_ids.pt")
    torch.save(per_step_logits, OUT / "golden_step_logits.pt")
    (OUT / "meta.txt").write_text(f"prompt={PROMPT!r}\nN={N}\nnew_ids={new_ids}\ndecoded={tok.decode(new_ids)!r}\n")
    print(f"[golden] new_ids={new_ids} decoded={tok.decode(new_ids)!r}", flush=True)
    print(f"[golden] saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
