"""Host (CPU) HF reference for WS6: greedy assisted decoding with google/gemma-4-26B-A4B-it-assistant.

Runs transformers' own MTP path (SinglePositionMultiTokenCandidateGenerator, K constant) on CPU and records,
per prompt, the generated tokens and the number of accepted drafts in every verify round.

  HF_HUB_OFFLINE=1 python mtp_host_reference.py <out.json> [num_prompts] [new_tokens] [K]
"""
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET, TARGET_REV = "google/gemma-4-26B-A4B-it", "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
DRAFTER, DRAFTER_REV = "google/gemma-4-26B-A4B-it-assistant", "6e5aaaf4c42b98394530b8fda2e95cadd65c151c"


def main():
    out = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    new_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 96
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    torch.set_num_threads(int(os.environ.get("THREADS", "16")))
    prompts = json.load(open(os.path.join(os.path.dirname(__file__), "mtp_prompts.json")))[:count]
    tok = AutoTokenizer.from_pretrained(TARGET, revision=TARGET_REV)
    target = AutoModelForCausalLM.from_pretrained(TARGET, revision=TARGET_REV, dtype=torch.bfloat16).eval()
    drafter = AutoModelForCausalLM.from_pretrained(DRAFTER, revision=DRAFTER_REV, dtype=torch.bfloat16).eval()
    drafter.generation_config.num_assistant_tokens = k
    drafter.generation_config.num_assistant_tokens_schedule = "constant"
    print("LOADED", type(target).__name__, type(drafter).__name__, flush=True)

    from transformers.generation import candidate_generator as cg

    rounds = []
    original = cg.SinglePositionMultiTokenCandidateGenerator.update_candidate_strategy

    def record(self, input_ids, scores, num_matches):
        rounds.append(int(num_matches))
        return original(self, input_ids, scores, num_matches)

    cg.SinglePositionMultiTokenCandidateGenerator.update_candidate_strategy = record
    report = {"K": k, "new_tokens": new_tokens, "runs": []}
    for text in prompts:
        chat = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
        ids = tok(chat, add_special_tokens=False, return_tensors="pt").input_ids
        rounds.clear()
        started = time.time()
        with torch.no_grad():
            result = target.generate(
                ids,
                assistant_model=drafter,
                max_new_tokens=new_tokens,
                do_sample=False,
                eos_token_id=None,
            )
        generated = result[0, ids.shape[1] :].tolist()
        # The first round is the prefill (no drafts); drop it from acceptance.
        accepted = rounds[1:]
        row = {
            "prompt": text,
            "prompt_tokens": ids.shape[1],
            "tokens": generated,
            "accepted_per_round": accepted,
            "mean_accepted": sum(accepted) / max(1, len(accepted)),
            "seconds": time.time() - started,
        }
        report["runs"].append(row)
        print("HOSTREF", f"{row['mean_accepted']:.3f}", len(accepted), repr(tok.decode(generated)[:120]), flush=True)
        json.dump(report, open(out, "w"), indent=1)
    allr = [a for r in report["runs"] for a in r["accepted_per_round"]]
    report["mean_accepted"] = sum(allr) / len(allr)
    json.dump(report, open(out, "w"), indent=1)
    print("HOSTREF_MEAN", report["mean_accepted"], flush=True)


if __name__ == "__main__":
    main()
