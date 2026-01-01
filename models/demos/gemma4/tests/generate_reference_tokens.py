# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced HF reference for the Gemma 4 token-accuracy test.

Runs the Hugging Face model once (CPU, bf16) over a fixed public-domain
passage and stores, for every position, the reference token that follows it,
the HF top-1 prediction and the HF top-5 predictions:

    reference_tokens: [1, N + 1] int64   chat-templated token ids (user turn + passage)
    top1_tokens:      [N]        int64   HF argmax at each position
    top5_tokens:      [N, 5]     int64   HF top-5 at each position
    score_start:      int                first token of the passage (score from here)

The TT test (test_token_accuracy.py) teacher-forces the same passage and
scores its own top-1 / top-5 against these, so the gate measures the TT
stack against the HF reference, not against the passage.

    HF_MODEL=google/gemma-4-31B-it python generate_reference_tokens.py [--tokens 512]
"""

import argparse
import os
import time

import torch

USER_TURN = "Recite the opening of A Tale of Two Cities by Charles Dickens, verbatim."

PASSAGE = (
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of "
    "foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, "
    "it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything "
    "before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other "
    "way - in short, the period was so far like the present period, that some of its noisiest authorities "
    "insisted on its being received, for good or for evil, in the superlative degree of comparison only. "
    "There were a king with a large jaw and a queen with a plain face, on the throne of England; there were a "
    "king with a large jaw and a queen with a fair face, on the throne of France. In both countries it was "
    "clearer than crystal to the lords of the State preserves of loaves and fishes, that things in general were "
    "settled for ever. It was the year of Our Lord one thousand seven hundred and seventy-five. Spiritual "
    "revelations were conceded to England at that favoured period, as at this. Mrs. Southcott had recently "
    "attained her five-and-twentieth blessed birthday, of whom a prophetic private in the Life Guards had "
    "heralded the sublime appearance by announcing that arrangements were made for the swallowing up of London "
    "and Westminster. Even the Cock-lane ghost had been laid only a round dozen of years, after rapping out its "
    "messages, as the spirits of this very year last past (supernaturally deficient in originality) rapped out "
    "theirs. Mere messages in the earthly order of events had lately come to the English Crown and People, from "
    "a congress of British subjects in America: which, strange to relate, have proved more important to the "
    "human race than any communications yet received through any of the chickens of the Cock-lane brood. "
    "France, less favoured on the whole as to matters spiritual than her sister of the shield and trident, "
    "rolled with exceeding smoothness down hill, making paper money and spending it. Under the guidance of her "
    "Christian pastors, she entertained herself, besides, with such humane achievements as sentencing a youth to "
    "have his hands cut off, his tongue torn out with pincers, and his body burned alive, because he had not "
    "kneeled down in the rain to do honour to a dirty procession of monks which passed within his view, at a "
    "distance of some fifty or sixty yards. It is likely enough that, rooted in the woods of France and Norway, "
    "there were growing trees, when that sufferer was put to death, already marked by the Woodman, Fate, to come "
    "down and be sawn into boards, to make a certain movable framework with a sack and a knife in it, terrible in "
    "history. It is likely enough that in the rough outhouses of some tillers of the heavy lands adjacent to "
    "Paris, there were sheltered from the weather that very day, rude carts, bespattered with rustic mire, "
    "snuffed about by pigs, and roosted in by poultry, which the Farmer, Death, had already set apart to be his "
    "tumbrils of the Revolution. But that Woodman and that Farmer, though they work unceasingly, work silently, "
    "and no one heard them as they went about with muffled tread: the rather, forasmuch as to entertain any "
    "suspicion that they were awake, was to be atheistical and traitorous."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=512, help="scored positions N")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    model_id = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    # The instruction-tuned checkpoints degenerate on raw text ("The capital of
    # France is" -> "111.11" on CPU, bf16 and fp32 alike) and answer correctly
    # under the chat template, so the passage is teacher-forced as the
    # assistant turn of a chat; only its tokens are scored.
    messages = [
        {"role": "user", "content": USER_TURN},
        {"role": "assistant", "content": PASSAGE},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix = text[: text.index(PASSAGE)]
    ids = tok.encode(text, add_special_tokens=False)
    score_start = len(tok.encode(prefix, add_special_tokens=False))
    n = args.tokens
    assert len(ids) >= n + 1, f"chat has {len(ids)} tokens, need {n + 1}"
    ids = torch.tensor(ids[: n + 1]).unsqueeze(0)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    print(f"loaded {model_id} ({type(model).__name__}) in {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    with torch.no_grad():
        logits = model(input_ids=ids[:, :n]).logits[0].float()  # [N, vocab]
    print(f"forward over {n} tokens in {time.time() - t0:.0f}s", flush=True)
    top5 = torch.topk(logits, k=5, dim=-1).indices
    top1 = top5[:, 0]
    true_next = ids[0, 1 : n + 1]
    scored = slice(score_start - 1, n)  # logits at p predict p + 1; passage starts at score_start
    print(
        f"HF vs passage over {n - score_start + 1} assistant-turn positions: "
        f"top1 {(top1[scored] == true_next[scored]).float().mean():.4f} "
        f"top5 {(top5[scored] == true_next[scored, None]).any(-1).float().mean():.4f}"
    )
    out = args.out or os.path.join(os.path.dirname(__file__), "reference_outputs", model_id.split("/")[-1] + ".refpt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(
        {
            "model": model_id,
            "reference_tokens": ids,
            "top1_tokens": top1,
            "top5_tokens": top5,
            "score_start": score_start,
        },
        out,
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
