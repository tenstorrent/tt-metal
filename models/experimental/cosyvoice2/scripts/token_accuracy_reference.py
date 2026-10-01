# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The PyTorch reference's next-token predictions along its own generated sequences, for teacher-forced token accuracy.

RUN IN THE REFERENCE VENV (see requirements-reference*.txt and scripts/reference_env.py):

    COSYVOICE2_REPO=<upstream checkout> $COSYVOICE2_REF_ENV/bin/python token_accuracy_reference.py \\
        --inputs <prepare_inputs.py dir> --run-dir <run_reference.py dir> --out-dir <dir>

For each corpus case, the full prompted sequence is built exactly as upstream's `Qwen2LM.inference` builds it:
`[sos, embed(prompt text + segment text), task_id, embed(prompt speech tokens)]`. The speech tokens the reference
generated for that case (`results.json`, `segment_tokens`) are appended. One no-cache forward gives the log-probs at
every generated position: the last prefix position predicts token 0, ..., and the last token's position predicts
the end. Writes `<case_id>.npz` with `tokens` [N], `top5` [N+1, 5] and `logprob_top5` [N+1, 5]. The TT side
(tests/e2e/test_token_accuracy.py) forces the same tokens through its decode loop and compares top-1.

The noise floor: `--precision bf16` runs the same forward with the whole LLM in bf16 (`bf16-fp32-head`: all but the
output head), and `--against <the fp32 run's --out-dir>` reports its top-1 agreement with fp32, as the TT test
does. That is the agreement a bf16 implementation of this exact model reaches with no port error at all.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference_env  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--precision", choices=("fp32", "bf16", "bf16-fp32-head"), default="fp32")
    ap.add_argument("--against", help="another run's --out-dir: report top-1 agreement with it")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import torch

    model = reference_env.load_upstream()
    lm = model.model.llm
    if args.precision != "fp32":
        lm.to(torch.bfloat16)
    if args.precision == "bf16-fp32-head":
        lm.llm_decoder.float()
    head_dtype = lm.llm_decoder.weight.dtype
    agreement = []  # (case_id, top-1 agrees [N+1], top-1 in the other run's top-5 [N+1], its top-1/top-2 margin)
    with open(os.path.join(args.run_dir, "results.json")) as fh:
        results = {r["case_id"]: r for r in json.load(fh)["results"]}
    for case_id, r in results.items():
        d = np.load(os.path.join(args.inputs, f"{case_id}.npz"))
        segments = json.loads(str(d["segment_text_ids_json"]))
        assert len(segments) == len(r["segment_tokens"]) == 1, "single-segment cases only"
        text = torch.tensor([segments[0]], dtype=torch.long)
        prompt_text = torch.from_numpy(d["prompt_text_ids"].astype(np.int64))
        prompt_speech = torch.from_numpy(d["llm_prompt_speech_tokens"].astype(np.int64))
        tokens = torch.tensor(r["segment_tokens"][0], dtype=torch.long)
        with torch.inference_mode():
            prefix = torch.cat(
                [
                    lm.llm_embedding.weight[lm.sos].reshape(1, 1, -1),
                    lm.llm.model.model.embed_tokens(torch.cat([prompt_text, text], dim=1)),
                    lm.llm_embedding.weight[lm.task_id].reshape(1, 1, -1),
                    lm.speech_embedding(prompt_speech),
                ],
                dim=1,
            )
            seq = torch.cat([prefix, lm.speech_embedding(tokens).unsqueeze(0)], dim=1)
            hidden = lm.llm.model(inputs_embeds=seq, output_hidden_states=True, return_dict=True, use_cache=False)
            last = hidden.hidden_states[-1][0, prefix.shape[1] - 1 :].to(head_dtype)
            logp = lm.llm_decoder(last).float().log_softmax(-1)  # [N + 1, V]
            top = logp.topk(5, dim=-1)
        np.savez(
            os.path.join(args.out_dir, f"{case_id}.npz"),
            tokens=tokens.numpy().astype(np.int32),
            top5=top.indices.numpy().astype(np.int32),
            logprob_top5=top.values.numpy().astype(np.float32),
        )
        agree = float((top.indices[:-1, 0] == tokens).float().mean())
        print(
            f"  {case_id:<34} {len(tokens)} tokens; the reference's own top-1 equals its sampled token at {agree:.1%}"
        )
        if args.against:
            other = np.load(os.path.join(args.against, f"{case_id}.npz"))
            assert (other["tokens"] == tokens.numpy()).all(), f"{case_id}: not the same forced sequence"
            top1 = top.indices[:, 0].numpy()
            agreement.append(
                (
                    case_id,
                    top1 == other["top5"][:, 0],
                    np.array([t in r for t, r in zip(top1, other["top5"])]),
                    other["logprob_top5"][:, 0] - other["logprob_top5"][:, 1],
                )
            )
    print(f"wrote {len(results)} cases to {args.out_dir}")
    if agreement:
        report_agreement(agreement, f"{args.precision} against {args.against}")
    return 0


def report_agreement(rows, title: str) -> None:
    """The same table and margin summary as tests/e2e/test_token_accuracy.py."""
    print(f"\n{title}\n  | case | positions | top-1 agreement | top-1 in the other's top-5 |\n  |---|---|---|---|")
    for case_id, agree, in_top5, _ in rows:
        print(f"  | {case_id} | {len(agree)} | {100 * agree.mean():.2f} % | {100 * in_top5.mean():.2f} % |")
    agree = np.concatenate([r[1] for r in rows])
    in_top5 = np.concatenate([r[2] for r in rows])
    print(f"  | **all** | {len(agree)} | **{100 * agree.mean():.2f} %** | {100 * in_top5.mean():.2f} % |")
    margins = np.concatenate([r[3][~r[1]] for r in rows])
    if len(margins):
        print(
            f"  the other run's top-1 / top-2 log-prob margin at the {len(margins)} disagreements: median "
            f"{np.median(margins):.3f}, 90th percentile {np.quantile(margins, 0.9):.3f}, max {margins.max():.3f}"
        )


if __name__ == "__main__":
    sys.exit(main())
