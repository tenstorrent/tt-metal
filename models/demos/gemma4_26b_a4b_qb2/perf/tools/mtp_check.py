"""WS6 check: greedy MTP speculative decode vs plain greedy (batch 1, full model).

  python mtp_check.py <out.json> [--prompts N] [--tokens T] [--k K] [--mode eager|traced] [--drafter-pos hf|anchor|committed]

Per prompt: plain greedy tokens (the traced batch-1 decode), speculative tokens, token identity, accepted drafts
per round, and (traced mode) the speculative decode tok/s.
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer

import ttnn
from models.demos.gemma4_26b_a4b_qb2.tt.generator import build_generator
from models.demos.gemma4_26b_a4b_qb2.tt.model import MODEL_ID, REVISION
from models.demos.gemma4_26b_a4b_qb2.tt.mtp.speculative import SpeculativeDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--k", default="5")
    parser.add_argument("--mode", default="eager")
    parser.add_argument("--drafter-pos", default="hf")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--ref-rows", action="store_true", help="also run plain greedy through the batched path")
    args = parser.parse_args()
    torch.set_num_threads(8)
    texts = json.load(open(os.path.join(os.path.dirname(__file__), "mtp_prompts.json")))[: args.prompts]
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1_500_000_000)
    report = {"args": vars(args), "runs": []}
    try:
        gen = build_generator(None, mesh, max_seq_len=args.max_seq_len)
        spec = None
        for k in [int(v) for v in args.k.split(",")]:
            spec = SpeculativeDecoder(
                gen, draft_len=k, drafter=None if spec is None else spec.drafter, drafter_cache_pos=args.drafter_pos
            )
            for text in texts:
                chat = tok.apply_chat_template(
                    [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
                )
                ids = tok.encode(chat, add_special_tokens=False)
                plain = [int(t) for t in gen.generate(list(ids), args.tokens, stop_on_eos=False)]
                plain_tps = gen.metrics.get("decode_tps")
                if args.mode == "traced":
                    spec.generate_traced(list(ids), args.tokens)  # warm-up / capture
                    tokens = spec.generate_traced(list(ids), args.tokens)
                else:
                    tokens = spec.generate(list(ids), args.tokens)
                rows_ref = None
                if args.ref_rows:
                    rows_ref = spec.generate_plain_rows(list(ids), args.tokens)
                    margins = spec.stats_rows["margins"]
                same = tokens == plain
                first_diff = next((i for i, (a, b) in enumerate(zip(tokens, plain)) if a != b), None)
                row = {
                    "k": k,
                    "prompt": text,
                    "identical": same,
                    "first_diff": first_diff,
                    "plain_tps": plain_tps,
                    "plain": plain,
                    "spec": tokens,
                    **spec.stats,
                }
                if rows_ref is not None:
                    row["rows_ref"] = rows_ref
                    row["identical_to_rows_ref"] = tokens == rows_ref
                    row["rows_ref_vs_plain_first_diff"] = next(
                        (i for i, (a, b) in enumerate(zip(rows_ref, plain)) if a != b), None
                    )
                    d = row["rows_ref_vs_plain_first_diff"]
                    row["margin_at_plain_divergence"] = margins[d] if d is not None else None
                    print(
                        "ROWSREF",
                        f"spec==rows_ref={row['identical_to_rows_ref']}",
                        f"rows_ref_vs_plain_first_diff={d}",
                        f"margin={row['margin_at_plain_divergence']}",
                        flush=True,
                    )
                report["runs"].append(row)
                print(
                    "MTP",
                    f"k={k}",
                    f"identical={same}",
                    f"first_diff={first_diff}",
                    f"accepted={spec.stats['mean_accepted']:.3f}",
                    f"tps={spec.stats.get('tps', 0):.1f}",
                    f"plain_tps={plain_tps or 0:.1f}",
                    repr(text[:40]),
                    flush=True,
                )
                if not same and first_diff is not None:
                    print("  plain:", repr(tok.decode(plain[max(0, first_diff - 5) : first_diff + 5])))
                    print("  spec :", repr(tok.decode(tokens[max(0, first_diff - 5) : first_diff + 5])))
                json.dump(report, open(args.out, "w"), indent=1)
        gen._release_trace()
    finally:
        ttnn.close_mesh_device(mesh)
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
