# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Reuse a pinned greedy HF control for bounded long-prompt TT top-1 checks.

Run only with exclusive device ownership. The saved oracle contains top-1 IDs,
not HF top-5/logits: this tool intentionally reports top-1 agreement only.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--messages", type=int, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--validate-only", action="store_true", help="Check exact prompt/reference without TTNN import")
    args = parser.parse_args()
    import torch
    from hf_eval_prefix_oracle import MODEL, REVISION
    from replay_eval_requests import TOOLS
    from transformers import AutoTokenizer

    oracle = json.loads(args.oracle.read_text())
    if oracle.get("model") != MODEL or oracle.get("revision") != REVISION or oracle.get("do_sample") is not False:
        raise ValueError("Oracle must be a greedy control at the exact pinned checkpoint")
    messages = json.loads(args.trajectory.read_text())["messages"][: args.messages]
    messages = [
        {k: v for k, v in m.items() if k in {"role", "content", "tool_calls", "tool_call_id"}} for m in messages
    ]
    for message in messages:
        if isinstance(message.get("content"), str):
            message["content"] = [{"type": "text", "text": message["content"]}]
        for call in message.get("tool_calls") or []:
            if isinstance(call["function"].get("arguments"), str):
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    tokens = tokenizer.apply_chat_template(
        messages, tools=TOOLS, add_generation_prompt=True, enable_thinking=True, tokenize=True, return_dict=False
    )
    digest = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
    if len(tokens) != oracle["prompt_tokens"] or digest != oracle["prompt_sha256"]:
        raise ValueError(f"Prompt does not match the saved HF control: {len(tokens)} tokens, sha256={digest}")
    generated = oracle["output_token_ids"]
    if not generated or len(tokens) + len(generated) > args.max_seq_len:
        raise ValueError("Invalid or out-of-capacity reference")
    if args.validate_only:
        print(json.dumps({"prompt_tokens": len(tokens), "prompt_sha256": digest, "reference_tokens": len(generated)}))
        return
    import ttnn
    from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import build_generator
    from models.common.readiness_check.run_teacher_forcing import _run_one_entry
    from models.common.readiness_check.schema import Reference, ReferenceEntry
    from models.common.readiness_check.teacher_forcing import TokenAccuracy

    reference = Reference(
        k=1,
        hf_model_id=MODEL,
        entries=[
            ReferenceEntry(
                prompt_text="Exact saved SWE tool-history prompt; content retained only in source artifact",
                prompt_tokens=torch.tensor([tokens], dtype=torch.int64),
                generated_tokens=torch.tensor([generated], dtype=torch.int64),
                topk_tokens=torch.tensor(generated, dtype=torch.int32).reshape(-1, 1),
                tf_prompt_len=len(tokens),
            )
        ],
    )
    report = {
        "scope": "long-prompt traced teacher forcing, HF top-1 only; not an eval reward or serving-speed result",
        "model": MODEL,
        "revision": REVISION,
        "prompt_tokens": len(tokens),
        "prompt_sha256": digest,
        "oracle_sha256": hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "generated_tokens": len(generated),
        "test_capacity": args.max_seq_len,
        "status": "started",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    try:
        started = time.monotonic()
        gen = build_generator(None, mesh, max_seq_len=args.max_seq_len, precision_config=args.policy)
        report["load_s"] = time.monotonic() - started
        report["runtime_policy"] = gen.model.precision_summary()
        acc = TokenAccuracy(reference)
        stats = _run_one_entry(generator=gen, acc=acc, entry_idx=0)
        report["result"] = {
            k: v for k, v in stats.items() if k not in {"top5", "top100", "matches_top5", "matches_top100"}
        }
        report["predicted_token_ids"] = acc.get_predicted_tokens()
        report["metrics"] = dict(gen.metrics)
        assert not gen.metrics["reduced_probe"]
        assert gen.metrics["counters"]["model_replays"] == len(generated) - 1
        assert gen.metrics["counters"]["sampling_replays"] == len(generated) - 1
        report["status"] = "completed"
        save()
    except Exception as error:
        report.update(status="runtime_error", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
