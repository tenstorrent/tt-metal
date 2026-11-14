#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Native-only private triage packets; final lm_eval samples remain authoritative."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from models.autoports.google_gemma_4_31b_it.tools.agentic_collect_gpqa import digest, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--digest", required=True, type=Path)
    parser.add_argument("--task-list", required=True, type=Path)
    args = parser.parse_args()
    from lm_eval.models.openai_completions import LocalChatCompletion
    from lm_eval.tasks import TaskManager, get_task_dict
    from lm_eval.tasks.gpqa.zeroshot.utils import preprocess
    from lm_eval.tasks.r1_evals import utils

    manifest = json.loads(args.task_list.read_text())["tasks"]
    ids = sorted(t["doc_id"] for t in manifest)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document IDs in task manifest")
    # Pinned doc_iterator yields dataset order, even for unsorted samples.
    expected = {str(t["doc_id"]): t for t in manifest}
    task = get_task_dict(["r1_gpqa_diamond"], task_manager=TaskManager())["r1_gpqa_diamond"]
    docs = {str(ids[ordinal]): doc for ordinal, doc in task.doc_iterator(samples=ids)}
    bodies = {b["id"]: b for b in digest.responses((args.digest / "responses.private.log").read_text().splitlines())}
    packets = args.digest / "analysis.private"
    packets.mkdir(mode=0o700, exist_ok=True)
    os.chmod(packets, 0o700)
    index = {
        "provisional": True,
        "official_score": False,
        "basis": "Exact native request choices matched uniquely to preprocessed official Correct Answer text; unchanged pinned main LocalChatCompletion.parse_generations and process_results_gpqa/BoxedChoiceFilter. Task default take_first; no shuffled re-render. Await main final samples. No aggregate score.",
        "filter_source": str(Path(utils.__file__).resolve()),
        "filter_source_sha256": hashlib.sha256(Path(utils.__file__).read_bytes()).hexdigest(),
        "tasks": [],
    }
    for statement_path in sorted((args.digest / "statements.private").glob("*.json")):
        statement = json.loads(statement_path.read_text())
        key = statement["task"]
        doc = docs[key]
        if sha(doc["Question"]) != expected[key]["question_sha256"]:
            raise ValueError("Official question identity mismatch")
        body = bodies.get(statement["native_response_id"])
        if not body:
            continue
        messages = statement["messages_or_prompt"]
        texts = (
            [m["content"] for m in messages if isinstance(m, dict) and isinstance(m.get("content"), str)]
            if isinstance(messages, list)
            else [messages]
        )
        matching = [text for text in texts if doc["Question"] in text]
        classification = "unknown"
        reason = None
        chosen = None
        gold = None
        choices = None
        if len(matching) != 1:
            reason = "Not exactly one native question-containing message"
        else:
            # Match literal template suffix and four line-start choice labels.
            remainder = matching[0].split(doc["Question"], 1)[1]
            match = re.fullmatch(
                r"\nChoices:\n\(A\) (.*?)\n\(B\) (.*?)\n\(C\) (.*?)\n\(D\) (.*?)\nPlease reason step by step, and your final answer must be only \(A,B,C or D\) within \\boxed\nAnswer:",
                remainder,
                re.S,
            )
            if not match:
                reason = "Native request does not exactly match pinned task template suffix"
            else:
                choices = list(match.groups())
                correct = preprocess(doc["Correct Answer"])
                positions = [i for i, c in enumerate(choices) if c == correct]
                if len(positions) != 1:
                    reason = "Official gold text has no unique exact match in native choices"
                else:
                    gold = "ABCD"[positions[0]]
                    native_doc = {"answer": f"({gold})", **{f"choice{i+1}": c for i, c in enumerate(choices)}}
                    outputs = LocalChatCompletion.parse_generations(body)
                    chosen = utils.choice_answer_clean(
                        utils._BOXED_CHOICE.apply([[outputs[0]]], [{**native_doc, "choices": choices}])[0][0]
                    )
                    classification = (
                        "provisional_correct"
                        if utils.process_results_gpqa(native_doc, [outputs[0]])["exact_match"]
                        else "provisional_incorrect"
                    )
        packet = {
            "task": key,
            "native_response_id": body["id"],
            "request_sha256": statement["native_request_sha256"],
            "statement": statement,
            "full_native_reply": body,
            "expected_letter": gold,
            "chosen_letter": chosen,
            "classification": classification,
            "unknown_reason": reason,
            "digest_path": str(args.digest / f"{key}.json"),
            "provisional": True,
            "official_score": False,
        }
        path = packets / f'{key}-{sha(body["id"])[:16]}.json'
        with path.open("w") as f:
            os.chmod(path, 0o600)
            f.write(json.dumps(packet, indent=2) + "\n")
        index["tasks"].append(
            {
                "task": key,
                "native_response_id": body["id"],
                "classification": classification,
                "unknown_reason": reason,
                "packet": str(path),
                "digest": str(args.digest / f"{key}.json"),
            }
        )
    (args.digest / "provisional_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(
        json.dumps(
            {
                "classified": len(index["tasks"]),
                "provisional_incorrect": [
                    r["task"] for r in index["tasks"] if r["classification"] == "provisional_incorrect"
                ],
                "unknown": [r["task"] for r in index["tasks"] if r["classification"] == "unknown"],
                "official_score_written": False,
            }
        )
    )


if __name__ == "__main__":
    main()
