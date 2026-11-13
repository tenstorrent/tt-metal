#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Map completed native GPQA calls before lm_eval writes scored samples."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from models.autoports.google_gemma_4_31b_it.tools.agentic_collect_gpqa import digest, sha, strings


def save_native_statement(private, task_id, boundary):
    """Preserve the exact served choices, never re-render shuffled dataset choices."""
    request = json.loads(boundary["request"])
    messages = request.get("messages", request.get("prompt"))
    request_hash = sha(boundary["request"])
    name = f"{task_id}-{sha(boundary['id'])[:16]}.json"
    path = private / name
    record = {
        "task": task_id,
        "native_response_id": boundary["id"],
        "native_request_sha256": request_hash,
        "source": "Exact logged native request messages/prompt; no dataset re-rendering",
        "messages_or_prompt": messages,
    }
    with path.open("w") as stream:
        os.chmod(path, 0o600)
        stream.write(json.dumps(record, indent=2) + "\n")
    return {"statement_file": str(path), "request_sha256": request_hash}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-log", required=True, type=Path)
    parser.add_argument("--task-list", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--output-cap", type=int, default=126976)
    args = parser.parse_args()
    from lm_eval.tasks import TaskManager, get_task_dict

    selected = json.loads(args.task_list.read_text())["tasks"]
    ids = [t["doc_id"] for t in selected]
    expected = {t["doc_id"]: t for t in selected}
    run = next(r for r in json.loads(args.runs.read_text())["runs"] if r["id"] == args.run_id)
    start = datetime.fromisoformat(run["start_utc"].replace("Z", "+00:00")).timestamp()
    end = datetime.fromisoformat(run["end_utc"].replace("Z", "+00:00")).timestamp() if run.get("end_utc") else None
    task = get_task_dict(["r1_gpqa_diamond"], task_manager=TaskManager())["r1_gpqa_diamond"]
    docs = {}
    args.out.mkdir(parents=True, exist_ok=True)
    private = args.out / "statements.private"
    private.mkdir(exist_ok=True, mode=0o700)
    os.chmod(private, 0o700)
    for ordinal, doc in task.doc_iterator(samples=ids):
        doc_id = ids[ordinal]
        if sha(doc["Question"]) != expected[doc_id]["question_sha256"]:
            raise ValueError("TaskManager document identity mismatch")
        docs[str(doc_id)] = doc["Question"]
        # Remove obsolete separately rendered statements from the earlier tool.
        (private / f"{doc_id}.txt").unlink(missing_ok=True)
    lines = args.server_log.read_text().splitlines()
    bodies = {body["id"]: body for body in digest.responses(iter(lines))}
    mapping = {key: [] for key in docs}
    matched_bodies = {}
    pending = []
    seen = set()
    for line in digest.framed(iter(lines)):
        if "eval_call=" not in line:
            continue
        row, _ = json.JSONDecoder().raw_decode(line.split("eval_call=", 1)[1])
        if row["start"] < start or (end is not None and row["start"] >= end):
            continue
        request = json.loads(row["request"])
        text = list(strings(request.get("messages", request.get("prompt", ""))))
        hits = [key for key, question in docs.items() if any(question in s for s in text)]
        if not hits:
            continue
        if len(hits) != 1:
            raise ValueError("Ambiguous native call task mapping")
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        key = hits[0]
        body = bodies.get(row["id"])
        statement = save_native_statement(private, key, row)
        mapping[key].append(
            {
                "id": row["id"],
                "start": row["start"],
                "end": row["end"],
                "usage": body.get("usage") if body else None,
                "finish_reason": body["choices"][0].get("finish_reason") if body else None,
                **statement,
            }
        )
        if body:
            matched_bodies[row["id"]] = body
        else:
            pending.append(row["id"])
    accounting = args.out / "live_calls.json"
    accounting.write_text(
        json.dumps(
            {
                "run": args.run_id,
                "scored": False,
                "mapping_basis": "Exact official Question substring plus manifest SHA256 and selected doc_iterator ID validation",
                "tasks": [{"task": key, "calls": calls} for key, calls in mapping.items()],
                "pending_response_bodies": pending,
            },
            indent=2,
        )
        + "\n"
    )
    # A private matched snapshot prevents earlier readiness/rerun bodies from
    # appearing as unmapped anomalies in this run's installed digest.
    snapshot = args.out / "responses.private.log"
    with snapshot.open("w") as stream:
        os.chmod(snapshot, 0o600)
        for body in matched_bodies.values():
            stream.write("response_body={" + json.dumps(body) + "}\n")
    summary = digest.digest_run([str(snapshot)], [str(accounting)], args.out, args.run_id, [], args.output_cap)
    print(digest.table(summary))
    print(
        json.dumps(
            {
                "completed_native_replies": len(matched_bodies),
                "pending_response_bodies": len(pending),
                "scores_generated": False,
            }
        )
    )


if __name__ == "__main__":
    main()
