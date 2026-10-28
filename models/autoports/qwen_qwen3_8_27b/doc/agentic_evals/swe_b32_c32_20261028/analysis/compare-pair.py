"""Compare authoritative per-arm analyses; preserve every task and missing value."""
import csv
import json
import sys
from pathlib import Path

combined, control, output = map(Path, sys.argv[1:])
arms = {"combined": json.loads(combined.read_text()), "control": json.loads(control.read_text())}
tasks = {}
for arm, data in arms.items():
    rows = [task for evaluation in data["evals"] for task in evaluation["tasks"]]
    names = [task["task_name"] for task in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate tasks in {arm}; resolve attempts explicitly")
    tasks[arm] = {task["task_name"]: task for task in rows}
rows = []
for name in sorted(set(tasks["combined"]) | set(tasks["control"])):
    row = {"task": name}
    pair = [tasks[arm].get(name) for arm in arms]
    row["same_task_checksum"] = all(pair) and pair[0]["task_checksum"] == pair[1]["task_checksum"]
    for field in ("messages_sha256", "prompt_ids_sha256"):
        values = [(task.get("first_request_identity") or {}).get(field) if task else None for task in pair]
        row["same_initial_" + field] = values[0] == values[1] if all(values) else None
    for arm in arms:
        task = tasks[arm].get(name)
        row[arm + "_present"] = task is not None
        if task is None:
            continue
        prefix = arm + "_"
        row[prefix + "reward"] = (task.get("verifier", {}).get("rewards") or {}).get("reward")
        row[prefix + "exit"] = task.get("agent_exit_status")
        row[prefix + "exception"] = task.get("exception_type")
        row[prefix + "trial_seconds"] = task.get("trial_wall", {}).get("seconds")
        row[prefix + "agent_seconds"] = task.get("phases", {}).get("agent_execution", {}).get("seconds")
        for key in ("started", "finished", "unfinished", "completion_tokens", "prompt_tokens", "cached_tokens"):
            row[prefix + key] = task.get("http_exact_session_match", {}).get(key)
        for key in ("comparable_turns", "full_prefix_preserved_turns", "lost_history_tokens", "unexplained_loss_beyond_quantization"):
            row[prefix + key] = task.get("whole_turn_prefix", {}).get(key)
    rows.append(row)
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"tasks": len(rows), "same_task_checksums": sum(row["same_task_checksum"] for row in rows), "output": str(output)}))
