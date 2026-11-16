"""Supplemental reply scan; emits suspects, never adjudicates a task."""
import argparse
import datetime
import hashlib
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
live = root / "current/swe-extended85-head-bfp8/live-tasks"
rows, sources = [], {}
responses = 0
pattern = re.compile(r"['’](ve|s|d|ll|re|m)['’]\1\b", re.I)
for log in sorted(live.glob("*/responses.private.log")):
    task = log.parent.name.removesuffix("-first")
    digest = json.loads((log.parent / "digest" / (task + ".json")).read_text())
    calls = {c["id"]: c["n"] for c in digest["calls"]}
    sources[str(log)] = hashlib.sha256(log.read_bytes()).hexdigest()
    seen = set()
    for line in log.read_text().splitlines():
        if "response_body=" not in line:
            continue
        raw = line.split("response_body=", 1)[1]
        body, _ = json.JSONDecoder().raw_decode(raw[1:] if raw.startswith("{{") else raw)
        rid = body["id"]
        assert rid not in seen and rid in calls
        seen.add(rid)
        responses += 1
        choice = body["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        tools = message.get("tool_calls") or []
        names = [t.get("function", {}).get("name") for t in tools]
        reasons = []
        if not tools:
            reasons.append("no_structured_tool_call")
        if any(n != "bash" for n in names):
            reasons.append("unexpected_tool_name")
        if "GetAll-bash" in content or "call:bash{" in content:
            reasons.append("protocol_text_in_content")
        text = json.dumps(message, ensure_ascii=False)
        doubled = [m.group() for m in pattern.finditer(text)]
        if doubled:
            reasons.append("doubled_contraction")
        if reasons:
            rows.append(
                {
                    "task": task,
                    "call": calls[rid],
                    "response_id": rid,
                    "reasons": reasons,
                    "tool_names": names,
                    "content_chars": len(content),
                    "content_prefix": content[:160],
                    "doubled_matches": doubled,
                }
            )
    assert seen == set(calls)
result = {
    "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "tasks": len(sources),
    "responses": responses,
    "suspects": rows,
    "sources": sources,
    "scope": "Supplemental pattern scan only. No anomaly verdict, absence of flags does not prove semantic normality.",
}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as f:
    json.dump(result, f, indent=2)
    f.write("\n")
print(json.dumps({k: result[k] for k in ["tasks", "responses", "suspects"]}))
