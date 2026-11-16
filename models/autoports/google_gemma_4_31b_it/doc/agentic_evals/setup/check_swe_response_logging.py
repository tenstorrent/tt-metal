"""Parent-invoked unscored logging check; prepared only, never run by setup agent."""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

model = Path("models/autoports/google_gemma_4_31b_it")
base = model / "doc/agentic_evals"
server = json.loads((base / "current/head-bfp8-rates/server/perf-b32-server.json").read_text())
expected = json.loads((base / "setup/head_bfp8_precision_config.json").read_text())
assert server["precision"] == expected
out = base / "current/swe-phase2"
out.mkdir(parents=True, exist_ok=True)
payload = {
    "model": server["model"],
    "messages": [{"role": "user", "content": "Reply with the single word READY."}],
    "max_tokens": 32,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1,
    "seed": 42,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
}
request = urllib.request.Request(
    server["base_url"].rstrip("/") + "/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=300) as response:
    result = json.load(response)
(out / "logging-probe.json").write_text(
    json.dumps({"unscored": True, "request": payload, "response": result}, indent=2) + "\n"
)
skill = Path("/home/mvasiljevic/v5-agentic/codex-home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.27/scripts")
subprocess.run(
    [
        sys.executable,
        str(skill / "reply_digest.py"),
        "--check-log",
        "--log",
        server["server_log"],
        "--id",
        result["id"],
        "--record",
        str(out / "logging_check.json"),
    ],
    check=True,
)
print(
    json.dumps(
        {"response_id": result["id"], "server_log": server["server_log"], "record": str(out / "logging_check.json")}
    )
)
