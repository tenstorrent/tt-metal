"""Read-only discriminating checks; never executes recorded task commands."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "nightly-swe-observable/collected"
p = next(root.glob("**/pydata__xarray-4966*/agent/mini-swe-agent.trajectory.json"))
data = json.loads(p.read_text())
m = data["messages"]
command = m[2]["extra"]["actions"][0]["command"]
assert command.startswith("curl -I https://observations.ipsl.fr/")
assert command.removeprefix("curl -I ") in m[1]["content"]
assert not any(x["role"] == "tool" for x in m[:2])
assert m[3]["extra"]["returncode"] == 0
assert "HTTP/1.1 400" in m[3]["extra"]["raw_output"]
assert "Server: Microsoft-IIS/10.0" in m[3]["extra"]["raw_output"]
config = json.loads((p.parent.parent / "config.json").read_text())
assert config["agent"]["env"]["BASH_ENV"] == "/root/.bashrc"
assert m[19]["extra"]["returncode"] == 0
assert "Decoded dtype: uint8" in m[19]["extra"]["raw_output"]
assert m[57]["extra"]["returncode"] == 0
assert "SUCCESS: Data matches expected signed values" in m[57]["extra"]["raw_output"]
assert data["info"]["exit_status"] == "Submitted"
assert data["info"]["model_stats"]["api_calls"] == 32
print(
    "PASS: first-action issue URL HEAD reached remote; activation configured; local reproduction executable; Submitted."
)
