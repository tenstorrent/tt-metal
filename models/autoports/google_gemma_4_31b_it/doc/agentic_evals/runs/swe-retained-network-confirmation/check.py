"""Read-only, narrow provenance check; never executes captured commands."""
import hashlib
import json
from pathlib import Path

r = Path(__file__).resolve().parent
source = next(
    (r.parent / "nightly-swe-decode-retention/collected").rglob(
        "pydata__xarray-4966__*/agent/mini-swe-agent.trajectory.json"
    )
)
raw = source.read_bytes()
messages = json.loads(raw)["messages"]
checks = []
assert "observations.ipsl.fr" in messages[1]["content"]
assert all(s in messages[9]["extra"]["raw_output"] for s in ["netCDF4", "pydap", "xarray"])
for index in (10, 12, 42):
    m = messages[index]
    tool = messages[index + 1]
    native = m["extra"]["response"]["choices"][0]["message"]["tool_calls"]
    assert native == m["tool_calls"]
    call = native[0]
    command = json.loads(call["function"]["arguments"])["command"]
    action = m["extra"]["actions"][0]
    assert action["command"] == command and action["tool_call_id"] == call["id"] == tool["tool_call_id"]
    assert "observations.ipsl.fr" in command and "xr.open_dataset" in command and "python reproduce_issue.py" in command
    assert (
        tool["extra"]["returncode"] == 0 and not tool["extra"]["exception_info"] and "[" in tool["extra"]["raw_output"]
    )
    checks.append(
        dict(
            assistant_message=index,
            tool_message=index + 1,
            native_action_exact=True,
            tool_call_id_exact=True,
            returncode=0,
            observed_data_values=True,
        )
    )
print(
    json.dumps(
        dict(
            source=str(source),
            source_sha256=hashlib.sha256(raw).hexdigest(),
            issue_supplied_endpoint=True,
            preceding_installed_dependencies=True,
            checks=checks,
        ),
        indent=2,
    )
)
