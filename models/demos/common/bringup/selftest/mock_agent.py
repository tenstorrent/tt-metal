# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Stand-in for ``claude -p`` in the orchestrator tests (BRINGUP_AGENT_CMD="python mock_agent.py").

Reads the brief path from the prompt, looks up ``<brief file name>`` in the JSON script at $MOCK_AGENT_SCRIPT, and
does what the entry says: {"write": {path: text}, "bash": [commands to report], "exit": rc}. It writes files relative
to its cwd (the repo) and prints stream-json events like the real CLI: init, one Bash tool_use per reported command,
and the result.
"""

import json
import os
import re
import sys
from pathlib import Path


def main():
    prompt = sys.argv[-1]
    agent = sys.argv[sys.argv.index("--agent") + 1] if "--agent" in sys.argv else None
    brief = Path(re.search(r"brief at (\S+) and", prompt).group(1))
    script = json.loads(Path(os.environ["MOCK_AGENT_SCRIPT"]).read_text())
    act = script.get(brief.name, {})
    calls = Path(os.environ["MOCK_AGENT_SCRIPT"]).with_suffix(".calls")
    with open(calls, "a") as f:
        f.write(f"{brief.name} {agent}\n")
    print(json.dumps({"type": "system", "subtype": "init", "session_id": f"sess-{brief.stem}", "model": "mock-model"}))
    for path, text in act.get("write", {}).items():
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    for cmd in act.get("bash", []):
        print(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {"command": cmd}}]},
                }
            )
        )
    rc = act.get("exit", 0)
    print(json.dumps({"type": "result", "result": "done", "is_error": rc != 0, "session_id": f"sess-{brief.stem}"}))
    sys.exit(rc)


if __name__ == "__main__":
    main()
