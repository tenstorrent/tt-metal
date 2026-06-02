#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Post BrAIn triage pipeline instructions to Slack."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BRAIN_UID = "U0AK4BVCFM0"
CHANNEL = "C0B1F8Z7GQM"
SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"


def build_message(include_lifecycle: bool) -> str:
    steps: list[str] = []
    if include_lifecycle:
        steps.append("`/run-lifecycle`")
    steps.extend(
        [
            "`/make-new-issues 1`",
            "`/assign-issue`",
            "`/draft-slack-message` (include pings for the owners you found)",
        ]
    )

    numbered = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, start=1))
    assign_step = len(steps) - 1
    draft_step = len(steps)

    return (
        f"<@{BRAIN_UID}> Run the full triage pipeline in order:\n\n"
        f"{numbered}\n\n"
        f"Combine steps {assign_step}-{draft_step} into a *single Slack message* -- one concise "
        "summary of what the issue is and who should take care of it. "
        "No separate messages per step."
    )


def post_slack_message(token: str, text: str) -> None:
    data = json.dumps({"channel": CHANNEL, "text": text}).encode()
    req = urllib.request.Request(
        SLACK_POST_MESSAGE_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Slack HTTP error: {exc}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Slack request failed: {exc}") from exc

    if not body.get("ok"):
        raise RuntimeError(f"Slack post failed: {body.get('error')}")


def write_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def main() -> int:
    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("SLACK_BOT_TOKEN is required", file=sys.stderr)
        return 1

    include_lifecycle = os.environ.get("INCLUDE_LIFECYCLE", "false").lower() == "true"
    text = build_message(include_lifecycle)
    post_slack_message(token, text)
    write_output("included_lifecycle", str(include_lifecycle).lower())

    if include_lifecycle:
        print("BrAIn notified (daily lifecycle step included).")
    else:
        print("BrAIn notified (lifecycle already run today — omitted).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
