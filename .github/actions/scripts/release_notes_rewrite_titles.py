#!/usr/bin/env python3
"""Rework unclear PR titles in the changelog at release time (MINFRA-978).

Backstop for the PR-time Copilot title suggestions: for each PR line in the
changelog, ask Claude whether the title reads clearly for release notes and,
only when it does not, replace it with a clearer one derived from the PR title
and description. Selective (already-clear titles are left untouched) and not
cached (every run re-evaluates each title).

Operates on the changelog line text only -- the GitHub PR title is unchanged.
Runs before layer grouping so each PR is evaluated once. Degrades gracefully:
with no API key, or on a per-PR failure, the original title is kept (a title
nicety must never fail a release).
"""
import json
import os
import re
import sys
import urllib.request

import anthropic

MODEL = os.environ.get("BUG_CHECKER_MODEL", "claude-sonnet-4-6")
# A PR line produced by the changelog config: "- <title> [PR <n>](<url>)"
PR_LINE = re.compile(
    r"^- (?P<title>.+?) (?P<link>\[PR \d+\]\(https://github\.com/"
    r"(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<num>\d+)\))\s*$"
)

INSTRUCTIONS = (
    "You improve pull-request titles for tt-metal release notes. Given a PR title and its "
    "description, decide whether the title reads clearly to someone outside the team. Fix "
    "spelling and grammar, replace unresolved references (bare ticket/PR numbers, 'the thing "
    "from yesterday'), and turn internal-only codenames or raw symbol names into plain external "
    "wording, keeping a meaningful component name when it aids clarity. Keep it concise and "
    "imperative. Do not change the technical meaning or invent scope the description does not "
    "support. If the title is already clear, keep it unchanged."
)

TOOL = {
    "name": "assess_title",
    "description": "Report whether the PR title needs a clearer rewrite for release notes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "needs_rewrite": {
                "type": "boolean",
                "description": "True only if the title is unclear, has errors, or uses internal-only references.",
            },
            "title": {
                "type": "string",
                "description": "The final title to use. Equal to the original when needs_rewrite is false.",
            },
        },
        "required": ["needs_rewrite", "title"],
    },
}


def pr_body(owner, repo, number, token):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=30)).get("body") or ""


def improved_title(client, title, body):
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        temperature=0,
        system=INSTRUCTIONS,
        messages=[{"role": "user", "content": f"Title: {title}\n\nDescription:\n{body[:4000]}"}],
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "assess_title"},
    )
    block = next((b for b in response.content if b.type == "tool_use"), None)
    if block and block.input.get("needs_rewrite") and block.input.get("title", "").strip():
        return block.input["title"].strip()
    return None


def main(path):
    key = os.environ.get("BUG_CHECKER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("::warning::no Anthropic API key; leaving titles unchanged", file=sys.stderr)
        return
    token = os.environ.get("GITHUB_TOKEN", "")
    client = anthropic.Anthropic(api_key=key)
    out = []
    for line in open(path, encoding="utf-8").read().splitlines():
        m = PR_LINE.match(line)
        if m:
            title = m.group("title")
            try:
                new = improved_title(client, title, pr_body(m.group("owner"), m.group("repo"), m.group("num"), token))
            except Exception as e:  # a title nicety must never fail a release
                print(f"::warning::PR {m.group('num')}: title rework skipped ({e})", file=sys.stderr)
                new = None
            if new and new != title:
                print(f"PR {m.group('num')}: rewrote title\n  - {title}\n  + {new}", file=sys.stderr)
                line = f"- {new} {m.group('link')}"
        out.append(line)
    open(path, "w", encoding="utf-8").write("\n".join(out).rstrip() + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
