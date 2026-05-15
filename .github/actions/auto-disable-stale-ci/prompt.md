Read the JSON file at __CANDIDATES_JSON_PATH__.

Task:
- Produce a machine-readable disable action plan for stale unresolved CI Slack threads.
- Only include entries where a disable PR should be attempted now.
- Prioritize highest age first.
- Cap actions to __MAX_ACTIONS__.
- Token-efficiency rule: do this in one pass; avoid verbose reasoning and avoid unnecessary extra analysis.

Decision rules:
- Ignore any candidate where `issue_closed` is true (these should already be excluded).
- Ignore/deprioritize any candidate where `progress_signal.defer_disable` is true.
- If `primary_issue_number` is missing, skip unless top-level text clearly contains enough failure context.
- During bootstrap testing, prefer `primary_issue_repo == "ebanerjeeTT/issue_dump"` when present.
- Prefer entries with clear CI failure context and issue references.
- If `fix_request_signal.requested` is true, include an explanatory `skipped` reason unless disable is still clearly required.
- Keep action scope minimal and specific.

Output contract:
- Do not call tools, shell commands, or subagents.
- At the very end, print exactly this marker on its own line:
===FINAL_DISABLE_ACTIONS_JSON===
- After that marker, output only JSON (no markdown) matching:
{
  "version": 1,
  "actions": [
    {
      "source_slack_ts": "string",
      "source_slack_permalink": "string",
      "issue_number": 12345,
      "issue_repo": "ebanerjeeTT/issue_dump",
      "issue_url": "https://github.com/ebanerjeeTT/issue_dump/issues/12345",
      "job_urls": ["optional job urls"],
      "disable_scope_hint": "one line",
      "branch_name_hint": "ci-disable-test-optional-slug",
      "pr_title": "ci: disable ...",
      "pr_body": "short body with rationale and non-closing issue ref"
    }
  ],
  "skipped": [
    {
      "source_slack_ts": "string",
      "reason": "short reason"
    }
  ]
}

Validation constraints:
- `actions` must be deterministic and deduplicated by `source_slack_ts`.
- `issue_number` must be an integer.
- `pr_title` and `pr_body` should be concise.
