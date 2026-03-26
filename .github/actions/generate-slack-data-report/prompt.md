Read the JSON file at __INPUT_JSON_PATH__.

Produce a concise markdown report with:
1) percentage of top-level messages that led to tests being disabled
2) percentage of top-level messages that led to developers fixing the problem
3) percentage of top-level messages/jobs still failing with no disable and no clear resolution
4) average time between first top-level message and first clear "fixed/resolved" signal

Requirements:
- Use only evidence from message text and thread replies in this JSON.
- State your classification heuristics explicitly in a "Methodology" section.
- Do not use subagents (`task`) and do not run shell/edit commands; work directly from the provided JSON content.
- Minimize intermediate narration. Keep work output concise until the final report section.
- Parse only from `messages[]`, each top-level item's `text`, and its `thread_replies[]` (plus useful text in `attachments[].text` / `attachments[].fallback` when present).
- Treat developer-posted PR links in the same thread as a strong signal and prioritize them over generic discussion text.
- Recognize GitHub links in Slack format, including:
  - `<https://github.com/tenstorrent/tt-metal/pull/12345|...>`
  - `https://github.com/tenstorrent/tt-metal/pull/12345`
  - `/pull/12345/changes` and PR links with `#issuecomment-...`
- Classification precedence (highest to lowest):
  1) **Disabled**: thread contains phrases like "PR to disable", "disable the test", "disable for now", or clear disable-intent around a PR link.
  2) **Fixed/Resolved**: thread contains fix-intent PR references ("PR to fix", "fix here", "addresses this", "tests now pass", "now merged", "looks resolved") and no stronger disable signal.
  3) **Unresolved**: no convincing disable/fix signal, or evidence is ambiguous/conflicting.
- If a PR link exists but status is uncertain, classify as unresolved unless nearby thread text explicitly indicates fix/resolve outcome.
- For average time-to-fix, use the timestamp delta between top-level `ts` and the earliest reply that provides a clear fixed/resolved signal.
- In the final report, include a short "High-confidence evidence examples" section with 3-5 concrete thread snippets (message ts + key phrase) that drove classifications.
- Include total top-level message count and per-category counts.
- If an item is ambiguous, classify as "still failing/unresolved" and mention ambiguity.
- Show percentages with one decimal place.
- If no resolved items exist, report average time as "N/A".
