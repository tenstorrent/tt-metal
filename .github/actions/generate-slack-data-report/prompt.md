Read the JSON file at __INPUT_JSON_PATH__.

Produce a concise markdown report with:
1) percentage of top-level messages that led to tests being disabled
2) percentage of top-level messages that led to developers fixing the problem
3) percentage of top-level messages/jobs still failing with no disable and no clear resolution
4) average time between first top-level message and first clear "fixed/resolved" signal

Requirements:
- Use only evidence from message text and thread replies in this JSON.
- State your classification heuristics explicitly in a "Methodology" section.
- Include total top-level message count and per-category counts.
- If an item is ambiguous, classify as "still failing/unresolved" and mention ambiguity.
- Show percentages with one decimal place.
- If no resolved items exist, report average time as "N/A".
- Output markdown only (no code fences).
