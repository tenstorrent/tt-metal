# Update Ticket

## Overview
Review one CI ticket and decide whether to close it, update it, or leave it unchanged based on the latest deterministic evidence from logs on `main`.

## Input
- **Required:** exactly one issue URL in `tenstorrent/tt-metal`.
- If no URL is provided, stop and ask for one.

## Steps
1. **Load ticket context**
   - Run `gh issue view` for the provided ticket.
   - Capture labels, current body quality, and referenced workflow/job/run links.
   - Follow `.cursor/rules/update-ticket.mdc`.

2. **Resolve target CI job**
   - Identify the job from the issue body/comments (prefer the "Most recent failing job URL" link).
   - If job identity is ambiguous, stop and ask the user which job the ticket should track.

3. **Collect current evidence from main**
   - Find the latest 3 completed runs on `main` for that exact job identity.
   - Download and read logs for all 3 runs.
   - Determine the terminal failure signature for each run from logs (not annotations).

4. **Choose one action**
   - **Close ticket** when:
     - the job/test is now passing, or
     - the 3 recent failing runs differ in root cause (not deterministic).
   - **Update ticket** when:
     - the same deterministic failure repeats in all 3 runs, and
     - the issue is stale/vague/poorly formatted or missing useful details.
   - **Leave unchanged** when:
     - the same deterministic failure repeats, and
     - the issue is already current, clear, and actionable.

5. **If closing**
   - Add a closure comment summarizing evidence and linking recent run/job URLs.
   - Close the issue with `gh issue close`.

6. **If updating**
   - Edit the issue body so it reflects current evidence.
   - Add a clear update note (or modify a pre-existing one) in the issue description with today's date (for example, `Last updated: YYYY-MM-DD`).
   - Improve clarity/readability (strong headings, concise bullets, concrete error excerpt).
   - Include concrete failure details from logs and refresh job links so "most recent failing job URL" and "last 3 failing instance URLs" point to the newest relevant runs from this review.
   - If still a CI maintenance issue, ensure label `glean CI maintenance` is present.

7. **If leaving unchanged**
   - Add no-op rationale in your final response to the user (why the ticket remains accurate).

## Notes
- Do not create a new issue; this command only operates on the provided ticket.
- Use logs as the source of truth; avoid annotation-only decisions.
- Prefer terminal/root failure over earlier incidental assertions.
