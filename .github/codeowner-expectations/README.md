# Codeowner Expectations

This directory contains **expectations files** — per-area review profiles that capture the recurring criteria a codeowner applies when reviewing PRs in their area.

## Purpose

- Give PR authors clear, searchable guidance before they implement a change.
- Give AI-assisted reviewers (BrAIn, `/bug-check`) concrete, repo-owned criteria to check before human reviewers are pinged.
- Reduce review churn by surfacing expectations at author time, not review time.

## How It Works

Each file in this directory corresponds to one or more paths in `CODEOWNERS`. When a PR changes files in a covered area, the relevant expectations file is surfaced to the automated reviewer and, optionally, to the PR author as a checklist.

Integration with the existing `/bug-check` workflow: the orchestrator loads matching expectations files alongside bug-pattern rules and passes them to Claude for a first-pass review.

## File Naming

Use the format `<area>.md` matching the dominant path component (e.g. `tt-metalium-api.md` for `tt_metal/api/`).

## Template

Copy `template.md` to start a new expectations file.

## Maintaining Expectations

- The expectations file for an area is owned by the same codeowners as the area itself (see `CODEOWNERS` entries below).
- Update the file whenever recurring review feedback reveals a gap.
- Keep entries short and actionable — this is a checklist, not a design doc.
- Flag items that are **hard blockers** (block merge) vs. **guidance** (informational).

## Coverage

| File | Paths covered | Codeowners |
|------|--------------|------------|
| `tt-metalium-api.md` | `tt_metal/api/` | `@tenstorrent/metalium-api-owners` |
| `tt-metalium-api-experimental.md` | `tt_metal/api/tt-metalium/experimental/` | `@akerteszTT @riverwuTT` and area owners |
| `fabric.md` | `tt_metal/fabric/`, `tt_metal/api/tt-metalium/experimental/fabric` | `@ubcheema @aliuTT` et al |
| `distributed.md` | `tt_metal/distributed/` | `@cfjchu @aliuTT @tt-asaigal @jbaumanTT` |
| `tt-stl.md` | `tt_stl/` | `@ayerofieiev-tt @akerteszTT @riverwuTT` |
| `hw.md` | `tt_metal/hw/` | `@abhullar-tt @jbaumanTT @nathan-TT @kstevensTT` |
| `ci-cd.md` | `.github/`, `cmake/`, `infra/` | `@tenstorrent/metalium-developers-infra` |
