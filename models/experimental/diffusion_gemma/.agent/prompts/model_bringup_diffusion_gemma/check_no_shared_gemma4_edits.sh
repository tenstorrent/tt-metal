#!/usr/bin/env bash
# Reusable DiffusionGemma gate: the shared Gemma-4 backbone must stay byte-for-byte
# unchanged. Every DiffusionGemma stage fix belongs inside
# models/experimental/diffusion_gemma/ (see $diffusion-gemma, plan.md F1/F2, risk R-new).
#
# Exit codes follow the multigoal convention: 0 pass, 2 critical failure, 3 checker error.
#
# DG_BASE_REF is the ref the DiffusionGemma work started from — the baseline the shared
# dirs must match. On a main-based DiffusionGemma dev branch that is `main` (the exact
# plan.md F1/F2 rule: `git diff main -- models/demos/gemma4` empty). On a branch stacked on
# other work (e.g. this skills branch sits on agentic-research/fast-models-fast, which itself
# carries unrelated shared-dir edits), set DG_BASE_REF to THAT base so the gate flags only
# shared edits introduced by DiffusionGemma work, not the pre-existing divergence.
set -u

# Anchor to the repo root so the cwd-relative git pathspecs below cannot silently match
# nothing (which would fail-open the very gate meant to protect the shared backbone).
ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "CHECKER ERROR: not inside a git worktree" >&2; exit 3; }
cd "$ROOT" || { echo "CHECKER ERROR: cannot cd to repo root $ROOT" >&2; exit 3; }

BASE="${DG_BASE_REF:-main}"

if ! git rev-parse --verify --quiet "$BASE" >/dev/null; then
  echo "CHECKER ERROR: base ref '$BASE' not found (set DG_BASE_REF)" >&2
  exit 3
fi

# Any change (staged, unstaged, or committed on this branch) to the shared backbone
# or other shared model dirs relative to the base ref is a critical violation.
SHARED_PATHS=(models/demos/gemma4 models/common models/tt_transformers)

violations="$(git diff --name-only "$BASE"...HEAD -- "${SHARED_PATHS[@]}" 2>/dev/null)"
worktree="$(git diff --name-only -- "${SHARED_PATHS[@]}" 2>/dev/null)"
staged="$(git diff --name-only --cached -- "${SHARED_PATHS[@]}" 2>/dev/null)"

all="$(printf '%s\n%s\n%s\n' "$violations" "$worktree" "$staged" | sort -u | sed '/^$/d')"

if [ -n "$all" ]; then
  echo "CRITICAL: shared backbone / shared dirs modified vs $BASE — move these into models/experimental/diffusion_gemma/:" >&2
  echo "$all" >&2
  exit 2
fi

echo "OK: no shared-directory edits vs $BASE (models/demos/gemma4, models/common, models/tt_transformers clean)"
exit 0
