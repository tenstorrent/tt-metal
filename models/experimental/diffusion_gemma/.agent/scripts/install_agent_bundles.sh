#!/usr/bin/env bash
# Link the DiffusionGemma agent bundles into the repo-root locations Claude Code and Cursor read.
#
# Both tools only look in a repo-root directory (`.claude/`, `.cursor/`), but DiffusionGemma work
# must not add tracked files outside models/experimental/diffusion_gemma/. So the content lives in
# the module and the root gets *untracked symlinks* that this script creates, plus matching
# .git/info/exclude entries (a local file, not a tracked one) so they never show up in git status.
#
# This is already how `.claude/` worked -- root `.claude` is in .gitignore and its skills were
# symlinks into `.agent/skills/`. This script makes that reproducible and extends it to `.cursor/`,
# whose 39 DG files used to be tracked at the root (moved into `.agent/cursor/` on 2026-07-30).
#
# Idempotent. Run from anywhere inside the checkout:
#   bash models/experimental/diffusion_gemma/.agent/scripts/install_agent_bundles.sh
#
# Exit codes: 0 pass, 3 checker error.
set -u

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "ERROR: not inside a git worktree" >&2
  exit 3
}
cd "$ROOT" || { echo "ERROR: cannot cd to repo root $ROOT" >&2; exit 3; }

AGENT="models/experimental/diffusion_gemma/.agent"
[ -d "$AGENT/skills" ] || { echo "ERROR: $AGENT/skills not found" >&2; exit 3; }

# link <link-path> <target-relative-to-repo-root>
# Computes the ../ prefix from the link's own depth so the symlink stays relative (and therefore
# valid inside a container, a worktree, or a differently-rooted clone).
link() {
  local link_path="$1" target="$2" depth up i
  depth=$(printf '%s' "${link_path%/*}" | tr -cd '/' | wc -c)
  up=""
  for ((i = 0; i <= depth; i++)); do up="../$up"; done
  mkdir -p "${link_path%/*}"
  # Only ever replace a symlink. A real file or directory there is somebody else's, so bail loudly
  # rather than deleting it.
  if [ -e "$link_path" ] && [ ! -L "$link_path" ]; then
    echo "  SKIP  $link_path (exists and is not a symlink -- not touching it)" >&2
    return 0
  fi
  rm -f "$link_path"
  ln -s "${up}${target}" "$link_path"
  echo "  link  $link_path -> ${up}${target}"
}

echo "Claude Code bundle (.claude/):"
for d in "$AGENT"/skills/*/; do
  link ".claude/skills/$(basename "$d")" "$AGENT/skills/$(basename "$d")"
done
for f in "$AGENT"/commands/*.md; do
  [ -e "$f" ] || continue
  link ".claude/commands/$(basename "$f")" "$AGENT/commands/$(basename "$f")"
done

echo "Cursor bundle (.cursor/):"
# One directory symlink: upstream owns .cursor/commands/ and .cursor/rules/, but .cursor/skills/ is
# entirely DiffusionGemma's, so it can be linked whole.
link ".cursor/skills" "$AGENT/cursor/skills"
# Commands must be linked file-by-file: a `.cursor/commands/dg/` directory symlink would namespace
# them as /dg/dg-01-... and change every command name.
for f in "$AGENT"/cursor/commands/dg-*.md; do
  [ -e "$f" ] || continue
  link ".cursor/commands/$(basename "$f")" "$AGENT/cursor/commands/$(basename "$f")"
done

# Keep the links out of git status without editing the tracked root .gitignore.
EXCLUDE="$(git rev-parse --git-path info/exclude)"
mkdir -p "$(dirname "$EXCLUDE")"
touch "$EXCLUDE"
added=0
for pat in '/.cursor/skills' '/.cursor/commands/dg-*.md'; do
  if ! grep -qxF "$pat" "$EXCLUDE"; then
    if [ "$added" -eq 0 ]; then
      printf '\n# DiffusionGemma agent bundle links (see .agent/scripts/install_agent_bundles.sh)\n' >>"$EXCLUDE"
      added=1
    fi
    printf '%s\n' "$pat" >>"$EXCLUDE"
  fi
done
echo "git exclude: $EXCLUDE ($( [ "$added" -eq 1 ] && echo 'entries added' || echo 'already current' ))"

# The whole point is a clean root, so prove it rather than asserting it. Only untracked (??) and
# newly-added (A) entries matter -- a staged deletion or rename is the move that got us here.
leaked="$(git status --porcelain -- .cursor .claude | grep -E '^(\?\?|A )' | sed '/^$/d')"
if [ -n "$leaked" ]; then
  echo "WARNING: these root paths would be committed:" >&2
  echo "$leaked" >&2
fi

echo "OK: agent bundles linked; no tracked files added at the repo root"
exit 0
