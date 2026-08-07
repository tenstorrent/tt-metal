#!/usr/bin/env bash
# skillexp — create a FRESH, ISOLATED run root for one cell.
#
#   bash newcell.sh <arm> <model_dir> [--activate] [--base-sha SHA]
#
# Implements PIPELINE-HARDENING.md §2. The point is not to *detect* contamination but to make the
# other arms' objects unreachable, so B34 cannot recur:
#
#   arm-scoped mirror   ~/skillexp/mirrors/<arm>.git   fetched with a refspec that admits ONLY
#                                                      base + this arm + fd-ready tags
#   cell root           ~/skillexp/cells/<arm>__<md>/  git clone --shared from that mirror
#
# Because the mirror holds no other arm, the cell's `git log --all` sees only its own history and
# `git show <other-arm-sha>` fails with "unknown revision". An operator audit fetch (the actual cause
# of B34) lands in the admin clone and no cell borrows from it.
#
# THE PATH TRAP (found while building this, would have silently corrupted every cell):
#   python_env/lib/*/site-packages/*.pth hardcode /home/mvasiljevic/tt-metal:
#       /home/mvasiljevic/tt-metal        /home/mvasiljevic/tt-metal/ttnn
#   A cell root at a NEW path would therefore import ttnn and models/autoports from the OLD checkout
#   while looking fresh -- measuring the wrong source with no symptom. So the canonical path stays
#   fixed and is a SYMLINK to the active cell; cells are swapped by repointing it. Build artifacts
#   keep working for the same reason (RPATHs resolve through the same canonical path).
set -uo pipefail

ARM=${1:?arm: nofuse-noadvise|fuse-noadvise|nofuse-advise|fuse-advise}
MD=${2:?model_dir}; shift 2
ACTIVATE=0; BASE_SHA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --activate) ACTIVATE=1; shift ;;
    --base-sha) BASE_SHA=${2:?}; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

ROOT=${SKILLEXP_ROOT:-$HOME/skillexp}
CANON=${SKILLEXP_CANON:-$HOME/tt-metal}          # the path python_env and the build are pinned to
ADMIN=$ROOT/admin/tt-metal                # full clone: sees everything, used by publish/review only
MIRROR=$ROOT/mirrors/$ARM.git
CELL=$ROOT/cells/${ARM}__${MD}
SHARED=$ROOT/shared
NS=${SKILLEXP_NS:-refs/heads/mvasiljevic/qb2/skillexp}
die() { echo "NEWCELL REFUSED: $*" >&2; exit 1; }
say() { echo "  $*"; }

case "$ARM" in nofuse-noadvise|fuse-noadvise|nofuse-advise|fuse-advise) ;; *) die "bad arm '$ARM'";; esac
mkdir -p "$ROOT"/{mirrors,cells,shared,admin}

# The source of truth for objects. Prefer the admin clone; fall back to the legacy checkout.
SRC=$ADMIN; [ -d "$ADMIN/.git" ] || SRC=$(readlink -f "$CANON")
[ -d "$SRC/.git" ] || die "no source clone at $SRC"

# ---------------------------------------------------------------- 1. arm-scoped mirror
# Refspec admits base, this arm, this arm's run branch, and fd-ready tags. Deliberately NOT
# refs/tags/skillexp/done/* -- fetching those is precisely what imported A's cell (B37).
if [ ! -d "$MIRROR" ]; then
  git init -q --bare "$MIRROR" || die "cannot create mirror"
  say "created arm-scoped mirror $MIRROR"
fi
git -C "$MIRROR" config --replace-all remote.origin.url "$SRC"
git -C "$MIRROR" config --unset-all remote.origin.fetch 2>/dev/null
for rs in "+$NS/base:$NS/base" "+$NS/$ARM:$NS/$ARM" "+$NS/run/$ARM:$NS/run/$ARM" \
          "+refs/tags/skillexp/fd-ready/*:refs/tags/skillexp/fd-ready/*"; do
  git -C "$MIRROR" config --add remote.origin.fetch "$rs"
done
git -C "$MIRROR" fetch -q --prune origin 2>/dev/null

# Assert the mirror really is arm-scoped: no OTHER arm's tip may resolve in it.
for other in nofuse-noadvise fuse-noadvise nofuse-advise fuse-advise; do
  [ "$other" = "$ARM" ] && continue
  osha=$(git -C "$SRC" rev-parse -q --verify "$NS/run/$other" 2>/dev/null) || continue
  if git -C "$MIRROR" rev-parse -q --verify "$osha^{commit}" >/dev/null 2>&1; then
    die "mirror resolves $other's tip ${osha:0:11} -- NOT arm-scoped, refusing (this is B34)"
  fi
done
say "mirror is arm-scoped: no other arm's tip resolves in it"

# ---------------------------------------------------------------- 2. fresh cell root
[ -e "$CELL" ] && { [ "$(readlink -f "$CANON")" = "$CELL" ] && die "cell $CELL is ACTIVE; deactivate first"
                    rm -rf "$CELL"; say "removed previous $CELL"; }
# --shared borrows objects from the mirror via alternates. Safe ONLY because the mirror is
# arm-scoped -- borrowing from an all-arms clone would silently restore the bug.
git clone -q --shared --single-branch --branch "${NS#refs/heads/}/$ARM" "$MIRROR" "$CELL" \
  || die "clone failed"
if [ -n "$BASE_SHA" ]; then
  git -C "$CELL" rev-parse -q --verify "$BASE_SHA^{commit}" >/dev/null \
    || die "pinned base $BASE_SHA not in this arm's mirror"
  git -C "$CELL" reset -q --hard "$BASE_SHA" || die "cannot pin to $BASE_SHA"
  say "pinned to base $BASE_SHA"
fi

# A fresh clone inherits no committer identity, and without one `git merge` fails BEFORE it can
# conflict. The first version of this script reported that as "FD merge conflicted", which is a
# different and much more alarming condition -- so set the identity, and keep the two apart below.
git -C "$CELL" config user.name  "$(git -C "$SRC" config user.name  || echo skillexp)"
git -C "$CELL" config user.email "$(git -C "$SRC" config user.email || echo skillexp@localhost)"

# bring in the shared functional decoder for this model
FDTAG="refs/tags/skillexp/fd-ready/$MD"
if git -C "$CELL" rev-parse -q --verify "$FDTAG" >/dev/null 2>&1; then
  if ! merr=$(git -C "$CELL" merge --no-edit "$FDTAG" 2>&1); then
    if [ -n "$(git -C "$CELL" diff --name-only --diff-filter=U)" ]; then
      die "FD merge CONFLICTED on: $(git -C "$CELL" diff --name-only --diff-filter=U | tr '\n' ' ')
  The arm branch should touch only .agents/ and the FD only models/ -- investigate before rerunning."
    fi
    die "FD merge failed without conflicting (environment, not content): ${merr%%$'\n'*}"
  fi
  say "merged $FDTAG"
else
  say "NOTE: no $FDTAG yet (expected only for a phase-1 FD cell)"
fi

# ---------------------------------------------------------------- 2b. enforce the arm condition
# Found by the first dry run: `base` ships a worked advisor capture for an UNRELATED model,
# models/autoports/openai_gpt_oss_20b/doc/optimized_decoder/shard_advise/report.json. It was
# therefore in the tree of all 16 cells, both noadvise arms included. Not contamination -- it
# predates the run -- but on a noadvise arm it is a reference leak: an agent that cannot generate
# advisor output can still read a worked example of it and copy the methodology.
#
# Deleting the shard-advise SKILL but leaving a finished advisor REPORT in the tree is not a
# suppressed factor. So strip it here, as an explicit recorded commit -- the clean-room pattern from
# agentic-research-setup: "make deletion an explicit local warning commit so the agent can see the
# reference was intentionally removed."
case "$ARM" in
  *noadvise)
    mapfile -t leaks < <(cd "$CELL" && git ls-files | grep -E 'doc/optimized_decoder/shard_advise/' || true)
    if [ ${#leaks[@]} -gt 0 ]; then
      ( cd "$CELL" && git rm -q -r --cached $(printf '%s\n' "${leaks[@]}" | sed 's#/[^/]*$##' | sort -u) >/dev/null 2>&1
        printf '%s\n' "${leaks[@]}" | sed 's#/[^/]*$##' | sort -u | while read -r d; do rm -rf "$d"; done
        cat > SHARD-ADVISE-REMOVED.md <<'EOM'
# Advisor artifacts removed to construct the `noadvise` condition

The base branch ships a completed `ttnn-advise` capture for an unrelated model. This arm suppresses
the shard advisor, so a worked example of its output must not be readable here either -- otherwise
the factor is only half-suppressed. Removed by newcell.sh; see PIPELINE-HARDENING.md §2.
EOM
        git add -A SHARD-ADVISE-REMOVED.md >/dev/null 2>&1
        git -c core.hooksPath=/dev/null commit -q --no-verify \
          -m "skillexp $ARM: remove advisor artifacts to construct the noadvise condition" >/dev/null 2>&1 )
      say "stripped ${#leaks[@]} advisor artifact(s) inherited from base (recorded as a commit)"
    fi
    ;;
esac

# ---------------------------------------------------------------- 3. shared build + env
# .gitignore has `/python_env/` and `build_*` -- the trailing slash matches a DIRECTORY, so the
# SYMLINKS created below are NOT ignored and `git add -A` tracks them (caught by negative test 2,
# which flagged python_env alongside the tampered file). Exclude them by exact name, locally.
printf 'build_Release\npython_env\n' >> "$CELL/.git/info/exclude"
for a in build_Release python_env; do
  if [ -e "$SHARED/$a" ]; then ln -sfn "$SHARED/$a" "$CELL/$a"; say "linked $a from shared"
  elif [ -e "$SRC/$a" ];    then ln -sfn "$SRC/$a"    "$CELL/$a"; say "linked $a from $SRC"
  else say "WARNING: no $a to link -- the cell cannot run until one exists"; fi
done

# ---------------------------------------------------------------- 4. isolation proof
echo
echo "isolation check (each must be 'unreachable'):"
fail=0
for other in nofuse-noadvise fuse-noadvise nofuse-advise fuse-advise; do
  [ "$other" = "$ARM" ] && continue
  osha=$(git -C "$SRC" rev-parse -q --verify "$NS/run/$other" 2>/dev/null) || continue
  if git -C "$CELL" rev-parse -q --verify "$osha^{commit}" >/dev/null 2>&1; then
    echo "  $other ${osha:0:11}  REACHABLE  <-- ISOLATION BROKEN"; fail=1
  else
    echo "  $other ${osha:0:11}  unreachable"
  fi
done
# the specific commit from B34
if git -C "$CELL" rev-parse -q --verify 9c405211e7f >/dev/null 2>&1; then
  echo "  B34 commit 9c405211e7f  REACHABLE  <-- ISOLATION BROKEN"; fail=1
else
  echo "  B34 commit 9c405211e7f  unreachable"
fi
[ "$fail" -eq 0 ] || die "isolation check failed -- cell left in place at $CELL for inspection"

# ---------------------------------------------------------------- 5. activate
if [ "$ACTIVATE" -eq 1 ]; then
  if [ -d "$CANON" ] && [ ! -L "$CANON" ]; then
    die "$CANON is a real directory. Move it to $ADMIN first (mv is instant, same filesystem):
    mkdir -p $ROOT/admin && mv $CANON $ADMIN
  Then rerun with --activate. Refusing to move 24G of someone else's checkout unasked."
  fi
  ln -sfn "$CELL" "$CANON"
  say "activated: $CANON -> $CELL"
  say "ttnn will now import from this cell (python_env .pth resolves through $CANON)"
else
  echo
  echo "cell ready (NOT activated): $CELL"
  echo "  activate with: ln -sfn $CELL $CANON   (or rerun with --activate)"
fi
