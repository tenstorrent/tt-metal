#!/usr/bin/env bash
# Diagnostics only: every failure here is data, not a job failure, so no
# errexit and an unconditional exit 0 at the end.
set -uo pipefail

HF_MODEL="${WEIGHTS_HF_MODEL:-}"
HUB="${WEIGHTS_HUB_DIR:-${HF_HUB_CACHE:-${HOME:-/root}/.cache/huggingface/hub}}"
TT_CACHE="${WEIGHTS_TT_CACHE_HOME:-${TT_CACHE_HOME:-}}"
LABEL="${WEIGHTS_LABEL:-}"
SUFFIX=""
[ -n "$LABEL" ] && SUFFIX=" ($LABEL)"

MODEL_NAME="${HF_MODEL//\//--}"
REPO_DIR="$HUB/models--$MODEL_NAME"

# Every listing is capped and time-boxed: the hub is a shared network mount
# holding hundreds of repos, and a hung mount must yield a timeout line in the
# log rather than eat the job's whole timeout-minutes budget.
MAX_LINES=200
run() {
  echo "\$ $*"
  timeout --preserve-status 60 "$@" 2>&1 | head -n "$MAX_LINES"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && echo "(exit $rc)"
  echo
}

echo "::group::🔍 Weights state: context$SUFFIX"
run date -u
echo "hostname:        $(hostname)"
echo "id:              $(id)"
echo "HF_MODEL:        ${HF_MODEL:-<unset>}"
echo "cache dir name:  models--$MODEL_NAME"
echo "HF_HUB_CACHE:    $HUB"
echo "TT_CACHE_HOME:   ${TT_CACHE:-<unset>}"
echo "HF_HUB_OFFLINE:  ${HF_HUB_OFFLINE:-<unset>}"
echo "HF_HOME:         ${HF_HOME:-<unset>}"
echo "HF_TOKEN set:    $([ -n "${HF_TOKEN:-}" ] && echo yes || echo no)"
echo "::endgroup::"

echo "::group::🔍 Weights state: mounts$SUFFIX"
# The container bind-mounts the host's /mnt/MLPerf at create time. If the
# host's weka mount unit dropped, the bind resolves to the empty underlying
# host directory and the filesystem type below reads as the host rootfs
# (overlay/ext4) instead of wekafs. That mismatch is the signature of a
# missing-weights run whose cause is the mount, not the sync.
for d in /mnt /mnt/MLPerf /mnt/MLPerf/huggingface "$HUB"; do
  if [ -e "$d" ]; then
    echo "statfs $d: fstype=$(stat -f -c '%T' "$d" 2>&1) blocks=$(stat -f -c '%b' "$d" 2>&1) free=$(stat -f -c '%a' "$d" 2>&1)"
  else
    echo "statfs $d: MISSING"
  fi
done
echo
if command -v findmnt >/dev/null 2>&1; then
  run findmnt --real -o TARGET,SOURCE,FSTYPE,OPTIONS
else
  run cat /proc/self/mounts
fi
echo "--- mountinfo entries covering the weights paths ---"
grep -E ' (/mnt|/mnt/MLPerf|/mnt/models)[ /]' /proc/self/mountinfo 2>&1 | head -n 40
echo
run df -h /mnt /mnt/MLPerf "$HUB"
run df -i /mnt /mnt/MLPerf "$HUB"
echo "::endgroup::"

echo "::group::🔍 Weights state: directory tree above the cache$SUFFIX"
run ls -la /mnt
run ls -la /mnt/MLPerf
run ls -la /mnt/MLPerf/huggingface
echo "hub repo count: $(timeout --preserve-status 60 ls -1 "$HUB" 2>/dev/null | wc -l)"
run ls -la "$HUB"
echo "::endgroup::"

echo "::group::🔍 Weights state: models--$MODEL_NAME$SUFFIX"
if [ -z "$HF_MODEL" ]; then
  echo "no model supplied; skipping repo-level inspection"
elif [ ! -d "$REPO_DIR" ]; then
  echo "❌ $REPO_DIR does not exist"
  echo "closest names in the hub:"
  # A partial match tells the infra team whether the repo is absent entirely or
  # merely cached under a different org/casing than the workflow asks for.
  find "$HUB" -maxdepth 1 -printf '%f\n' 2>/dev/null | grep -i -- "${MODEL_NAME##*--}" | head -n 20
else
  run ls -la "$REPO_DIR"
  run ls -la "$REPO_DIR/refs"
  for ref in "$REPO_DIR"/refs/*; do
    [ -f "$ref" ] || continue
    rev="$(cat "$ref" 2>/dev/null)"
    echo "ref $(basename "$ref") -> $rev"
    # Byte-exact, because huggingface_hub takes the ref file verbatim: a stray
    # trailing newline makes it look for snapshots/<rev>\n and fail offline
    # while the tree below still looks perfectly healthy.
    echo "  raw bytes ($(wc -c < "$ref")): $(od -c "$ref" 2>/dev/null | head -n 1)"
    if [ -d "$REPO_DIR/snapshots/$rev" ]; then
      echo "  snapshot dir present"
    else
      echo "  ❌ snapshot dir $REPO_DIR/snapshots/$rev MISSING"
    fi
  done
  echo
  run ls -la "$REPO_DIR/snapshots"
  for snap in "$REPO_DIR"/snapshots/*; do
    [ -d "$snap" ] || continue
    echo "--- snapshot $(basename "$snap") ---"
    # Long listing on purpose: the symlink targets and sizes are the payload.
    # shellcheck disable=SC2012
    ls -la "$snap" 2>&1 | head -n 60
    echo "  files: $(find "$snap" -mindepth 1 2>/dev/null | wc -l)"
    # HF snapshots are symlinks into blobs/. Dangling links mean a partial or
    # interrupted sync: the metadata looks complete while the payload is gone.
    dangling="$(find "$snap" -xtype l 2>/dev/null)"
    if [ -n "$dangling" ]; then
      echo "  ❌ dangling symlinks: $(printf '%s\n' "$dangling" | wc -l)"
      printf '%s\n' "$dangling" | head -n 20
    else
      echo "  no dangling symlinks"
    fi
    echo
  done
  echo "blobs: $(find "$REPO_DIR/blobs" -mindepth 1 2>/dev/null | wc -l) files"
  run du -sh --apparent-size "$REPO_DIR"
  # .incomplete files are huggingface_hub's in-flight download markers; their
  # presence means a sync was cut off rather than never started.
  echo "--- .incomplete / .lock markers ---"
  find "$REPO_DIR" -name '*.incomplete' -o -name '*.lock' 2>/dev/null | head -n 20
  echo
  echo "--- read test on the largest blob (catches stale handles and EACCES) ---"
  biggest="$(find "$REPO_DIR/blobs" -type f -printf '%s %p\n' 2>/dev/null | sort -rn | head -n 1 | cut -d' ' -f2-)"
  if [ -n "$biggest" ]; then
    echo "reading first 1 MiB of $biggest"
    timeout --preserve-status 60 dd if="$biggest" of=/dev/null bs=1M count=1 2>&1 | tail -n 2
  else
    echo "no blobs to read"
  fi
fi
echo "::endgroup::"

echo "::group::🔍 Weights state: TT cache$SUFFIX"
if [ -n "$TT_CACHE" ]; then
  run ls -la "$TT_CACHE"
  run ls -la "$TT_CACHE/$MODEL_NAME"
else
  echo "TT_CACHE_HOME unset"
fi
echo "::endgroup::"

echo "::group::🔍 Weights state: huggingface_hub offline resolution$SUFFIX"
# The authoritative check: resolve exactly the way vLLM does, so the verdict
# here matches the server's own success or LocalEntryNotFoundError.
# -u so the traceback on stderr stays interleaved in order with stdout once
# both are merged into the pipe below. HF_HUB_CACHE is forced to the same dir
# the listings above walked, so the two halves cannot disagree.
HF_MODEL="$HF_MODEL" HF_HUB_CACHE="$HUB" timeout --preserve-status 120 python3 -u - <<'PY' 2>&1 | head -n 60
import os
import traceback

model = os.environ.get("HF_MODEL", "")
try:
    import huggingface_hub
    from huggingface_hub import constants, snapshot_download, try_to_load_from_cache
except Exception:
    traceback.print_exc()
    raise SystemExit(0)

print("huggingface_hub:", huggingface_hub.__version__)
print("constants.HF_HUB_CACHE:", constants.HF_HUB_CACHE)
print("constants.HF_HUB_OFFLINE:", constants.HF_HUB_OFFLINE)

if not model:
    raise SystemExit(0)

for filename in ("config.json", "tokenizer_config.json", "model.safetensors.index.json"):
    try:
        print(f"try_to_load_from_cache({filename}):", try_to_load_from_cache(model, filename))
    except Exception as exc:
        print(f"try_to_load_from_cache({filename}) raised:", repr(exc))

try:
    print("snapshot_download(local_files_only=True):", snapshot_download(model, local_files_only=True))
except Exception:
    print("snapshot_download(local_files_only=True) raised:")
    traceback.print_exc()
PY
echo "::endgroup::"

# One-line verdict plus a run annotation, so a scan of the run summary shows
# whether the weights were there without opening the log groups.
if [ -n "$HF_MODEL" ] && [ ! -d "$REPO_DIR" ]; then
  echo "::warning title=weights-missing::$HF_MODEL is not cached at $REPO_DIR on this runner. See the '🔍 Weights state' log groups for the mount and directory state."
  echo "❌ weights NOT present: $REPO_DIR"
else
  echo "✅ weights directory present: $REPO_DIR"
fi

exit 0
