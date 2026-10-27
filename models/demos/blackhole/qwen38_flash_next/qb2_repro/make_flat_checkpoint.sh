#!/usr/bin/env bash
# Build the flat checkpoint dir the server needs (it rejects symlinked files): hard-link every file of an HF snapshot.
#   make_flat_checkpoint.sh SRC_SNAPSHOT_DIR DEST_DIR
# SRC is e.g. ~/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/<rev>. DEST must be on the same filesystem.
set -euo pipefail
[[ $# -eq 2 ]] || { sed -n '2,4p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
src=$(cd -- "$1" && pwd -P) || exit 2
mkdir -p -- "$2"
dest=$(cd -- "$2" && pwd -P)
[[ "$src" != "$dest" ]] || { echo "SRC and DEST are the same directory" >&2; exit 2; }

first=$(find -L "$src" -type f -print -quit)
[[ -n "$first" ]] || { echo "no files under $src" >&2; exit 2; }
if [[ "$(stat -L -c %d -- "$first")" != "$(stat -c %d -- "$dest")" ]]; then
    echo "ERROR: $src and $dest are on different filesystems; hard links are impossible." >&2
    echo "Copy instead:  cp -rL \"$src/.\" \"$dest/\"   (needs the full checkpoint size again, about 336 GiB)" >&2
    exit 1
fi

n=0
while IFS= read -r -d '' f; do
    rel=${f#"$src"/}
    mkdir -p -- "$dest/$(dirname -- "$rel")"
    if [[ ! -e "$dest/$rel" ]]; then
        ln -- "$(readlink -f -- "$f")" "$dest/$rel"
    fi
    n=$((n + 1))
done < <(find -L "$src" -type f -print0)

links=$(find "$dest" -type l | wc -l)
if [[ "$links" != 0 ]]; then
    echo "ERROR: $links symlinks remain under $dest" >&2
    exit 1
fi
echo "OK: $n files hard-linked into $dest, no symlinks. Total size: $(du -shL -- "$dest" | cut -f1)"
echo "Use:  export HF_MODEL=$dest"
