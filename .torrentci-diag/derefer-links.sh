#!/bin/sh
# TEMPORARY (torrentci#2377 prototype): replace every symlink under $1 with a
# copy of its target; drop dangling links. The cache archive refuses links.
set -eu
find "$1" -type l | while IFS= read -r link; do
  target=$(readlink -f "$link" || true)
  rm -f "$link"
  if [ -n "$target" ] && [ -e "$target" ]; then cp -a "$target" "$link"; fi
done
echo "links left: $(find "$1" -type l | wc -l)"
