#!/bin/bash

# Specify the two files to compare
cached_diff="infra/links.diff"
generated_diff="infra/links-generated.diff"

find . -type f -wholename "./infra/*" -prune -o -type f -wholename "./.git/*" -prune -o -type f -print | xargs -n 1 -I{} sed -i 's/tenstorrent-metal\/tt-metal/tenstorrent\/tt-metal/g' {}
find . -type f -wholename "./infra/*" -prune -o -type f -wholename "./.git/*" -prune -o -type f -print | xargs -n 1 -I{} sed -i 's/tenstorrent-metal\.github\.io\/tt-metal/tenstorrent\.github\.io\/tt-metal/g' {}

git diff > "$generated_diff"

if diff "$cached_diff" "$generated_diff"; then
    echo "No differences found."
else
    echo "Differences found."
    exit 1
fi
