#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Regenerate every .png in this directory from its .d2 source.
#
# Needs two tools, neither a dependency of anything else here:
#
#   d2            https://d2lang.com -- a single static Go binary
#   rsvg-convert  apt-get install librsvg2-bin
#
# d2 can emit PNG directly, but only by driving a headless browser through
# Playwright, whose download endpoints now 404. So d2 renders SVG (to a temp
# file, since only the PNGs are committed here) and librsvg rasterises it.
#
# Pinned to the elk layout engine and d2's default theme: a re-render produces
# a diff only when a source actually changed.
set -euo pipefail

cd "$(dirname "$0")"
command -v d2 >/dev/null || { echo "d2 not found -- see the header of this script" >&2; exit 1; }
command -v rsvg-convert >/dev/null || { echo "rsvg-convert not found -- see the header" >&2; exit 1; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# 2x native for legible text, capped so no PNG runs away.
MAX_W=2000

for src in *.d2; do
    base="${src%.d2}"
    d2 --layout=elk --theme=0 --pad=40 "$src" "$tmp/$base.svg" >/dev/null

    native=$(sed -n 's/.*viewBox="0 0 \([0-9]*\).*/\1/p' "$tmp/$base.svg" | head -1)
    want=$((native * 2))
    [ "$want" -gt "$MAX_W" ] && want=$MAX_W
    rsvg-convert -w "$want" -o "$base.png" "$tmp/$base.svg"

    printf '%-24s -> %s  %s\n' "$src" "$base.png" "$(file -b "$base.png" | cut -d, -f2 | tr -d ' ')"
done
