#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Regenerate every .svg and .png in this directory from its .d2 source.
#
# Install the two tools it needs -- neither is a dependency of anything else here:
#
#   curl -fsSL https://d2lang.com/install.sh | sh -s --
#   apt-get install -y librsvg2-bin
#
# Why two tools. d2 renders SVG by itself, but its PNG encoder drives a headless
# browser through Playwright, and that path is dead: d2 v0.7.1 pins
# playwright-go v0.4702.0, which hardcodes driver version 1.47.2 and the retired
# playwright.azureedge.net mirrors. The successor host
# (playwright.download.prss.microsoft.com) currently returns HTTP 400 for driver
# zips of any version -- see upstream playwright issues 38273, 38967 and 40084 -- and the
# one live alternative mirror has aged 1.47.2 out. PLAYWRIGHT_DOWNLOAD_HOST is
# read by that version but cannot help, because the version itself is a
# compile-time constant. So librsvg rasterises the SVG that d2 already produced.
#
# Both outputs are committed: the SVG is the reviewable text artifact, the PNG is
# what survives export into slides and issue comments.
#
# Pinned to the elk layout engine and d2's default theme: a re-render produces a
# diff only when a source actually changed.
set -euo pipefail

cd "$(dirname "$0")"
command -v d2 >/dev/null || { echo "d2 not found -- see the header of this script" >&2; exit 1; }
command -v rsvg-convert >/dev/null || { echo "rsvg-convert not found -- see the header" >&2; exit 1; }

# 2x native for legible text, capped so no PNG runs away.
MAX_W=2000

for src in *.d2; do
    base="${src%.d2}"
    d2 --layout=elk --theme=0 --pad=40 "$src" "$base.svg" >/dev/null

    native=$(sed -n 's/.*viewBox="0 0 \([0-9]*\).*/\1/p' "$base.svg" | head -1)
    want=$((native * 2))
    [ "$want" -gt "$MAX_W" ] && want=$MAX_W
    rsvg-convert -w "$want" -o "$base.png" "$base.svg"

    printf '%-24s -> %-24s %s\n' "$src" "$base.svg" "$(file -b "$base.png" | cut -d, -f2 | tr -d ' ')"
done
