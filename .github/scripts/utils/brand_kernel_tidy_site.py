#!/usr/bin/env python3
"""Give the exported CodeChecker site a title and a favicon.

`CodeChecker parse --export html` titles every page "Plist HTML Viewer" and
references no icon, so a published report is indistinguishable from any other
tab and shows the browser's default globe. Neither is configurable, hence this
pass over the emitted HTML.

The export is flat -- index.html, statistics.html and every *.plist.html sit in
one directory -- so a relative icon href resolves from all of them.
"""

import argparse
import pathlib
import re
import sys

SITE_NAME = "tt-metal kernel clang-tidy"

# Magnifier on Tenstorrent purple. An SVG stays crisp at any tab density and is
# small enough to read in review, which a base64 data URI would not be.
FAVICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">
  <rect width="16" height="16" rx="3" fill="#7C68FA"/>
  <circle cx="7" cy="7" r="3.1" fill="none" stroke="#fff" stroke-width="1.6"/>
  <path d="M9.6 9.6 L12.4 12.4" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>
</svg>
"""

TITLE_RE = re.compile(r"<title>.*?</title>", re.DOTALL)
ICON_LINK = '<link rel="icon" href="favicon.svg">'

# CodeChecker names finding pages <leg>__reports__<source>_clang-tidy_<hash>.plist.html
SOURCE_RE = re.compile(r"__reports__(.+?)_clang-tidy_[0-9a-f]{32}\.plist\.html$")


def title_for(path: pathlib.Path) -> str:
    if path.name == "index.html":
        return SITE_NAME
    if path.name == "statistics.html":
        return f"Statistics \u00b7 {SITE_NAME}"
    m = SOURCE_RE.search(path.name)
    return f"{m.group(1)} \u00b7 {SITE_NAME}" if m else SITE_NAME


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("site", type=pathlib.Path, help="directory holding the exported HTML")
    args = ap.parse_args()

    if not args.site.is_dir():
        print(f"no such directory: {args.site}", file=sys.stderr)
        return 1

    (args.site / "favicon.svg").write_text(FAVICON)

    titled = linked = 0
    for page in args.site.glob("*.html"):
        html = page.read_text(errors="surrogateescape")
        html, n = TITLE_RE.subn(f"<title>{title_for(page)}</title>", html, count=1)
        titled += bool(n)
        if ICON_LINK not in html:
            # After the title so the icon is not separated from it; falls back to
            # <head> for a page that somehow carries no title at all.
            anchor = "</title>" if n else "<head>"
            html = html.replace(anchor, f"{anchor}\n    {ICON_LINK}", 1)
            linked += 1
        page.write_text(html, errors="surrogateescape")

    print(f"titled {titled} pages, linked favicon into {linked}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
