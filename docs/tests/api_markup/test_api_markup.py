# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Guards the contract between tt_theme.css and the HTML the docs build emits.

Two failures are worth catching before a release:

* the stylesheet stops matching the markup, so the API pages silently lose
  their design;
* a rule hides documented content that nothing puts back. That is what
  happened in #53397 — the CSS shipped with rules hiding the Parameters /
  Keyword Arguments / Returns fields and the supported-dtypes table, and the
  script meant to rebuild them was in a pull request that had not merged.

Run with:  pytest docs/tests/api_markup
Needs the pinned doc requirements (docs/requirements-docs.txt) plus lxml,
cssselect and tinycss2.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

tinycss2 = pytest.importorskip("tinycss2")
pytest.importorskip("cssselect")
from lxml import html as LH  # noqa: E402
from lxml.cssselect import CSSSelector  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixture"
STYLESHEET = HERE.parents[1] / "source" / "common" / "_static" / "tt_theme.css"

#: Selectors tt_theme.css relies on. If autodoc or Breathe change shape, or a
#: build-time transform stops running, these stop matching and the design is
#: quietly lost.
DESIGN_SELECTORS = [
    ".rst-content dl.py.function > dt.sig",
    ".rst-content dt.sig .sig-name",
    ".rst-content dl.py > dd > p:first-child",
    ".rst-content dl.field-list > dt",
    ".rst-content dl.field-list > dd > ul.simple > li",
    ".rst-content .tt-api-param-name",
    ".rst-content .tt-api-param-name em",
    ".rst-content .tt-api-param-desc",
    ".rst-content .tt-api-card-list",
    ".rst-content .tt-api-card-name",
    ".rst-content .admonition-example",
    ".rst-content .admonition-example > .admonition-title",
]


@pytest.fixture(scope="module")
def page(tmp_path_factory) -> LH.HtmlElement:
    out = tmp_path_factory.mktemp("html")
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-q", "-E", "-b", "html", str(FIXTURE), str(out)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    doc = LH.parse(str(out / "index.html")).getroot()
    body = doc.find("body")

    # The fixture builds without the site shell, so stand in for the two hooks
    # the deployed pages carry. Both matter: rules are written against
    # `.rst-content`, and several are guarded on `body:has(.tt-top-nav)` — a
    # guard that is always true in production, so the test has to model it or
    # it would pass on a stylesheet that blanks the live pages.
    if not doc.cssselect(".rst-content"):
        main = doc.cssselect("[role=main]")
        target = main[0] if main else body
        target.set("class", (target.get("class") or "") + " rst-content")
    if not doc.cssselect(".tt-top-nav"):
        shell = LH.Element("div")
        shell.set("class", "tt-top-nav")
        body.insert(0, shell)
    return doc


def _select(doc, selector: str):
    """Evaluate a selector, resolving a :has() guard cssselect cannot parse."""
    guard = re.match(r"^\s*\w*:has\(([^)]*)\)\s+(.*)$", selector)
    if guard:
        if not doc.cssselect(guard.group(1)):
            return []
        selector = guard.group(2)
    try:
        return CSSSelector(selector, translator="html")(doc)
    except Exception:
        return []


@pytest.mark.parametrize("selector", DESIGN_SELECTORS)
def test_stylesheet_still_matches_the_markup(page, selector):
    assert _select(page, selector), (
        f"{selector!r} matches nothing in the built page: the stylesheet and " f"the generated HTML have drifted apart."
    )


def _anchor_id(heading) -> str:
    """Resolve a heading to the anchor docs-toc.js will link it to.

    Mirrors the lookup order in ``docs-toc.js``: the heading's own id, then the
    id of the enclosing section, then the target of Sphinx's own permalink.
    """
    if heading.get("id"):
        return heading.get("id")
    for parent in heading.iterancestors():
        is_section = parent.tag == "section" or "section" in (parent.get("class") or "").split()
        if is_section and parent.get("id"):
            return parent.get("id")
    link = heading.cssselect("a.headerlink")
    if link and (link[0].get("href") or "").startswith("#"):
        return link[0].get("href")[1:]
    return ""


def test_every_heading_resolves_to_a_unique_anchor(page):
    """The right-hand TOC is only navigable while this holds.

    Sphinx puts the anchor on the enclosing ``<section>``, never on the ``h2``
    itself, and dedupes repeated titles there (``usage``, then ``id1``). If a
    heading had no resolvable anchor the entry would be dead, and if two
    resolved to the same value the second entry would jump to the first
    section — which is what deriving the id from the heading text produced.

    This locks the HTML side of the contract only. It cannot exercise the
    script, which would need a JS runtime in the docs test environment.
    """
    roots = page.cssselect("[role=main]") or page.cssselect(".rst-content")
    headings = [h for h in page.cssselect("h2, h3") if any(h is root or root in h.iterancestors() for root in roots)]
    assert headings, "fixture built no headings, so this test would prove nothing"

    #: The RTD theme renders the permalink as a Font Awesome glyph (U+F0C1);
    #: plain Sphinx uses a pilcrow. Neither belongs in a failure message.
    PERMALINK_GLYPHS = "\uf0c1\u00b6"

    def label(h) -> str:
        text = " ".join("".join(h.itertext()).split())
        return text.strip(PERMALINK_GLYPHS + " ")

    dead = [label(h) for h in headings if not _anchor_id(h)]
    assert not dead, f"headings with no anchor to link to: {dead}"

    by_anchor: dict[str, list[str]] = {}
    for h in headings:
        by_anchor.setdefault(_anchor_id(h), []).append(label(h))
    collisions = {a: names for a, names in by_anchor.items() if len(names) > 1}
    assert not collisions, f"several headings share one anchor, so the TOC cannot reach them all: {collisions}"

    repeated = {n for names in by_anchor.values() for n in names}
    assert len(repeated) < len(headings), (
        "the fixture no longer contains two sections with the same title, so "
        "the deduped-anchor path is not covered any more"
    )


def _hiding_selectors() -> list[str]:
    rules = tinycss2.parse_stylesheet(STYLESHEET.read_text(), skip_whitespace=True, skip_comments=True)
    out = []
    for rule in rules:
        if rule.type != "qualified-rule":
            continue
        if not re.search(r"display\s*:\s*none", tinycss2.serialize(rule.content)):
            continue
        for selector in tinycss2.serialize(rule.prelude).split(","):
            selector = selector.strip()
            if selector:
                out.append(selector)
    return out


def test_no_rule_hides_documented_content(page):
    """A decoration may be hidden; a paragraph of documentation may not."""
    roots = page.cssselect("[role=main]") or page.cssselect(".rst-content")

    def inside_article(element) -> bool:
        return any(element is root or root in element.iterancestors() for root in roots)

    def carries_prose(element) -> bool:
        return len("".join(element.itertext()).split()) >= 5

    offenders = {}
    for selector in _hiding_selectors():
        hidden = [e for e in _select(page, selector) if inside_article(e) and carries_prose(e)]
        if hidden:
            offenders[selector] = len(hidden)

    assert not offenders, (
        "these rules hide documented content, and nothing in the build puts " f"it back: {offenders}. See #53397."
    )
