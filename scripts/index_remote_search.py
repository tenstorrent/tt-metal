#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Crawl the built docs in output/ and push them to the OpenSearch-backed
docs search service so the in-page search modal has something to query.

Configured entirely through environment variables (see main()); intended to run
as a step in the Pages deploy workflow after `python build_docs.py`."""

from __future__ import annotations

import html
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

# Keep batches small: the indexer Lambda fans each batch out to an OpenSearch
# _bulk call, and API Gateway caps the request at ~29s. 200 docs in one POST
# times out (HTTP 504) on the larger catalogs; 50 stays comfortably under it.
MAX_BATCH_SIZE = 50
TIMEOUT_SECONDS = 30


def _strip_html_to_text(content: str) -> str:
    # Remove script/style blocks first so they do not pollute search text.
    content = re.sub(r"<script\b[^>]*>.*?</script>", " ", content, flags=re.IGNORECASE | re.DOTALL)
    content = re.sub(r"<style\b[^>]*>.*?</style>", " ", content, flags=re.IGNORECASE | re.DOTALL)
    # Strip tags.
    content = re.sub(r"<[^>]+>", " ", content)
    # Decode entities and normalize whitespace.
    content = html.unescape(content)
    content = re.sub(r"\s+", " ", content).strip()
    return content


def _extract_title(content: str, fallback: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", content, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return fallback
    return _strip_html_to_text(match.group(1)) or fallback


def _extract_article_body(content: str) -> str:
    """Return just the main article region of a Sphinx/RTD page.

    The RTD theme wraps the real page content in an element carrying
    ``itemprop="articleBody"``; everything else (sidebar, version selector,
    top/side navigation, footer) is boilerplate chrome that repeats on every
    page. Indexing the whole document would put that chrome into every body and
    make unrelated pages match on navigation terms, distorting relevance.

    Falls back to the full document when no article-body marker is present
    (e.g. custom landing pages), so nothing is silently dropped.
    """
    start_match = re.search(
        r"<(\w+)\b[^>]*\bitemprop\s*=\s*[\"']articleBody[\"'][^>]*>",
        content,
        flags=re.IGNORECASE,
    )
    if not start_match:
        return content

    tag = start_match.group(1)
    open_re = re.compile(r"<" + re.escape(tag) + r"\b", re.IGNORECASE)
    close_re = re.compile(r"</" + re.escape(tag) + r"\s*>", re.IGNORECASE)

    # Walk forward from the opening tag, tracking nesting of the same tag, to
    # find the matching close (handles nested <section>/<div> etc.).
    depth = 1
    pos = start_match.end()
    while depth > 0:
        next_close = close_re.search(content, pos)
        if not next_close:
            # Unbalanced markup: take everything after the opening tag.
            return content[start_match.end() :]
        next_open = open_re.search(content, pos)
        if next_open and next_open.start() < next_close.start():
            depth += 1
            pos = next_open.end()
        else:
            depth -= 1
            pos = next_close.end()
    return content[start_match.end() : next_close.start()]


def _iter_html_files(output_root: Path) -> Iterable[Path]:
    for path in output_root.rglob("*.html"):
        rel = path.relative_to(output_root).as_posix()
        if rel.startswith("_static/") or rel.startswith("_sources/"):
            continue
        if "/_static/" in rel or "/_sources/" in rel:
            continue
        # Skip Sphinx's own search/genindex shell pages — they carry no content.
        name = path.name
        if name in {"search.html", "genindex.html"}:
            continue
        yield path


def _build_documents(
    output_root: Path, site_base_url: str, catalog: str, version: str, id_namespace: str
) -> list[dict]:
    site_base_url = site_base_url.rstrip("/")
    docs: list[dict] = []

    for html_file in _iter_html_files(output_root):
        rel = html_file.relative_to(output_root).as_posix()
        raw = html_file.read_text(encoding="utf-8", errors="ignore")
        title = _extract_title(raw, rel)
        # Index only the article body, not the surrounding navigation chrome.
        body = _strip_html_to_text(_extract_article_body(raw))
        doc_id = f"{catalog}:{id_namespace}:{version}:{rel}"
        url = f"{site_base_url}/{rel}"
        docs.append({"id": doc_id, "title": title, "body": body, "url": url})

    return docs


def _chunks(items: list[dict], size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _post_json(url: str, payload: dict, api_key: str) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        method="POST",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            return response.getcode(), response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8", errors="replace")


def main() -> int:
    api_base = os.environ.get("SEARCH_API_BASE", "").rstrip("/")
    source_id = os.environ.get("DOC_CATALOG_SOURCE_ID", "").strip()
    api_key = os.environ.get("DOCS_SEARCH_INGEST_API_KEY", "").strip()
    output_dir = os.environ.get("DOCS_OUTPUT_DIR", "output").strip()
    site_base = os.environ.get("DOC_SITE_BASE_URL", "").rstrip("/")
    version = os.environ.get("DOCS_INDEX_VERSION", "latest").strip()
    id_namespace = os.environ.get("DOCS_ID_NAMESPACE", "tenstorrent").strip()

    if not api_base:
        print("Missing SEARCH_API_BASE", file=sys.stderr)
        return 2
    if not source_id:
        print("Missing DOC_CATALOG_SOURCE_ID", file=sys.stderr)
        return 2
    if not api_key:
        print("Missing DOCS_SEARCH_INGEST_API_KEY", file=sys.stderr)
        return 2
    if not site_base:
        print("Missing DOC_SITE_BASE_URL", file=sys.stderr)
        return 2
    if not id_namespace:
        print("Missing DOCS_ID_NAMESPACE", file=sys.stderr)
        return 2

    output_root = Path(output_dir)
    if not output_root.exists():
        print(f"Output directory not found: {output_root}", file=sys.stderr)
        return 2

    docs = _build_documents(output_root, site_base, source_id, version, id_namespace)
    if not docs:
        print("No HTML docs found to index.", file=sys.stderr)
        return 1

    endpoint = f"{api_base}/v1/index/{source_id}"
    print(f"Indexing {len(docs)} documents to {endpoint}")

    indexed = 0
    for batch in _chunks(docs, MAX_BATCH_SIZE):
        payload = {"version": version, "documents": batch}
        status, body = _post_json(endpoint, payload, api_key)
        if status < 200 or status >= 300:
            print(f"Indexing failed: HTTP {status}", file=sys.stderr)
            print(body, file=sys.stderr)
            if status == 403:
                print(
                    (
                        "Hint: DOCS_SEARCH_INGEST_API_KEY must be the API key VALUE, "
                        "not the API key ID (for example, not '427rdpxpb6')."
                    ),
                    file=sys.stderr,
                )
            return 1
        indexed += len(batch)
        print(f"Indexed {indexed}/{len(docs)}")

    print("Indexing complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
