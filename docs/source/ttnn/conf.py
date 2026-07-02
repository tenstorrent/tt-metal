# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import collections
from pathlib import Path

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../common"))
sys.path.append(os.path.abspath("./_ext"))

from docs_versions import get_published_versions

project = "TT-NN"
copyright = "Tenstorrent"
author = "Tenstorrent"

_docs_version = os.environ.get("DOCS_VERSION", "latest")

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_parser",
]

# sphinx crashes when trying to introspect real ttnn (C extension issues in main process).
# Mock ttnn so pages are accessible, even with empty stubs.
# TODO: build ttnn from source in separate environment to get real docs.
autodoc_mock_imports = ["ttnn"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_notes = True
napoleon_use_param = True
napoleon_use_rtype = False

templates_path = ["_templates", "../common/_templates"]
exclude_patterns = [
    "**/CMakeLists.txt",
    "**/tutorials-dev.txt",
    "**/tutorials_venv.sh",
    "**/tutorials_env/**",
]

_METAL_BASE = "https://docs.tenstorrent.com/tt-metal/latest/"
_GLOBAL_CSS = "https://docs.tenstorrent.com/_static/tt_theme.css"

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "titles_only": True,
    "navigation_depth": 2,
}
html_logo = "../common/images/tt_logo.svg"
html_favicon = "../common/images/favicon.png"
# Single-version site: published at a flat path, no version switcher.
html_baseurl = f"{_METAL_BASE}ttnn/"
html_static_path = ["_static", "../common/_static"]

_docs_site_base = os.environ.get("DOC_SITE_BASE_URL", "https://docs.tenstorrent.com/tt-metal/latest").rstrip("/")

html_css_files = [_GLOBAL_CSS]

html_context = {
    "logo_link_url": "https://docs.tenstorrent.com/",
    "docs_site_base": _docs_site_base,
    "docs_project_subpath": "ttnn",
}

nbsphinx_execute = "never"


def setup(app):
    app.add_css_file("tt_theme.css")
    app.add_js_file("api_style.js")


breathe_projects = {"ttmetaldoxygen": "../../doxygen_build/xml/"}
breathe_default_project = "ttmetaldoxygen"
