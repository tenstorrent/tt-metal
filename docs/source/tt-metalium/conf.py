# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "TT-Metalium"
copyright = "Tenstorrent"
author = "Tenstorrent"

_docs_version = os.environ.get("DOCS_VERSION", "latest")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_parser",
]

try:
    import breathe
    extensions.append("breathe")
except ImportError:
    pass

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_notes = True
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ["_templates", "../common/_templates"]
exclude_patterns = []

import subprocess as _sp

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
html_baseurl = f"{_METAL_BASE}tt-metalium/"
html_static_path = ["_static", "../common/_static"]

html_css_files = [_GLOBAL_CSS]

html_context = {
    "logo_link_url": "https://docs.tenstorrent.com/",
}


def setup(app):
    app.add_css_file("tt_theme.css")
    app.add_js_file("api_style.js")


breathe_projects = {"ttmetaldoxygen": "../../doxygen_build/xml/"}
breathe_default_project = "ttmetaldoxygen"
