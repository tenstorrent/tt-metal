# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Minimal Sphinx project exercising every API markup shape the theme styles.

It deliberately does not import ttnn: the point is to pin the *shape* of the
HTML autodoc, Breathe and autosummary produce, which is what tt_theme.css and
tt_api_build.py are written against.
"""
import os
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
DOCS_SOURCE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "source"))

sys.path.insert(0, HERE)
sys.path.append(os.path.join(DOCS_SOURCE, "common", "_ext"))

project = "api-markup-fixture"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "tt_api_build",
]

# Kept in step with docs/source/*/conf.py.
copybutton_selector = "div.highlight pre, .rst-content dl.py > dt.sig, .rst-content dl.cpp > dt.sig"
copybutton_exclude = ".linenos, .gp, .headerlink"
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False
autosummary_generate = False

html_theme = "sphinx_rtd_theme"
html_show_sourcelink = False
