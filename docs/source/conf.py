# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import collections

sys.path.insert(0, os.path.abspath("."))

MetalSphinxConfig = collections.namedtuple("MetalSphinxConfig", ["fullname", "shortname"])

config_lookup = {
    "tt-metalium": MetalSphinxConfig(fullname="TT-Metalium", shortname="tt-metalium"),
    "ttnn": MetalSphinxConfig(fullname="TT-NN", shortname="ttnn"),
}

if "REQUESTED_DOCS_PKG" not in os.environ:
    raise Exception("REQUESTED_DOCS_PKG needs to be supplied, either tt-metalium or ttnn")

metal_sphinx_config = config_lookup[os.environ["REQUESTED_DOCS_PKG"]]

# -- Project information -----------------------------------------------------

project = metal_sphinx_config.fullname
copyright = "Tenstorrent"
author = "Tenstorrent"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.email",
    "sphinx.ext.mathjax",
    "breathe",
    "myst_parser",
]

# For markdown and RST files
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Email settings
email_automode = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_logo = "images/tt_logo.svg"
html_favicon = "images/cropped-favicon-32x32.png"
html_baseurl = f"/{metal_sphinx_config.shortname}/" + os.environ["DOCS_VERSION"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


def setup(app):
    app.add_css_file("tt_theme.css")


# Breathe configs
breathe_projects = {"ttmetaldoxygen": "../../doxygen_build/xml/"}
breathe_default_project = "ttmetaldoxygen"
