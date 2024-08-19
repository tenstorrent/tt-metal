# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    	?= -W
SPHINXBUILD   	?= sphinx-build
SOURCEDIR     	= source/$(REQUESTED_DOCS_PKG)
BUILDDIR      	= build
DOXYGENDIR    	= doxygen_build
HTMLDIR       	= $(BUILDDIR)/html
TT_METALIUM_BUILDDIR	= $(BUILDDIR)/tt-metalium
TTNN_BUILDDIR	= $(BUILDDIR)/ttnn
PORT          	?= 8888
DOCS_VERSION       	?= latest

GITHUB_TOKEN    ?= INSERT_TOKEN_HERE
TTNN_SWEEPS_DIR = source/ttnn/ttnn_sweeps
TTNN_API_DIR    = source/ttnn/ttnn/api

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help default clean html publish build_doxygen sphinx_build_dir html/ttnn html/tt-metalium ttnn_sweeps

default: html

all: clean html server

clean:
	@rm -rf $(BUILDDIR)
	@rm -rf $(DOXYGENDIR)
	@rm -rf $(TTNN_API_DIR)

build_doxygen: clean
	@cd .. && doxygen

sphinx_build_dir: clean
	@mkdir -p $(BUILDDIR)

html/tt-metalium: build_doxygen sphinx_build_dir
	@DOCS_VERSION=$(DOCS_VERSION) REQUESTED_DOCS_PKG=tt-metalium $(SPHINXBUILD) "$(SOURCEDIR)/tt-metalium" "$(TT_METALIUM_BUILDDIR)"

html/ttnn: build_doxygen sphinx_build_dir
	@DOCS_VERSION=$(DOCS_VERSION) REQUESTED_DOCS_PKG=ttnn $(SPHINXBUILD) "$(SOURCEDIR)/ttnn" "$(TTNN_BUILDDIR)"

html: html/tt-metalium html/ttnn
	mkdir -p $(HTMLDIR)
	mv -f $(TT_METALIUM_BUILDDIR) $(HTMLDIR)/tt-metalium
	mv -f $(TTNN_BUILDDIR) $(HTMLDIR)/ttnn
	cp source/index.html $(HTMLDIR)/

ttnn_sweeps:
	@echo "Note that GITHUB_TOKEN must be set before calling this"
	@cd .. && python tests/ttnn/sweep_tests/build_rst_sweep_results.py --dir docs/$(TTNN_SWEEPS_DIR) --token $(GITHUB_TOKEN)

server:
	@echo "Navigate to: \033[4;33mlocalhost:$(PORT)/index.html\033[0m"
	@cd $(HTMLDIR) && python -m http.server $(PORT)
