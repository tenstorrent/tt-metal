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

# register_python_operation doesn't load the doc string when fast_runtime_mode is enabled.
# disable fast_runtime_mode to enable loading the doc string.
TTNN_FLAGS      = TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}'
GITHUB_TOKEN    ?= INSERT_TOKEN_HERE
TTNN_API_DIR    = source/ttnn/ttnn/api

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help default clean html publish build_doxygen sphinx_build_dir html/ttnn html/tt-metalium

default: html

all: clean html server

clean:
	@rm -rf $(BUILDDIR)
	@rm -rf $(DOXYGENDIR)
	@rm -rf $(TTNN_API_DIR)

build_doxygen: clean
	@cd .. && doxygen docs/Doxyfile

sphinx_build_dir: clean
	@mkdir -p $(BUILDDIR)

html/tt-metalium: build_doxygen sphinx_build_dir
	@DOCS_VERSION=$(DOCS_VERSION) REQUESTED_DOCS_PKG=tt-metalium $(SPHINXBUILD) "$(SOURCEDIR)/tt-metalium" "$(TT_METALIUM_BUILDDIR)" $(SPHINXOPTS) $(O)

html/ttnn: build_doxygen sphinx_build_dir
	@ ${TTNN_FLAGS} DOCS_VERSION=$(DOCS_VERSION) REQUESTED_DOCS_PKG=ttnn $(SPHINXBUILD) "$(SOURCEDIR)/ttnn" "$(TTNN_BUILDDIR)" $(SPHINXOPTS) $(O)

html: html/tt-metalium html/ttnn
	mkdir -p $(HTMLDIR)
	mv -f $(TT_METALIUM_BUILDDIR) $(HTMLDIR)/tt-metalium
	mv -f $(TTNN_BUILDDIR) $(HTMLDIR)/ttnn
	cp source/index.html $(HTMLDIR)/


server:
	@echo "Navigate to: \033[4;33mlocalhost:$(PORT)/index.html\033[0m"
	@cd $(HTMLDIR) && python -m http.server $(PORT)
