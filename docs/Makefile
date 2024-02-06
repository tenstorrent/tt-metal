# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    	?= -W
SPHINXBUILD   	?= sphinx-build
SOURCEDIR     	= source
BUILDDIR      	= build
DOXYGENDIR    	= doxygen_build
HTMLDIR       	= $(BUILDDIR)/html
PORT          	?= 8888
DOCS_VERSION       	?= latest
#TODO: Update when folder created on web server with updated name
#PUBLIC_DOCDIR   = yyz-webservice-02:/var/www/html/docs/<PROJECT-NAME>
BRANCH = $(shell git rev-parse --abbrev-ref HEAD)

GITHUB_TOKEN    ?= INSERT_TOKEN_HERE
TTNN_SWEEPS_DIR = $(HTMLDIR)/ttnn_sweeps

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help default clean html publish

default: html

all: clean html server

clean:
	@rm -rf $(BUILDDIR)
	@rm -rf $(DOXYGENDIR)

html:
	@cd .. && doxygen
	@DOCS_VERSION=$(DOCS_VERSION) $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

ttnn_sweeps/check_directory:
	@if [ -d "$(TTNN_SWEEPS_DIR)" ]; then \
		echo "Error: ttnn sweeps dir $(TTNN_SWEEPS_DIR) exists already."; \
		exit 1; \
	else \
		mkdir -p $(TTNN_SWEEPS_DIR); \
	fi

ttnn_sweeps:
	@echo "Note that GITHUB_TOKEN must be set before calling this"
	@cd .. && python tests/ttnn/sweep_tests/build_html_sweep_results.py --dir docs/$(TTNN_SWEEPS_DIR) --token $(GITHUB_TOKEN)

server:
	@echo "Navigate to: \033[4;33mlocalhost:$(PORT)/index.html\033[0m"
	@cd $(HTMLDIR) && python -m http.server $(PORT)

# This will need to updated to
# - Use proper branch name for main
# - Upload doxygen XML assets with build
publish:
ifeq ($(BRANCH), master)
	@echo "Publishing docs to server"
	@chmod -R g+w $(HTMLDIR)/*
	@rsync --delete-before --delete-excluded --omit-dir-times -avz $(HTMLDIR)/* $(PUBLIC_DOCDIR)
else
	@echo "Publishing docs to server failed, only supported on master branch"
	@echo "You are currently on branch: $(BRANCH)"
endif
