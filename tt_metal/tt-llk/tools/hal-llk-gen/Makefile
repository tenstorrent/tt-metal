# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Thin wrapper around ./hal-llk-gen (see README.md). Examples:
#   make check
#   make run NAME=9a
#   make run NAME=9 REPEAT=3 DETACH=1
#   make run NAME=9b PROFILE=hal-test-gen STAGES="harness postcheck perf static dyn"
#   make compare NAMES="9-1 9-2 9-3"

TOOL    := ./hal-llk-gen
PROFILE ?= run8
NAME    ?=
NAMES   ?= $(NAME)
STAGES  ?=
REPEAT  ?=
DETACH  ?=

OPTS := --profile $(PROFILE) $(if $(STAGES),--stages "$(STAGES)") $(if $(REPEAT),--repeat $(REPEAT)) $(if $(DETACH),--detach)

.PHONY: help check dry-run run continue trivial follow status stop summary compare profiles selftest

help:
	@$(TOOL) help

check:
	@$(TOOL) check $(OPTS)

dry-run:
	@$(TOOL) dry-run $(NAME) $(OPTS)

run:
	@test -n "$(NAMES)" || { echo "usage: make run NAME=<name> [PROFILE=run8] [STAGES=\"...\"] [REPEAT=N] [DETACH=1]"; exit 2; }
	@$(TOOL) run $(NAMES) $(OPTS)

continue:
	@test -n "$(NAME)" -a -n "$(STAGES)" || { echo "usage: make continue NAME=<name> STAGES=\"perf static\""; exit 2; }
	@$(TOOL) continue $(NAME) $(OPTS)

trivial:
	@$(TOOL) trivial $(NAME) $(OPTS)

follow:
	@$(TOOL) follow $(NAME)

status:
	@$(TOOL) status $(NAME)

stop:
	@$(TOOL) stop $(NAME)

summary:
	@$(TOOL) summary $(NAME)

compare:
	@$(TOOL) compare $(if $(NAMES),$(NAMES),--all)

profiles:
	@$(TOOL) profiles

selftest:
	@$(TOOL) selftest
