# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

scan: scan.cpp tensix_isa.h elf32.h
	$(CXX) $(CXXFLAGS) -o $@ scan.cpp

clean:
	rm -f scan

.PHONY: clean
