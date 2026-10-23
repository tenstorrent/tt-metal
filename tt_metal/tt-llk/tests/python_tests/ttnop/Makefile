# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra


scan-%: scan.cpp tensix_isa.h elf32.h
	$(CXX) $(CXXFLAGS) -DARCH_$(shell echo $* | tr a-z A-Z) -o $@ scan.cpp

clean:
	rm -f scan-*

check:
	python3 tests/ttnop_check.py

check-hang:
	python3 tests/ttnop_check.py --hang

.PHONY: clean check check-hang
