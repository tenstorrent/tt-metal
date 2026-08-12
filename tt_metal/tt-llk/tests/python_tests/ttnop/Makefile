# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

scan: scan.cpp tensix_isa.h elf32.h
	$(CXX) $(CXXFLAGS) -o $@ scan.cpp

clean:
	rm -f scan

.PHONY: clean

# --- Metal backend (TTNOP_METAL=1) -------------------------------------------
# None of this is a prerequisite of `scan`: the LLK sweep must keep building with
# no tt-metal checkout in sight.

TT_METAL_HOME ?= $(abspath ../../../../..)
TT_METAL_CACHE ?= $(if $(wildcard $(HOME)),$(HOME)/.cache/tt-metal-cache,/tmp/tt-metal-cache)
# The build_Release/lib/ copy is often left stale after a rebuild.
TT_METAL_LIB ?= $(TT_METAL_HOME)/build_Release/tt_metal

.PHONY: metal_shim
metal_shim: libttnop_metal.so

# Only tt_memory.h is needed, so keep the include root thin rather than pulling
# in the rest of metal's header tree.
libttnop_metal.so: metal_shim.cpp
	$(CXX) -std=c++20 -O2 -Wall -Wextra -fPIC -shared -I$(TT_METAL_HOME)/tt_metal \
		-o $@ metal_shim.cpp -L$(TT_METAL_LIB) -ltt_metal -Wl,-rpath,$(TT_METAL_LIB)

# 512B = 128 words against the 102 the cave needs (100 fillers + the displaced
# instruction + the jump back).
CAVE_BYTES ?= 512
CAVE_DEFINE ?=

.PHONY: metal_cave metal_cave_disable _cave_linkers
metal_cave: CAVE_DEFINE = -DTTNOP_CAVE_BYTES=$(CAVE_BYTES)
metal_cave: _cave_linkers
metal_cave_disable: _cave_linkers

# Mirrors the add_custom_command in tt_metal/hw/CMakeLists.txt that preprocesses
# main.ld — same flags, same outputs — so only the cave define differs. Keep in
# step with it. Rebuilding hw_toolchain reverts these, so re-run after a build.
_cave_linkers:
	@for arch in wormhole blackhole; do \
		out="$(TT_METAL_HOME)/runtime/hw/toolchain/$$arch"; mkdir -p "$$out"; \
		for trisc in 0 1 2; do \
			$(CXX) -DTYPE_KERNEL -DCOMPILE_FOR_TRISC=$$trisc \
				-DARCH_$$(echo $$arch | tr a-z A-Z) $(CAVE_DEFINE) \
				-I"$(TT_METAL_HOME)/tt_metal/hw/inc/internal/tt-1xx/$$arch" \
				-E -P -x c -o "$$out/kernel_trisc$$trisc.ld" \
				"$(TT_METAL_HOME)/tt_metal/hw/toolchain/main.ld" || exit $$?; \
		done; \
	done
	@# KernelCompileHash is build_key + hlk_desc + compute_hash, so it cannot see a
	@# linker change: kernels cached under the other setting would be reused silently.
	@rm -rf $(TT_METAL_CACHE)/*/kernels
	@echo "ttnop: cave $(if $(CAVE_DEFINE),$(CAVE_BYTES)B,OFF); kernel cache dropped"
