# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
#
# On-chip token sampling on the L2CPU: the sampling library (lib/), the firmware application (fw/, built by the
# generic tools/l2cpu/fw/Makefile with APP=sampling) and the layout mirror.
#
#   make                  firmware images bh-irq-sampling, bh-poll-sampling (+ QEMU test images), host library
#   make test             mirror test + library host tests + firmware QEMU suites (irq, poll)
#   make lib-test         library: host unit tests, RISC-V build + nm check, QEMU bit-exactness corpus
#   make qemu-test        firmware QEMU suites only
#   make mirror-test      include/l2cpu_sampling.h <-> host/l2cpu/sampling/layout.py
#
# Variables: PY (python with numpy + pytest), SAMPLING_FAST=1|0 (library fast paths, default 1), ZB=1|0 (Zba + Zbb,
#            default 1), LOGDIR,
#            CORPUS_NPY (optional recorded rows for the library's QEMU corpus, see README)

HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
FW := $(abspath $(HERE)../fw)
PY ?= python3
LOGDIR ?= $(HERE)build/logs
SAMPLING_FAST ?= 1
ZB ?= 1
FWARGS := APP=sampling SAMPLING_FAST=$(SAMPLING_FAST) ZB=$(ZB) LOGDIR=$(LOGDIR)

.PHONY: all images lib test lib-test qemu-test mirror-test clean

all: images lib

images:
	$(MAKE) -C $(FW) image PLATFORM=bh NOTIFY=irq $(FWARGS)
	$(MAKE) -C $(FW) image PLATFORM=bh NOTIFY=poll $(FWARGS)
	$(MAKE) -C $(FW) image PLATFORM=qemu NOTIFY=irq TEST=1 $(FWARGS)
	$(MAKE) -C $(FW) image PLATFORM=qemu NOTIFY=poll TEST=1 $(FWARGS)
	$(MAKE) -C $(FW) resident
	@md5sum $(FW)/build/bh-irq-sampling/fw.bin $(FW)/build/bh-poll-sampling/fw.bin

lib:
	$(MAKE) -C $(HERE)lib PY=$(PY)

mirror-test:
	$(PY) $(HERE)tools/gen_sampling_py.py --check
	$(PY) $(HERE)tests/test_sampling_mirror.py

lib-test:
	$(MAKE) -C $(HERE)lib test riscv qemu-test PY=$(PY) ZB=$(ZB) LOGDIR=$(LOGDIR)

qemu-test: images
	$(MAKE) -C $(FW) qemu-run NOTIFY=irq $(FWARGS)
	$(MAKE) -C $(FW) qemu-run NOTIFY=poll $(FWARGS)

test: mirror-test lib-test qemu-test

clean:
	$(MAKE) -C $(HERE)lib clean
	rm -rf $(FW)/build/*-sampling $(HERE)build
