# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
#
# Build fragment of the sampling application, included by tools/l2cpu/fw/Makefile when APP names this directory.
#   SAMPLING_FAST=1|0  the sampling library's bf16 fast paths + fused copy/prepare (default 1; 0 = reference paths,
#                      same tokens)
#   RVV=1|0            the sampling library's RVV paths (default 1)
#   ZB=1|0             build the library and the copy with Zba + Zbb (the x280 executes both; default 1; 0 = rv64gcv)
#   L2CPU_LINK_INC     directory of l2cpu_link.h (default tools/l2cpu/tensix/kernels)
SAMPLING_FAST ?= 1
RVV ?= 1
ZB ?= 1
APP_NAME := sampling
APP_ROOT := $(abspath $(APP_DIR)/..)
L2CPU_LINK_INC ?= $(L2CPU)/tensix/kernels
APP_INC := -I$(APP_ROOT)/include -I$(APP_ROOT)/lib -I$(L2CPU_LINK_INC)
APP_CFLAGS := -ffp-contract=off -fno-fast-math -DL2S_SAMPLING_FAST=$(SAMPLING_FAST) -DX280S_RVV_BUILD=$(RVV)
# Built with V (their own flags): the sampling library and the bulk copy.
APP_VSRCS := $(APP_ROOT)/lib/x280s.c $(APP_DIR)/copy_rvv.c
APP_VFLAGS := -march=rv64gcv$(if $(filter 1,$(ZB)),_zba_zbb) -mabi=lp64d -mcmodel=medany -O2 -g -ffp-contract=off -fno-fast-math -ffreestanding \
	-fno-builtin -fno-tree-loop-distribute-patterns -fno-common -ffunction-sections -fdata-sections \
	-fno-stack-protector -fno-pic -std=c11 -Wall -Wextra -Werror $(if $(filter 1,$(RVV)),-DX280S_RVV) \
	$(if $(filter 1,$(SAMPLING_FAST)),,-DX280S_RVV_NOFAST) -I$(APP_ROOT)/lib
APP_HDRS := $(APP_ROOT)/include/l2cpu_sampling.h $(APP_ROOT)/lib/x280s.h $(L2CPU_LINK_INC)/l2cpu_link.h
# QEMU test flavour: the application's own test driver replaces the generic one.
APP_TEST_DRIVER := $(APP_DIR)/test_driver.c
