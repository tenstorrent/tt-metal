# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0

# L2CPU firmware runtime + an application (default: examples/heartbeat).
#
#   make                      bh-irq, bh-poll, QEMU test images, resident blobs, PIC check
#   make qemu-test            header mirror test + QEMU tests (irq and poll)
#   make image PLATFORM=bh NOTIFY=irq [APP=dir|name] [BASE=0x...]   -> build/<platform>-<notify>[-test][-<app>]/fw.bin
#   make resident             build/resident/resident-{bh,qemu}.bin (written by the host before the release)
#   make pic-check            the flat image is position independent (two BASEs, same bytes)
#
# Variables: PLATFORM=bh|qemu  NOTIFY=irq|poll  TEST=0|1  APP=<dir with app.c, or a name: tools/l2cpu/<name>/fw or
#            examples/<name>>  BASE=<link address>
#            FLAVOUR=<build dir name>  CYCLES=<QEMU restart cycles>  FWVER=<0x...u, FW_VERSION override>  V=1

HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
L2CPU := $(abspath $(HERE)..)
INC := $(L2CPU)/include

CROSS = riscv64-unknown-elf-
CC = $(CROSS)gcc
OBJCOPY = $(CROSS)objcopy
OBJDUMP = $(CROSS)objdump
SIZE = $(CROSS)size
PY = python3
QEMU = qemu-system-riscv64
LOGDIR ?= $(HERE)build/logs

PLATFORM ?= qemu
NOTIFY ?= irq
TEST ?= 0
APP ?= $(L2CPU)/examples/heartbeat
APP_DIR := $(patsubst %/,%,$(dir $(abspath $(firstword $(wildcard $(APP)/app.c $(L2CPU)/$(APP)/fw/app.c $(L2CPU)/examples/$(APP)/app.c)))))
ifeq ($(APP_DIR),)
$(error APP=$(APP): no app.c found)
endif
# Optional application build fragment: APP_NAME (flavour suffix), APP_INC, APP_CFLAGS (app.c only), APP_VSRCS +
# APP_VFLAGS (sources built with their own flags, e.g. with V), APP_HDRS, APP_TEST_DRIVER (replaces test_driver.c).
-include $(APP_DIR)/app.mk
CYCLES ?= 1000
ifeq ($(PLATFORM),qemu)
BASE ?= 0x80200000
else
BASE ?= 0x400030200000
endif

FLAVOUR ?= $(PLATFORM)-$(NOTIFY)$(if $(filter 1,$(TEST)),-test)$(if $(APP_NAME),-$(APP_NAME))
B := $(HERE)build/$(FLAVOUR)
V ?= 0
Q := $(if $(filter 1,$(V)),,@)

ARCH := -mabi=lp64d -mcmodel=medany
COMMON := -O2 -g -ffreestanding -nostdlib -fno-builtin -fno-tree-loop-distribute-patterns -fno-common \
	-ffunction-sections -fdata-sections -fno-stack-protector -fno-pic -fno-jump-tables \
	-Wall -Wextra -Werror -Wno-unused-parameter
DEFS := -DL2CPU_TEST=$(TEST) -DL2CPU_NOTIFY_POLL=$(if $(filter poll,$(NOTIFY)),1,0) \
	-DL2CPU_RESTART_CYCLES=$(CYCLES)u $(if $(FWVER),-DL2CPU_FW_VERSION_OVERRIDE=$(FWVER))
ifeq ($(PLATFORM),bh)
DEFS += -DL2CPU_PLATFORM_BH=1 -DL2CPU_BH_HWINIT=1
PLAT_SRC := platform_bh.c
else
DEFS += -DL2CPU_PLATFORM_QEMU=1
PLAT_SRC := platform_qemu.c
endif
CFLAGS := -march=rv64gc $(ARCH) $(COMMON) $(DEFS) -I$(HERE) -I$(INC) $(APP_INC)
LDFLAGS := -nostdlib -nostartfiles -Wl,--gc-sections -Wl,--no-warn-rwx-segments -Wl,--defsym=BASE=$(BASE) -T $(HERE)link.ld

SRCS := main.c mailbox.c trap.c log.c libc.c $(PLAT_SRC) $(if $(filter 1,$(TEST)),$(if $(APP_TEST_DRIVER),,test_driver.c))
VOBJS := $(patsubst %.c,$(B)/v_%.o,$(notdir $(APP_VSRCS)))
OBJS := $(B)/start.o $(patsubst %.c,$(B)/%.o,$(SRCS)) $(B)/app.o $(VOBJS) \
	$(if $(and $(filter 1,$(TEST)),$(APP_TEST_DRIVER)),$(B)/app_test_driver.o)
HDRS := $(wildcard $(HERE)*.h) $(INC)/l2cpu_boot.h $(INC)/l2cpu_hw.h $(APP_HDRS)

.PHONY: FORCE all image clean qemu-test qemu-run mirror-test pic-check resident

all:
	$(MAKE) image PLATFORM=bh NOTIFY=irq
	$(MAKE) image PLATFORM=bh NOTIFY=poll
	$(MAKE) image PLATFORM=qemu NOTIFY=irq TEST=1
	$(MAKE) image PLATFORM=qemu NOTIFY=poll TEST=1
	$(MAKE) resident
	$(MAKE) pic-check
	@md5sum $(HERE)build/bh-irq/fw.bin $(HERE)build/bh-poll/fw.bin $(HERE)build/resident/resident-bh.bin

image: $(B)/fw.bin $(B)/fw.lst
	@$(SIZE) $(B)/fw.elf | tail -1 | awk '{printf "  %-16s text %7d data %5d bss %7d  -> $(B)/fw.bin\n", "$(FLAVOUR)", $$1, $$2, $$3}'

$(B):
	@mkdir -p $@

# rebuild a flavour when its flags change
FLAGS_TEXT := $(CFLAGS) | $(LDFLAGS) | $(APP_DIR)$(if $(APP_VSRCS), | $(APP_CFLAGS) | $(APP_VFLAGS))
$(B)/.flags: FORCE | $(B)
	@echo '$(FLAGS_TEXT)' | cmp -s - $@ || echo '$(FLAGS_TEXT)' > $@

$(B)/start.o: $(HERE)start.S $(HERE)fw_asm.h $(INC)/l2cpu_boot.h $(B)/.flags | $(B)
	@echo "  AS  $(FLAVOUR)/$(notdir $@)"
	$(Q)$(CC) -march=rv64gc $(ARCH) $(DEFS) -I$(HERE) -I$(INC) -c -o $@ $<

$(B)/%.o: $(HERE)%.c $(HDRS) $(B)/.flags | $(B)
	@echo "  CC  $(FLAVOUR)/$(notdir $@)"
	$(Q)$(CC) $(CFLAGS) -c -o $@ $<

$(B)/app.o: $(APP_DIR)/app.c $(HDRS) $(B)/.flags | $(B)
	@echo "  CC  $(FLAVOUR)/app.o ($(notdir $(APP_DIR)))"
	$(Q)$(CC) $(CFLAGS) $(APP_CFLAGS) -c -o $@ $<

$(B)/app_test_driver.o: $(APP_TEST_DRIVER) $(HDRS) $(B)/.flags | $(B)
	@echo "  CC  $(FLAVOUR)/app_test_driver.o"
	$(Q)$(CC) $(CFLAGS) $(APP_CFLAGS) -c -o $@ $<

define VRULE
$(B)/v_$(notdir $(1:.c=.o)): $(1) $(HDRS) $(B)/.flags | $(B)
	@echo "  CC  $(FLAVOUR)/v_$(notdir $(1:.c=.o)) (app, own flags)"
	$(Q)$(CC) $(APP_VFLAGS) -c -o $$@ $$<
endef
$(foreach f,$(APP_VSRCS),$(eval $(call VRULE,$(f))))

$(B)/fw.elf: $(OBJS) $(HERE)link.ld
	@echo "  LD  $(FLAVOUR)/fw.elf at BASE=$(BASE)"
	$(Q)$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJS) -lgcc

$(B)/fw.bin: $(B)/fw.elf
	$(Q)$(OBJCOPY) -O binary $< $@

$(B)/fw.lst: $(B)/fw.elf
	$(Q)$(OBJDUMP) -d -S $< > $@

pic-check:
	$(MAKE) image PLATFORM=bh NOTIFY=irq BASE=0x400030200000 B=$(HERE)build/pic-a >/dev/null
	$(MAKE) image PLATFORM=bh NOTIFY=irq BASE=0x400047a00000 B=$(HERE)build/pic-b >/dev/null
	cmp $(HERE)build/pic-a/fw.bin $(HERE)build/pic-b/fw.bin
	@echo "pic-check PASS: bh-irq image identical when linked at 0x400030200000 and 0x400047a00000"

# ---- resident page blobs ------------------------------------------------------------------------------------------
RES_DIR := $(HERE)build/resident
$(RES_DIR)/resident-%.bin: $(HERE)resident.S $(INC)/l2cpu_boot.h
	@mkdir -p $(RES_DIR)
	@echo "  AS  resident-$*"
	$(Q)$(CC) -march=rv64gc $(ARCH) -nostdlib -nostartfiles -Wl,-Ttext=0 -Wl,--no-warn-rwx-segments -I$(INC) \
		$(if $(filter bh,$*),-DL2CPU_PLATFORM_BH=1) -o $(RES_DIR)/resident-$*.elf $<
	$(Q)$(OBJCOPY) -O binary $(RES_DIR)/resident-$*.elf $@
	$(Q)$(OBJDUMP) -d $(RES_DIR)/resident-$*.elf > $(RES_DIR)/resident-$*.lst
	@test $$(stat -c %s $@) -eq $$(printf %d 0x2000) || { echo "resident blob size != L2CPU_RES_SIZE"; exit 1; }
resident: $(RES_DIR)/resident-bh.bin $(RES_DIR)/resident-qemu.bin

# ---- host mirror of l2cpu_boot.h ----------------------------------------------------------------------------------
mirror-test:
	$(PY) $(L2CPU)/tools/gen_boot_py.py --check
	$(PY) $(L2CPU)/tests/test_boot_mirror.py

# ---- QEMU -----------------------------------------------------------------------------------------------------------
QEMU_FLAGS = -M virt -cpu rv64,v=true,vlen=512,elen=64,vext_spec=v1.0,zba=true,zbb=true -smp 5 -m 1G \
	-bios none -nographic -serial stdio -monitor none
# -bios none makes the reset ROM jump to the DRAM base: the generic loader sets each hart's pc instead (the QEMU
# equivalent of the four reset vectors). The region prefix is preloaded with 0xA5 garbage (a DRAM buffer is not
# zeroed either) and the resident page is loaded where the host would write it.
QEMU_REGION = 0x80000000
QEMU_TIMEOUT ?= 1200
QEMU_RESIDENT = 0x803A0000
QEMU_BOOT = -device loader,file=$(HERE)build/garbage.bin,addr=$(QEMU_REGION),force-raw=on \
	-device loader,file=$(RES_DIR)/resident-qemu.bin,addr=$(QEMU_RESIDENT),force-raw=on \
	-device loader,file=$(1) $(foreach h,0 1 2 3 4,-device loader,addr=$(BASE),cpu-num=$(h))

$(HERE)build/garbage.bin:
	@mkdir -p $(dir $@)
	$(PY) -c "import sys; sys.stdout.buffer.write(bytes([0xA5]) * 0x200000)" > $@

qemu-run: $(HERE)build/garbage.bin resident
	$(MAKE) image PLATFORM=qemu NOTIFY=$(NOTIFY) TEST=1
	@mkdir -p $(LOGDIR)
	@log=$(LOGDIR)/qemu-$(NOTIFY)$(if $(APP_NAME),-$(APP_NAME)).log; \
	echo "running QEMU ($(NOTIFY)$(if $(APP_NAME), $(APP_NAME))), log $$log"; \
	timeout $(QEMU_TIMEOUT) $(QEMU) $(QEMU_FLAGS) $(call QEMU_BOOT,$(HERE)build/qemu-$(NOTIFY)-test$(if $(APP_NAME),-$(APP_NAME))/fw.elf) > $$log 2>&1; rc=$$?; \
	grep -v 'vector version is not specified' $$log | grep -E '^(PASS|FAIL|ALL)|TRAP|PARK' | head -80; \
	if [ $$rc -ne 0 ] || ! grep -q '^ALL PASS' $$log; then echo "qemu-test $(NOTIFY) FAILED (exit $$rc), see $$log"; exit 1; fi; \
	echo "qemu-test $(NOTIFY) PASS"

qemu-test: mirror-test
	$(MAKE) qemu-run NOTIFY=irq
	$(MAKE) qemu-run NOTIFY=poll

clean:
	rm -rf $(HERE)build
