ifeq ($(ARCH_NAME),$(filter $(ARCH_NAME),wormhole wormhole_b0))
ERISC_MAKE = BUDA_HOME=$(BUDA_HOME) $(MAKE) -C src/firmware/riscv/targets/erisc
ERISC_MAKE_CLEAN = BUDA_HOME=$(BUDA_HOME) $(MAKE) -C src/firmware/riscv/targets/erisc clean
else
ERISC_MAKE = @echo 'Skipping Erisc build for Grayskull.'
ERISC_MAKE_CLEAN = @echo 'Skipping Erisc clean for Grayskull.'
endif

.PHONY: src/firmware
src/firmware: $(BUDA_HOME)/src/ckernels
	BUDA_HOME=$(BUDA_HOME) $(MAKE) -C src/firmware/riscv/targets/brisc
	BUDA_HOME=$(BUDA_HOME) $(MAKE) -C src/firmware/riscv/targets/ncrisc
	$(ERISC_MAKE)

src/firmware/clean:
	BUDA_HOME=$(BUDA_HOME) $(MAKE) -C src/firmware/riscv/targets/brisc clean
	BUDA_HOME=$(BUDA_HOME) $(MAKE) -C src/firmware/riscv/targets/ncrisc clean
	$(ERISC_MAKE_CLEAN)
