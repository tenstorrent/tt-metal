ifeq ($(ARCH_NAME),$(filter $(ARCH_NAME),wormhole wormhole_b0))
ERISC_MAKE = TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C hw/firmware/riscv/targets/erisc
ERISC_MAKE_CLEAN = TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C hw/firmware/riscv/targets/erisc clean
else
ERISC_MAKE = @echo 'Skipping Erisc build for Grayskull.'
ERISC_MAKE_CLEAN = @echo 'Skipping Erisc clean for Grayskull.'
endif

hw/firmware:
	TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C linker_scripts
	TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C linker_scripts
	$(info firmware/module.mk: Erisc build is skipped for wormhole_b0 currently)

hw/firmware/clean:
	$(info firmware/module.mk: Erisc build is skipped for wormhole_b0 currently)
