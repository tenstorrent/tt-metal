ifeq ($(ARCH_NAME),$(filter $(ARCH_NAME),wormhole wormhole_b0))
ERISC_MAKE = TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C src/firmware/riscv/targets/erisc
ERISC_MAKE_CLEAN = TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C src/firmware/riscv/targets/erisc clean
else
ERISC_MAKE = @echo 'Skipping Erisc build for Grayskull.'
ERISC_MAKE_CLEAN = @echo 'Skipping Erisc clean for Grayskull.'
endif

src/firmware:
	TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C $(TT_METAL_HOME)/tt_metal/src/firmware/riscv/targets/brisc linker_scripts
	TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C $(TT_METAL_HOME)/tt_metal/src/firmware/riscv/targets/ncrisc linker_scripts

src/firmware/clean:
	TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C $(TT_METAL_HOME)/tt_metal/src/firmware/riscv/targets/brisc clean
	TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C $(TT_METAL_HOME)/tt_metal/src/firmware/riscv/targets/ncrisc clean
	$(ERISC_MAKE_CLEAN)
