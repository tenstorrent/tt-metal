.PHONY: src/ckernels
src/ckernels:
	TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C $(TT_METAL_HOME)/tt_metal/src/ckernels/gen
	TT_METAL_HOME=$(TT_METAL_HOME) OUTPUT_DIR=$(TT_METAL_HOME)/build/src/ckernels/out LINKER_SCRIPT_NAME=trisc0.ld TEST=unused FIRMWARE_NAME=unused $(MAKE) -C $(TT_METAL_HOME)/tt_metal/src/ckernels linker_scripts
	TT_METAL_HOME=$(TT_METAL_HOME) OUTPUT_DIR=$(TT_METAL_HOME)/build/src/ckernels/out LINKER_SCRIPT_NAME=trisc1.ld TEST=unused FIRMWARE_NAME=unused $(MAKE) -C $(TT_METAL_HOME)/tt_metal/src/ckernels linker_scripts
	TT_METAL_HOME=$(TT_METAL_HOME) OUTPUT_DIR=$(TT_METAL_HOME)/build/src/ckernels/out LINKER_SCRIPT_NAME=trisc2.ld TEST=unused FIRMWARE_NAME=unused $(MAKE) -C $(TT_METAL_HOME)/tt_metal/src/ckernels linker_scripts

src/ckernels/clean:
	TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C $(TT_METAL_HOME)/tt_metal/src/ckernels/gen clean
	TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C $(TT_METAL_HOME)/tt_metal/src/ckernels/gen clean
