OUTPUT_DIR=$(TT_METAL_HOME)/build/hw/toolchain

ifeq ("$(ARCH_NAME)", "wormhole_b0")
	DEV_MEM_MAP=$(TT_METAL_HOME)/tt_metal/hw/inc/wormhole/dev_mem_map.h
	INCLUDES = -I$(TT_METAL_HOME)/tt_metal/hw/inc/wormhole
else
	DEV_MEM_MAP=$(TT_METAL_HOME)/tt_metal/hw/inc/$(ARCH_NAME)/dev_mem_map.h
	INCLUDES = -I$(TT_METAL_HOME)/tt_metal/hw/inc/$(ARCH_NAME)
endif

hw/toolchain: $(OUTPUT_DIR)/idle-erisc.ld $(OUTPUT_DIR)/brisc.ld $(OUTPUT_DIR)/ncrisc.ld $(OUTPUT_DIR)/trisc0.ld $(OUTPUT_DIR)/trisc1.ld $(OUTPUT_DIR)/trisc2.ld

hw/toolchain/clean:
	rm -rf $(OUTPUT_DIR)

$(OUTPUT_DIR)/%.ld: toolchain/%.ld $(DEV_MAM_MAP) | $(OUTPUT_DIR)
	$(CXX) $(DEFINES) $(DEFS) $(INCLUDES) -E -P -x c -o $@ $<

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)
