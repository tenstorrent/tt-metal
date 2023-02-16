# CKTI - C Kernel Test Image

include $(BUDA_HOME)/common/common.mk

OUTPUT_DIR = out

INCLUDES = \
	-I$(CKERNELS_DIR)/$(ARCH_NAME)/common/inc \
	-I$(CKERNELS_DIR)/gen/out \
	-I$(FIRMWARE_RISCV_DIR)/common \
	-I$(REGSPECS_DIR)/inc \
	-I$(CMDASSEMBLER_INC) \
	-I$(BUDA_HOME)/src/software/ttl/include \
	-I$(BUDA_HOME)/vendor/tclap/include \
  -I$(BUDA_HOME)/vendor/tenstorrent-repositories/magic_enum/include \
  -I$(BUDA_HOME)/vendor/yaml-cpp/include \

TARGET = $(OUTPUT_DIR)/ckti
OBJS = $(OUTPUT_DIR)/ckti.o

LDFLAGS += -lboost_system

FINAL_LDFLAGS = -MMD -MP -g -std=c++17 -Werror $(CMDASSEMBLER_LDFLAGS) $(LDFLAGS)
FINAL_CXXFLAGS = -MMD -MP -g -std=c++17 -Werror $(INCLUDES) $(CXXFLAGS)


.PHONY: all
all : $(TARGET)

$(TARGET): $(OBJS) $(CMDASSEMBLER_LIBRARY) $(YAML_LIBRARY)
	@mkdir -p $(dir $@)
	$(CXX) -o $(TARGET) $^ $(FINAL_LDFLAGS)
	$(PRINT_SUCCESS)

$(OUTPUT_DIR)/ckti.o: ckti.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(FINAL_CXXFLAGS) -c $^ -o $@

.PHONY: clean
clean:
	@rm -Rf $(OUTPUT_DIR)

include $(BUDA_HOME)/infra/common_targets.mk
