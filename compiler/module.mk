# Every variable in subdir must be prefixed with subdir (emulating a namespace)
COMPILER_INCLUDES = \
	-Icompiler \
	-Ithird_party/json \
	-I$(BUDA_HOME)

COMPILER_CFLAGS = $(CFLAGS) -Werror

include compiler/graph_lib/module.mk
include compiler/reportify/module.mk
include compiler/graph_deserializer/module.mk
include compiler/graph_utils/module.mk

compiler: compiler/graph_lib compiler/reportify compiler/graph_deserializer compiler/graph_utils
