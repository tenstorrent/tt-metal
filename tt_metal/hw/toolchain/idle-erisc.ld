#include "dev_mem_map.h"
#include "idle-erisc-memory.ld"

REGION_ALIAS("REGION_CODE", IDLE_ERISC_CODE)
REGION_ALIAS("REGION_DATA", IDLE_ERISC_LOCAL_DATA_MEM)
REGION_ALIAS("REGION_STACK", IDLE_ERISC_STACK_MEM)

__firmware_stack_size = MEM_IERISC_STACK_SIZE;

#include "idle-erisc-sections.ld"

_Z11kernel_initv = _etext;
