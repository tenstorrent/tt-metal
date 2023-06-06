#include "dev_mem_map.h"
#include "tensix-memory.ld"

REGION_ALIAS("REGION_CODE", TRISC0_CODE)
REGION_ALIAS("REGION_DATA", TRISC0_LOCAL_DATA_MEM)
REGION_ALIAS("REGION_STACK", TRISC0_STACK_MEM)

__firmware_stack_size = MEM_TRISC0_STACK_SIZE;

#include "tensix-sections.ld"

_Z11kernel_initv = _etext;
