#include "dev_mem_map.h"
#include "tensix-memory.ld"

REGION_ALIAS("REGION_CODE", TRISC2_CODE)
REGION_ALIAS("REGION_DATA", TRISC2_LOCAL_DATA_MEM)
REGION_ALIAS("REGION_STACK", TRISC2_STACK_MEM)

__firmware_stack_size = MEM_TRISC2_STACK_SIZE;

#include "tensix-sections.ld"

_Z11kernel_initv = _etext;
