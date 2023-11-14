#include "dev_mem_map.h"
#include "tensix-memory.ld"

REGION_ALIAS("REGION_CODE", TRISC1_CODE)
REGION_ALIAS("REGION_DATA", TRISC1_LOCAL_DATA_MEM)
REGION_ALIAS("REGION_STACK", TRISC1_STACK_MEM)

__firmware_stack_size = MEM_TRISC1_STACK_SIZE;

#include "tensix-sections.ld"

_Z11kernel_initv = _etext;
