#include "dev_mem_map.h"
#include "tensix-memory.ld"

REGION_ALIAS("REGION_CODE", TRISC2_CODE)
REGION_ALIAS("REGION_DATA", TRISC_LOCAL_DATA_MEM)
REGION_ALIAS("REGION_RODATA", TRISC_LOCAL_DATA_MEM)
REGION_ALIAS("REGION_LDM", TRISC_LOCAL_DATA_MEM)

__firmware_start = ORIGIN(TRISC2_CODE);
__firmware_stack_size = MEM_TRISC2_STACK_SIZE;

#include "tensix-sections.ld"
