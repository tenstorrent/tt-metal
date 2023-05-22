#include "dev_mem_map.h"
#include "tensix-memory.ld"

REGION_ALIAS("REGION_CODE", TRISC0_CODE)
REGION_ALIAS("REGION_DATA", TRISC_LOCAL_DATA_MEM)
REGION_ALIAS("REGION_RODATA", TRISC_LOCAL_DATA_MEM)
REGION_ALIAS("REGION_LDM", TRISC_LOCAL_DATA_MEM)

__firmware_start = ORIGIN(TRISC0_CODE);
__firmware_stack_size = MEM_TRISC0_STACK_SIZE;

#include "tensix-sections.ld"
