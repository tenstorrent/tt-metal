#include "dev_mem_map.h"
#include "tensix-memory.ld"

REGION_ALIAS("REGION_CODE", TRISC2_CODE)
REGION_ALIAS("REGION_DATA", LOCAL_DATA_MEM_TRISC)
REGION_ALIAS("REGION_RODATA", LOCAL_DATA_MEM_TRISC)
REGION_ALIAS("REGION_LDM", LOCAL_DATA_MEM_TRISC)

__firmware_start = ORIGIN(TRISC2_CODE);
__firmware_stack_size = 768;
__local_mem_rodata_start_addr = ORIGIN(LOCAL_DATA_MEM_TRISC);

#include "tensix-sections.ld"
