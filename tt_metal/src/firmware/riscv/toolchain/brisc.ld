#include "dev_mem_map.h"
#include "tensix-memory.ld"

#if DEBUG_MODE == 1
REGION_ALIAS("REGION_CODE", BRISC_CODE)
REGION_ALIAS("REGION_DATA", BRISC_CODE)
REGION_ALIAS("REGION_STACK", BRISC_CODE)
#else
REGION_ALIAS("REGION_CODE", BRISC_CODE)
REGION_ALIAS("REGION_DATA", BRISC_LOCAL_DATA_MEM)
REGION_ALIAS("REGION_STACK", BRISC_STACK_MEM)
#endif

__firmware_stack_size = MEM_BRISC_STACK_SIZE;

#include "tensix-sections.ld"

_Z11kernel_initv = _etext;
