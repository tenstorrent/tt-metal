INCLUDE "tensix-memory.ld"

/* This TRISC is an override location remapped to physical TRISC0 */

REGION_ALIAS("REGION_CODE", TRISC0_OVRD_CODE)
REGION_ALIAS("REGION_DATA", TRISC0_OVRD_CODE)
REGION_ALIAS("REGION_RODATA", TRISC0_OVRD_CODE)
REGION_ALIAS("REGION_LDM", LOCAL_DATA_MEM_TRISC)

__firmware_start = ORIGIN(TRISC0_OVRD_CODE); 
__firmware_stack_size = 1024;

INCLUDE "tensix-sections.ld"
INCLUDE "tensix-address.ld"

