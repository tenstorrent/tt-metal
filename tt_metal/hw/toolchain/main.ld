#if defined(TYPE_FIRMWARE) + defined(TYPE_KERNEL) != 1
#error "Exactly one of TYPE_FIRMWARE and TYPE_KERNEL must be defined"
#endif

#if (defined(COMPILE_FOR_BRISC) + defined(COMPILE_FOR_NCRISC) +         \
     defined(COMPILE_FOR_IERISC) + defined(COMPILE_FOR_SLAVE_IERISC) +  \
     defined(COMPILE_FOR_TRISC) + defined(COMPILE_FOR_AERISC)) != 1
#error "Exactly one COMPILE_FOR_${PROC}RISC must be defined"
#endif

#include "dev_mem_map.h"
#include "memory.ld"

#define FIRMWARE_STACK_SIZE LENGTH(REGION_STACK)
#if defined(COMPILE_FOR_NCRISC)
#define KERNEL_ENTRY_SYMBOL _Z11kernel_initm
#endif

#include "sections.ld"
