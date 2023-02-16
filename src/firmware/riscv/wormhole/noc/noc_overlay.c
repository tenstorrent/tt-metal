
#include "noc_overlay.h"

#include <stdint.h>
#include <stdbool.h>
#include "noc_overlay_parameters.h"


#ifdef TB_NOC

#include "noc_api_dpi.h"

#else

#define STREAM_WRITE_REG(id, reg_idx, val) ( ( *( (volatile uint32_t*)( OVERLAY_REGS_START_ADDR + ((id)*STREAM_REG_SPACE_SIZE) + ((reg_idx)*4)) ) ) = (val) )
#define STREAM_READ_REG(id, reg_idx )        ( *( (volatile uint32_t*)( OVERLAY_REGS_START_ADDR + ((id)*STREAM_REG_SPACE_SIZE) + ((reg_idx)*4)) ) )

#endif


