#ifndef _NOC_MACROS_H_
#define _NOC_MACROS_H_

#ifndef COMPILE_FOR_EMULE
#define COMPILE_FOR_EMULE 0
#endif

#if COMPILE_FOR_EMULE
#include "src/t6ifc/emule-core/ncrisc_fw_interface.h"

namespace ncrisc_fw {
/** @brief Gateway for this firmware to interact with emule. (This firmware is always run in a unique thread) */
inline thread_local emule::NocControlRiscInterface *emule_api = nullptr;
}  // namespace ncrisc_fw

#undef NOC_STREAM_READ_REG
#define NOC_STREAM_READ_REG(stream_id, reg_id) ncrisc_fw::emule_api->read_stream_register(stream_id, reg_id)
#undef NOC_STREAM_WRITE_REG
#define NOC_STREAM_WRITE_REG(stream_id, reg_id, val) ncrisc_fw::emule_api->write_stream_register(stream_id, reg_id, val)

// do translation for emule
#define MAP_PTR_IF_EMULE(expr) (ncrisc_fw::emule_api->firmwarePointerToEmule(expr))
#define REVERSE_MAP_PTR_IF_EMULE(expr) (ncrisc_fw::emule_api->emulePointerToFirmware(expr))

#else  // COMPILE_FOR_EMULE

// make it a no-op for versim/board
#define MAP_PTR_IF_EMULE(expr) (expr)
#define REVERSE_MAP_PTR_IF_EMULE(expr) (expr)

#endif  // COMPILE_FOR_EMULE

#define NOC_READ_NB(noc, cmd_buf, src_addr, dest_addr, len_bytes)      \
  noc_fast_read(noc, cmd_buf, src_addr, dest_addr, len_bytes);


#define NOC_READ_NB_ANY_LEN(noc, cmd_buf, src_addr, dest_addr, len_bytes)  \
  noc_fast_read_any_len(noc, cmd_buf, src_addr, dest_addr, len_bytes);


#define NOC_READ_NB_LOCAL_128BIT(noc, cmd_buf, src_addr, dest_addr)      \
  noc_fast_read_local_128bit(noc, cmd_buf, src_addr, dest_addr);


#define NOC_WRITE_NB(noc, cmd_buf, src_addr, dest_addr, len_bytes, vc, mcast, linked, num_dests, physical_grid, dram_x, dram_y)  \
  noc_fast_write(noc, cmd_buf, src_addr, dest_addr, len_bytes, vc, mcast, linked, num_dests);


#define NOC_WRITE_NB_ANY_LEN(noc, cmd_buf, src_addr, dest_addr, len_bytes, vc, mcast, linked, num_dests, physical_grid, dram_x, dram_y)  \
  noc_fast_write_any_len(noc, cmd_buf, src_addr, dest_addr, len_bytes, vc, mcast, linked, num_dests);

#endif
