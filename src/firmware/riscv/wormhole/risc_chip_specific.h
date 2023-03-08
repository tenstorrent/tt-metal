#pragma once

#include <stdint.h>

////

int get_epoch_table_x(int my_x, int my_y) __attribute__((const));
int get_epoch_table_y(int my_x, int my_y) __attribute__((const));

inline uint16_t op_pack_tiles_ptr_add(uint16_t a, uint16_t b) {
#ifdef RISC_B0_HW
  return (a + b) & 0x3FF;
#else
  return a + b;
#endif
}

inline uint16_t op_pack_tiles_ptr_sub(uint16_t a, uint16_t b) {
#ifdef RISC_B0_HW
  return (a - b) & 0x3FF;
#else
  return a - b;
#endif
}
