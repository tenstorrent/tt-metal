// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "risc_chip_specific.h"

////

int get_epoch_table_x(int my_x, int my_y) {
#if NO_DISTRIBUTED_EPOCH_TABLES == 1
  return 0;
#else
  int epoch_x;
#ifdef ERISC
  if (my_x <= 4) {
    epoch_x = 0;
  } else {
    epoch_x = 5;
  }
#else
  switch (my_y) {
    case 0:
    case 11:
    case 1:
    case 7:
    case 5:
    case 6:
      if (my_x >= 5)
        epoch_x = 5;
      else
        epoch_x = 0;
      break;

    default:
      epoch_x = 5;
      break;
  }
#endif
  return epoch_x;
#endif
}


int get_epoch_table_y(int my_x, int my_y) {
#if NO_DISTRIBUTED_EPOCH_TABLES == 1
  return 0;
#else
  return my_y;
#endif
}
