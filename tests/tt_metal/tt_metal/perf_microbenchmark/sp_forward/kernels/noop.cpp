// SPDX-License-Identifier: Apache-2.0
// Receive-only worker: no work. Multicast writes into this core's L1 are acked by NoC hardware;
// the kernel just needs to exist so the core is part of the program.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {}
