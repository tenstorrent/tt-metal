// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

void kernel_main() {

        // Nothing to move. Print respond message.

        DPRINT_DATA0(DPRINT << "Void outbound kernel is running." << ENDL()<< ENDL());
        DPRINT_DATA1(DPRINT << "Void outbound kernel is running." << ENDL()<< ENDL());

}
