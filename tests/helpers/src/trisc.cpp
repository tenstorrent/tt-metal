// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ckernel.h"
#include "ckernel_addr_map.h"
#include "ckernel_pcbuf.h"
#include "ckernel_main.h"
#include "ckernel_globals.h"
#include <l1_address_map.h>
#include <tensix.h>
// Necessary for ckernel variables
#include "ckernel_helper.h"
#include "params.h"

#ifdef LLK_TRISC_UNPACK
	volatile uint32_t* mailbox = (volatile uint32_t*)(0x19FFC);
#elif defined(LLK_TRISC_MATH)
	volatile uint32_t* mailbox = (volatile uint32_t*)(0x19FF8);
#elif defined(LLK_TRISC_PACK)
	volatile uint32_t* mailbox = (volatile uint32_t*)(0x19FF4);
#endif


int main()
{

	*mailbox = 0x2; // write value different than 1 to mailbox to indicate kernel is running

	for (int i = 0; i < 64; i++){
		regfile[i] = 0;
	}

	reset_cfg_state_id();
	reset_dest_offset_id();

	tensix_sync();
    run_kernel();

	*mailbox = KERNEL_COMPLETE; // 0x1

	for(;;){}
}
