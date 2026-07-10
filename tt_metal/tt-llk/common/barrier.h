// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "llk_assert.h"

template <std::size_t NumParticipants>
class ReentrantBarrier
{
private:

    std::uint32_t arrive[NumParticipants];
    std::uint32_t release   = 0;    

    void coordinator()
    {
        ckernel::fence_compiler();
        for (size_t i = 1; i < 1 + sizeof...(Participant); i++)
        {
            while (ckernel::load_force(arrive[i]) != flag)
            {   
                ckernel::invalidate_data_cache();
                asm volatile("nop; nop; nop; nop; nop; nop; nop; nop;\n\t");
            }
        }
        ckernel::fence_compiler(); // only compiler because branch on load will block the pipeline until the load completes
        ckernel::store_blocking(release, flag);
    }

    void worker()
    {
        ckernel::fence_compiler(); // doesn't prevent previous stores from reordering, 
        while(ckernel::load_force(release) != flag) {
            ckernel::invalidate_data_cache();
            asm volatile("nop; nop; nop; nop; nop; nop; nop; nop;\n\t");
        }
        ckernel::fence_compiler(); // only compiler because branch on load will block the pipeline until the load completes
    }

public:
    void init()
    {
        // fixme: only works in llk infra becuse we don't have COMPILE_FOR_TRISC in ttmetal QSR
        LLK_ASSERT(ENV_LLK_INFRA, "ReentrantBarrier is only supported in LLK infra.");  
        
        // fixme: these storeblocking are too heavy, only the last one is required if it fits in on bank and is aligned.
        for(size_t i = 0; i < 1 + sizeof...(Participant); i++)
        {
            ckernel::store_blocking(arrive[i], 0);
        }
        ckernel::store_blocking(release, 0);
    }

    void sync()
    {
        // fixme: find a smarter way to support this on QSR
#if !defined(COMPILE_FOR_TRISC)
        LLK_ASSERT(false, "ReentrantBarrier currently only supports TRISC sync");
#endif
        if constexpr(COMPILE_FOR_TRISC == 0)
        {
            // fixme: coordinator should be determined by the template param.
            coordinator();
        }
        else
        {
            worker();
        }
    }

};
