#include <cstdint>

#include "sanitizer/api.h"


const ThreadOutputContext thread_context;


void init_reconfig_pass_clobbers() {

    const FsmState current {
        FsmStateType::Initialized,
        Operation::Pack
    };

    const FsmState next {
        FsmStateType::Reconfigured,
        Operation::None
    };

    fsm_check(
        thread_context,
        current,
        next
    );

}




void kernel_main(
    std::uint32_t case
) {



}
