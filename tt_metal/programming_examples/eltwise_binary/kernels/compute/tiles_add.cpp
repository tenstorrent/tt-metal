#include <cstdint>
#include "compile_time_args.h"
#include "compute_kernel_api.h"

using namespace std;

namespace NAMESPACE {
void MAIN {
    uint32_t src0_addr = 0;
    uint32_t src1_addr = 0;
    uint32_t dst_addr = 0;

    volatile uint32_t* src0_ptr = (uint32_t*)(src0_addr);
    volatile uint32_t* src1_ptr = (uint32_t*)(src1_addr);

    volatile uint32_t* dst_ptr = (uint32_t*)(dst_addr);

    volatile float* src0_floats = (volatile float*)(src0_ptr);
    volatile float* src1_floats = (volatile float*)(src1_ptr);

    volatile float* dst_floats = (volatile float*)(dst_ptr);

    uint32_t num_elements = 0;
    for (uint32_t i = 0; i < num_elements; i++) {
        dst_floats[i] = src0_floats[i] + src1_floats[i];
    }
}
}  // namespace NAMESPACE
