#include "compile_time_args.h"
#include <cstdint>

constexpr uint32_t num_writes = get_compile_time_arg_val(0);
constexpr uint32_t done_flag = get_compile_time_arg_val(1);
constexpr uint32_t buffer_base = get_compile_time_arg_val(2);

inline void done() {
    auto done_flag_ptr = reinterpret_cast<volatile uint32_t*>(done_flag);
    done_flag_ptr[0] = 1;
}

void kernel_main() {
    auto buffer_ptr = reinterpret_cast<volatile uint32_t*>(buffer_base);
    for (uint32_t expected_value = 0; expected_value < num_writes; expected_value++) {
        // NOLINTNEXTLINE(bugprone-infinite-loop)
        do {
            invalidate_l1_cache();
        } while (buffer_ptr[expected_value] != expected_value);
    }

    for (uint32_t index = 0; index < num_writes; index++) {
        // multiple each value by 2
        buffer_ptr[index] *= 2;
    }

    done();
}
