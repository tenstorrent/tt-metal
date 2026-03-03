// Minimal diagnostic: test if SemaphoreDescriptor initial_value=1 works.
// Just waits for semaphore 0 to be >= 1 (should pass immediately if initial_value=1).

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t SEM_TEST = 0;
    volatile tt_l1_ptr uint32_t* test_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_TEST));
    noc_semaphore_wait(test_sem, 1);
    // If we reach here, initial_value=1 worked correctly.
}
