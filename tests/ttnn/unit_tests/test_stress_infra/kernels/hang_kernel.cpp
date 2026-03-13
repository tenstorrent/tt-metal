// Deliberately hanging kernel — waits for data that never arrives
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Wait on CB 0 for 1 tile — no producer will ever push data here
    cb_wait_front(0, 1);
}
