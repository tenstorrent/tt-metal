#include "deployment_common.hpp"

std::atomic_bool g_stop_requested = false;
std::atomic_bool g_stop_message_printed = false;

void handle_sigint(int) {
    g_stop_requested.store(true);

    if (!g_stop_message_printed.exchange(true)) {
        const char msg[] = "\nSIGINT received, waiting to finish current test...\n";
        write(2, msg, sizeof msg - 1);
    }
}
