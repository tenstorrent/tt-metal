// Simple tool to check what buffers the server thinks are still allocated
#include <iostream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>

#define TT_ALLOC_SERVER_SOCKET "/tmp/tt_allocation_server.sock"

struct __attribute__((packed)) AllocMessage {
    enum Type : uint8_t { ALLOC = 1, FREE = 2, QUERY = 3, RESPONSE = 4 };

    Type type;
    uint8_t pad1[3];
    int32_t device_id;
    uint64_t size;
    uint8_t buffer_type;
    uint8_t pad2[3];
    int32_t process_id;
    uint64_t buffer_id;
    uint64_t timestamp;

    uint64_t dram_allocated;
    uint64_t l1_allocated;
    uint64_t l1_small_allocated;
    uint64_t trace_allocated;
};

const char* buffer_type_name(uint8_t type) {
    switch (type) {
        case 0: return "DRAM";
        case 1: return "L1";
        case 2: return "SYSTEM_MEMORY";
        case 3: return "L1_SMALL";
        case 4: return "TRACE";
        default: return "UNKNOWN";
    }
}

int main() {
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Failed to create socket\n";
        return 1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, TT_ALLOC_SERVER_SOCKET, sizeof(addr.sun_path) - 1);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "Failed to connect to server. Is it running?\n";
        close(sock);
        return 1;
    }

    std::cout << "Querying all devices for remaining allocations...\n\n";

    for (int dev = 0; dev < 8; dev++) {
        AllocMessage query;
        memset(&query, 0, sizeof(query));
        query.type = AllocMessage::QUERY;
        query.device_id = dev;

        send(sock, &query, sizeof(query), 0);

        AllocMessage response;
        recv(sock, &response, sizeof(response), 0);

        if (response.dram_allocated > 0 || response.l1_allocated > 0) {
            std::cout << "Device " << dev << ":\n";
            if (response.dram_allocated > 0) {
                std::cout << "  DRAM: " << response.dram_allocated << " bytes\n";
            }
            if (response.l1_allocated > 0) {
                std::cout << "  L1: " << response.l1_allocated << " bytes\n";
            }
            std::cout << "\n";
        }
    }

    close(sock);

    std::cout << "\nTo see WHICH specific buffers these are, check the server code.\n";
    std::cout << "The server needs to be modified to return buffer details.\n";

    return 0;
}
