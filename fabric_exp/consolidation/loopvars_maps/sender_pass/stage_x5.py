"""stage_x5.py: v4txo5 = v4txo4 + X5 (V4_SENDER_OWNERSHIP.md §3 #8, "Planned: one copy of the cold worker-connection
path"): the library's worker interface gets a base holding its connection state whatever the channel's slot count
(EdmChannelWorkerConnection<NOC>; main's router unchanged), and the router has one out-of-line worker connection event
for every worker channel; the opening credit (preset, credits so far) stays per channel."""
import os, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
src, dst = f"{V}/v4txo4", f"{V}/" + os.environ.get("X5_NAME", "v4txo5")
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(src, dst, symlinks=True)

# the library: the connection state in a base of its own
p = f"{dst}/fabric_erisc_datamover_channels.hpp"
s = open(p).read()
def rep(o, n, count=1):
    global s
    assert s.count(o) == count, (s.count(o), o[:90])
    s = s.replace(o, n)
rep("""// Base class for channel worker interfaces
// Derived classes implement specific counter management strategies.
template <uint8_t WORKER_HANDSHAKE_NOC, typename DERIVED>
struct EdmChannelWorkerInterface {
    EdmChannelWorkerInterface() :
        worker_location_info_ptr(nullptr),
        cached_worker_semaphore_address(0),
        connection_live_semaphore(nullptr),
        sender_sync_noc_cmd_buf(write_at_cmd_buf) {}
""", """// A sender channel's connection to its producer, whatever the channel's slot count: the producer's location info, the
// connection semaphore, the producer's credit address (cached when it connects) and the command buffer of its credits.
// One type for every channel, so code that only opens and closes connections exists once.
template <uint8_t WORKER_HANDSHAKE_NOC>
struct EdmChannelWorkerConnection {
    EdmChannelWorkerConnection() :
        worker_location_info_ptr(nullptr),
        cached_worker_semaphore_address(0),
        connection_live_semaphore(nullptr),
        sender_sync_noc_cmd_buf(write_at_cmd_buf) {}

    EdmChannelWorkerConnection(
        volatile EDMChannelWorkerLocationInfo* worker_location_info_ptr,
        volatile tt_l1_ptr uint32_t* const connection_live_semaphore,
        uint8_t sender_sync_noc_cmd_buf) :
        worker_location_info_ptr(worker_location_info_ptr),
        cached_worker_semaphore_address(0),
        connection_live_semaphore(connection_live_semaphore),
        sender_sync_noc_cmd_buf(sender_sync_noc_cmd_buf) {}

    [[nodiscard]] FORCE_INLINE uint32_t get_worker_semaphore_address() const {
        return cached_worker_semaphore_address & 0xFFFFFFFF;
    }

    template <bool RISC_CPU_DATA_CACHE_ENABLED, uint8_t MY_ETH_CHANNEL = USE_DYNAMIC_CREDIT_ADDR>
    FORCE_INLINE void cache_producer_noc_addr() {
        router_invalidate_l1_cache<RISC_CPU_DATA_CACHE_ENABLED>();
        const auto& worker_info = *worker_location_info_ptr;
        uint64_t worker_semaphore_address;
        worker_semaphore_address = get_noc_addr(
            (uint32_t)worker_info.worker_xy.x, (uint32_t)worker_info.worker_xy.y, worker_info.worker_semaphore_address);
        this->cached_worker_semaphore_address = worker_semaphore_address;
    }

    [[nodiscard]] FORCE_INLINE bool has_worker_teardown_request() const {
        return *connection_live_semaphore == tt::tt_fabric::connection_interface::close_connection_request_value;
    }

    // Closes the connection: the semaphore back to unused, `read_counter` (the credits returned so far) into the
    // producer's location info, and +1 on the producer's teardown semaphore.
    template <bool posted = false, bool RISC_CPU_DATA_CACHE_ENABLED>
    FORCE_INLINE void close_connection(uint32_t read_counter) const {
        router_invalidate_l1_cache<RISC_CPU_DATA_CACHE_ENABLED>();
        const auto& worker_info = *worker_location_info_ptr;
        uint64_t worker_semaphore_address = get_noc_addr(
            (uint32_t)worker_info.worker_xy.x,
            (uint32_t)worker_info.worker_xy.y,
            worker_info.worker_teardown_semaphore_address);

        *this->connection_live_semaphore = tt::tt_fabric::connection_interface::unused_connection_value;

        worker_location_info_ptr->edm_read_counter = read_counter;

        noc_semaphore_inc<posted>(worker_semaphore_address, 1, WORKER_HANDSHAKE_NOC);
    }

    volatile tt_l1_ptr EDMChannelWorkerLocationInfo* worker_location_info_ptr;
    uint64_t cached_worker_semaphore_address = 0;
    volatile tt_l1_ptr uint32_t* const connection_live_semaphore;
    uint8_t sender_sync_noc_cmd_buf;
};

// Base class for channel worker interfaces
// Derived classes implement specific counter management strategies.
template <uint8_t WORKER_HANDSHAKE_NOC, typename DERIVED>
struct EdmChannelWorkerInterface : EdmChannelWorkerConnection<WORKER_HANDSHAKE_NOC> {
    using Connection = EdmChannelWorkerConnection<WORKER_HANDSHAKE_NOC>;
    using Connection::worker_location_info_ptr;
    using Connection::cached_worker_semaphore_address;
    using Connection::connection_live_semaphore;
    using Connection::sender_sync_noc_cmd_buf;

    EdmChannelWorkerInterface() : Connection() {}
""")
rep("""        uint8_t edm_read_counter_initial_value) :
        worker_location_info_ptr(worker_location_info_ptr),
        cached_worker_semaphore_address(0),
        connection_live_semaphore(connection_live_semaphore),
        sender_sync_noc_cmd_buf(sender_sync_noc_cmd_buf) {""", """        uint8_t edm_read_counter_initial_value) :
        Connection(worker_location_info_ptr, connection_live_semaphore, sender_sync_noc_cmd_buf) {""")
# the old base's copies of the moved members and methods
a = s.index("template <uint8_t WORKER_HANDSHAKE_NOC, typename DERIVED>\nstruct EdmChannelWorkerInterface :")
b = s.index("\n};\n", a) + 1
body = s[a:b]
for o in ["""    [[nodiscard]] FORCE_INLINE uint32_t get_worker_semaphore_address() const {
        return cached_worker_semaphore_address & 0xFFFFFFFF;
    }
""", """    template <bool RISC_CPU_DATA_CACHE_ENABLED, uint8_t MY_ETH_CHANNEL = USE_DYNAMIC_CREDIT_ADDR>
    FORCE_INLINE void cache_producer_noc_addr() {
        router_invalidate_l1_cache<RISC_CPU_DATA_CACHE_ENABLED>();
        const auto& worker_info = *worker_location_info_ptr;
        uint64_t worker_semaphore_address;
        worker_semaphore_address = get_noc_addr(
            (uint32_t)worker_info.worker_xy.x, (uint32_t)worker_info.worker_xy.y, worker_info.worker_semaphore_address);
        this->cached_worker_semaphore_address = worker_semaphore_address;
    }
""", """    [[nodiscard]] FORCE_INLINE bool has_worker_teardown_request() const {
        return *connection_live_semaphore == tt::tt_fabric::connection_interface::close_connection_request_value;
    }
""", """    volatile tt_l1_ptr EDMChannelWorkerLocationInfo* worker_location_info_ptr;
    uint64_t cached_worker_semaphore_address = 0;
    volatile tt_l1_ptr uint32_t* const connection_live_semaphore;
    uint8_t sender_sync_noc_cmd_buf;
"""]:
    assert body.count(o) == 1, o[:70]
    body = body.replace(o, "")
s = s[:a] + body + s[b:]
open(p, "w").write(s)

# the router: one connection event for every worker channel
p = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router/router_sender.hpp"
s = open(p).read()
a = s.index("// Worker channel CH's connection semaphore changed (open or close request)")
b = s.index("\n}\n", a) + 3
s = s[:a] + """// A worker channel's connection, whatever its slot count (the library's connection state).
using WorkerConnection = tt::tt_fabric::EdmChannelWorkerConnection<WORKER_CREDIT_NOC>;

// A worker channel's connection semaphore changed (open or close request): handled out of line, one copy for every
// worker channel; returns the value the semaphore holds while nothing changes. Opening caches the worker's credit
// address (the caller then sends the opening credit); closing writes the credits returned so far (`returned`) into the
// worker's location info and signals its teardown semaphore.
__attribute__((noinline)) uint32_t worker_connection_event(
    WorkerConnection& connection, uint32_t semaphore, uint32_t expected, uint32_t returned) {
    using namespace tt::tt_fabric::connection_interface;
    if (expected == unused_connection_value) {
        if (connect_is_requested(semaphore)) {
            connection.cache_producer_noc_addr<ENABLE_RISC_CPU_DATA_CACHE, MY_ETH_CHANNEL>();
            return open_connection_value;
        }
        return unused_connection_value;
    }
    if (semaphore == close_connection_request_value) {
        connection.close_connection<true, ENABLE_RISC_CPU_DATA_CACHE>(returned);
        return unused_connection_value;
    }
    return open_connection_value;
}

// Worker channel CH's connection event (open or close request), out of line: the shared event, then, where it opened
// the connection, the channel's credit command buffer programmed for the worker and the credits returned so far sent on
// it. A thin wrapper, so the pass calls one function per channel, as before.
template <size_t CH>
__attribute__((noinline)) uint32_t connection_event(uint32_t semaphore, uint32_t expected, uint32_t returned) {
    const uint32_t state = worker_connection_event(worker_interface<CH>(), semaphore, expected, returned);
    if (expected == tt::tt_fabric::connection_interface::unused_connection_value &&
        state != tt::tt_fabric::connection_interface::unused_connection_value) {
        preset_worker_credit<CH>();
        credit_worker<CH>(returned, true);
    }
    return state;
}
""" + s[b:]
o = """        state.connection_state = worker_connection_event<CH>(semaphore, state.connection_state, returned);"""
n = """        state.connection_state = connection_event<CH>(semaphore, state.connection_state, returned);"""
assert s.count(o) == 1
s = s.replace(o, n)
open(p, "w").write(s)
print("made", dst)
