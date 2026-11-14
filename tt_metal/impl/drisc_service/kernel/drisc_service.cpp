// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DRISC service kernel: copies data between NOC endpoints and within its GDDR bank on host request.

#include <cstddef>
#include <cstdint>

#include "bounce_arena.hpp"
#include "hostdev/drisc_service/abi.hpp"
#include "kernel_utils.hpp"
#include "l1_layout.hpp"
#include "transaction.hpp"

// Each in-flight transaction uses its ring index as its NIU transaction ID. Control-page publishes get the last ID, and
// batch window pulls share the ones between, so none aliases a transaction's completion counter.
constexpr uint32_t PUBLISH_TRID = NOC_MAX_TRANSACTION_ID;
constexpr uint32_t PULL_TRIDS_MASK = ((1u << PUBLISH_TRID) - 1) & ~((1u << MAX_INFLIGHT_TRANSACTIONS) - 1);
static_assert(__builtin_popcount(PULL_TRIDS_MASK) == abi::PULL_TRIDS);

// The guard linker script holds the kernel's text below the batch windows.
asm(".globl drisc_service_text_limit\n.set drisc_service_text_limit, %0" ::"i"(l1::BATCH_WINDOWS_START));

// do_crt1 from firmware_common.h, which compiles only with the JIT's full define set. Both are in substitutes.o.
extern "C" void wzerorange(uint32_t* start, uint32_t* end);
void l1_to_local_mem_copy(uint32_t* dst, uint32_t tt_l1_ptr* src, L1WordCount len);
static void crt1() {
    extern uint32_t __ldm_bss_start[], __ldm_bss_end[], __ldm_data_start[], __ldm_data_end[];
    wzerorange(__ldm_bss_start, __ldm_bss_end);
    l1_to_local_mem_copy(
        __ldm_data_start,
        reinterpret_cast<uint32_t tt_l1_ptr*>(MEM_DRISC_INIT_LOCAL_L1_BASE_SCRATCH),
        l1_word_count_from_range(__ldm_data_start, __ldm_data_end));
}

// Read by the host out of the ELF before boot (abi::BUILD_BLOCK_SECTION).
__attribute__((section(".drisc_service_build"), used)) const abi::KernelBuildBlock build_block = {
    .magic = abi::BUILD_BLOCK_MAGIC,
    .abi_version = abi::ABI_VERSION,
};

// The host writes this block in our L1; the kernel writes only the header and the acks.
volatile abi::L1CommandBlock* const command_block =
    reinterpret_cast<volatile abi::L1CommandBlock*>(abi::L1CommandBlockLayout::base_addr);

// Copy one command out of L1 the host or the NOC writes (queue entry or batch
// slot). A volatile struct doesn't bind to the implicit copy ctor, so go field
// by field (scalar reads).
abi::KernelCommand read_command(const volatile abi::KernelCommand& entry) {
    abi::KernelCommand cmd;
    cmd.src_addr = entry.src_addr;
    cmd.dst_addr = entry.dst_addr;
    cmd.size = entry.size;
    cmd.src_noc_xy_addr = entry.src_noc_xy_addr;
    cmd.dst_noc_xy_addr = entry.dst_noc_xy_addr;
    return cmd;
}

Transaction transactions[MAX_INFLIGHT_TRANSACTIONS];

// The seat's bank, from the header the host wrote before boot.
uint64_t bank_size;
uint16_t bank_noc_xy_addr[abi::BANK_ENDPOINTS];

// The GDDR DMA engine can read `bytes` at `src`: inside the seat's bank, in 16 B units. SeatClient::validate() mirrors
// this.
bool dma_readable(const abi::NocEndpoint& src, uint32_t bytes) {
    if (bytes % abi::LOCAL_SIZE_ALIGN != 0 || src.local_addr > bank_size - bytes) {
        return false;
    }
    for (const uint16_t xy : bank_noc_xy_addr) {
        if (src.noc_xy_addr == xy) {
            return true;
        }
    }
    return false;
}

// The kernel's view of one client slot. ATTACHED: carve its commands and
// publish its done_count. DETACHING: the client left; stop carving, issue none
// of its NoC writes not yet started, and free the slot once none is in flight.
enum SlotState : uint8_t {
    SLOT_IDLE,
    SLOT_ATTACHED,
    SLOT_DETACHING,
};

// Carve cursor: the client's command being chopped into transactions — a queue command, or one entry of a CMD_BATCH's
// list. A command occupies the cursor until its last transaction is emitted; a CMD_BATCH occupies its queue entry until
// its list's last transaction is emitted.
enum CarveState : uint8_t {
    CARVE_IDLE,         // no command; the next one comes from the client's queue
    CARVE_SINGLE,       // cmd is a queue command
    CARVE_BATCH_ENTRY,  // cmd is an entry of the batch list
    CARVE_BATCH_NEXT,   // between entries; the next one may still be on its way into the window
};

struct CarveCursor {
    abi::KernelCommand cmd{};
    uint32_t off = 0;  // bytes of cmd emitted so far
    abi::NocEndpoint batch_list{};
    CarveState state = CARVE_IDLE;
    uint8_t batch_count = 0;    // entries in the list
    uint8_t batch_next = 0;     // next entry to load into cmd
    uint8_t halves_landed = 0;  // window halves of the list pulled into L1
    uint8_t pull_trid = 0;      // TRID of the next half's pull in flight; 0 = none
};

struct Client {
    abi::NocEndpoint control_page{};
    uint32_t gen = 0;
    uint32_t accepted = 0;   // commands fully carved
    uint32_t completed = 0;  // commands retired
    uint32_t inflight = 0;   // transactions not yet retired
    SlotState state = SLOT_IDLE;
    CarveCursor carve;
};
static_assert(abi::MAX_CLIENTS <= 32, "retire_transactions tracks clients in a 32-bit mask");
static_assert(abi::BATCH_MAX_COMMANDS <= 255, "CarveCursor counts list entries in a byte");
Client clients[abi::MAX_CLIENTS];

// Batch window pull TRIDs not in use, one bit per TRID.
uint32_t free_pull_trids = PULL_TRIDS_MASK;

// Writes issued as of the last publish; scratch may be reused once that many have left L1. Set after noc_boot:
// the NIU's sent counter survives RISC reset, so 0 is not a safe start.
uint32_t scratch_watermark;
// Relies on acked host writes being visible before this one.
void publish_done_count(const Client& client) {
    while (!noc_nonposted_writes_sent_since(noc_index, scratch_watermark));
    *reinterpret_cast<volatile uint32_t*>(l1::SCRATCH) = client.completed;
    while (!noc_try_write_packet_with_trid(
        l1::SCRATCH,
        client.control_page.at_offset(offsetof(abi::ControlPage, done_count)),
        sizeof(client.completed),
        PUBLISH_TRID));
    scratch_watermark = noc_nonposted_writes_checkpoint(noc_index);
}
// A bit per client slot in SLOT_ATTACHED, and the round-robin start of the next pick.
uint32_t attached_mask;
uint32_t next_slot;
constexpr uint32_t ALL_SLOTS_MASK = (1u << abi::MAX_CLIENTS) - 1;

// DMA requests issued and not yet retired, per stream.
struct DmaInFlight {
    uint32_t reads = 0;
    uint32_t writes = 0;
};

uint32_t window_entry_addr(uint32_t slot, uint32_t entry) {
    return l1::BATCH_WINDOWS_START + slot * l1::WINDOW_SIZE +
           entry % (2 * l1::WINDOW_HALF_COMMANDS) * sizeof(abi::KernelCommand);
}

// Retires the cursor's window pull once it lands, then pulls the list's next half once the window half it lands in has
// been fully loaded. At most one pull per client is in flight.
FORCE_INLINE void pump_window(CarveCursor& carve, uint32_t slot) {
    if (carve.pull_trid != 0) {
        if (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, carve.pull_trid)) {
            return;
        }
        free_pull_trids |= 1u << carve.pull_trid;
        carve.pull_trid = 0;
        ++carve.halves_landed;
    }
    const uint32_t first = carve.halves_landed * l1::WINDOW_HALF_COMMANDS;
    if (first >= carve.batch_count || first > carve.batch_next + l1::WINDOW_HALF_COMMANDS || free_pull_trids == 0) {
        return;
    }
    const uint32_t entries =
        carve.batch_count - first < l1::WINDOW_HALF_COMMANDS ? carve.batch_count - first : l1::WINDOW_HALF_COMMANDS;
    carve.pull_trid = __builtin_ctz(free_pull_trids);
    free_pull_trids &= free_pull_trids - 1;
    noc_async_read_with_trid(
        window_entry_addr(slot, first),
        carve.batch_list.at_offset(first * sizeof(abi::KernelCommand)),
        entries * sizeof(abi::KernelCommand),
        carve.pull_trid);
}

// True once the client's cursor holds a command with bytes left to emit. Loads the client's next queue command or
// batch entry into the cursor, and keeps its batch window pulled ahead.
FORCE_INLINE bool carve_ready(uint32_t slot) {
    Client& client = clients[slot];
    CarveCursor& carve = client.carve;
    if (carve.state == CARVE_IDLE) {
        if (command_block->slots[slot].doorbell == client.accepted) {
            return false;
        }
        carve.cmd =
            read_command(command_block->slots[slot].command_queue[client.accepted % abi::CLIENT_QUEUE_CAPACITY]);
        carve.off = 0;
        if (abi::command_kind(carve.cmd) != abi::CMD_BATCH) {
            carve.state = CARVE_SINGLE;
            return true;
        }
        carve.batch_list = abi::NocEndpoint{carve.cmd.src_addr, carve.cmd.src_noc_xy_addr};
        carve.batch_count = carve.cmd.size / sizeof(abi::KernelCommand);
        carve.batch_next = 0;
        carve.halves_landed = 0;
        carve.state = CARVE_BATCH_NEXT;
    }
    if (carve.state != CARVE_BATCH_NEXT) {
        return true;
    }
    if (carve.batch_next % l1::WINDOW_HALF_COMMANDS == 0) {
        pump_window(carve, slot);
    }
    if (carve.batch_next >= carve.halves_landed * l1::WINDOW_HALF_COMMANDS) {
        return false;
    }
    carve.cmd =
        read_command(*reinterpret_cast<const volatile abi::KernelCommand*>(window_entry_addr(slot, carve.batch_next)));
    ++carve.batch_next;
    carve.off = 0;
    carve.state = CARVE_BATCH_ENTRY;
    return true;
}

// Round-robin over the attached clients: the first from next_slot whose cursor has bytes to emit. Turns average
// CHUNK_SIZE bytes (see turn_bytes), so clients share the seat equally by bytes.
FORCE_INLINE bool pick_client(uint32_t& slot) {
    const uint32_t start = next_slot;
    const uint32_t mask = attached_mask;
    uint32_t pending = (mask >> start | mask << (abi::MAX_CLIENTS - start)) & ALL_SLOTS_MASK;
    for (; pending != 0; pending &= pending - 1) {
        slot = (start + __builtin_ctz(pending)) % abi::MAX_CLIENTS;
        if (carve_ready(slot)) {
            return true;
        }
    }
    return false;
}

// The picked client keeps the seat until it has carved CHUNK_SIZE bytes or has nothing to carve, so a run of small
// commands is picked once per chunk, like one large command. While another client is attached, a piece is cut to what
// is left of the turn, rounded down to 64 B so pieces stay aligned: every turn carves CHUNK_SIZE to within 64 B, and
// clients share the seat equally by bytes whatever their command sizes. A client alone takes whole chunks.
// turn_bytes = CHUNK_SIZE: no client holds the seat.
uint32_t turn_slot;
uint32_t turn_bytes = l1::CHUNK_SIZE;

// Fill the freshly claimed transaction and start its read leg. trid == ring index.
FORCE_INLINE void start_transaction(
    Transaction& t,
    uint32_t trid,
    uint32_t slot,
    uint32_t payload_bytes,
    const BounceArena::Allocation& extent,
    DmaInFlight& dma) {
    const CarveCursor& carve = clients[slot].carve;
    t.src = abi::NocEndpoint{carve.cmd.src_addr, carve.cmd.src_noc_xy_addr}.at_offset(carve.off);
    t.dst = abi::NocEndpoint{carve.cmd.dst_addr, carve.cmd.dst_noc_xy_addr}.at_offset(carve.off);
    t.l1_addr = extent.addr;
    t.payload_bytes = payload_bytes;
    t.alloc_bytes = extent.alloc_bytes;
    // Last of the queue command: a single command's last transaction, or the
    // last transaction of a batch list's last command.
    t.last_of_cmd = carve.off + payload_bytes == carve.cmd.size &&
                    (carve.state == CARVE_SINGLE || carve.batch_next == carve.batch_count);
    t.client_slot = static_cast<uint8_t>(slot);
    const bool local = abi::command_kind(carve.cmd) == abi::CMD_LOCAL;
    t.dma_read_pending = local || dma_readable(t.src, payload_bytes);
    t.dma_write = local;
    if (t.dma_read_pending) {
        experimental::dma_async_read<false>(dma_read_stream, t.src.local_addr, t.l1_addr, payload_bytes);
        ++dma.reads;
    } else {
        noc_async_read_with_trid(t.l1_addr, t.src, payload_bytes, trid);
    }
}

// Accept: carve client commands into transactions and start their read legs,
// until the transaction ring or the arena says stop. Never waits on
// completion — a full pipe just means the next tick retries.
FORCE_INLINE void accept_transactions(TransactionRing& txn_ring, BounceArena& bounce_arena, DmaInFlight& dma) {
    while (!txn_ring.full()) {
        if (turn_bytes >= l1::CHUNK_SIZE) {
            uint32_t picked;
            if (!pick_client(picked)) {
                return;
            }
            turn_slot = picked;
            turn_bytes = 0;
            next_slot = (picked + 1) % abi::MAX_CLIENTS;
        }
        const uint32_t slot = turn_slot;
        Client& client = clients[slot];
        CarveCursor& carve = client.carve;
        const bool shared = (attached_mask & ~(1u << slot)) != 0;
        uint32_t turn = turn_bytes;
        bool ready;
        while ((ready = carve_ready(slot))) {
            const uint32_t remaining = carve.cmd.size - carve.off;
            const uint32_t limit = shared ? (l1::CHUNK_SIZE - turn) & ~63u : l1::CHUNK_SIZE;
            if (limit == 0) {
                turn = l1::CHUNK_SIZE;
                break;
            }
            const uint32_t payload_bytes = remaining > limit ? limit : remaining;
            BounceArena::Allocation extent;
            if (!bounce_arena.try_alloc(payload_bytes, extent)) {
                turn_bytes = turn;
                return;
            }
            const uint32_t trid = txn_ring.claim_trid();
            Transaction& t = transactions[trid];
            start_transaction(t, trid, slot, payload_bytes, extent, dma);
            ++client.inflight;
            turn += payload_bytes;
            carve.off += payload_bytes;
            if (carve.off == carve.cmd.size) {
                carve.state = t.last_of_cmd ? CARVE_IDLE : CARVE_BATCH_NEXT;
            }
            if (t.last_of_cmd) {
                ++client.accepted;
            }
            if (turn >= l1::CHUNK_SIZE || txn_ring.full()) {
                break;
            }
        }
        // A client with nothing to carve ends its turn and owes nothing.
        if (!ready) {
            turn_bytes = l1::CHUNK_SIZE;
            continue;
        }
        turn_bytes = turn;
    }
}

// Retires writing transactions in ring order up to the first whose writes have not acked. Returns a bit per client
// slot whose done_count changed.
FORCE_INLINE uint32_t retire_transactions(TransactionRing& txn_ring, BounceArena& bounce_arena, DmaInFlight& dma) {
    uint32_t done_changed = 0;
    while (txn_ring.has_writing()) {
        const uint32_t trid = txn_ring.oldest_writing_trid();
        if (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, trid)) {
            break;
        }
        const Transaction& t = transactions[trid];
        // A DMA stream completes in order, so the oldest of its requests is done
        // once fewer than all of them are outstanding.
        if (t.dma_write) {
            if (experimental::dma_get_writes_outstanding(dma_write_stream) >= dma.writes) {
                break;
            }
            --dma.writes;
        }
        bounce_arena.free(t.alloc_bytes);
        Client& client = clients[t.client_slot];
        --client.inflight;
        if (t.last_of_cmd) {
            ++client.completed;
            done_changed |= 1u << t.client_slot;
        }
        txn_ring.retire();
    }
    return done_changed;
}

// Issues writes for reading transactions in ring order up to the first whose read has not landed, then publishes the
// done_count of each client in done_changed. A done_count is published after its retired writes acked, so a visible
// done_count implies visible data. A detaching client's NoC writes are dropped: its host memory may already be
// unpinned.
FORCE_INLINE void issue_writes(TransactionRing& txn_ring, DmaInFlight& dma, uint32_t done_changed) {
    while (txn_ring.has_reading()) {
        const uint32_t trid = txn_ring.oldest_reading_trid();
        Transaction& t = transactions[trid];
        if (t.dma_read_pending) {
            if (experimental::dma_get_reads_outstanding(dma_read_stream) >= dma.reads) {
                break;
            }
            --dma.reads;
            t.dma_read_pending = false;
        }
        // Once its first write packet is out, the TRID counts the write, not the
        // read.
        if (txn_ring.write_issued_bytes() == 0 && !ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid)) {
            break;
        }
        if (t.dma_write) {
            experimental::dma_async_write<false>(dma_write_stream, t.l1_addr, t.dst.local_addr, t.payload_bytes);
            ++dma.writes;
            txn_ring.mark_write_issued();
            continue;
        }
        if (txn_ring.write_issued_bytes() == 0 && clients[t.client_slot].state == SLOT_DETACHING) {
            txn_ring.mark_write_issued();
            continue;
        }
        while (txn_ring.write_issued_bytes() < t.payload_bytes) {
            const uint32_t done = txn_ring.write_issued_bytes();
            const uint32_t n =
                noc_try_write_packet_with_trid(t.l1_addr + done, t.dst.at_offset(done), t.payload_bytes - done, trid);
            if (n == 0) {
                break;
            }
            txn_ring.add_write_issued_bytes(n);
        }
        if (txn_ring.write_issued_bytes() < t.payload_bytes) {
            break;
        }
        txn_ring.mark_write_issued();
    }
    for (uint32_t s = 0; done_changed != 0; s++, done_changed >>= 1) {
        const Client& client = clients[s];
        if ((done_changed & 1) == 0 || client.state != SLOT_ATTACHED) {
            continue;
        }
        publish_done_count(client);
    }
}

// Attach slot s if its request is an ATTACH the kernel has not answered yet.
// The host zeroed the doorbell and filled the control page before writing the
// request.
bool try_attach(uint32_t s) {
    volatile abi::ClientSlot& slot = command_block->slots[s];
    const uint32_t request = slot.request;
    if (abi::slot_word_type(request) != abi::SLOT_ATTACH ||
        abi::slot_word_gen(request) == abi::slot_word_gen(slot.ack)) {
        return false;
    }
    Client& client = clients[s];
    client.gen = abi::slot_word_gen(request);
    client.accepted = 0;
    client.completed = 0;
    client.control_page.local_addr = slot.control_page.local_addr;
    client.control_page.noc_xy_addr = slot.control_page.noc_xy_addr;
    client.state = SLOT_ATTACHED;
    attached_mask |= 1u << s;
    slot.ack = request;
    return true;
}

// One pass over the client slots: attach, notice exits, free drained slots.
// False once no client is left and none is attaching: the kernel stops.
bool scan_slots() {
    command_block->header.heartbeat = command_block->header.heartbeat + 1;
    command_block->header.free_pull_trid_count = __builtin_popcount(free_pull_trids);
    bool any_client = false;
    for (uint32_t s = 0; s < abi::MAX_CLIENTS; s++) {
        Client& client = clients[s];
        volatile abi::ClientSlot& slot = command_block->slots[s];
        if (client.state == SLOT_ATTACHED && slot.exit_mailbox == abi::slot_word(abi::SLOT_DETACH, client.gen)) {
            client.state = SLOT_DETACHING;
            attached_mask &= ~(1u << s);
            if (s == turn_slot) {
                turn_bytes = l1::CHUNK_SIZE;
            }
            // Drop the rest of the command being carved for it; a window pull
            // in flight must land before its window and TRID are reused.
            CarveCursor& carve = client.carve;
            if (carve.pull_trid != 0) {
                while (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, carve.pull_trid));
                free_pull_trids |= 1u << carve.pull_trid;
                carve.pull_trid = 0;
            }
            carve.state = CARVE_IDLE;
        }
        if (client.state == SLOT_DETACHING && client.inflight == 0) {
            client.state = SLOT_IDLE;
            slot.ack = abi::slot_word(abi::SLOT_FREED, client.gen);
        }
        if (client.state == SLOT_IDLE) {
            try_attach(s);
        }
        any_client |= client.state != SLOT_IDLE;
    }
    if (any_client) {
        return true;
    }
    // A client may be attaching right now: it saw RUNNING and wrote its
    // request after the pass above. Announce STOPPING and look once more, so it
    // is served without a relaunch. Correctness does not rest on this look: a
    // client launches only on STOPPED, and on STOPPING waits for ack or STOPPED.
    command_block->header.state = abi::SERVICE_STOPPING;
    for (uint32_t s = 0; s < abi::MAX_CLIENTS; s++) {
        any_client |= try_attach(s);
    }
    if (any_client) {
        command_block->header.state = abi::SERVICE_RUNNING;
    }
    return any_client;
}

int main() {
    crt1();
    noc_boot();
    scratch_watermark = noc_nonposted_writes_checkpoint(noc_index);
    dma_boot();
    bank_size = command_block->header.bank_size;
    for (uint32_t i = 0; i < abi::BANK_ENDPOINTS; i++) {
        bank_noc_xy_addr[i] = command_block->header.bank_noc_xy_addr[i];
    }
    command_block->header.build.magic = abi::BUILD_BLOCK_MAGIC;
    command_block->header.build.abi_version = abi::ABI_VERSION;
    command_block->header.state = abi::SERVICE_RUNNING;

    TransactionRing txn_ring;
    BounceArena bounce_arena(l1::BOUNCE_BUFFER_START, l1::BOUNCE_BUFFER_SIZE);
    DmaInFlight dma;

    // main loop: retire -> accept -> issue writes, one non-blocking pass per tick. Ticks
    // never wait on NOC completion, so slot scans keep running through
    // sustained load. The first tick scans, attaching the launching client.
    for (uint32_t spins = 0;; ++spins) {
        if (spins % abi::SLOT_SCAN_TICK_INTERVAL == 0 && !scan_slots()) {
            break;
        }
        const uint32_t done_changed = retire_transactions(txn_ring, bounce_arena, dma);
        accept_transactions(txn_ring, bounce_arena, dma);
        issue_writes(txn_ring, dma, done_changed);
    }

    // No client is left, so every transaction retired and every carve is idle.
    // The last done_count publish must land before STOPPED tells the host it may reset the core.
    while (!ncrisc_noc_nonposted_writes_flushed(noc_index));
    command_block->header.state = abi::SERVICE_STOPPED;
    return 0;
}
