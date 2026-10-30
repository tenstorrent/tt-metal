// Stage 1 placement test (compile-only): a sender-like loop. MODE selects how the forwarding records are placed:
//   0  nothing (the compiler's choice; today's router)
//   1  asm volatile("" : "+m"(record)) at the end of each record's visit
//   2  volatile record fields
//   3  asm("" : "+m"(record)) (not volatile) at the end of each record's visit
//   4  asm volatile("" : "+m"(table)) once per pass, after the forwarding visits
//   5  asm volatile("" : "+m"(table)) once, before the loop (outside the contract: nothing holds it in the loop)
// N is the number of forwarding records (1..8). COLD=1 keeps the credit address in the record; COLD=0 reads it from
// a caller-owned array at credit time.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>

#ifndef MODE
#define MODE 0
#endif
#ifndef N
#define N 3
#endif
#ifndef COLD
#define COLD 1
#endif
#ifndef IFSEP
#define IFSEP 0  // 1: in_flight is a loop local, not a table member
#endif
#define FORCE_INLINE inline __attribute__((always_inline))

template <uint32_t A>
FORCE_INLINE uint32_t rd() { return *reinterpret_cast<volatile uint32_t*>(A); }
template <uint32_t A>
FORCE_INLINE void wr(uint32_t v) { *reinterpret_cast<volatile uint32_t*>(A) = v; }

template <typename T>
FORCE_INLINE void keep_in_memory(T& obj) {
#if MODE == 1 || MODE == 4
    asm volatile("" : "+m"(obj));
#elif MODE == 3
    asm("" : "+m"(obj));
#endif
}

#if MODE == 2
#define VF volatile
#else
#define VF
#endif

constexpr uint32_t SIZE = 4352;
constexpr uint32_t FIRST[9] = {0x10000, 0x20000, 0x30000, 0x40000, 0x50000, 0x60000, 0x70000, 0x80000, 0x90000};
constexpr uint32_t LAST(size_t ch) { return FIRST[ch] + 7 * SIZE; }
constexpr uint32_t STREAM(size_t ch) { return 0xFFB40000 + ch * 0x1000; }
constexpr uint32_t COMPLETION(size_t ch) { return 0x3000 + ch * 4; }
constexpr uint32_t TXQ = 0xFFB90000;

struct Record {
    VF uint32_t slot;
    VF uint32_t sent;
    VF uint32_t completed;
#if COLD
    VF uint64_t credit_addr;
#endif
};
struct Table {
    Record r[N];
#if !IFSEP
    uint32_t in_flight;
#endif
};
#if IFSEP
#define IN_FLIGHT in_flight
#else
#define IN_FLIGHT t.in_flight
#endif
struct FarRing {
    uint32_t free_slots;
    uint32_t next_slot;
};
struct Worker {
    uint32_t slot, sent, completed;
    volatile uint32_t* semaphore;
    uint32_t expected;
};

#ifndef CALLS
#define CALLS 1
#endif
#if CALLS
extern void credit_upstream(uint64_t addr, uint32_t n);  // out of line, as the router's credit helpers
extern void credit_worker(uint32_t n);
extern uint32_t connection_event(uint32_t semaphore, uint32_t expected);
#else  // the same effects as register writes, no calls in the loop
FORCE_INLINE void credit_upstream(uint64_t addr, uint32_t n) { wr<0xFFB20000>(uint32_t(addr)); wr<0xFFB20004>(uint32_t(addr >> 32)); wr<0xFFB20008>(n); }
FORCE_INLINE void credit_worker(uint32_t n) { wr<0xFFB2000c>(n); }
FORCE_INLINE uint32_t connection_event(uint32_t semaphore, uint32_t expected) { wr<0xFFB20010>(semaphore); return semaphore ^ expected; }
#endif

// Send one packet from `slot` into the far ring (a few register writes, as the router's send).
template <size_t CH>
FORCE_INLINE void send(uint32_t& slot, FarRing& far, uint32_t& sent) {
    wr<TXQ + 0x10>(slot);
    wr<TXQ + 0x14>(far.next_slot);
    wr<TXQ + 0x18>(SIZE);
    wr<TXQ + 0x1c>(1);
    far.next_slot = far.next_slot == 0x90000 + 7 * SIZE ? 0x90000 : far.next_slot + SIZE;
    far.free_slots--;
    slot = slot == LAST(CH) ? FIRST[CH] : slot + SIZE;
    sent++;
    wr<STREAM(CH) + 0x20>(1);
}

// The free-slots registers of channels 0..N, read as one group between fences.
template <size_t... I>
FORCE_INLINE uint32_t pending(std::index_sequence<I...>) {
    uint32_t regs[sizeof...(I)];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    ((regs[I] = rd<STREAM(I)>()), ...);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return ((static_cast<uint32_t>(regs[I] != 8) << I) | ...);
}

FORCE_INLINE void visit_worker(Worker& w, bool has_packet, FarRing& far) {
    if (has_packet && far.free_slots != 0) {
        send<0>(w.slot, far, w.sent);
    }
    const bool read = w.sent != w.completed;
    uint32_t reported = 0;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (read) {
        reported = rd<COMPLETION(0)>();
    }
    const uint32_t semaphore = *w.semaphore;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (read && reported != w.completed) {
        far.free_slots += reported - w.completed;
        w.completed = reported;
        credit_worker(reported);
    }
    if (semaphore != w.expected) {
        w.expected = connection_event(semaphore, w.expected);
    }
}

template <size_t CH>
FORCE_INLINE void visit_forwarding(Table& t, uint32_t& in_flight, bool has_packet, FarRing& far, const uint64_t* upstream) {
    Record& r = t.r[CH - 1];
    uint32_t sent = r.sent;
    uint32_t completed = r.completed;
    if (has_packet && far.free_slots != 0) {
        uint32_t slot = r.slot;
        send<CH>(slot, far, sent);
        r.slot = slot;
        r.sent = sent;
    }
    if (sent != completed) {
        const uint32_t c = rd<COMPLETION(CH)>() - completed;
        if (c != 0) {
            completed += c;
            r.completed = completed;
            far.free_slots += c;
#if COLD
            credit_upstream(r.credit_addr, c);
#else
            credit_upstream(upstream[CH - 1], c);
#endif
        }
    }
    IN_FLIGHT = sent != completed ? (IN_FLIGHT | (1u << CH)) : (IN_FLIGHT & ~(1u << CH));
    keep_in_memory(r);
}

template <size_t... I>
FORCE_INLINE void visit_all(Table& t, uint32_t& in_flight, uint32_t pend, FarRing& far, const uint64_t* upstream, std::index_sequence<I...>) {
    const uint32_t flying = IN_FLIGHT;
    ((((pend | flying) >> (I + 1)) & 1u ? visit_forwarding<I + 1>(t, in_flight, ((pend >> (I + 1)) & 1u) != 0, far, upstream)
                                            : void()),
     ...);
}

__attribute__((noinline)) void run(const uint64_t* upstream, volatile uint32_t* semaphore) {
    Table t{};
#if COLD
    for (size_t i = 0; i < N; i++) {
        t.r[i].credit_addr = upstream[i];
    }
#endif
    for (size_t i = 0; i < N; i++) {
        t.r[i].slot = FIRST[i + 1];
    }
#if MODE == 5  // once, before the loop only
    asm volatile("" : "+m"(t));
#endif
    Worker w{FIRST[0], 0, 0, semaphore, 0};
    FarRing far{8, 0x90000};
    uint32_t rounds = 0;
    uint32_t in_flight = 0;
    while (rd<0x20100>() == 0) {  // termination flag in L1
        for (int n = 0; n < 32; n++) {
            const uint32_t pend = pending(std::make_index_sequence<N + 1>{});
            visit_worker(w, (pend & 1u) != 0, far);
            if (((pend | IN_FLIGHT) & (((1u << N) - 1) << 1)) != 0) {
                visit_all(t, in_flight, pend, far, upstream, std::make_index_sequence<N>{});
            }
#if MODE == 4
            keep_in_memory(t);
#endif
        }
        rounds += w.sent;
        wr<0x20104>(rounds);
    }
}
