// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Engram host work in C++ (ENGRAM_HOST_IMPL=cpp): the n-gram hash, the bfloat8_b packer and the gather of packed
// table rows into TILE bytes; the same integer arithmetic as tt/engram_bfp8.py's numba kernels, bit for bit
// (tests/test_engram_host_cpp.py). Built and loaded by tt/engram_host_cpp.py (torch.utils.cpp_extension.load; no
// ttnn build). Every entry point takes raw pointers the Python wrapper checked, releases the GIL once for the whole
// call and splits the work over a persistent pool of std::threads (``threads`` a call, the caller among them).

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

// ---------------------------------------------------------------------------------------------- the thread pool
// A call with ``threads`` wakes workers 0 .. threads - 2 only (each its own futex word: the others keep sleeping);
// they and the caller take task indices from one atomic counter. A woken worker spins ``spin_us`` after its last task
// (the next round of the same call, e.g. the gather after the hash, starts on hot threads), then sleeps. One call at a
// time (callers from several Python threads queue on run_mutex).
inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

inline void futex_wait(std::atomic<uint32_t>* addr, uint32_t seen) {
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(addr), FUTEX_WAIT_PRIVATE, seen, nullptr, nullptr, 0);
}

inline void futex_wake(std::atomic<uint32_t>* addr) {
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(addr), FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
}

class Pool {
public:
    std::atomic<int64_t> spin_us{100};

    void parallel_for(int64_t n_tasks, int threads, const std::function<void(int64_t)>& fn) {
        if (n_tasks <= 0) {
            return;
        }
        threads = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>({threads, n_tasks, kMaxWorkers + 1})));
        if (threads == 1) {
            for (int64_t i = 0; i < n_tasks; ++i) {
                fn(i);
            }
            return;
        }
        std::lock_guard<std::mutex> run(run_mutex_);
        grow(threads - 1);
        fn_ = &fn;
        n_tasks_ = n_tasks;
        next_.store(0, std::memory_order_relaxed);
        pending_.store(threads - 1, std::memory_order_relaxed);
        for (int i = 0; i < threads - 1; ++i) {
            slots_[i].word.fetch_add(1, std::memory_order_release);
            futex_wake(&slots_[i].word);
        }
        drain();
        while (pending_.load(std::memory_order_acquire) != 0) {
            cpu_relax();
        }
        fn_ = nullptr;
    }

private:
    static constexpr int kMaxWorkers = 255;

    struct alignas(64) Slot {
        std::atomic<uint32_t> word{0};  // (bumped once a call that wakes this worker: the futex word)
    };

    void grow(int n) {
        while (n_workers_ < n) {
            const int idx = n_workers_++;
            std::thread(&Pool::loop, this, idx, slots_[idx].word.load()).detach();  // (lives as long as the process)
        }
    }

    void drain() {
        for (;;) {
            int64_t i = next_.fetch_add(1, std::memory_order_relaxed);
            if (i >= n_tasks_) {
                return;
            }
            (*fn_)(i);
        }
    }

    void loop(int idx, uint32_t seen) {
        std::atomic<uint32_t>& word = slots_[idx].word;
        for (;;) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto spin = std::chrono::microseconds(spin_us.load(std::memory_order_relaxed));
            uint32_t g;
            int k = 0;
            while ((g = word.load(std::memory_order_acquire)) == seen) {
                cpu_relax();
                if ((++k & 255) == 0 && std::chrono::steady_clock::now() - t0 > spin) {
                    futex_wait(&word, seen);  // (returns at once if the word moved on: no lost wake-up)
                }
            }
            seen = g;
            drain();
            pending_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    Slot slots_[kMaxWorkers];
    int n_workers_ = 0;
    std::mutex run_mutex_;
    const std::function<void(int64_t)>* fn_ = nullptr;
    int64_t n_tasks_ = 0;
    std::atomic<int64_t> next_{0};
    std::atomic<int> pending_{0};
};

Pool& pool() {
    static Pool* p = new Pool();  // (never destroyed: no join at interpreter exit while a worker may still sleep)
    return *p;
}

// ---------------------------------------------------------------------------------------------- the n-gram hash
struct HashArgs {
    const int64_t* ids;        // [s]
    const int64_t* tail;       // [tail_len]
    const int64_t* token_map;  // [vocab]
    const int64_t* mult;       // [m]
    const int64_t* primes;     // [m - 1, heads]
    const int64_t* offsets;    // [(m - 1) * heads]
    int64_t s, tail_len, vocab, m, heads, pad_id, dead;
    int64_t* out;          // [s, (m - 1) * heads]
    int64_t num_rows = 0;  // > 0: also count the rows outside [0, num_rows) (the gather's range check, fused)
};

constexpr int64_t kHashBlock = 256;

// engram_bfp8._hash_kernel: per position the rolling XOR of tok * mult[shift] (int64 multiply wrapping), the
// remainder with Python's sign rule (the divisor's sign), plus the (n, head) offset. Returns ids outside the vocab;
// adds the rows outside the table to ``bad_rows`` (num_rows > 0).
int64_t hash_block(const HashArgs& a, int64_t blk, std::atomic<int64_t>& bad_rows) {
    int64_t out_of_table = 0;
    const int64_t t0 = blk * kHashBlock, t1 = std::min(a.s, t0 + kHashBlock);
    const int64_t width = (a.m - 1) * a.heads;
    int64_t bad = 0;
    for (int64_t t = t0; t < t1; ++t) {
        bool blocked = false;
        uint64_t rolling = 0;
        for (int64_t shift = 0; shift < a.m; ++shift) {
            const int64_t j = a.tail_len - shift + t;
            int64_t src;
            if (j < a.tail_len) {
                src = a.tail[j];
            } else {
                const int64_t id = a.ids[j - a.tail_len];
                if (id < 0 || id >= a.vocab) {
                    ++bad;
                    src = a.dead;
                } else {
                    src = a.token_map[id];
                }
            }
            if (src == a.dead) {
                blocked = true;
            }
            const int64_t tok = blocked ? a.pad_id : src;
            const uint64_t prod = static_cast<uint64_t>(tok) * static_cast<uint64_t>(a.mult[shift]);
            if (shift == 0) {
                rolling = prod;
                continue;
            }
            rolling ^= prod;
            const int64_t r0 = static_cast<int64_t>(rolling);
            for (int64_t h = 0; h < a.heads; ++h) {
                const int64_t p = a.primes[(shift - 1) * a.heads + h];
                int64_t r = r0 % p;
                if (r != 0 && ((r < 0) != (p < 0))) {
                    r += p;
                }
                const int64_t c = (shift - 1) * a.heads + h;
                const int64_t row = r + a.offsets[c];
                a.out[t * width + c] = row;
                out_of_table += (row < 0) | (row >= a.num_rows);
            }
        }
    }
    if (a.num_rows > 0 && out_of_table) {
        bad_rows += out_of_table;
    }
    return bad;
}

int64_t run_hash(const HashArgs& a, int threads, std::atomic<int64_t>& bad_rows) {
    std::atomic<int64_t> bad{0};
    const int64_t blocks = (a.s + kHashBlock - 1) / kHashBlock;
    pool().parallel_for(blocks, threads, [&](int64_t b) {
        int64_t n = hash_block(a, b, bad_rows);
        if (n) {
            bad += n;
        }
    });
    return bad.load();
}

// ---------------------------------------------------------------------------------------------- the TILE gather
struct TileArgs {
    const uint8_t* packed;  // [num_rows, row_bytes] (g exponent bytes, then g * 16 magnitudes)
    int64_t num_rows, row_bytes;
    const int64_t* ids;  // [>= valid, slots]
    int64_t slots, valid, q;
    std::vector<uint8_t*> outs;  // per chip [q * slots * g * 16 * 17 / 16]
};

constexpr int64_t kTileRows = 32;

// engram_bfp8._tile_chips_kernel for rows [32 b, 32 b + 32) of the chunk (a tile row of one chip: q % 32 == 0).
void tile_block(const TileArgs& a, int64_t blk) {
    // (every field in a local: the uint8_t stores may alias anything, so fields read through ``a`` reload each store)
    const int64_t g = a.row_bytes / 17, row_bytes = a.row_bytes, slots = a.slots, q = a.q, valid = a.valid;
    const int64_t tcols = slots * g / 2;
    const int64_t s = q * static_cast<int64_t>(a.outs.size());
    const uint8_t* const packed = a.packed;
    const int64_t* const ids = a.ids;
    const int64_t t0 = blk * kTileRows, t1 = std::min(s, t0 + kTileRows);
    for (int64_t t = t0; t < t1; ++t) {
        uint8_t* const out = a.outs[t / q];
        const int64_t lt = t % q;
        const int64_t tr = lt / 32, ri = lt % 32;
        const int64_t fr = ri / 16, rr = ri % 16;
        for (int64_t j = 0; j < slots; ++j) {
            const int64_t rid = t < valid ? ids[t * slots + j] : -1;
            const uint8_t* const row = rid < 0 ? nullptr : packed + rid * row_bytes;
            for (int64_t k = 0; k < g; ++k) {
                const int64_t col = j * g + k;  // (16-column group: c0 = 16 col)
                const int64_t f = fr * 2 + (col & 1);
                uint8_t* const tile = out + (tr * tcols + (col >> 1)) * 1088;
                uint8_t* const mo = tile + 64 + f * 256 + rr * 16;
                if (row == nullptr) {
                    tile[f * 16 + rr] = 0;
                    std::memset(mo, 0, 16);
                } else {
                    tile[f * 16 + rr] = row[k];
                    std::memcpy(mo, row + g + k * 16, 16);
                }
            }
        }
    }
}

int64_t count_bad(const TileArgs& a, int threads) {
    const int64_t s = a.q * static_cast<int64_t>(a.outs.size());
    const int64_t n = std::min(a.valid, s);
    std::atomic<int64_t> bad{0};
    pool().parallel_for((n + 1023) / 1024, threads, [&](int64_t b) {
        const int64_t e = std::min(n, (b + 1) * 1024) * a.slots;
        int64_t c = 0;
        for (int64_t i = b * 1024 * a.slots; i < e; ++i) {
            c += (a.ids[i] < 0) | (a.ids[i] >= a.num_rows);
        }
        if (c) {
            bad += c;
        }
    });
    return bad.load();
}

// Returns the ids outside the table (then no byte is written), else 0. ``checked``: the caller counted them.
int64_t run_tile(const TileArgs& a, int threads, bool checked = false) {
    const int64_t bad = checked ? 0 : count_bad(a, threads);
    if (bad) {
        return bad;
    }
    const int64_t s = a.q * static_cast<int64_t>(a.outs.size());
    // (contiguous rows a thread, as numba's static prange: each thread streams through its own chips' bytes)
    const int64_t blocks = (s + kTileRows - 1) / kTileRows;
    const int64_t parts = std::min<int64_t>(threads, blocks);
    pool().parallel_for(parts, threads, [&](int64_t p) {
        for (int64_t b = blocks * p / parts; b < blocks * (p + 1) / parts; ++b) {
            tile_block(a, b);
        }
    });
    return 0;
}

// ---------------------------------------------------------------------------------------------- the bfp8 packer
// engram_bfp8._numba_kernel (pack): per 16 values the largest (exp_t[b] + scale) of the non-zero bytes (from 0),
// stored as uint8 (wraps as numba's store), and per value byte_t[b, min(gap, 31)] (0 for +-0).
constexpr int64_t kPackBlock = 64;

}  // namespace

// ---------------------------------------------------------------------------------------------- the bindings
int64_t ngram_hash(
    uintptr_t ids,
    uintptr_t tail,
    uintptr_t token_map,
    uintptr_t mult,
    uintptr_t primes,
    uintptr_t offsets,
    int64_t s,
    int64_t tail_len,
    int64_t vocab,
    int64_t m,
    int64_t heads,
    int64_t pad_id,
    int64_t dead,
    uintptr_t out,
    int threads) {
    HashArgs a{
        reinterpret_cast<const int64_t*>(ids),
        reinterpret_cast<const int64_t*>(tail),
        reinterpret_cast<const int64_t*>(token_map),
        reinterpret_cast<const int64_t*>(mult),
        reinterpret_cast<const int64_t*>(primes),
        reinterpret_cast<const int64_t*>(offsets),
        s,
        tail_len,
        vocab,
        m,
        heads,
        pad_id,
        dead,
        reinterpret_cast<int64_t*>(out)};
    std::atomic<int64_t> unused{0};
    return run_hash(a, threads, unused);
}

int64_t tile_chips(
    uintptr_t packed,
    int64_t num_rows,
    int64_t row_bytes,
    uintptr_t ids,
    int64_t slots,
    int64_t valid,
    const std::vector<uintptr_t>& outs,
    int64_t q,
    int threads) {
    TileArgs a{
        reinterpret_cast<const uint8_t*>(packed),
        num_rows,
        row_bytes,
        reinterpret_cast<const int64_t*>(ids),
        slots,
        valid,
        q,
        {}};
    for (auto p : outs) {
        a.outs.push_back(reinterpret_cast<uint8_t*>(p));
    }
    return run_tile(a, threads);
}

// The HOST_HIDE job of one engram layer in one call (one GIL release): the hash of ids [valid] into h [valid, slots],
// then those rows' TILE bytes into every chip's share. Returns (ids outside the vocab, rows outside the table).
std::pair<int64_t, int64_t> hash_tile_chips(
    uintptr_t ids,
    uintptr_t tail,
    uintptr_t token_map,
    uintptr_t mult,
    uintptr_t primes,
    uintptr_t offsets,
    int64_t valid,
    int64_t tail_len,
    int64_t vocab,
    int64_t m,
    int64_t heads,
    int64_t pad_id,
    int64_t dead,
    uintptr_t h,
    uintptr_t packed,
    int64_t num_rows,
    int64_t row_bytes,
    const std::vector<uintptr_t>& outs,
    int64_t q,
    int threads) {
    HashArgs ha{
        reinterpret_cast<const int64_t*>(ids),
        reinterpret_cast<const int64_t*>(tail),
        reinterpret_cast<const int64_t*>(token_map),
        reinterpret_cast<const int64_t*>(mult),
        reinterpret_cast<const int64_t*>(primes),
        reinterpret_cast<const int64_t*>(offsets),
        valid,
        tail_len,
        vocab,
        m,
        heads,
        pad_id,
        dead,
        reinterpret_cast<int64_t*>(h),
        num_rows};
    std::atomic<int64_t> bad_rows{0};
    const int64_t bad_ids = run_hash(ha, threads, bad_rows);
    if (bad_ids || bad_rows.load()) {
        return {bad_ids, bad_rows.load()};
    }
    TileArgs ta{
        reinterpret_cast<const uint8_t*>(packed),
        num_rows,
        row_bytes,
        reinterpret_cast<const int64_t*>(h),
        (m - 1) * heads,
        valid,
        q,
        {}};
    for (auto p : outs) {
        ta.outs.push_back(reinterpret_cast<uint8_t*>(p));
    }
    return {0, run_tile(ta, threads, /*checked=*/true)};
}

// TtEngramRows.fill_tp's host work (HOST_HIDE=on, resident table) in one call: the valid prefix of the chunk's ids
// (``valid`` < 0: up to the first id outside the vocabulary, the engine's pad, as TtEngramRows._valid), the new tail
// (TtEngramRows.next_tail; checked against ``want_tail`` when given), the hash of ids [valid] and those rows' TILE
// bytes into every chip's share (rows past valid 0). ids int64 or int32 (``ids_int32``). Returns (valid, ids outside
// the vocabulary, rows outside the table, tail matches want_tail, hash us, gather us); on a bad id / row nothing is
// written to ``outs``.
py::tuple engram_job(
    uintptr_t ids_p,
    bool ids_int32,
    int64_t s,
    int64_t valid,
    uintptr_t tail_p,
    uintptr_t want_tail_p,
    uintptr_t new_tail_p,
    uintptr_t token_map,
    uintptr_t mult,
    uintptr_t primes,
    uintptr_t offsets,
    int64_t tail_len,
    int64_t vocab,
    int64_t m,
    int64_t heads,
    int64_t pad_id,
    int64_t dead,
    uintptr_t packed,
    int64_t num_rows,
    int64_t row_bytes,
    const std::vector<uintptr_t>& outs,
    int64_t q,
    int threads) {
    int64_t r_valid = 0, r_bad_ids = 0, r_bad_rows = 0;
    bool tail_ok = true;
    double hash_us = 0, tile_us = 0;
    {
        py::gil_scoped_release nogil;
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<int64_t> ids64;
        const int64_t* ids = reinterpret_cast<const int64_t*>(ids_p);
        if (ids_int32) {
            const int32_t* i32 = reinterpret_cast<const int32_t*>(ids_p);
            ids64.assign(i32, i32 + s);
            ids = ids64.data();
        }
        if (valid < 0) {
            valid = s;
            for (int64_t t = 0; t < s; ++t) {
                if (ids[t] < 0 || ids[t] >= vocab) {
                    valid = t;
                    break;
                }
            }
        }
        valid = std::max<int64_t>(0, std::min(valid, s));
        r_valid = valid;
        // the new tail: [tail | token_map[ids[valid - tail_len : valid]]][-tail_len:]
        const int64_t* tail = reinterpret_cast<const int64_t*>(tail_p);
        const int64_t* tmap = reinterpret_cast<const int64_t*>(token_map);
        std::vector<int64_t> comp(tail, tail + tail_len);
        for (int64_t t = std::max<int64_t>(0, valid - tail_len); t < valid; ++t) {
            comp.push_back(ids[t] >= 0 && ids[t] < vocab ? tmap[ids[t]] : dead);
        }
        int64_t* new_tail = reinterpret_cast<int64_t*>(new_tail_p);
        for (int64_t i = 0; i < tail_len; ++i) {
            new_tail[i] = comp[comp.size() - tail_len + i];
        }
        if (want_tail_p) {
            const int64_t* want = reinterpret_cast<const int64_t*>(want_tail_p);
            tail_ok = std::equal(new_tail, new_tail + tail_len, want);
        }
        const int64_t slots = (m - 1) * heads;
        std::vector<int64_t> h(static_cast<size_t>(valid * slots));
        std::atomic<int64_t> bad_rows{0};
        if (valid) {
            HashArgs ha{
                ids,
                tail,
                tmap,
                reinterpret_cast<const int64_t*>(mult),
                reinterpret_cast<const int64_t*>(primes),
                reinterpret_cast<const int64_t*>(offsets),
                valid,
                tail_len,
                vocab,
                m,
                heads,
                pad_id,
                dead,
                h.data(),
                num_rows};
            r_bad_ids = run_hash(ha, threads, bad_rows);
        }
        r_bad_rows = bad_rows.load();
        const auto t1 = std::chrono::steady_clock::now();
        if (r_bad_ids == 0 && r_bad_rows == 0) {
            TileArgs ta{reinterpret_cast<const uint8_t*>(packed), num_rows, row_bytes, h.data(), slots, valid, q, {}};
            for (auto p : outs) {
                ta.outs.push_back(reinterpret_cast<uint8_t*>(p));
            }
            run_tile(ta, threads, /*checked=*/true);
        }
        const auto t2 = std::chrono::steady_clock::now();
        hash_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        tile_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
    }
    return py::make_tuple(r_valid, r_bad_ids, r_bad_rows, tail_ok, hash_us, tile_us);
}

void pack(
    uintptr_t w,
    int64_t w_stride,
    uintptr_t s,
    int64_t s_stride,
    int64_t n,
    int64_t d,
    int64_t n_scales,
    uintptr_t exps,
    int64_t e_stride,
    uintptr_t mags,
    int64_t m_stride,
    uintptr_t exp_t,
    uintptr_t zero_t,
    uintptr_t byte_t,
    int64_t gaps,
    int threads) {
    const uint8_t* W = reinterpret_cast<const uint8_t*>(w);
    const uint8_t* S = reinterpret_cast<const uint8_t*>(s);
    uint8_t* E = reinterpret_cast<uint8_t*>(exps);
    uint8_t* M = reinterpret_cast<uint8_t*>(mags);
    const int32_t* ET = reinterpret_cast<const int32_t*>(exp_t);
    const uint8_t* ZT = reinterpret_cast<const uint8_t*>(zero_t);  // (bool bytes)
    const uint8_t* BT = reinterpret_cast<const uint8_t*>(byte_t);  // [256, gaps]
    const int64_t per_scale = d / n_scales;
    pool().parallel_for((n + kPackBlock - 1) / kPackBlock, threads, [&](int64_t b) {
        const int64_t r1 = std::min(n, (b + 1) * kPackBlock);
        for (int64_t r = b * kPackBlock; r < r1; ++r) {
            const uint8_t* wr = W + r * w_stride;
            const uint8_t* sr = S + r * s_stride;
            uint8_t* mr = M + r * m_stride;
            for (int64_t k = 0; k < d / 16; ++k) {
                const int32_t sc = sr[(k * 16) / per_scale];
                int64_t mx = 0;
                for (int j = 0; j < 16; ++j) {
                    const uint8_t v = wr[k * 16 + j];
                    if (!ZT[v]) {
                        mx = std::max<int64_t>(mx, ET[v] + sc);
                    }
                }
                E[r * e_stride + k] = static_cast<uint8_t>(mx);
                for (int j = 0; j < 16; ++j) {
                    const uint8_t v = wr[k * 16 + j];
                    mr[k * 16 + j] = ZT[v] ? 0 : BT[v * gaps + std::min<int64_t>(mx - (ET[v] + sc), gaps - 1)];
                }
            }
        }
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.doc() = "engram host work (hash, bfp8 pack, TILE gather) in C++; see tt/engram_host_cpp.py";
    auto nogil = py::call_guard<py::gil_scoped_release>();
    mod.def("ngram_hash", &ngram_hash, nogil);
    mod.def("tile_chips", &tile_chips, nogil);
    mod.def("hash_tile_chips", &hash_tile_chips, nogil);
    mod.def("pack", &pack, nogil);
    mod.def("engram_job", &engram_job);  // (releases the GIL itself: it builds its result tuple after)
    mod.def("set_spin_us", [](int64_t us) { pool().spin_us.store(us); });
}
