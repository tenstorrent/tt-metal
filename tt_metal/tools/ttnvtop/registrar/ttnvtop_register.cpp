// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Writer side of the ttnvtop program registry. See
// `../common/program_registry.hpp` for the on-disk schema and
// `ttnvtop_register.hpp` for the public API contract.
//
// Design notes:
//   * The hot path (env var unset) is a single atomic-bool load: we cache
//     the enable decision inside `WriterState` and the function-local
//     static initializer runs exactly once per process, guarded by C++11
//     magic statics.
//   * Initialization (env-var check, open, ftruncate, mmap, header init)
//     lives in a `__attribute__((cold))` helper so the compiler parks it
//     away from hot-path code.
//   * Per-call concurrency is lock-free: `fetch_add` on
//     `header->write_cursor` claims a slot, then a plain memcpy writes
//     the entry. The circular buffer has no drop path — the newest
//     `kRegistryCapacity` entries always win, which is what the viewer
//     expects.
//   * Failure modes (missing /dev/shm, ftruncate EPERM, mmap refused) log
//     one stderr line and flip `enabled=false`; `register_program` keeps
//     returning instantly thereafter.

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../common/program_registry.hpp"
#include "ttnvtop_register.hpp"

namespace ttnvtop {
namespace {

struct WriterState {
    bool enabled = false;
    int fd = -1;
    void* map = nullptr;
    size_t map_size = 0;
    RegistryHeader* header = nullptr;
    RegistryEntry* entries = nullptr;
};

// Cold path: runs exactly once per process via magic-static init. Checks
// the env-var gate, maps the SHM file, and (if we created it fresh)
// initializes the header. Any failure leaves `enabled=false`.
__attribute__((cold)) WriterState init_state() {
    WriterState s;

    const char* gate = std::getenv(kRegistryEnvVar);
    if (gate == nullptr || std::strcmp(gate, "1") != 0) {
        return s;  // disabled; hot path will see enabled=false
    }

    const size_t want = registry_file_size();

    // O_CREAT | O_RDWR so the first-to-arrive writer creates the file.
    // 0644 is fine — /dev/shm is user-writable and the viewer only reads.
    const int fd = ::open(kRegistryShmPath, O_CREAT | O_RDWR, 0644);
    if (fd < 0) {
        std::fprintf(
            stderr,
            "ttnvtop_register: open(%s) failed: %s; registration disabled\n",
            kRegistryShmPath,
            std::strerror(errno));
        return s;
    }

    // ftruncate to the exact layout size. If another writer already sized
    // it correctly this is a no-op. If someone sized it smaller we grow.
    struct stat st{};
    if (::fstat(fd, &st) == 0 && static_cast<size_t>(st.st_size) < want) {
        if (::ftruncate(fd, static_cast<off_t>(want)) != 0) {
            std::fprintf(
                stderr,
                "ttnvtop_register: ftruncate(%s, %zu) failed: %s; registration disabled\n",
                kRegistryShmPath,
                want,
                std::strerror(errno));
            ::close(fd);
            return s;
        }
    } else if (st.st_size == 0) {
        // fstat failed or file was zero-sized; try to size it anyway.
        if (::ftruncate(fd, static_cast<off_t>(want)) != 0) {
            std::fprintf(
                stderr,
                "ttnvtop_register: ftruncate(%s, %zu) failed: %s; registration disabled\n",
                kRegistryShmPath,
                want,
                std::strerror(errno));
            ::close(fd);
            return s;
        }
    }

    void* map = ::mmap(nullptr, want, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        std::fprintf(
            stderr,
            "ttnvtop_register: mmap(%s, %zu) failed: %s; registration disabled\n",
            kRegistryShmPath,
            want,
            std::strerror(errno));
        ::close(fd);
        return s;
    }

    auto* header = static_cast<RegistryHeader*>(map);
    auto* entries = reinterpret_cast<RegistryEntry*>(static_cast<char*>(map) + sizeof(RegistryHeader));

    // Initialize the header if the magic is absent (fresh file or a stale
    // file from a previous writer with a different layout). We deliberately
    // do NOT reset write_cursor on attach to an existing file with valid
    // magic — that would race with a concurrent writer. Instead we overwrite
    // slots by wrapping, which is what the reader expects.
    if (std::memcmp(header->magic, kRegistryMagic, sizeof(kRegistryMagic)) != 0 ||
        header->version != kRegistryVersion || header->entry_size != static_cast<uint16_t>(sizeof(RegistryEntry)) ||
        header->capacity != kRegistryCapacity) {
        std::memcpy(header->magic, kRegistryMagic, sizeof(kRegistryMagic));
        header->version = kRegistryVersion;
        header->entry_size = static_cast<uint16_t>(sizeof(RegistryEntry));
        header->capacity = kRegistryCapacity;
        for (uint32_t i = 0; i < 4; ++i) {
            header->reserved[i] = 0;
        }
        header->write_cursor.store(0, std::memory_order_relaxed);
        // Zero the entry region so readers see deterministic empty slots.
        std::memset(entries, 0, static_cast<size_t>(kRegistryCapacity) * sizeof(RegistryEntry));
    }

    // Always refresh writer_pid/epoch_us — the most-recent writer owns these
    // fields and the viewer uses writer_pid to detect stale registries.
    header->writer_pid = static_cast<uint32_t>(::getpid());
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    header->epoch_us = static_cast<uint64_t>(ts.tv_sec) * 1'000'000ull + static_cast<uint64_t>(ts.tv_nsec) / 1000ull;

    s.enabled = true;
    s.fd = fd;
    s.map = map;
    s.map_size = want;
    s.header = header;
    s.entries = entries;
    return s;
}

WriterState& state() {
    static WriterState s = init_state();
    return s;
}

}  // namespace

void register_program(uint32_t runtime_id, const char* name) {
    WriterState& s = state();
    if (!s.enabled) {
        return;  // hot path exits here in one atomic-bool load
    }

    const uint32_t total = s.header->write_cursor.fetch_add(1, std::memory_order_relaxed);
    const uint32_t slot = total % kRegistryCapacity;
    RegistryEntry& e = s.entries[slot];

    e.runtime_id = runtime_id;
    e.pid = static_cast<uint32_t>(::getpid());
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    e.epoch_us = static_cast<uint64_t>(ts.tv_sec) * 1'000'000ull + static_cast<uint64_t>(ts.tv_nsec) / 1000ull;

    if (name == nullptr) {
        name = "(unnamed)";
    }
    // strnlen bounds the scan; -1 leaves room for the null terminator.
    const size_t n = ::strnlen(name, kRegistryNameMax - 1);
    std::memcpy(e.name, name, n);
    e.name[n] = '\0';
    // Zero the v2 cycle field. The collector will overwrite it once it
    // observes ring traffic for this kernel_id.
    e.cycles_in_window = 0u;
}

}  // namespace ttnvtop
