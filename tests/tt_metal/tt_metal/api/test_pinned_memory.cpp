// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Coverage for the device-access mode of experimental::PinnedMemory: which mapping permissions Create() will
// hand out, how PinnedMemoryCache reconciles a request against an already-cached mapping with different
// permissions, and what the transfer paths do when a caller supplies a mapping the device may only read.
//
// Device-based tests use GenericMeshDeviceFixture and require hardware access.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <tt_stl/span.hpp>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include "impl/context/metal_context.hpp"
#include "tt_metal/distributed/pinned_memory_cache.hpp"
#include "tt_metal/impl/dispatch/vector_aligned.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using PinnedMemoryTestSuite = GenericMeshDeviceFixture;

// A device-read-only mapping cannot be a D2H destination: the device would NOC-write into an IOMMU mapping that
// faults, which shows up as a hung read rather than a test failure. issue_read_buffer_dispatch_command_sequence
// guards against that by excluding read-only mappings from the direct pinned-write path. Nothing else in the tree
// reaches that guard -- every in-tree read pins ReadWrite -- so this attaches a read-only pin to a read transfer
// explicitly and checks the data still arrives via the fallback.
TEST_F(PinnedMemoryTestSuite, EnqueueReadShardsWithReadOnlyPinnedMemoryUsesUnpinnedPath) {
    const auto pinning_params = experimental::GetMemoryPinningParameters(*mesh_device_);
    if (!pinning_params.can_map_to_noc || !pinning_params.supports_read_only) {
        GTEST_SKIP() << "Device-read-only pinned NOC mappings are not available";
        return;
    }
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size, .buffer_type = BufferType::DRAM, .bottom_up = true};

    const uint32_t tiles_per_device = 128;
    const uint32_t bytes_per_device = tiles_per_device * single_tile_size;

    ReplicatedBufferConfig global_buffer_config{.size = bytes_per_device};
    auto mesh_buffer = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    std::vector<uint32_t> src(bytes_per_device / sizeof(uint32_t), 0);
    std::iota(src.begin(), src.end(), 0);

    distributed::MeshCoordinate coord(0, 0);
    auto write_transfer = distributed::ShardDataTransfer{coord}
                              .host_data(static_cast<void*>(const_cast<uint32_t*>(src.data())))
                              .region(BufferRegion(0, bytes_per_device));
    mesh_device_->mesh_command_queue().enqueue_write_shards(mesh_buffer, {write_transfer}, /*blocking=*/true);

    auto dst = std::make_shared<vector_aligned<uint32_t>>(bytes_per_device / sizeof(uint32_t), 0);
    uint32_t* dst_ptr_aligned = reinterpret_cast<uint32_t*>(dst->data());
    HostBuffer host_buffer(ttsl::Span<uint32_t>(dst_ptr_aligned, bytes_per_device / sizeof(uint32_t)), MemoryPin(dst));

    // The host memory is writable; only the device's view of it is restricted, which is what makes this the case
    // the guard has to catch rather than one the driver would reject up front.
    auto coordinate_range_set = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    std::shared_ptr<experimental::PinnedMemory> pinned_read_only = experimental::PinnedMemory::Create(
        *mesh_device_,
        coordinate_range_set,
        host_buffer,
        /*map_to_noc=*/true,
        experimental::PinnedMemoryDeviceAccess::ReadOnly);
    ASSERT_TRUE(pinned_read_only);
    ASSERT_EQ(pinned_read_only->get_device_access(), experimental::PinnedMemoryDeviceAccess::ReadOnly);

    auto read_transfer = distributed::ShardDataTransfer{coord}
                             .host_data(static_cast<void*>(dst_ptr_aligned))
                             .region(BufferRegion(0, bytes_per_device));
    experimental::ShardDataTransferSetPinnedMemory(read_transfer, pinned_read_only);
    mesh_device_->mesh_command_queue().enqueue_read_shards({read_transfer}, mesh_buffer, /*blocking=*/true);

    std::vector<uint32_t> dst_aligned(dst_ptr_aligned, dst_ptr_aligned + (bytes_per_device / sizeof(uint32_t)));

    EXPECT_EQ(dst_aligned, src);
}

TEST_F(PinnedMemoryTestSuite, PinnedMemoryCacheReadOnlyRequestReusesReadWriteMapping) {
    const auto pinning_params = experimental::GetMemoryPinningParameters(*mesh_device_);
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes() == 0 ||
        !pinning_params.can_map_to_noc) {
        GTEST_SKIP() << "Pinned NOC mappings are not available";
    }

    auto storage = std::make_shared<vector_aligned<uint32_t>>(1024, 0);
    HostBuffer host_buffer(ttsl::Span<uint32_t>(storage->data(), storage->size()), MemoryPin(storage));
    const auto coord = *MeshCoordinateRange(mesh_device_->shape()).begin();
    const auto range = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    auto& cache = experimental::PinnedMemoryCache::instance();

    auto read_write =
        cache.try_pin(*mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadWrite);
    ASSERT_TRUE(read_write);
    ASSERT_EQ(read_write->get_device_access(), experimental::PinnedMemoryDeviceAccess::ReadWrite);

    auto read_only =
        cache.try_pin(*mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadOnly);
    ASSERT_TRUE(read_only);
    EXPECT_EQ(read_only.get(), read_write.get());
    EXPECT_EQ(read_only->get_device_access(), experimental::PinnedMemoryDeviceAccess::ReadWrite);
}

TEST_F(PinnedMemoryTestSuite, PinnedMemoryCacheCreatesBestSupportedReadOnlyMapping) {
    const auto pinning_params = experimental::GetMemoryPinningParameters(*mesh_device_);
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes() == 0 ||
        !pinning_params.can_map_to_noc) {
        GTEST_SKIP() << "Pinned NOC mappings are not available";
    }

    auto storage = std::make_shared<vector_aligned<uint32_t>>(1024, 0);
    HostBuffer host_buffer(ttsl::Span<uint32_t>(storage->data(), storage->size()), MemoryPin(storage));
    const auto coord = *MeshCoordinateRange(mesh_device_->shape()).begin();
    const auto range = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));

    auto pinned = experimental::PinnedMemoryCache::instance().try_pin(
        *mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadOnly);
    ASSERT_TRUE(pinned);
    const auto expected_access = pinning_params.supports_read_only ? experimental::PinnedMemoryDeviceAccess::ReadOnly
                                                                   : experimental::PinnedMemoryDeviceAccess::ReadWrite;
    EXPECT_EQ(pinned->get_device_access(), expected_access);
}

// The mirror image of the read-only tests above: they need the feature present, this one needs it absent. That
// makes it the only one of the group that runs on the current fleet, which is still below the KMD 2.9.0 floor,
// and the only coverage of PinnedMemory::Create's public contract. PinnedMemoryCache::try_pin cannot cover it:
// try_pin widens an unsupported read-only request to ReadWrite, while Create -- reachable directly by callers --
// must reject it up front rather than let a raw UMD error escape a tt-metal API.
TEST_F(PinnedMemoryTestSuite, PinnedMemoryCreateRejectsReadOnlyWhenUnsupported) {
    const auto pinning_params = experimental::GetMemoryPinningParameters(*mesh_device_);
    if (pinning_params.supports_read_only) {
        GTEST_SKIP() << "Requires a system without device-read-only pinning support";
    }

    auto storage = std::make_shared<vector_aligned<uint32_t>>(1024, 0);
    HostBuffer host_buffer(ttsl::Span<uint32_t>(storage->data(), storage->size()), MemoryPin(storage));
    const auto coord = *MeshCoordinateRange(mesh_device_->shape()).begin();
    const auto range = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));

    try {
        experimental::PinnedMemory::Create(
            *mesh_device_, range, host_buffer, /*map_to_noc=*/true, experimental::PinnedMemoryDeviceAccess::ReadOnly);
        FAIL() << "Create() must reject a ReadOnly request on a system without support for it";
    } catch (const std::exception& e) {
        // Checking the text, not just that something threw: the contract is that the caller is told which
        // parameter to consult, so a bare throw from somewhere else in Create() would not satisfy it.
        EXPECT_NE(std::string(e.what()).find("supports_read_only"), std::string::npos) << e.what();
    }
}

TEST_F(PinnedMemoryTestSuite, PinnedMemoryCacheReadWriteRequestReplacesUnreferencedReadOnlyMapping) {
    const auto pinning_params = experimental::GetMemoryPinningParameters(*mesh_device_);
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes() == 0 ||
        !pinning_params.can_map_to_noc || !pinning_params.supports_read_only) {
        GTEST_SKIP() << "Device-read-only pinned NOC mappings are not available";
    }

    auto storage = std::make_shared<vector_aligned<uint32_t>>(1024, 0);
    HostBuffer host_buffer(ttsl::Span<uint32_t>(storage->data(), storage->size()), MemoryPin(storage));
    const auto coord = *MeshCoordinateRange(mesh_device_->shape()).begin();
    const auto range = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    auto& cache = experimental::PinnedMemoryCache::instance();

    auto read_only =
        cache.try_pin(*mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadOnly);
    ASSERT_TRUE(read_only);
    std::weak_ptr<experimental::PinnedMemory> old_mapping = read_only;
    read_only.reset();

    auto read_write =
        cache.try_pin(*mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadWrite);
    ASSERT_TRUE(read_write);
    EXPECT_TRUE(old_mapping.expired());
    EXPECT_EQ(read_write->get_device_access(), experimental::PinnedMemoryDeviceAccess::ReadWrite);
}

TEST_F(PinnedMemoryTestSuite, PinnedMemoryCacheReadWriteRequestRejectsHeldReadOnlyMapping) {
    const auto pinning_params = experimental::GetMemoryPinningParameters(*mesh_device_);
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_pinned_memory_cache_limit_bytes() == 0 ||
        !pinning_params.can_map_to_noc || !pinning_params.supports_read_only) {
        GTEST_SKIP() << "Device-read-only pinned NOC mappings are not available";
    }

    auto storage = std::make_shared<vector_aligned<uint32_t>>(1024, 0);
    HostBuffer host_buffer(ttsl::Span<uint32_t>(storage->data(), storage->size()), MemoryPin(storage));
    const auto coord = *MeshCoordinateRange(mesh_device_->shape()).begin();
    const auto range = MeshCoordinateRangeSet(MeshCoordinateRange(coord, coord));
    auto& cache = experimental::PinnedMemoryCache::instance();

    auto read_only =
        cache.try_pin(*mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadOnly);
    ASSERT_TRUE(read_only);
    auto read_write =
        cache.try_pin(*mesh_device_, range, host_buffer, true, experimental::PinnedMemoryDeviceAccess::ReadWrite);
    EXPECT_EQ(read_write, nullptr);
}

// Host-only tests of the check behind try_pin's read-only decline (#63423); they need no device or KMD support.
using experimental::find_private_file_mapping;
using experimental::MapsSource;
using experimental::PrivateFileMapping;

// Whether PROCMAP_QUERY (Linux 6.11+) answers here, asked without the code under test; a sandbox may deny it.
// The request is the uapi's _IOWR('f', 17, struct procmap_query): size, flags (COVERING_OR_NEXT_VMA), address, ...
bool procmap_query_works() {
    std::array<uint64_t, 13> query{sizeof(query), 0x10, reinterpret_cast<uintptr_t>(&query)};
    const int maps = open("/proc/thread-self/maps", O_RDONLY | O_CLOEXEC);
    const bool works = maps >= 0 && ioctl(maps, _IOWR('f', 17, decltype(query)), query.data()) == 0;
    if (maps >= 0) {
        close(maps);
    }
    return works;
}

// read_only_pin_allowed's warnings, recorded through its `warn` parameter. A test cannot capture them from tt-logger:
// libtt_metal.so keeps its own logger registry, so its warnings follow TT_LOGGER_LEVEL, _TYPES and _FILE.
struct RecordedWarnings {
    std::vector<std::string> seen;
    std::function<void(std::string_view)> sink() {
        return [this](std::string_view warning) { seen.emplace_back(warning); };
    }
};
constexpr std::string_view kUnknownWarning = "could not rule out a private file mapping";

// Adjacent 2 MiB VMAs from 0x7f0000000000: anonymous, shared file, private file, anonymous, deleted private file.
constexpr const char* kMapsText =
    "\n"
    "00400000-00452000 r-xp 00000000 08:02 173521                     /usr/bin/python3\n"
    "7f0000000000-7f0000200000 rw-p 00000000 00:00 0 \n"
    "7f0000200000-7f0000400000 r--s 00000000 08:02 99                 /data/shared.bin\n"
    "7f0000400000-7f0000600000 r--p 00000000 08:02 100                /data/w.safetensors\n"
    "7f0000600000-7f0000800000 rw-p 00000000 00:00 0 \n"
    "7f0000800000-7f0000a00000 r--p 1a2b3c4d5 08:02 101               /data/old weights.bin (deleted)\n"
    "7ffd00000000-7ffd00021000 rw-p 00000000 00:00 0                  [stack]\n";
constexpr uintptr_t kMapsBase = 0x7f0000000000;
constexpr uintptr_t kVma = 0x200000;

PrivateFileMapping in_text(const std::string& text, uintptr_t begin, uintptr_t end) {
    std::istringstream maps(text);
    return experimental::find_private_file_mapping_in_maps_text(maps, begin, end);
}

TEST(PinnedMemoryCachePrivateFileMappingTest, MapsTextClassifiesEachVma) {
    constexpr auto kNo = PrivateFileMapping::NotFound;
    constexpr auto kYes = PrivateFileMapping::Found;
    EXPECT_EQ(in_text(kMapsText, kMapsBase, kMapsBase + kVma), kNo);                                    // anonymous
    EXPECT_EQ(in_text(kMapsText, kMapsBase + kVma, kMapsBase + 2 * kVma), kNo);                         // shared file
    EXPECT_EQ(in_text(kMapsText, kMapsBase + 2 * kVma, kMapsBase + 3 * kVma), kYes);                    // private file
    EXPECT_EQ(in_text(kMapsText, kMapsBase + 3 * kVma, kMapsBase + 4 * kVma), kNo);                     // anonymous
    EXPECT_EQ(in_text(kMapsText, kMapsBase + 4 * kVma + 0x1000, kMapsBase + 4 * kVma + 0x2000), kYes);  // 9 digits
    EXPECT_EQ(in_text(kMapsText, kMapsBase, kMapsBase + 2 * kVma), kNo);  // anonymous and shared
    EXPECT_EQ(in_text(kMapsText, kMapsBase + kVma - 0x1000, kMapsBase + 2 * kVma + 0x1000), kYes);  // into private
    EXPECT_EQ(in_text(kMapsText, 0x400000, 0x401000), kYes);  // the executable's text
}

// Maps that do not cover the whole range cannot rule out a private file mapping, so they answer Unknown.
TEST(PinnedMemoryCachePrivateFileMappingTest, MapsTextThatMissesTheRangeIsUnknown) {
    constexpr auto kUnknown = PrivateFileMapping::Unknown;
    EXPECT_EQ(in_text(kMapsText, 0x7ffe00000000, 0x7ffe00001000), kUnknown);  // above every VMA
    EXPECT_EQ(in_text("", kMapsBase, kMapsBase + kVma), kUnknown);            // no maps at all
    EXPECT_EQ(in_text(kMapsText, 0x452000, 0x453000), kUnknown);              // in a gap between VMAs
    const std::string garbled = "7f0000000000-7f0000200000 rw-p\n7f0000200000-7f0000400000 r--s 0 08:02 99\n";
    EXPECT_EQ(in_text(garbled, kMapsBase, kMapsBase + 2 * kVma), kUnknown);  // an overlapping line that fails to parse
}

// Real adjacent VMAs, checked on whichever path this process offers: PROCMAP_QUERY if it answers, else the text.
TEST(PinnedMemoryCachePrivateFileMappingTest, FindsPrivateFileMappingAcrossVmas) {
    const auto expected_source = procmap_query_works() ? MapsSource::Ioctl : MapsSource::Text;
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const int fd = memfd_create("tt_metal_private_mapping_check", MFD_CLOEXEC);
    ASSERT_NE(fd, -1);
    ASSERT_EQ(ftruncate(fd, static_cast<off_t>(2 * page)), 0);
    // anonymous | shared memfd | private memfd | anonymous
    auto* region =
        static_cast<char*>(mmap(nullptr, 4 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    ASSERT_NE(region, MAP_FAILED);
    std::shared_ptr<void> unmap(region, [page](void* mapping) { munmap(mapping, 4 * page); });
    ASSERT_EQ(mmap(region + page, page, PROT_READ, MAP_SHARED | MAP_FIXED, fd, 0), region + page);
    ASSERT_EQ(mmap(region + 2 * page, page, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, page), region + 2 * page);
    ASSERT_EQ(close(fd), 0);

    // A page the test maps and unmaps again is a hole that no VMA covers, whatever the address layout. The cases live
    // on the stack and the hole is queried first, so no allocation can land in it before its query.
    void* hole = mmap(nullptr, page, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(hole, MAP_FAILED);
    const std::array<std::tuple<const void*, size_t, PrivateFileMapping>, 5> cases = {{
        {hole, page, PrivateFileMapping::Unknown},
        {region, 2 * page, PrivateFileMapping::NotFound},
        {region, 4 * page, PrivateFileMapping::Found},
        {region + 2 * page + 100, 1, PrivateFileMapping::Found},
        {region + 3 * page, page, PrivateFileMapping::NotFound},
    }};
    ASSERT_EQ(munmap(hole, page), 0);
    for (const auto& [address, size, expected] : cases) {
        auto source = MapsSource::None;
        EXPECT_EQ(find_private_file_mapping(address, size, "/proc/thread-self/maps", &source), expected);
        EXPECT_EQ(source, expected_source);
    }
}

// A regular file answers PROCMAP_QUERY with ENOTTY, so this takes the text path that kernels before 6.11 use.
TEST(PinnedMemoryCachePrivateFileMappingTest, FallsBackToMapsText) {
    const auto path = std::filesystem::temp_directory_path() / ("tt_metal_maps_" + std::to_string(getpid()));
    std::ofstream(path) << kMapsText;
    const auto* base = reinterpret_cast<const void*>(kMapsBase);
    auto source = MapsSource::None;
    EXPECT_EQ(find_private_file_mapping(base, 2 * kVma, path.c_str(), &source), PrivateFileMapping::NotFound);
    EXPECT_EQ(source, MapsSource::Text);
    EXPECT_EQ(find_private_file_mapping(base, 3 * kVma, path.c_str(), &source), PrivateFileMapping::Found);
    EXPECT_EQ(source, MapsSource::Text);
    std::filesystem::remove(path);
}

// Maps that cannot be opened, read or that come back empty answer Unknown, which try_pin declines.
TEST(PinnedMemoryCachePrivateFileMappingTest, UnreadableOrEmptyMapsIsUnknown) {
    const int value = 0;
    EXPECT_EQ(find_private_file_mapping(&value, sizeof(value), "/nonexistent/maps"), PrivateFileMapping::Unknown);
    EXPECT_EQ(find_private_file_mapping(&value, sizeof(value), "/"), PrivateFileMapping::Unknown);  // read() fails
    EXPECT_EQ(find_private_file_mapping(&value, sizeof(value), "/dev/null"), PrivateFileMapping::Unknown);  // empty
    const auto* near_top = reinterpret_cast<const void*>(UINTPTR_MAX - 10);
    EXPECT_EQ(find_private_file_mapping(near_top, 100), PrivateFileMapping::Unknown);  // the range wraps around
}

// try_pin's decision for each answer: a private file mapping and Unknown are declined, and an anonymous answer goes on
// to pin. Only Unknown warns, and only the first time. Removing the decline or the warning fails this test.
TEST(PinnedMemoryCachePrivateFileMappingTest, ReadOnlyPinDecision) {
    const int value = 0;
    RecordedWarnings warnings;
    std::once_flag unknown_warning;
    const auto allowed = [&](PrivateFileMapping mapping) {
        return experimental::read_only_pin_allowed(&value, 4, mapping, unknown_warning, warnings.sink());
    };
    EXPECT_FALSE(allowed(PrivateFileMapping::Found));
    EXPECT_TRUE(allowed(PrivateFileMapping::NotFound));
    EXPECT_TRUE(warnings.seen.empty());  // neither warns, though the flag is still unused
    EXPECT_FALSE(allowed(PrivateFileMapping::Unknown));
    EXPECT_FALSE(allowed(PrivateFileMapping::Unknown));
    ASSERT_EQ(warnings.seen.size(), 1u);
    EXPECT_NE(warnings.seen[0].find(kUnknownWarning), std::string::npos) << warnings.seen[0];
}

// The overload try_pin calls shares one flag across the process, so Unknown answers warn at most once in all (none
// here if one in an earlier test of this process already did).
TEST(PinnedMemoryCachePrivateFileMappingTest, ReadOnlyPinDecisionWarnsOncePerProcess) {
    const int value = 0;
    RecordedWarnings warnings;
    EXPECT_FALSE(experimental::read_only_pin_allowed(&value, 4, PrivateFileMapping::Unknown, warnings.sink()));
    EXPECT_FALSE(experimental::read_only_pin_allowed(&value, 4, PrivateFileMapping::Unknown, warnings.sink()));
    EXPECT_LE(warnings.seen.size(), 1u);
}

// The same decision on real mappings, as try_pin makes it: a private memfd view is declined, anonymous memory is not.
TEST(PinnedMemoryCachePrivateFileMappingTest, ReadOnlyPinDecisionOnRealMappings) {
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const int fd = memfd_create("tt_metal_pin_decision_check", MFD_CLOEXEC);
    ASSERT_NE(fd, -1);
    ASSERT_EQ(ftruncate(fd, static_cast<off_t>(page)), 0);
    void* private_view = mmap(nullptr, page, PROT_READ, MAP_PRIVATE, fd, 0);
    ASSERT_NE(private_view, MAP_FAILED);
    ASSERT_EQ(close(fd), 0);
    void* anonymous = mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(anonymous, MAP_FAILED);
    std::once_flag unknown_warning;
    auto allowed = [&](void* address) {
        return experimental::read_only_pin_allowed(
            address, page, find_private_file_mapping(address, page), unknown_warning);
    };
    EXPECT_FALSE(allowed(private_view));
    EXPECT_TRUE(allowed(anonymous));
    EXPECT_EQ(munmap(private_view, page), 0);
    EXPECT_EQ(munmap(anonymous, page), 0);
}

// After the main thread exits, /proc/self/maps reads empty; the calling thread's maps still list its mappings.
TEST(PinnedMemoryCachePrivateFileMappingTest, ChecksCallingThreadAfterMainThreadExits) {
    const std::string death_test_style = GTEST_FLAG_GET(death_test_style);
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            std::thread([] {
                const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
                const int fd = memfd_create("tt_metal_thread_exit_check", MFD_CLOEXEC);
                const bool sized = fd != -1 && ftruncate(fd, static_cast<off_t>(page)) == 0;
                void* private_view = sized ? mmap(nullptr, page, PROT_READ, MAP_PRIVATE, fd, 0) : MAP_FAILED;
                void* anonymous = mmap(nullptr, page, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                usleep(100000);  // let the main thread finish exiting
                const bool ok = private_view != MAP_FAILED && anonymous != MAP_FAILED &&
                                find_private_file_mapping(private_view, page) == PrivateFileMapping::Found &&
                                find_private_file_mapping(anonymous, page) == PrivateFileMapping::NotFound;
                std::_Exit(ok ? 0 : 1);
            }).detach();
            syscall(SYS_exit, 0);  // ends only this thread, as pthread_exit does, without an unwind gtest would catch
        },
        ::testing::ExitedWithCode(0),
        "");
    GTEST_FLAG_SET(death_test_style, death_test_style);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
