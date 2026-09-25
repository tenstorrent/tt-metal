// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tests for large tensor uploads that the device reads from host memory pinned in chunks by a worker pool
// (tt_metal/impl/tensor/pinned_upload.cpp). The chunked path runs on Blackhole with IOMMU and read-only pinning;
// the fixture skips elsewhere.

#include <gtest/gtest.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>
#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/memory_pin_access.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt_stl/aligned_allocator.hpp>
#include <tt_stl/span.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

#include "impl/context/metal_context.hpp"
#include "impl/tensor/pinned_upload.hpp"
#include "tt_metal/distributed/pinned_memory_cache.hpp"

namespace tt::tt_metal {
namespace {

// A row of 1024 uint32 is one 4 KiB page, so chunks are exactly 8 MiB (2048 rows) and every row offset is aligned.
constexpr uint32_t k_row_words = 1024;
constexpr size_t k_row_bytes = k_row_words * sizeof(uint32_t);
constexpr size_t k_rows_per_chunk = (8 * 1024 * 1024) / k_row_bytes;
constexpr size_t k_host_alignment = 4096;

using AlignedWords = std::vector<uint32_t, ttsl::aligned_allocator<uint32_t, k_host_alignment>>;

TensorSpec row_major_spec(size_t rows, uint32_t row_words, const MemoryConfig& memory_config = MemoryConfig{}) {
    return TensorSpec(
        Shape{1, 1, static_cast<uint32_t>(rows), row_words},
        TensorLayout(DataType::UINT32, Layout::ROW_MAJOR, memory_config));
}

HostTensor host_tensor_over(HostBuffer buffer, const TensorSpec& spec, const distributed::MeshShape& mesh_shape) {
    auto dhb = DistributedHostBuffer::create(mesh_shape);
    distributed::MeshCoordinateRange range(mesh_shape);
    std::vector<distributed::MeshCoordinate> coords(range.begin(), range.end());
    dhb.emplace_shards(coords, [&](const distributed::MeshCoordinate&) { return buffer; });
    return host_tensor_from_buffer_with_topology(
        std::move(dhb), spec, TensorTopology::create_fully_replicated_tensor_topology(mesh_shape));
}

std::vector<uint32_t> read_back(distributed::MeshCommandQueue& cq, const MeshTensor& device_tensor) {
    HostTensor result = cq.enqueue_read_tensor(device_tensor);
    auto shard = result.buffer().get_shard(*distributed::MeshCoordinateRange(device_tensor.device().shape()).begin());
    // Compare as words whatever the dtype: block-float host buffers hold packed uint32 words.
    auto bytes = shard->view_bytes();
    std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
    std::memcpy(words.data(), bytes.data(), words.size() * sizeof(uint32_t));
    return words;
}

// First mismatching word index, or -1.
int64_t first_mismatch(ttsl::Span<const uint32_t> expected, const std::vector<uint32_t>& actual) {
    if (expected.size() != actual.size()) {
        return 0;
    }
    for (size_t i = 0; i < expected.size(); i++) {
        if (expected[i] != actual[i]) {
            return static_cast<int64_t>(i);
        }
    }
    return -1;
}

class ScopedPinnedUploadThreads {
public:
    explicit ScopedPinnedUploadThreads(uint32_t num_threads) :
        previous_(MetalContext::instance().rtoptions().get_pinned_upload_threads()) {
        MetalContext::instance().rtoptions().set_pinned_upload_threads(num_threads);
    }
    ~ScopedPinnedUploadThreads() { MetalContext::instance().rtoptions().set_pinned_upload_threads(previous_); }

    ScopedPinnedUploadThreads(const ScopedPinnedUploadThreads&) = delete;
    ScopedPinnedUploadThreads& operator=(const ScopedPinnedUploadThreads&) = delete;

private:
    uint32_t previous_ = 0;
};

class TensixPinnedUploadFixture : public MeshDevice1x1Fixture {
protected:
    void SetUp() override {
        MeshDevice1x1Fixture::SetUp();
        if (IsSkipped()) {
            return;
        }
        const auto params = experimental::GetMemoryPinningParameters(*mesh_device_);
        const auto& rtoptions = MetalContext::instance().rtoptions();
        if (!MetalContext::instance().hal().get_supports_64_bit_pcie_addressing() || params.max_pins == 0 ||
            !params.can_map_to_noc || !params.supports_read_only) {
            GTEST_SKIP() << "Chunked pinned uploads need Blackhole with IOMMU and read-only page pinning";
        }
        if (rtoptions.get_pinned_upload_threads() == 0 || rtoptions.get_pinned_memory_cache_limit_bytes() == 0) {
            GTEST_SKIP() << "Chunked pinned uploads are disabled by TT_METAL_PINNED_UPLOAD_THREADS or "
                            "TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES";
        }
    }

    // Uploads `rows` rows of distinct words from a fresh mutable buffer and checks the device copy. Returns whether a
    // PinnedMemoryCache entry was added, which only the whole-shard path does.
    bool upload_and_verify(size_t rows) {
        auto words = std::make_shared<AlignedWords>(rows * k_row_words);
        std::iota(words->begin(), words->end(), static_cast<uint32_t>(rows));
        HostBuffer buffer(ttsl::Span<uint32_t>(words->data(), words->size()), MemoryPin(words));
        auto spec = row_major_spec(rows, k_row_words);
        auto host_tensor = host_tensor_over(buffer, spec, mesh_device_->shape());

        auto& cache = experimental::PinnedMemoryCache::instance();
        const size_t entries_before = cache.num_entries();
        auto& cq = mesh_device_->mesh_command_queue();
        MeshTensor device_tensor = cq.enqueue_write_tensor(host_tensor);
        const bool cached = cache.num_entries() != entries_before;

        const auto actual = read_back(cq, device_tensor);
        EXPECT_EQ(first_mismatch(ttsl::Span<const uint32_t>(words->data(), words->size()), actual), -1)
            << "rows=" << rows;
        return cached;
    }
};

TEST_F(TensixPinnedUploadFixture, UploadWithPartialLastChunk) {
    // 33 MiB plus 37 rows: above the 32 MiB pinned-write threshold, with a short final chunk.
    EXPECT_FALSE(upload_and_verify(33 * 256 + 37));
}

TEST_F(TensixPinnedUploadFixture, UploadOfWholeChunks) { EXPECT_FALSE(upload_and_verify(5 * k_rows_per_chunk)); }

TEST_F(TensixPinnedUploadFixture, UploadOf1GiB) { EXPECT_FALSE(upload_and_verify(1024 * 1024 * 1024 / k_row_bytes)); }

// Memory that is not device-immutable may be reused as soon as the upload returns.
TEST_F(TensixPinnedUploadFixture, MutableHostMemoryIsReadBeforeReturn) {
    const size_t rows = 9 * k_rows_per_chunk + 3;
    auto words = std::make_shared<AlignedWords>(rows * k_row_words);
    std::iota(words->begin(), words->end(), 7u);
    const AlignedWords expected = *words;
    HostBuffer buffer(ttsl::Span<uint32_t>(words->data(), words->size()), MemoryPin(words));
    auto host_tensor = host_tensor_over(buffer, row_major_spec(rows, k_row_words), mesh_device_->shape());

    auto& cq = mesh_device_->mesh_command_queue();
    MeshTensor device_tensor = cq.enqueue_write_tensor(host_tensor);
    std::fill(words->begin(), words->end(), 0xDEADBEEFu);
    EXPECT_EQ(pinned_upload::num_pending(*mesh_device_), 0u);

    const auto actual = read_back(cq, device_tensor);
    EXPECT_EQ(first_mismatch(ttsl::Span<const uint32_t>(expected.data(), expected.size()), actual), -1);
}

// Device-immutable memory (a read-only file mapping) may still be read after the upload returns. Closing the device
// without finish() must wait for those reads and release every mapping.
TEST_F(TensixPinnedUploadFixture, ImmutableUploadsCompleteAtDeviceClose) {
    constexpr size_t rows = 6 * k_rows_per_chunk + 11;
    constexpr size_t size_bytes = rows * k_row_bytes;
    AlignedWords expected(rows * k_row_words);
    std::iota(expected.begin(), expected.end(), 3u);

    const auto path =
        std::filesystem::temp_directory_path() / ("tt_pinned_upload_test_" + std::to_string(getpid()) + ".bin");
    {
        FILE* file = std::fopen(path.c_str(), "wb");
        ASSERT_NE(file, nullptr) << std::strerror(errno);
        ASSERT_EQ(std::fwrite(expected.data(), 1, size_bytes, file), size_bytes);
        ASSERT_EQ(std::fclose(file), 0);
    }
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    std::filesystem::remove(path);
    ASSERT_GE(fd, 0) << std::strerror(errno);

    // Each upload gets its own mapping, as ttnn.load_tensor does, so every one takes the chunked path.
    auto num_released = std::make_shared<std::atomic<int>>(0);
    auto map_file = [&]() -> HostBuffer {
        void* mapping = mmap(nullptr, size_bytes, PROT_READ, MAP_SHARED, fd, 0);
        TT_FATAL(mapping != MAP_FAILED, "mmap failed: {}", std::strerror(errno));
        MemoryPin pin(std::shared_ptr<void>(mapping, [num_released](void* addr) {
            munmap(addr, size_bytes);
            num_released->fetch_add(1);
        }));
        experimental::MemoryPinMarkDeviceImmutable(pin);
        return HostBuffer(ttsl::Span<uint32_t>(static_cast<uint32_t*>(mapping), rows * k_row_words), pin);
    };

    constexpr int k_num_uploads = 6;
    auto spec = row_major_spec(rows, k_row_words);
    auto& cq = mesh_device_->mesh_command_queue();
    {
        // Read one back, so the mapping is known to be what the device receives.
        auto host_tensor = host_tensor_over(map_file(), spec, mesh_device_->shape());
        MeshTensor device_tensor = cq.enqueue_write_tensor(host_tensor);
        const auto actual = read_back(cq, device_tensor);
        EXPECT_EQ(first_mismatch(ttsl::Span<const uint32_t>(expected.data(), expected.size()), actual), -1);
    }
    {
        std::vector<MeshTensor> device_tensors;
        for (int i = 1; i < k_num_uploads; i++) {
            device_tensors.push_back(
                cq.enqueue_write_tensor(host_tensor_over(map_file(), spec, mesh_device_->shape())));
        }
    }
    close(fd);
    mesh_device_->close();
    EXPECT_EQ(pinned_upload::num_pending(*mesh_device_), 0u);
    pinned_upload::wait_for_storage_release();
    EXPECT_EQ(num_released->load(), k_num_uploads);
    mesh_device_.reset();
}

TEST_F(TensixPinnedUploadFixture, FinishReleasesImmutableUploads) {
    const size_t rows = 5 * k_rows_per_chunk + 1;
    auto words = std::make_shared<AlignedWords>(rows * k_row_words);
    std::iota(words->begin(), words->end(), 11u);
    MemoryPin pin(words);
    experimental::MemoryPinMarkDeviceImmutable(pin);
    HostBuffer buffer(ttsl::Span<uint32_t>(words->data(), words->size()), pin);
    auto host_tensor = host_tensor_over(buffer, row_major_spec(rows, k_row_words), mesh_device_->shape());

    auto& cq = mesh_device_->mesh_command_queue();
    MeshTensor device_tensor = cq.enqueue_write_tensor(host_tensor);
    cq.finish();
    EXPECT_EQ(pinned_upload::num_pending(*mesh_device_), 0u);
    const auto actual = read_back(cq, device_tensor);
    EXPECT_EQ(first_mismatch(ttsl::Span<const uint32_t>(words->data(), words->size()), actual), -1);
}

// Sharded device buffers are not eligible for chunking; they take the whole-shard path and still arrive intact.
TEST_F(TensixPinnedUploadFixture, ShardedBufferFallsBack) {
    auto dram_grid = mesh_device_->dram_grid_size();
    CoreRangeSet dram_cores(CoreRange(CoreCoord{0, 0}, CoreCoord{dram_grid.x - 1, 0}));
    const size_t rows = 40 * 256;
    MemoryConfig sharded(
        BufferType::DRAM, NdShardSpec{Shape{1, 1, 32, k_row_words}, dram_cores, ShardOrientation::ROW_MAJOR});

    auto words = std::make_shared<AlignedWords>(rows * k_row_words);
    std::iota(words->begin(), words->end(), 5u);
    HostBuffer buffer(ttsl::Span<uint32_t>(words->data(), words->size()), MemoryPin(words));
    auto host_tensor = host_tensor_over(buffer, row_major_spec(rows, k_row_words, sharded), mesh_device_->shape());

    auto& cq = mesh_device_->mesh_command_queue();
    MeshTensor device_tensor = cq.enqueue_write_tensor(host_tensor);
    const auto actual = read_back(cq, device_tensor);
    EXPECT_EQ(first_mismatch(ttsl::Span<const uint32_t>(words->data(), words->size()), actual), -1);
}

// Rows of 1000 words (4000 B) are padded to the DRAM alignment on device, so the pinned path does not apply.
TEST_F(TensixPinnedUploadFixture, PaddedPagesFallBack) {
    constexpr uint32_t row_words = 1000;
    const size_t rows = 9000;
    auto spec = row_major_spec(rows, row_words);
    ASSERT_GT(rows * row_words * sizeof(uint32_t), pinned_upload::k_pin_write_threshold_bytes);

    auto words = std::make_shared<AlignedWords>(rows * row_words);
    std::iota(words->begin(), words->end(), 9u);
    HostBuffer buffer(ttsl::Span<uint32_t>(words->data(), words->size()), MemoryPin(words));
    auto host_tensor = host_tensor_over(buffer, spec, mesh_device_->shape());

    auto& cq = mesh_device_->mesh_command_queue();
    MeshTensor device_tensor = cq.enqueue_write_tensor(host_tensor);
    const auto actual = read_back(cq, device_tensor);
    EXPECT_EQ(first_mismatch(ttsl::Span<const uint32_t>(words->data(), words->size()), actual), -1);
}

// A host base off the L1 read alignment cannot be read directly by the device; the upload copies instead.
TEST_F(TensixPinnedUploadFixture, UnalignedHostBaseFallsBack) {
    const size_t rows = 36 * 256;
    auto words = std::make_shared<AlignedWords>(rows * k_row_words + 1);
    std::iota(words->begin(), words->end(), 13u);
    // One word past an aligned address.
    HostBuffer buffer(ttsl::Span<uint32_t>(words->data() + 1, rows * k_row_words), MemoryPin(words));
    auto host_tensor = host_tensor_over(buffer, row_major_spec(rows, k_row_words), mesh_device_->shape());

    auto& cache = experimental::PinnedMemoryCache::instance();
    const size_t entries_before = cache.num_entries();
    auto& cq = mesh_device_->mesh_command_queue();
    MeshTensor device_tensor = cq.enqueue_write_tensor(host_tensor);
    EXPECT_NE(cache.num_entries(), entries_before) << "An unaligned base must take the whole-shard path";
    const auto actual = read_back(cq, device_tensor);
    EXPECT_EQ(first_mismatch(ttsl::Span<const uint32_t>(words->data() + 1, rows * k_row_words), actual), -1);
}

TEST_F(TensixPinnedUploadFixture, ZeroThreadsUsesWholeShardPin) {
    ScopedPinnedUploadThreads threads(0);
    EXPECT_TRUE(upload_and_verify(35 * 256 + 5));
}

TEST_F(TensixPinnedUploadFixture, SecondUploadOfSameBufferUsesPinCache) {
    const size_t rows = 34 * 256 + 9;
    auto words = std::make_shared<AlignedWords>(rows * k_row_words);
    std::iota(words->begin(), words->end(), 17u);
    HostBuffer buffer(ttsl::Span<uint32_t>(words->data(), words->size()), MemoryPin(words));
    auto host_tensor = host_tensor_over(buffer, row_major_spec(rows, k_row_words), mesh_device_->shape());

    auto& cache = experimental::PinnedMemoryCache::instance();
    auto& cq = mesh_device_->mesh_command_queue();
    const size_t entries_before = cache.num_entries();
    MeshTensor first = cq.enqueue_write_tensor(host_tensor);
    EXPECT_EQ(cache.num_entries(), entries_before);
    MeshTensor second = cq.enqueue_write_tensor(host_tensor);
    EXPECT_EQ(cache.num_entries(), entries_before + 1);
    const auto actual = read_back(cq, second);
    EXPECT_EQ(first_mismatch(ttsl::Span<const uint32_t>(words->data(), words->size()), actual), -1);
}

// Block-float tiles have page sizes that are not powers of two (bfloat4_b: 576 B, bfloat8_b: 1088 B), so chunks are
// multiples of lcm(page, 4 KiB) and the region offsets of every chunk must still land on page boundaries.
void upload_block_float_and_verify(distributed::MeshDevice& mesh_device, DataType dtype, uint32_t rows, uint32_t cols) {
    const TensorSpec spec(Shape{rows, cols}, TensorLayout(dtype, PageConfig(Layout::TILE), MemoryConfig{}));
    const size_t size_bytes = spec.compute_packed_buffer_size_bytes();
    ASSERT_GT(size_bytes, pinned_upload::k_pin_write_threshold_bytes);
    ASSERT_NE(size_bytes % pinned_upload::k_max_chunk_bytes, 0u);

    auto words = std::make_shared<AlignedWords>(size_bytes / sizeof(uint32_t));
    uint32_t state = 0x12345678u ^ rows ^ (cols << 8);
    for (auto& word : *words) {
        state = state * 1664525u + 1013904223u;
        word = state;
    }
    HostBuffer buffer(ttsl::Span<uint32_t>(words->data(), words->size()), MemoryPin(words));
    auto host_tensor = host_tensor_over(buffer, spec, mesh_device.shape());

    auto& cache = experimental::PinnedMemoryCache::instance();
    const size_t entries_before = cache.num_entries();
    auto& cq = mesh_device.mesh_command_queue();
    MeshTensor device_tensor = cq.enqueue_write_tensor(host_tensor);
    EXPECT_EQ(cache.num_entries(), entries_before) << "Expected the chunked path, not a whole-shard cached pin";
    const auto actual = read_back(cq, device_tensor);
    EXPECT_EQ(first_mismatch(ttsl::Span<const uint32_t>(words->data(), words->size()), actual), -1);
}

// 7168 x 10240 bfloat4_b: five 8.26 MB expert-weight-sized chunks (576 B pages).
TEST_F(TensixPinnedUploadFixture, UploadBfloat4bShard) {
    upload_block_float_and_verify(*mesh_device_, DataType::BFLOAT4_B, 7168, 10240);
}

// A bfloat4_b shard whose chunks round up to the 36 KiB granule, leaving a shorter last chunk.
TEST_F(TensixPinnedUploadFixture, UploadBfloat4bShardWithShortLastChunk) {
    upload_block_float_and_verify(*mesh_device_, DataType::BFLOAT4_B, 7168, 9728);
}

// 7168 x 5120 bfloat8_b (1088 B pages).
TEST_F(TensixPinnedUploadFixture, UploadBfloat8bShard) {
    upload_block_float_and_verify(*mesh_device_, DataType::BFLOAT8_B, 7168, 5120);
}

// A bfloat8_b shard whose chunks round up to the 68 KiB granule, leaving a shorter last chunk.
TEST_F(TensixPinnedUploadFixture, UploadBfloat8bShardWithShortLastChunk) {
    upload_block_float_and_verify(*mesh_device_, DataType::BFLOAT8_B, 3232, 11264);
}

size_t num_chunks_for(size_t shard_bytes, size_t chunk_bytes) { return (shard_bytes + chunk_bytes - 1) / chunk_bytes; }

// A host base that meets the L1 read alignment but not the PCIe alignment makes every chunk start with an unaligned
// head, which the pinned write sends inline from host_data + region offset. 36 KiB rows keep chunk offsets page
// multiples without being 64 B aligned relative to the unaligned base.
TEST_F(TensixPinnedUploadFixture, HostBaseOffPcieAlignment) {
    constexpr uint32_t row_words = 9216;
    const size_t rows = 1025;
    constexpr size_t offset_words = 4;  // 16 B
    auto words = std::make_shared<AlignedWords>(rows * row_words + offset_words);
    std::iota(words->begin(), words->end(), 0u);
    HostBuffer buffer(ttsl::Span<uint32_t>(words->data() + offset_words, rows * row_words), MemoryPin(words));
    auto host_tensor = host_tensor_over(buffer, row_major_spec(rows, row_words), mesh_device_->shape());

    auto& cache = experimental::PinnedMemoryCache::instance();
    const size_t entries_before = cache.num_entries();
    auto& cq = mesh_device_->mesh_command_queue();
    MeshTensor device_tensor = cq.enqueue_write_tensor(host_tensor);
    EXPECT_EQ(cache.num_entries(), entries_before) << "Expected the chunked path, not a whole-shard cached pin";
    const auto actual = read_back(cq, device_tensor);
    EXPECT_EQ(first_mismatch(ttsl::Span<const uint32_t>(words->data() + offset_words, rows * row_words), actual), -1);
}

TEST(PinnedUploadChunking, ExpertShardIsOneChunk) {
    // 7168 x 2048 bfloat4_b: 14336 tiles of 576 B.
    constexpr size_t shard_bytes = 14336 * 576;
    EXPECT_EQ(pinned_upload::chunk_bytes_for(shard_bytes, 576), shard_bytes);
}

TEST(PinnedUploadChunking, ChunksAreBalancedGranuleMultiples) {
    const size_t os_page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    for (size_t page_bytes : {size_t{576}, size_t{1088}, size_t{2048}, size_t{4096}}) {
        const size_t granule = std::lcm(page_bytes, os_page);
        for (size_t num_pages : {size_t{17024}, size_t{35552}, size_t{65537}, size_t{262144}}) {
            const size_t shard_bytes = num_pages * page_bytes;
            const size_t chunk_bytes = pinned_upload::chunk_bytes_for(shard_bytes, page_bytes);
            ASSERT_NE(chunk_bytes, 0u);
            EXPECT_LE(chunk_bytes, pinned_upload::k_max_chunk_bytes);
            EXPECT_EQ(chunk_bytes % granule, 0u) << "page " << page_bytes;
            const size_t fewest_chunks = num_chunks_for(shard_bytes, pinned_upload::k_max_chunk_bytes);
            const size_t num_chunks = num_chunks_for(shard_bytes, chunk_bytes);
            // Rounding every chunk up to the granule can cost at most one extra chunk.
            EXPECT_LE(num_chunks, fewest_chunks + 1) << "page " << page_bytes << " pages " << num_pages;
            // Balanced: the last chunk is short by less than one granule per chunk.
            const size_t last_chunk = shard_bytes - (num_chunks - 1) * chunk_bytes;
            EXPECT_GT(last_chunk + num_chunks * granule, chunk_bytes)
                << "page " << page_bytes << " pages " << num_pages;
        }
    }
}

TEST(PinnedUploadChunking, GranuleLargerThanChunkLimitDisablesChunking) {
    // 32772 B pages share only a factor of 4 with a 4 KiB OS page, so their lcm exceeds 8 MiB.
    EXPECT_EQ(pinned_upload::chunk_bytes_for(size_t{32772} * 2048, 32772), 0u);
}

TEST(MemoryPinDeviceImmutability, CopiesShareTheMark) {
    auto storage = std::make_shared<int>(0);
    MemoryPin pin(storage);
    MemoryPin copy = pin;
    EXPECT_FALSE(experimental::MemoryPinIsDeviceImmutable(copy));
    experimental::MemoryPinMarkDeviceImmutable(pin);
    EXPECT_TRUE(experimental::MemoryPinIsDeviceImmutable(copy));
    EXPECT_TRUE(experimental::MemoryPinIsDeviceImmutable(MemoryPin(copy)));
    EXPECT_FALSE(experimental::MemoryPinIsDeviceImmutable(MemoryPin(storage)));
    EXPECT_FALSE(experimental::MemoryPinIsDeviceImmutable(MemoryPin()));
}

}  // namespace
}  // namespace tt::tt_metal
