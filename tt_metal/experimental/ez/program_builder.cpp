// SPDX-FileCopyrightText: (c) 2026 Olof Johansson <olof@lixom.net>
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/ez/program_builder.hpp>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::experimental::ez {

namespace {
// Convert CoreSpec to the variant that CreateSemaphore expects.
std::variant<CoreRange, CoreRangeSet> to_semaphore_core_spec(const CoreSpec& cs) {
    return std::visit(
        [](auto&& v) -> std::variant<CoreRange, CoreRangeSet> {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, CoreCoord>) {
                return CoreRange(v, v);
            } else if constexpr (std::is_same_v<T, CoreRange>) {
                return v;
            } else {
                return v;
            }
        },
        cs);
}
}  // namespace

// --- KernelRef ---

KernelRef::KernelRef(ProgramBuilder& builder, KernelHandle handle, CoreSpec core_spec) :
    builder_(builder), handle_(handle), core_spec_(std::move(core_spec)) {}

KernelRef& KernelRef::runtime_args(std::initializer_list<uint32_t> args) {
    SetRuntimeArgs(builder_.program_, handle_, core_spec_, args);
    return *this;
}

KernelRef& KernelRef::runtime_args(const std::vector<uint32_t>& args) {
    SetRuntimeArgs(builder_.program_, handle_, core_spec_, tt::stl::Span<const uint32_t>(args));
    return *this;
}

KernelRef& KernelRef::runtime_args(std::function<std::vector<uint32_t>(const CoreCoord&)> fn) {
    auto set_for_core = [&](const CoreCoord& core) {
        auto args = fn(core);
        SetRuntimeArgs(
            builder_.program_,
            handle_,
            std::variant<CoreCoord, CoreRange, CoreRangeSet>(core),
            tt::stl::Span<const uint32_t>(args));
    };

    std::visit(
        [&](auto&& cs) {
            using T = std::decay_t<decltype(cs)>;
            if constexpr (std::is_same_v<T, CoreCoord>) {
                set_for_core(cs);
            } else if constexpr (std::is_same_v<T, CoreRange>) {
                for (const auto& core : cs) {
                    set_for_core(core);
                }
            } else if constexpr (std::is_same_v<T, CoreRangeSet>) {
                for (const auto& range : cs.ranges()) {
                    for (const auto& core : range) {
                        set_for_core(core);
                    }
                }
            }
        },
        core_spec_);
    return *this;
}

KernelRef& KernelRef::runtime_args_at(const CoreCoord& core, const std::vector<uint32_t>& args) {
    SetRuntimeArgs(
        builder_.program_,
        handle_,
        std::variant<CoreCoord, CoreRange, CoreRangeSet>(core),
        tt::stl::Span<const uint32_t>(args));
    return *this;
}

ProgramBuilder& KernelRef::done() { return builder_; }

KernelHandle KernelRef::handle() const { return handle_; }

// --- ProgramBuilder ---

ProgramBuilder::ProgramBuilder(const CoreSpec& core_spec) :
    program_(CreateProgram()), default_core_spec_(core_spec) {}

CoreSpec ProgramBuilder::active_core_spec() {
    if (override_core_spec_.has_value()) {
        CoreSpec cs = std::move(*override_core_spec_);
        override_core_spec_.reset();
        return cs;
    }
    return default_core_spec_;
}

ProgramBuilder& ProgramBuilder::cb(tt::CBIndex index, uint32_t num_tiles, tt::DataFormat fmt) {
    auto ts = tt::tile_size(fmt);
    auto cs = active_core_spec();

    CreateCircularBuffer(
        program_,
        cs,
        CircularBufferConfig(num_tiles * ts, {{index, fmt}}).set_page_size(index, ts));
    return *this;
}

ProgramBuilder& ProgramBuilder::cb(
    tt::CBIndex index, tt::DataFormat fmt, uint32_t num_tiles, uint32_t page_size) {
    auto cs = active_core_spec();

    CreateCircularBuffer(
        program_,
        cs,
        CircularBufferConfig(num_tiles * page_size, {{index, fmt}}).set_page_size(index, page_size));
    return *this;
}

ProgramBuilder& ProgramBuilder::cb(const CircularBufferConfig& config) {
    auto cs = active_core_spec();

    CreateCircularBuffer(program_, cs, config);
    return *this;
}

ProgramBuilder& ProgramBuilder::cb(
    tt::CBIndex index,
    const std::shared_ptr<distributed::MeshBuffer>& l1_buffer,
    uint32_t num_tiles,
    tt::DataFormat fmt) {
    auto ts = tt::tile_size(fmt);
    auto* backing = l1_buffer->get_backing_buffer();
    TT_FATAL(backing != nullptr, "L1 MeshBuffer has no backing buffer (was it constructed with an explicit address?)");
    // If num_tiles is 0, infer from the buffer size.
    if (num_tiles == 0) {
        num_tiles = static_cast<uint32_t>(backing->aligned_size_per_bank() / ts);
    }
    auto cs = active_core_spec();

    CreateCircularBuffer(
        program_,
        cs,
        CircularBufferConfig(num_tiles * ts, {{index, fmt}})
            .set_page_size(index, ts)
            .set_globally_allocated_address(*backing));
    return *this;
}

KernelRef& ProgramBuilder::reader(
    const std::string& path,
    const std::vector<std::shared_ptr<distributed::MeshBuffer>>& buffers,
    const std::vector<uint32_t>& compile_args) {
    // compile_args first, then TensorAccessorArgs — matches the dominant codebase convention.
    std::vector<uint32_t> all_args(compile_args);
    for (const auto& buf : buffers) {
        TT_FATAL(
            !is_sharded(buf->device_local_config().sharding_args.buffer_layout()),
            "reader() requires interleaved buffers; use the lower-level CreateKernel API for sharded buffers");
        TensorAccessorArgs(*buf).append_to(all_args);
    }

    auto cs = active_core_spec();
    auto defs = active_defines();
    auto nca = active_named_compile_args();

    auto handle = CreateKernel(program_, path, cs, ReaderDataMovementConfig{all_args, defs, nca});
    kernel_refs_.push_back(std::unique_ptr<KernelRef>(new KernelRef(*this, handle, cs)));
    return *kernel_refs_.back();
}

KernelRef& ProgramBuilder::writer(
    const std::string& path,
    const std::vector<std::shared_ptr<distributed::MeshBuffer>>& buffers,
    const std::vector<uint32_t>& compile_args) {
    // compile_args first, then TensorAccessorArgs — matches the dominant codebase convention.
    std::vector<uint32_t> all_args(compile_args);
    for (const auto& buf : buffers) {
        TT_FATAL(
            !is_sharded(buf->device_local_config().sharding_args.buffer_layout()),
            "writer() requires interleaved buffers; use the lower-level CreateKernel API for sharded buffers");
        TensorAccessorArgs(*buf).append_to(all_args);
    }

    auto cs = active_core_spec();
    auto defs = active_defines();
    auto nca = active_named_compile_args();

    auto handle = CreateKernel(program_, path, cs, WriterDataMovementConfig{all_args, defs, nca});
    kernel_refs_.push_back(std::unique_ptr<KernelRef>(new KernelRef(*this, handle, cs)));
    return *kernel_refs_.back();
}

KernelRef& ProgramBuilder::compute(
    const std::string& path,
    MathFidelity fidelity,
    const std::vector<uint32_t>& compile_args) {
    auto cs = active_core_spec();
    auto defs = active_defines();
    auto nca = active_named_compile_args();

    auto handle = CreateKernel(
        program_,
        path,
        cs,
        ComputeConfig{
            .math_fidelity = fidelity,
            .compile_args = compile_args,
            .defines = defs,
            .named_compile_args = nca});
    kernel_refs_.push_back(std::unique_ptr<KernelRef>(new KernelRef(*this, handle, cs)));
    return *kernel_refs_.back();
}

KernelRef& ProgramBuilder::compute(const std::string& path, const ComputeConfig& config) {
    auto cs = active_core_spec();

    auto handle = CreateKernel(program_, path, cs, config);
    kernel_refs_.push_back(std::unique_ptr<KernelRef>(new KernelRef(*this, handle, cs)));
    return *kernel_refs_.back();
}

KernelRef& ProgramBuilder::kernel(
    const std::string& path,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config) {
    auto cs = active_core_spec();

    auto handle = CreateKernel(program_, path, cs, config);
    kernel_refs_.push_back(std::unique_ptr<KernelRef>(new KernelRef(*this, handle, cs)));
    return *kernel_refs_.back();
}

ProgramBuilder& ProgramBuilder::on(const CoreSpec& core_spec) {
    override_core_spec_ = core_spec;
    return *this;
}

ProgramBuilder& ProgramBuilder::defines(const std::map<std::string, std::string>& defs) {
    override_defines_ = defs;
    return *this;
}

ProgramBuilder& ProgramBuilder::named_args(const std::unordered_map<std::string, uint32_t>& args) {
    override_named_compile_args_ = args;
    return *this;
}

std::map<std::string, std::string> ProgramBuilder::active_defines() {
    if (override_defines_.has_value()) {
        auto defs = std::move(*override_defines_);
        override_defines_.reset();
        return defs;
    }
    return {};
}

std::unordered_map<std::string, uint32_t> ProgramBuilder::active_named_compile_args() {
    if (override_named_compile_args_.has_value()) {
        auto nca = std::move(*override_named_compile_args_);
        override_named_compile_args_.reset();
        return nca;
    }
    return {};
}

uint32_t ProgramBuilder::semaphore(uint32_t initial_value) {
    return CreateSemaphore(program_, to_semaphore_core_spec(active_core_spec()), initial_value);
}

uint32_t ProgramBuilder::semaphore(const CoreSpec& cores, uint32_t initial_value) {
    return CreateSemaphore(program_, to_semaphore_core_spec(cores), initial_value);
}

Program ProgramBuilder::build() {
    TT_FATAL(!built_, "ProgramBuilder::build() can only be called once");
    built_ = true;
    return std::move(program_);
}

}  // namespace tt::tt_metal::experimental::ez
