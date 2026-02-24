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

KernelRef::KernelRef(ProgramBuilder& builder, Type type, std::string path, CoreSpec core_spec) :
    builder_(builder), type_(type), path_(std::move(path)), core_spec_(std::move(core_spec)) {}

KernelRef& KernelRef::defines(const std::map<std::string, std::string>& defs) {
    for (const auto& [k, v] : defs) {
        defines_[k] = v;
    }
    return *this;
}

KernelRef& KernelRef::named_args(const std::unordered_map<std::string, uint32_t>& args) {
    for (const auto& [k, v] : args) {
        named_compile_args_[k] = v;
    }
    return *this;
}

KernelRef& KernelRef::runtime_args(std::initializer_list<uint32_t> args) {
    auto captured = std::vector<uint32_t>(args);
    deferred_runtime_args_.emplace_back(
        [captured](Program& program, KernelHandle handle, const CoreSpec& cs) {
            SetRuntimeArgs(program, handle, cs, captured);
        });
    return *this;
}

KernelRef& KernelRef::runtime_args(const std::vector<uint32_t>& args) {
    deferred_runtime_args_.emplace_back(
        [args](Program& program, KernelHandle handle, const CoreSpec& cs) {
            SetRuntimeArgs(program, handle, cs, tt::stl::Span<const uint32_t>(args));
        });
    return *this;
}

KernelRef& KernelRef::runtime_args(std::function<std::vector<uint32_t>(const CoreCoord&)> fn) {
    deferred_runtime_args_.emplace_back(
        [fn = std::move(fn)](Program& program, KernelHandle handle, const CoreSpec& cs) {
            auto set_for_core = [&](const CoreCoord& core) {
                auto args = fn(core);
                SetRuntimeArgs(
                    program,
                    handle,
                    std::variant<CoreCoord, CoreRange, CoreRangeSet>(core),
                    tt::stl::Span<const uint32_t>(args));
            };

            std::visit(
                [&](auto&& v) {
                    using T = std::decay_t<decltype(v)>;
                    if constexpr (std::is_same_v<T, CoreCoord>) {
                        set_for_core(v);
                    } else if constexpr (std::is_same_v<T, CoreRange>) {
                        for (const auto& core : v) {
                            set_for_core(core);
                        }
                    } else if constexpr (std::is_same_v<T, CoreRangeSet>) {
                        for (const auto& range : v.ranges()) {
                            for (const auto& core : range) {
                                set_for_core(core);
                            }
                        }
                    }
                },
                cs);
        });
    return *this;
}

KernelRef& KernelRef::runtime_args_at(const CoreCoord& core, const std::vector<uint32_t>& args) {
    deferred_runtime_args_.emplace_back(
        [core, args](Program& program, KernelHandle handle, const CoreSpec&) {
            SetRuntimeArgs(
                program,
                handle,
                std::variant<CoreCoord, CoreRange, CoreRangeSet>(core),
                tt::stl::Span<const uint32_t>(args));
        });
    return *this;
}

// Forwarding methods to ProgramBuilder.

ProgramBuilder& KernelRef::cb(tt::CBIndex index, uint32_t num_tiles, tt::DataFormat fmt) {
    return builder_.cb(index, num_tiles, fmt);
}

ProgramBuilder& KernelRef::cb(tt::CBIndex index, tt::DataFormat fmt, uint32_t num_tiles, uint32_t page_size) {
    return builder_.cb(index, fmt, num_tiles, page_size);
}

ProgramBuilder& KernelRef::cb(
    tt::CBIndex index,
    const std::shared_ptr<distributed::MeshBuffer>& l1_buffer,
    uint32_t num_tiles,
    tt::DataFormat fmt) {
    return builder_.cb(index, l1_buffer, num_tiles, fmt);
}

ProgramBuilder& KernelRef::cb(const CircularBufferConfig& config) {
    return builder_.cb(config);
}

KernelRef& KernelRef::reader(
    const std::string& path,
    const std::vector<std::shared_ptr<distributed::MeshBuffer>>& buffers,
    const std::vector<uint32_t>& compile_args) {
    return builder_.reader(path, buffers, compile_args);
}

KernelRef& KernelRef::writer(
    const std::string& path,
    const std::vector<std::shared_ptr<distributed::MeshBuffer>>& buffers,
    const std::vector<uint32_t>& compile_args) {
    return builder_.writer(path, buffers, compile_args);
}

KernelRef& KernelRef::compute(
    const std::string& path,
    MathFidelity fidelity,
    const std::vector<uint32_t>& compile_args) {
    return builder_.compute(path, fidelity, compile_args);
}

KernelRef& KernelRef::compute(const std::string& path, const ComputeConfig& config) {
    return builder_.compute(path, config);
}

KernelRef& KernelRef::kernel(
    const std::string& path,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config) {
    return builder_.kernel(path, config);
}

ProgramBuilder& KernelRef::on(const CoreSpec& core_spec) {
    return builder_.on(core_spec);
}

ProgramBuilder& KernelRef::defines_next(const std::map<std::string, std::string>& defs) {
    return builder_.defines(defs);
}

ProgramBuilder& KernelRef::named_args_next(const std::unordered_map<std::string, uint32_t>& args) {
    return builder_.named_args(args);
}

uint32_t KernelRef::semaphore(uint32_t initial_value) {
    return builder_.semaphore(initial_value);
}

uint32_t KernelRef::semaphore(const CoreSpec& cores, uint32_t initial_value) {
    return builder_.semaphore(cores, initial_value);
}

Program KernelRef::build() {
    return builder_.build();
}

void KernelRef::materialize(Program& program) {
    KernelHandle handle;

    switch (type_) {
        case Type::Reader: {
            std::vector<uint32_t> all_args(compile_args_);
            for (const auto& buf : buffers_) {
                TT_FATAL(
                    !is_sharded(buf->device_local_config().sharding_args.buffer_layout()),
                    "reader() requires interleaved buffers; use the lower-level CreateKernel API for sharded buffers");
                TensorAccessorArgs(*buf).append_to(all_args);
            }
            handle = CreateKernel(
                program, path_, core_spec_, ReaderDataMovementConfig{all_args, defines_, named_compile_args_});
            break;
        }
        case Type::Writer: {
            std::vector<uint32_t> all_args(compile_args_);
            for (const auto& buf : buffers_) {
                TT_FATAL(
                    !is_sharded(buf->device_local_config().sharding_args.buffer_layout()),
                    "writer() requires interleaved buffers; use the lower-level CreateKernel API for sharded buffers");
                TensorAccessorArgs(*buf).append_to(all_args);
            }
            handle = CreateKernel(
                program, path_, core_spec_, WriterDataMovementConfig{all_args, defines_, named_compile_args_});
            break;
        }
        case Type::Compute: {
            handle = CreateKernel(
                program,
                path_,
                core_spec_,
                ComputeConfig{
                    .math_fidelity = fidelity_,
                    .compile_args = compile_args_,
                    .defines = defines_,
                    .named_compile_args = named_compile_args_});
            break;
        }
        case Type::Custom: {
            TT_FATAL(custom_config_.has_value(), "Custom kernel has no config");
            handle = CreateKernel(program, path_, core_spec_, *custom_config_);
            break;
        }
    }

    for (const auto& action : deferred_runtime_args_) {
        action(program, handle, core_spec_);
    }
}

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
    auto cs = active_core_spec();
    auto defs = active_defines();
    auto nca = active_named_compile_args();

    kernel_refs_.push_back(
        std::unique_ptr<KernelRef>(new KernelRef(*this, KernelRef::Type::Reader, path, cs)));
    auto& ref = *kernel_refs_.back();
    ref.buffers_ = buffers;
    ref.compile_args_ = compile_args;
    ref.defines_ = std::move(defs);
    ref.named_compile_args_ = std::move(nca);
    return ref;
}

KernelRef& ProgramBuilder::writer(
    const std::string& path,
    const std::vector<std::shared_ptr<distributed::MeshBuffer>>& buffers,
    const std::vector<uint32_t>& compile_args) {
    auto cs = active_core_spec();
    auto defs = active_defines();
    auto nca = active_named_compile_args();

    kernel_refs_.push_back(
        std::unique_ptr<KernelRef>(new KernelRef(*this, KernelRef::Type::Writer, path, cs)));
    auto& ref = *kernel_refs_.back();
    ref.buffers_ = buffers;
    ref.compile_args_ = compile_args;
    ref.defines_ = std::move(defs);
    ref.named_compile_args_ = std::move(nca);
    return ref;
}

KernelRef& ProgramBuilder::compute(
    const std::string& path,
    MathFidelity fidelity,
    const std::vector<uint32_t>& compile_args) {
    auto cs = active_core_spec();
    auto defs = active_defines();
    auto nca = active_named_compile_args();

    kernel_refs_.push_back(
        std::unique_ptr<KernelRef>(new KernelRef(*this, KernelRef::Type::Compute, path, cs)));
    auto& ref = *kernel_refs_.back();
    ref.fidelity_ = fidelity;
    ref.compile_args_ = compile_args;
    ref.defines_ = std::move(defs);
    ref.named_compile_args_ = std::move(nca);
    return ref;
}

KernelRef& ProgramBuilder::compute(const std::string& path, const ComputeConfig& config) {
    auto cs = active_core_spec();

    kernel_refs_.push_back(
        std::unique_ptr<KernelRef>(new KernelRef(*this, KernelRef::Type::Custom, path, cs)));
    auto& ref = *kernel_refs_.back();
    ref.custom_config_ = config;
    return ref;
}

KernelRef& ProgramBuilder::kernel(
    const std::string& path,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config) {
    auto cs = active_core_spec();

    kernel_refs_.push_back(
        std::unique_ptr<KernelRef>(new KernelRef(*this, KernelRef::Type::Custom, path, cs)));
    auto& ref = *kernel_refs_.back();
    ref.custom_config_ = config;
    return ref;
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

    for (auto& ref : kernel_refs_) {
        ref->materialize(program_);
    }

    return std::move(program_);
}

}  // namespace tt::tt_metal::experimental::ez
