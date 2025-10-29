// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <umd/device/types/core_coordinates.hpp>
#include <string>

#include "api/tt-metalium/data_types.hpp"
#include "api/tt-metalium/kernel_types.hpp"
#include "api/tt-metalium/runtime_args_data.hpp"
#include "api/tt-metalium/device.hpp"
#include "core_coord.hpp"
#include "hal_types.hpp"
#include "jit_build/jit_build_settings.hpp"
#include "jit_build/jit_build_options.hpp"
#include "program/program_impl.hpp"
#include <enchantum/enchantum.hpp>
#include "llrt.hpp"

namespace tt::tt_metal {

struct KernelSource {
    enum SourceType { FILE_PATH, SOURCE_CODE };

    std::string source_;
    SourceType source_type_;
    // if source_type_ is FILE_PATH, file pointed by path_ exists at time of construction
    std::filesystem::path path_;

    KernelSource(const std::string& source, const SourceType& source_type);

    std::string name() const {
        std::string name;
        if (this->source_type_ == SourceType::FILE_PATH) {
            const std::size_t start_pos_of_name = this->source_.rfind("/") + 1;
            const std::size_t pos_of_dot = this->source_.rfind(".");
            name = this->source_.substr(start_pos_of_name, (pos_of_dot - start_pos_of_name));
        } else {
            name = "Kernel_Source_Code";
        }
        return name;
    }
};


class Kernel {
public:
    using Config = std::variant<DataMovementConfig, EthernetConfig, ComputeConfig>;

    virtual ~Kernel() = default;

    std::string name() const;

    const KernelSource &kernel_source() const { return kernel_src_; }

    const CoreRangeSet &core_range_set() const { return core_range_set_; }

    const std::set<CoreCoord>& cores_with_runtime_args() const { return core_with_runtime_args_; }

    const std::map<std::string, std::string>& defines() const { return defines_; }

    const std::set<CoreCoord> &logical_cores() const;

    std::vector<CoreRange> logical_coreranges() const;

    bool is_on_logical_core(const CoreCoord &logical_core) const;

    std::vector<uint32_t> compile_time_args() const { return compile_time_args_; }
    std::unordered_map<std::string, uint32_t> named_compile_time_args() const { return named_compile_time_args_; }

    std::vector<uint32_t> & runtime_args(const CoreCoord &logical_core);
    RuntimeArgsData & runtime_args_data(const CoreCoord &logical_core);
    std::vector< std::vector< std::vector<uint32_t>> > & runtime_args();
    std::vector< std::vector< RuntimeArgsData > > & runtime_args_data();
    void set_runtime_args_count(CoreRangeSet& core_ranges, uint32_t count);
    std::vector<uint32_t> & common_runtime_args();
    RuntimeArgsData & common_runtime_args_data();
    void set_common_runtime_args_count(uint32_t count);
    uint32_t get_common_runtime_args_count() const { return this->common_runtime_args_count_; }
    uint32_t dispatch_class() { return this->dispatch_class_; }

    virtual bool configure(IDevice* device, const CoreCoord &logical_core, uint32_t base_address, const uint32_t offsets[]) const = 0;

    virtual Config config() const = 0;

    std::string compute_hash() const;

    virtual const std::string& get_full_kernel_name() const = 0;

    void validate_runtime_args_size(size_t num_unique_rt_args, size_t num_common_rt_args, const CoreCoord& logical_core);
    void set_runtime_args(const CoreCoord &logical_core, stl::Span<const uint32_t> runtime_args);
    void set_common_runtime_args(stl::Span<const uint32_t> runtime_args);

    int get_watcher_kernel_id() const { return watcher_kernel_id_; }

    // Get the corresponding core type, processor class, and processor type of the kernel as defined by HAL.
    // The processor type is per-binary, where 0 <= index < expected_num_binaries.
    HalProgrammableCoreType get_kernel_programmable_core_type() const { return this->programmable_core_type_; }
    HalProcessorClassType get_kernel_processor_class() const { return this->processor_class_; }
    virtual uint32_t get_kernel_processor_type(int index) const = 0;

    CoreType get_kernel_core_type() const;
    void set_full_name(const std::string& s) { kernel_full_name_ = s; }
    void add_defines(const std::map<std::string, std::string>& defines);

    virtual uint8_t expected_num_binaries() const = 0;
    virtual uint32_t get_binary_packed_size(IDevice* device, int index) const = 0;

    bool is_idle_eth() const;

    // Collects metadata of the kernel and the binaries within the kernel if device is non-null
    // Note: device is nullable
    detail::KernelMeta meta(IDevice* device) const;

protected:
    HalProgrammableCoreType programmable_core_type_;
    HalProcessorClassType processor_class_;

    int watcher_kernel_id_{};
    KernelSource kernel_src_;
    std::string kernel_full_name_;  // Name + hash
    CoreRangeSet core_range_set_;
    uint8_t dispatch_class_{};
    std::vector<uint32_t> compile_time_args_;
    std::unordered_map<std::string, uint32_t> named_compile_time_args_;
    std::vector< std::vector< std::vector<uint32_t>> > core_to_runtime_args_;
    std::vector< std::vector< RuntimeArgsData> > core_to_runtime_args_data_;
    uint32_t common_runtime_args_count_;
    std::vector<uint32_t> common_runtime_args_;
    RuntimeArgsData common_runtime_args_data_{};
    std::set<CoreCoord> core_with_runtime_args_;
    std::size_t max_runtime_args_per_core_;             // For validation
    CoreCoord core_with_max_runtime_args_;              // For validation
    std::map<std::string, std::string> defines_;        // preprocessor defines. this is to be able to generate generic instances.
    std::set<CoreCoord> logical_cores_;

    virtual std::string config_hash() const = 0;

private:
    void register_kernel_with_watcher();

    Kernel(
        HalProgrammableCoreType programmable_core_type,
        HalProcessorClassType processor_class,
        const KernelSource& kernel_src,
        const CoreRangeSet& core_range_set,
        const std::vector<uint32_t>& compile_args,
        const std::map<std::string, std::string>& defines = {},
        const std::unordered_map<std::string, uint32_t>& named_compile_args = {});

    // Only allow KernelImpl to inherit from Kernel.
    friend class KernelImpl;
};

class KernelImpl : public Kernel, public JitBuildSettings {
public:
    const std::vector<const ll_api::memory*>& binaries(uint32_t build_key) const;
    uint32_t get_binary_packed_size(IDevice* device, int index) const override;
    uint32_t get_binary_text_size(IDevice* device, int index) const;
    void set_binaries(uint32_t build_key, std::vector<const ll_api::memory*>&& binaries);

    const std::string& get_full_kernel_name() const override;
    void process_defines(std::function<void(const std::string& define, const std::string& value)>) const override;
    void process_compile_time_args(std::function<void(const std::vector<uint32_t>& values)>) const override;
    void process_named_compile_time_args(
        std::function<void(const std::unordered_map<std::string, uint32_t>& named_args)>) const override;
    bool binaries_exist_on_disk(const IDevice* device) const;

    virtual void set_build_options(JitBuildOptions& build_options) const {}
    virtual void generate_binaries(IDevice* device, JitBuildOptions& build_options) const = 0;
    virtual void read_binaries(IDevice* device) = 0;

    void register_kernel_elf_paths_with_watcher(IDevice& device) const;

    static KernelImpl& from(Kernel& kernel) {
        // KernelImpl and subclasses are the only implementations of Kernel.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        return static_cast<KernelImpl&>(kernel);
    }

    static const KernelImpl& from(const Kernel& kernel) {
        // KernelImpl and subclasses are the only implementations of Kernel.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        return static_cast<const KernelImpl&>(kernel);
    }

protected:
    KernelImpl(
        HalProgrammableCoreType programmable_core_type,
        HalProcessorClassType processor_class,
        const KernelSource& kernel_src,
        const CoreRangeSet& core_range_set,
        const std::vector<uint32_t>& compile_args,
        const std::map<std::string, std::string>& defines,
        const std::unordered_map<std::string, uint32_t>& named_compile_args) :
        Kernel(
            programmable_core_type,
            processor_class,
            kernel_src,
            core_range_set,
            compile_args,
            defines,
            named_compile_args) {}
    // DataMovement kernels have one binary each and Compute kernels have three binaries
    // Different set of binaries per device because kernel compilation is device dependent
    // TODO: break this dependency by https://github.com/tenstorrent/tt-metal/issues/3381
    std::unordered_map<ChipId, std::vector<const ll_api::memory*>> binaries_;

    std::vector<std::string> file_paths(IDevice& device) const;
};

class DataMovementKernel : public KernelImpl {
public:
    DataMovementKernel(const KernelSource& kernel_src, const CoreRangeSet& cr_set, const DataMovementConfig& config) :
        KernelImpl(
            HalProgrammableCoreType::TENSIX,
            HalProcessorClassType::DM,
            kernel_src,
            cr_set,
            config.compile_args,
            config.defines,
            config.named_compile_args),
        config_(config) {
        this->dispatch_class_ = DISPATCH_CLASS_TENSIX_DM0 + enchantum::to_underlying(config.processor);
    }

    ~DataMovementKernel() override = default;

    uint32_t get_kernel_processor_type(int index) const override;
    void generate_binaries(IDevice* device, JitBuildOptions& build_options) const override;
    void read_binaries(IDevice* device) override;

    bool configure(
        IDevice* device, const CoreCoord& logical_core, uint32_t base_address, const uint32_t offsets[]) const override;

    Config config() const override { return this->config_; }

    void process_defines(std::function<void(const std::string& define, const std::string& value)>) const override;

    std::string_view get_compiler_opt_level() const override;

    std::string_view get_linker_opt_level() const override;

private:
    const DataMovementConfig config_;

    uint8_t expected_num_binaries() const override;

    std::string config_hash() const override;
};

class EthernetKernel : public KernelImpl {
public:
    EthernetKernel(const KernelSource& kernel_src, const CoreRangeSet& cr_set, const EthernetConfig& config) :
        KernelImpl(
            config.eth_mode == Eth::IDLE ? HalProgrammableCoreType::IDLE_ETH : HalProgrammableCoreType::ACTIVE_ETH,
            HalProcessorClassType::DM,
            kernel_src,
            cr_set,
            config.compile_args,
            config.defines,
            config.named_compile_args),
        config_(config) {
        this->dispatch_class_ = DISPATCH_CLASS_ETH_DM0 + enchantum::to_underlying(config.processor);
    }

    ~EthernetKernel() override = default;

    uint32_t get_kernel_processor_type(int index) const override;
    void generate_binaries(IDevice* device, JitBuildOptions& build_options) const override;
    void read_binaries(IDevice* device) override;

    bool configure(
        IDevice* device, const CoreCoord& logical_core, uint32_t base_address, const uint32_t offsets[]) const override;

    Config config() const override { return this->config_; }

    void process_defines(std::function<void(const std::string& define, const std::string& value)>) const override;

    std::string_view get_compiler_opt_level() const override;

    std::string_view get_linker_opt_level() const override;

private:
    const EthernetConfig config_;

    uint8_t expected_num_binaries() const override;

    std::string config_hash() const override;
};

class ComputeKernel : public KernelImpl {
public:
    ComputeKernel(const KernelSource& kernel_src, const CoreRangeSet& cr_set, const ComputeConfig& config) :
        KernelImpl(
            HalProgrammableCoreType::TENSIX,
            HalProcessorClassType::COMPUTE,
            kernel_src,
            cr_set,
            config.compile_args,
            config.defines,
            config.named_compile_args),
        config_(config) {
        // Note: it's wrong to use HalProcessorClassType here, because DM == 0 and COMPUTE == 1,
        // but DISPATCH_CLASS_TENSIX_COMPUTE == 2.
        this->dispatch_class_ = DISPATCH_CLASS_TENSIX_COMPUTE;
    }

    ~ComputeKernel() override = default;

    uint32_t get_kernel_processor_type(int index) const override;
    void set_build_options(JitBuildOptions& build_options) const override;
    void generate_binaries(IDevice* device, JitBuildOptions& build_options) const override;
    void read_binaries(IDevice* device) override;

    bool configure(
        IDevice* device, const CoreCoord& logical_core, uint32_t base_address, const uint32_t offsets[]) const override;

    Config config() const override { return this->config_; }

    void process_defines(std::function<void(const std::string& define, const std::string& value)>) const override;

    std::string_view get_compiler_opt_level() const override;

    std::string_view get_linker_opt_level() const override;

private:
    const ComputeConfig config_;

    uint8_t expected_num_binaries() const override;

    std::string config_hash() const override;
};

}  // namespace tt::tt_metal
