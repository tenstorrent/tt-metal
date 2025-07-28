// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "api/tt-metalium/kernel.hpp"
#include "jit_build/jit_build_settings.hpp"
#include "jit_build/jit_build_options.hpp"

namespace tt::tt_metal {

class KernelImpl : public Kernel, public JitBuildSettings {
public:
    const std::vector<const ll_api::memory*>& binaries(uint32_t build_key) const;
    uint32_t get_binary_packed_size(IDevice* device, int index) const override;
    uint32_t get_binary_text_size(IDevice* device, int index) const;
    void set_binaries(uint32_t build_key, std::vector<const ll_api::memory*>&& binaries);

    const std::string& get_full_kernel_name() const override;
    void process_defines(std::function<void(const std::string& define, const std::string& value)>) const override;
    void process_compile_time_args(std::function<void(const std::vector<uint32_t>& values)>) const override;

    virtual void set_build_options(JitBuildOptions& build_options) const {}
    virtual void generate_binaries(IDevice* device, JitBuildOptions& build_options) const = 0;
    virtual bool binaries_exist_on_disk(const IDevice* device) const = 0;
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
        const KernelSource& kernel_src,
        const CoreRangeSet& core_range_set,
        const std::vector<uint32_t>& compile_args,
        const std::map<std::string, std::string>& defines) :
        Kernel(kernel_src, core_range_set, compile_args, defines) {}
    // DataMovement kernels have one binary each and Compute kernels have three binaries
    // Different set of binaries per device because kernel compilation is device dependent
    // TODO: break this dependency by https://github.com/tenstorrent/tt-metal/issues/3381
    std::unordered_map<chip_id_t, std::vector<const ll_api::memory*>> binaries_;

    virtual uint8_t expected_num_binaries() const = 0;

    virtual std::vector<std::string> file_paths(IDevice& device) const = 0;
};

class DataMovementKernel : public KernelImpl {
public:
    DataMovementKernel(const KernelSource& kernel_src, const CoreRangeSet& cr_set, const DataMovementConfig& config) :
        KernelImpl(kernel_src, cr_set, config.compile_args, config.defines), config_(config) {
        this->dispatch_class_ =
            magic_enum::enum_integer(HalProcessorClassType::DM) + magic_enum::enum_integer(config.processor);
    }

    ~DataMovementKernel() override = default;

    RISCV processor() const override;

    void generate_binaries(IDevice* device, JitBuildOptions& build_options) const override;
    bool binaries_exist_on_disk(const IDevice* device) const override;
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

    std::vector<std::string> file_paths(IDevice& device) const override;
};

class EthernetKernel : public KernelImpl {
public:
    EthernetKernel(const KernelSource& kernel_src, const CoreRangeSet& cr_set, const EthernetConfig& config) :
        KernelImpl(kernel_src, cr_set, config.compile_args, config.defines), config_(config) {
        this->dispatch_class_ =
            magic_enum::enum_integer(HalProcessorClassType::DM) + magic_enum::enum_integer(config.processor);
    }

    ~EthernetKernel() override = default;

    RISCV processor() const override;

    void generate_binaries(IDevice* device, JitBuildOptions& build_options) const override;
    bool binaries_exist_on_disk(const IDevice* device) const override;
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
    std::vector<std::string> file_paths(IDevice& device) const override;
};

class ComputeKernel : public KernelImpl {
public:
    ComputeKernel(const KernelSource& kernel_src, const CoreRangeSet& cr_set, const ComputeConfig& config) :
        KernelImpl(kernel_src, cr_set, config.compile_args, config.defines), config_(config) {
        this->dispatch_class_ = magic_enum::enum_integer(HalProcessorClassType::COMPUTE);
    }

    ~ComputeKernel() override = default;

    RISCV processor() const override;

    void set_build_options(JitBuildOptions& build_options) const override;
    void generate_binaries(IDevice* device, JitBuildOptions& build_options) const override;
    bool binaries_exist_on_disk(const IDevice* device) const override;
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
    std::vector<std::string> file_paths(IDevice& device) const override;
};

}  // namespace tt::tt_metal
