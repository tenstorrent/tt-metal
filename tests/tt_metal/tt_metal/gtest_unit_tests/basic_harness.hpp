#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"

class CommandQueueHarness : public ::testing::Test {
   protected:
    tt::ARCH arch;
    Device* device;
    std::unique_ptr<CommandQueue> cq;
    u32 pcie_id;

    void SetUp() override {
        this->arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());

        if (this->arch != tt::ARCH::GRAYSKULL) {
            GTEST_SKIP();
        }

        const int pci_express_slot = 0;
        this->device = tt::tt_metal::CreateDevice(arch, pci_express_slot);
        tt::tt_metal::InitializeDevice(this->device, tt::tt_metal::MemoryAllocator::L1_BANKING);
        this->cq = std::make_unique<CommandQueue>(this->device);

        this->pcie_id = 0;
    }

    void TearDown() override {
        if (this->arch != tt::ARCH::GRAYSKULL) {
            GTEST_SKIP();
        }
        tt::tt_metal::CloseDevice(this->device);
    }
};

class DeviceHarness : public ::testing::Test {
   protected:
    tt::ARCH arch;
    Device* device;
    u32 pcie_id;

    void SetUp() override {
        this->arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        const int pci_express_slot = 0;
        this->pcie_id = pci_express_slot;
        this->device = tt::tt_metal::CreateDevice(arch, pci_express_slot);
        tt::tt_metal::InitializeDevice(this->device);
    }

    void TearDown() override {
        tt::tt_metal::CloseDevice(this->device);
    }
};

class CoreCoordHarness : public ::testing::Test {
   protected:
    CoreRange cr1 = {.start={0, 0}, .end={1, 1}};
    CoreRange cr2 = {.start={3, 3}, .end={5, 4}};
    CoreRange cr3 = {.start={1, 2}, .end={2, 2}};
    CoreRange cr4 = {.start={0, 0}, .end={5, 4}};
    CoreRange cr5 = {.start={1, 0}, .end={6, 4}};
    CoreRange cr6 = {.start={0, 0}, .end={6, 4}};
    CoreRange cr7 = {.start={2, 0}, .end={7, 4}};
    CoreRange cr8 = {.start={0, 0}, .end={7, 4}};
    CoreRange single_core = {.start={1, 1}, .end={1, 1}};

};
