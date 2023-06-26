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
        const int pci_express_slot = 0;
        this->device = tt::tt_metal::CreateDevice(arch, pci_express_slot);
        tt::tt_metal::InitializeDevice(this->device, tt::tt_metal::MemoryAllocator::L1_BANKING);
        this->cq = std::make_unique<CommandQueue>(this->device);

        this->pcie_id = 0;
    }

    void TearDown() override { tt::tt_metal::CloseDevice(this->device); }
};
