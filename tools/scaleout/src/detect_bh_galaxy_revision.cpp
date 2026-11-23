// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <enchantum/enchantum.hpp>
#include <iostream>
#include <memory>
#include <tt-umd/pcie/pci_device.hpp>

#include <node/node.hpp>

// Prints the detected BH Galaxy node type (BH_GALAXY_REV_AB or BH_GALAXY_REV_C) as the last line of stdout.
// Torus cabling differs between revisions, so scripts use this to pick the matching cabling descriptor.
int main() {
    if (tt::umd::PCIDevice::enumerate_devices().empty()) {
        std::cerr << "No PCIe devices found; cannot detect BH Galaxy revision" << std::endl;
        return 1;
    }
    std::cout << enchantum::to_string(tt::scaleout_tools::get_bh_glx_rev_from_system()) << std::endl;
    return 0;
}
