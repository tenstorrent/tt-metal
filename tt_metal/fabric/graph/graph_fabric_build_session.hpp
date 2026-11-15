// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>

#include "physical_graph_lowering.hpp"
#include "tt_metal/fabric/fabric_builder.hpp"

namespace tt::tt_fabric::graph {

// Builds and checks the fabric graph once per launch, then builds each chip from it.
// Tensix mux/UDM fabrics aren't covered by the graph yet and get the stock builder.
class GraphFabricBuildSession : public FabricBuildSession {
public:
    explicit GraphFabricBuildSession(FabricContext& fabric_context);

    std::unique_ptr<FabricBuilder> make_device_builder(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        FabricContext& fabric_context,
        CoreType dispatch_core_type) const override;

private:
    std::optional<PhysicalGraph> graph_;
};

}  // namespace tt::tt_fabric::graph
