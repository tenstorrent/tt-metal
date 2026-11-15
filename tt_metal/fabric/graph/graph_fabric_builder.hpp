// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>

#include "physical_graph_lowering.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "tt_metal/fabric/compute_mesh_router_builder.hpp"
#include "tt_metal/fabric/fabric_builder.hpp"

namespace tt::tt_fabric::graph {

// Builds one chip's routers and connections from the graph. Stock discovery still supplies the
// live eth channels; the stock router builders then emit the kernels.
class GraphFabricBuilder : public FabricBuilder {
public:
    GraphFabricBuilder(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        FabricContext& fabric_context,
        CoreType dispatch_core_type,
        const PhysicalGraph& graph);

    void connect_routers() override;

protected:
    std::unique_ptr<FabricRouterBuilder> make_router(const RouterLocation& location) override;

private:
    RouterVcShape shape_of(RouterIndex router) const;
    void check_layout(RouterIndex router, const FabricRouterBuilder& built) const;
    ComputeMeshRouterBuilder& builder_of(RouterIndex router);

    const PhysicalGraph& graph_;
    std::map<std::pair<RoutingDirection, uint32_t>, RouterIndex> on_chip_;  // (direction, plane) -> this chip's routers
    std::map<RouterIndex, chan_id_t> eth_chan_;
};

}  // namespace tt::tt_fabric::graph
