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

// Builds one chip's routers and connections from the graph: their eth channels, channel layouts, link
// peers, sender flags and the chip's handshake master. Stock discovery only checks the eth channels; the
// stock router builders then emit the kernels.
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
    chan_id_t pick_master_router_chan() const override;
    KernelCreationContext kernel_creation_context() const override;

private:
    RouterVcShape shape_of(RouterIndex router) const;
    RouterGraphInputs inputs_of(RouterIndex router) const;
    ComputeMeshRouterBuilder& builder_of(RouterIndex router);

    const PhysicalGraph& graph_;
    std::map<chan_id_t, RouterIndex> on_chip_;  // this chip's routers by eth channel
};

}  // namespace tt::tt_fabric::graph
