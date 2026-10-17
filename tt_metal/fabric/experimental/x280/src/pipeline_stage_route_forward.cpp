// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

extern "C" {
#include "rt/x280_idle.h"
#include "rt/x280_pinned_input.h"
#include "x280_hart_protocol.h"
}

#include "backends/x280_route_forward.hpp"
#include "pipeline_stage_common.h"
#include "pipeline_stage_frontend_cxx.hpp"
#include "x280_section_load.hpp"
#include "route_forward_args.h"

namespace {

constexpr uint64_t kParticipants = 1u;
constexpr uint32_t kBackendId = PIPELINE_STAGE_BACKEND_ID_X280_ROUTE_FORWARD;

}  // namespace

extern "C" int main(uint64_t hartid) {
    pipeline_stage::backend_route_forward::state_t state;
    state.result = nullptr;
    state.samples = nullptr;
    state.header = nullptr;

    const bool loaded = x280_load_section<pipeline_stage_route_forward_args_block_t>(
        hartid, kParticipants, state.as_uint32, PIPELINE_STAGE_ROUTE_FORWARD_ARGS_BYTES);

    const bool ok = loaded && validate_section(state.args.section) && layout_is_sane(state.args.layout) &&
                    pipeline_stage::backend_route_forward::sanity_check(state);

    if (!ok) {
        pipeline_stage::refuse_section(state, loaded, kBackendId);
        return 0;
    }

    pipeline_stage::bind_result(state, kBackendId, state.args.section.hop_count);
    if (!pipeline_stage::admit_pinned_input(state, kBackendId)) {
        return 0;
    }

    pipeline_stage::backend_route_forward::seed_route(state);

    using Backend = pipeline_stage::backend_route_forward::Backend;
    pipeline_stage::run_member<Backend>(state);
    return 0;
}
