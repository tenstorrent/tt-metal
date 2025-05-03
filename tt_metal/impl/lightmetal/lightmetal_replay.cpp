// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/lightmetal_replay.hpp>
#include <utility>

#include "lightmetal_binary.hpp"
#include "tt_metal/impl/lightmetal/lightmetal_replay_impl.hpp"

namespace tt::tt_metal {

LightMetalReplay::LightMetalReplay(LightMetalBinary&& binary, IDevice* device) :
    pimpl_(std::make_unique<detail::LightMetalReplayImpl>(std::move(binary), device)) {}

LightMetalReplay::~LightMetalReplay() = default;

LightMetalReplay::LightMetalReplay(LightMetalReplay&&) noexcept = default;
LightMetalReplay& LightMetalReplay::operator=(LightMetalReplay&&) noexcept = default;

bool LightMetalReplay::run() { return pimpl_->run(); }

}  // namespace tt::tt_metal
