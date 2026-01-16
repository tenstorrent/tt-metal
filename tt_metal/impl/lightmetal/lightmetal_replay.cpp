// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/lightmetal/lightmetal_replay.hpp>
#include <tt-metalium/experimental/lightmetal/lightmetal_binary.hpp>
#include <utility>

#include "impl/lightmetal/lightmetal_replay_impl.hpp"

namespace tt::tt_metal::experimental::lightmetal {

LightMetalReplay::LightMetalReplay(LightMetalBinary&& binary, IDevice* device) :
    pimpl_(std::make_unique<detail::LightMetalReplayImpl>(std::move(binary), device)) {}

LightMetalReplay::~LightMetalReplay() = default;

LightMetalReplay::LightMetalReplay(LightMetalReplay&&) noexcept = default;
LightMetalReplay& LightMetalReplay::operator=(LightMetalReplay&&) noexcept = default;

bool LightMetalReplay::run() { return pimpl_->run(); }

}  // namespace tt::tt_metal::experimental::lightmetal
