// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/program/program_cache_impl.hpp"

#include <tt_stl/assert.hpp>

#include <utility>

namespace tt::tt_metal::program_cache::detail {

ProgramCache::ProgramCache() : impl_(std::make_unique<ProgramCacheImpl>()) {}

ProgramCache::ProgramCache(ProgramCacheImpl impl) : impl_(std::make_unique<ProgramCacheImpl>(std::move(impl))) {}

ProgramCache::ProgramCache(ProgramCache&& other) noexcept = default;
ProgramCache& ProgramCache::operator=(ProgramCache&& other) noexcept = default;
ProgramCache::~ProgramCache() = default;

ProgramCacheImpl& ProgramCache::impl() {
    TT_FATAL(impl_ != nullptr, "ProgramCache impl is null");
    return *impl_;
}

const ProgramCacheImpl& ProgramCache::impl() const {
    TT_FATAL(impl_ != nullptr, "ProgramCache impl is null");
    return *impl_;
}

bool ProgramCache::contains(const ProgramCacheKey& program_key) const { return impl().contains(program_key); }

CachedProgramFactory& ProgramCache::get(const ProgramCacheKey& program_key) { return impl().get(program_key); }

void ProgramCache::insert(const ProgramCacheKey& program_key, CachedProgramFactory&& program) {
    impl().insert(program_key, std::move(program));
}

bool ProgramCache::cache_misses_allowed() const { return impl().cache_misses_allowed(); }

bool ProgramCache::is_enabled() const { return impl().is_enabled(); }

}  // namespace tt::tt_metal::program_cache::detail
