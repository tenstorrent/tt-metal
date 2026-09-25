// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "job_path.hpp"

#include <utility>

namespace bench {

template <typename Fn>
void push_pop_invoke(Fn* slot, Fn&& job) {
    *slot = std::move(job);      // push: move-assign into a live slot
    Fn task = std::move(*slot);  // pop: move-assign out into the worker's local
    task();
}

// The callable's concrete type never reaches this TU, so one instantiation per contender is enough.
template void push_pop_invoke<StdFunction<void()>>(StdFunction<void()>*, StdFunction<void()>&&);
template void push_pop_invoke<ZooFunction<void()>>(ZooFunction<void()>*, ZooFunction<void()>&&);
template void push_pop_invoke<Fu2Function<void()>>(Fu2Function<void()>*, Fu2Function<void()>&&);

}  // namespace bench
