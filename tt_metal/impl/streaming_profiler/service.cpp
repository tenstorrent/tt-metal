// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/service.hpp"

#include <array>
#include <memory>

#include <tt_stl/indestructible.hpp>

#include "impl/streaming_profiler/ops_csv.hpp"
#include "impl/streaming_profiler/tracy_consumer.hpp"
#include "impl/streaming_profiler/zone_csv.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal::streaming_profiler {



Service& service() {
    static ttsl::Indestructible<Service> instance;
    return instance.get();
}

    }
}

    }
        }
    }
}

bool Service::is_active() const {
    std::lock_guard<std::mutex> lk(mu_);
}

}  // namespace tt::tt_metal::streaming_profiler
