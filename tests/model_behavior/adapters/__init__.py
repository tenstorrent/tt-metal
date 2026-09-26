# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Keep imports lazy: collecting scenarios and testing the driver needs no TT runtime.
from tests.model_behavior.adapters.profiles import PROFILES

ADAPTERS = {
    "galaxy-llama70b": "tests.model_behavior.adapters.galaxy_llama70b",
    **{name: "tests.model_behavior.adapters.transformers" for name in PROFILES},
}
