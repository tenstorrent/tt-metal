# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared kernel configuration and helpers for agent tools."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from codegen.config.settings import settings

# ---------------------------------------------------------------------------
# Kernel type configuration
# ---------------------------------------------------------------------------

ARCH_DIR_MAP = {
    "quasar": "tt_llk_quasar",
    "blackhole": "tt_llk_blackhole",
    "wormhole": "tt_llk_wormhole_b0",
}

KERNEL_CONFIGS = {
    "sfpu": {
        "file_pattern": "common/inc/sfpu/ckernel_sfpu_{op}.h",
        "stem_prefix": "ckernel_sfpu_",
        "detect_patterns": ["/sfpu/", "ckernel_sfpu_"],
        "template_includes": [
            '#include "ckernel_ops.h"',
            '#include "ckernel_trisc_common.h"',
            '#include "cmath_common.h"',
        ],
        "wrapper_includes": [
            '#include "ckernel_trisc_common.h"',
            '#include "cmath_common.h"',
        ],
        "namespace_open": "namespace ckernel::sfpu {",
        "namespace_close": "}  // namespace ckernel::sfpu",
        "using_namespaces": [
            "using namespace ckernel;",
            "using namespace ckernel::sfpu;",
        ],
        "init_name": "_init_{op}_",
        "impl_name": "_calculate_{op}_",
        "uninit_name": None,
    },
    "math": {
        "file_pattern": "llk_lib/llk_math_{op}.h",
        "stem_prefix": "llk_math_",
        "detect_patterns": ["llk_math_"],
        "template_includes": [
            '#include "llk_math_common.h"',
        ],
        "wrapper_includes": [
            '#include "ckernel_trisc_common.h"',
            '#include "llk_math_common.h"',
        ],
        "namespace_open": None,
        "namespace_close": None,
        "using_namespaces": ["using namespace ckernel;"],
        "init_name": "_llk_math_{op}_init_",
        "impl_name": "_llk_math_{op}_",
        "uninit_name": "_llk_math_{op}_uninit_",
    },
    "pack": {
        "file_pattern": "llk_lib/llk_pack_{op}.h",
        "stem_prefix": "llk_pack_",
        "detect_patterns": ["llk_pack_"],
        "template_includes": [
            '#include "ckernel_trisc_common.h"',
            '#include "cpack_common.h"',
            '#include "llk_pack_common.h"',
        ],
        "wrapper_includes": [
            '#include "ckernel_trisc_common.h"',
            '#include "llk_pack_common.h"',
        ],
        "namespace_open": None,
        "namespace_close": None,
        "using_namespaces": ["using namespace ckernel;"],
        "init_name": "_llk_pack_{op}_init_",
        "impl_name": "_llk_pack_{op}_",
        "uninit_name": "_llk_pack_{op}_uninit_",
    },
    "unpack": {
        "file_pattern": "llk_lib/llk_unpack_{op}.h",
        "stem_prefix": "llk_unpack_",
        "detect_patterns": ["llk_unpack_"],
        "template_includes": [
            '#include "ckernel_trisc_common.h"',
            '#include "cunpack_common.h"',
            '#include "llk_unpack_common.h"',
        ],
        "wrapper_includes": [
            '#include "ckernel_trisc_common.h"',
            '#include "llk_unpack_common.h"',
        ],
        "namespace_open": None,
        "namespace_close": None,
        "using_namespaces": [
            "using namespace ckernel;",
            "using namespace ckernel::trisc;",
        ],
        "init_name": "_llk_unpack_{op}_init_",
        "impl_name": "_llk_unpack_{op}_",
        "uninit_name": "_llk_unpack_{op}_uninit_",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def detect_kernel_type(filepath: Path) -> str:
    """Detect kernel type from file path or name."""
    path_str = str(filepath)
    for ktype, config in KERNEL_CONFIGS.items():
        for pattern in config["detect_patterns"]:
            if pattern in path_str or pattern in filepath.name:
                return ktype
    return "sfpu"


def get_op_name(filepath: Path, kernel_type: str) -> str:
    """Extract operation name from filename using kernel type config."""
    return filepath.stem.replace(KERNEL_CONFIGS[kernel_type]["stem_prefix"], "")


def get_function_names(op: str, kernel_type: str) -> tuple[str, str | None, str | None]:
    """Return (impl_name, init_name, uninit_name) for a kernel."""
    config = KERNEL_CONFIGS[kernel_type]
    impl_name = config["impl_name"].format(op=op)
    init_name = config["init_name"].format(op=op) if config["init_name"] else None
    uninit_name = (
        config["uninit_name"].format(op=op) if config.get("uninit_name") else None
    )
    return impl_name, init_name, uninit_name
