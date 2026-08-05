# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
LEDGER_PATH = REPO_ROOT / "helper_design/mcast_pipe/migration/ledger.json"


def _migrated_sources():
    ledger = json.loads(LEDGER_PATH.read_text())
    kernels = [REPO_ROOT / entry["kernel"] for entry in ledger["entries"] if entry["status"] == "migrated"]
    factories = {
        REPO_ROOT / binding["factory"] for binding in ledger["host_bindings"] if binding["status"] == "migrated"
    }
    return kernels, sorted(factories)


def test_migrated_host_bindings_treat_helper_outputs_as_opaque_ranges():
    _, factories = _migrated_sources()
    declaration = re.compile(r"\b(?:const\s+)?auto\s+([A-Za-z_]\w*)\s*=\s*([^;]+);", re.DOTALL)

    violations = []
    for factory in factories:
        source = factory.read_text()
        helper_outputs = {
            name
            for name, expression in declaration.findall(source)
            if "compile_time_args(" in expression or "runtime_args(" in expression
        }
        for name in helper_outputs:
            if re.search(rf"\b{re.escape(name)}\s*\[", source):
                violations.append(f"{factory.relative_to(REPO_ROOT)} indexes helper output `{name}`")

    assert not violations, "Helper argument vectors must be copied as complete ranges:\n" + "\n".join(violations)


def test_migrated_kernels_do_not_repeat_rotating_span_as_a_template_argument():
    kernels, _ = _migrated_sources()
    third_template_argument = re.compile(r"\bMcastArgs\s*<\s*[^,>]+\s*,\s*[^,>]+\s*,")

    violations = [
        str(kernel.relative_to(REPO_ROOT)) for kernel in kernels if third_template_argument.search(kernel.read_text())
    ]

    assert not violations, "McastArgs rotating span must come only from the v10 CT wire:\n" + "\n".join(violations)


def test_sort_row_start_readiness_is_pipe_owned():
    kernels, factories = _migrated_sources()
    sort_kernels = [path for path in kernels if "single_row_multi_core.cpp" in path.name]
    sort_factories = [path for path in factories if path.name == "sort_program_factory.cpp"]

    assert len(sort_factories) == 1
    violations = [
        str(path.relative_to(REPO_ROOT)) for path in sort_kernels if "cores_to_coordinator_ready" in path.read_text()
    ]
    factory_source = sort_factories[0].read_text()
    assert ".handshake = true" in factory_source
    assert "row_start_mcast" in factory_source and "substage_mcast" in factory_source
    assert not violations, "Sort row-start readiness must remain inside the handshaked Pipe:\n" + "\n".join(violations)


OPAQUE_BOUNDARY_RULES = {
    "reader_bmm_tile_layout_in1_sender_writer_padding.cpp": [
        (r"rt_args_idx\s*\+=\s*4", "manual runtime skip assumes the helper wire width"),
    ],
    "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp": [
        (r"get_compile_time_arg_val\((?:1[0-9]|2[0-2])\)", "fixed compile-time tail bypasses McastArgs"),
    ],
    "activation_reader_width_sharded.cpp": [
        (
            r"get_compile_time_arg_val\((?:17|18|20|21|22|24|25|26|27|28)\)",
            "fixed compile-time tail bypasses McastArgs",
        ),
        (r"load_config_tensor_if_in_dram<(?:26|27|28)", "fixed config tail bypasses McastArgs"),
    ],
    "reader_mcast_sender_unary_sharded_gn_v2.cpp": [
        (r"get_compile_time_arg_val\((?:1[5-9]|2[0-2])\)", "fixed operation tail bypasses the mcast chain"),
    ],
    "reader_mcast_receiver_unary_sharded_gn_v2.cpp": [
        (r"get_compile_time_arg_val\((?:1[5-9]|20)\)", "fixed operation tail bypasses the mcast chain"),
    ],
    "welford_reader_mcast_sender_unary_sharded_gn_v2.cpp": [
        (r"get_compile_time_arg_val\((?:1[5-9]|2[0-5])\)", "fixed operation tail bypasses the mcast chain"),
    ],
    "welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp": [
        (r"get_compile_time_arg_val\((?:1[5-9]|2[0-3])\)", "fixed operation tail bypasses the mcast chain"),
    ],
}


@pytest.mark.parametrize("kernel,rules", OPAQUE_BOUNDARY_RULES.items(), ids=OPAQUE_BOUNDARY_RULES)
def test_migrated_kernels_resume_from_named_mcast_boundaries(kernel, rules):
    kernels, _ = _migrated_sources()
    matching_paths = [path for path in kernels if path.name == kernel]
    assert len(matching_paths) == 1, f"Expected one migrated ledger entry for {kernel}, found {matching_paths}"

    source = matching_paths[0].read_text()
    violations = [message for pattern, message in rules if re.search(pattern, source)]
    assert not violations, f"{matching_paths[0].relative_to(REPO_ROOT)}: " + "; ".join(violations)
