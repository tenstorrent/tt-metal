#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Script to generate configuration serialization code for Inspector.
Parses rtoptions.hpp for no-arg public getters and rtoptions.cpp for
the EnvVarID enum, then generates functions that serialize both into
ConfigurationEntry objects.
"""

import re
import sys


# Return types that can be formatted with fmt::format("{}", ...)
SIMPLE_RETURN_TYPES = {
    "bool",
    "int",
    "unsigned",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "size_t",
    "float",
    "double",
    "std::string",
    "const std::string&",
    "std::filesystem::path",
    "const std::filesystem::path&",
}

# Getters that take parameters — handled in the hand-written section
SKIP_GETTERS = {
    # Parameterized getters (take RunTimeDebugFeatures, CoreType, etc.)
    # These are expanded manually over the enum values in the hand-written section
    "get_feature_enabled",
    "get_feature_cores",
    "get_feature_all_cores",
    "get_feature_chip_ids",
    "get_feature_all_chips",
    "get_feature_processors",
    "get_feature_file_name",
    "get_feature_one_file_per_risc",
    "get_feature_prepend_device_core_risc",
    "get_feature_targets",
    "get_feature_hash_string",
}


def parse_getters(header_file: str) -> list[tuple[str, str]]:
    """
    Parse rtoptions.hpp and extract all no-arg public const getter methods.
    Returns list of (return_type, method_name) tuples.
    """
    with open(header_file, "r") as f:
        content = f.read()

    pattern = r"^\s+([\w:<>&,\s]+?)\s+(get_\w+)\(\)\s*const"

    getters = []
    for match in re.finditer(pattern, content, re.MULTILINE):
        return_type = match.group(1).strip()
        method_name = match.group(2).strip()

        if method_name in SKIP_GETTERS:
            continue

        getters.append((return_type, method_name))

    return getters


def parse_env_var_ids(cpp_file: str) -> list[str]:
    """
    Parse rtoptions.cpp and extract all EnvVarID enum values.
    Returns list of env var name strings.
    """
    with open(cpp_file, "r") as f:
        content = f.read()

    # Find the enum block
    enum_match = re.search(r"enum class EnvVarID\s*\{(.*?)\};", content, re.DOTALL)
    if not enum_match:
        raise ValueError("Could not find EnvVarID enum in rtoptions.cpp")

    enum_body = enum_match.group(1)

    # Extract enum value names (skip comments and blank lines)
    env_vars = []
    for line in enum_body.split("\n"):
        line = line.strip()
        # Remove inline comments
        line = re.sub(r"//.*", "", line).strip()
        # Remove trailing comma
        line = line.rstrip(",").strip()
        # Skip empty lines and section headers
        if not line or line.startswith("//") or line.startswith("/*"):
            continue
        # Should be a valid identifier
        if re.match(r"^[A-Z_][A-Z0-9_]*$", line):
            env_vars.append(line)

    return env_vars


def is_optional_type(return_type: str) -> bool:
    """Check if the return type is std::optional<T>."""
    return return_type.startswith("std::optional<")


def is_simple_type(return_type: str) -> bool:
    """Check if the return type can be directly formatted."""
    if return_type in SIMPLE_RETURN_TYPES:
        return True
    if is_optional_type(return_type):
        inner = return_type[len("std::optional<") : -1].strip()
        return inner in SIMPLE_RETURN_TYPES or inner in {
            "uint32_t",
            "uint64_t",
            "int32_t",
            "int64_t",
        }
    return False


# Types that need enchantum::to_string()
ENCHANTUM_ENUM_TYPES = {
    "TargetDevice",
    "tt_metal::DispatchCoreType",
    "tt_metal::DispatchCoreAxis",
    "tt_metal::KernelBuildOptLevel",
    "tt::tt_fabric::FabricReliabilityMode",
}


def is_enum_type(return_type: str) -> bool:
    """Check if this is a known enum type (possibly wrapped in optional)."""
    inner = return_type
    if is_optional_type(return_type):
        inner = return_type[len("std::optional<") : -1].strip()
    # Strip const& etc
    inner = inner.replace("const ", "").replace("&", "").strip()
    return any(inner.endswith(e) for e in ENCHANTUM_ENUM_TYPES)


def getter_name_to_config_name(method_name: str) -> str:
    """Convert get_foo_bar to foo_bar."""
    assert method_name.startswith("get_")
    return method_name[4:]


def generate_getter_entry(return_type: str, method_name: str) -> str:
    """Generate the serialization code for a single getter."""
    config_name = getter_name_to_config_name(method_name)

    if is_simple_type(return_type):
        if is_optional_type(return_type):
            return f"""    try {{
        auto val = rt.{method_name}();
        entries.push_back({{"{config_name}", val.has_value() ? fmt::format("{{}}", val.value()) : "(unset)", "RtOptions"}});
    }} catch (...) {{
        entries.push_back({{"{config_name}", "(unset)", "RtOptions"}});
    }}
"""
        else:
            return f"""    try {{
        entries.push_back({{"{config_name}", fmt::format("{{}}", rt.{method_name}()), "RtOptions"}});
    }} catch (...) {{
        entries.push_back({{"{config_name}", "(unset)", "RtOptions"}});
    }}
"""
    elif is_enum_type(return_type):
        if is_optional_type(return_type):
            return f"""    try {{
        auto val = rt.{method_name}();
        entries.push_back({{"{config_name}", val.has_value() ? fmt::format("{{}}", static_cast<int>(val.value())) : "(unset)", "RtOptions"}});
    }} catch (...) {{
        entries.push_back({{"{config_name}", "(unset)", "RtOptions"}});
    }}
"""
        else:
            return f"""    try {{
        entries.push_back({{"{config_name}", fmt::format("{{}}", static_cast<int>(rt.{method_name}())), "RtOptions"}});
    }} catch (...) {{
        entries.push_back({{"{config_name}", "(unset)", "RtOptions"}});
    }}
"""
    elif "std::chrono::duration" in return_type:
        return f"""    try {{
        entries.push_back({{"{config_name}", fmt::format("{{}}s", rt.{method_name}().count()), "RtOptions"}});
    }} catch (...) {{
        entries.push_back({{"{config_name}", "(unset)", "RtOptions"}});
    }}
"""
    elif "std::set<std::string>" in return_type:
        return f"""    try {{
        const auto& set_val = rt.{method_name}();
        std::string joined;
        for (const auto& s : set_val) {{
            if (!joined.empty()) joined += ", ";
            joined += s;
        }}
        entries.push_back({{"{config_name}", joined.empty() ? "(empty)" : joined, "RtOptions"}});
    }} catch (...) {{
        entries.push_back({{"{config_name}", "(unset)", "RtOptions"}});
    }}
"""
    else:
        # Unknown complex type — skip with a comment
        return f"    // Skipped: {return_type} {method_name}() — unsupported type\n"


def generate_source(getters: list[tuple[str, str]], env_vars: list[str]) -> str:
    """Generate the source file with serialization functions."""

    source = """// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Auto-generated file - DO NOT EDIT
// Generated from llrt/rtoptions.hpp and llrt/rtoptions.cpp by generate_rtoptions_config.py

#include <tt-metalium/experimental/inspector_config.hpp>
#include "llrt/rtoptions.hpp"
#include <cstdlib>
#include <fmt/format.h>
#include <string>
#include <vector>

namespace tt::tt_metal::inspector {

std::vector<ConfigurationEntry> get_environment_config_entries() {
    std::vector<ConfigurationEntry> entries;
"""

    for env_var in env_vars:
        source += f"    {{\n"
        source += f'        const char* val = std::getenv("{env_var}");\n'
        source += f"        if (val) {{\n"
        source += f'            entries.push_back({{"{env_var}", std::string(val), "Environment"}});\n'
        source += f"        }}\n"
        source += f"    }}\n"

    source += """
    return entries;
}

std::vector<ConfigurationEntry> get_rtoptions_config_entries(const tt::llrt::RunTimeOptions& rt) {
    std::vector<ConfigurationEntry> entries;
"""

    for return_type, method_name in getters:
        source += generate_getter_entry(return_type, method_name)

    # Hand-written section: parameterized getters expanded over RunTimeDebugFeatures
    source += """
    // ---- Parameterized getters expanded over RunTimeDebugFeatures ----
    static const char* feature_names[] = {"dprint", "read_debug_delay", "write_debug_delay", "atomic_debug_delay", "enable_l1_data_cache"};
    for (int i = 0; i < tt::llrt::RunTimeDebugFeatureCount; ++i) {
        auto feature = static_cast<tt::llrt::RunTimeDebugFeatures>(i);
        const char* fname = feature_names[i];
        try {
            entries.push_back({fmt::format("feature_{}_enabled", fname), fmt::format("{}", rt.get_feature_enabled(feature)), "RtOptions"});
        } catch (...) {}
        try {
            entries.push_back({fmt::format("feature_{}_file_name", fname), rt.get_feature_file_name(feature), "RtOptions"});
        } catch (...) {}
        try {
            entries.push_back({fmt::format("feature_{}_one_file_per_risc", fname), fmt::format("{}", rt.get_feature_one_file_per_risc(feature)), "RtOptions"});
        } catch (...) {}
        try {
            entries.push_back({fmt::format("feature_{}_prepend_device_core_risc", fname), fmt::format("{}", rt.get_feature_prepend_device_core_risc(feature)), "RtOptions"});
        } catch (...) {}
        try {
            entries.push_back({fmt::format("feature_{}_all_chips", fname), fmt::format("{}", rt.get_feature_all_chips(feature)), "RtOptions"});
        } catch (...) {}
    }

    // ---- DispatchCoreConfig ----
    try {
        auto config = rt.get_dispatch_core_config();
        static const char* dispatch_core_types[] = {"WORKER", "ETH"};
        static const char* dispatch_core_axes[] = {"ROW", "COL"};
        auto type_idx = static_cast<int>(config.get_dispatch_core_type());
        auto axis_idx = static_cast<int>(config.get_dispatch_core_axis());
        entries.push_back({"dispatch_core_config_type", (type_idx < 2) ? dispatch_core_types[type_idx] : fmt::format("{}", type_idx), "RtOptions"});
        entries.push_back({"dispatch_core_config_axis", (axis_idx < 2) ? dispatch_core_axes[axis_idx] : fmt::format("{}", axis_idx), "RtOptions"});
    } catch (...) {
        entries.push_back({"dispatch_core_config", "(unset)", "RtOptions"});
    }

    // ---- FabricTelemetrySettings ----
    try {
        const auto& fts = rt.get_fabric_telemetry_settings();
        entries.push_back({"fabric_telemetry_enabled", fmt::format("{}", fts.enabled), "RtOptions"});
        entries.push_back({"fabric_telemetry_chips_monitor_all", fmt::format("{}", fts.chips.monitor_all), "RtOptions"});
        entries.push_back({"fabric_telemetry_channels_monitor_all", fmt::format("{}", fts.channels.monitor_all), "RtOptions"});
        entries.push_back({"fabric_telemetry_eriscs_monitor_all", fmt::format("{}", fts.eriscs.monitor_all), "RtOptions"});
        entries.push_back({"fabric_telemetry_stats_mask", fmt::format("{}", fts.stats_mask), "RtOptions"});
    } catch (...) {
        entries.push_back({"fabric_telemetry_settings", "(unset)", "RtOptions"});
    }
"""

    source += """
    return entries;
}

}  // namespace tt::tt_metal::inspector
"""
    return source


def generate_header() -> str:
    """Generate the header file for the serialization functions."""
    return """// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Auto-generated file - DO NOT EDIT
// Generated from llrt/rtoptions.hpp and llrt/rtoptions.cpp by generate_rtoptions_config.py

#pragma once

#include <tt-metalium/experimental/inspector_config.hpp>
#include <vector>

namespace tt::llrt {
class RunTimeOptions;
}

namespace tt::tt_metal::inspector {

std::vector<ConfigurationEntry> get_environment_config_entries();
std::vector<ConfigurationEntry> get_rtoptions_config_entries(const tt::llrt::RunTimeOptions& rt);

}  // namespace tt::tt_metal::inspector
"""


def main():
    if len(sys.argv) != 5:
        print("Usage: generate_rtoptions_config.py <rtoptions_hpp> <rtoptions_cpp> <output_header> <output_source>")
        sys.exit(1)

    header_file = sys.argv[1]
    cpp_file = sys.argv[2]
    output_header = sys.argv[3]
    output_source = sys.argv[4]

    # Parse rtoptions getters
    all_getters = parse_getters(header_file)

    handled = []
    skipped = []
    for rt, name in all_getters:
        if is_simple_type(rt) or is_enum_type(rt) or "std::chrono::duration" in rt or "std::set<std::string>" in rt:
            handled.append((rt, name))
        else:
            skipped.append((rt, name))

    if skipped:
        print(f"Skipped {len(skipped)} getters with unsupported types:")
        for rt, name in skipped:
            print(f"  {rt} {name}()")

    print(f"Generating serialization for {len(handled)} rtoptions getters (+ parameterized feature getters + structs)")

    # Parse EnvVarID enum
    env_vars = parse_env_var_ids(cpp_file)
    print(f"Generating serialization for {len(env_vars)} environment variables")

    header_content = generate_header()
    with open(output_header, "w") as f:
        f.write(header_content)

    source_content = generate_source(handled, env_vars)
    with open(output_source, "w") as f:
        f.write(source_content)


if __name__ == "__main__":
    main()
