#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Generate skeleton LLK kernel templates with empty function bodies.

Usage (from codegen/ directory):
    python -m scripts.agent_tools.kernel_template sigmoid --type sfpu --arch quasar
    python -m scripts.agent_tools.kernel_template reduce --type math --arch quasar
    python -m scripts.agent_tools.kernel_template untilize --type pack --arch blackhole
    python -m scripts.agent_tools.kernel_template tilize --type unpack --arch quasar
"""

import argparse
from pathlib import Path

from . import ARCH_DIR_MAP, KERNEL_CONFIGS, get_function_names, settings


def scaffold_kernel(op: str, kernel_type: str, arch: str) -> Path:
    """Create a skeleton kernel header with empty function bodies."""
    config = KERNEL_CONFIGS[kernel_type]
    arch_dir = ARCH_DIR_MAP[arch]
    rel_path = config["file_pattern"].format(op=op)
    full_path = settings.tt_llk_root / arch_dir / rel_path

    impl_name, init_name, uninit_name = get_function_names(op, kernel_type)

    lines = [
        "// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        "#pragma once",
        "#include <cstdint>",
    ]
    for inc in config["template_includes"]:
        lines.append(inc)
    lines.append("")

    if config["namespace_open"]:
        lines.append(config["namespace_open"])
    else:
        for ns in config["using_namespaces"]:
            lines.append(ns)
    lines.append("")

    if init_name:
        lines.append(f"inline void {init_name}() {{")
        lines.append("}")
        lines.append("")

    lines.append(f"inline void {impl_name}() {{")
    lines.append("}")
    lines.append("")

    if uninit_name:
        lines.append(f"inline void {uninit_name}() {{")
        lines.append("}")
        lines.append("")

    if config["namespace_close"]:
        lines.append(config["namespace_close"])
        lines.append("")

    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text("\n".join(lines))
    return full_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate skeleton LLK kernel templates"
    )
    parser.add_argument("op", help="Operation name (e.g. sigmoid, reduce, untilize)")
    parser.add_argument(
        "--type",
        dest="kernel_type",
        required=True,
        choices=KERNEL_CONFIGS.keys(),
        help="Kernel type",
    )
    parser.add_argument(
        "--arch",
        default="quasar",
        choices=ARCH_DIR_MAP.keys(),
        help="Target architecture (default: quasar)",
    )
    args = parser.parse_args()

    path = scaffold_kernel(args.op, args.kernel_type, args.arch)

    impl_name, init_name, uninit_name = get_function_names(args.op, args.kernel_type)

    print(f"Created: {path}")
    print(f"  Type: {args.kernel_type}")
    print(f"  Arch: {args.arch}")
    print(f"  Functions:")
    if init_name:
        print(f"    init:   {init_name}()")
    print(f"    impl:   {impl_name}()")
    if uninit_name:
        print(f"    uninit: {uninit_name}()")


if __name__ == "__main__":
    main()
