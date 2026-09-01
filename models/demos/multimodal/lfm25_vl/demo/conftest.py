# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import argparse


def _cli_bool(value):
    """argparse bool that accepts 0/1, true/false (Python bool('False') is True)."""
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected boolean string, got {value!r}")


def pytest_addoption(parser):
    parser.addoption(
        "--dummy_weights",
        action="store",
        default=False,
        type=_cli_bool,
        help="Use dummy/random weights instead of loading HF checkpoints.",
    )
