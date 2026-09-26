# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Select an expert control without passing JSON through Tracy's shell parser."""

import argparse
import json
import sys

from models.autoports.google_gemma_4_26b_a4b_it.tests import run_optimized_decoder


def main():
    cases = {
        "indexed22k44": {"expert_grid": [11, 2]},
        "indexed_separatek44": {"expert_split": True},
        "expanded44k44": {"indexed_experts": False},
    }
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--expert-case", choices=cases, required=True)
    args, rest = parser.parse_known_args()
    if "--default-overrides" in rest:
        parser.error("The named case owns its default overrides")
    previous = sys.argv
    try:
        sys.argv = [sys.argv[0], "--defaults", "--default-overrides", json.dumps(cases[args.expert_case]), *rest]
        run_optimized_decoder.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
