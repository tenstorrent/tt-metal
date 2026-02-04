#!/usr/bin/env python3
"""Parse TTSim skip list YAML and output pytest --deselect arguments."""

import shlex
import sys
import yaml


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <architecture>", file=sys.stderr)
        sys.exit(1)

    arch = sys.argv[1]
    skip_list_file = ".github/ttsim-skip-list.yaml"

    with open(skip_list_file) as f:
        data = yaml.safe_load(f)

    common = data.get("common") or []
    arch_specific = data.get(arch) or []

    skips = common + arch_specific
    # Use shlex.quote for paths with special characters to ensure proper shell escaping
    print(" ".join(f"--deselect={shlex.quote(path)}" for path in skips))


if __name__ == "__main__":
    main()
