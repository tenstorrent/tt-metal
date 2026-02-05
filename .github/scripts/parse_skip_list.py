#!/usr/bin/env python3
"""Parse TTSim skip list YAML and output pytest --deselect arguments."""

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

    skips = data.get(arch) or []
    # Output simple --deselect args (no special chars in paths now)
    print(" ".join(f"--deselect={path}" for path in skips))


if __name__ == "__main__":
    main()
