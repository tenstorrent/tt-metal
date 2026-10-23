# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Start the Gemma4 prefill service with its production defaults."""

import os
from pathlib import Path


def main():
    os.environ.setdefault("PREFILL_MANIFEST", str(Path(__file__).with_name("manifest.json")))
    from models.demos.common.prefill.runners.prefill_runner import main as serve

    serve()


if __name__ == "__main__":
    main()
