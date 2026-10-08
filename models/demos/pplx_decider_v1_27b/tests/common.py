# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path


def load_example_state(example, ckpt_dir):
    if "state_file" in example:
        return (Path(ckpt_dir) / example["state_file"]).read_text()
    return example["state"]


def ids_sha256(input_ids):
    return hashlib.sha256(json.dumps(input_ids).encode()).hexdigest()
