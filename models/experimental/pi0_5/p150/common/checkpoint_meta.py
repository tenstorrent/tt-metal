# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for reading pi0.5 checkpoint metadata.

Different checkpoint variants (lerobot finetune, openpi upstream, base) ship
different `action_horizon` values. Hardcoding the horizon causes silent PCC
degradation against checkpoints trained for a different horizon (positions
beyond the trained range pull untrained position embeddings through the
network and amplify torch-vs-TTNN divergence). See the
`action-horizon-from-config` memory entry for measured impact.
"""

import json
from pathlib import Path


def action_horizon_from_checkpoint(d: Path, default: int = 50) -> int:
    """Read action_horizon from the checkpoint's config.json.

    Accepts either openpi's `action_horizon` or lerobot's `chunk_size`.
    Falls back to `default` if config.json is missing or unparseable.
    """
    cfg = Path(d) / "config.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text())
            for key in ("action_horizon", "chunk_size"):
                if key in data:
                    return int(data[key])
        except (ValueError, json.JSONDecodeError):
            pass
    return default
