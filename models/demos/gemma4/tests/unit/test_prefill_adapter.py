# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

from models.demos.common.prefill.adapter import get_adapter


def test_gemma4_prefill_adapter_is_registered_and_import_light():
    sys.modules.pop("models.demos.gemma4.tt.tt_prefill_runtime", None)
    adapter = get_adapter("gemma4_31b")

    assert adapter.name == "gemma4_31b"
    assert adapter.model_config.NUM_LAYERS == 60
    assert adapter.pipeline_activation_emb_tp_sharded is False
    assert "models.demos.gemma4.tt.tt_prefill_runtime" not in sys.modules


def test_gemma4_prefill_manifest_matches_supported_runtime():
    manifest_path = Path("models/demos/gemma4/tt/runners/manifests/gemma4_31b.json")
    env = json.loads(manifest_path.read_text())["env"]

    assert env["PREFILL_MODEL"] == "gemma4_31b"
    assert (int(env["PREFILL_SP"]), int(env["PREFILL_TP"])) == (8, 4)
    assert int(env["PREFILL_NUM_LAYERS"]) == 60
    assert int(env["PREFILL_CHUNK_SIZE"]) == 8192
    assert int(env["PREFILL_MAX_SEQ_LEN"]) == 65536
    assert int(env["PREFILL_NUM_USERS"]) == 8
    assert env["PREFILL_USE_TRACE"] == "1"
