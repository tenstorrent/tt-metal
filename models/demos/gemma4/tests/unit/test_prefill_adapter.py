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
    assert int(env["PREFILL_MAX_SEQ_LEN"]) == 262144
    assert int(env["PREFILL_NUM_USERS"]) == 2
    assert env["PREFILL_USE_TRACE"] == "1"
    assert env["PREFILL_LAYER_ACK_D2H"] == "1"


def test_gemma4_cache_head_dim_matches_chunk_table():
    from models.demos.gemma4.tt.attention.global_kv_cache import GLOBAL_PACKED_DIM, SLIDING_HEAD_DIM
    from models.demos.gemma4.tt.runners.kv_chunk_table import CONFIG_NAMES

    adapter = get_adapter("gemma4_31b")

    assert len(CONFIG_NAMES) == 36
    assert adapter.cache_head_dim(0) == GLOBAL_PACKED_DIM
    assert adapter.cache_head_dim(3) == GLOBAL_PACKED_DIM
    assert adapter.cache_head_dim(4) == SLIDING_HEAD_DIM
    assert adapter.cache_head_dim(19) == SLIDING_HEAD_DIM
    assert adapter.cache_head_dim(20) == SLIDING_HEAD_DIM
    assert adapter.cache_head_dim(35) == SLIDING_HEAD_DIM
    assert adapter.cache_head_dim(36) is None
    assert adapter.cache_head_dim(-1) is None
