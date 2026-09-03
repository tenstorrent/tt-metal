# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import yaml

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


def test_gemma4_disaggregated_prefill_binding_matches_serving_contract():
    binding_path = Path("models/demos/gemma4/tt/runners/manifests/gemma4_binding_disagg_migration_1rank.yaml")
    binding = yaml.safe_load(binding_path.read_text())
    env = binding["global_env"]

    assert binding["rank_bindings"] == [{"rank": 0, "mesh_id": 0, "mesh_host_rank": 0, "env_overrides": {}}]
    assert binding["mesh_graph_desc_path"].endswith("single_bh_galaxy_torus_x_graph_descriptor.textproto")
    assert env["PREFILL_MANIFEST"].endswith("gemma4_31b.json")
    assert (int(env["PREFILL_SP"]), int(env["PREFILL_TP"])) == (8, 4)
    assert int(env["PREFILL_NUM_LAYERS"]) == 60
    assert int(env["PREFILL_CHUNK_SIZE"]) == 8192
    assert int(env["PREFILL_MAX_SEQ_LEN"]) == 262144
    assert int(env["PREFILL_NUM_USERS"]) == 2
    assert env["PREFILL_H2D_SERVICE_ID"] == "gemma4_prefill"
    assert env["PREFILL_USE_TRACE"] == "1"
    assert env["PREFILL_ENABLE_MIGRATION"] == "1"
    assert env["PREFILL_ENABLE_LAYER_ACK"] == "1"
    assert env["PREFILL_MIGRATION_CMD_QUEUE"] == "/mig_ep0_cmd"
    assert env["PREFILL_MIGRATION_TABLE_QUEUE"] == "/mig_ep0_table"
    assert env["PREFILL_MIGRATION_RESP_QUEUE"] == "/mig_ep0_resp"
    assert env["PREFILL_MIGRATION_CLIENT_DIR"] == "${PREFILL_MIGRATION_CLIENT_DIR}"
    assert env["GEMMA4_PREFILL_LOAD_FULL_WEIGHTS"] == "${GEMMA4_PREFILL_LOAD_FULL_WEIGHTS}"
    assert env["HF_HOME"] == "${TT_CACHE_PATH}/hf"
    assert "PREFILL_GEMMA4_SLIDING_CACHE_LEN" not in env
