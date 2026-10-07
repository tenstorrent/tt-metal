# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side launch contract for the pinned eight-process TP4 serving stack."""

import ast
import hashlib
import json
import math
import os
import re
from pathlib import Path

from models.demos.qwen38_27b_qb2.tt.precision import precision_fingerprint

PLUGIN_REVISION = "b7e4292e4193cba20abe9c7c68ce489201b2e36b"
MODEL_NAME = "Qwen/Qwen3.8-27B"
OPTIMIZATION_ENV = {
    "QWEN_DECODE_BUCKETS": "1",
    "QWEN_COMPACT_DECODE_RESIDUAL": "1",
    "QWEN_COMPACT_DECODE_MLP": "1",
    "QWEN_BATCHED_DECODE_ROPE": "1",
    "QWEN_COMPACT_DECODE_ATTENTION": "1",
    "QWEN_VLLM_KV_POOL_TOKENS": "1050592",
    "QWEN_BATCHED_PREFILL": "1",
    "QWEN_PREFILL_RESIDUAL_LAYOUT": "sharded_replicated_norm",
    "QWEN_PREFILL_BATCHED_HEAD": "1",
    "QWEN_PREFILL_SKIP_INTERMEDIATE_HEAD": "1",
    "QWEN_PREFILL_STARTUP_WARMUP": "1",
    "QWEN_VLLM_HOST_COMPATIBILITY": "1",
}


def qualified_runtime_environment(precision=None):
    """Preserve the precision artifact covered by the G0 source check."""
    environment = dict(OPTIMIZATION_ENV)
    override = os.environ.get("QWEN_PRECISION_CONFIG")
    if override:
        environment["QWEN_PRECISION_CONFIG"] = override if override == "baseline" else str(Path(override).resolve())
    if precision is not None:
        environment["QWEN_EXPECTED_PRECISION_SHA256"] = precision_fingerprint(precision)
    return environment


def verify_worker_precision(log, expected, *, require_all=True):
    """Reject any reported mismatch during startup; require eight actual workers before eval."""
    pattern = r"\([^\n)]*pid=(\d+)\).*Qwen3\.8 vLLM precision: (\{[^\n]*\})"
    workers = {}
    for pid, value in re.findall(pattern, log):
        policy = ast.literal_eval(value)
        if policy != expected:
            raise ValueError(f"Serving worker {pid} precision differs from the qualified policy")
        workers[int(pid)] = precision_fingerprint(policy)
    if require_all and len(workers) != 8:
        raise ValueError("Serving log must confirm qualified precision on all eight workers")
    return workers


def model_source_hashes(source):
    source = Path(source)
    files = sorted(p for p in (source / "tt").rglob("*") if p.suffix in (".py", ".cpp", ".hpp", ".h"))
    files.append(source / "config/precision.json")
    hashes = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    override = os.environ.get("QWEN_PRECISION_CONFIG")
    if override:
        # Bind the actual override too. Otherwise a passing default-policy G0
        # receipt could silently authorize a different precision at serving.
        content = b"baseline" if override == "baseline" else Path(override).read_bytes()
        hashes["effective_precision_override"] = hashlib.sha256(content).hexdigest()
    return hashes


def verify_qualified_source(receipt, source):
    current = model_source_hashes(source)
    if receipt.get("source_sha256") != current:
        raise ValueError("Model source or precision differs from the eight-replica qualification")
    return current


def qualified_groups(receipt):
    """Never launch a different chip partition under a passing G0 receipt."""
    if (
        receipt.get("passed") is not True
        or receipt.get("state") != "completed"
        or receipt.get("replicas_requested") != 8
        or receipt.get("replicas_executed") != 8
        or receipt.get("replica_mesh") != [1, 4]
        or receipt.get("topology") != "linear"
    ):
        raise ValueError("Eight TP4 replicas must complete Linear G0 qualification before serving")
    comparisons = receipt.get("comparisons", [])
    if len(comparisons) != 8 or {row.get("replica") for row in comparisons} != set(range(8)):
        raise ValueError("Qualification must include all eight replica timing comparisons")
    if any(not math.isfinite(row["ratio"]) or not 0 < row["ratio"] <= 1.03 for row in comparisons):
        raise ValueError("Qualification exceeds the 3 percent concurrent TPOT regression gate")
    loaded = receipt.get("loaded", [])
    if len(loaded) != 8 or {row.get("replica") for row in loaded} != set(range(8)):
        raise ValueError("Qualification must identify all eight physical chip groups")
    groups = [row["device_ids"] for row in sorted(loaded, key=lambda row: row["replica"])]
    ids = [chip for group in groups for chip in group]
    if (
        any(len(group) != 4 for group in groups)
        or any(type(chip) is not int or chip < 0 for chip in ids)
        or len(set(ids)) != 32
    ):
        raise ValueError("Qualification groups must contain 32 distinct nonnegative physical chip IDs")
    return [",".join(map(str, group)) for group in groups]


def additional_config(groups):
    # These are the pinned plugin's serialized discovery results. Seeding them
    # preserves the chip partition actually qualified by G0 instead of silently
    # rediscovering through the transformer's (4,8) parent-mesh reshape. The
    # normal standard-DP worker binding and conflict validation still run.
    return {
        "_tt_standard_dp_visible_groups": groups,
        "_tt_standard_dp_mesh_grids": {group: [1, 4] for group in groups},
        "tt": {
            "fabric_config": "FABRIC_1D",
            "fabric_max_packet_payload_size_bytes": 8192,
            "l1_small_size": 24576,
            "sample_on_device_mode": "all",
            "trace_mode": "decode_only",
            "trace_region_size": 200000000,
        },
    }


def server_command(task_root, checkpoint, groups, *, port=8000):
    return [
        str(Path(task_root) / "serving_env/bin/python"),
        "-u",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(checkpoint),
        "--served-model-name",
        MODEL_NAME,
        "--hf-overrides",
        json.dumps({"architectures": ["TTQwen38ForCausalLM"]}),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--data-parallel-size",
        "8",
        "--block-size",
        "32",
        "--max-num-seqs",
        "16",
        "--max-model-len",
        "262144",
        "--max-num-batched-tokens",
        "262144",
        "--max-logprobs",
        "-1",
        "--async-scheduling",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--no-enable-log-requests",
        "--no-enable-log-outputs",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--enable-auto-tool-choice",
        "--additional-config",
        json.dumps(additional_config(groups)),
    ]


def verify_worker_bindings(log, groups):
    pattern = (
        r"TT worker standard-DP binding: data_parallel_index=\d+ "
        r"data_parallel_rank_local=(\d+) TT_VISIBLE_DEVICES=([0-9,]+) MESH_DEVICE="
    )
    bindings = {}
    for rank, group in re.findall(pattern, log):
        rank = int(rank)
        if rank >= len(groups) or group != groups[rank]:
            raise ValueError("Serving worker chip assignment differs from G0 qualification")
        bindings[rank] = group
    if len(bindings) != 8:
        raise ValueError("Serving log does not confirm all eight standard-DP workers")
    return bindings
