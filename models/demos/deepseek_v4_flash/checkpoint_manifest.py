# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.key_mapping import normalize_hf_key
from models.demos.deepseek_v4_flash.manifest import MODEL_NAME

REAL_CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1
REAL_CHECKPOINT_REPO_ID = "deepseek-ai/DeepSeek-V4-Flash"
MODEL_INDEX_FILENAME = "model.safetensors.index.json"
SUPPORTED_TOPOLOGIES = ("t3k", "galaxy")

_TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
)

_REAL_FLASH_SIGNATURE = {
    "vocab_size": 129280,
    "hidden_size": 4096,
    "moe_intermediate_size": 2048,
    "num_hidden_layers": 43,
    "num_nextn_predict_layers": 1,
    "num_attention_heads": 64,
    "head_dim": 512,
    "q_lora_rank": 1024,
    "n_routed_experts": 256,
    "num_experts_per_tok": 6,
}

_TINY_SYNTHETIC_SIGNATURE = {
    "vocab_size": 64,
    "hidden_size": 32,
    "moe_intermediate_size": 32,
    "head_dim": 8,
    "q_lora_rank": 16,
    "o_lora_rank": 16,
}

_EXPERT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\.(?P<projection>w1|w2|w3)\." r"(?P<kind>weight|scale)$"
)
_MTP_EXPERT_RE = re.compile(r"^mtp\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\.(w1|w2|w3)\.(weight|scale)$")
_LAYER_RE = re.compile(r"^layers\.(?P<layer>\d+)\.")
_MTP_RE = re.compile(r"^mtp\.(?P<layer>\d+)\.")


def build_checkpoint_manifest(snapshot_dir: str | Path) -> dict[str, Any]:
    """Build a JSON-serializable manifest for a local HF snapshot.

    This reads config/tokenizer/index metadata only. If the HF index JSON is
    absent, safetensor files are opened for headers/keys but tensor data is not
    materialized.
    """

    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"HF snapshot directory does not exist: {snapshot_dir}")

    root_config = _read_json_object(snapshot_dir / "config.json")
    inference_config_path = snapshot_dir / "inference" / "config.json"
    inference_config = _read_json_object(inference_config_path) if inference_config_path.is_file() else None
    config = DeepSeekV4FlashConfig.from_hf_configs(root_config, inference_config)
    config_manifest = config.to_manifest_dict()
    checkpoint_kind = _classify_checkpoint(config_manifest)

    weight_index = _load_weight_index(snapshot_dir)
    canonical_names = [_canonical_weight_name(name) for name in weight_index["weight_map"]]
    expected_names = _expected_weight_names(config)
    validation = _validate_weight_names(
        canonical_names,
        expected_names,
        tensor_index_available=weight_index["tensor_index_available"],
        unreadable_shards=weight_index["unreadable_shards"],
        missing_shard_files=weight_index["missing_shard_files"],
    )

    return {
        "schema_version": REAL_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "model_name": MODEL_NAME,
        "source": {
            "snapshot_dir": str(snapshot_dir),
            "repo_id": REAL_CHECKPOINT_REPO_ID,
            "checkpoint_kind": checkpoint_kind,
            "is_real_hf_snapshot": checkpoint_kind == "real_hf",
            "is_tiny_synthetic": checkpoint_kind == "tiny_synthetic",
        },
        "files": {
            "config": "config.json",
            "inference_config": "inference/config.json" if inference_config_path.is_file() else None,
            "tokenizer_files": _existing_files(snapshot_dir, _TOKENIZER_FILENAMES),
            "weight_index": MODEL_INDEX_FILENAME if (snapshot_dir / MODEL_INDEX_FILENAME).is_file() else None,
            "safetensor_shards": weight_index["shards"],
            "shard_count": len(weight_index["shards"]),
            "missing_shard_files": weight_index["missing_shard_files"],
        },
        "config": config_manifest,
        "dimensions": _dimension_summary(config_manifest),
        "mla_indexer": _mla_indexer_summary(config_manifest),
        "weights": {
            "tensor_index_available": weight_index["tensor_index_available"],
            "tensor_count": len(weight_index["weight_map"]),
            "indexed_total_size_bytes": weight_index["total_size"],
            "coverage": _coverage_summary(canonical_names, expected_names),
            "observed": _observed_name_summary(canonical_names),
            "validation": validation,
        },
    }


def build_preprocessing_plan(manifest: dict[str, Any], topology: str) -> dict[str, Any]:
    """Return a dry-run plan for future real-weight preprocessing."""

    validate_checkpoint_manifest(manifest)
    topology = _validate_topology(topology)
    coverage_counts = manifest["weights"]["coverage"]["counts"]
    config = manifest["config"]
    expert_dtype = config.get("expert_dtype")

    actions = [
        _plan_action(
            family="config_tokenizer",
            count=1 + len(manifest["files"]["tokenizer_files"]),
            action="copy",
            source_format="json/tokenizer metadata",
            planned_format="unchanged",
            support="planned",
            note="Copy model config, inference config when present, and tokenizer-side files before tensor conversion.",
        ),
        _plan_action(
            family="embedding_lm_head_norm",
            count=_count_families(coverage_counts, "embedding", "lm_head", "final_norm", "hc_head"),
            action="copy_or_shard",
            source_format=config["torch_dtype"],
            planned_format="BF16 initially; shard per topology once runtime contract is fixed",
            support="planned",
            note="Keep simple until real vocabulary/head sharding is exercised on hardware.",
        ),
        _plan_action(
            family="attention_mla",
            count=_count_families(
                coverage_counts, "attention_weights", "attention_scales", "attention_norms", "attention_sinks"
            ),
            action="copy_quantize_or_pack",
            source_format="FP8 e4m3 weights with UE8M0 scales where .scale tensors are present; BF16/FP32 norms and sinks",
            planned_format="BF16/BFP8/BFP4 pending measurement",
            support="planned_placeholder",
            note="Includes q_a/q_b/wkv/wo_a/wo_b and MLA attention sink/norm tensors.",
        ),
        _plan_action(
            family="compressor_indexer",
            count=_count_families(coverage_counts, "mla_compressor", "indexer", "indexer_compressor"),
            action="copy_quantize_or_pack",
            source_format="BF16/FP32 plus FP8 scale tensors where present",
            planned_format="BF16/BFP8/BFP4 pending measurement",
            support="planned_placeholder",
            note="Covers compressed-attention compressor and ratio-4 indexer tensors.",
        ),
        _plan_action(
            family="router",
            count=_count_families(coverage_counts, "router"),
            action="copy",
            source_format="BF16/FP32 router weights, FP32 bias, int32 hash tid2eid",
            planned_format="BF16/FP32/int32",
            support="planned",
            note="Hash layers use tid2eid; later layers use learned gate bias.",
        ),
        _plan_action(
            family="shared_experts",
            count=_count_families(coverage_counts, "shared_expert_weights", "shared_expert_scales"),
            action="copy_quantize_or_pack",
            source_format="FP8 e4m3 weights with UE8M0 scales",
            planned_format="BF16/BFP8/BFP4 pending measurement",
            support="planned_placeholder",
            note="Shared expert path can be staged before routed expert distribution.",
        ),
        _plan_action(
            family="routed_experts",
            count=_count_families(coverage_counts, "routed_expert_weights", "routed_expert_scales"),
            action="repack_or_convert",
            source_format=_expert_source_format(expert_dtype),
            planned_format="TT BFP4/expert-packed format likely; exact ABI pending real-kernel measurement",
            support="planned_placeholder",
            note=(
                "DeepSeek source experts use FP4 tensor storage, while the tiny path currently assumes its own "
                "uint8 packed debug ABI; real conversion must explicitly map source FP4 blocks and scales."
            ),
        ),
        _plan_action(
            family="hypercompressed_mixing",
            count=_count_families(coverage_counts, "hypercompressed"),
            action="copy",
            source_format="BF16/FP32",
            planned_format="BF16/FP32",
            support="planned",
            note="Covers hc_attn/hc_ffn layer mixing metadata.",
        ),
        _plan_action(
            family="mtp_next_token_prediction",
            count=_count_families(
                coverage_counts,
                "mtp",
                "mtp_attention",
                "mtp_hypercompressed",
                "mtp_routed_experts",
                "mtp_shared_experts",
            ),
            action="none",
            source_format="same families under mtp.*",
            planned_format="unsupported in this slice",
            support="unsupported",
            note="Real Flash has one next-token prediction layer; tiny runtime and converter do not consume it yet.",
        ),
        _plan_action(
            family="tensor_caches",
            count=0,
            action="none",
            source_format="not present in HF checkpoint",
            planned_format="unsupported in this slice",
            support="unsupported",
            note="No TT tensor caches or full conversion artifacts are produced by this dry run.",
        ),
    ]

    return {
        "schema_version": REAL_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "model_name": MODEL_NAME,
        "dry_run": True,
        "source": manifest["source"],
        "topology": _topology_plan(topology),
        "checkpoint": {
            "tensor_count": manifest["weights"]["tensor_count"],
            "shard_count": manifest["files"]["shard_count"],
            "indexed_total_size_bytes": manifest["weights"]["indexed_total_size_bytes"],
            "checkpoint_kind": manifest["source"]["checkpoint_kind"],
        },
        "actions": actions,
        "unsupported": [action for action in actions if action["support"] == "unsupported"],
        "format_notes": [
            "Expert weights likely target BFP4 or an expert-packed variant after real-kernel measurement.",
            "Attention/MLA weights may target BF16, BFP8, or BFP4 depending on accuracy and bandwidth measurements.",
            (
                "DeepSeek FP4 source uses torch.float4_e2m1fn_x2-style packed weights with UE8M0 scales; "
                "the current tiny TT ABI stores uint8-packed debug FP4 with float scales, so this dry run marks "
                "expert repacking as a contract item rather than doing conversion."
            ),
        ],
    }


def validate_checkpoint_manifest(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected manifest object, got {type(manifest).__name__}")
    if manifest.get("schema_version") != REAL_CHECKPOINT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint manifest schema_version {manifest.get('schema_version')!r}")
    if manifest.get("model_name") != MODEL_NAME:
        raise ValueError(f"Expected model_name {MODEL_NAME!r}, got {manifest.get('model_name')!r}")
    for field in ("source", "files", "config", "dimensions", "weights"):
        if not isinstance(manifest.get(field), dict):
            raise ValueError(f"Manifest missing {field} object")
    validation = manifest["weights"].get("validation")
    if not isinstance(validation, dict):
        raise ValueError("Manifest weights missing validation object")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a DeepSeek V4 Flash HF checkpoint without loading tensors.")
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--topology", choices=SUPPORTED_TOPOLOGIES, default="t3k")
    parser.add_argument("--dry-run", action="store_true", help="Also emit a preprocessing dry-run plan.")
    args = parser.parse_args()

    manifest = build_checkpoint_manifest(args.snapshot_dir)
    output: dict[str, Any] = {"manifest": manifest}
    if args.dry_run:
        output["preprocessing_plan"] = build_preprocessing_plan(manifest, args.topology)
    print(json.dumps(output, indent=2, sort_keys=True))


def _load_weight_index(snapshot_dir: Path) -> dict[str, Any]:
    index_path = snapshot_dir / MODEL_INDEX_FILENAME
    if index_path.is_file():
        index = _read_json_object(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Expected non-empty weight_map in {index_path}")
        mapped = {str(key): str(value) for key, value in weight_map.items()}
        shards = sorted(set(mapped.values()))
        return {
            "weight_map": mapped,
            "shards": shards,
            "total_size": _optional_int(index.get("metadata", {}).get("total_size")),
            "tensor_index_available": True,
            "unreadable_shards": [],
            "missing_shard_files": _missing_shard_files(snapshot_dir, shards),
        }

    shard_paths = sorted(snapshot_dir.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No {MODEL_INDEX_FILENAME} or safetensor shards found in {snapshot_dir}")

    weight_map: dict[str, str] = {}
    unreadable_shards: list[dict[str, str]] = []
    for shard_path in shard_paths:
        if _is_lfs_pointer(shard_path):
            unreadable_shards.append({"shard": shard_path.name, "reason": "git_lfs_pointer"})
            continue
        try:
            from safetensors import safe_open

            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    weight_map[str(key)] = shard_path.name
        except Exception as exc:
            unreadable_shards.append({"shard": shard_path.name, "reason": type(exc).__name__})

    return {
        "weight_map": weight_map,
        "shards": [path.name for path in shard_paths],
        "total_size": None,
        "tensor_index_available": bool(weight_map),
        "unreadable_shards": unreadable_shards,
        "missing_shard_files": [],
    }


def _coverage_summary(canonical_names: list[str], expected_names: set[str]) -> dict[str, Any]:
    actual_counts = Counter(_tensor_family(name) for name in canonical_names)
    expected_counts = Counter(_tensor_family(name) for name in expected_names)
    family_names = sorted(set(actual_counts) | set(expected_counts))
    return {
        "counts": {family: int(actual_counts[family]) for family in family_names},
        "expected_counts": {family: int(expected_counts[family]) for family in family_names},
        "categories": [
            {
                "family": family,
                "present": int(actual_counts[family]),
                "expected": int(expected_counts[family]),
                "complete": int(actual_counts[family]) >= int(expected_counts[family]),
            }
            for family in family_names
        ],
    }


def _validate_weight_names(
    canonical_names: list[str],
    expected_names: set[str],
    *,
    tensor_index_available: bool,
    unreadable_shards: list[dict[str, str]],
    missing_shard_files: list[str],
) -> dict[str, Any]:
    actual_names = set(canonical_names)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    errors = []
    warnings = []
    if not tensor_index_available:
        errors.append("No tensor names were available from the HF index or safetensor headers.")
    if missing:
        errors.append(f"Missing {len(missing)} expected DeepSeek V4 Flash tensor name(s).")
    if unexpected:
        warnings.append(f"Found {len(unexpected)} tensor name(s) outside the expected DeepSeek V4 Flash patterns.")
    if unreadable_shards:
        warnings.append(f"{len(unreadable_shards)} safetensor shard(s) could not be read for metadata.")
    if missing_shard_files:
        warnings.append(f"{len(missing_shard_files)} shard file(s) referenced by the HF index are absent locally.")
    status = "error" if errors else "warning" if warnings else "ok"
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "missing_required_count": len(missing),
        "missing_required_examples": missing[:50],
        "unexpected_count": len(unexpected),
        "unexpected_examples": unexpected[:50],
        "unreadable_shards": unreadable_shards,
    }


def _expected_weight_names(config: DeepSeekV4FlashConfig) -> set[str]:
    names = {"embed.weight", "head.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"}
    if _classify_checkpoint(config.to_manifest_dict()) != "tiny_synthetic":
        names.add("norm.weight")

    for layer in range(config.num_hidden_layers):
        names.update(_expected_layer_names(f"layers.{layer}", config, compress_ratio=config.compress_ratios[layer]))
        names.add(
            f"layers.{layer}.ffn.gate.tid2eid" if layer < config.num_hash_layers else f"layers.{layer}.ffn.gate.bias"
        )

    for layer in range(config.num_nextn_predict_layers):
        names.update(_expected_mtp_names(f"mtp.{layer}", config))
    return names


def _expected_layer_names(prefix: str, config: DeepSeekV4FlashConfig, *, compress_ratio: int) -> set[str]:
    names = {
        f"{prefix}.attn_norm.weight",
        f"{prefix}.ffn_norm.weight",
        f"{prefix}.hc_attn_base",
        f"{prefix}.hc_attn_fn",
        f"{prefix}.hc_attn_scale",
        f"{prefix}.hc_ffn_base",
        f"{prefix}.hc_ffn_fn",
        f"{prefix}.hc_ffn_scale",
        f"{prefix}.attn.attn_sink",
        f"{prefix}.attn.q_norm.weight",
        f"{prefix}.attn.kv_norm.weight",
        f"{prefix}.attn.wq_a.weight",
        f"{prefix}.attn.wq_a.scale",
        f"{prefix}.attn.wq_b.weight",
        f"{prefix}.attn.wq_b.scale",
        f"{prefix}.attn.wkv.weight",
        f"{prefix}.attn.wkv.scale",
        f"{prefix}.attn.wo_a.weight",
        f"{prefix}.attn.wo_a.scale",
        f"{prefix}.attn.wo_b.weight",
        f"{prefix}.attn.wo_b.scale",
        f"{prefix}.ffn.gate.weight",
    }
    for projection in ("w1", "w2", "w3"):
        names.add(f"{prefix}.ffn.shared_experts.{projection}.weight")
        names.add(f"{prefix}.ffn.shared_experts.{projection}.scale")
        for expert in range(config.n_routed_experts):
            names.add(f"{prefix}.ffn.experts.{expert}.{projection}.weight")
            names.add(f"{prefix}.ffn.experts.{expert}.{projection}.scale")

    if compress_ratio:
        names.update(
            {
                f"{prefix}.attn.compressor.ape",
                f"{prefix}.attn.compressor.norm.weight",
                f"{prefix}.attn.compressor.wgate.weight",
                f"{prefix}.attn.compressor.wkv.weight",
            }
        )
    if compress_ratio == 4:
        names.update(
            {
                f"{prefix}.attn.indexer.compressor.ape",
                f"{prefix}.attn.indexer.compressor.norm.weight",
                f"{prefix}.attn.indexer.compressor.wgate.weight",
                f"{prefix}.attn.indexer.compressor.wkv.weight",
                f"{prefix}.attn.indexer.weights_proj.weight",
                f"{prefix}.attn.indexer.wq_b.weight",
                f"{prefix}.attn.indexer.wq_b.scale",
            }
        )
    return names


def _expected_mtp_names(prefix: str, config: DeepSeekV4FlashConfig) -> set[str]:
    names = _expected_layer_names(prefix, config, compress_ratio=0)
    names.add(f"{prefix}.ffn.gate.bias")
    names.update(
        {
            f"{prefix}.e_proj.weight",
            f"{prefix}.e_proj.scale",
            f"{prefix}.h_proj.weight",
            f"{prefix}.h_proj.scale",
            f"{prefix}.enorm.weight",
            f"{prefix}.hnorm.weight",
            f"{prefix}.norm.weight",
            f"{prefix}.hc_head_base",
            f"{prefix}.hc_head_fn",
            f"{prefix}.hc_head_scale",
        }
    )
    return names


def _tensor_family(name: str) -> str:
    if name == "embed.weight":
        return "embedding"
    if name == "head.weight":
        return "lm_head"
    if name == "norm.weight":
        return "final_norm"
    if name.startswith("hc_head_"):
        return "hc_head"
    if name.startswith("mtp."):
        if _MTP_EXPERT_RE.match(name):
            return "mtp_routed_experts"
        if ".ffn.shared_experts." in name:
            return "mtp_shared_experts"
        if ".attn." in name:
            return "mtp_attention"
        if ".hc_" in name:
            return "mtp_hypercompressed"
        return "mtp"
    if _EXPERT_RE.match(name):
        return "routed_expert_scales" if name.endswith(".scale") else "routed_expert_weights"
    if ".ffn.shared_experts." in name:
        return "shared_expert_scales" if name.endswith(".scale") else "shared_expert_weights"
    if ".ffn.gate." in name:
        return "router"
    if ".attn.indexer.compressor." in name:
        return "indexer_compressor"
    if ".attn.indexer." in name:
        return "indexer"
    if ".attn.compressor." in name:
        return "mla_compressor"
    if ".attn." in name:
        if name.endswith(".scale"):
            return "attention_scales"
        if name.endswith("norm.weight"):
            return "attention_norms"
        if name.endswith("attn_sink"):
            return "attention_sinks"
        if name.endswith(".weight"):
            return "attention_weights"
        return "attention_other"
    if name.endswith(".attn_norm.weight") or name.endswith(".ffn_norm.weight"):
        return "layer_norms"
    if ".hc_attn_" in name or ".hc_ffn_" in name:
        return "hypercompressed"
    return "unknown"


def _observed_name_summary(canonical_names: list[str]) -> dict[str, Any]:
    layers = sorted({int(match.group("layer")) for name in canonical_names if (match := _LAYER_RE.match(name))})
    mtp_layers = sorted({int(match.group("layer")) for name in canonical_names if (match := _MTP_RE.match(name))})
    routed_experts = sorted(
        {int(match.group("expert")) for name in canonical_names if (match := _EXPERT_RE.match(name))}
    )
    return {
        "layer_ids": layers,
        "layer_count": len(layers),
        "mtp_layer_ids": mtp_layers,
        "mtp_layer_count": len(mtp_layers),
        "routed_expert_count": len(routed_experts),
        "routed_expert_id_min": routed_experts[0] if routed_experts else None,
        "routed_expert_id_max": routed_experts[-1] if routed_experts else None,
    }


def _dimension_summary(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "vocab_size": config["vocab_size"],
        "hidden_size": config["hidden_size"],
        "moe_intermediate_size": config["moe_intermediate_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "num_nextn_predict_layers": config["num_nextn_predict_layers"],
        "num_hash_layers": config["num_hash_layers"],
        "n_routed_experts": config["n_routed_experts"],
        "n_shared_experts": config["n_shared_experts"],
        "num_experts_per_tok": config["num_experts_per_tok"],
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config["num_key_value_heads"],
        "head_dim": config["head_dim"],
        "qk_rope_head_dim": config["qk_rope_head_dim"],
    }


def _mla_indexer_summary(config: dict[str, Any]) -> dict[str, Any]:
    compress_ratios = list(config["compress_ratios"])
    return {
        "q_lora_rank": config["q_lora_rank"],
        "o_lora_rank": config["o_lora_rank"],
        "o_groups": config["o_groups"],
        "sliding_window": config["sliding_window"],
        "compress_rope_theta": config["compress_rope_theta"],
        "compress_ratios": compress_ratios,
        "compress_ratio_counts": {str(ratio): compress_ratios.count(ratio) for ratio in sorted(set(compress_ratios))},
        "index_n_heads": config["index_n_heads"],
        "index_head_dim": config["index_head_dim"],
        "index_topk": config["index_topk"],
    }


def _classify_checkpoint(config: dict[str, Any]) -> str:
    if all(config.get(key) == value for key, value in _REAL_FLASH_SIGNATURE.items()):
        return "real_hf"
    if all(config.get(key) == value for key, value in _TINY_SYNTHETIC_SIGNATURE.items()):
        return "tiny_synthetic"
    return "hf_snapshot"


def _topology_plan(topology: str) -> dict[str, Any]:
    if topology == "t3k":
        return {
            "name": "t3k",
            "mesh_shape_hint": [2, 4],
            "device_count": 8,
            "parallelism_hint": {
                "decode": {"tp": 4, "ep": 2, "sp": 1, "dp": 1},
                "prefill": {"tp": 4, "ep": 1, "sp": 2, "dp": 2},
            },
        }
    if topology == "galaxy":
        return {
            "name": "galaxy",
            "mesh_shape_hint": [8, 4],
            "device_count": 32,
            "parallelism_hint": {
                "decode": {"tp": 4, "ep": 8, "sp": 1, "dp": 1},
                "prefill": {"tp": 4, "ep": 1, "sp": 8, "dp": 8},
            },
            "note": "Draft full-model target; tune once the allocated Galaxy topology is fixed.",
        }
    raise ValueError(f"Unsupported topology {topology!r}")


def _plan_action(
    *,
    family: str,
    count: int,
    action: str,
    source_format: str,
    planned_format: str,
    support: str,
    note: str,
) -> dict[str, Any]:
    return {
        "family": family,
        "tensor_count": int(count),
        "action": action,
        "source_format": source_format,
        "planned_format": planned_format,
        "support": support,
        "note": note,
    }


def _expert_source_format(expert_dtype: str | None) -> str:
    if expert_dtype == "fp4":
        return "DeepSeek FP4, torch.float4_e2m1fn_x2-style packed weights with UE8M0 scales"
    if expert_dtype is None:
        return "unknown expert dtype; infer from safetensor metadata in the real converter"
    return str(expert_dtype)


def _canonical_weight_name(name: str) -> str:
    return normalize_hf_key(name).canonical


def _count_families(counts: dict[str, int], *families: str) -> int:
    return sum(int(counts.get(family, 0)) for family in families)


def _validate_topology(topology: str) -> str:
    if topology not in SUPPORTED_TOPOLOGIES:
        raise ValueError(f"topology must be one of {SUPPORTED_TOPOLOGIES}, got {topology!r}")
    return topology


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def _existing_files(base: Path, filenames: tuple[str, ...]) -> list[str]:
    return [filename for filename in filenames if (base / filename).is_file()]


def _missing_shard_files(snapshot_dir: Path, shard_names: list[str]) -> list[str]:
    return [name for name in shard_names if not (snapshot_dir / name).is_file()]


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(64).startswith(b"version https://git-lfs.github.com/spec/")


if __name__ == "__main__":
    main()
