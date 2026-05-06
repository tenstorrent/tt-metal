# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.manifest import TT_MANIFEST_FILENAME, load_tt_manifest
from models.demos.deepseek_v4_flash.weight_inventory import (
    WeightInventoryReport,
    build_weight_inventory_report,
    estimate_max_seq_len_supported,
)

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
MODEL_INDEX_FILENAME = "model.safetensors.index.json"
TOKENIZER_FILENAMES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json")
ENCODING_FILENAMES = ("encoding/encoding_dsv4.py",)
_MAX_EVIDENCE_ITEMS = 8

RuntimePhase = Literal["prefill", "decode", "generate"]


@dataclass(frozen=True)
class RuntimeBlocker:
    code: str
    phase: str
    detail: str
    next_action: str
    evidence: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "phase": self.phase,
            "detail": self.detail,
            "next_action": self.next_action,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class RuntimeBlockerReport:
    phase: RuntimePhase
    model_id: str
    snapshot_dir: str
    preprocessed_dir: str
    blocked: bool
    blockers: tuple[RuntimeBlocker, ...]
    max_seq_len_supported: int | None
    estimated_max_seq_len_if_device_path_ready: int | None
    configured_max_position_embeddings: int | None
    device_topology: str
    tracing_enabled: bool
    batch_size: int
    dtype_quantization: str
    context: Mapping[str, Any]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "model_id": self.model_id,
            "snapshot_dir": self.snapshot_dir,
            "preprocessed_dir": self.preprocessed_dir,
            "blocked": self.blocked,
            "blockers": [blocker.to_mapping() for blocker in self.blockers],
            "max_seq_len_supported": self.max_seq_len_supported,
            "estimated_max_seq_len_if_device_path_ready": self.estimated_max_seq_len_if_device_path_ready,
            "configured_max_position_embeddings": self.configured_max_position_embeddings,
            "device_topology": self.device_topology,
            "tracing_enabled": self.tracing_enabled,
            "batch_size": self.batch_size,
            "dtype_quantization": self.dtype_quantization,
            "context": dict(self.context),
        }


@dataclass(frozen=True)
class HardwareMeshProbeResult:
    target_topology: str
    visible_devices: str
    mesh_graph_desc_path: str
    opened: bool
    available_devices: int | None
    returncode: int | None
    timed_out: bool
    error: str | None
    evidence: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "target_topology": self.target_topology,
            "visible_devices": self.visible_devices,
            "mesh_graph_desc_path": self.mesh_graph_desc_path,
            "opened": self.opened,
            "available_devices": self.available_devices,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "error": self.error,
            "evidence": list(self.evidence),
        }


class DeepSeekRuntimeBlocked(RuntimeError):
    """Raised when the integrated runtime cannot execute without an invalid fallback."""

    def __init__(self, report: RuntimeBlockerReport):
        self.report = report
        codes = ", ".join(blocker.code for blocker in report.blockers)
        super().__init__(f"DeepSeek V4 Flash runtime is blocked in {report.phase}: {codes}")


class HostBoundaryViolation(RuntimeError):
    """Raised when guarded model execution attempts a known host fallback boundary."""


@dataclass(frozen=True)
class GuardedSymbol:
    module_name: str
    attr_path: str
    label: str


class RuntimeHostBoundaryGuard(AbstractContextManager["RuntimeHostBoundaryGuard"]):
    """Patch known host readback/fallback helpers while primary runtime code executes."""

    def __init__(self, symbols: Sequence[GuardedSymbol] | None = None):
        self._symbols = tuple(symbols) if symbols is not None else default_guarded_symbols()
        self._patches: list[tuple[object, str, object]] = []
        self.guarded_labels: list[str] = []

    def __enter__(self) -> "RuntimeHostBoundaryGuard":
        for symbol in self._symbols:
            target = _resolve_guard_target(symbol)
            if target is None:
                continue
            parent, attr = target
            original = getattr(parent, attr)
            setattr(parent, attr, _blocked_host_boundary(symbol.label))
            self._patches.append((parent, attr, original))
            self.guarded_labels.append(symbol.label)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for parent, attr, original in reversed(self._patches):
            setattr(parent, attr, original)
        self._patches.clear()


@dataclass(frozen=True)
class PrefillState:
    input_ids: torch.Tensor
    prompt_length: int
    current_position: int
    cache_owner: Literal["device"]
    page_table: object
    layer_caches: tuple[object, ...]


@dataclass(frozen=True)
class GenerationResult:
    token_ids: tuple[int, ...]
    report: Mapping[str, Any]


class TtDeepSeekV4FlashRuntime:
    """Primary batch-1 runtime contract for real DeepSeek V4 Flash execution.

    This class is intentionally the only public model path for the bringup. Until
    every mandatory operation is resident in the TTNN/paged/traced path, calls
    fail with a structured blocker report instead of silently falling back to the
    scaffold smoke runners or torch reference code.
    """

    def __init__(
        self,
        snapshot_dir: str | Path,
        *,
        preprocessed_dir: str | Path | None = None,
        mesh_shape: tuple[int, int] = (2, 4),
        batch_size: int = 1,
        tracing_enabled: bool = True,
        dtype_quantization: str = "real checkpoint: FP8 non-expert + FP4 experts, TTNN decode target BF16/BFP8/BFP4",
    ):
        if batch_size != 1:
            raise ValueError(f"DeepSeek V4 Flash runtime is batch-1 only in this bringup, got batch_size={batch_size}")
        self.snapshot_dir = Path(snapshot_dir).expanduser().resolve()
        self.preprocessed_dir = _resolve_preprocessed_dir(self.snapshot_dir, preprocessed_dir)
        self.mesh_shape = tuple(int(dim) for dim in mesh_shape)
        if self.mesh_shape != (2, 4):
            raise ValueError(f"Blackhole Loudbox target requires mesh_shape=(2, 4), got {self.mesh_shape}")
        self.batch_size = int(batch_size)
        self.tracing_enabled = bool(tracing_enabled)
        self.dtype_quantization = dtype_quantization
        self._raw_config: dict[str, Any] | None = None
        self._config: DeepSeekV4FlashConfig | None = None
        self._config_error: str | None = None
        self._preprocessed_manifest: dict[str, Any] | None = None
        self._preprocessed_error: str | None = None
        self._preprocessed_missing_artifacts: tuple[str, ...] = ()
        self._weight_inventory: WeightInventoryReport | None = None
        self._weight_inventory_error: str | None = None
        self._load_config_if_available()
        self._load_preprocessed_if_available()
        self._load_weight_inventory_if_available()

    @classmethod
    def from_hf_snapshot(cls, snapshot_dir: str | Path, **kwargs) -> "TtDeepSeekV4FlashRuntime":
        return cls(snapshot_dir, **kwargs)

    @property
    def config(self) -> DeepSeekV4FlashConfig | None:
        return self._config

    def preflight(
        self,
        phase: RuntimePhase = "generate",
        *,
        context: Mapping[str, Any] | None = None,
        check_hardware: bool = False,
        hardware_timeout_s: float = 90.0,
        visible_devices: str = "0,1,2,3,4,5,6,7",
    ) -> RuntimeBlockerReport:
        report_context = {} if context is None else dict(context)
        blockers = [
            *self._asset_blockers(phase),
            *self._implementation_blockers(phase),
        ]
        if check_hardware:
            hardware_blockers, hardware_context = self._hardware_blockers(
                phase,
                timeout_s=hardware_timeout_s,
                visible_devices=visible_devices,
            )
            blockers.extend(hardware_blockers)
            report_context["hardware_preflight"] = hardware_context.to_mapping()
        return RuntimeBlockerReport(
            phase=phase,
            model_id=MODEL_ID,
            snapshot_dir=str(self.snapshot_dir),
            preprocessed_dir=str(self.preprocessed_dir),
            blocked=bool(blockers),
            blockers=tuple(blockers),
            max_seq_len_supported=None if blockers else self.configured_max_position_embeddings,
            estimated_max_seq_len_if_device_path_ready=self._estimated_max_seq_len_if_device_path_ready(),
            configured_max_position_embeddings=self.configured_max_position_embeddings,
            device_topology=f"Blackhole Loudbox {self.mesh_shape[0]}x{self.mesh_shape[1]} ({self.mesh_shape[0] * self.mesh_shape[1]} devices)",
            tracing_enabled=self.tracing_enabled,
            batch_size=self.batch_size,
            dtype_quantization=self.dtype_quantization,
            context=report_context,
        )

    def prefill(self, input_ids: Sequence[int] | torch.Tensor) -> PrefillState:
        ids = self._normalize_input_ids(input_ids, label="input_ids")
        report = self.preflight("prefill", context={"prompt_length": int(ids.shape[1])})
        self._raise_if_blocked(report)
        raise AssertionError("unreachable until preflight blockers are cleared")

    def decode_step(self, state: PrefillState, token_id: int | torch.Tensor | None = None) -> tuple[int, PrefillState]:
        if not isinstance(state, PrefillState):
            raise TypeError(f"state must be PrefillState, got {type(state).__name__}")
        next_token = None if token_id is None else self._normalize_decode_token(token_id)
        report = self.preflight(
            "decode",
            context={
                "current_position": state.current_position,
                "has_supplied_token": next_token is not None,
                "cache_owner": state.cache_owner,
            },
        )
        self._raise_if_blocked(report)
        raise AssertionError("unreachable until preflight blockers are cleared")

    def generate(self, input_ids: Sequence[int] | torch.Tensor, *, max_new_tokens: int) -> GenerationResult:
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
        ids = self._normalize_input_ids(input_ids, label="input_ids")
        report = self.preflight(
            "generate",
            context={"prompt_length": int(ids.shape[1]), "requested_new_tokens": int(max_new_tokens)},
        )
        self._raise_if_blocked(report)
        raise AssertionError("unreachable until preflight blockers are cleared")

    @property
    def configured_max_position_embeddings(self) -> int | None:
        if self._raw_config is None:
            return None
        value = self._raw_config.get("max_position_embeddings")
        if value is None:
            rope_scaling = self._raw_config.get("rope_scaling")
            if isinstance(rope_scaling, dict):
                value = rope_scaling.get("original_max_position_embeddings")
        return None if value is None else int(value)

    def _load_config_if_available(self) -> None:
        config_path = self.snapshot_dir / "config.json"
        if not config_path.is_file():
            return
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                raw_config = json.load(handle)
            if not isinstance(raw_config, dict):
                raise ValueError(f"{config_path} must contain a JSON object")
            self._raw_config = raw_config
            self._config = DeepSeekV4FlashConfig.from_model_path(self.snapshot_dir)
        except Exception as exc:  # noqa: BLE001 - surfaced as a structured blocker.
            self._config_error = str(exc)

    def _load_preprocessed_if_available(self) -> None:
        if not self.preprocessed_dir.is_dir():
            return
        try:
            manifest = load_tt_manifest(self.preprocessed_dir)
            self._preprocessed_manifest = manifest
            self._preprocessed_missing_artifacts = tuple(_missing_preprocessed_artifacts(self.preprocessed_dir, manifest))
        except Exception as exc:  # noqa: BLE001 - surfaced as a structured blocker.
            self._preprocessed_error = str(exc)

    def _load_weight_inventory_if_available(self) -> None:
        if not self._has_complete_tt_preprocessed_weights:
            return
        try:
            self._weight_inventory = build_weight_inventory_report(self.preprocessed_dir, mesh_shape=self.mesh_shape)
        except Exception as exc:  # noqa: BLE001 - surfaced as a structured blocker.
            self._weight_inventory_error = str(exc)

    def _asset_blockers(self, phase: RuntimePhase) -> list[RuntimeBlocker]:
        blockers: list[RuntimeBlocker] = []
        if not self.snapshot_dir.is_dir():
            blockers.append(
                RuntimeBlocker(
                    code="checkpoint.snapshot_missing",
                    phase=phase,
                    detail=f"HF snapshot directory does not exist: {self.snapshot_dir}",
                    next_action="Download or point DSV4_FLASH_REAL_SNAPSHOT_DIR at deepseek-ai/DeepSeek-V4-Flash.",
                )
            )
            return blockers
        if not (self.snapshot_dir / "config.json").is_file():
            blockers.append(
                RuntimeBlocker(
                    code="checkpoint.config_missing",
                    phase=phase,
                    detail="config.json is required before any real model runtime can be constructed.",
                    next_action="Materialize the deepseek-ai/DeepSeek-V4-Flash snapshot with config.json present.",
                )
            )
        if self._config_error is not None:
            blockers.append(
                RuntimeBlocker(
                    code="checkpoint.config_invalid",
                    phase=phase,
                    detail=self._config_error,
                    next_action="Fix the local HF snapshot/config mismatch before running TTNN.",
                )
            )
        if not (self.snapshot_dir / MODEL_INDEX_FILENAME).is_file():
            blockers.append(
                RuntimeBlocker(
                    code="checkpoint.weight_index_missing",
                    phase=phase,
                    detail=f"{MODEL_INDEX_FILENAME} is required for real weight materialization.",
                    next_action="Fetch the full safetensors snapshot, not just config/tokenizer assets.",
                )
            )
        else:
            blockers.extend(self._weight_index_blockers(phase))
        missing_tokenizer = [name for name in TOKENIZER_FILENAMES if not (self.snapshot_dir / name).is_file()]
        if missing_tokenizer:
            blockers.append(
                RuntimeBlocker(
                    code="tokenizer.assets_missing",
                    phase=phase,
                    detail=f"Missing tokenizer-compatible assets: {missing_tokenizer}",
                    next_action="Fetch tokenizer.json, tokenizer_config.json, and generation_config.json from the HF snapshot.",
                )
            )
        missing_encoding = [name for name in ENCODING_FILENAMES if not (self.snapshot_dir / name).is_file()]
        if missing_encoding:
            blockers.append(
                RuntimeBlocker(
                    code="tokenizer.encoding_missing",
                    phase=phase,
                    detail=f"Missing DeepSeek V4 prompt encoding assets: {missing_encoding}",
                    next_action="Fetch the encoding/encoding_dsv4.py asset so token IDs are produced from the model-compatible prompt format.",
                )
            )
        blockers.extend(self._preprocessed_blockers(phase))
        if self._weight_inventory_error is not None:
            blockers.append(
                RuntimeBlocker(
                    code="weights.inventory_invalid",
                    phase=phase,
                    detail=f"Could not build TT-preprocessed weight inventory: {self._weight_inventory_error}",
                    next_action="Fix the TT-preprocessed artifacts so the runtime can plan device ownership.",
                    evidence=(str(self.preprocessed_dir),),
                )
            )
        return blockers

    def _preprocessed_blockers(self, phase: RuntimePhase) -> list[RuntimeBlocker]:
        if not self.preprocessed_dir.is_dir():
            return []
        if self._preprocessed_error is not None:
            return [
                RuntimeBlocker(
                    code="weights.tt_manifest_invalid",
                    phase=phase,
                    detail=f"Could not load TT-preprocessed manifest at {self.preprocessed_dir / TT_MANIFEST_FILENAME}: {self._preprocessed_error}",
                    next_action="Regenerate the full TT-preprocessed checkpoint from the real HF snapshot.",
                    evidence=(str(self.preprocessed_dir / TT_MANIFEST_FILENAME),),
                )
            ]
        if self._preprocessed_missing_artifacts:
            return [
                RuntimeBlocker(
                    code="weights.tt_artifacts_missing",
                    phase=phase,
                    detail=f"{len(self._preprocessed_missing_artifacts)} TT-preprocessed artifacts referenced by the manifest are missing.",
                    next_action="Regenerate the TT-preprocessed checkpoint; do not run from a partial materialization directory.",
                    evidence=self._preprocessed_missing_artifacts[:_MAX_EVIDENCE_ITEMS],
                )
            ]
        return []

    def _weight_index_blockers(self, phase: RuntimePhase) -> list[RuntimeBlocker]:
        index_path = self.snapshot_dir / MODEL_INDEX_FILENAME
        try:
            with index_path.open("r", encoding="utf-8") as handle:
                index = json.load(handle)
        except Exception as exc:  # noqa: BLE001 - surfaced as a structured blocker.
            return [
                RuntimeBlocker(
                    code="checkpoint.weight_index_invalid",
                    phase=phase,
                    detail=f"Could not parse {MODEL_INDEX_FILENAME}: {exc}",
                    next_action="Replace the local index with the real model.safetensors.index.json from the HF snapshot.",
                    evidence=(str(index_path),),
                )
            ]
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return [
                RuntimeBlocker(
                    code="checkpoint.weight_index_invalid",
                    phase=phase,
                    detail=f"{MODEL_INDEX_FILENAME} does not contain a non-empty weight_map.",
                    next_action="Fetch the full real checkpoint index before constructing TTNN weights.",
                    evidence=(str(index_path),),
                )
            ]

        shard_names = tuple(sorted({str(name) for name in weight_map.values()}))
        missing = tuple(name for name in shard_names if not (self.snapshot_dir / name).is_file())
        blockers: list[RuntimeBlocker] = []
        if missing:
            blockers.append(
                RuntimeBlocker(
                    code="checkpoint.weight_shards_missing",
                    phase=phase,
                    detail=f"{len(missing)} safetensors shards referenced by the index are missing.",
                    next_action="Resume the HF snapshot download until every indexed safetensors shard is present.",
                    evidence=missing[:_MAX_EVIDENCE_ITEMS],
                )
            )
        pointer_shards = tuple(name for name in shard_names if _looks_like_lfs_pointer(self.snapshot_dir / name))
        if pointer_shards:
            blockers.append(
                RuntimeBlocker(
                    code="checkpoint.weight_shards_pointer",
                    phase=phase,
                    detail=f"{len(pointer_shards)} safetensors shards are Git LFS pointers, not real weights.",
                    next_action="Run git-lfs/HF snapshot download so the branch has real checkpoint bytes.",
                    evidence=pointer_shards[:_MAX_EVIDENCE_ITEMS],
                )
            )
        return blockers

    def _implementation_blockers(self, phase: RuntimePhase) -> list[RuntimeBlocker]:
        weight_blocker = self._weight_materialization_blocker(phase)
        blockers = [
            RuntimeBlocker(
                code="runtime.full_model_device_path",
                phase=phase,
                detail=(
                    "The current branch still exposes scaffold/smoke slices; no integrated all-layer Blackhole "
                    "runtime owns prefill, decode, cache, logits, and token selection end to end."
                ),
                next_action="Replace the staged slices with one all-layer TtDeepSeekV4FlashRuntime execution graph.",
                evidence=(
                    "models/demos/deepseek_v4_flash/ttnn_model.py",
                    "models/demos/deepseek_v4_flash/real_traceable_decode_smoke.py",
                ),
            ),
            weight_blocker,
        ]
        if phase in ("prefill", "generate"):
            blockers.extend(
                [
                    RuntimeBlocker(
                        code="prefill.device_only_attention",
                        phase=phase,
                        detail=(
                            "Prefill compressor, indexer top-k, and sparse attention still have host fallback "
                            "boundaries in the current TTNN modules."
                        ),
                        next_action="Move compressor pooling, indexer scoring/top-k, and sparse attention reduction into TTNN/device kernels.",
                        evidence=(
                            "models/demos/deepseek_v4_flash/ttnn_prefill_compressor.py",
                            "models/demos/deepseek_v4_flash/ttnn_prefill_indexer.py",
                            "models/demos/deepseek_v4_flash/ttnn_sparse_attention.py",
                        ),
                    ),
                    RuntimeBlocker(
                        code="prefill.device_resident_cache_seed",
                        phase=phase,
                        detail="Prefill does not yet seed the model-owned paged decode cache entirely on device.",
                        next_action="Allocate paged K/V, sparse/indexer, and compressor state in device memory during prefill.",
                        evidence=("models/demos/deepseek_v4_flash/ttnn_prefill_attention_block.py",),
                    ),
                ]
            )
        if phase in ("decode", "generate"):
            blockers.extend(
                [
                    RuntimeBlocker(
                        code="decode.paged_sparse_attention",
                        phase=phase,
                        detail=(
                            "Decode has paged SDPA probes, but the integrated DeepSeek sparse/indexer selected-row "
                            "attention path is not implemented as the mandatory decode path."
                        ),
                        next_action="Make paged selected-row sparse attention the only decode attention path and wire it to real cache state.",
                        evidence=("models/demos/deepseek_v4_flash/real_traceable_decode_smoke.py",),
                    ),
                    RuntimeBlocker(
                        code="decode.device_resident_cache",
                        phase=phase,
                        detail="Current decode cache ownership is host/scaffold-oriented rather than model-owned paged device state.",
                        next_action="Carry K/V, compressed-KV, indexer, RoPE/current-position, and page-table state as TTNN tensors.",
                        evidence=("models/demos/deepseek_v4_flash/ttnn_decode_cache.py",),
                    ),
                    RuntimeBlocker(
                        code="decode.traced_moe_routing_dispatch",
                        phase=phase,
                        detail="MoE router top-k, expert dispatch, and combine still cross host boundaries.",
                        next_action="Implement traced/device router selection, dispatch metadata, expert execution, and combine on the 2x4 mesh.",
                        evidence=(
                            "models/demos/deepseek_v4_flash/ttnn_router.py",
                            "models/demos/deepseek_v4_flash/ttnn_moe_block.py",
                            "models/demos/deepseek_v4_flash/ttnn_expert_group.py",
                        ),
                    ),
                    RuntimeBlocker(
                        code="decode.traced_token_selection",
                        phase=phase,
                        detail="Autoregressive token selection/feed-forward is not yet in the traced device path.",
                        next_action="Keep final logits, argmax/top-k token selection, and next-token feed as trace-safe device operations.",
                    ),
                ]
            )
        return blockers

    def _hardware_blockers(
        self,
        phase: RuntimePhase,
        *,
        timeout_s: float,
        visible_devices: str,
    ) -> tuple[list[RuntimeBlocker], HardwareMeshProbeResult]:
        probe = _run_ttnn_mesh_open_probe(
            mesh_shape=self.mesh_shape,
            visible_devices=visible_devices,
            timeout_s=timeout_s,
        )
        if probe.opened:
            return [], probe
        detail = (
            f"Could not open required {probe.target_topology} for the integrated runtime"
            f" (available_devices={probe.available_devices}, visible_devices={probe.visible_devices})."
        )
        if probe.error:
            detail += f" TTNN/UMD error: {probe.error}"
        return [
            RuntimeBlocker(
                code="hardware.mesh_open_failed",
                phase=phase,
                detail=detail,
                next_action=(
                    "Fix the Blackhole Loudbox fabric/topology so the 2x4 mesh opens without constraining "
                    "TT_VISIBLE_DEVICES; single-device visibility can validate primitives but is not sufficient "
                    "for the accepted batch-1 runtime."
                ),
                evidence=probe.evidence[:_MAX_EVIDENCE_ITEMS],
            )
        ], probe

    def _weight_materialization_blocker(self, phase: RuntimePhase) -> RuntimeBlocker:
        if not self._has_complete_tt_preprocessed_weights:
            return RuntimeBlocker(
                code="runtime.full_weight_materialization",
                phase=phase,
                detail=(
                    "The primary runtime does not have a complete TT-preprocessed checkpoint directory available; "
                    f"expected {self.preprocessed_dir / TT_MANIFEST_FILENAME}."
                ),
                next_action="Run the full HF-to-TT checkpoint conversion and pass the resulting directory to the runtime.",
                evidence=("models/demos/deepseek_v4_flash/converter.py",),
            )
        return RuntimeBlocker(
            code="runtime.device_weight_ownership",
            phase=phase,
            detail=(
                "The full TT-preprocessed checkpoint exists, but the primary runtime still does not instantiate "
                "all layer/expert weights as model-owned device tensors on the 2x4 Blackhole mesh."
            ),
            next_action="Build the runtime weight owner that loads every preprocessed artifact into the execution layout.",
            evidence=(
                str(self.preprocessed_dir / TT_MANIFEST_FILENAME),
                "models/demos/deepseek_v4_flash/ttnn_model.py",
            ),
        )

    @property
    def _has_complete_tt_preprocessed_weights(self) -> bool:
        return (
            self._preprocessed_manifest is not None
            and self._preprocessed_error is None
            and not self._preprocessed_missing_artifacts
        )

    def _estimated_max_seq_len_if_device_path_ready(self) -> int | None:
        if self._config is None or self._weight_inventory is None:
            return None
        return estimate_max_seq_len_supported(
            self._config,
            self._weight_inventory,
            device_dram_bytes=_device_dram_bytes_from_env(),
            cache_dtype_bytes=2,
            safety_margin_bytes=1 << 30,
        )

    def _normalize_input_ids(self, input_ids: Sequence[int] | torch.Tensor, *, label: str) -> torch.Tensor:
        if isinstance(input_ids, torch.Tensor):
            ids = input_ids.detach().cpu().to(torch.long)
        else:
            ids = torch.tensor([list(input_ids)], dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if tuple(ids.shape[:1]) != (1,):
            raise ValueError(f"{label} must have batch size 1, got shape {tuple(ids.shape)}")
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError(f"{label} must have shape [1, tokens] with tokens > 0, got {tuple(ids.shape)}")
        if torch.any(ids < 0):
            raise ValueError(f"{label} must contain non-negative token IDs")
        if self._config is not None and torch.any(ids >= int(self._config.vocab_size)):
            raise ValueError(f"{label} contains token IDs outside vocab_size={int(self._config.vocab_size)}")
        return ids.contiguous()

    def _normalize_decode_token(self, token_id: int | torch.Tensor) -> int:
        if isinstance(token_id, torch.Tensor):
            token = int(token_id.detach().cpu().reshape(-1)[0].item())
        else:
            token = int(token_id)
        if token < 0:
            raise ValueError(f"decode token must be non-negative, got {token}")
        if self._config is not None and token >= int(self._config.vocab_size):
            raise ValueError(f"decode token {token} is outside vocab_size={int(self._config.vocab_size)}")
        return token

    @staticmethod
    def _raise_if_blocked(report: RuntimeBlockerReport) -> None:
        if report.blocked:
            raise DeepSeekRuntimeBlocked(report)


def default_guarded_symbols() -> tuple[GuardedSymbol, ...]:
    return (
        GuardedSymbol("ttnn", "to_torch", "ttnn.to_torch"),
        GuardedSymbol("models.demos.deepseek_v4_flash.cpu_reference", "sparse_attention", "cpu_reference.sparse_attention"),
        GuardedSymbol("models.demos.deepseek_v4_flash.cpu_reference", "indexer_topk", "cpu_reference.indexer_topk"),
        GuardedSymbol("models.demos.deepseek_v4_flash.cpu_reference", "compressor_prefill", "cpu_reference.compressor_prefill"),
        GuardedSymbol("models.demos.deepseek_v4_flash.cpu_reference", "v4_router", "cpu_reference.v4_router"),
        GuardedSymbol(
            "models.demos.deepseek_v4_flash.cpu_reference",
            "combine_routed_experts",
            "cpu_reference.combine_routed_experts",
        ),
        GuardedSymbol("models.demos.deepseek_v4_flash.ttnn_sparse_attention", "sparse_attention", "sparse_attention host import"),
        GuardedSymbol("models.demos.deepseek_v4_flash.ttnn_prefill_indexer", "indexer_topk", "indexer_topk host import"),
    )


def _resolve_guard_target(symbol: GuardedSymbol) -> tuple[object, str] | None:
    try:
        target = importlib.import_module(symbol.module_name)
    except Exception:  # noqa: BLE001 - optional TTNN modules may not import in CPU-only tooling.
        return None
    path_parts = symbol.attr_path.split(".")
    for part in path_parts[:-1]:
        if not hasattr(target, part):
            return None
        target = getattr(target, part)
    attr = path_parts[-1]
    if not hasattr(target, attr):
        return None
    return target, attr


def _blocked_host_boundary(label: str):
    def blocked(*args, **kwargs):
        raise HostBoundaryViolation(f"Host fallback/readback boundary is forbidden in model execution: {label}")

    return blocked


def _resolve_preprocessed_dir(snapshot_dir: Path, preprocessed_dir: str | Path | None) -> Path:
    if preprocessed_dir is not None:
        return Path(preprocessed_dir).expanduser().resolve()
    env_dir = os.environ.get("DSV4_FLASH_TT_PREPROCESSED_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    snapshot_name = snapshot_dir.name
    if snapshot_name.endswith("_hf"):
        materialized_name = snapshot_name.removesuffix("_hf") + "_tt_preprocessed"
    else:
        materialized_name = snapshot_name + "_tt_preprocessed"
    return snapshot_dir.with_name(materialized_name).resolve()


def _device_dram_bytes_from_env() -> int:
    value = os.environ.get("DSV4_FLASH_DEVICE_DRAM_GIB")
    if value is None:
        return 32 * 1024**3
    try:
        gib = float(value)
    except ValueError as exc:
        raise ValueError(f"DSV4_FLASH_DEVICE_DRAM_GIB must be numeric, got {value!r}") from exc
    if gib <= 0:
        raise ValueError(f"DSV4_FLASH_DEVICE_DRAM_GIB must be positive, got {value!r}")
    return int(gib * 1024**3)


def _default_mesh_graph_desc_path(mesh_shape: tuple[int, int]) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    descriptor_name = {
        (1, 1): "p150_mesh_graph_descriptor.textproto",
        (1, 2): "p150_x2_mesh_graph_descriptor.textproto",
        (2, 2): "p150_x4_mesh_graph_descriptor.textproto",
        (2, 4): "p150_x8_mesh_graph_descriptor.textproto",
    }.get(tuple(mesh_shape))
    if descriptor_name is None:
        descriptor_name = "p150_x8_mesh_graph_descriptor.textproto"
    return repo_root / "tt_metal" / "fabric" / "mesh_graph_descriptors" / descriptor_name


def _run_ttnn_mesh_open_probe(
    *,
    mesh_shape: tuple[int, int],
    visible_devices: str,
    timeout_s: float,
) -> HardwareMeshProbeResult:
    mesh_graph_desc_path = _default_mesh_graph_desc_path(mesh_shape)
    env = dict(os.environ)
    env["TT_VISIBLE_DEVICES"] = visible_devices
    env["TT_MESH_GRAPH_DESC_PATH"] = str(mesh_graph_desc_path)
    env.setdefault("TT_METAL_HOME", str(Path(__file__).resolve().parents[3]))
    env.setdefault("TT_METAL_RUNTIME_ROOT", str(Path(__file__).resolve().parents[3]))

    probe_code = f"""
import json
import traceback

result = {{
    "available_devices": None,
    "opened": False,
    "error": None,
    "error_type": None,
}}
try:
    import ttnn

    result["available_devices"] = int(ttnn.GetNumAvailableDevices())
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape({int(mesh_shape[0])}, {int(mesh_shape[1])}))
    result["opened"] = True
    result["device_repr"] = repr(device)
    ttnn.close_mesh_device(device)
except Exception as exc:  # noqa: BLE001 - returned as structured preflight evidence.
    result["error_type"] = type(exc).__name__
    result["error"] = str(exc).split("\\n")[0]
    result["traceback_tail"] = traceback.format_exc().splitlines()[-8:]
print("__DSV4_FLASH_HARDWARE_PREFLIGHT__" + json.dumps(result, sort_keys=True))
"""

    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe_code],
            env=env,
            cwd=str(Path(__file__).resolve().parents[3]),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = "\n".join(part for part in (exc.stdout, exc.stderr) if part)
        return HardwareMeshProbeResult(
            target_topology=_format_topology(mesh_shape),
            visible_devices=visible_devices,
            mesh_graph_desc_path=str(mesh_graph_desc_path),
            opened=False,
            available_devices=None,
            returncode=None,
            timed_out=True,
            error=f"TTNN mesh-open probe timed out after {timeout_s:.1f}s",
            evidence=_tail_lines(output),
        )

    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    payload: dict[str, Any] | None = None
    for line in completed.stdout.splitlines():
        if line.startswith("__DSV4_FLASH_HARDWARE_PREFLIGHT__"):
            try:
                payload = json.loads(line.removeprefix("__DSV4_FLASH_HARDWARE_PREFLIGHT__"))
            except json.JSONDecodeError:
                payload = None

    available_devices = None
    opened = False
    error = None
    if payload is not None:
        available_devices = payload.get("available_devices")
        if available_devices is not None:
            available_devices = int(available_devices)
        opened = bool(payload.get("opened"))
        error = payload.get("error")
    elif completed.returncode != 0:
        error = f"TTNN mesh-open probe exited with returncode={completed.returncode}"

    return HardwareMeshProbeResult(
        target_topology=_format_topology(mesh_shape),
        visible_devices=visible_devices,
        mesh_graph_desc_path=str(mesh_graph_desc_path),
        opened=opened and completed.returncode == 0,
        available_devices=available_devices,
        returncode=int(completed.returncode),
        timed_out=False,
        error=None if opened and completed.returncode == 0 else error,
        evidence=_tail_lines(output),
    )


def _format_topology(mesh_shape: tuple[int, int]) -> str:
    rows, cols = (int(mesh_shape[0]), int(mesh_shape[1]))
    return f"Blackhole Loudbox {rows}x{cols} ({rows * cols} devices)"


def _tail_lines(output: str, *, limit: int = 16) -> tuple[str, ...]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return tuple(lines[-limit:])


def _missing_preprocessed_artifacts(preprocessed_dir: Path, manifest: Mapping[str, Any]) -> list[str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return [TT_MANIFEST_FILENAME]
    filenames: list[str] = []
    copied_files = artifacts.get("copied_files", [])
    if isinstance(copied_files, list):
        filenames.extend(str(filename) for filename in copied_files)
    for field in ("non_expert_safetensors", "expert_safetensors"):
        values = artifacts.get(field, [])
        if isinstance(values, list):
            filenames.extend(str(filename) for filename in values)
    metadata = artifacts.get("metadata_safetensors")
    if isinstance(metadata, str):
        filenames.append(metadata)
    return [filename for filename in filenames if not (preprocessed_dir / filename).is_file()]


def _looks_like_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        prefix = handle.read(64)
    return prefix.startswith(b"version https://git-lfs.github.com/spec/")


def runtime_summary_from_exception(exc: DeepSeekRuntimeBlocked) -> dict[str, Any]:
    return exc.report.to_mapping()


def timed_call(fn, *args, **kwargs) -> tuple[Any, float]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start
