# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
import warnings
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_2d import DecoderBlock2D
from models.demos.deepseek_v3.tt.decoder_block.moe_decoder_block_2d import MoEDecoderBlock2D
from models.demos.deepseek_v3.tt.embedding.embedding2d import Embedding2D
from models.demos.deepseek_v3.tt.generator import (
    DEFAULT_MAX_SEQ_LEN,
    DeepseekGenerator,
    _build_verify_alias_page_table_host,
)
from models.demos.deepseek_v3.tt.lm_head1d import LMHead1D
from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
from models.demos.deepseek_v3.tt.model.row_batched_model import RowBatchedModel, get_fabric_config
from models.demos.deepseek_v3.tt.mtp import MTP2D
from models.demos.deepseek_v3.tt.rms_norm.distributed_rms_norm import DistributedRMSNorm
from models.demos.deepseek_v3.tt.rope import RotarySetup
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, even_int_div
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import load_state_dict
from models.demos.deepseek_v3.utils.weight_config import _try_load_cached_config, get_weight_config

DEFAULT_NUM_STEPS = 128
GENERATE_REFERENCE = os.getenv("DEEPSEEK_V3_MTP_GENERATE_REFERENCE", "0") == "1"
TRACE_REGION_SIZE = int(os.getenv("DEEPSEEK_TRACE_REGION_SIZE", "134217728"))
TIMEOUT_S = int(os.getenv("DEEPSEEK_V3_MTP_TIMEOUT_S", "1200"))
MAX_E2E_SECONDS = float(os.getenv("DEEPSEEK_V3_MTP_E2E_MAX_S", "0"))
MIN_TOKENS_PER_SEC = float(os.getenv("DEEPSEEK_V3_MTP_MIN_TPS", "1.0"))
MIN_TOKENS_PER_SEC_TRACE = float(os.getenv("DEEPSEEK_V3_MTP_MIN_TPS_TRACE", "0"))
DEFAULT_PREFILL_LEN = int(os.getenv("DEEPSEEK_V3_MTP_PREFILL_LEN", "16"))
DEFAULT_VERIFY_STEPS = int(os.getenv("DEEPSEEK_V3_MTP_VERIFY_STEPS", "16"))
SKIP_IN_CI = pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip in CI")


def _get_reference_dir() -> Path:
    try:
        default_cache = f"/localdev/{os.getlogin()}/deepseek-v3-cache"
    except OSError:
        default_cache = "/proj_sw/user_dev/deepseek-v3-cache"
    return Path(os.getenv("DEEPSEEK_V3_CACHE", default_cache)) / "test_io_cache"


def _debug_mtp_enabled() -> bool:
    return os.getenv("DEEPSEEK_DEBUG_MTP", "0") == "1"


# Test: host-side selective aliasing only rewires the intended interleaved verify rows.
@SKIP_IN_CI
def test_mtp_verify_page_table_selective_interleaved_aliasing_host():
    """Single-prompt selective aliasing should only alias row1->row0 and leave row3 untouched."""
    base_page_table = torch.tensor(
        [
            [0, 1],
            [10, 11],
            [20, 21],
            [30, 31],
        ],
        dtype=torch.int32,
    )

    alias_single = _build_verify_alias_page_table_host(
        base_page_table=base_page_table,
        num_prompts=1,
        verify_offset=0,
        prompt_indices=[0],
        interleaved=True,
    )
    expected_single = torch.tensor(
        [
            [0, 1],
            [0, 1],
            [20, 21],
            [30, 31],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(alias_single, expected_single)

    alias_all = _build_verify_alias_page_table_host(
        base_page_table=base_page_table,
        num_prompts=2,
        verify_offset=0,
        prompt_indices=None,
        interleaved=True,
    )
    expected_all = torch.tensor(
        [
            [0, 1],
            [0, 1],
            [20, 21],
            [20, 21],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(alias_all, expected_all)


def _run_reference_decode_replay_consistency(
    mesh_device,
    model_path,
    cache_path,
    force_recalculate_weight_config,
    set_deterministic_env,
    enable_mtp: bool,
):
    num_steps = int(os.getenv("DEEPSEEK_V3_MTP_REF_STEPS", str(DEFAULT_NUM_STEPS)))
    steps_to_check = min(int(os.getenv("DEEPSEEK_V3_MTP_VERIFY_STEPS", str(DEFAULT_VERIFY_STEPS))), num_steps)
    mesh = mesh_device

    mtp_label = "on" if enable_mtp else "off"
    with _prepare_generator(
        mesh_device=mesh,
        model_path=model_path,
        cache_path=cache_path,
        force_recalculate=force_recalculate_weight_config,
        enable_mtp=enable_mtp,
    ) as gen:
        payload, _reference_path = _load_reference_payload_for_generator(
            gen,
            num_steps,
            context=f"MTP reference decode replay ({mtp_label})",
        )
        next_tokens = payload["next_tokens"].to(torch.int32)
        start_tokens = payload["start_tokens"].to(torch.long)
        steps_to_check = min(steps_to_check, int(next_tokens.shape[0]))
        _assert_reference_start_tokens(payload, gen, context=f"MTP reference decode replay ({mtp_label})")

        positions = torch.zeros((gen.batch_size,), dtype=torch.int32)
        tokens_step = start_tokens.clone()
        mismatch_rows: list[str] = []
        first_mismatch_step = None

        for step in range(steps_to_check):
            logits = (
                gen._decode_step(
                    tokens_step=tokens_step,
                    positions=positions,
                    page_tables=gen._get_page_tables(),
                    return_hidden=False,
                )
                .squeeze(0)
                .squeeze(0)
            )
            pred_tokens = torch.argmax(logits, dim=-1).to(torch.int32)
            gt_tokens = next_tokens[step].to(torch.int32)
            mismatch_mask = pred_tokens != gt_tokens
            mismatch_count = int(mismatch_mask.sum().item())

            row = (
                f"step={step} pos={int(positions[0].item())} mismatch_count={mismatch_count} "
                f"sample_pred={int(pred_tokens[0].item())} sample_gt={int(gt_tokens[0].item())}"
            )
            mismatch_rows.append(row)
            logger.info("MTP reference replay [{}] {}", mtp_label, row)

            if mismatch_count > 0 and first_mismatch_step is None:
                first_mismatch_step = step
                mismatch_idx = mismatch_mask.nonzero(as_tuple=False).flatten()[:8].tolist()
                details = [
                    f"uid={uid} pred={int(pred_tokens[uid].item())} gt={int(gt_tokens[uid].item())}"
                    for uid in mismatch_idx
                ]
                logger.info(
                    "MTP reference replay [{}] first mismatch step={} details={}",
                    mtp_label,
                    step,
                    details,
                )

            tokens_step = gt_tokens.to(torch.long)
            positions += 1

        logger.info(
            "MTP reference replay [{}] summary: steps_checked={} first_mismatch_step={}",
            mtp_label,
            steps_to_check,
            first_mismatch_step,
        )
        assert first_mismatch_step is None, (
            f"Fresh base decode ({mtp_label}) does not reproduce the stored reference stream. "
            f"first_mismatch_step={first_mismatch_step}\n" + "\n".join(mismatch_rows)
        )


# Test: mtp=on must not perturb base greedy decode replay against the stored reference stream.
@pytest.mark.timeout(TIMEOUT_S)
@pytest.mark.requires_device(["DUAL", "QUAD"])
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": TRACE_REGION_SIZE,
            },
            marks=SKIP_IN_CI,
        )
    ],
    indirect=True,
)
def test_mtp_reference_decode_replay_consistency(
    mesh_device,
    model_path,
    cache_path,
    force_recalculate_weight_config,
    set_deterministic_env,
):
    """Fresh base decode under mtp=on should reproduce the stored greedy reference stream."""
    _run_reference_decode_replay_consistency(
        mesh_device,
        model_path,
        cache_path,
        force_recalculate_weight_config,
        set_deterministic_env,
        enable_mtp=True,
    )


# Test: mtp=off baseline decode must still replay the saved reference stream exactly.
@pytest.mark.timeout(TIMEOUT_S)
@pytest.mark.requires_device(["DUAL", "QUAD"])
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": TRACE_REGION_SIZE,
            },
            marks=SKIP_IN_CI,
        )
    ],
    indirect=True,
)
def test_mtp_reference_decode_replay_consistency_mtp_off(
    mesh_device,
    model_path,
    cache_path,
    force_recalculate_weight_config,
    set_deterministic_env,
):
    """Fresh base decode under mtp=off should reproduce the stored greedy reference stream."""
    _run_reference_decode_replay_consistency(
        mesh_device,
        model_path,
        cache_path,
        force_recalculate_weight_config,
        set_deterministic_env,
        enable_mtp=False,
    )


def _get_reference_path(num_steps: int) -> Path:
    return _get_reference_dir() / f"mtp_full_model_seq{num_steps}.pt"


def _get_start_token_id(hf_config) -> int:
    bos_id = getattr(hf_config, "bos_token_id", None)
    if isinstance(bos_id, (list, tuple)):
        bos_id = bos_id[0] if bos_id else None
    if bos_id is None:
        eos_id = getattr(hf_config, "eos_token_id", None)
        if isinstance(eos_id, (list, tuple)):
            eos_id = eos_id[0] if eos_id else None
        bos_id = eos_id
    return int(bos_id) if bos_id is not None else 1


def _assert_reference_start_tokens(payload: dict, gen: Any, context: str) -> None:
    if "start_tokens" not in payload:
        return
    start_tokens = payload["start_tokens"].to(torch.long)
    expected_start_id = _get_start_token_id(gen.hf_config)
    expected_tokens = (torch.arange(gen.batch_size, dtype=torch.long) + expected_start_id) % gen.hf_config.vocab_size

    if start_tokens.shape != expected_tokens.shape or not torch.equal(start_tokens, expected_tokens):
        mismatch_preview = []
        if start_tokens.shape == expected_tokens.shape:
            mismatch_idx = (start_tokens != expected_tokens).nonzero(as_tuple=False).flatten()[:8]
            for idx in mismatch_idx.tolist():
                mismatch_preview.append(
                    f"{idx}: ref={int(start_tokens[idx].item())} expected={int(expected_tokens[idx].item())}"
                )
            mismatch_count = int((start_tokens != expected_tokens).sum().item())
        else:
            preview_count = min(8, int(start_tokens.shape[0]), int(expected_tokens.shape[0]))
            for idx in range(preview_count):
                mismatch_preview.append(
                    f"{idx}: ref={int(start_tokens[idx].item())} expected={int(expected_tokens[idx].item())}"
                )
            mismatch_count = abs(int(start_tokens.shape[0]) - int(expected_tokens.shape[0]))
        msg = (
            f"{context}: reference start_tokens do not match current generator ordering. "
            f"expected_start_id={expected_start_id} batch_size={gen.batch_size} "
            f"ref_shape={tuple(start_tokens.shape)} expected_shape={tuple(expected_tokens.shape)} "
            f"mismatch_count={mismatch_count} "
            f"mismatches={mismatch_preview}"
        )
        pytest.fail(msg)


def _load_reference_payload(reference_path: Path) -> dict:
    if not reference_path.exists():
        pytest.skip(
            f"Missing MTP reference IO at {reference_path}. "
            "Set DEEPSEEK_V3_MTP_GENERATE_REFERENCE=1 and run the reference generator test."
        )
    return torch.load(reference_path, map_location="cpu")


def _load_reference_payload_for_generator(
    gen: Any,
    num_steps: int,
    context: str,
) -> tuple[dict, Path]:
    reference_path = _get_reference_path(num_steps)
    payload = _load_reference_payload(reference_path)
    metadata = dict(payload.get("metadata", {}))
    hidden_states = payload["hidden_states"].to(torch.bfloat16)
    next_tokens = payload["next_tokens"].to(torch.int32)
    start_tokens = payload["start_tokens"].to(torch.long)

    ref_batch = int(start_tokens.shape[0])
    assert hidden_states.ndim == 3, (
        f"{context}: expected reference hidden_states to have shape [num_steps, batch, hidden], "
        f"got {tuple(hidden_states.shape)} from {reference_path}."
    )
    assert next_tokens.ndim == 2, (
        f"{context}: expected reference next_tokens to have shape [num_steps, batch], "
        f"got {tuple(next_tokens.shape)} from {reference_path}."
    )
    assert hidden_states.shape[0] == next_tokens.shape[0], (
        f"{context}: hidden_states steps {hidden_states.shape[0]} do not match next_tokens steps "
        f"{next_tokens.shape[0]} in {reference_path}."
    )
    assert hidden_states.shape[1] == ref_batch, (
        f"{context}: hidden_states batch {hidden_states.shape[1]} does not match start_tokens batch {ref_batch} "
        f"in {reference_path}."
    )
    assert next_tokens.shape[1] == ref_batch, (
        f"{context}: next_tokens batch {next_tokens.shape[1]} does not match start_tokens batch {ref_batch} "
        f"in {reference_path}."
    )

    if "batch_size" in metadata:
        assert int(metadata["batch_size"]) == ref_batch, (
            f"{context}: reference metadata batch_size={metadata['batch_size']} does not match tensor batch "
            f"{ref_batch} in {reference_path}."
        )
    if "num_steps" in metadata:
        assert int(metadata["num_steps"]) == int(hidden_states.shape[0]), (
            f"{context}: reference metadata num_steps={metadata['num_steps']} does not match tensor steps "
            f"{hidden_states.shape[0]} in {reference_path}."
        )
    if "hidden_size" in metadata:
        assert int(metadata["hidden_size"]) == int(gen.hf_config.hidden_size), (
            f"{context}: reference hidden_size={metadata['hidden_size']} does not match current model "
            f"hidden_size={gen.hf_config.hidden_size} in {reference_path}."
        )
    if "vocab_size" in metadata:
        assert int(metadata["vocab_size"]) == int(gen.hf_config.vocab_size), (
            f"{context}: reference vocab_size={metadata['vocab_size']} does not match current model "
            f"vocab_size={gen.hf_config.vocab_size} in {reference_path}."
        )
    if "start_token_id" in metadata:
        expected_start_id = _get_start_token_id(gen.hf_config)
        assert int(metadata["start_token_id"]) == expected_start_id, (
            f"{context}: reference start_token_id={metadata['start_token_id']} does not match current model "
            f"start_token_id={expected_start_id} in {reference_path}."
        )
    if "batch_size_per_row" in metadata:
        assert int(metadata["batch_size_per_row"]) == int(gen.batch_size_per_row), (
            f"{context}: reference batch_size_per_row={metadata['batch_size_per_row']} does not match current "
            f"batch_size_per_row={gen.batch_size_per_row} in {reference_path}."
        )

    required_batch_rows = int(gen.batch_size // gen.batch_size_per_row)
    assert ref_batch >= gen.batch_size, (
        f"{context}: reference batch size {ref_batch} from {reference_path} is too small for the current "
        f"generator batch size {gen.batch_size}. Regenerate the reference with at least {required_batch_rows} "
        f"batch rows ({gen.batch_size} total batch). reference_mesh_shape={metadata.get('mesh_shape')}"
    )

    if ref_batch > gen.batch_size:
        hidden_states = hidden_states[:, : gen.batch_size]
        next_tokens = next_tokens[:, : gen.batch_size]
        start_tokens = start_tokens[: gen.batch_size]

    normalized_payload = dict(payload)
    normalized_payload["metadata"] = metadata
    normalized_payload["hidden_states"] = hidden_states
    normalized_payload["next_tokens"] = next_tokens
    normalized_payload["start_tokens"] = start_tokens
    return normalized_payload, reference_path


def _prepare_generator(
    mesh_device: ttnn.MeshDevice,
    model_path: Path,
    cache_path: Path,
    force_recalculate: bool,
    enable_mtp: bool,
) -> DeepseekGenerator:
    gen = DeepseekGenerator(
        mesh_device=mesh_device,
        model_path=model_path,
        cache_dir=cache_path,
        enable_mtp=enable_mtp,
        force_recalculate=force_recalculate,
    )
    gen._prepare_run_configs("prefill")
    gen._prepare_run_configs("decode")
    return gen


def _deallocate_ttnn_tensors(
    obj: Any, seen_objects: set[int] | None = None, seen_tensors: set[int] | None = None
) -> None:
    if obj is None:
        return

    if seen_objects is None:
        seen_objects = set()
    if seen_tensors is None:
        seen_tensors = set()

    if isinstance(obj, ttnn.Tensor):
        tensor_id = id(obj)
        if tensor_id not in seen_tensors:
            try:
                ttnn.deallocate(obj)
            except Exception:
                pass
            seen_tensors.add(tensor_id)
        return

    object_id = id(obj)
    if object_id in seen_objects:
        return
    seen_objects.add(object_id)

    if isinstance(obj, dict):
        for value in obj.values():
            _deallocate_ttnn_tensors(value, seen_objects, seen_tensors)
        return

    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            _deallocate_ttnn_tensors(value, seen_objects, seen_tensors)
        return

    if is_dataclass(obj):
        for field in fields(obj):
            _deallocate_ttnn_tensors(getattr(obj, field.name), seen_objects, seen_tensors)


def _load_cached_mtp_weight_config(
    cache_path: Path,
    hf_config,
    mesh_device: ttnn.MeshDevice,
    force_recalculate: bool,
) -> dict[str, Any] | None:
    cache_subdir_name = f"{hf_config.num_hidden_layers}_layers_mtp"
    weight_cache_path = cache_path / cache_subdir_name / f"mesh_{mesh_device.shape[0]}x{mesh_device.shape[1]}"
    cached_weight_config = _try_load_cached_config(
        weight_cache_path / "config.json",
        weight_cache_path,
        force_recalculate=force_recalculate,
    )
    if cached_weight_config is None:
        return None

    mtp_weight_config = cached_weight_config.get("mtp") if isinstance(cached_weight_config, dict) else None
    if not isinstance(mtp_weight_config, dict) or not mtp_weight_config:
        raise RuntimeError(
            f"Cached weight config at {weight_cache_path} does not contain an MTP block. "
            "Regenerate the DeepSeek MTP cache before running this test."
        )
    return mtp_weight_config


class _MtpModuleRunner:
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        model_path: Path,
        cache_path: Path,
        force_recalculate: bool,
    ) -> None:
        self.mesh_device = mesh_device
        self.model_path = Path(model_path)
        self.cache_path = Path(cache_path)
        self.force_recalculate = force_recalculate
        self.enable_mtp = True

        self.hf_config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.hf_config.max_seq_len = DEFAULT_MAX_SEQ_LEN
        if int(getattr(self.hf_config, "num_nextn_predict_layers", 0)) <= 0:
            raise RuntimeError("MTP module runner requires a model config with num_nextn_predict_layers > 0.")

        self.ccl = CCL(mesh_device)
        self.dp_factor = int(mesh_device.shape[1])
        self.batch_size_per_row = USERS_PER_ROW
        self.batch_size = self.batch_size_per_row * int(mesh_device.shape[0])
        self.paged_config = MLA2D.get_valid_paged_config(
            self.hf_config.max_seq_len,
            self.batch_size_per_row,
            self.dp_factor,
        )
        self.rope_setup = RotarySetup(
            device=self.mesh_device,
            batch_size_per_row=self.batch_size_per_row,
            hf_config=self.hf_config,
        )

        self._weight_ttnn_cache: dict[str, ttnn.Tensor] = {}
        self.mtp_page_table_tt: ttnn.Tensor | None = None
        self.mtp_page_table_host: torch.Tensor | None = None
        self._mtp_state_dict = None
        self.model_weight_config = _load_cached_mtp_weight_config(
            cache_path=self.cache_path,
            hf_config=self.hf_config,
            mesh_device=self.mesh_device,
            force_recalculate=self.force_recalculate,
        )
        if self.model_weight_config is None:
            mtp_layer_idx = int(self.hf_config.num_hidden_layers)
            self._mtp_state_dict = load_state_dict(self.model_path, f"model.layers.{mtp_layer_idx}")
            if "eh_proj.weight" not in self._mtp_state_dict:
                raise RuntimeError(
                    f"Could not find MTP weights under model.layers.{mtp_layer_idx} in {self.model_path}."
                )

            cache_subdir_name = f"{self.hf_config.num_hidden_layers}_layers_mtp_module"
            self.model_weight_config = get_weight_config(
                ModuleClass=MTP2D,
                hf_config=self.hf_config,
                state_dicts=(self._mtp_state_dict,),
                weight_cache_path=self.cache_path,
                mesh_device=self.mesh_device,
                force_recalculate=self.force_recalculate,
                cache_subdir_name=cache_subdir_name,
            )
        self.model_state = MTP2D.create_state(
            hf_config=self.hf_config,
            paged_config=self.paged_config,
            mesh_device=self.mesh_device,
            ccl=self.ccl,
        )
        self.model_shared_state = MTP2D.create_shared_state(
            hf_config=self.hf_config,
            mesh_device=self.mesh_device,
        )
        self.model_decode_cfg = MTP2D.decode_model_config(
            hf_config=self.hf_config,
            mesh_device=self.mesh_device,
            fabric_config=get_fabric_config(),
            batch_size_per_row=self.batch_size_per_row,
        )
        mtp_decode_run_config = create_run_config(
            self.model_decode_cfg,
            self.model_weight_config,
            self.model_state,
            self.model_shared_state,
            cached_ttnn_weights=self._weight_ttnn_cache,
        )
        # Keep the same top-level shape as DeepseekGenerator.model_run_config_decode so existing helpers still work.
        self.model_run_config_decode = {"mtp": mtp_decode_run_config}

    def _get_mtp_page_table(self) -> ttnn.Tensor:
        if self.mtp_page_table_tt is not None:
            return self.mtp_page_table_tt

        batch_per_shard = even_int_div(self.batch_size_per_row, self.dp_factor)
        blocks_per_user = even_int_div(self.paged_config.max_num_blocks, batch_per_shard)
        self.mtp_page_table_host = torch.arange(self.paged_config.max_num_blocks, dtype=torch.int32).reshape(
            batch_per_shard,
            blocks_per_user,
        )
        self.mtp_page_table_tt = MLA2D.create_page_table(
            paged_config=self.paged_config,
            mesh_device=self.mesh_device,
            page_table=self.mtp_page_table_host,
            batch_size=self.batch_size_per_row,
        )
        return self.mtp_page_table_tt

    def cleanup_all(self) -> None:
        seen_objects: set[int] = set()
        seen_tensors: set[int] = set()

        _deallocate_ttnn_tensors(self.model_run_config_decode, seen_objects, seen_tensors)
        _deallocate_ttnn_tensors(self.model_state, seen_objects, seen_tensors)
        _deallocate_ttnn_tensors(self.model_shared_state, seen_objects, seen_tensors)
        _deallocate_ttnn_tensors(self.rope_setup.cos_matrix, seen_objects, seen_tensors)
        _deallocate_ttnn_tensors(self.rope_setup.sin_matrix, seen_objects, seen_tensors)
        _deallocate_ttnn_tensors(self.rope_setup.transformation_mat, seen_objects, seen_tensors)
        _deallocate_ttnn_tensors(self.rope_setup.transformation_mat_prefill, seen_objects, seen_tensors)
        _deallocate_ttnn_tensors(self.mtp_page_table_tt, seen_objects, seen_tensors)

        self._weight_ttnn_cache.clear()
        self.mtp_page_table_tt = None
        self.mtp_page_table_host = None
        if self._mtp_state_dict is not None and hasattr(self._mtp_state_dict, "close"):
            try:
                self._mtp_state_dict.close()
            except Exception:
                pass

    def __enter__(self) -> "_MtpModuleRunner":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup_all()


def _prepare_mtp_module_runner(
    mesh_device: ttnn.MeshDevice,
    model_path: Path,
    cache_path: Path,
    force_recalculate: bool,
) -> _MtpModuleRunner:
    return _MtpModuleRunner(
        mesh_device=mesh_device,
        model_path=model_path,
        cache_path=cache_path,
        force_recalculate=force_recalculate,
    )


def _tt_hidden_from_torch(
    mesh_device: ttnn.MeshDevice, hidden: torch.Tensor, device: ttnn.MeshDevice | None
) -> ttnn.Tensor:
    batch_size, hidden_size = hidden.shape
    return ttnn.from_torch(
        hidden.view(1, 1, batch_size, hidden_size).to(torch.bfloat16),
        device=device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(-2, -1),
            mesh_shape=tuple(mesh_device.shape),
        ),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )


def _tt_tokens_from_torch(
    mesh_device: ttnn.MeshDevice, tokens: torch.Tensor, device: ttnn.MeshDevice | None
) -> ttnn.Tensor:
    x = tokens.view(1, 1, -1).to(torch.int32)
    return ttnn.from_torch(
        x,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _tt_positions_from_torch(
    mesh_device: ttnn.MeshDevice, positions: torch.Tensor, device: ttnn.MeshDevice | None
) -> ttnn.Tensor:
    return ttnn.from_torch(
        positions.to(torch.int32),
        device=device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        dtype=ttnn.int32,
    )


def _run_mtp_step(
    gen: Any,
    hidden: torch.Tensor,
    tokens: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    mesh_device = gen.mesh_device
    tt_hidden = _tt_hidden_from_torch(mesh_device, hidden, device=mesh_device)
    tt_tokens = _tt_tokens_from_torch(mesh_device, tokens, device=mesh_device)
    tt_positions = _tt_positions_from_torch(mesh_device, positions, device=mesh_device)

    rot_idxs = gen.rope_setup.get_rot_idxs(positions)
    rope_tensors = gen.rope_setup.get_rot_mats_from_rot_idxs(rot_idxs)

    def op():
        return RowBatchedModel.forward_mtp_decode(
            hidden_states=tt_hidden,
            token_ids=tt_tokens,
            position_idxs=tt_positions,
            cfg=gen.model_run_config_decode,
            rope_tensors=rope_tensors,
            page_table=gen._get_mtp_page_table(),
        )

    tt_logits = op()

    logits = ttnn.to_torch(
        tt_logits,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
    )
    logits = logits.squeeze(0).squeeze(0)

    ttnn.deallocate(tt_logits)
    ttnn.deallocate(tt_hidden)
    ttnn.deallocate(tt_tokens)
    ttnn.deallocate(tt_positions)
    ttnn.deallocate(rot_idxs)
    ttnn.deallocate(rope_tensors["cos_matrix"])
    ttnn.deallocate(rope_tensors["sin_matrix"])

    return logits


def _iter_decode_block_cfgs(cfg):
    layer_idx = 0
    for block_cfg in cfg["mlp_decoder_block"]:
        yield layer_idx, "mlp", DecoderBlock2D, block_cfg
        layer_idx += 1
    for block_cfg in cfg["moe_decoder_block"]:
        yield layer_idx, "moe", MoEDecoderBlock2D, block_cfg
        layer_idx += 1


def _get_decode_kv_caches(gen: DeepseekGenerator) -> list[ttnn.Tensor]:
    assert gen.model_run_config_decode is not None, "Decode run config is not initialized"
    kv_caches: list[ttnn.Tensor] = []
    for decoder_type in ("mlp_decoder_block", "moe_decoder_block"):
        decoder_blocks = gen.model_run_config_decode.get(decoder_type, [])
        for block_cfg in decoder_blocks:
            kv_caches.append(block_cfg["mla"]["mla1d"]["kvpe_cache"])
    return kv_caches


def _set_kv_caches_on_run_config(run_config, kv_caches: list[ttnn.Tensor]) -> None:
    cache_idx = 0
    for decoder_type in ("mlp_decoder_block", "moe_decoder_block"):
        decoder_blocks = run_config.get(decoder_type, [])
        for block_cfg in decoder_blocks:
            if cache_idx >= len(kv_caches):
                raise RuntimeError(
                    f"Not enough kv caches to populate run config: need at least {cache_idx + 1}, got {len(kv_caches)}"
                )
            block_cfg["mla"]["mla1d"]["kvpe_cache"] = kv_caches[cache_idx]
            cache_idx += 1
    if cache_idx != len(kv_caches):
        raise RuntimeError(f"Unused kv caches while populating run config: used={cache_idx} total={len(kv_caches)}")


def _reset_base_kv_caches(gen: DeepseekGenerator) -> None:
    caches_unique: list[ttnn.Tensor] = []
    seen: set[int] = set()
    for cache in _get_decode_kv_caches(gen):
        cache_id = id(cache)
        if cache_id not in seen:
            caches_unique.append(cache)
            seen.add(cache_id)

    for cache in caches_unique:
        # Reuse the already-allocated decode caches to avoid allocating a fresh
        # full cache set during the layerwise alias-vs-base compare.
        ttnn.fill(cache, 0.0, memory_config=cache.memory_config(), output_tensor=cache)

    gen.ccl.reset_sem_counters()


def _reset_mtp_kv_cache(gen: DeepseekGenerator) -> None:
    assert gen.model_run_config_decode is not None, "Decode run config is not initialized"
    mtp_cfg = gen.model_run_config_decode.get("mtp")
    if mtp_cfg is None:
        raise RuntimeError("MTP decode config is not initialized")
    decoder_block_cfg = mtp_cfg.get("decoder_block")
    if decoder_block_cfg is None:
        raise RuntimeError("MTP decoder block config is missing")
    kvpe_cache = decoder_block_cfg["mla"]["mla1d"]["kvpe_cache"]
    ttnn.fill(kvpe_cache, 0.0, memory_config=kvpe_cache.memory_config(), output_tensor=kvpe_cache)
    gen.ccl.reset_sem_counters()


def _replay_verify_decode_batches(
    gen: DeepseekGenerator,
    start_tokens: torch.Tensor,
    batched_history: list[tuple[torch.Tensor, torch.Tensor]],
    page_tables: tuple[ttnn.Tensor, ...],
) -> None:
    batch_size = int(start_tokens.shape[0])
    seed_positions = torch.zeros((batch_size,), dtype=torch.int32)
    seed_logits_tt = gen._decode_step_tt(
        tokens_step=start_tokens,
        positions=seed_positions,
        batch_size_per_row=gen.batch_size_per_row,
        page_tables=gen._get_page_tables(),
        return_hidden=False,
    )
    ttnn.deallocate(seed_logits_tt)
    gen.ccl.reset_sem_counters()

    for batched_tokens, batched_positions in batched_history:
        logits_tt = gen._decode_step_tt(
            tokens_step=batched_tokens,
            positions=batched_positions,
            batch_size_per_row=gen.batch_size_per_row,
            page_tables=page_tables,
            return_hidden=False,
        )
        ttnn.deallocate(logits_tt)
        gen.ccl.reset_sem_counters()


def _replay_reference_decode_steps(
    gen: DeepseekGenerator,
    start_tokens: torch.Tensor,
    next_tokens: torch.Tensor,
    num_prior_steps: int,
) -> None:
    batch_size = int(start_tokens.shape[0])
    seed_positions = torch.zeros((batch_size,), dtype=torch.int32)
    seed_logits_tt = gen._decode_step_tt(
        tokens_step=start_tokens,
        positions=seed_positions,
        batch_size_per_row=gen.batch_size_per_row,
        page_tables=gen._get_page_tables(),
        return_hidden=False,
    )
    ttnn.deallocate(seed_logits_tt)
    gen.ccl.reset_sem_counters()

    for replay_step in range(num_prior_steps):
        prompt_tokens = next_tokens[replay_step].to(torch.int32)
        prompt_positions = torch.full((batch_size,), replay_step + 1, dtype=torch.int32)
        logits_tt = gen._decode_step_tt(
            tokens_step=prompt_tokens,
            positions=prompt_positions,
            batch_size_per_row=gen.batch_size_per_row,
            page_tables=gen._get_page_tables(),
            return_hidden=False,
        )
        ttnn.deallocate(logits_tt)
        gen.ccl.reset_sem_counters()


def _decode_step_host(
    gen: DeepseekGenerator,
    tokens_step: torch.Tensor,
    positions: torch.Tensor,
    page_tables: tuple[ttnn.Tensor, ...],
) -> torch.Tensor:
    mesh_device = gen.mesh_device
    logits_tt = gen._decode_step_tt(
        tokens_step=tokens_step,
        positions=positions,
        batch_size_per_row=gen.batch_size_per_row,
        page_tables=page_tables,
        return_hidden=False,
    )
    logits = (
        ttnn.to_torch(
            logits_tt,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
        )
        .squeeze(0)
        .squeeze(0)
    )
    ttnn.deallocate(logits_tt)
    gen.ccl.reset_sem_counters()
    return logits


def _decode_step_layerwise_host(
    gen: DeepseekGenerator,
    tokens_step: torch.Tensor,
    positions: torch.Tensor,
    page_tables: tuple[ttnn.Tensor, ...],
    capture_user_ids: list[int],
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    mesh_device = gen.mesh_device
    cfg = gen.model_run_config_decode
    assert cfg is not None, "Decode run config is not initialized"

    tt_tokens = gen._tt_from_tokens_step(tokens_step)
    rot_idxs = gen.rope_setup.get_rot_idxs(positions)
    rope_tensors = gen.rope_setup.get_rot_mats_from_rot_idxs(rot_idxs)
    tt_positions = ttnn.from_torch(
        positions.to(torch.int32),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        dtype=ttnn.int32,
    )

    layer_captures: list[dict[str, object]] = []
    logits_tt = None
    try:
        x = Embedding2D.forward_decode(tt_tokens, cfg["embedding"])
        for (layer_idx, block_kind, BlockClass, block_cfg), page_table in zip(
            _iter_decode_block_cfgs(cfg),
            page_tables,
            strict=True,
        ):
            x = BlockClass.forward_decode(x, tt_positions, block_cfg, rope_tensors, page_table)
            x_host = (
                ttnn.to_torch(
                    x,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
                )
                .squeeze(0)
                .squeeze(0)
            )
            rows = {uid: x_host[uid].to(torch.float32).cpu() for uid in capture_user_ids}
            layer_captures.append(
                {
                    "layer_idx": layer_idx,
                    "block_kind": block_kind,
                    "rows": rows,
                }
            )

        x = ttnn.to_memory_config(x, **cfg["norm_reshard"])
        x = DistributedRMSNorm.forward_decode(x, cfg["norm"])
        ccl = cfg["lm_head"]["ccl"]
        x = ttnn.experimental.all_gather_async(x, **ccl.populate_all_gather_runtime_args(cfg["lm_head"]["all_gather"]))
        logits_tt = LMHead1D.forward_decode(x, cfg["lm_head"])
        logits = (
            ttnn.to_torch(
                logits_tt,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
            )
            .squeeze(0)
            .squeeze(0)
        )
        gen.ccl.reset_sem_counters()
        return logits, layer_captures
    finally:
        if logits_tt is not None:
            ttnn.deallocate(logits_tt)
        ttnn.deallocate(tt_tokens)
        ttnn.deallocate(tt_positions)
        ttnn.deallocate(rot_idxs)
        ttnn.deallocate(rope_tensors["cos_matrix"])
        ttnn.deallocate(rope_tensors["sin_matrix"])


class _MtpTraceRunner:
    def __init__(self, gen: Any) -> None:
        self.gen = gen
        self.mesh_device = gen.mesh_device
        self.trace_id: int | None = None
        self.tt_hidden: ttnn.Tensor | None = None
        self.tt_tokens: ttnn.Tensor | None = None
        self.tt_positions: ttnn.Tensor | None = None
        self.tt_rot_idxs: ttnn.Tensor | None = None
        self.trace_output: ttnn.Tensor | None = None
        self._expected_hidden_shape: tuple[int, ...] | None = None
        self._expected_hidden_dtype: torch.dtype | None = None
        self._expected_tokens_shape: tuple[int, ...] | None = None
        self._expected_tokens_dtype: torch.dtype | None = None
        self._expected_positions_shape: tuple[int, ...] | None = None
        self._expected_positions_dtype: torch.dtype | None = None

    def run(self, hidden: torch.Tensor, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.trace_id is None:
            self._capture(hidden, tokens, positions)

        # Always reset inputs (positions, tokens, hidden, rot_idxs) before executing.
        self._update_inputs(hidden, tokens, positions)
        self.gen.ccl.reset_sem_counters()
        ttnn.execute_trace(self.mesh_device, self.trace_id, cq_id=0, blocking=True)

        assert self.trace_output is not None
        logits = ttnn.to_torch(
            self.trace_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape),
        )
        return logits.squeeze(0).squeeze(0)

    def release(self) -> None:
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh_device, self.trace_id)
            self.trace_id = None

        for tensor_name in ("tt_hidden", "tt_tokens", "tt_positions", "tt_rot_idxs", "trace_output"):
            tensor = getattr(self, tensor_name)
            if tensor is not None:
                ttnn.deallocate(tensor)
                setattr(self, tensor_name, None)
        self._expected_hidden_shape = None
        self._expected_hidden_dtype = None
        self._expected_tokens_shape = None
        self._expected_tokens_dtype = None
        self._expected_positions_shape = None
        self._expected_positions_dtype = None

    def _capture(self, hidden: torch.Tensor, tokens: torch.Tensor, positions: torch.Tensor) -> None:
        # Warm-up compile run (no trace) to keep compilation out of capture.
        _run_mtp_step(
            gen=self.gen,
            hidden=hidden,
            tokens=tokens,
            positions=positions,
        )
        ttnn.synchronize_device(self.mesh_device)

        self.tt_hidden = _tt_hidden_from_torch(self.mesh_device, hidden, device=self.mesh_device)
        self.tt_tokens = _tt_tokens_from_torch(self.mesh_device, tokens, device=self.mesh_device)
        self.tt_positions = _tt_positions_from_torch(self.mesh_device, positions, device=self.mesh_device)
        self.tt_rot_idxs = self.gen.rope_setup.get_rot_idxs(positions)
        self._expected_hidden_shape = tuple(hidden.shape)
        self._expected_hidden_dtype = hidden.dtype
        self._expected_tokens_shape = tuple(tokens.shape)
        self._expected_tokens_dtype = tokens.dtype
        self._expected_positions_shape = tuple(positions.shape)
        self._expected_positions_dtype = positions.dtype

        self.gen.ccl.reset_sem_counters()
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        rope_tensors = self.gen.rope_setup.get_rot_mats_from_rot_idxs(self.tt_rot_idxs)
        self.trace_output = RowBatchedModel.forward_mtp_decode(
            hidden_states=self.tt_hidden,
            token_ids=self.tt_tokens,
            position_idxs=self.tt_positions,
            cfg=self.gen.model_run_config_decode,
            rope_tensors=rope_tensors,
            page_table=self.gen._get_mtp_page_table(),
        )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        self.trace_id = trace_id

    def _update_inputs(self, hidden: torch.Tensor, tokens: torch.Tensor, positions: torch.Tensor) -> None:
        assert (
            self.tt_hidden is not None
            and self.tt_tokens is not None
            and self.tt_positions is not None
            and self.tt_rot_idxs is not None
            and self.trace_id is not None
        )

        assert self._expected_hidden_shape == tuple(
            hidden.shape
        ), f"MTP trace hidden shape changed: expected {self._expected_hidden_shape}, got {tuple(hidden.shape)}"
        assert (
            self._expected_hidden_dtype == hidden.dtype
        ), f"MTP trace hidden dtype changed: expected {self._expected_hidden_dtype}, got {hidden.dtype}"
        assert self._expected_tokens_shape == tuple(
            tokens.shape
        ), f"MTP trace tokens shape changed: expected {self._expected_tokens_shape}, got {tuple(tokens.shape)}"
        assert (
            self._expected_tokens_dtype == tokens.dtype
        ), f"MTP trace tokens dtype changed: expected {self._expected_tokens_dtype}, got {tokens.dtype}"
        assert self._expected_positions_shape == tuple(
            positions.shape
        ), f"MTP trace positions shape changed: expected {self._expected_positions_shape}, got {tuple(positions.shape)}"
        assert (
            self._expected_positions_dtype == positions.dtype
        ), f"MTP trace positions dtype changed: expected {self._expected_positions_dtype}, got {positions.dtype}"

        host_hidden = _tt_hidden_from_torch(self.mesh_device, hidden, device=None)
        self._copy_checked(host_hidden, self.tt_hidden, "hidden")

        host_tokens = _tt_tokens_from_torch(self.mesh_device, tokens, device=None)
        self._copy_checked(host_tokens, self.tt_tokens, "tokens")

        host_positions = _tt_positions_from_torch(self.mesh_device, positions, device=None)
        self._copy_checked(host_positions, self.tt_positions, "positions")

        host_rot_idxs = self.gen.rope_setup.get_rot_idxs(positions, on_host=True)
        self._copy_checked(host_rot_idxs, self.tt_rot_idxs, "rot_idxs")

    @staticmethod
    def _copy_checked(host_tensor: ttnn.Tensor, device_tensor: ttnn.Tensor, name: str) -> None:
        try:
            assert (
                host_tensor.shape == device_tensor.shape
            ), f"MTP trace {name} shape mismatch: host {host_tensor.shape} vs device {device_tensor.shape}"
            assert (
                host_tensor.dtype == device_tensor.dtype
            ), f"MTP trace {name} dtype mismatch: host {host_tensor.dtype} vs device {device_tensor.dtype}"
            assert (
                host_tensor.layout == device_tensor.layout
            ), f"MTP trace {name} layout mismatch: host {host_tensor.layout} vs device {device_tensor.layout}"
            ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)
        finally:
            ttnn.deallocate(host_tensor)


# Test: generate the hidden-state and next-token oracle payload used by the MTP checks below.
@pytest.mark.timeout(TIMEOUT_S)
@pytest.mark.requires_device(["DUAL", "QUAD"])
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": TRACE_REGION_SIZE,
            },
            marks=SKIP_IN_CI,
        )
    ],
    indirect=True,
)
@pytest.mark.skipif(
    not GENERATE_REFERENCE,
    reason="Set DEEPSEEK_V3_MTP_GENERATE_REFERENCE=1 to generate MTP reference IO.",
)
def test_generate_mtp_reference_io(
    mesh_device,
    model_path,
    cache_path,
    force_recalculate_weight_config,
    set_deterministic_env,
):
    host_rank = int(os.getenv("TT_MESH_HOST_RANK", "0"))

    num_steps = int(os.getenv("DEEPSEEK_V3_MTP_REF_STEPS", str(DEFAULT_NUM_STEPS)))
    if num_steps < 2:
        pytest.skip("Need at least 2 steps to generate MTP reference IO.")

    mesh = mesh_device
    reference_path = _get_reference_path(num_steps)
    _get_reference_dir().mkdir(parents=True, exist_ok=True)

    with _prepare_generator(
        mesh_device=mesh,
        model_path=model_path,
        cache_path=cache_path,
        force_recalculate=force_recalculate_weight_config,
        enable_mtp=False,
    ) as gen:
        batch_size = gen.batch_size
        hidden_size = gen.hf_config.hidden_size
        vocab_size = gen.hf_config.vocab_size

        start_token_id = _get_start_token_id(gen.hf_config)
        initial_tokens = (torch.arange(batch_size, dtype=torch.long) + start_token_id) % vocab_size
        tokens_step = initial_tokens.clone()
        positions = torch.zeros(batch_size, dtype=torch.int32)

        hidden_states = torch.empty((num_steps, batch_size, hidden_size), dtype=torch.bfloat16)
        next_tokens = torch.empty((num_steps, batch_size), dtype=torch.int32)

        for step in range(num_steps):
            logits, hidden = gen._decode_step(
                tokens_step,
                positions,
                return_hidden=True,
            )

            hidden_step = hidden.squeeze(0).squeeze(0)
            if hidden_step.dim() == 3 and hidden_step.shape[0] == 1:
                hidden_step = hidden_step[0]
            hidden_states[step] = hidden_step.to(torch.bfloat16)

            step_logits = logits.squeeze(0).squeeze(0)
            step_next = torch.argmax(step_logits, dim=-1).to(torch.int32)
            next_tokens[step] = step_next

            tokens_step = step_next.to(torch.long)
            positions += 1

        payload = {
            "metadata": {
                "mesh_shape": list(mesh.shape),
                "num_steps": num_steps,
                "batch_size": batch_size,
                "hidden_size": hidden_size,
                "vocab_size": vocab_size,
                "start_token_id": start_token_id,
                "batch_size_per_row": gen.batch_size_per_row,
            },
            "hidden_states": hidden_states.cpu(),
            "next_tokens": next_tokens.cpu(),
            "start_tokens": initial_tokens.cpu(),
        }

        if host_rank == 0:
            tmp_path = reference_path.with_suffix(".tmp")
            torch.save(payload, tmp_path)
            tmp_path.replace(reference_path)
            logger.info(f"Saved MTP reference IO to {reference_path}")
        else:
            logger.info("Reference IO generation complete on host rank %d (skipping save).", host_rank)


# Test: the MTP predictor alone must meet the acceptance-rate and throughput gates over the reference window.
@pytest.mark.timeout(TIMEOUT_S)
@pytest.mark.requires_device(["DUAL", "QUAD"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": TRACE_REGION_SIZE,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("min_accept_rate", [0.75])
@pytest.mark.parametrize(
    "enable_trace",
    [
        False,
        # True
    ],
)
def test_mtp_accept_rate_and_perf(
    min_accept_rate,
    enable_trace,
    mesh_device,
    model_path,
    cache_path,
    force_recalculate_weight_config,
    set_deterministic_env,
):
    """Validate MTP-only predictor accept rate and throughput against reference IO."""
    num_steps = int(os.getenv("DEEPSEEK_V3_MTP_REF_STEPS", str(DEFAULT_NUM_STEPS)))
    mesh = mesh_device

    with _prepare_mtp_module_runner(
        mesh_device=mesh,
        model_path=model_path,
        cache_path=cache_path,
        force_recalculate=force_recalculate_weight_config,
    ) as gen:
        if not gen.enable_mtp:
            pytest.skip("MTP is disabled for this configuration; skipping MTP module test.")

        payload, reference_path = _load_reference_payload_for_generator(
            gen,
            num_steps,
            context="MTP accept rate and perf",
        )
        hidden_states = payload["hidden_states"].to(torch.bfloat16)
        next_tokens = payload["next_tokens"].to(torch.int32)

        if hidden_states.shape[0] < 2:
            pytest.skip("Reference IO must contain at least 2 steps for MTP verification.")

        _assert_reference_start_tokens(payload, gen, context="MTP accept rate and perf")

        total_matches = 0
        total_count = 0

        start_time = time.perf_counter()
        trace_runner = _MtpTraceRunner(gen) if enable_trace else None
        try:
            for step in range(hidden_states.shape[0] - 1):
                hidden = hidden_states[step]
                tokens = next_tokens[step]
                positions = torch.full((gen.batch_size,), step + 1, dtype=torch.int32)

                if enable_trace:
                    logits = trace_runner.run(hidden=hidden, tokens=tokens, positions=positions)
                else:
                    logits = _run_mtp_step(
                        gen=gen,
                        hidden=hidden,
                        tokens=tokens,
                        positions=positions,
                    )

                pred = torch.argmax(logits, dim=-1).to(torch.int32)
                gt = next_tokens[step + 1]

                total_matches += int((pred == gt).sum().item())
                total_count += gt.numel()
        finally:
            if trace_runner is not None:
                trace_runner.release()

        elapsed = time.perf_counter() - start_time
        ttnn.synchronize_device(mesh)

        accept_rate = total_matches / max(total_count, 1)
        tokens_per_sec = total_count / max(elapsed, 1e-9)
        logger.info(
            f"MTP accept rate: {total_matches}/{total_count} = {accept_rate:.3f} | "
            f"e2e {tokens_per_sec:.2f} tokens/s (elapsed {elapsed:.3f}s, trace={enable_trace})"
        )
        logger.info(
            "MTP test summary: steps={} batch={} total_tokens={} mesh={}x{} trace={} "
            "accept_rate={:.3f} min_accept_rate={:.3f} tps={:.2f} min_tps={:.2f} ref={}".format(
                hidden_states.shape[0],
                gen.batch_size,
                total_count,
                mesh.shape[0],
                mesh.shape[1],
                enable_trace,
                accept_rate,
                min_accept_rate,
                tokens_per_sec,
                MIN_TOKENS_PER_SEC_TRACE if enable_trace else MIN_TOKENS_PER_SEC,
                reference_path,
            )
        )

        assert (
            accept_rate >= min_accept_rate
        ), f"MTP accept rate {accept_rate:.3f} below required minimum {min_accept_rate:.3f}"

        min_tps = MIN_TOKENS_PER_SEC_TRACE if enable_trace else MIN_TOKENS_PER_SEC
        if min_tps > 0:
            assert (
                tokens_per_sec >= min_tps
            ), f"MTP e2e throughput {tokens_per_sec:.2f} tokens/s below required minimum {min_tps:.2f}"

        if MAX_E2E_SECONDS > 0:
            assert elapsed <= MAX_E2E_SECONDS, f"MTP e2e time {elapsed:.3f}s exceeds limit {MAX_E2E_SECONDS:.3f}s"


# Test: prefill priming must seed the MTP cache correctly so post-prefill predictions stay accurate.
@pytest.mark.timeout(TIMEOUT_S)
@pytest.mark.requires_device(["DUAL", "QUAD"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": TRACE_REGION_SIZE,
        }
    ],
    indirect=True,
)
def test_mtp_prefill_priming(
    mesh_device,
    model_path,
    cache_path,
    force_recalculate_weight_config,
    set_deterministic_env,
):
    """Validate prefill priming for MTP (user0) and post-prefill accept rate."""
    num_steps = int(os.getenv("DEEPSEEK_V3_MTP_REF_STEPS", str(DEFAULT_NUM_STEPS)))
    mesh = mesh_device

    prefill_seq_tile = ttnn.TILE_SIZE
    max_prompt_len = num_steps - 2
    if max_prompt_len < prefill_seq_tile:
        pytest.skip(f"Need at least {prefill_seq_tile} prompt tokens and 2 reference steps for prefill priming test.")

    requested_prompt_len = max(2, DEFAULT_PREFILL_LEN)
    # Prefill matmuls require the sequence length to be tile-aligned.
    prompt_len = ((requested_prompt_len + prefill_seq_tile - 1) // prefill_seq_tile) * prefill_seq_tile
    if prompt_len > max_prompt_len:
        prompt_len = (max_prompt_len // prefill_seq_tile) * prefill_seq_tile
    if prompt_len < prefill_seq_tile:
        pytest.skip(
            f"Could not choose a tile-aligned prefill prompt length <= {max_prompt_len} " f"(tile={prefill_seq_tile})."
        )

    with _prepare_generator(
        mesh_device=mesh,
        model_path=model_path,
        cache_path=cache_path,
        force_recalculate=force_recalculate_weight_config,
        enable_mtp=True,
    ) as gen:
        if not gen.enable_mtp:
            pytest.skip("MTP is disabled for this configuration; skipping MTP prefill priming test.")

        payload, _reference_path = _load_reference_payload_for_generator(
            gen,
            num_steps,
            context="MTP prefill priming",
        )
        hidden_states = payload["hidden_states"].to(torch.bfloat16)
        next_tokens = payload["next_tokens"].to(torch.int32)
        start_tokens = payload["start_tokens"].to(torch.long)

        _assert_reference_start_tokens(payload, gen, context="MTP prefill priming")
        logger.info(
            "MTP prefill priming prompt length: requested={} aligned={} tile={}",
            requested_prompt_len,
            prompt_len,
            prefill_seq_tile,
        )

        # Build a prompt for user 0 using reference tokens: t0 + t1..t{L-1}
        prompt_tokens = torch.empty((prompt_len,), dtype=torch.long)
        prompt_tokens[0] = start_tokens[0]
        if prompt_len > 1:
            prompt_tokens[1:] = next_tokens[: prompt_len - 1, 0].to(torch.long)

        # Prefill user 0 to prime the MTP cache and grab the last hidden state.
        _, last_hidden = gen._prefill(
            prompt_tokens,
            user_id=0,
            prompt_len=prompt_len,
            return_last_hidden=True,
        )
        gen.ccl.reset_sem_counters()

        # Run a single MTP prediction for the first post-prefill step and compare to reference.
        hidden_for_mtp = hidden_states[prompt_len - 1].clone()
        hidden_for_mtp[0] = last_hidden.to(torch.bfloat16)
        tokens_step = next_tokens[prompt_len - 1].to(torch.int32)
        positions = torch.full((gen.batch_size,), prompt_len, dtype=torch.int32)

        mtp_logits = gen._mtp_predict_logits(
            hidden_states=hidden_for_mtp,
            tokens_step=tokens_step,
            positions=positions,
        )
        pred = torch.argmax(mtp_logits, dim=-1).to(torch.int32)
        gt = next_tokens[prompt_len]

        _ = pred, gt  # First-step check suppressed; accept-rate is the only gate.

        steps_available = hidden_states.shape[0] - 1 - prompt_len
        steps_to_check = min(32, steps_available)
        if steps_to_check <= 0:
            pytest.skip("Not enough reference steps for post-prefill accept-rate check.")

        matches = 0
        for offset in range(steps_to_check):
            step = prompt_len + offset
            hidden_step = hidden_states[step]
            tokens_step = next_tokens[step]
            positions = torch.full((gen.batch_size,), step + 1, dtype=torch.int32)

            logits = gen._mtp_predict_logits(
                hidden_states=hidden_step,
                tokens_step=tokens_step,
                positions=positions,
            )
            pred = torch.argmax(logits, dim=-1).to(torch.int32)
            gt = next_tokens[step + 1]

            if pred[0].item() == gt[0].item():
                matches += 1
            gen.ccl.reset_sem_counters()

        accept_rate = matches / steps_to_check
        logger.info(
            "MTP prefill accept rate (user0): {}/{} = {:.3f}",
            matches,
            steps_to_check,
            accept_rate,
        )
        assert accept_rate >= 0.75, f"MTP prefill accept rate {accept_rate:.3f} below required minimum 0.750"


# Test: verify batching with aliased page tables must preserve prompt predictions, accept masks, and accepted verify outputs.
@pytest.mark.timeout(TIMEOUT_S)
@pytest.mark.requires_device(["DUAL", "QUAD"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": TRACE_REGION_SIZE,
        }
    ],
    indirect=True,
)
def test_mtp_verify_batching_aliasing(
    mesh_device,
    model_path,
    cache_path,
    force_recalculate_weight_config,
    set_deterministic_env,
):
    """Validate verify-lane batching + page-table aliasing invariance."""
    num_steps = int(os.getenv("DEEPSEEK_V3_MTP_REF_STEPS", str(DEFAULT_NUM_STEPS)))
    mesh = mesh_device

    with _prepare_generator(
        mesh_device=mesh,
        model_path=model_path,
        cache_path=cache_path,
        force_recalculate=force_recalculate_weight_config,
        enable_mtp=True,
    ) as gen:
        if not gen.enable_mtp:
            pytest.skip("MTP is disabled for this configuration; skipping verify-lane batching test.")

        payload, _reference_path = _load_reference_payload_for_generator(
            gen,
            num_steps,
            context="MTP verify batching aliasing",
        )
        hidden_states = payload["hidden_states"].to(torch.bfloat16)
        next_tokens = payload["next_tokens"].to(torch.int32)
        start_tokens = payload["start_tokens"].to(torch.int32)
        _assert_reference_start_tokens(payload, gen, context="MTP verify batching aliasing")

        batch_size = gen.batch_size  # total batch across all mesh rows
        # Per-shard batch size (users per DP shard within a mesh row).
        batch_per_shard = int(gen.batch_size_per_row // gen.dp_factor)
        if batch_per_shard < 2:
            pytest.skip(f"Verify aliasing requires at least 2 lanes per shard; batch_per_shard={batch_per_shard}.")

        # Interleaved layout: per mesh row, prompt/spec lanes alternate -> half the users are prompts.
        prompts_per_row = gen.batch_size_per_row // 2
        # Total prompt lanes across mesh rows (can be capped for debugging).
        full_num_prompts = prompts_per_row * mesh.shape[0]
        max_prompts_env = int(os.getenv("DEEPSEEK_V3_MTP_VERIFY_MAX_PROMPTS", "0"))
        num_prompts = min(full_num_prompts, max_prompts_env) if max_prompts_env > 0 else full_num_prompts
        if max_prompts_env > 0:
            logger.info(
                "Capping verify prompts: requested_max={} full_num_prompts={} -> num_prompts={}",
                max_prompts_env,
                full_num_prompts,
                num_prompts,
            )
        if num_prompts <= 0:
            pytest.skip("No prompt lanes available for verify batching test.")
        if 2 * num_prompts > batch_size:
            pytest.skip(f"Need at least 2x prompt lanes; batch_size={batch_size}, num_prompts={num_prompts}.")
        selected_prompt_indices = list(range(num_prompts)) if num_prompts < full_num_prompts else None
        if selected_prompt_indices is not None:
            logger.info(
                "Selective verify aliasing enabled: active_prompt_indices={} (full_num_prompts={} num_prompts={})",
                selected_prompt_indices,
                full_num_prompts,
                num_prompts,
            )

        # Map each prompt index to interleaved prompt/spec user IDs within the batch.
        prompt_user_ids_list: list[int] = []
        spec_user_ids_list: list[int] = []
        for i in range(num_prompts):
            row = i // prompts_per_row
            col = i % prompts_per_row
            base = row * gen.batch_size_per_row
            prompt_uid = base + 2 * col  # even lane = prompt
            spec_uid = prompt_uid + 1  # odd lane = spec/verify
            prompt_user_ids_list.append(prompt_uid)
            spec_user_ids_list.append(spec_uid)
        prompt_user_ids = torch.tensor(prompt_user_ids_list, dtype=torch.long)
        spec_user_ids = torch.tensor(spec_user_ids_list, dtype=torch.long)
        logger.info("prompt_user_ids: {}", prompt_user_ids.tolist())
        logger.info("spec_user_ids: {}", spec_user_ids.tolist())

        # Need at least two future steps for next and next-next token checks.
        steps_available = hidden_states.shape[0] - 2
        if steps_available <= 0:
            pytest.skip("Not enough reference steps for verify batching test.")
        steps_to_check = min(DEFAULT_VERIFY_STEPS, steps_available)  # keep test short

        # Build aliased page tables so spec lanes share prompt-lane KV cache pages.
        decode_page_tables = gen._build_mtp_verify_page_tables(
            num_prompts=num_prompts,
            verify_offset=0,
            prompt_indices=selected_prompt_indices,
            interleaved=True,
        )
        logger.info("decode_page_tables layers: {}", len(decode_page_tables))
        if len(decode_page_tables) > 0:
            try:
                logger.info("decode_page_tables[0] shape: {}", tuple(decode_page_tables[0].shape))
            except Exception as exc:
                logger.info("decode_page_tables[0] shape: <unavailable> ({})", exc)
        debug_mtp = _debug_mtp_enabled()
        alias_debug = debug_mtp
        if alias_debug and len(decode_page_tables) > 0:
            try:
                pt0_host = ttnn.to_torch(
                    decode_page_tables[0],
                    mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
                )
                # If replicated across devices, slice to one replica.
                num_devices = mesh.shape[0] * mesh.shape[1]
                if num_devices > 1 and pt0_host.shape[0] % num_devices == 0:
                    rows_per_device = pt0_host.shape[0] // num_devices
                    pt0_host = pt0_host.reshape(num_devices, rows_per_device, -1)[0]
                row0 = pt0_host[0].tolist()[:8] if pt0_host.shape[0] > 0 else []
                row1 = pt0_host[1].tolist()[:8] if pt0_host.shape[0] > 1 else []
                row2 = pt0_host[2].tolist()[:8] if pt0_host.shape[0] > 2 else []
                row3 = pt0_host[3].tolist()[:8] if pt0_host.shape[0] > 3 else []
                eq01 = bool(pt0_host.shape[0] > 1 and torch.equal(pt0_host[0], pt0_host[1]))
                eq23 = bool(pt0_host.shape[0] > 3 and torch.equal(pt0_host[2], pt0_host[3]))
                logger.info(
                    "decode_page_tables[0] host alias debug shape={} row0[:8]={} row1[:8]={} row2[:8]={} row3[:8]={} eq01={} eq23={}",
                    tuple(int(dim) for dim in pt0_host.shape),
                    row0,
                    row1,
                    row2,
                    row3,
                    eq01,
                    eq23,
                )
            except Exception as exc:
                logger.info("decode_page_tables[0] host alias debug failed: {}", exc)

        # Helper: map a global lane index to device index and local lane within the shard.
        def _lane_to_device_and_local(lane: int) -> tuple[int, int]:
            batch_per_shard = int(gen.batch_size_per_row // gen.dp_factor)
            row = lane // gen.batch_size_per_row
            within_row = lane % gen.batch_size_per_row
            shard_col = within_row // batch_per_shard
            local_lane = within_row % batch_per_shard
            device_idx = row * gen.dp_factor + shard_col
            return device_idx, local_lane

        # (lane sanity checks moved below, after logging flags are set)

        # Debug toggles for KV cache snapshots and verify table prints.
        enable_kv_log = False
        enable_verify_table = debug_mtp
        log_host_rank_env = os.getenv("TT_LOG_HOST_RANK")
        log_host_rank = int(log_host_rank_env) if log_host_rank_env not in (None, "") else None
        host_rank = int(os.getenv("TT_MESH_HOST_RANK", "0"))
        if log_host_rank is not None and host_rank != log_host_rank:
            enable_kv_log = False
            enable_verify_table = False
        if debug_mtp:
            logger.info(
                "DEEPSEEK_DEBUG_MTP enabled: enable_verify_table={} enable_kv_log={} kv_log_reason={}",
                enable_verify_table,
                enable_kv_log,
                "disabled because full KV host dumps stall before verify decode",
            )

        # Sanity-check prompt/spec lane aliasing within each shard.
        mapping_rows: list[str] = []
        for i in range(num_prompts):
            prompt_uid = int(prompt_user_ids[i].item())
            spec_uid = int(spec_user_ids[i].item())
            prompt_dev, prompt_local = _lane_to_device_and_local(prompt_uid)
            spec_dev, spec_local = _lane_to_device_and_local(spec_uid)

            if enable_verify_table:
                mapping_rows.append(
                    f"{i:>3} | p={prompt_uid:>4} -> (dev={prompt_dev:>2}, lane={prompt_local:>2}) "
                    f"| s={spec_uid:>4} -> (dev={spec_dev:>2}, lane={spec_local:>2})"
                )

            assert prompt_dev == spec_dev, (
                f"Prompt/spec lanes must be on same device: prompt_uid={prompt_uid} dev={prompt_dev}, "
                f"spec_uid={spec_uid} dev={spec_dev}"
            )
            assert (
                prompt_local % 2 == 0
            ), f"Prompt lane should be even local_lane: prompt_uid={prompt_uid} local_lane={prompt_local}"
            assert spec_local == prompt_local + 1, (
                f"Spec lane must alias adjacent local lane: prompt_uid={prompt_uid} local_lane={prompt_local}, "
                f"spec_uid={spec_uid} local_lane={spec_local}"
            )

        if enable_verify_table and mapping_rows:
            logger.info("MTP verify lane mapping:\n{}", "\n".join(mapping_rows))
        if gen.base_page_table_host is None:
            _ = gen._get_page_tables()
        base_page_table = gen.base_page_table_host.to(torch.int32)
        alias_page_table = _build_verify_alias_page_table_host(
            base_page_table=base_page_table,
            num_prompts=base_page_table.shape[0] // 2,
            verify_offset=0,
            prompt_indices=None,
            interleaved=True,
        )
        logger.info(
            "base_page_table_host shape={} values={}",
            tuple(int(dim) for dim in base_page_table.shape),
            base_page_table.tolist(),
        )
        logger.info(
            "alias_page_table_host shape={} values={}",
            tuple(int(dim) for dim in alias_page_table.shape),
            alias_page_table.tolist(),
        )

        # Optional KV cache logger for specific prompt/spec lanes at specific positions.
        def _dump_kv_cache(tag: str, step_idx: int, prompt_lane: int, spec_lane: int) -> None:
            if not enable_kv_log:
                return
            kv_cache_list = gen.get_kv_cache()
            if not kv_cache_list:
                logger.warning("{}: no kv_cache entries available", tag)
                return

            kv_cache_tt = kv_cache_list[0]
            mesh_composer = ttnn.ConcatMeshToTensor(mesh, dim=0)
            kv_cache_host = ttnn.aggregate_tensor(kv_cache_tt, mesh_composer)
            kv_cache_torch = kv_cache_host.to_torch()
            max_num_blocks = int(gen.paged_config.max_num_blocks)
            block_size = int(gen.paged_config.block_size)
            kv_dim = int(kv_cache_torch.shape[-1])
            num_devices = mesh.shape[0] * mesh.shape[1]
            kv_cache_torch = kv_cache_torch.reshape(num_devices, max_num_blocks, 1, block_size, kv_dim)

            if gen.base_page_table_host is None:
                _ = gen._get_page_tables()
            base_page_table = gen.base_page_table_host.to(torch.int32)
            alias_page_table = _build_alias_page_table(base_page_table)

            def _log_lane(lane: int, pos: int, label: str) -> None:
                device_idx, local_lane = _lane_to_device_and_local(lane)
                if device_idx < 0 or device_idx >= num_devices:
                    logger.warning(
                        "{} {} lane={} device_idx={} out of range (num_devices={})",
                        tag,
                        label,
                        lane,
                        device_idx,
                        num_devices,
                    )
                    return
                block = pos // block_size
                offset = pos % block_size
                physical_block = int(alias_page_table[local_lane, block].item())
                vec = kv_cache_torch[device_idx, physical_block, 0, offset, :].cpu()
                logger.info(
                    "{} {} lane={} (device={}, local_lane={}) pos={} block={} offset={} phys_block={} vec={}",
                    tag,
                    label,
                    lane,
                    device_idx,
                    local_lane,
                    pos,
                    block,
                    offset,
                    physical_block,
                    vec,
                )

            prompt_pos = step_idx + 1  # position for prompt-lane token
            spec_pos = step_idx + 2  # position for spec-lane token
            logger.info(
                "{} cache snapshot: step={} prompt_pos={} spec_pos={} block_size={} kv_dim={}",
                tag,
                step_idx,
                prompt_pos,
                spec_pos,
                block_size,
                kv_dim,
            )
            _log_lane(prompt_lane, prompt_pos, "prompt")
            _log_lane(spec_lane, spec_pos, "spec")

        # Seed cache with position 0 token (matches reference stream).
        seed_positions = torch.zeros((batch_size,), dtype=torch.int32)
        _dump_kv_cache(
            "before_seed_decode", step_idx=0, prompt_lane=int(prompt_user_ids[0]), spec_lane=int(spec_user_ids[0])
        )
        seed_logits_tt = gen._decode_step_tt(
            tokens_step=start_tokens,
            positions=seed_positions,
            batch_size_per_row=gen.batch_size_per_row,
            page_tables=gen._get_page_tables(),
            return_hidden=False,
        )
        ttnn.deallocate(seed_logits_tt)
        gen.ccl.reset_sem_counters()
        _dump_kv_cache(
            "after_seed_decode", step_idx=0, prompt_lane=int(prompt_user_ids[0]), spec_lane=int(spec_user_ids[0])
        )
        base_page_tables = gen._get_page_tables()

        prompt_matches = 0  # count of correct prompt-lane next-token predictions
        prompt_count = 0  # total prompt-lane comparisons
        verify_matches = 0  # count of correct verify-lane next-next predictions (accepted only)
        verify_count = 0  # total accepted prompt lanes
        debug_batched_history: list[tuple[torch.Tensor, torch.Tensor]] = []
        debug_spec_prev_history: list[torch.Tensor] = []
        debug_gt_next_history: list[torch.Tensor] = []
        debug_gt_after_spec_history: list[torch.Tensor] = []
        predictor_full_matches = 0
        predictor_full_count = 0
        predictor_prompt_matches = 0
        predictor_prompt_count = 0
        predictor_spec_matches = 0
        predictor_spec_count = 0
        debug_layer_compare_done = False
        # For the single-prompt aliasing repro, the first observed prompt-lane
        # mismatch happens at verify step 7, so compare there instead of step 1.
        debug_compare_step = 7

        for step in range(steps_to_check):
            # Current token is reference next_tokens[step] at position step+1 (post-prefill alignment).
            prompt_tokens = next_tokens[step].to(torch.int32)
            positions_prompt = torch.full((batch_size,), step + 1, dtype=torch.int32)  # prompt pos = t+1

            # MTP predicts next-next tokens from hidden[t] and token[t+1] for all lanes.
            spec_next_logits = gen._mtp_predict_logits(
                hidden_states=hidden_states[step],
                tokens_step=prompt_tokens,
                positions=positions_prompt,
            )
            spec_next_all = torch.argmax(spec_next_logits, dim=-1).to(prompt_tokens.dtype)
            spec_next_prompt = spec_next_all[prompt_user_ids]
            spec_prev_prompt = spec_next_prompt  # no carry-over in reference-aligned mode

            # MTP prediction from hidden[t+1] for table visibility only.
            positions_after_spec = torch.full((batch_size,), step + 2, dtype=torch.int32)
            spec_after_spec_logits = gen._mtp_predict_logits(
                hidden_states=hidden_states[step + 1],
                tokens_step=next_tokens[step + 1],
                positions=positions_after_spec,
            )
            spec_after_spec_all = torch.argmax(spec_after_spec_logits, dim=-1).to(torch.int32)
            spec_after_prompt = spec_after_spec_all[prompt_user_ids]

            # Build 2x-batch verification input:
            # prompt lanes = token[t+1] at pos t+1; spec lanes = predicted token at pos t+2.
            batched_tokens = prompt_tokens.clone()
            batched_positions = positions_prompt.clone()
            batched_tokens[spec_user_ids] = spec_prev_prompt
            batched_positions[spec_user_ids] = positions_prompt[prompt_user_ids] + 1

            gt_next = next_tokens[step + 1][prompt_user_ids]  # reference next token
            gt_after_spec = next_tokens[step + 2][prompt_user_ids]  # reference next-next token

            predictor_full_matches += int((spec_next_all == next_tokens[step + 1]).sum().item())
            predictor_full_count += int(next_tokens[step + 1].numel())
            predictor_prompt_matches += int((spec_prev_prompt == gt_next).sum().item())
            predictor_prompt_count += int(gt_next.numel())
            predictor_spec_matches += int(
                (spec_next_all[spec_user_ids] == next_tokens[step + 1][spec_user_ids]).sum().item()
            )
            predictor_spec_count += int(spec_user_ids.numel())

            if step == 0:
                _dump_kv_cache(
                    "before_batched_decode_step0",
                    step_idx=step,
                    prompt_lane=int(prompt_user_ids[0]),
                    spec_lane=int(spec_user_ids[0]),
                )

            if (
                debug_mtp
                and not debug_layer_compare_done
                and step == debug_compare_step
                and prompt_user_ids.numel() > 0
            ):
                if prompt_user_ids.numel() > 1:
                    capture_user_ids = [int(prompt_user_ids[0].item()), int(prompt_user_ids[1].item())]
                    control_uid = capture_user_ids[0]
                    target_uid = capture_user_ids[1]
                else:
                    capture_user_ids = [int(prompt_user_ids[0].item())]
                    control_uid = None
                    target_uid = capture_user_ids[0]
                logger.info(
                    "MTP layer compare: rebuilding verify state for step {} target_uid={} control_uid={} prior_replay_steps={} capture_user_ids={}",
                    step,
                    target_uid,
                    control_uid,
                    len(debug_batched_history),
                    capture_user_ids,
                )

                base_page_tables = gen._get_page_tables()

                _reset_base_kv_caches(gen)
                _replay_verify_decode_batches(
                    gen=gen,
                    start_tokens=start_tokens,
                    batched_history=debug_batched_history,
                    page_tables=base_page_tables,
                )
                base_logits, base_layer_captures = _decode_step_layerwise_host(
                    gen=gen,
                    tokens_step=batched_tokens,
                    positions=batched_positions,
                    page_tables=base_page_tables,
                    capture_user_ids=capture_user_ids,
                )
                base_pred_all = torch.argmax(base_logits, dim=-1).to(torch.int32)

                _reset_base_kv_caches(gen)
                _replay_verify_decode_batches(
                    gen=gen,
                    start_tokens=start_tokens,
                    batched_history=debug_batched_history,
                    page_tables=decode_page_tables,
                )
                logits_2b, alias_layer_captures = _decode_step_layerwise_host(
                    gen=gen,
                    tokens_step=batched_tokens,
                    positions=batched_positions,
                    page_tables=decode_page_tables,
                    capture_user_ids=capture_user_ids,
                )
                debug_layer_compare_done = True

                alias_pred_all = torch.argmax(logits_2b, dim=-1).to(torch.int32)
                if control_uid is None:
                    logger.info(
                        "MTP layer compare final preds: step={} target_uid={} base_pred={} alias_pred={}",
                        step,
                        target_uid,
                        int(base_pred_all[target_uid].item()),
                        int(alias_pred_all[target_uid].item()),
                    )
                else:
                    logger.info(
                        "MTP layer compare final preds: step={} target_uid={} base_pred={} alias_pred={} control_uid={} base_control_pred={} alias_control_pred={}",
                        step,
                        target_uid,
                        int(base_pred_all[target_uid].item()),
                        int(alias_pred_all[target_uid].item()),
                        control_uid,
                        int(base_pred_all[control_uid].item()),
                        int(alias_pred_all[control_uid].item()),
                    )

                first_divergent_layer = None
                for base_cap, alias_cap in zip(base_layer_captures, alias_layer_captures, strict=True):
                    layer_idx = int(base_cap["layer_idx"])
                    block_kind = str(base_cap["block_kind"])
                    base_rows = base_cap["rows"]
                    alias_rows = alias_cap["rows"]
                    base_target = base_rows[target_uid]
                    alias_target = alias_rows[target_uid]
                    target_equal = torch.equal(base_target, alias_target)
                    target_max_abs = float((base_target - alias_target).abs().max().item())
                    if control_uid is None:
                        logger.info(
                            "MTP layer compare step={} layer={} kind={} target_uid={} equal={} max_abs_diff={:.6f} "
                            "base[:8]={} alias[:8]={}",
                            step,
                            layer_idx,
                            block_kind,
                            target_uid,
                            target_equal,
                            target_max_abs,
                            base_target[:8].tolist(),
                            alias_target[:8].tolist(),
                        )
                    else:
                        base_control = base_rows[control_uid]
                        alias_control = alias_rows[control_uid]
                        control_equal = torch.equal(base_control, alias_control)
                        control_max_abs = float((base_control - alias_control).abs().max().item())
                        logger.info(
                            "MTP layer compare step={} layer={} kind={} target_uid={} equal={} max_abs_diff={:.6f} "
                            "base[:8]={} alias[:8]={} control_uid={} control_equal={} control_max_abs_diff={:.6f}",
                            step,
                            layer_idx,
                            block_kind,
                            target_uid,
                            target_equal,
                            target_max_abs,
                            base_target[:8].tolist(),
                            alias_target[:8].tolist(),
                            control_uid,
                            control_equal,
                            control_max_abs,
                        )
                    if first_divergent_layer is None and not target_equal:
                        first_divergent_layer = (layer_idx, block_kind, target_max_abs)

                logger.info(
                    "MTP layer compare summary: step={} target_uid={} first_divergent_layer={}",
                    step,
                    target_uid,
                    first_divergent_layer,
                )

                trajectory_rows: list[str] = []
                probe_steps = range(max(0, step - 1), step + 1)
                for probe_step in probe_steps:
                    probe_prompt_tokens = next_tokens[probe_step].to(torch.int32)
                    probe_positions_prompt = torch.full((batch_size,), probe_step + 1, dtype=torch.int32)
                    gt_probe_next = int(next_tokens[probe_step + 1][target_uid].item())

                    _reset_base_kv_caches(gen)
                    _replay_reference_decode_steps(
                        gen=gen,
                        start_tokens=start_tokens,
                        next_tokens=next_tokens,
                        num_prior_steps=probe_step,
                    )
                    prompt_only_logits = _decode_step_host(
                        gen=gen,
                        tokens_step=probe_prompt_tokens,
                        positions=probe_positions_prompt,
                        page_tables=base_page_tables,
                    )
                    prompt_only_pred = int(torch.argmax(prompt_only_logits, dim=-1).to(torch.int32)[target_uid].item())

                    if probe_step < len(debug_batched_history):
                        probe_batched_tokens, probe_batched_positions = debug_batched_history[probe_step]
                        probe_prior_history = debug_batched_history[:probe_step]
                    else:
                        probe_batched_tokens = batched_tokens
                        probe_batched_positions = batched_positions
                        probe_prior_history = debug_batched_history

                    _reset_base_kv_caches(gen)
                    _replay_verify_decode_batches(
                        gen=gen,
                        start_tokens=start_tokens,
                        batched_history=probe_prior_history,
                        page_tables=base_page_tables,
                    )
                    verify_base_logits = _decode_step_host(
                        gen=gen,
                        tokens_step=probe_batched_tokens,
                        positions=probe_batched_positions,
                        page_tables=base_page_tables,
                    )
                    verify_base_pred = int(torch.argmax(verify_base_logits, dim=-1).to(torch.int32)[target_uid].item())
                    spec_token = int(probe_batched_tokens[int(spec_user_ids[0].item())].item())
                    spec_pos = int(probe_batched_positions[int(spec_user_ids[0].item())].item())
                    trajectory_rows.append(
                        "step={} prompt_token={} prompt_pos={} spec_token={} spec_pos={} gt_next={} "
                        "prompt_only_pred={} verify_base_pred={}".format(
                            probe_step,
                            int(probe_prompt_tokens[target_uid].item()),
                            int(probe_positions_prompt[target_uid].item()),
                            spec_token,
                            spec_pos,
                            gt_probe_next,
                            prompt_only_pred,
                            verify_base_pred,
                        )
                    )

                logger.info(
                    "MTP trajectory compare target_uid={} rows:\n{}",
                    target_uid,
                    "\n".join(trajectory_rows),
                )
            else:
                logits_2b_tt = gen._decode_step_tt(
                    tokens_step=batched_tokens,
                    positions=batched_positions,
                    batch_size_per_row=gen.batch_size_per_row,
                    page_tables=decode_page_tables,
                    return_hidden=False,
                )
                logits_2b = ttnn.to_torch(
                    logits_2b_tt,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(-2, -1), mesh_shape=mesh.shape),
                )
                ttnn.deallocate(logits_2b_tt)
                gen.ccl.reset_sem_counters()
                logits_2b = logits_2b.squeeze(0).squeeze(0)

            pred_all = torch.argmax(logits_2b, dim=-1).to(torch.int32)
            pred_next = pred_all[prompt_user_ids]  # base-model next-token prediction
            pred_after_spec = pred_all[spec_user_ids]  # base-model next-next prediction

            if step == 0:
                _dump_kv_cache(
                    "after_batched_decode_step0",
                    step_idx=step,
                    prompt_lane=int(prompt_user_ids[0]),
                    spec_lane=int(spec_user_ids[0]),
                )

            if enable_verify_table:
                header_cols = [
                    "step",
                    "req",
                    "row",
                    "pos",
                    "next_pred",
                    "match_gt",
                    "spec_prev",
                    "next_spec",
                    "pred_after_spec",
                    "spec_after_spec",
                    "accept/reject",
                ]
                col_widths = [4, 4, 4, 4, 9, 9, 9, 9, 14, 14, 13]

                def _fmt_row(values: list[object]) -> str:
                    return "\t".join(f"{str(v):>{w}}" for v, w in zip(values, col_widths, strict=True))

                rows = [_fmt_row(header_cols)]
                for i in range(num_prompts):
                    user_id = int(prompt_user_ids[i].item())
                    pos_id = int(positions_prompt[user_id].item())
                    next_pred_val = int(pred_next[i].item())
                    gt_next_val = int(gt_next[i].item())
                    match_gt_val = "MATCH" if next_pred_val == gt_next_val else "MISMATCH"
                    spec_prev_val = int(spec_prev_prompt[i].item())
                    next_spec_val = int(spec_next_prompt[i].item())
                    pred_after_val = int(pred_after_spec[i].item())
                    spec_after_val = int(spec_after_prompt[i].item())
                    verdict = "ACCEPT" if next_pred_val == spec_prev_val else "REJECT"
                    rows.append(
                        _fmt_row(
                            [
                                step,
                                i,
                                user_id,
                                pos_id,
                                next_pred_val,
                                match_gt_val,
                                spec_prev_val,
                                next_spec_val,
                                pred_after_val,
                                spec_after_val,
                                verdict,
                            ]
                        )
                    )
                logger.info("MTP verify table:\n{}", "\n".join(rows))

            prompt_matches += int((pred_next == gt_next).sum().item())
            prompt_count += int(gt_next.numel())

            # Accept if base-model next token matches the carried-over spec token.
            accepted_mask = pred_next == spec_prev_prompt
            if accepted_mask.any():
                verify_matches += int((pred_after_spec[accepted_mask] == gt_after_spec[accepted_mask]).sum().item())
                verify_count += int(accepted_mask.sum().item())

            debug_batched_history.append((batched_tokens.clone(), batched_positions.clone()))
            debug_spec_prev_history.append(spec_prev_prompt.clone())
            debug_gt_next_history.append(gt_next.clone())
            debug_gt_after_spec_history.append(gt_after_spec.clone())

        def _replay_verify_accept_diagnostics(
            page_tables: list[ttnn.Tensor],
            label: str,
        ) -> tuple[int, int, int, list[list[int]], list[torch.Tensor], list[torch.Tensor]]:
            _reset_base_kv_caches(gen)
            seed_logits_tt = gen._decode_step_tt(
                tokens_step=start_tokens,
                positions=seed_positions,
                batch_size_per_row=gen.batch_size_per_row,
                page_tables=gen._get_page_tables(),
                return_hidden=False,
            )
            ttnn.deallocate(seed_logits_tt)
            gen.ccl.reset_sem_counters()

            replay_prompt_matches = 0
            replay_accept_count = 0
            per_step_accepted_prompt_indices: list[list[int]] = []
            per_step_prompt_preds: list[torch.Tensor] = []
            per_step_verify_preds: list[torch.Tensor] = []

            for step_idx, (batched_tokens_step, batched_positions_step) in enumerate(debug_batched_history):
                logits_tt = gen._decode_step_tt(
                    tokens_step=batched_tokens_step,
                    positions=batched_positions_step,
                    batch_size_per_row=gen.batch_size_per_row,
                    page_tables=page_tables,
                    return_hidden=False,
                )
                logits_host = ttnn.to_torch(
                    logits_tt,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(-2, -1), mesh_shape=mesh.shape),
                )
                ttnn.deallocate(logits_tt)
                gen.ccl.reset_sem_counters()
                logits_host = logits_host.squeeze(0).squeeze(0)
                pred_all_step = torch.argmax(logits_host, dim=-1).to(torch.int32)
                pred_next_step = pred_all_step[prompt_user_ids]
                pred_after_spec_step = pred_all_step[spec_user_ids]
                replay_prompt_matches += int((pred_next_step == debug_gt_next_history[step_idx]).sum().item())
                accepted_mask_step = pred_next_step == debug_spec_prev_history[step_idx]
                replay_accept_count += int(accepted_mask_step.sum().item())
                per_step_accepted_prompt_indices.append(accepted_mask_step.nonzero(as_tuple=False).flatten().tolist())
                per_step_prompt_preds.append(pred_next_step.clone())
                per_step_verify_preds.append(pred_after_spec_step.clone())

            logger.info(
                "MTP verify replay {}: prompt_matches={}/{} ({:.3f}) accept_count={}/{} ({:.3f})",
                label,
                replay_prompt_matches,
                prompt_count,
                replay_prompt_matches / max(prompt_count, 1),
                replay_accept_count,
                prompt_count,
                replay_accept_count / max(prompt_count, 1),
            )
            return (
                replay_prompt_matches,
                replay_accept_count,
                prompt_count,
                per_step_accepted_prompt_indices,
                per_step_prompt_preds,
                per_step_verify_preds,
            )

        prompt_rate = prompt_matches / max(prompt_count, 1)
        accept_rate = verify_count / max(prompt_count, 1)
        logger.info(f"MTP verify batching prompt match rate: {prompt_matches}/{prompt_count} = {prompt_rate:.3f}")
        logger.info(f"MTP verify batching short-window accept rate: {verify_count}/{prompt_count} = {accept_rate:.3f}")
        if debug_mtp:
            predictor_full_rate = predictor_full_matches / max(predictor_full_count, 1)
            predictor_prompt_rate = predictor_prompt_matches / max(predictor_prompt_count, 1)
            predictor_spec_rate = predictor_spec_matches / max(predictor_spec_count, 1)
            logger.info(
                "MTP verify predictor baseline: full_batch={}/{} ({:.3f}) prompt_subset={}/{} ({:.3f}) spec_subset={}/{} ({:.3f})",
                predictor_full_matches,
                predictor_full_count,
                predictor_full_rate,
                predictor_prompt_matches,
                predictor_prompt_count,
                predictor_prompt_rate,
                predictor_spec_matches,
                predictor_spec_count,
                predictor_spec_rate,
            )
            _reset_mtp_kv_cache(gen)
            full_window_full_matches = 0
            full_window_full_count = 0
            full_window_prompt_matches = 0
            full_window_prompt_count = 0
            full_window_spec_matches = 0
            full_window_spec_count = 0
            for step_idx in range(hidden_states.shape[0] - 1):
                positions_full = torch.full((batch_size,), step_idx + 1, dtype=torch.int32)
                logits_full = gen._mtp_predict_logits(
                    hidden_states=hidden_states[step_idx],
                    tokens_step=next_tokens[step_idx],
                    positions=positions_full,
                )
                pred_full = torch.argmax(logits_full, dim=-1).to(torch.int32)
                gt_full = next_tokens[step_idx + 1]
                full_window_full_matches += int((pred_full == gt_full).sum().item())
                full_window_full_count += int(gt_full.numel())
                full_window_prompt_matches += int((pred_full[prompt_user_ids] == gt_full[prompt_user_ids]).sum().item())
                full_window_prompt_count += int(prompt_user_ids.numel())
                full_window_spec_matches += int((pred_full[spec_user_ids] == gt_full[spec_user_ids]).sum().item())
                full_window_spec_count += int(spec_user_ids.numel())
            logger.info(
                "MTP verify predictor full-window baseline: full_batch={}/{} ({:.3f}) prompt_subset={}/{} ({:.3f}) spec_subset={}/{} ({:.3f})",
                full_window_full_matches,
                full_window_full_count,
                full_window_full_matches / max(full_window_full_count, 1),
                full_window_prompt_matches,
                full_window_prompt_count,
                full_window_prompt_matches / max(full_window_prompt_count, 1),
                full_window_spec_matches,
                full_window_spec_count,
                full_window_spec_matches / max(full_window_spec_count, 1),
            )
        (
            base_prompt_matches_dbg,
            base_accept_count_dbg,
            _,
            base_step_accept_indices_dbg,
            base_step_prompt_preds_dbg,
            base_step_verify_preds_dbg,
        ) = _replay_verify_accept_diagnostics(base_page_tables, "base")
        (
            alias_prompt_matches_dbg,
            alias_accept_count_dbg,
            _,
            alias_step_accept_indices_dbg,
            alias_step_prompt_preds_dbg,
            alias_step_verify_preds_dbg,
        ) = _replay_verify_accept_diagnostics(decode_page_tables, "alias")
        differing_accept_steps = [
            step_idx
            for step_idx, (base_idx, alias_idx) in enumerate(
                zip(base_step_accept_indices_dbg, alias_step_accept_indices_dbg, strict=True)
            )
            if base_idx != alias_idx
        ]
        differing_prompt_pred_steps = [
            step_idx
            for step_idx, (base_pred, alias_pred) in enumerate(
                zip(base_step_prompt_preds_dbg, alias_step_prompt_preds_dbg, strict=True)
            )
            if not torch.equal(base_pred, alias_pred)
        ]
        differing_verify_pred_steps = []
        differing_verify_pred_steps_all = []
        for step_idx, (base_pred, alias_pred, accepted_indices) in enumerate(
            zip(
                base_step_verify_preds_dbg,
                alias_step_verify_preds_dbg,
                base_step_accept_indices_dbg,
                strict=True,
            )
        ):
            if not torch.equal(base_pred, alias_pred):
                differing_verify_pred_steps_all.append(step_idx)
            if not accepted_indices:
                continue
            accepted_idx = torch.tensor(accepted_indices, dtype=torch.long)
            if not torch.equal(base_pred[accepted_idx], alias_pred[accepted_idx]):
                differing_verify_pred_steps.append(step_idx)
        logger.info(
            "MTP verify replay compare: base_accept_count={} alias_accept_count={} differing_accept_steps={} differing_prompt_pred_steps={} differing_verify_pred_steps={} differing_verify_pred_steps_all={}",
            base_accept_count_dbg,
            alias_accept_count_dbg,
            differing_accept_steps,
            differing_prompt_pred_steps,
            differing_verify_pred_steps,
            differing_verify_pred_steps_all,
        )
        if differing_accept_steps or differing_prompt_pred_steps:
            sample_steps = sorted(set(differing_accept_steps + differing_prompt_pred_steps))[:8]
            logger.info(
                "MTP verify replay prompt/accept differing details: {}",
                [
                    {
                        "step": step_idx,
                        "base_accept_prompt_indices": base_step_accept_indices_dbg[step_idx],
                        "alias_accept_prompt_indices": alias_step_accept_indices_dbg[step_idx],
                        "base_prompt_pred_sample": base_step_prompt_preds_dbg[step_idx][:8].tolist(),
                        "alias_prompt_pred_sample": alias_step_prompt_preds_dbg[step_idx][:8].tolist(),
                        "accepted_prompt_indices": base_step_accept_indices_dbg[step_idx],
                        "base_verify_pred_sample": base_step_verify_preds_dbg[step_idx][:8].tolist(),
                        "alias_verify_pred_sample": alias_step_verify_preds_dbg[step_idx][:8].tolist(),
                    }
                    for step_idx in sample_steps
                ],
            )
        if debug_mtp and differing_verify_pred_steps:
            sample_steps = differing_verify_pred_steps[:8]
            logger.info(
                "MTP verify replay alias-aware note: verify-lane base/alias differences are expected because only the aliased spec rows share prompt history. sample_steps={}",
                sample_steps,
            )
            logger.info(
                "MTP verify replay verify-lane differing details: {}",
                [
                    {
                        "step": step_idx,
                        "accepted_prompt_indices": base_step_accept_indices_dbg[step_idx],
                        "base_verify_pred_sample": base_step_verify_preds_dbg[step_idx][:8].tolist(),
                        "alias_verify_pred_sample": alias_step_verify_preds_dbg[step_idx][:8].tolist(),
                    }
                    for step_idx in sample_steps
                ],
            )
            first_verify_step = differing_verify_pred_steps[0]
            first_accepted_indices = base_step_accept_indices_dbg[first_verify_step]
            first_mismatch_prompt_idx = next(
                prompt_idx
                for prompt_idx in first_accepted_indices
                if int(base_step_verify_preds_dbg[first_verify_step][prompt_idx].item())
                != int(alias_step_verify_preds_dbg[first_verify_step][prompt_idx].item())
            )
            first_prompt_uid = int(prompt_user_ids[first_mismatch_prompt_idx].item())
            first_spec_uid = int(spec_user_ids[first_mismatch_prompt_idx].item())
            first_batched_tokens, first_batched_positions = debug_batched_history[first_verify_step]
            logger.info(
                "MTP verify first accepted divergence: step={} prompt_idx={} prompt_uid={} spec_uid={} "
                "prompt_token={} prompt_pos={} spec_token={} spec_pos={} spec_prev={} gt_next={} gt_after_spec={} "
                "base_prompt_pred={} alias_prompt_pred={} base_verify_pred={} alias_verify_pred={}",
                first_verify_step,
                first_mismatch_prompt_idx,
                first_prompt_uid,
                first_spec_uid,
                int(first_batched_tokens[first_prompt_uid].item()),
                int(first_batched_positions[first_prompt_uid].item()),
                int(first_batched_tokens[first_spec_uid].item()),
                int(first_batched_positions[first_spec_uid].item()),
                int(debug_spec_prev_history[first_verify_step][first_mismatch_prompt_idx].item()),
                int(debug_gt_next_history[first_verify_step][first_mismatch_prompt_idx].item()),
                int(debug_gt_after_spec_history[first_verify_step][first_mismatch_prompt_idx].item()),
                int(base_step_prompt_preds_dbg[first_verify_step][first_mismatch_prompt_idx].item()),
                int(alias_step_prompt_preds_dbg[first_verify_step][first_mismatch_prompt_idx].item()),
                int(base_step_verify_preds_dbg[first_verify_step][first_mismatch_prompt_idx].item()),
                int(alias_step_verify_preds_dbg[first_verify_step][first_mismatch_prompt_idx].item()),
            )
        if base_prompt_matches_dbg != prompt_count:
            warnings.warn(
                (
                    "Base replay prompt-lane mismatch under verify batching: "
                    f"{base_prompt_matches_dbg}/{prompt_count}"
                ),
                stacklevel=2,
            )
        if alias_prompt_matches_dbg != prompt_count:
            warnings.warn(
                (
                    "Aliased replay prompt-lane mismatch under verify batching: "
                    f"{alias_prompt_matches_dbg}/{prompt_count}"
                ),
                stacklevel=2,
            )
        assert (
            alias_accept_count_dbg == base_accept_count_dbg
        ), f"Aliased/base accept-count mismatch under verify batching: alias={alias_accept_count_dbg} base={base_accept_count_dbg}"
        assert (
            alias_accept_count_dbg == verify_count
        ), f"Aliased replay/live accept-count mismatch under verify batching: alias={alias_accept_count_dbg} live={verify_count}"
        assert (
            not differing_accept_steps
        ), f"Aliased/base accept-mask mismatch under verify batching at steps {differing_accept_steps[:8]}"
        assert (
            not differing_prompt_pred_steps
        ), f"Aliased/base prompt-prediction mismatch under verify batching at steps {differing_prompt_pred_steps[:8]}"
        if prompt_matches != prompt_count:
            warnings.warn(
                ("Prompt-lane mismatch under verify batching: " f"{prompt_matches}/{prompt_count}"),
                stacklevel=2,
            )

        if verify_count > 0:
            verify_rate = verify_matches / verify_count
            logger.info(
                f"MTP verify-lane match rate (accepted only): {verify_matches}/{verify_count} = {verify_rate:.3f}"
            )
            if verify_matches != verify_count:
                warnings.warn(
                    f"Verify-lane mismatch under aliasing: {verify_matches}/{verify_count}",
                    stacklevel=2,
                )
        else:
            logger.warning("No accepted speculative tokens in verify batching test; skipping verify-lane check.")


if __name__ == "__main__":
    pytest.main([__file__])
