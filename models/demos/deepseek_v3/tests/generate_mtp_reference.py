# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.tests.test_mtp import (
    SKIP_IN_CI,
    TIMEOUT_S,
    TRACE_REGION_SIZE,
    _get_reference_dir,
    _get_start_token_id,
    _prepare_generator,
    _prepare_mtp_module_runner,
    _tt_hidden_from_torch,
    _tt_positions_from_torch,
    _tt_tokens_from_torch,
)
from models.demos.deepseek_v3.tt.decoder_block.moe_decoder_block_2d import MoEDecoderBlock2D
from models.demos.deepseek_v3.tt.embedding.embedding2d import Embedding2D
from models.demos.deepseek_v3.tt.lm_head1d import LMHead1D
from models.demos.deepseek_v3.tt.mtp import _has_distinct_buffer
from models.demos.deepseek_v3.tt.rms_norm.distributed_rms_norm import DistributedRMSNorm

GENERATE_MTP_INTERMEDIATE_REFERENCE = os.getenv("DEEPSEEK_V3_MTP_GENERATE_INTERMEDIATE_REFERENCE", "0") == "1"
DEFAULT_CAPTURE_USERS = int(os.getenv("DEEPSEEK_V3_MTP_CAPTURE_USERS", "64"))
DEFAULT_CAPTURE_STEPS = int(os.getenv("DEEPSEEK_V3_MTP_CAPTURE_STEPS", "128"))


class _MtpStepCapture(NamedTuple):
    decoder_input: torch.Tensor
    decoder_output: torch.Tensor
    spec_tokens: torch.Tensor


def _get_capture_reference_path(num_users: int, num_steps: int) -> Path:
    return _get_reference_dir() / f"mtp_intermediate_u{num_users}_seq{num_steps}.pt"


def _tt_to_host_batch(mesh_device: ttnn.MeshDevice, tensor: ttnn.Tensor) -> torch.Tensor:
    host = ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
    )
    host = host.squeeze(0).squeeze(0)
    if host.dim() == 3 and host.shape[0] == 1:
        host = host[0]
    return host


def _deallocate_rope_tensors(rope_tensors: dict[str, ttnn.Tensor] | None) -> None:
    if rope_tensors is None:
        return
    for key in ("cos_matrix", "sin_matrix"):
        tensor = rope_tensors.get(key)
        if tensor is not None:
            ttnn.deallocate(tensor)


def _capture_mtp_step(
    runner,
    hidden_states: torch.Tensor,
    token_ids: torch.Tensor,
    position_idxs: torch.Tensor,
) -> _MtpStepCapture:
    mesh_device = runner.mesh_device
    cfg = runner.model_run_config_decode["mtp"]
    ccl = cfg["ccl"]

    tt_hidden = _tt_hidden_from_torch(mesh_device, hidden_states, device=mesh_device)
    tt_tokens = _tt_tokens_from_torch(mesh_device, token_ids, device=mesh_device)
    tt_positions = _tt_positions_from_torch(mesh_device, position_idxs, device=mesh_device)

    rot_idxs = runner.rope_setup.get_rot_idxs(position_idxs)
    rope_tensors = runner.rope_setup.get_rot_mats_from_rot_idxs(rot_idxs)
    mtp_page_table = runner._get_mtp_page_table()

    token_emb = Embedding2D.forward_decode(tt_tokens, cfg["embedding"])

    hidden_norm_in = ttnn.to_memory_config(tt_hidden, **cfg["hidden_norm_reshard"])
    hidden_norm = DistributedRMSNorm.forward_decode(hidden_norm_in, cfg["hidden_norm"])
    if _has_distinct_buffer(hidden_norm_in, tt_hidden):
        ttnn.deallocate(hidden_norm_in)

    token_norm_in = ttnn.to_memory_config(token_emb, **cfg["token_norm_reshard"])
    token_norm = DistributedRMSNorm.forward_decode(token_norm_in, cfg["token_norm"])
    if _has_distinct_buffer(token_norm_in, token_emb):
        ttnn.deallocate(token_norm_in)
    ttnn.deallocate(token_emb)

    hidden_full = ttnn.experimental.all_gather_async(
        hidden_norm, **ccl.populate_all_gather_runtime_args(cfg["norm_all_gather"])
    )
    ttnn.deallocate(hidden_norm)
    token_full = ttnn.experimental.all_gather_async(
        token_norm, **ccl.populate_all_gather_runtime_args(cfg["norm_all_gather"])
    )
    ttnn.deallocate(token_norm)

    orig_hidden_full = hidden_full
    orig_token_full = token_full
    assert (
        hidden_full.shape[2] == token_full.shape[2]
    ), f"MTP hidden/token length mismatch: hidden_full.shape={hidden_full.shape} token_full.shape={token_full.shape}"

    concat_in = ttnn.concat([token_full, hidden_full], **cfg["concat"])
    ttnn.deallocate(hidden_full)
    ttnn.deallocate(token_full)
    if orig_hidden_full is not hidden_full:
        ttnn.deallocate(orig_hidden_full)
    if orig_token_full is not token_full:
        ttnn.deallocate(orig_token_full)

    eh_out = ttnn.linear(concat_in, **cfg["eh_proj"]["linear"])
    ttnn.deallocate(concat_in)

    decoder_in = ttnn.to_memory_config(eh_out, **cfg["decoder_input_reshard"])
    if _has_distinct_buffer(decoder_in, eh_out):
        ttnn.deallocate(eh_out)
    decoder_input_host = _tt_to_host_batch(mesh_device, decoder_in).to(torch.bfloat16)

    decoder_out = MoEDecoderBlock2D.forward_decode(
        decoder_in,
        tt_positions,
        cfg["decoder_block"],
        rope_tensors,
        mtp_page_table,
    )
    decoder_output_host = _tt_to_host_batch(mesh_device, decoder_out).to(torch.bfloat16)
    ttnn.deallocate(decoder_in)

    head_norm_in = ttnn.to_memory_config(decoder_out, **cfg["head_norm_reshard"])
    if _has_distinct_buffer(head_norm_in, decoder_out):
        ttnn.deallocate(decoder_out)
    head_norm_out = DistributedRMSNorm.forward_decode(head_norm_in, cfg["head_norm"])
    ttnn.deallocate(head_norm_in)

    head_full = ttnn.experimental.all_gather_async(
        head_norm_out, **ccl.populate_all_gather_runtime_args(cfg["head_all_gather"])
    )
    ttnn.deallocate(head_norm_out)
    logits_tt = LMHead1D.forward_decode(head_full, cfg["head"])
    logits_host = _tt_to_host_batch(mesh_device, logits_tt)
    spec_tokens = torch.argmax(logits_host, dim=-1).to(torch.int32)

    ttnn.deallocate(logits_tt)
    ttnn.deallocate(tt_hidden)
    ttnn.deallocate(tt_tokens)
    ttnn.deallocate(tt_positions)
    ttnn.deallocate(rot_idxs)
    _deallocate_rope_tensors(rope_tensors)

    return _MtpStepCapture(
        decoder_input=decoder_input_host,
        decoder_output=decoder_output_host,
        spec_tokens=spec_tokens,
    )


def _normalize_decode_hidden(hidden: torch.Tensor) -> torch.Tensor:
    hidden = hidden.squeeze(0).squeeze(0)
    if hidden.dim() == 3 and hidden.shape[0] == 1:
        hidden = hidden[0]
    if hidden.dim() != 2:
        raise RuntimeError(f"Unexpected base hidden shape after decode gather: {tuple(hidden.shape)}")
    return hidden.to(torch.bfloat16)


def _normalize_decode_logits(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.squeeze(0).squeeze(0)
    if logits.dim() != 2:
        raise RuntimeError(f"Unexpected base logits shape after decode gather: {tuple(logits.shape)}")
    return logits


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
    not GENERATE_MTP_INTERMEDIATE_REFERENCE,
    reason="Set DEEPSEEK_V3_MTP_GENERATE_INTERMEDIATE_REFERENCE=1 to generate MTP intermediate reference data.",
)
def test_generate_mtp_reference_with_intermediates(
    mesh_device,
    model_path,
    cache_path,
    force_recalculate_weight_config,
    set_deterministic_env,
):
    """Capture a base-stream-driven MTP reference artifact with decoder intermediates.

    The capture is aligned to logical sequence positions, not to the interleaved
    BASE/SPEC execution layout used by the end-to-end MTP decode path. This keeps
    one record per emitted sequence position for each captured request.
    """

    host_rank = int(os.getenv("TT_MESH_HOST_RANK", "0"))
    capture_users = DEFAULT_CAPTURE_USERS
    capture_steps = DEFAULT_CAPTURE_STEPS
    if capture_users <= 0:
        pytest.skip("Need DEEPSEEK_V3_MTP_CAPTURE_USERS > 0.")
    if capture_steps <= 0:
        pytest.skip("Need DEEPSEEK_V3_MTP_CAPTURE_STEPS > 0.")

    reference_path = _get_capture_reference_path(capture_users, capture_steps)
    _get_reference_dir().mkdir(parents=True, exist_ok=True)

    with _prepare_generator(
        mesh_device=mesh_device,
        model_path=model_path,
        cache_path=cache_path,
        force_recalculate=force_recalculate_weight_config,
        enable_mtp=False,
    ) as base_gen, _prepare_mtp_module_runner(
        mesh_device=mesh_device,
        model_path=model_path,
        cache_path=cache_path,
        force_recalculate=force_recalculate_weight_config,
    ) as mtp_runner:
        if capture_users > base_gen.batch_size:
            pytest.skip(
                f"Requested {capture_users} captured users, but the base generator batch size is only {base_gen.batch_size}."
            )
        if capture_users > mtp_runner.batch_size:
            pytest.skip(
                f"Requested {capture_users} captured users, but the MTP batch size is only {mtp_runner.batch_size}."
            )
        assert (
            base_gen.batch_size == mtp_runner.batch_size
        ), f"Base/MTP batch mismatch: base={base_gen.batch_size} mtp={mtp_runner.batch_size}"

        batch_size = base_gen.batch_size
        hidden_size = int(base_gen.hf_config.hidden_size)
        vocab_size = int(base_gen.hf_config.vocab_size)
        start_token_id = _get_start_token_id(base_gen.hf_config)

        initial_tokens = (torch.arange(batch_size, dtype=torch.long) + start_token_id) % vocab_size
        tokens_step = initial_tokens.clone()
        positions = torch.zeros((batch_size,), dtype=torch.int32)

        base_hidden_states = torch.empty((capture_steps, capture_users, hidden_size), dtype=torch.bfloat16)
        base_hidden_positions = torch.empty((capture_steps, capture_users), dtype=torch.int32)
        base_output_tokens = torch.empty((capture_steps, capture_users), dtype=torch.int32)
        base_output_positions = torch.empty((capture_steps, capture_users), dtype=torch.int32)
        mtp_input_tokens = torch.empty((capture_steps, capture_users), dtype=torch.int32)
        mtp_input_positions = torch.empty((capture_steps, capture_users), dtype=torch.int32)
        mtp_decoder_inputs = torch.empty((capture_steps, capture_users, hidden_size), dtype=torch.bfloat16)
        mtp_decoder_outputs = torch.empty((capture_steps, capture_users, hidden_size), dtype=torch.bfloat16)
        mtp_speculation_tokens = torch.empty((capture_steps, capture_users), dtype=torch.int32)
        mtp_speculation_positions = torch.empty((capture_steps, capture_users), dtype=torch.int32)

        for step in range(capture_steps):
            logits, hidden = base_gen._decode_step(
                tokens_step=tokens_step,
                positions=positions,
                return_hidden=True,
            )
            hidden_step = _normalize_decode_hidden(hidden)
            logits_step = _normalize_decode_logits(logits)
            base_next_tokens = torch.argmax(logits_step, dim=-1).to(torch.int32)

            mtp_positions = positions + 1
            mtp_capture = _capture_mtp_step(
                runner=mtp_runner,
                hidden_states=hidden_step,
                token_ids=base_next_tokens,
                position_idxs=mtp_positions,
            )

            user_slice = slice(0, capture_users)
            base_hidden_states[step] = hidden_step[user_slice]
            base_hidden_positions[step] = positions[user_slice]
            base_output_tokens[step] = base_next_tokens[user_slice]
            base_output_positions[step] = mtp_positions[user_slice]
            mtp_input_tokens[step] = base_next_tokens[user_slice]
            mtp_input_positions[step] = mtp_positions[user_slice]
            mtp_decoder_inputs[step] = mtp_capture.decoder_input[user_slice]
            mtp_decoder_outputs[step] = mtp_capture.decoder_output[user_slice]
            mtp_speculation_tokens[step] = mtp_capture.spec_tokens[user_slice]
            mtp_speculation_positions[step] = mtp_positions[user_slice] + 1

            tokens_step = base_next_tokens.to(torch.long)
            positions += 1
            base_gen.ccl.reset_sem_counters()
            mtp_runner.ccl.reset_sem_counters()

            logger.info(
                "Captured MTP intermediate reference step {} / {} sample_user0: base_pos={} base_token={} spec_pos={} spec_token={}",
                step + 1,
                capture_steps,
                int(base_output_positions[step, 0].item()),
                int(base_output_tokens[step, 0].item()),
                int(mtp_speculation_positions[step, 0].item()),
                int(mtp_speculation_tokens[step, 0].item()),
            )

        payload = {
            "metadata": {
                "mesh_shape": list(mesh_device.shape),
                "capture_users": capture_users,
                "capture_steps": capture_steps,
                "batch_size": batch_size,
                "hidden_size": hidden_size,
                "vocab_size": vocab_size,
                "start_token_id": start_token_id,
                "batch_size_per_row": base_gen.batch_size_per_row,
                "description": (
                    "Base-stream-driven MTP reference capture. "
                    "base_hidden_states are pre-norm base-model h[t], "
                    "base_output_tokens / mtp_input_tokens are base-model token[t+1], "
                    "and mtp_speculation_tokens are the MTP prediction for token[t+2]."
                ),
            },
            "captured_user_ids": torch.arange(capture_users, dtype=torch.int32),
            "start_tokens": initial_tokens[:capture_users].cpu(),
            "base_hidden_states": base_hidden_states.cpu(),
            "base_hidden_positions": base_hidden_positions.cpu(),
            "base_output_tokens": base_output_tokens.cpu(),
            "base_output_positions": base_output_positions.cpu(),
            "mtp_input_tokens": mtp_input_tokens.cpu(),
            "mtp_input_positions": mtp_input_positions.cpu(),
            "mtp_decoder_inputs": mtp_decoder_inputs.cpu(),
            "mtp_decoder_outputs": mtp_decoder_outputs.cpu(),
            "mtp_speculation_tokens": mtp_speculation_tokens.cpu(),
            "mtp_speculation_positions": mtp_speculation_positions.cpu(),
        }

        if host_rank == 0:
            tmp_path = reference_path.with_suffix(".tmp")
            torch.save(payload, tmp_path)
            tmp_path.replace(reference_path)
            logger.info(f"Saved MTP intermediate reference IO to {reference_path}")
        else:
            logger.info("MTP intermediate reference capture complete on host rank %d (skipping save).", host_rank)
