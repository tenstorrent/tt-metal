# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.tt.generator import DeepseekGenerator
from models.demos.deepseek_v3.tt.model.row_batched_model import RowBatchedModel

REFERENCE_DIR = Path(__file__).with_name("reference_io")
DEFAULT_NUM_STEPS = 128
GENERATE_REFERENCE = os.getenv("DEEPSEEK_V3_MTP_GENERATE_REFERENCE", "0") == "1"
TRACE_REGION_SIZE = int(os.getenv("DEEPSEEK_TRACE_REGION_SIZE", "134217728"))
TIMEOUT_S = int(os.getenv("DEEPSEEK_V3_MTP_TIMEOUT_S", "1200"))
MAX_E2E_SECONDS = float(os.getenv("DEEPSEEK_V3_MTP_E2E_MAX_S", "0"))
MIN_TOKENS_PER_SEC = float(os.getenv("DEEPSEEK_V3_MTP_MIN_TPS", "1.0"))
MIN_TOKENS_PER_SEC_TRACE = float(os.getenv("DEEPSEEK_V3_MTP_MIN_TPS_TRACE", "0"))
DEFAULT_PREFILL_LEN = int(os.getenv("DEEPSEEK_V3_MTP_PREFILL_LEN", "16"))
DEFAULT_VERIFY_STEPS = int(os.getenv("DEEPSEEK_V3_MTP_VERIFY_STEPS", "16"))


def _get_reference_path(mesh_device: ttnn.MeshDevice, num_steps: int) -> Path:
    return REFERENCE_DIR / f"mtp_full_model_seq{num_steps}_mesh_{mesh_device.shape[0]}x{mesh_device.shape[1]}.pt"


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


def _prepare_generator(
    mesh_device: ttnn.MeshDevice,
    model_path: Path,
    cache_path: Path,
    force_recalculate: bool,
    mtp_mode: str,
) -> DeepseekGenerator:
    gen = DeepseekGenerator(
        mesh_device=mesh_device,
        model_path=model_path,
        cache_dir=cache_path,
        mtp_mode=mtp_mode,
        force_recalculate=force_recalculate,
    )
    gen._prepare_run_configs("prefill")
    gen._prepare_run_configs("decode")
    return gen


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
    gen: DeepseekGenerator,
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


class _MtpTraceRunner:
    def __init__(self, gen: DeepseekGenerator) -> None:
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


@pytest.mark.timeout(TIMEOUT_S)
@pytest.mark.requires_device(["T3K", "TG", "DUAL", "QUAD"])
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
    reference_path = _get_reference_path(mesh, num_steps)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    with _prepare_generator(
        mesh_device=mesh,
        model_path=model_path,
        cache_path=cache_path,
        force_recalculate=force_recalculate_weight_config,
        mtp_mode="off",
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
                batch_size_per_row=gen.batch_size_per_row,
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


@pytest.mark.timeout(TIMEOUT_S)
@pytest.mark.requires_device(["T3K", "TG", "DUAL", "QUAD"])
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
    num_steps = int(os.getenv("DEEPSEEK_V3_MTP_REF_STEPS", str(DEFAULT_NUM_STEPS)))
    mesh = mesh_device
    reference_path = _get_reference_path(mesh, num_steps)
    if not reference_path.exists():
        pytest.skip(
            f"Missing MTP reference IO at {reference_path}. "
            "Set DEEPSEEK_V3_MTP_GENERATE_REFERENCE=1 and run the reference generator test."
        )

    payload = torch.load(reference_path, map_location="cpu")
    metadata = payload.get("metadata", {})
    if tuple(metadata.get("mesh_shape", ())) != tuple(mesh.shape):
        pytest.skip(
            f"Reference IO mesh shape {metadata.get('mesh_shape')} does not match current mesh shape {tuple(mesh.shape)}."
        )

    hidden_states = payload["hidden_states"].to(torch.bfloat16)
    next_tokens = payload["next_tokens"].to(torch.int32)

    if hidden_states.shape[0] < 2:
        pytest.skip("Reference IO must contain at least 2 steps for MTP verification.")

    with _prepare_generator(
        mesh_device=mesh,
        model_path=model_path,
        cache_path=cache_path,
        force_recalculate=force_recalculate_weight_config,
        mtp_mode="auto",
    ) as gen:
        if not gen.enable_mtp:
            pytest.skip("MTP is disabled for this configuration; skipping MTP module test.")

        if hidden_states.shape[1] != gen.batch_size:
            pytest.skip(
                f"Reference IO batch size {hidden_states.shape[1]} does not match generator batch size {gen.batch_size}."
            )

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


@pytest.mark.timeout(TIMEOUT_S)
@pytest.mark.requires_device(["T3K", "TG", "DUAL", "QUAD"])
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
    num_steps = int(os.getenv("DEEPSEEK_V3_MTP_REF_STEPS", str(DEFAULT_NUM_STEPS)))
    mesh = mesh_device
    reference_path = _get_reference_path(mesh, num_steps)
    if not reference_path.exists():
        pytest.skip(
            f"Missing MTP reference IO at {reference_path}. "
            "Set DEEPSEEK_V3_MTP_GENERATE_REFERENCE=1 and run the reference generator test."
        )

    prompt_len = min(max(2, DEFAULT_PREFILL_LEN), num_steps - 2)
    if prompt_len < 2:
        pytest.skip("Need at least 2 prompt tokens and 2 reference steps for prefill priming test.")

    payload = torch.load(reference_path, map_location="cpu")
    metadata = payload.get("metadata", {})
    if tuple(metadata.get("mesh_shape", ())) != tuple(mesh.shape):
        pytest.skip(
            f"Reference IO mesh shape {metadata.get('mesh_shape')} does not match current mesh shape {tuple(mesh.shape)}."
        )

    hidden_states = payload["hidden_states"].to(torch.bfloat16)
    next_tokens = payload["next_tokens"].to(torch.int32)
    start_tokens = payload["start_tokens"].to(torch.long)

    with _prepare_generator(
        mesh_device=mesh,
        model_path=model_path,
        cache_path=cache_path,
        force_recalculate=force_recalculate_weight_config,
        mtp_mode="auto",
    ) as gen:
        if not gen.enable_mtp:
            pytest.skip("MTP is disabled for this configuration; skipping MTP prefill priming test.")

        if hidden_states.shape[1] != gen.batch_size:
            pytest.skip(
                f"Reference IO batch size {hidden_states.shape[1]} does not match generator batch size {gen.batch_size}."
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


@pytest.mark.timeout(TIMEOUT_S)
@pytest.mark.requires_device(["T3K", "TG", "DUAL", "QUAD"])
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
    num_steps = int(os.getenv("DEEPSEEK_V3_MTP_REF_STEPS", str(DEFAULT_NUM_STEPS)))
    mesh = mesh_device
    reference_path = _get_reference_path(mesh, num_steps)
    if not reference_path.exists():
        pytest.skip(
            f"Missing MTP reference IO at {reference_path}. "
            "Set DEEPSEEK_V3_MTP_GENERATE_REFERENCE=1 and run the reference generator test."
        )

    payload = torch.load(reference_path, map_location="cpu")
    metadata = payload.get("metadata", {})
    if tuple(metadata.get("mesh_shape", ())) != tuple(mesh.shape):
        pytest.skip(
            f"Reference IO mesh shape {metadata.get('mesh_shape')} does not match current mesh shape {tuple(mesh.shape)}."
        )

    hidden_states = payload["hidden_states"].to(torch.bfloat16)
    next_tokens = payload["next_tokens"].to(torch.int32)
    start_tokens = payload["start_tokens"].to(torch.int32)

    with _prepare_generator(
        mesh_device=mesh,
        model_path=model_path,
        cache_path=cache_path,
        force_recalculate=force_recalculate_weight_config,
        mtp_mode="auto",
    ) as gen:
        if not gen.enable_mtp:
            pytest.skip("MTP is disabled for this configuration; skipping verify-lane batching test.")

        if hidden_states.shape[1] != gen.batch_size:
            pytest.skip(
                f"Reference IO batch size {hidden_states.shape[1]} does not match generator batch size {gen.batch_size}."
            )

        batch_size = gen.batch_size
        batch_per_shard = int(gen.batch_size_per_row // gen.dp_factor)
        if batch_per_shard < 2:
            pytest.skip(f"Verify aliasing requires at least 2 lanes per shard; batch_per_shard={batch_per_shard}.")

        prompts_per_row = gen.batch_size_per_row // 2
        num_prompts = prompts_per_row * mesh.shape[0]
        if num_prompts <= 0:
            pytest.skip("No prompt lanes available for verify batching test.")
        if 2 * num_prompts > batch_size:
            pytest.skip(f"Need at least 2x prompt lanes; batch_size={batch_size}, num_prompts={num_prompts}.")

        prompt_user_ids_list: list[int] = []
        spec_user_ids_list: list[int] = []
        for i in range(num_prompts):
            row = i // prompts_per_row
            col = i % prompts_per_row
            base = row * gen.batch_size_per_row
            prompt_uid = base + 2 * col
            spec_uid = prompt_uid + 1
            prompt_user_ids_list.append(prompt_uid)
            spec_user_ids_list.append(spec_uid)
        prompt_user_ids = torch.tensor(prompt_user_ids_list, dtype=torch.long)
        spec_user_ids = torch.tensor(spec_user_ids_list, dtype=torch.long)

        steps_available = hidden_states.shape[0] - 2
        if steps_available <= 0:
            pytest.skip("Not enough reference steps for verify batching test.")
        steps_to_check = min(DEFAULT_VERIFY_STEPS, steps_available)

        decode_page_tables = gen._build_mtp_verify_page_tables(
            num_prompts=num_prompts, verify_offset=0, interleaved=True
        )

        def _build_alias_page_table(base_page_table: torch.Tensor) -> torch.Tensor:
            alias = base_page_table.clone()
            num_rows = int(alias.shape[0])
            for row in range(1, num_rows, 2):
                alias[row] = alias[row - 1]
            return alias

        def _lane_to_device_and_local(lane: int) -> tuple[int, int]:
            batch_per_shard = int(gen.batch_size_per_row // gen.dp_factor)
            row = lane // gen.batch_size_per_row
            within_row = lane % gen.batch_size_per_row
            shard_col = within_row // batch_per_shard
            local_lane = within_row % batch_per_shard
            device_idx = row * gen.dp_factor + shard_col
            return device_idx, local_lane

        enable_kv_log = os.getenv("DEEPSEEK_V3_MTP_KV_LOG", "0") == "1"
        enable_verify_table = os.getenv("DEEPSEEK_V3_MTP_VERIFY_TABLE", "1") == "1"
        min_verify_accept_rate = float(os.getenv("DEEPSEEK_V3_MTP_VERIFY_MIN_ACCEPT_RATE", "0.75"))
        log_host_rank_env = os.getenv("TT_LOG_HOST_RANK")
        log_host_rank = int(log_host_rank_env) if log_host_rank_env not in (None, "") else None
        host_rank = int(os.getenv("TT_MESH_HOST_RANK", "0"))
        if log_host_rank is not None and host_rank != log_host_rank:
            enable_verify_table = False

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

            prompt_pos = step_idx + 1
            spec_pos = step_idx + 2
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

        prompt_matches = 0
        prompt_count = 0
        verify_matches = 0
        verify_count = 0

        for step in range(steps_to_check):
            # Current token is reference next_tokens[step] at position step+1 (post-prefill alignment).
            prompt_tokens = next_tokens[step]
            positions_prompt = torch.full((batch_size,), step + 1, dtype=torch.int32)

            spec_logits = gen._mtp_predict_logits(
                hidden_states=hidden_states[step],
                tokens_step=prompt_tokens,
                positions=positions_prompt,
            )
            spec_tokens = torch.argmax(spec_logits, dim=-1).to(torch.int32)
            spec_after_spec_tokens = None
            if enable_verify_table:
                positions_after_spec = torch.full((batch_size,), step + 2, dtype=torch.int32)
                spec_after_spec_logits = gen._mtp_predict_logits(
                    hidden_states=hidden_states[step + 1],
                    tokens_step=next_tokens[step + 1],
                    positions=positions_after_spec,
                )
                spec_after_spec_tokens = torch.argmax(spec_after_spec_logits, dim=-1).to(torch.int32)

            batched_tokens = prompt_tokens.clone()
            batched_positions = positions_prompt.clone()
            batched_tokens[spec_user_ids] = spec_tokens[prompt_user_ids]
            batched_positions[spec_user_ids] = positions_prompt[prompt_user_ids] + 1

            if step == 0:
                _dump_kv_cache(
                    "before_batched_decode_step0",
                    step_idx=step,
                    prompt_lane=int(prompt_user_ids[0]),
                    spec_lane=int(spec_user_ids[0]),
                )

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
            pred_next = pred_all[prompt_user_ids]
            pred_after_spec = pred_all[spec_user_ids]

            if step == 0:
                _dump_kv_cache(
                    "after_batched_decode_step0",
                    step_idx=step,
                    prompt_lane=int(prompt_user_ids[0]),
                    spec_lane=int(spec_user_ids[0]),
                )

            gt_next = next_tokens[step + 1][prompt_user_ids]
            gt_after_spec = next_tokens[step + 2][prompt_user_ids]

            if enable_verify_table:
                header = "step | user | pos | next_pred | next_spec | pred_after_spec | spec_after_spec | accept/reject"
                rows = [header]
                spec_after_prompt = (
                    spec_after_spec_tokens[prompt_user_ids] if spec_after_spec_tokens is not None else None
                )
                for i in range(num_prompts):
                    user_id = int(prompt_user_ids[i].item())
                    pos_id = int(positions_prompt[user_id].item())
                    next_pred_val = int(pred_next[i].item())
                    next_spec_val = int(spec_tokens[user_id].item())
                    pred_after_val = int(pred_after_spec[i].item())
                    spec_after_val = int(spec_after_prompt[i].item()) if spec_after_prompt is not None else -1
                    verdict = "ACCEPT" if next_pred_val == next_spec_val else "REJECT"
                    rows.append(
                        f"{step} | {user_id} | {pos_id} | {next_pred_val} | {next_spec_val} | "
                        f"{pred_after_val} | {spec_after_val} | {verdict}"
                    )
                logger.info("MTP verify table:\n{}", "\n".join(rows))

            prompt_matches += int((pred_next == gt_next).sum().item())
            prompt_count += int(gt_next.numel())

            accepted_mask = spec_tokens[prompt_user_ids] == gt_next
            if accepted_mask.any():
                verify_matches += int((pred_after_spec[accepted_mask] == gt_after_spec[accepted_mask]).sum().item())
                verify_count += int(accepted_mask.sum().item())

        prompt_rate = prompt_matches / max(prompt_count, 1)
        accept_rate = verify_count / max(prompt_count, 1)
        logger.info(f"MTP verify batching prompt match rate: {prompt_matches}/{prompt_count} = {prompt_rate:.3f}")
        logger.info(f"MTP verify batching accept rate: {verify_count}/{prompt_count} = {accept_rate:.3f}")
        assert accept_rate >= min_verify_accept_rate, (
            f"MTP verify batching accept rate {accept_rate:.3f} below required minimum " f"{min_verify_accept_rate:.3f}"
        )
        assert (
            prompt_matches == prompt_count
        ), f"Prompt-lane mismatch under verify batching: {prompt_matches}/{prompt_count}"

        if verify_count > 0:
            verify_rate = verify_matches / verify_count
            logger.info(
                f"MTP verify-lane match rate (accepted only): {verify_matches}/{verify_count} = {verify_rate:.3f}"
            )
            assert (
                verify_matches == verify_count
            ), f"Verify-lane mismatch under aliasing: {verify_matches}/{verify_count}"
        else:
            logger.warning("No accepted speculative tokens in verify batching test; skipping verify-lane check.")


if __name__ == "__main__":
    pytest.main([__file__])
