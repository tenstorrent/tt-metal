# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek V3 B1 model (host interface).

Orchestrates prefill (token-by-token prompt processing) and autoregressive decode
via injectable write/read callables. Caller provides write_fn(token_tensor) and
read_fn(output_tensor); e.g. Pipeline.write_token / Pipeline.read_output.

Algorithm (prefill-by-decode then generation):
  - Prefill: for i = 0..S-1, call with input_ids = x[i] (B, 1); device uses/updates
    cache; ignore logits for i < S-1. prefill() returns the last step output (logits
    in real decoder) so caller can sample y0.
  - Start generation: last_logits = prefill(prompt_tokens); y0 = sample(last_logits).
  - Generation loop: for t = 0,1,..., feed y[t] (B, 1) via decode_step(), get logits,
    sample y[t+1], repeat.

Input tensor shape (H2D):
  - Only (B, 1) is supported: one token per batch element per step. The embedding layer
    runs on device; the host sends token IDs (int32). Payload size is B * TOKEN_ID_BYTES.

Interface vs real decoder:
  - One write and one read per step. The real decoder will also need per-step
    position (cur_pos_tensor / kv_cache_write_index); the engine tracks position for
    when the protocol is extended.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from loguru import logger

import ttnn

# Token IDs are int32 over the socket; payload size per step is B * TOKEN_ID_BYTES.
TOKEN_ID_BYTES: int = 4

ACTIVATION_DIM = 7168

ACTIVATION_SIZE_BYTES = ACTIVATION_DIM * 2

# Socket page_size must be PCIe-aligned (see h2d_socket.cpp). Must match demo stage TOKEN_PAGE_SIZE_BYTES (64).
PCIE_PAGE_ALIGNMENT_BYTES: int = 64


def align_up(value: int, alignment: int) -> int:
    """Round value up to the next multiple of alignment."""
    return (value + alignment - 1) // alignment * alignment


def page_size_bytes(batch_size: int) -> int:
    """PCIe-aligned page (and FIFO) size in bytes for (B, 1) token IDs. Use for socket creation."""
    return align_up(batch_size * TOKEN_ID_BYTES, PCIE_PAGE_ALIGNMENT_BYTES)


def create_output_buffer(page_size_datums: int) -> ttnn.Tensor:
    """Allocate a host output tensor (1, page_size_datums) int32 for socket read_tensor."""
    torch_output = torch.zeros(1, page_size_datums, dtype=torch.int32)
    return ttnn.from_torch(torch_output, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)


def to_padded_input(
    token: torch.Tensor | ttnn.Tensor,
    batch_size: int,
    page_size_datums: int,
) -> ttnn.Tensor:
    """Copy (B, 1) token into a PCIe-aligned padded buffer for write_tensor."""
    if isinstance(token, ttnn.Tensor):
        token = ttnn.to_torch(token)
    torch_padded = torch.zeros(1, page_size_datums, dtype=torch.int32)
    torch_padded[0, :batch_size] = token.flatten()[:batch_size]
    return ttnn.from_torch(torch_padded, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)


class DeepSeekV3:
    """
    Host-side model interface for prefill and decode via injectable write/read.
    Tracks position for compatibility with the real decoder (position_ids, kv_cache_write_index).
    Caller manages I/O lifecycle (e.g. Pipeline.setup_and_run() or HostInterface.run/terminate).
    """

    def __init__(
        self,
        write_fn: Callable[[ttnn.Tensor], None],
        read_fn: Callable[[ttnn.Tensor], None],
        batch_size: int = 1,
        prev_rank: int | None = None,
        next_rank: int | None = None,
        outputs_tokens: bool = False,
    ) -> None:
        """
        Args:
            write_fn: Called with a token tensor (PCIe-aligned, page_size_bytes(batch_size)).
            read_fn: Called with an output tensor; implementation fills it (e.g. Pipeline.read_output).
            batch_size: Batch size B. Current implementation supports only B=1;
                payload size is B * TOKEN_ID_BYTES (int32).
            prev_rank: MPI rank of the upstream pipeline stage (None for the first stage).
            next_rank: MPI rank of the downstream pipeline stage (None for the last stage).
            outputs_tokens: If True, this stage outputs token IDs (not activations).
                Used for LM head and passthrough stages that produce small token payloads.
        """
        if batch_size != 1:
            raise ValueError(f"DeepSeekV3 currently supports only batch_size=1, got {batch_size}")
        self._write_fn = write_fn
        self._read_fn = read_fn
        self.batch_size = batch_size
        self._prev_rank = ttnn.Rank(prev_rank) if prev_rank is not None else None
        self._next_rank = ttnn.Rank(next_rank) if next_rank is not None else None

        self._activation_size_datums: int = ACTIVATION_SIZE_BYTES // TOKEN_ID_BYTES
        self._token_size_datums: int = PCIE_PAGE_ALIGNMENT_BYTES // TOKEN_ID_BYTES

        self._position: int = 0
        self._input_buffer = create_output_buffer(self._activation_size_datums)
        if outputs_tokens or not next_rank:
            self._output_buffer: ttnn.Tensor = create_output_buffer(self._token_size_datums)
        else:
            self._output_buffer: ttnn.Tensor = create_output_buffer(self._activation_size_datums)
        logger.debug(f"Creating DeepSeekV3 model with batch size {batch_size}")

    def prefill(
        self,
        prompt_tokens: list[ttnn.Tensor] | None,
        num_iterations: int | None = None,
        ref_hidden_states: torch.Tensor | None = None,
        ref_logits: torch.Tensor | None = None,
        layer_idx: int | None = None,
        logits_tensor_fn: Callable[[], ttnn.Tensor] | None = None,
        mesh_device: ttnn.MeshDevice | None = None,
        save_outputs_dir: str | Path | None = None,
        pcc_threshold: float = 0.90,
    ) -> ttnn.Tensor:
        """Prefill-by-decode with optional per-token validation and tensor dumping.

        Runs the normal pipeline prefill. When ref_hidden_states or ref_logits
        are provided, compares device outputs against reference per token.
        """
        from safetensors.torch import save_file

        from models.common.utility_functions import comp_pcc

        is_lm_head = logits_tensor_fn is not None
        validating = ref_hidden_states is not None or ref_logits is not None
        all_device_outputs = []

        def _run_one_iteration(token_idx: int):
            self._position += 1
            if self._next_rank is not None:
                ttnn.send_tensor(self._output_buffer, self._next_rank)

            if not validating:
                return

            if is_lm_head and ref_logits is not None:
                scores = logits_tensor_fn()
                device_logits = (
                    ttnn.to_torch(
                        scores,
                        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1),
                    )
                    .float()
                    .reshape(-1)
                )
                all_device_outputs.append(device_logits.unsqueeze(0))

                ref_row = ref_logits[token_idx].float().reshape(-1)
                min_len = min(device_logits.shape[0], ref_row.shape[0])
                passing, pcc = comp_pcc(ref_row[:min_len], device_logits[:min_len], pcc_threshold)
                max_abs_diff = (ref_row[:min_len] - device_logits[:min_len]).abs().max().item()
                device_argmax = int(device_logits.argmax().item())
                ref_argmax = int(ref_row.argmax().item())
                sampled_token = int(ttnn.to_torch(self._output_buffer).to(torch.int32).flatten()[0].item())

                logger.info(
                    f"  token {token_idx}: PCC={pcc:.6f} pass={passing} "
                    f"max_diff={max_abs_diff:.4f} "
                    f"device_sampled={sampled_token} "
                    f"device_argmax={device_argmax} ref_argmax={ref_argmax} "
                    f"match={device_argmax == ref_argmax}"
                )
            elif ref_hidden_states is not None:
                raw = ttnn.to_torch(self._output_buffer)
                device_output = raw.contiguous().view(torch.uint8).view(torch.bfloat16).float().reshape(-1)
                all_device_outputs.append(device_output)

                ref_row = ref_hidden_states[token_idx].float().reshape(-1)
                passing, pcc = comp_pcc(ref_row, device_output, pcc_threshold)
                max_abs_diff = (ref_row - device_output).abs().max().item()
                logger.info(
                    f"  Layer {layer_idx} token {token_idx}: "
                    f"PCC={pcc:.6f} pass={passing} max_diff={max_abs_diff:.4f}"
                )

        if prompt_tokens is not None:
            num_tokens = len(prompt_tokens)
            if validating:
                label = "LM head" if is_lm_head else f"Layer {layer_idx}"
                logger.info(f"{label}: running {num_tokens} tokens through pipeline")
            for i, token in enumerate(prompt_tokens):
                self._write_fn(token)
                self._read_fn(self._output_buffer)
                _run_one_iteration(i)
        else:
            assert self._prev_rank is not None, "Non-first stage must have a prev_rank"
            assert num_iterations is not None, "Non-first stage must be given num_iterations"
            num_tokens = num_iterations
            if validating:
                label = "LM head" if is_lm_head else f"Layer {layer_idx}"
                logger.info(f"{label}: running {num_tokens} tokens through pipeline")
            for i in range(num_iterations):
                ttnn.recv_tensor(self._input_buffer, self._prev_rank)
                self._write_fn(self._input_buffer)
                self._read_fn(self._output_buffer)
                _run_one_iteration(i)

        if validating and all_device_outputs and save_outputs_dir is not None:
            save_dir = Path(save_outputs_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            stacked = torch.cat(all_device_outputs) if is_lm_head else torch.stack(all_device_outputs)
            if is_lm_head:
                save_file({"logits": stacked}, str(save_dir / "logits.safetensors"))
                logger.info(f"Saved device logits {list(stacked.shape)} to {save_dir / 'logits.safetensors'}")
            else:
                key = f"decoder_output_layer_{layer_idx}"
                save_file({key: stacked}, str(save_dir / f"{key}.safetensors"))
                logger.info(f"Saved {key} {list(stacked.shape)} to {save_dir / f'{key}.safetensors'}")

    def decode_step(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """
        Single decode step: send input via write_fn, receive output via read_fn.
        Returns the output tensor. Increments position.

        Args:
            input_tensor: Token IDs (B, 1), torch or ttnn.

        Returns:
            Output tensor; valid data is first batch_size elements.
        """
        assert len(input_tensor.shape) == 2, f"Input tensor shape must be (B, 1), got {input_tensor.shape}"
        assert (
            input_tensor.shape[0] == self.batch_size
        ), f"Input tensor batch size must be {self.batch_size}, got {input_tensor.shape[0]}"

        padded_input = to_padded_input(input_tensor, self.batch_size, self._page_size_datums)
        self._write_fn(padded_input)
        self._read_fn(self._output_buffer)
        self._position += 1
        return self._output_buffer

    def prefill_with_trace(
        self,
        trace_dir: str | Path,
        layer_idx: int,
        pcc_threshold: float = 0.90,
        save_hidden_states_path: str | Path | None = None,
    ) -> dict:
        """Run a single stage across all trace tokens and validate the output.

        For each token in the trace, loads the previous layer's output as input,
        pushes it through the pipeline, reads back the result, and collects all
        outputs. After all tokens are processed, computes PCC across the full
        concatenated sequence.

        Args:
            trace_dir: Path to the debug trace directory containing hidden_states.safetensors.
            layer_idx: The decoder layer index this stage processes.
            pcc_threshold: Minimum PCC to pass validation.
            save_hidden_states_path: If set, saves device outputs to this safetensors file
                with key "decoder_output_layer_{layer_idx}".

        Returns:
            Dict with keys: layer, pcc, passing, max_abs_diff, num_tokens.
        """
        from safetensors.torch import safe_open, save_file

        from models.common.utility_functions import comp_pcc

        trace_dir = Path(trace_dir)
        trace_file = str(trace_dir / "hidden_states.safetensors")

        output_buffer = ttnn.from_torch(
            torch.zeros(1, ACTIVATION_DIM, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        prev_layer = layer_idx - 1
        with safe_open(trace_file, framework="pt") as f:
            all_trace_inputs = f.get_tensor(f"decoder_output_layer_{prev_layer}")
            all_trace_expected = f.get_tensor(f"decoder_output_layer_{layer_idx}")

        num_tokens = all_trace_inputs.shape[0]
        logger.info(f"Layer {layer_idx}: running {num_tokens} tokens from trace")

        all_device_outputs = []
        for token_idx in range(num_tokens):
            trace_input = all_trace_inputs[token_idx].unsqueeze(0)
            input_tensor = ttnn.from_torch(
                trace_input.to(torch.bfloat16).reshape(1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

            self._write_fn(input_tensor)
            self._read_fn(output_buffer)
            self._position += 1

            if self._next_rank is not None:
                ttnn.send_tensor(output_buffer, self._next_rank)

            device_output = ttnn.to_torch(output_buffer).reshape(-1).float()
            all_device_outputs.append(device_output)

            if (token_idx + 1) % 50 == 0 or token_idx == num_tokens - 1:
                logger.info(f"Layer {layer_idx}: completed token {token_idx + 1}/{num_tokens}")

        all_device_outputs_cat = torch.stack(all_device_outputs)
        device_flat = all_device_outputs_cat.reshape(-1)
        trace_flat = all_trace_expected.float().reshape(-1)

        passing, pcc = comp_pcc(trace_flat, device_flat, pcc_threshold)
        max_abs_diff = (trace_flat - device_flat).abs().max().item()

        logger.info(
            f"Layer {layer_idx} PCC (all {num_tokens} tokens): {pcc} (threshold={pcc_threshold}, pass={passing})"
        )
        logger.info(f"Layer {layer_idx} max abs diff: {max_abs_diff}")
        logger.info(f"Layer {layer_idx} trace max magnitude: {trace_flat.abs().max().item()}")
        logger.info(f"Layer {layer_idx} device max magnitude: {device_flat.abs().max().item()}")

        if save_hidden_states_path is not None:
            save_hidden_states_path = Path(save_hidden_states_path)
            save_hidden_states_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                {f"decoder_output_layer_{layer_idx}": all_device_outputs_cat},
                str(save_hidden_states_path),
            )
            logger.info(f"Saved hidden states [{num_tokens}, {ACTIVATION_DIM}] to {save_hidden_states_path}")

        return {
            "layer": layer_idx,
            "pcc": pcc,
            "passing": passing,
            "max_abs_diff": max_abs_diff,
            "num_tokens": num_tokens,
        }

    def prefill_with_trace_lmhead(
        self,
        trace_dir: str | Path,
        logits_file: str | Path,
        input_layer_idx: int,
        logits_tensor_fn: Callable[[], ttnn.Tensor],
        mesh_device: ttnn.MeshDevice,
        pcc_threshold: float = 0.90,
        save_logits_path: str | Path | None = None,
    ) -> dict:
        """Validate the LM head stage against reference logits.

        For each token, feeds the last decoder layer's output into the LM head,
        reads the on-device logits tensor back, compares per-token against
        reference, and prints per-token PCC + argmax.

        Args:
            trace_dir: Path to debug trace directory with hidden_states.safetensors.
            logits_file: Path to reference logits safetensors file.
            input_layer_idx: Decoder layer index whose output is the LM head input.
            logits_tensor_fn: Callable returning the on-device logits ttnn.Tensor.
            mesh_device: MeshDevice for ConcatMeshToTensor readback.
            pcc_threshold: Minimum PCC to pass.
            save_logits_path: If set, saves all device logits to this safetensors file.

        Returns:
            Dict with keys: stage, per_token_results, overall_pcc, overall_passing, num_tokens.
        """
        from safetensors.torch import safe_open, save_file

        from models.common.utility_functions import comp_pcc

        trace_dir = Path(trace_dir)
        trace_file = str(trace_dir / "hidden_states.safetensors")
        logits_file = str(logits_file)

        with safe_open(trace_file, framework="pt") as f:
            all_trace_inputs = f.get_tensor(f"decoder_output_layer_{input_layer_idx}")

        with safe_open(logits_file, framework="pt") as f:
            all_ref_logits = f.get_tensor("logits")

        num_tokens = all_trace_inputs.shape[0]
        vocab_size = all_ref_logits.shape[-1]
        logger.info(f"LM head: running {num_tokens} tokens (input=decoder_output_layer_{input_layer_idx})")
        logger.info(f"LM head: reference logits shape: {list(all_ref_logits.shape)}")

        all_device_logits = []
        per_token_results = []

        for token_idx in range(num_tokens):
            trace_input = all_trace_inputs[token_idx].unsqueeze(0)
            input_tensor = ttnn.from_torch(
                trace_input.to(torch.bfloat16).reshape(1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

            self._write_fn(input_tensor)
            self._read_fn(self._output_buffer)
            self._position += 1

            if self._next_rank is not None:
                ttnn.send_tensor(self._output_buffer, self._next_rank)

            scores_tensor = logits_tensor_fn()
            device_logits = (
                ttnn.to_torch(
                    scores_tensor,
                    mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1),
                )
                .float()
                .reshape(-1)
            )

            if token_idx == 0:
                logger.info(f"LM head: device logits shape: [{num_tokens}, {device_logits.shape[0]}]")

            ref_logits = all_ref_logits[token_idx].float().reshape(-1)
            min_len = min(device_logits.shape[0], ref_logits.shape[0])
            device_cmp = device_logits[:min_len]
            ref_cmp = ref_logits[:min_len]

            passing, pcc = comp_pcc(ref_cmp, device_cmp, pcc_threshold)
            max_abs_diff = (ref_cmp - device_cmp).abs().max().item()
            device_argmax = int(device_logits.argmax().item())
            ref_argmax = int(ref_logits.argmax().item())
            sampled_token = int(ttnn.to_torch(self._output_buffer).to(torch.int32).flatten()[0].item())

            logger.info(
                f"  token {token_idx}: PCC={pcc:.6f} pass={passing} "
                f"max_diff={max_abs_diff:.4f} "
                f"device_sampled={sampled_token} "
                f"device_argmax={device_argmax} ref_argmax={ref_argmax} "
                f"match={device_argmax == ref_argmax}"
            )

            per_token_results.append(
                {
                    "token_idx": token_idx,
                    "pcc": pcc,
                    "passing": passing,
                    "max_abs_diff": max_abs_diff,
                    "device_argmax": device_argmax,
                    "ref_argmax": ref_argmax,
                    "argmax_match": device_argmax == ref_argmax,
                }
            )
            all_device_logits.append(device_logits.unsqueeze(0))

        all_device_logits_cat = torch.cat(all_device_logits, dim=0)
        ref_flat = all_ref_logits[:num_tokens].float().reshape(-1)
        device_flat = all_device_logits_cat.reshape(-1)
        min_len = min(device_flat.shape[0], ref_flat.shape[0])
        overall_passing, overall_pcc = comp_pcc(ref_flat[:min_len], device_flat[:min_len], pcc_threshold)

        logger.info(f"{'=' * 80}")
        logger.info(f"LM head summary ({num_tokens} tokens):")
        logger.info(f"  Overall PCC: {overall_pcc} (threshold={pcc_threshold}, pass={overall_passing})")
        logger.info(f"  Argmax matches: {sum(r['argmax_match'] for r in per_token_results)}/{num_tokens}")
        logger.info(f"{'=' * 80}")

        if save_logits_path is not None:
            save_logits_path = Path(save_logits_path)
            save_logits_path.parent.mkdir(parents=True, exist_ok=True)
            save_file({"logits": all_device_logits_cat}, str(save_logits_path))
            logger.info(f"Saved device logits [{num_tokens}, {all_device_logits_cat.shape[1]}] to {save_logits_path}")

        return {
            "stage": "lm_head",
            "per_token_results": per_token_results,
            "overall_pcc": overall_pcc,
            "overall_passing": overall_passing,
            "num_tokens": num_tokens,
        }

    @property
    def position(self) -> int:
        """Current sequence position (number of tokens processed so far)."""
        return self._position
