# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import getpass
import json
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file

import ttnn
from models.common.auto_compose import to_torch_auto_compose


def _get_git_metadata() -> dict[str, str | None]:
    repo_root = Path(__file__).resolve().parents[4]

    def _git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        value = result.stdout.strip()
        return value or None

    return {
        "git_branch": _git("branch", "--show-current"),
        "git_commit": _git("rev-parse", "HEAD"),
    }


class BitSculptTraceCollector:
    """Collect BitSculpt-compatible debug tensors from the TT DeepSeek path.

    The collector stores one CPU tensor per artifact key and writes the same
    on-disk contract used by ``../bit_sculpt``:

    - ``expert_routing.safetensors``
    - ``hidden_states.safetensors``
    - ``kv_cache.safetensors``
    - ``metadata.json``
    """

    def __init__(
        self,
        *,
        hf_config,
        mesh_device: ttnn.MeshDevice,
        model_id: str,
        prompt: str,
        token_ids: list[int],
        token_strings: list[str],
        save_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.hf_config = hf_config
        self.mesh_device = mesh_device
        self.mesh_shape = tuple(mesh_device.shape)
        self.model_id = model_id
        self.prompt = prompt
        self.token_ids = token_ids
        self.token_strings = token_strings
        self.save_dtype = save_dtype
        self.seq_len = len(token_ids)
        self._buffers: dict[str, torch.Tensor] = {}

    @property
    def n_layers(self) -> int:
        return int(self.hf_config.num_hidden_layers)

    @property
    def moe_layer_offset(self) -> int:
        return int(self.hf_config.first_k_dense_replace)

    @property
    def hidden_dim(self) -> int:
        return int(self.hf_config.hidden_size)

    @property
    def kv_dim(self) -> int:
        return int(self.hf_config.kv_lora_rank + self.hf_config.qk_rope_head_dim)

    @property
    def top_k(self) -> int:
        return int(self.hf_config.num_experts_per_tok)

    def capture_hidden(self, artifact: str, layer_idx: int, tensor: ttnn.Tensor) -> None:
        torch_tensor = self._to_torch_mesh(tensor, dims=(0, -1))
        self._store(f"{artifact}_layer_{layer_idx}", self._flatten_prefill_tensor(torch_tensor, self.hidden_dim))

    def capture_routing(
        self,
        layer_idx: int,
        topk_weights: ttnn.Tensor,
        topk_indices: ttnn.Tensor,
    ) -> None:
        weights_torch = self._to_torch_mesh(topk_weights, dims=(-2, 0))
        indices_torch = self._to_torch_mesh(topk_indices, dims=(-2, 0))

        self._store(
            f"expert_weights_layer_{layer_idx}",
            self._flatten_prefill_tensor(weights_torch, self.top_k),
        )
        self._store(
            f"expert_ids_layer_{layer_idx}",
            self._flatten_prefill_tensor(indices_torch, self.top_k).to(torch.int32),
        )

    def capture_compressed_kv(self, layer_idx: int, tensor: ttnn.Tensor) -> None:
        torch_tensor = to_torch_auto_compose(tensor, device=self.mesh_device)
        self._store(f"compressed_kv_layer_{layer_idx}", self._flatten_prefill_tensor(torch_tensor, self.kv_dim))

    def validate(self) -> None:
        hidden_expected = self.n_layers
        moe_expected = self.n_layers - self.moe_layer_offset

        for layer_idx in range(self.n_layers):
            for prefix in ("decoder_output", "post_mla_residual", "post_attn_norm", "compressed_kv"):
                key = f"{prefix}_layer_{layer_idx}"
                if key not in self._buffers:
                    raise AssertionError(f"Missing trace artifact: {key}")

        for layer_idx in range(self.moe_layer_offset, self.n_layers):
            for prefix in ("expert_ids", "expert_weights"):
                key = f"{prefix}_layer_{layer_idx}"
                if key not in self._buffers:
                    raise AssertionError(f"Missing trace artifact: {key}")

        decoder_count = sum(key.startswith("decoder_output_layer_") for key in self._buffers)
        post_mla_count = sum(key.startswith("post_mla_residual_layer_") for key in self._buffers)
        post_attn_count = sum(key.startswith("post_attn_norm_layer_") for key in self._buffers)
        kv_count = sum(key.startswith("compressed_kv_layer_") for key in self._buffers)
        expert_id_count = sum(key.startswith("expert_ids_layer_") for key in self._buffers)
        expert_weight_count = sum(key.startswith("expert_weights_layer_") for key in self._buffers)

        assert decoder_count == hidden_expected
        assert post_mla_count == hidden_expected
        assert post_attn_count == hidden_expected
        assert kv_count == hidden_expected
        assert expert_id_count == moe_expected
        assert expert_weight_count == moe_expected

    def save(self, output_dir: str | Path, run_tag: str) -> Path:
        run_dir = Path(output_dir) / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        routing_tensors: dict[str, torch.Tensor] = {}
        hidden_tensors: dict[str, torch.Tensor] = {}
        kv_tensors: dict[str, torch.Tensor] = {}

        for key, tensor in sorted(self._buffers.items()):
            if key.startswith("expert_ids_") or key.startswith("expert_weights_"):
                routing_tensors[key] = tensor
            elif key.startswith("compressed_kv_"):
                kv_tensors[key] = tensor
            else:
                hidden_tensors[key] = tensor

        if routing_tensors:
            save_file(routing_tensors, str(run_dir / "expert_routing.safetensors"))
        if hidden_tensors:
            save_file(hidden_tensors, str(run_dir / "hidden_states.safetensors"))
        if kv_tensors:
            save_file(kv_tensors, str(run_dir / "kv_cache.safetensors"))

        git_metadata = _get_git_metadata()
        metadata = {
            "prompt": self.prompt,
            "token_ids": self.token_ids,
            "token_strings": self.token_strings,
            "n_tokens": self.seq_len,
            "n_layers": self.n_layers,
            "moe_layer_offset": self.moe_layer_offset,
            "kv_lora_rank": int(self.hf_config.kv_lora_rank),
            "model_id": self.model_id,
            "hidden_dim": self.hidden_dim,
            "n_experts": int(self.hf_config.n_routed_experts),
            "top_k": self.top_k,
            "save_dtype": str(self.save_dtype),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "username": getpass.getuser(),
            "hostname": socket.gethostname(),
            "command_line": shlex.join(sys.argv),
            "git_branch": git_metadata["git_branch"],
            "git_commit": git_metadata["git_commit"],
        }
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return run_dir

    def _store(self, key: str, tensor: torch.Tensor) -> None:
        if key in self._buffers:
            raise AssertionError(f"Duplicate trace artifact captured: {key}")

        out = tensor.detach().cpu().contiguous()
        if out.is_floating_point():
            out = out.to(self.save_dtype)
        self._buffers[key] = out

    def _to_torch_mesh(self, tensor: ttnn.Tensor, dims: tuple[int, int]) -> torch.Tensor:
        return ttnn.to_torch(
            tensor,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.mesh_device,
                mesh_shape=self.mesh_shape,
                dims=dims,
            ),
        )

    def _flatten_prefill_tensor(self, tensor: torch.Tensor, feature_dim: int) -> torch.Tensor:
        if tensor.ndim < 1:
            raise ValueError(f"Expected at least rank-2 tensor, got shape {tuple(tensor.shape)}")
        flat = tensor.reshape(-1, tensor.shape[-1])
        return flat[: self.seq_len, :feature_dim]
