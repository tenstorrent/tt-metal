# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The vision tower on the mesh: one image's pixel patches -> merged visual features, replicated per die.

Every die runs the whole tower on the same input (no collectives); only the merger's second linear is column-sharded
over the mesh so a die produces exactly its hidden columns, the placement of the language model's embedding output
(per die ``[1, 1, T, 2560 / dies]`` BF16 TILE DRAM-interleaved, hidden-sharded over mesh axis 1, T = N / 4 merged
tokens in the processor's token order), so the splice into the prompt embedding is a per-die row copy.

Per image (eager ops, no trace: N varies): the host pads N to a tile multiple, computes the position embedding
(bilinear resample of the learned table), the rotary cos / sin in the device's interleaved layout and, when there is
padding or more than one frame, the attention windows; the device runs patch embed + position add, 27 blocks
(LayerNorm -> fused qkv -> heads -> rotary -> windowed non-causal SDPA -> concat -> proj, residual, LayerNorm -> fc1 with
GELU-tanh -> fc2, residual) and the merger (LayerNorm -> 2x2 group -> fc1 -> GELU -> column-sharded fc2).  Padded
patches attend only among themselves (their own window) and land in rows past T.  Weights are BF16 and resident
between ``load()`` and ``free()``; ``resident_bytes()`` is the MODELED footprint the admission ledger applies.
Numerics are measured against ``vision_reference.py`` in ``tests/test_vision_tower.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Mapping

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tools.checkpoint_budget import vision_resident_layout
from models.demos.blackhole.qwen38_flash_next.ttnn.vision_layout import TILE, device_cos_sin, padded_rows, tower_layout
from models.demos.blackhole.qwen38_flash_next.vision_reference import (
    VisionTowerConfig,
    attention_segments,
    interpolated_pos_embed,
    rotary_cos_sin,
    vision_position_ids,
    vision_tensor_shapes,
)

DRAM = ttnn.DRAM_MEMORY_CONFIG
# MEASURED on one p150 die (2026-09-29, six grids from 396 to 65,536 patches, the allocator sampled after every transient
# buffer): the tower's live activation peak is 17.3-17.7 KB per padded patch per die, at block 1's SDPA output (fused
# qkv + q/k/v heads + attention output alive together); 18.4 KB on the smallest padded grid.  The admission term rounds up.
PEAK_ACTIVATION_BYTES_PER_PATCH = 18_432
# Linears: BF16 operands, HiFi2 with FP32 accumulation.  LayerNorm, rotary and attention: HiFi4 with FP32 accumulation.
LINEAR_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)
EXACT_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


@dataclass
class VisionHostInputs:
    """One image's device inputs prepared on the host (FP32, tile-padded rows)."""

    pixels: torch.Tensor  # [1, 1, rows, patch_dim]
    position_embedding: torch.Tensor  # [1, 1, rows, hidden]
    cos: torch.Tensor  # [1, 1, rows, PADDED_HEAD_DIM]
    sin: torch.Tensor
    windows: torch.Tensor  # int32 cumulative attention window boundaries [0, .., patches(, rows)]
    patches: int
    rows: int
    tokens: int


@dataclass
class VisionTowerOutput:
    features: ttnn.Tensor  # per die [1, 1, rows / 4, out_hidden / dies]
    tokens: int  # real feature rows (T = N / 4); rows beyond hold the padding patches' merges
    patches: int
    rows: int
    timing: dict[str, float]
    trace: dict[str, object] | None = None  # {"embed": tensor, "blocks": [tensor per block]} when requested


@dataclass
class _Block:
    norm1_weight: ttnn.Tensor
    norm1_bias: ttnn.Tensor
    qkv_weight: ttnn.Tensor
    qkv_bias: ttnn.Tensor
    proj_weight: ttnn.Tensor
    proj_bias: ttnn.Tensor
    norm2_weight: ttnn.Tensor
    norm2_bias: ttnn.Tensor
    fc1_weight: ttnn.Tensor
    fc1_bias: ttnn.Tensor
    fc2_weight: ttnn.Tensor
    fc2_bias: ttnn.Tensor

    def tensors(self) -> list[ttnn.Tensor]:
        return [getattr(self, name) for name in self.__dataclass_fields__]


@dataclass
class _Weights:
    patch_weight: ttnn.Tensor
    patch_bias: ttnn.Tensor
    blocks: list[_Block]
    merger_norm_weight: ttnn.Tensor
    merger_norm_bias: ttnn.Tensor
    merger_fc1_weight: ttnn.Tensor
    merger_fc1_bias: ttnn.Tensor
    merger_fc2_weight: ttnn.Tensor
    merger_fc2_bias: ttnn.Tensor
    rotary_transformation: ttnn.Tensor
    uploaded: list[ttnn.Tensor] = field(default_factory=list)


def sdpa_chunk_size(rows: int) -> int:
    """The largest power of two that divides ``rows``, capped at 256 for long rows and 64 below 2048 (the prefill
    attention's empirical chunks); ``rows`` is a tile multiple so the result is at least 32."""

    chunk = 256 if rows >= 2048 else 64
    while rows % chunk:
        chunk //= 2
    return chunk


def _mesh_shape(mesh_device) -> tuple[int, int]:
    shape = tuple(int(value) for value in mesh_device.shape)
    if len(shape) != 2:
        raise ValueError(f"expected a two-dimensional mesh, got shape {shape}")
    return shape


class VisionTower:
    """The device tower over a host BF16 state dict (``Qwen38Checkpoint.vision_state_dict()``)."""

    def __init__(
        self, mesh_device, state_dict: Mapping[str, torch.Tensor], config: VisionTowerConfig = VisionTowerConfig()
    ):
        self.mesh_device = mesh_device
        self.config = config
        self.mesh_shape = _mesh_shape(mesh_device)
        self.dies = self.mesh_shape[1]
        if config.out_hidden_size % self.dies:
            raise ValueError(f"out_hidden_size {config.out_hidden_size} does not split over {self.dies} dies")
        expected = vision_tensor_shapes(config)
        if set(state_dict) != set(expected):
            raise ValueError("vision state dict names differ from the checkpoint's tower")
        for name, shape in expected.items():
            if tuple(state_dict[name].shape) != shape:
                raise ValueError(f"{name}: expected shape {shape}, got {tuple(state_dict[name].shape)}")
        self.state_dict = state_dict
        self.position_table = state_dict["pos_embed.weight"].to(torch.float32)
        grid = mesh_device.compute_with_storage_grid_size()
        self.core_grid = (int(grid.x), int(grid.y))
        self.scale = config.head_dim**-0.5
        self._replicate = ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig(
                [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], ttnn.MeshShape(*self.mesh_shape)
            ),
        )
        self._shard_columns = ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig(
                [ttnn.PlacementReplicate(), ttnn.PlacementShard(3)], ttnn.MeshShape(*self.mesh_shape)
            ),
        )
        self.weights: _Weights | None = None

    # ------------------------------------------------------------------------------------------------------------------
    # Residency
    # ------------------------------------------------------------------------------------------------------------------

    def resident_bytes(self) -> int:
        """MODELED BF16 bytes per die of the resident weights (``checkpoint_budget.vision_resident_layout``)."""

        return vision_resident_layout(mesh_size=self.dies)["device_total"]

    def dram_banks(self) -> int:
        return int(ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.DRAM).num_banks)

    def resident_bytes_per_bank(self) -> int:
        return -(-self.resident_bytes() // self.dram_banks())

    def peak_activation_bytes_per_bank(self, patches: int) -> int:
        """The transient DRAM an image of ``patches`` patches needs per bank while the tower runs (MEASURED law:
        ``PEAK_ACTIVATION_BYTES_PER_PATCH`` per padded row per die, spread over the banks)."""

        if patches <= 0:
            raise ValueError(f"patches must be positive, got {patches}")
        return -(-padded_rows(patches) * PEAK_ACTIVATION_BYTES_PER_PATCH // self.dram_banks())

    def _upload_weight(
        self, host: torch.Tensor, *, shard_columns: bool = False, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    ) -> ttnn.Tensor:
        while host.ndim < 4:
            host = host.unsqueeze(0)
        tensor = ttnn.from_torch(
            host,
            dtype=dtype,
            layout=layout,
            device=self.mesh_device,
            memory_config=DRAM,
            mesh_mapper=self._shard_columns if shard_columns else self._replicate,
        )
        if self.weights is not None:
            self.weights.uploaded.append(tensor)
        return tensor

    def load(self) -> int:
        """Upload every weight (BF16, DRAM interleaved, replicated; the merger's fc2 column-sharded); returns
        ``resident_bytes()``."""

        if self.weights is not None:
            return self.resident_bytes()
        layout = tower_layout(self.state_dict, self.config)
        self.weights = _Weights(
            patch_weight=None,
            patch_bias=None,
            blocks=[],
            merger_norm_weight=None,
            merger_norm_bias=None,
            merger_fc1_weight=None,
            merger_fc1_bias=None,
            merger_fc2_weight=None,
            merger_fc2_bias=None,
            rotary_transformation=None,
        )
        weights = self.weights
        weights.patch_weight = self._upload_weight(layout["patch_weight"])
        weights.patch_bias = self._upload_weight(layout["patch_bias"])
        for block in layout["blocks"]:
            weights.blocks.append(_Block(**{name: self._upload_weight(tensor) for name, tensor in block.items()}))
        weights.merger_norm_weight = self._upload_weight(layout["merger_norm_weight"])
        weights.merger_norm_bias = self._upload_weight(layout["merger_norm_bias"])
        weights.merger_fc1_weight = self._upload_weight(layout["merger_fc1_weight"])
        weights.merger_fc1_bias = self._upload_weight(layout["merger_fc1_bias"])
        weights.merger_fc2_weight = self._upload_weight(layout["merger_fc2_weight"], shard_columns=True)
        weights.merger_fc2_bias = self._upload_weight(layout["merger_fc2_bias"], shard_columns=True)
        weights.rotary_transformation = self._upload_weight(layout["rotary_transformation"])
        return self.resident_bytes()

    def free(self) -> None:
        if self.weights is None:
            return
        for tensor in self.weights.uploaded:
            ttnn.deallocate(tensor)
        self.weights = None

    def _require_weights(self) -> _Weights:
        if self.weights is None:
            raise RuntimeError("vision tower weights are not loaded; call load() first")
        return self.weights

    # ------------------------------------------------------------------------------------------------------------------
    # Host preparation
    # ------------------------------------------------------------------------------------------------------------------

    def prepare(self, pixel_patches: torch.Tensor, grid_thw: torch.Tensor, rows: int | None = None) -> VisionHostInputs:
        """The host side of one image.  ``rows`` (a tile multiple of at least the patch count; default the smallest)
        is the padded row count the device programs are compiled for: a served process passes its bucket."""

        config = self.config
        if pixel_patches.ndim != 2 or pixel_patches.shape[1] != config.patch_dim:
            raise ValueError(f"pixel patches must be [N, {config.patch_dim}], got {tuple(pixel_patches.shape)}")
        segments = attention_segments(grid_thw)
        patches = segments[-1][1]
        if pixel_patches.shape[0] != patches:
            raise ValueError(
                f"{pixel_patches.shape[0]} pixel patches do not match grid_thw {grid_thw.tolist()} ({patches})"
            )
        if patches % config.merge_unit:
            raise ValueError(f"{patches} patches are not a multiple of the merge unit {config.merge_unit}")
        rows = padded_rows(patches) if rows is None else int(rows)
        if rows < patches or rows % TILE:
            raise ValueError(f"rows {rows} must be a tile multiple of at least the {patches} patches")
        pixels = torch.zeros(rows, config.patch_dim, dtype=torch.float32)
        pixels[:patches] = pixel_patches.to(torch.float32)
        position = torch.zeros(rows, config.hidden_size, dtype=torch.float32)
        position[:patches] = interpolated_pos_embed(self.position_table, grid_thw, config)
        cos, sin = rotary_cos_sin(vision_position_ids(grid_thw, config), config)
        cos_dev, sin_dev = device_cos_sin(cos, sin, rows=rows)
        # The attention windows are always given (one segment per frame, the padded rows their own window), so a
        # row count has ONE compiled program set whether or not an image fills it: the prewarmed bucket serves
        # every image that pads to it.
        boundaries = [0] + [end for _, end in segments]
        if rows != patches:
            boundaries.append(rows)
        windows = torch.tensor(boundaries, dtype=torch.int32)
        return VisionHostInputs(
            pixels=pixels.reshape(1, 1, rows, config.patch_dim),
            position_embedding=position.reshape(1, 1, rows, config.hidden_size),
            cos=cos_dev,
            sin=sin_dev,
            windows=windows,
            patches=patches,
            rows=rows,
            tokens=patches // config.merge_unit,
        )

    def _upload_inputs(self, inputs: VisionHostInputs) -> dict[str, ttnn.Tensor | None]:
        return {
            "pixels": self._upload_activation(inputs.pixels),
            "position_embedding": self._upload_activation(inputs.position_embedding),
            "cos": self._upload_activation(inputs.cos),
            "sin": self._upload_activation(inputs.sin),
            "windows": (
                None
                if inputs.windows is None
                else ttnn.from_torch(
                    inputs.windows,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.mesh_device,
                    memory_config=DRAM,
                    mesh_mapper=self._replicate,
                )
            ),
        }

    def _upload_activation(self, host: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
        return ttnn.from_torch(
            host,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=DRAM,
            mesh_mapper=self._replicate,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Device stages
    # ------------------------------------------------------------------------------------------------------------------

    def _linear(
        self, x: ttnn.Tensor, weight: ttnn.Tensor, bias: ttnn.Tensor, activation: str | None = None
    ) -> ttnn.Tensor:
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            memory_config=DRAM,
            dtype=ttnn.bfloat16,
            compute_kernel_config=LINEAR_COMPUTE,
            activation=activation,
        )

    def _layer_norm(self, x: ttnn.Tensor, weight: ttnn.Tensor, bias: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(
            x,
            epsilon=self.config.layer_norm_eps,
            weight=weight,
            bias=bias,
            memory_config=DRAM,
            compute_kernel_config=EXACT_COMPUTE,
        )

    def embed_patches(self, pixels: ttnn.Tensor, position_embedding: ttnn.Tensor) -> ttnn.Tensor:
        weights = self._require_weights()
        projected = self._linear(pixels, weights.patch_weight, weights.patch_bias)
        hidden = ttnn.add(projected, position_embedding, memory_config=DRAM)
        ttnn.deallocate(projected)
        return hidden

    def attention_block(
        self,
        index: int,
        normed: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        windows: ttnn.Tensor | None,
        probe: Callable[[str], None] | None = None,
    ) -> ttnn.Tensor:
        weights = self._require_weights()
        block = weights.blocks[index]
        rows = normed.shape[-2]
        qkv = self._linear(normed, block.qkv_weight, block.qkv_bias)
        if probe is not None:
            probe(f"block{index}.qkv")
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_heads,
            transpose_k_heads=False,
            memory_config=DRAM,
        )
        ttnn.deallocate(qkv)
        if probe is not None:
            probe(f"block{index}.heads")
        q_rot = ttnn.experimental.rotary_embedding_llama(
            q, cos, sin, weights.rotary_transformation, is_decode_mode=False, compute_kernel_config=EXACT_COMPUTE
        )
        ttnn.deallocate(q)
        k_rot = ttnn.experimental.rotary_embedding_llama(
            k, cos, sin, weights.rotary_transformation, is_decode_mode=False, compute_kernel_config=EXACT_COMPUTE
        )
        ttnn.deallocate(k)
        chunk = sdpa_chunk_size(rows)
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.core_grid, q_chunk_size=chunk, k_chunk_size=chunk, exp_approx_mode=False
        )
        if probe is not None:
            probe(f"block{index}.rotary")
        attended = ttnn.transformer.scaled_dot_product_attention(
            q_rot,
            k_rot,
            v,
            is_causal=False,
            scale=self.scale,
            program_config=program_config,
            compute_kernel_config=EXACT_COMPUTE,
            memory_config=DRAM,
            cu_window_seqlens=windows,
        )
        if probe is not None:
            probe(f"block{index}.sdpa")
        ttnn.deallocate(q_rot)
        ttnn.deallocate(k_rot)
        ttnn.deallocate(v)
        concatenated = ttnn.experimental.nlp_concat_heads(attended, memory_config=DRAM)
        ttnn.deallocate(attended)
        output = self._linear(concatenated, block.proj_weight, block.proj_bias)
        ttnn.deallocate(concatenated)
        return output

    def mlp_block(self, index: int, normed: ttnn.Tensor, probe: Callable[[str], None] | None = None) -> ttnn.Tensor:
        block = self._require_weights().blocks[index]
        inner = self._linear(normed, block.fc1_weight, block.fc1_bias, activation="gelu_tanh")
        if probe is not None:
            probe(f"block{index}.fc1")
        output = self._linear(inner, block.fc2_weight, block.fc2_bias)
        ttnn.deallocate(inner)
        return output

    def tower_block(
        self,
        index: int,
        hidden: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        windows: ttnn.Tensor | None,
        probe: Callable[[str], None] | None = None,
    ) -> ttnn.Tensor:
        """One block on the BF16 residual stream (the measured stock form: an FP32 stream gained +0.0004 PCC at the merger
        for 10 % more time); ``hidden`` is consumed (deallocated); ``probe`` samples the allocator after the transient
        buffers of each step exist (the peak the admission needs)."""

        block = self._require_weights().blocks[index]
        normed = self._layer_norm(hidden, block.norm1_weight, block.norm1_bias)
        attended = self.attention_block(index, normed, cos, sin, windows, probe)
        ttnn.deallocate(normed)
        residual = ttnn.add(hidden, attended, memory_config=DRAM)
        ttnn.deallocate(attended)
        ttnn.deallocate(hidden)
        normed = self._layer_norm(residual, block.norm2_weight, block.norm2_bias)
        mlp = self.mlp_block(index, normed, probe)
        ttnn.deallocate(normed)
        output = ttnn.add(residual, mlp, memory_config=DRAM)
        ttnn.deallocate(mlp)
        ttnn.deallocate(residual)
        return output

    def merge_patches(self, hidden: ttnn.Tensor, probe: Callable[[str], None] | None = None) -> ttnn.Tensor:
        """``[1, 1, rows, hidden]`` -> per die ``[1, 1, rows / 4, out_hidden / dies]``."""

        weights = self._require_weights()
        config = self.config
        rows = hidden.shape[-2]
        normed = self._layer_norm(hidden, weights.merger_norm_weight, weights.merger_norm_bias)
        row_major = ttnn.to_layout(normed, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(normed)
        # In row-major layout the 2x2 group is a pure reinterpretation of the rows; the reshape may alias the buffer,
        # so the two handles are released by allocation state, never twice.
        grouped_rm = ttnn.reshape(row_major, (1, 1, rows // config.merge_unit, config.merged_hidden_size))
        grouped = ttnn.to_layout(grouped_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(grouped_rm)
        if row_major.is_allocated():
            ttnn.deallocate(row_major)
        inner = self._linear(grouped, weights.merger_fc1_weight, weights.merger_fc1_bias, activation="gelu")
        ttnn.deallocate(grouped)
        if probe is not None:
            probe("merger.fc1")
        features = self._linear(inner, weights.merger_fc2_weight, weights.merger_fc2_bias)
        ttnn.deallocate(inner)
        return features

    # ------------------------------------------------------------------------------------------------------------------
    # Whole image
    # ------------------------------------------------------------------------------------------------------------------

    def run_image(
        self,
        pixel_patches: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        trace: bool = False,
        probe: Callable[[str], None] | None = None,
        rows: int | None = None,
    ) -> VisionTowerOutput:
        """One image.  ``trace`` keeps the block-0 input and every block output on device for the tests; ``probe``
        is called with a stage name after each stage (allocator sampling); ``rows`` is the served row bucket."""

        started = time.perf_counter()
        inputs = self.prepare(pixel_patches, grid_thw, rows)
        prepared = time.perf_counter()
        device_inputs = self._upload_inputs(inputs)
        uploaded = time.perf_counter()
        cos, sin, windows = device_inputs["cos"], device_inputs["sin"], device_inputs["windows"]
        hidden = self.embed_patches(device_inputs["pixels"], device_inputs["position_embedding"])
        ttnn.deallocate(device_inputs["pixels"])
        ttnn.deallocate(device_inputs["position_embedding"])
        kept: dict[str, object] | None = {"embed": hidden, "blocks": []} if trace else None
        if probe is not None:
            probe("embed")
        for index in range(self.config.depth):
            block_input = hidden if not trace else ttnn.clone(hidden, memory_config=DRAM)
            hidden = self.tower_block(index, block_input, cos, sin, windows, probe)
            if trace:
                kept["blocks"].append(hidden)
            if probe is not None:
                probe(f"block{index}")
        features = self.merge_patches(hidden, probe)
        if not trace:
            ttnn.deallocate(hidden)
        for tensor in (cos, sin, windows):
            if tensor is not None:
                ttnn.deallocate(tensor)
        if probe is not None:
            probe("merger")
        ttnn.synchronize_device(self.mesh_device)
        finished = time.perf_counter()
        timing = {
            "host_prepare_s": prepared - started,
            "upload_s": uploaded - prepared,
            "device_s": finished - uploaded,
            "total_s": finished - started,
        }
        return VisionTowerOutput(features, inputs.tokens, inputs.patches, inputs.rows, timing, kept)

    # ------------------------------------------------------------------------------------------------------------------
    # Host views (tests, diagnostics)
    # ------------------------------------------------------------------------------------------------------------------

    def prewarm_rows(self, rows: int) -> dict[str, float]:
        """Compile the served program set of one row bucket: two zero images, one whose grid fills ``rows`` exactly
        (2 x rows/2 patches; the attention windows ``[0, rows]``) and one four patches short of it (2 x (rows/2 - 2);
        the windows ``[0, rows - 4, rows]``), since the windowed attention's program is keyed on the window tensor's
        shape as well as the rows: every image of the bucket is one of the two forms.  The outputs are released;
        returns the two forwards' timings summed."""

        if rows % (2 * TILE) or rows < 2 * TILE:
            raise ValueError(f"a prewarm bucket must be a positive multiple of {2 * TILE} rows, got {rows}")
        timing: dict[str, float] = {}
        for patches in (rows, rows - 4):
            grid_thw = torch.tensor([[1, 2, patches // 2]], dtype=torch.long)
            output = self.run_image(torch.zeros(patches, self.config.patch_dim), grid_thw, rows=rows)
            ttnn.deallocate(output.features)
            for key, value in output.timing.items():
                timing[key] = timing.get(key, 0.0) + value
        return timing

    def features_to_torch(self, output: VisionTowerOutput) -> torch.Tensor:
        """``[T, out_hidden]`` FP32: the dies' column shards concatenated in mesh order, the padding rows dropped."""

        shards = [ttnn.to_torch(shard).to(torch.float32) for shard in ttnn.get_device_tensors(output.features)]
        return torch.cat(shards, dim=-1).reshape(-1, self.config.out_hidden_size)[: output.tokens]

    def replicated_to_torch(self, tensor: ttnn.Tensor, rows: int | None = None) -> torch.Tensor:
        """Die 0's copy of a replicated ``[1, 1, rows, width]`` activation as ``[rows, width]`` FP32."""

        host = ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).to(torch.float32)
        host = host.reshape(-1, host.shape[-1])
        return host if rows is None else host[:rows]

    def run_block(self, index: int, hidden: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Block ``index`` alone on a host ``[N, hidden]`` input (the reference's, cast to BF16); ``[N, hidden]`` FP32."""

        rows = padded_rows(hidden.shape[0])
        padded = torch.zeros(rows, hidden.shape[1], dtype=torch.float32)
        padded[: hidden.shape[0]] = hidden.to(torch.float32)
        inputs = self.prepare(torch.zeros(hidden.shape[0], self.config.patch_dim), grid_thw)
        device_inputs = self._upload_inputs(inputs)
        device_hidden = self._upload_activation(padded.reshape(1, 1, rows, hidden.shape[1]))
        output = self.tower_block(
            index, device_hidden, device_inputs["cos"], device_inputs["sin"], device_inputs["windows"]
        )
        result = self.replicated_to_torch(output, hidden.shape[0])
        for tensor in (output, *(t for t in device_inputs.values() if t is not None)):
            ttnn.deallocate(tensor)
        return result

    def _padded_upload(self, host: torch.Tensor) -> tuple[ttnn.Tensor, int]:
        rows = padded_rows(host.shape[0])
        padded = torch.zeros(rows, host.shape[1], dtype=torch.float32)
        padded[: host.shape[0]] = host.to(torch.float32)
        return self._upload_activation(padded.reshape(1, 1, rows, host.shape[1])), rows

    def run_attention(self, index: int, normed: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Block ``index``'s attention alone on a host ``[N, hidden]`` LayerNorm output; ``[N, hidden]`` FP32."""

        inputs = self.prepare(torch.zeros(normed.shape[0], self.config.patch_dim), grid_thw)
        device_inputs = self._upload_inputs(inputs)
        device_normed, _ = self._padded_upload(normed)
        output = self.attention_block(
            index, device_normed, device_inputs["cos"], device_inputs["sin"], device_inputs["windows"]
        )
        result = self.replicated_to_torch(output, normed.shape[0])
        for tensor in (output, device_normed, *(t for t in device_inputs.values() if t is not None)):
            ttnn.deallocate(tensor)
        return result

    def run_mlp(self, index: int, normed: torch.Tensor) -> torch.Tensor:
        """Block ``index``'s MLP (fc1 with fused GELU-tanh, fc2) alone on a host ``[N, hidden]`` input; FP32."""

        device_normed, _ = self._padded_upload(normed)
        output = self.mlp_block(index, device_normed)
        result = self.replicated_to_torch(output, normed.shape[0])
        ttnn.deallocate(output)
        ttnn.deallocate(device_normed)
        return result

    def run_merger(self, hidden: torch.Tensor) -> torch.Tensor:
        """The merger alone on a host ``[N, hidden]`` input; ``[N / 4, out_hidden]`` FP32."""

        rows = padded_rows(hidden.shape[0])
        padded = torch.zeros(rows, hidden.shape[1], dtype=torch.float32)
        padded[: hidden.shape[0]] = hidden.to(torch.float32)
        device_hidden = self._upload_activation(padded.reshape(1, 1, rows, hidden.shape[1]))
        features = self.merge_patches(device_hidden)
        ttnn.deallocate(device_hidden)
        output = VisionTowerOutput(features, hidden.shape[0] // self.config.merge_unit, hidden.shape[0], rows, {})
        result = self.features_to_torch(output)
        ttnn.deallocate(features)
        return result
