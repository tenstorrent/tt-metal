# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import Mode


class LMHead(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        tt_ccl,
        dtype,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        max_columns_per_device,  # too many columns per device lead to L1 OOM
        prefetcher=None,
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.dtype = dtype
        self.vocab_size = args.vocab_size
        self.padded_vocab_size = args.padded_vocab_size
        self.num_devices = args.num_devices
        self.prefetcher = prefetcher

        size_per_device = self.padded_vocab_size // self.num_devices

        tile_size = 32
        max_columns_per_device_ring_mm = math.ceil((max_columns_per_device) / tile_size) * tile_size
        max_columns_per_device_dram_sharded = max_columns_per_device

        self.model_config = args.get_model_config()

        num_splits_ring_mm = math.ceil(size_per_device / max_columns_per_device_ring_mm)
        num_splits_dram_sharded = math.ceil(size_per_device / max_columns_per_device_dram_sharded)

        self.split_sizes_dram_sharded = [min(size_per_device, max_columns_per_device_dram_sharded)] * (
            num_splits_dram_sharded - 1
        )
        self.split_sizes_dram_sharded.append(size_per_device - sum(self.split_sizes_dram_sharded))  # remaining columns

        self.split_sizes_ring_mm = [min(size_per_device, max_columns_per_device_ring_mm)] * (num_splits_ring_mm - 1)
        self.split_sizes_ring_mm.append(size_per_device - sum(self.split_sizes_ring_mm))  # remaining columns

        # Read the raw (vocab_size, dim) weight from the state dict.
        torch_output_weights = state_dict[f"{state_dict_prefix}output.weight"]

        def _dram_sharded_cache_fn(i: int, n_cols: int):
            if args.dummy_weights:
                return None
            return (
                weight_cache_path
                / f"output_lm_head_{len(self.split_sizes_dram_sharded)}_split_shard_{i}_{n_cols}_mode_0"
            )

        self.output_weights_dram_sharded = self._build_dram_sharded_output_weights(
            torch_output_weights,
            cache_file_name_fn=_dram_sharded_cache_fn,
        )

        # ring_mm mirror: only built when the prefetcher is enabled. Kept
        # inline because LMHead.update() does not currently support this
        # path (see _update_output_weights_ring_mm).
        self.output_weights_ring_mm = []
        if self.prefetcher is not None:
            permuted_padded = self._permute_and_pad_output_weights(torch_output_weights)
            for i, split_size in enumerate(self.split_sizes_ring_mm):
                device_splits = []
                for device in range(self.num_devices):
                    start = device * size_per_device + sum(self.split_sizes_ring_mm[:i])
                    end = start + split_size
                    device_splits.append(permuted_padded[:, start:end])
                combined_split = torch.cat(device_splits, dim=-1)

                cache_file_name = (
                    None
                    if args.dummy_weights
                    else weight_cache_path
                    / f"output_lm_head_{len(self.split_sizes_ring_mm)}_split_shard_{i}_{combined_split.shape[-1]}_mode_1"
                )

                def pad_to_power_of_2(n):
                    if n <= 0:
                        return 1
                    return 1 << (n - 1).bit_length()

                memory_config = args.create_dram_sharded_mem_config(
                    k=args.dim,
                    n=pad_to_power_of_2(math.ceil(combined_split.shape[-1] / self.num_devices)),
                    dram_grid=self.prefetcher.to_core_range_set(self.prefetcher.dram_banks()),
                )
                self.output_weights_ring_mm.append(
                    ttnn.as_tensor(
                        combined_split,
                        device=mesh_device,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
                        layout=ttnn.TILE_LAYOUT,
                        dtype=dtype,
                        memory_config=memory_config,
                        cache_file_name=cache_file_name,
                    )
                )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def _permute_and_pad_output_weights(self, torch_output_weights):
        """``state_dict``-shape (vocab_size, dim) -> column-wise (dim, padded_vocab_size).

        Mirrors the original constructor preamble: transpose to put the
        vocab axis last, then right-pad the vocab dim with zeros so it
        evenly divides across devices.
        """
        weights = torch_output_weights.permute(1, 0)
        if self.vocab_size < self.padded_vocab_size:
            padding_size = self.padded_vocab_size - self.vocab_size
            weights = torch.cat(
                [
                    weights,
                    torch.zeros(weights.shape[0], padding_size, dtype=weights.dtype),
                ],
                dim=-1,
            )
        return weights

    def _build_dram_sharded_output_weights(self, torch_output_weights, cache_file_name_fn=None):
        """Convert raw ``(vocab_size, dim)`` weight into the per-chunk
        DRAM-sharded ``ttnn.Tensor`` list that mirrors
        ``self.output_weights_dram_sharded``.

        Used by both ``__init__`` (to populate the buffers initially) and
        ``LMHead.update`` (to build replacement tensors which are then
        ``ttnn.copy``-d into the existing buffers).

        Args:
            torch_output_weights: torch tensor with shape
                ``(vocab_size, dim)``, i.e. exactly the layout of
                ``state_dict[f"{prefix}output.weight"]``.
            cache_file_name_fn: optional ``(chunk_idx, n_cols_after_concat)
                -> Path | None``. When called during ``update()`` this is
                ``None`` so no caching happens.

        Returns:
            list[ttnn.Tensor] of length ``len(self.split_sizes_dram_sharded)``,
            each shaped ``(dim, n_cols)`` and sharded along ``dim=-1`` via
            ``ShardTensorToMesh``.
        """
        permuted_padded = self._permute_and_pad_output_weights(torch_output_weights)
        size_per_device = self.padded_vocab_size // self.num_devices
        split_sizes = self.split_sizes_dram_sharded

        out = []
        for i, split_size in enumerate(split_sizes):
            device_splits = []
            for device in range(self.num_devices):
                start = device * size_per_device + sum(split_sizes[:i])
                end = start + split_size
                device_splits.append(permuted_padded[:, start:end])
            combined_split = torch.cat(device_splits, dim=-1)

            memory_config = self.args.create_dram_sharded_mem_config(
                k=self.args.dim,
                n=math.ceil(combined_split.shape[-1] / self.num_devices),
            )

            cache_file_name = (
                cache_file_name_fn(i, combined_split.shape[-1]) if cache_file_name_fn is not None else None
            )

            out.append(
                ttnn.as_tensor(
                    combined_split,
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
                    layout=ttnn.TILE_LAYOUT,
                    dtype=self.dtype,
                    memory_config=memory_config,
                    cache_file_name=cache_file_name,
                )
            )
        return out

    @staticmethod
    def _inplace_copy(src: ttnn.Tensor, dst: ttnn.Tensor, target_dtype) -> None:
        """Convert ``src`` to ``dst``'s layout/dtype/shape/memcfg, then
        ``ttnn.copy`` it into ``dst``. Mirrors ``Attention._inplace_copy``.
        """
        converted = src

        if converted.layout != dst.layout:
            converted = ttnn.to_layout(converted, layout=dst.layout)

        if converted.dtype != target_dtype:
            converted = ttnn.typecast(converted, dtype=target_dtype)

        if tuple(converted.shape) != tuple(dst.shape):
            converted = ttnn.reshape(converted, list(dst.shape))

        if converted.memory_config() != dst.memory_config():
            converted = ttnn.to_memory_config(converted, dst.memory_config())

        ttnn.copy(input_a=converted, input_b=dst)

    def _update_output_weights_dram_sharded(self, torch_output_weights) -> None:
        """In-place replace every chunk of ``self.output_weights_dram_sharded``.

        Rebuilds the per-chunk replacement tensors with no caching (so we
        don't poison the on-disk weight cache with whatever the user
        passed in), then ``ttnn.copy``-s each into the existing buffer.
        Buffer addresses are preserved across the update.
        """
        new_tensors = self._build_dram_sharded_output_weights(torch_output_weights, cache_file_name_fn=None)
        for new_t, dst in zip(new_tensors, self.output_weights_dram_sharded):
            self._inplace_copy(new_t, dst, self.dtype)

    def _update_output_weights_ring_mm(self, torch_output_weights) -> None:
        """In-place replace every chunk of ``self.output_weights_ring_mm``.

        The ring-mm path uses a different per-chunk split, a power-of-two
        padding on the column count, and a prefetcher-derived DRAM grid,
        so it is structurally distinct from the dram_sharded mirror and
        requires its own builder. Not yet implemented.
        """
        raise NotImplementedError("LMHead.update for output_weights_ring_mm (prefetcher path) is not yet implemented")

    def update(self, torch_output_weights) -> None:
        """In-place replace the on-device LM-head weights via ``ttnn.copy``.

        Accepts a single non-sharded torch tensor shaped exactly like
        ``state_dict[f"{state_dict_prefix}output.weight"]`` in
        ``__init__`` -- i.e. ``(vocab_size, dim)``. This method then does
        the same permute / pad / split / shard work the constructor does,
        and copies the resulting per-chunk tensors into the existing
        device buffers (no reallocation, so any captured trace remains
        valid).

        Updates ``self.output_weights_dram_sharded``. When the prefetcher
        is enabled the ring-mm mirror also has to be kept in sync; that
        path is not yet implemented (see ``_update_output_weights_ring_mm``).
        """
        self._update_output_weights_dram_sharded(torch_output_weights)
        if len(self.output_weights_ring_mm) > 0:
            self._update_output_weights_ring_mm(torch_output_weights)

    def forward(self, x: ttnn.Tensor, debug_input_torch=None, debug_weight_torch=None):
        outputs = []
        use_prefetcher = self.prefetcher is not None and self.prefetcher.mode == Mode.DECODE
        split_sizes = self.split_sizes_ring_mm if use_prefetcher else self.split_sizes_dram_sharded
        program_configs = [
            self.args.get_lm_head_program_config(split_size, self.prefetcher if use_prefetcher else None)
            for split_size in split_sizes
        ]

        output_weights = self.output_weights_ring_mm if use_prefetcher else self.output_weights_dram_sharded

        self.lm_head_output_memory_config = self.args.get_lm_head_output_mem_config(
            Mode.DECODE if use_prefetcher else Mode.PREFILL, self.prefetcher if use_prefetcher else None
        )

        for i, (weight, pc) in enumerate(zip(output_weights, program_configs)):
            output = ttnn.linear(
                x,
                weight,
                compute_kernel_config=self.compute_kernel_config,
                program_config=pc,
                memory_config=self.lm_head_output_memory_config,
                dtype=self.args.lm_head_dtype if hasattr(self.args, "lm_head_dtype") else ttnn.bfloat8_b,
                sub_device_id=self.prefetcher.worker_sub_device_id if use_prefetcher else None,
            )
            output = ttnn.to_memory_config(
                output,
                memory_config=self.args.get_lm_head_sharded_output_mem_config(
                    self.prefetcher if use_prefetcher else None
                ),
            )

            outputs.append(output)

        ttnn.deallocate(x)

        # Concatenate the outputs
        # outputs shape: a list of tensors, each tensor is 1,1,32,size_per_device per device
        output = ttnn.concat(
            outputs,
            dim=-1,
            memory_config=ttnn.L1_MEMORY_CONFIG if not use_prefetcher else ttnn.DRAM_MEMORY_CONFIG,
            sub_core_grids=self.prefetcher.all_worker_cores_range_set if use_prefetcher else None,
        )

        # Only use reshard mem config for ring_mm mode
        if use_prefetcher:
            output = ttnn.to_memory_config(
                output,
                memory_config=self.args.get_lm_head_reshard_mem_config(self.prefetcher),
            )

        output = tt_all_reduce(
            output,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=1,
            dim=3 if self.args.is_galaxy else 0,
            memory_config=output.memory_config(),
            dtype=self.args.ccl_dtype,
            sharded=False,
            use_composite=True,
            subdevice_id=self.prefetcher.worker_sub_device_id if use_prefetcher else None,
        )

        return output
