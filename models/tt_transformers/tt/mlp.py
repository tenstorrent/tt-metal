# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import copy_to_buffer
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import Mode, pad_to_size
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup


class MLP(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        model_config,
        state_dict_prefix=None,
        prefetcher=None,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.dim = args.dim
        self.model_config = model_config
        self.layer_num = layer_num

        # Define the prefetcher object
        self.prefetcher = prefetcher

        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)
        torch_weight = lambda name: torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
        pad_hidden_dim = lambda tensor, dim: pad_to_size(tensor, dim=dim, size=args.hidden_dim)
        # If padding was applied (e.g. via env var), add the unpadded hidden dim to the cache name to avoid loading incorrect weights
        hidden_dim_string = f".hidden_dim_{args.hidden_dim}" if args.hidden_dim != args.unpadded_hidden_dim else ""

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}{hidden_dim_string}"

        w1_w3_mem_config = args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)
        w2_mem_config = args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim)

        # TODO Clean up this code. With sharding, we load the normal weights and then shard them
        # Note: unsqueeze(0).unsqueeze(0) makes weights 4D [1, 1, H, W] to match attention weights
        # This is required for the dram_prefetcher to correctly interpret all weights
        def as_sharded_tensor(name, type, dims):
            # First get the raw weight and transpose it
            raw_weight = torch_weight(name[:2])  # This is 2D: [H, W]
            # Pad if needed
            padded_weight = pad_hidden_dim(raw_weight, dims[0] if args.is_galaxy else dims[-1])
            # Make 4D: [1, 1, H, W] - CRITICAL for prefetcher to work correctly
            torch_tensor = padded_weight.unsqueeze(0).unsqueeze(0)

            result = ttnn.as_tensor(
                torch_tensor,
                dtype=type,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=args.cluster_shape),
                layout=ttnn.TILE_LAYOUT,
                memory_config=(
                    ttnn.DRAM_MEMORY_CONFIG if args.is_galaxy else w2_mem_config if "w2" in name else w1_w3_mem_config
                ),
                cache_file_name=cache_name(name),
            )
            return result

        # Sharded weights
        w1_dims = (-1, -2) if args.is_galaxy else (-2, -1)
        w2_dims = (-2, -1) if args.is_galaxy else (-1, -2)

        layer_num = max(layer_num, 0)  # cross_block uses the configuration of the first decoder

        # When prefetcher is enabled, use consistent dtypes across all layers to avoid
        # race conditions caused by different block sizes
        use_prefetcher = prefetcher is not None

        self.decoders_optimizations = self.args.decoders_optimizations

        self.ff1_3_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.FF1_FF3, prefetcher=use_prefetcher
        )
        self.ff2_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.FF2, prefetcher=use_prefetcher
        )

        self.w1 = as_sharded_tensor(
            "w1_sharded", self.ff1_3_dtype, dims=w1_dims
        )  # bfp4 normally ok here but sub .99 pcc for llama 3.1 weights
        self.w2 = as_sharded_tensor("w2_sharded", self.ff2_dtype, dims=w2_dims)
        self.w3 = as_sharded_tensor("w3_sharded", self.ff1_3_dtype, dims=w1_dims)

        # Decode weights for the DRAM-core tensor prefetcher (receiver-contiguous layout, see dram_core_prefetch.py)
        self.dcp = getattr(args, "dram_core_prefetch", None)
        if self.dcp is not None:

            def rc_weight(name, dtype, dims):
                raw = pad_hidden_dim(torch_weight(name), dims[-1]).unsqueeze(0).unsqueeze(0)
                return self.dcp.make_weight(raw, dtype, cache_name(f"{name}_rc64{hidden_dim_string}"))

            self.w1_rc = rc_weight("w1", self.ff1_3_dtype, w1_dims)
            self.w3_rc = rc_weight("w3", self.ff1_3_dtype, w1_dims)
            self.w2_rc = rc_weight("w2", self.ff2_dtype, w2_dims)
            self.dcp.register("w1", self.w1_rc, args.dim, args.hidden_dim)
            self.dcp.register("w3", self.w3_rc, args.dim, args.hidden_dim)
            self.dcp.register("w2", self.w2_rc, args.hidden_dim, args.dim)

        # Decode-only hidden-dim chunks for the DRAM prefetcher when a whole w2 block does not fit L1 (see
        # ModelArgs.mlp_hidden_split_for_prefetcher). Prefill keeps using the unsplit tensors above.
        self.hidden_split = args.mlp_hidden_split_for_prefetcher(prefetcher)
        self.w_split = []
        if self.hidden_split:
            assert not args.is_galaxy and args.num_devices == 1, "MLP hidden split is only wired for one device"

            def as_split_tensor(name, dtype, start, size, split_cols):
                raw = torch_weight(name[:2])  # [dim, hidden] for w1/w3, [hidden, dim] for w2
                raw = pad_hidden_dim(raw, -1 if split_cols else -2)
                chunk = raw[:, start : start + size] if split_cols else raw[start : start + size, :]
                mem = (
                    args.create_dram_sharded_mem_config(args.dim, size)
                    if split_cols
                    else args.create_dram_sharded_mem_config(size, args.dim)
                )
                return ttnn.as_tensor(
                    chunk.unsqueeze(0).unsqueeze(0),
                    dtype=dtype,
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(-2, -1), mesh_shape=args.cluster_shape),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=mem,
                    cache_file_name=cache_name(f"{name}.split_{start}_{size}"),
                )

            start = 0
            for size in self.hidden_split:
                self.w_split.append(
                    {
                        "h": size,
                        "w1": as_split_tensor("w1_sharded", self.ff1_3_dtype, start, size, True),
                        "w3": as_split_tensor("w3_sharded", self.ff1_3_dtype, start, size, True),
                        "w2": as_split_tensor("w2_sharded", self.ff2_dtype, start, size, False),
                    }
                )
                start += size

        # Default activation is SILU
        self.activation_type = (
            args.mlp_activation_type if hasattr(args, "mlp_activation_type") else ttnn.UnaryOpType.SILU
        )

        # Insert the tensors into the prefetcher if it is used
        if self.prefetcher is not None:

            def register_weights():
                if self.w_split:
                    # Same order the split decode path consumes them: per chunk w1, w3, then w2
                    for ws in self.w_split:
                        self.prefetcher.insert_tensor(ws["w1"])
                        self.prefetcher.insert_tensor(ws["w3"])
                        self.prefetcher.insert_tensor(ws["w2"])
                    return
                self.prefetcher.insert_tensor(self.w1)
                self.prefetcher.insert_tensor(self.w3)
                self.prefetcher.insert_tensor(self.w2)

            self.prefetcher.register_callback(register_weights)

    def update(
        self,
        *,
        gate_proj: ttnn.Tensor,
        up_proj: ttnn.Tensor,
        down_proj: ttnn.Tensor,
    ) -> None:
        """In-place replace the on-device MLP weights via ``ttnn.copy``.

        HF-format input (see ``LLAMA_WEIGHT_TRANSFER.md``): ``gate_proj``,
        ``up_proj`` are ``(1, 1, I, H)`` and ``down_proj`` is ``(1, 1, H, I)``
        (HF Linear wrapped in two unit dims; ``H=args.dim``, ``I=hidden_dim``),
        bf16, TILE, DRAM-interleaved, replicated.

        Internal storage is the HF weight transposed; ``update`` transposes each
        input on device (mirroring the constructor's ``torch.transpose``) then
        ``copy_to_buffer``s into the existing buffers, preserving addresses (so
        captured traces and the prefetcher's recorded addresses stay valid).

        Caveats: hidden-dim padding is not handled (asserted off for
        Llama-3.2-1B-Instruct); the multi-chip replicated -> 2D-sharded mesh
        projection is not inserted (a no-op on the 1x1 transfer case).
        """
        assert not self.w_split, "MLP.update does not yet refresh the prefetcher hidden-dim split copies of w1/w2/w3"
        assert self.args.num_devices == 1, (
            f"MLP.update for num_devices > 1 is not yet implemented "
            f"(got num_devices={self.args.num_devices}); w1/w2/w3 are "
            "2D-sharded on a mesh and need a ttnn.mesh_partition into the "
            "sharded layout before copy."
        )
        assert self.args.hidden_dim == self.args.unpadded_hidden_dim, (
            f"MLP.update does not yet support hidden_dim padding "
            f"(hidden_dim={self.args.hidden_dim}, "
            f"unpadded_hidden_dim={self.args.unpadded_hidden_dim}); pad on the "
            "caller side or extend update() with an on-device ttnn.pad."
        )

        w1_internal = ttnn.transpose(gate_proj, -2, -1)
        w3_internal = ttnn.transpose(up_proj, -2, -1)
        w2_internal = ttnn.transpose(down_proj, -2, -1)

        copy_to_buffer(w1_internal, self.w1, self.ff1_3_dtype)
        copy_to_buffer(w3_internal, self.w3, self.ff1_3_dtype)
        copy_to_buffer(w2_internal, self.w2, self.ff2_dtype)

    def _forward_decode_hidden_split(self, x, activation_dtype, cfg_ff1_3, cfg_ff2, w2_output_dtype):
        """Decode MLP through the DRAM prefetcher ring with the hidden dim in chunks: per chunk
        silu(x @ w1_c) * (x @ w3_c) @ w2_c, partial outputs summed. Single device only (no CCL)."""
        pf = self.prefetcher
        out = None
        for ws in self.w_split:
            h = ws["h"]
            pc13 = self.args.get_mlp_ff1_3_prg_config(Mode.DECODE, 1, pf, hidden_chunk=h)
            mem13 = self.args.get_mlp_ff1_3_mem_config(Mode.DECODE, pf, hidden_chunk=h)
            w1_out = ttnn.linear(
                x,
                ws["w1"],
                dtype=activation_dtype or ttnn.bfloat16,
                compute_kernel_config=cfg_ff1_3,
                program_config=pc13,
                memory_config=mem13,
                global_cb=pf.global_cb,
                sub_device_id=pf.worker_sub_device_id,
            )
            w3_out = ttnn.linear(
                x,
                ws["w3"],
                dtype=activation_dtype or ttnn.bfloat16,
                compute_kernel_config=cfg_ff1_3,
                program_config=pc13,
                memory_config=mem13,
                global_cb=pf.global_cb,
                sub_device_id=pf.worker_sub_device_id,
            )
            w2_in = ttnn.mul(
                w1_out,
                w3_out,
                input_tensor_a_activations=[self.activation_type],
                dtype=activation_dtype or ttnn.bfloat8_b,
                memory_config=w1_out.memory_config(),
            )
            ttnn.deallocate(w3_out)
            ttnn.deallocate(w1_out)
            w2_out = ttnn.linear(
                w2_in,
                ws["w2"],
                compute_kernel_config=cfg_ff2,
                dtype=w2_output_dtype,
                program_config=self.args.get_mlp_ff2_prg_config(Mode.DECODE, 1, pf, hidden_chunk=h),
                memory_config=self.args.get_mlp_ff2_mem_config(Mode.DECODE, pf),
                global_cb=pf.global_cb,
                sub_device_id=pf.worker_sub_device_id,
            )
            ttnn.deallocate(w2_in)
            if out is None:
                out = w2_out
            else:
                summed = ttnn.add(out, w2_out, memory_config=out.memory_config())
                ttnn.deallocate(out)
                ttnn.deallocate(w2_out)
                out = summed
        ttnn.deallocate(x)
        out = ttnn.to_memory_config(out, self.args.get_mlp_output_mem_config(Mode.DECODE, pf))
        return out

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]
        chunk_size = self.model_config.get("MLP_PREFILL_CHUNK_SIZE", seq_len)
        if mode == Mode.PREFILL and seq_len > chunk_size:
            # The MLP is independent across tokens. Bound its gate/up/product intermediates
            # without splitting attention, which may require a complete sliding window.
            outputs = [
                self.forward(x[..., start : min(start + chunk_size, seq_len), :], mode)
                for start in range(0, seq_len, chunk_size)
            ]
            ttnn.deallocate(x)
            output = ttnn.concat(outputs, dim=-2, memory_config=outputs[0].memory_config())
            for chunk in outputs:
                ttnn.deallocate(chunk)
            return output
        TG = self.args.is_galaxy
        layer_num = max(self.layer_num, 0)  # cross_block uses the configuration of the first decoder
        activation_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
        )
        li_ff1_3_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_FF1_FF3, configuration=self.args
        )

        if mode == Mode.PREFILL and seq_len >= self.args.prefill_len_cutoff:  # 512 if Blackhole, 1024 if Wormhole
            # Reshape input to to fit on device and parallelize computation
            x = ttnn.reshape(x, [1, seq_len // self.args.prefill_len_cutoff, self.args.prefill_len_cutoff, -1])

        # In decode mode (seqlen <= 32) do DRAM sharded matmuls
        # These use HiFi2; this drops 1 bit of the activations but would be FLOP-bound on 12 cores with HiFi4
        if mode == Mode.DECODE and self.w_split:
            li_ff2_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
                decoder_id=layer_num, op=OpGroup.LI_FF2, configuration=self.args
            )
            return self._forward_decode_hidden_split(
                x,
                activation_dtype,
                li_ff1_3_compute_kernel_cfg,
                li_ff2_compute_kernel_cfg,
                activation_dtype or ttnn.bfloat16,
            )

        if mode == Mode.DECODE and not TG and self.prefetcher is None and self.dcp is not None:
            li_ff2_cfg = self.decoders_optimizations.get_math_fidelity(
                decoder_id=layer_num, op=OpGroup.LI_FF2, configuration=self.args
            )
            w1_out = self.dcp.linear(
                x,
                self.w1_rc,
                self.dim,
                self.args.hidden_dim,
                activation_dtype or ttnn.bfloat16,
                li_ff1_3_compute_kernel_cfg,
            )
            w3_out = self.dcp.linear(
                x,
                self.w3_rc,
                self.dim,
                self.args.hidden_dim,
                activation_dtype or ttnn.bfloat16,
                li_ff1_3_compute_kernel_cfg,
            )
            ttnn.deallocate(x)
            w2_in = ttnn.mul(
                w1_out,
                w3_out,
                input_tensor_a_activations=[self.activation_type],
                dtype=activation_dtype or ttnn.bfloat8_b,
                memory_config=w1_out.memory_config(),
            )
            ttnn.deallocate(w3_out)
            ttnn.deallocate(w1_out)
            w2_out = self.dcp.linear(
                w2_in, self.w2_rc, self.args.hidden_dim, self.dim, activation_dtype or ttnn.bfloat16, li_ff2_cfg
            )
            ttnn.deallocate(w2_in)
            return ttnn.to_memory_config(w2_out, self.args.get_mlp_output_mem_config(mode, self.prefetcher))

        pc_1 = self.args.get_mlp_ff1_3_prg_config(mode, seq_len, self.prefetcher)
        pc_2 = self.args.get_mlp_ff2_prg_config(mode, seq_len, self.prefetcher)
        pc_3 = self.args.get_mlp_ff1_3_prg_config(mode, seq_len, self.prefetcher)

        use_tg_decode_no_prefetch = TG and mode == Mode.DECODE and self.prefetcher is None
        if use_tg_decode_no_prefetch:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
            core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_1 else None,
            compute_kernel_config=li_ff1_3_compute_kernel_cfg,
            program_config=None if use_tg_decode_no_prefetch else pc_1,
            memory_config=(
                ttnn.DRAM_MEMORY_CONFIG
                if use_tg_decode_no_prefetch
                else self.args.get_mlp_ff1_3_mem_config(mode, self.prefetcher)
            ),
            global_cb=self.prefetcher.global_cb if self.prefetcher is not None and mode == Mode.DECODE else None,
            sub_device_id=self.prefetcher.worker_sub_device_id
            if self.prefetcher is not None and mode == Mode.DECODE
            else None,
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
            core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_3 else None,
            compute_kernel_config=li_ff1_3_compute_kernel_cfg,
            program_config=None if use_tg_decode_no_prefetch else pc_3,
            memory_config=(
                ttnn.DRAM_MEMORY_CONFIG
                if use_tg_decode_no_prefetch
                else self.args.get_mlp_ff1_3_mem_config(mode, self.prefetcher)
            ),
            global_cb=self.prefetcher.global_cb if self.prefetcher is not None and mode == Mode.DECODE else None,
            sub_device_id=self.prefetcher.worker_sub_device_id
            if self.prefetcher is not None and mode == Mode.DECODE
            else None,
        )
        ttnn.deallocate(x)

        if TG:
            # if mode == "decode" and self.dim!=8192:
            #     w1_out = ttnn.to_memory_config(w1_out, ttnn.DRAM_MEMORY_CONFIG)
            #     w3_out = ttnn.to_memory_config(w3_out, ttnn.DRAM_MEMORY_CONFIG)
            if self.dim == 8192 or mode == Mode.PREFILL:
                input_mem_cfg = w1_out.memory_config()

                cluster_axis = 1
                w1_out = ttnn.experimental.reduce_scatter_minimal_async(
                    w1_out,
                    persistent_output_buffers=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                    num_links=self.tt_ccl.get_num_links(cluster_axis),
                    cluster_axis=cluster_axis,
                    memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"] if mode == Mode.DECODE else None,
                    intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=self.args.ccl_topology(),
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )

                w3_out = ttnn.experimental.reduce_scatter_minimal_async(
                    w3_out,
                    persistent_output_buffers=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                    num_links=1,
                    cluster_axis=cluster_axis,
                    memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"] if mode == Mode.DECODE else None,
                    intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=self.args.ccl_topology(),
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )
            else:
                # NOTE: In MLP All-reduce hard codes to 2 links, so we do not get the dynamic link count from the CCL class
                # to avoid any performance regressions.
                w1_out = tt_all_reduce(
                    w1_out,
                    self.mesh_device,
                    self.tt_ccl,
                    cluster_axis=1,
                    num_all_gather_links=2,
                    sharded=mode == Mode.DECODE and not use_tg_decode_no_prefetch,
                    topology=self.args.ccl_topology(),
                    memory_config=(
                        ttnn.DRAM_MEMORY_CONFIG
                        if use_tg_decode_no_prefetch
                        else self.model_config["FF1_OUT_GATHERED_MEMCFG"]
                        if mode == Mode.DECODE
                        else None
                    ),
                )
                w3_out = tt_all_reduce(
                    w3_out,
                    self.mesh_device,
                    self.tt_ccl,
                    cluster_axis=1,
                    num_all_gather_links=2,
                    sharded=mode == Mode.DECODE and not use_tg_decode_no_prefetch,
                    topology=self.args.ccl_topology(),
                    memory_config=(
                        ttnn.DRAM_MEMORY_CONFIG
                        if use_tg_decode_no_prefetch
                        else self.model_config["FF1_OUT_GATHERED_MEMCFG"]
                        if mode == Mode.DECODE
                        else None
                    ),
                )

        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[self.activation_type],
            dtype=activation_dtype or ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )

        if (
            mode == Mode.DECODE
            and not TG
            and self.prefetcher is None
            and not (
                self.args.use_wide_dram_sharded_decode_grid()
                and (self.args.num_devices == 1 or self.args.pcie_tp_mlp_grids())
            )
        ):
            # w2 may use a different core grid, this is a no-op if they already match. On the wide P150 grid the
            # w1/w3 outputs already sit on 64 cores with 7-tile shards, which is what w2's in0 needs, and a
            # row-major 64-core activation streams as fast as the 8x8 grid, so the reshard is skipped.
            w2_in = ttnn.to_memory_config(w2_in, self.args.get_mlp_binary_mult_mem_config(mode))

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        if TG and (self.dim == 8192 or mode == Mode.PREFILL):
            cluster_axis = 1
            w2_in = ttnn.experimental.all_gather_async(
                w2_in,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
                num_links=2,
                cluster_axis=1,
                topology=self.args.ccl_topology(),
                memory_config=input_mem_cfg,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

            if mode == Mode.DECODE:
                w2_in = ttnn.to_memory_config(w2_in, ttnn.L1_MEMORY_CONFIG)

        li_ff2_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_FF2, configuration=self.args
        )

        w2_output_dtype = activation_dtype
        if w2_output_dtype is None:
            w2_output_dtype = ttnn.bfloat16

        if mode != Mode.DECODE and self.args.use_minimal_prefill_matmul(seq_len):
            is_qwen3_32b_t3k = self.args.base_model_name == "Qwen3-32B" and self.args.device_name == "T3K"
            if not is_qwen3_32b_t3k:
                # None makes minimal_matmul inherit w2_in.dtype. Only the
                # validated Qwen3-32B/T3K path overrides that default.
                w2_output_dtype = None

            w2_out = ttnn.experimental.minimal_matmul(
                w2_in,
                self.w2,
                dtype=w2_output_dtype,
                compute_kernel_config=li_ff2_compute_kernel_cfg,
                config=pc_2,
            )
        else:
            if TG:
                w2_output_dtype = self.args.ccl_dtype

            w2_out = ttnn.linear(
                w2_in,
                self.w2,
                compute_kernel_config=li_ff2_compute_kernel_cfg,
                dtype=w2_output_dtype,
                program_config=None if use_tg_decode_no_prefetch else pc_2,
                memory_config=(
                    ttnn.DRAM_MEMORY_CONFIG
                    if use_tg_decode_no_prefetch
                    else self.args.get_mlp_ff2_mem_config(mode, self.prefetcher)
                ),
                core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
                global_cb=self.prefetcher.global_cb if self.prefetcher is not None and mode == Mode.DECODE else None,
                sub_device_id=self.prefetcher.worker_sub_device_id
                if self.prefetcher is not None and mode == Mode.DECODE
                else None,
            )
        ttnn.deallocate(w2_in)

        w2_out_reduced = tt_all_reduce(
            w2_out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=0 if (TG and self.dim < 8192) else 3,
            sharded=mode == Mode.DECODE and not use_tg_decode_no_prefetch,
            memory_config=(
                ttnn.DRAM_MEMORY_CONFIG
                if use_tg_decode_no_prefetch
                else self.args.get_mlp_ff2_all_reduce_mem_config(mode, w2_out)
            ),
            rs_memory_config=self.model_config["MLP_RS_CONFIG"]["rs_memory_config"]
            if mode == Mode.DECODE
            else ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.args.ccl_dtype,
            use_composite=True if self.dim == 8192 else False,
            topology=self.args.ccl_topology(),
            chunks_per_sync=self.model_config["MLP_RS_CONFIG"]["chunks_per_sync"] if mode == Mode.DECODE else 10,
            num_workers_per_link=self.model_config["MLP_RS_CONFIG"]["num_workers_per_link"]
            if mode == Mode.DECODE
            else 2,
            subdevice_id=self.prefetcher.worker_sub_device_id
            if mode == Mode.DECODE and self.prefetcher is not None
            else None,
        )
        # Ensure dim 0 and 1 are 1
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

        if mode == Mode.DECODE:
            w2_out_reduced = ttnn.to_memory_config(
                w2_out_reduced,
                self.args.get_mlp_output_mem_config(mode, self.prefetcher),
            )

        return w2_out_reduced
