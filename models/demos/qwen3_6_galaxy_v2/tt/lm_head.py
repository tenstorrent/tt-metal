# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class LMHead(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        dtype,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        max_columns_per_device=128256 // 4,  # larger values per device lead to OOM or hangs
        tt_ccl=None,
        prefetcher_setup=None,
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.vocab_size = args.vocab_size
        self.padded_vocab_size = args.padded_vocab_size
        self.num_devices = args.num_devices
        self.tt_ccl = tt_ccl

        size_per_device = self.padded_vocab_size // self.num_devices
        num_splits = math.ceil(size_per_device / max_columns_per_device)

        split_sizes = [min(size_per_device, max_columns_per_device)] * (num_splits - 1)
        split_sizes.append(size_per_device - sum(split_sizes))  # remaining columns

        # Split the output weights
        torch_output_weights = state_dict[f"{state_dict_prefix}output.weight"].permute(1, 0)

        self.output_weights = []
        self.output_weights_decode = []
        self.output_weights_prefill = []
        num_splits = 1
        cache_file_name_decode = (
            None
            if args.dummy_weights
            else weight_cache_path / f"output_lm_head_{num_splits}_split_shard_0_dram_width_sharded_decode"
        )
        cache_file_name_prefill = (
            None
            if args.dummy_weights
            else weight_cache_path / f"output_lm_head_{num_splits}_split_shard_0_dram_prefill"
        )
        padded_lm_head = torch.zeros(1, 1, args.dim, self.padded_vocab_size)
        padded_lm_head[:, :, :, : self.vocab_size] = torch_output_weights

        if args.is_70b:
            memory_config_decode = args.create_dram_sharded_mem_config_lm_head(
                k=args.dim // 4, n=self.padded_vocab_size // 8
            )
            memory_config_prefill = ttnn.DRAM_MEMORY_CONFIG
        else:
            # Qwen3-32B / Qwen3.6 / olmo: small-model branch.  When dim==2048
            # the lm-head is interleaved DRAM; otherwise it is DRAM-sharded
            # along K to match the 8-bank ring matmul.  Decode and prefill
            # share the same memcfg here (the 70B branch above splits them
            # because that path uses a dedicated lm_head DRAM sharding).
            shared_mc = (
                ttnn.DRAM_MEMORY_CONFIG
                if args.dim == 2048
                else args.create_dram_sharded_mem_config(k=args.dim // 4, n=self.padded_vocab_size // 8)
            )
            memory_config_decode = shared_mc
            memory_config_prefill = shared_mc

        for i in range(num_splits):
            index = i * self.padded_vocab_size // num_splits
            self.output_weights_decode.append(  # (2k, 16k) 128* 1024
                ttnn.as_tensor(
                    padded_lm_head[..., index : index + self.padded_vocab_size // num_splits],
                    device=mesh_device,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, 2), mesh_shape=args.cluster_shape),
                    layout=ttnn.TILE_LAYOUT,
                    dtype=dtype,
                    memory_config=memory_config_decode,
                    cache_file_name=cache_file_name_decode,
                )
            )
            self.output_weights_prefill.append(  # (2k, 16k) 128* 1024
                ttnn.as_tensor(
                    padded_lm_head[..., index : index + self.padded_vocab_size // num_splits],
                    device=mesh_device,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, 2), mesh_shape=args.cluster_shape),
                    layout=ttnn.TILE_LAYOUT,
                    dtype=dtype,
                    memory_config=memory_config_prefill,
                    cache_file_name=cache_file_name_prefill,
                )
            )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            dst_full_sync_en=True,
        )

        self.program_configs = [args.model_config["LM_HEAD_TG_RING_PROGCFG"]] * num_splits
        self.output_memory_config = args.model_config["LM_HEAD_OUT_RING_MEMCFG"]
        self.prefill_pc = args.model_config["LM_HEAD_PREFILL_PROGCFG"]

    def forward_on_host(self, x: ttnn.Tensor):
        x_torch = ttnn.to_torch(
            x,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.mesh_device,
                dims=(0, 3),
                mesh_shape=(8, 4),
            ),
        )  # [8, 1, 32, 2048 * 4]
        x_torch = x_torch[:1]

        weight_torch = ttnn.to_torch(
            self.output_weights[0],
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device=self.mesh_device, dims=(3, 2), mesh_shape=list(self.mesh_device.shape)
            ),
        )

        output_torch = torch.matmul(x_torch.float(), weight_torch.float())

        output = ttnn.as_tensor(
            output_torch,
            dtype=ttnn.bfloat8_b,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device=self.mesh_device, dims=(3, None), mesh_shape=list(self.mesh_device.shape)
            ),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        return [output]

    def forward(self, x: ttnn.Tensor, worker_sub_device_id, mode):
        outputs = []
        num_links = self.args.model_config["GALAXY_NUM_LINKS"]
        # On the all-bf8 low-L1 decode path (QWEN36_ATTN_OUT_BF8=1), emit bf8 logits so the
        # decode lm_head all-reduce reduction CB (sized from this input dtype) matches the bf8
        # tt_lm_head_buffer bank (else CB=516096 bf16 > 274176 bf8 bank). bf16 otherwise.
        _lmhead_dtype = ttnn.bfloat8_b if os.environ.get("QWEN36_ATTN_OUT_BF8", "0") == "1" else ttnn.bfloat16
        # QWEN36_LM_HEAD_PLAIN_DECODE (default ON for qwen3.6): the decode-mode RING lm_head matmul
        # (LM_HEAD_TG_RING_PROGCFG + 32-row ring reshards) produces garbage row-0 logits on the
        # batch-1 decode tail (0.05 PCC vs the proven prefill lm_head, while the backbone row-0 hidden
        # is 0.99). The ring path was never validated end-to-end (the coherent inline demo uses the
        # PREFILL lm_head). Route decode through the SAME minimal_matmul the prefill lm_head uses
        # (identical weights output_weights_decode==output_weights_prefill), then reduce via the
        # decode RS+AG below. Previously this flag was referenced NOWHERE (a no-op).
        _plain_decode = (
            getattr(self.args, "is_qwen36", False)
            and os.environ.get("QWEN36_LM_HEAD_PLAIN_DECODE", "1") == "1"
            and isinstance(self.prefill_pc, ttnn.MinimalMatmulConfig)
        )
        if mode == "decode":
            for weight, pc in zip(self.output_weights_decode, self.program_configs):
                if _plain_decode:
                    _xin = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
                    output = ttnn.experimental.minimal_matmul(
                        input_tensor=_xin,
                        weight_tensor=weight,
                        config=self.prefill_pc,
                        compute_kernel_config=self.compute_kernel_config,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    ttnn.deallocate(_xin)
                else:
                    x = ttnn.to_memory_config(x, self.args.model_config["SHARDED_LM_HEAD_INPUT_32_RING_MEMCFG"])
                    output = ttnn.linear(
                        x,
                        weight,
                        compute_kernel_config=self.compute_kernel_config,
                        program_config=pc,
                        memory_config=self.output_memory_config,
                        dtype=_lmhead_dtype,  # bf8 on the all-bf8 path to match the bf8 all-reduce buffer
                        sub_device_id=worker_sub_device_id,
                    )
                    output = ttnn.to_memory_config(output, self.args.model_config["LM_HEAD_OUT_RING_RESHARD_MEMCFG"])

                outputs.append(output)
        else:
            for weight, pc in zip(self.output_weights_prefill, self.program_configs):
                # qwen3.6 / olmo: LM_HEAD_PREFILL_PROGCFG is a MinimalMatmulConfig
                # which ttnn.linear does not accept — must use
                # ttnn.experimental.minimal_matmul instead.  The original
                # `ttnn.linear(program_config=MinimalMatmulConfig)` call raises
                # a TypeError at runtime (V2-7b 4L logits regression).
                if isinstance(self.prefill_pc, ttnn.MinimalMatmulConfig):
                    output = ttnn.experimental.minimal_matmul(
                        input_tensor=x,
                        weight_tensor=weight,
                        config=self.prefill_pc,
                        compute_kernel_config=self.compute_kernel_config,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                else:
                    output = ttnn.linear(
                        x,
                        weight,
                        compute_kernel_config=self.compute_kernel_config,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        program_config=self.prefill_pc,
                        dtype=ttnn.bfloat16,  # bf16 logits — match the bf16 SAMPLING buffer (minimal_matmul branch already bf16)
                    )
                x.deallocate(True)
                outputs.append(output)

        outputs_reduced = []
        _is_qwen36 = getattr(self.args, "is_qwen36", False)
        for output in outputs:
            if _is_qwen36 and mode == "decode":
                # qwen3.6 decode lm_head reduce — match olmo3 (no-prefetcher port): decompose
                # the col-axis all-reduce into reduce_scatter + all_gather over DRAM. This is
                # still DECODE-mode CCL (separate from prefill), but uses NO persistent L1
                # all-reduce buffer, so it avoids the core-(0,0) static-CB clash that the
                # tt_lm_head_buffer_l1 path hits at qwen3.6's 4x-larger vocab (64512).
                # olmo3 feeds a DRAM (interleaved) lm_head output to its reduce_scatter; qwen3.6's
                # output is L1-sharded (the RESHARD), so move it to DRAM first — that routes
                # line_reduce_scatter to the generic ttnn.reduce_scatter (DRAM-capable) instead of
                # the sharded-only llama_reduce_scatter.
                # If output is already DRAM-interleaved (the minimal_matmul / plain-decode path) use
                # it directly — to_memory_config returns a new wrapper over the SAME buffer, so
                # deallocating `output` would free the reduce_scatter's input (use-after-free, the
                # earlier "Buffer is not allocated" crash). output_dram is freed after the RS below.
                _mc = output.memory_config()
                if _mc.buffer_type == ttnn.BufferType.DRAM and _mc.shard_spec is None:
                    output_dram = output
                else:
                    output_dram = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(output)
                rs_output = self.tt_ccl.line_reduce_scatter(
                    output_dram,
                    cluster_axis=1,
                    num_links=num_links,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    use_noc1_only=False,
                )
                ttnn.deallocate(output_dram)
                ag_output = self.tt_ccl.line_all_gather(
                    rs_output,
                    dim=3,
                    cluster_axis=1,
                    num_links=num_links,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.deallocate(rs_output)
                outputs_reduced.append(ag_output)
            else:
                output_reduced = self.tt_ccl.line_all_reduce(
                    output,
                    cluster_axis=1,
                    num_links=num_links,
                    memory_config=output.memory_config(),
                    lm_head=True,
                    buffer_key="LM_HEAD",
                )  # self.output_memory_config
                outputs_reduced.append(
                    ttnn.sharded_to_interleaved(output_reduced, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                )
        return outputs_reduced
