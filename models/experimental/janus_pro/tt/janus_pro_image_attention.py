"""
Self-attention for the Janus-Pro-7B vision tower.

HF reference: `vision_model.encoder.layers[i].self_attn` (`ModelArgs.reference_vision_attention`).

The Q/K/V weights are fused into a single wqkv tensor and projected with one ttnn.linear, then split via
ttnn.nlp_create_qkv_heads. The qkv bias rides inside that matmul at every device count; the O projection's
bias does so only on a single device, because with a mesh it is applied after the all-reduce.
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtJanusProImageAttention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.num_devices = configuration.num_devices

        self.hidden_size = configuration.vision_dim
        self.n_heads = configuration.vision_attn_n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.n_kv_heads = self.n_heads

        self.n_local_heads = self.n_heads // configuration.num_devices
        self.n_local_kv_heads = self.n_kv_heads // configuration.num_devices

        self.dtype = dtype

        # Both body projections take bfloat8_b on each side, so HiFi2's second pass reads mantissa
        # bits the operands do not carry beyond LoFi's coverage.
        self.compute_kernel_config_lofi = configuration.compute_kernel_config_lofi
        # SDPA is the tower's only compute-bound op -- all three TRISCs sit at 98.5% of its
        # duration -- and the shared config runs it at HiFi4, 64 cycles per tile against HiFi2's
        # 32. The activations reaching it are already bfloat8_b, so the extra passes read bits
        # they do not carry.
        self.compute_kernel_config_sdpa = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self.configuration = configuration
        self.use_fused_qkv_sdpa = configuration.VISION_FUSED_QKV_SDPA

        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (f"{state_dict_prefix}{name}")

        wq_str = f"{state_dict_prefix}wq.weight"
        wk_str = f"{state_dict_prefix}wk.weight"
        wv_str = f"{state_dict_prefix}wv.weight"
        wo_str = f"{state_dict_prefix}wo.weight"

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % configuration.num_devices == 0
        assert self.n_kv_heads % configuration.num_devices == 0

        # Janus-Pro head_dim (64) is already a tile multiple, so no head-dim padding is needed.
        wq = self.state_dict[wq_str]
        wk = self.state_dict[wk_str]
        wv = self.state_dict[wv_str]
        wo = self.state_dict[wo_str]

        wq_chunked, wk_chunked, wv_chunked = (torch.chunk(w, configuration.num_devices) for w in [wq, wk, wv])

        self.wqkv = ttnn.as_tensor(
            torch.concat(
                [
                    torch.concat(
                        [
                            torch.transpose(
                                wq_chunked[i],
                                -2,
                                -1,
                            ),
                            torch.transpose(
                                wk_chunked[i],
                                -2,
                                -1,
                            ),
                            torch.transpose(
                                wv_chunked[i],
                                -2,
                                -1,
                            ),
                        ],
                        dim=-1,
                    )
                    for i in range(configuration.num_devices)
                ],
                dim=-1,
            ),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name("wqkv_sharded"),
        )

        bq_str = f"{state_dict_prefix}wq.bias"
        bk_str = f"{state_dict_prefix}wk.bias"
        bv_str = f"{state_dict_prefix}wv.bias"
        bo_str = f"{state_dict_prefix}wo.bias"

        if bq_str in self.state_dict:
            bq = self.state_dict[bq_str]
            bk = self.state_dict[bk_str]
            bv = self.state_dict[bv_str]

            bq_chunked, bk_chunked, bv_chunked = (torch.chunk(b, configuration.num_devices) for b in [bq, bk, bv])

            self.bqkv = ttnn.as_tensor(
                torch.concat(
                    [
                        torch.concat(
                            [
                                bq_chunked[i],
                                bk_chunked[i],
                                bv_chunked[i],
                            ],
                            dim=-1,
                        )
                        for i in range(configuration.num_devices)
                    ],
                    dim=-1,
                ),
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
                dtype=self.dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name("bqkv_sharded"),
            )
            # ttnn.linear wants the bias as a row, not a bare vector.
            self.bqkv = ttnn.reshape(self.bqkv, [1, -1])
        else:
            self.bqkv = None

        self.wo = ttnn.as_tensor(
            torch.transpose(
                wo,
                -2,
                -1,
            ),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-2),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name("wo_sharded"),
        )

        if bo_str in self.state_dict:
            self.bo = ttnn.as_tensor(
                self.state_dict[bo_str],
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name("bo_replicated"),
            )
            self.bo = ttnn.reshape(self.bo, [1, -1])
        else:
            self.bo = None

        # `bo` may ride inside the output matmul only when nothing reduces after it. With more
        # than one device the all-reduce would fold the bias in once per device.
        self.fuse_bo = self.bo is not None and self.num_devices == 1

        self.scale = self.head_dim**-0.5

    def forward(self, x_11SH, mask=None):
        seq_len = x_11SH.shape[-2]
        batch_size = x_11SH.shape[0]

        # Reshape required:
        # ttnn.embedding returns [b, s, d]
        # ttnn.nlp_create_qkv_heads expects [b, 1, s, d]
        if len(x_11SH.shape) == 3:
            x_11SH = ttnn.reshape(x_11SH, (batch_size, 1, seq_len, -1))

        # The bias rides inside: wqkv and bqkv shard on the same axis with no reduce between them,
        # unlike `bo` below. The bfloat8_b output propagates -- nlp_create_qkv_heads passes the dtype
        # through to q/k/v and SDPA's output takes q's, so the wo matmul's in0 is bfloat8_b too.
        qkv_program_config = self.configuration.vision_qkv_program_config(batch_size, seq_len)
        # A sharded output is taken only on one device: otherwise it flows into the all-gather,
        # which would carry a shard spec covering a fraction of the gathered tensor.
        shard_qkv_output = qkv_program_config is not None and self.num_devices == 1
        xqkv_fused = ttnn.linear(
            x_11SH,
            self.wqkv,
            bias=self.bqkv,
            dtype=ttnn.bfloat8_b,
            # Sharding the output drops the writer loop, and this projection writes three
            # times the tiles per core that wo and c_proj do.
            memory_config=(ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG if shard_qkv_output else ttnn.DRAM_MEMORY_CONFIG),
            compute_kernel_config=self.compute_kernel_config_lofi,
            program_config=qkv_program_config,
        )
        ttnn.deallocate(x_11SH)

        # nlp_create_qkv_heads shards only over a grid dividing num_q_heads, so 16 cores at
        # most against this matmul's 48; the shard cannot carry through.
        if shard_qkv_output:
            interleaved = ttnn.sharded_to_interleaved(xqkv_fused, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(xqkv_fused)
            xqkv_fused = interleaved

        # TODO: get this from model_config, and derive k_chunk from the sequence instead of pinning it
        # to the tower's 576 -- a sequence that is not a multiple of it pads to a second, all-padding
        # chunk plus a mask CB.
        #
        # k_chunk over the whole key sequence gives one inner iteration, so the softmax reduces once.
        # q_chunk sets the parallelism; exp_approx_mode off measured bit-identical. See PERF.md.
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8), q_chunk_size=192, k_chunk_size=576, exp_approx_mode=False
        )

        if self.use_fused_qkv_sdpa:
            # No head split runs: the reader takes q, k and v as strided windows of the fused tensor.
            attn_output_1QSD = ttnn.transformer.fused_qkv_sdpa(
                xqkv_fused,
                self.n_local_heads,
                attn_mask=mask,
                scale=self.scale,
                program_config=sdpa_cfg,
                compute_kernel_config=self.compute_kernel_config_sdpa,
                # nlp_concat_heads is the only consumer.
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(xqkv_fused)
        else:
            q_heads_1QSD, k_heads_1KSD, v_heads_1VSD = ttnn.experimental.nlp_create_qkv_heads(
                xqkv_fused,
                num_heads=self.n_local_heads,
                num_kv_heads=self.n_local_kv_heads,
                transpose_k_heads=False,
                # SDPA is the only consumer and reads all three back immediately, so a DRAM round
                # trip buys nothing. This op is pure data movement -- the write is most of what it
                # costs -- and an L1 write stays on the core. The mcast fan-out that makes L1 lose
                # on a matmul in0 has no counterpart here, since nothing multicasts these.
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(xqkv_fused)

            attn_output_1QSD = ttnn.transformer.scaled_dot_product_attention(
                q_heads_1QSD,
                k_heads_1KSD,
                v_heads_1VSD,
                is_causal=False,
                scale=self.scale,
                attn_mask=mask,
                program_config=sdpa_cfg,
                compute_kernel_config=self.compute_kernel_config_sdpa,
                # Same reasoning as q/k/v above: nlp_concat_heads is the only consumer.
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            # deallocate keys and values
            ttnn.deallocate(q_heads_1QSD)
            ttnn.deallocate(k_heads_1KSD)
            ttnn.deallocate(v_heads_1VSD)

        ###
        # Output matmul
        ###
        attn_output_11SH = ttnn.experimental.nlp_concat_heads(
            attn_output_1QSD,
            # L1, like q/k/v and SDPA's output above, even though `wo` reads this as a matmul in0.
            # The in0 penalty this file records elsewhere comes from a *sharded* source, which makes
            # every core multicast its own piece; an interleaved L1 source under an explicit 2D
            # config just puts the read nearer. Measured -0.7 us per `wo` instance.
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output_1QSD)

        wo_program_config = self.configuration.vision_wo_program_config(batch_size, seq_len)
        shard_wo_output = wo_program_config is not None and self.num_devices == 1
        output_11SH = ttnn.linear(
            attn_output_11SH,
            self.wo,
            bias=self.bo if self.fuse_bo else None,
            compute_kernel_config=self.compute_kernel_config_lofi,
            program_config=wo_program_config,
            # bfloat8_b: this is the attention branch's contribution to the residual, read once
            # by the block's add. The residual stream it lands in stays bfloat16.
            dtype=ttnn.bfloat8_b,
            # A sharded output compiles the writer loop out of the kernel that also reads in1,
            # so the matmul stops issuing one NOC write per output tile. The shard spec follows
            # from the program config's per_core_M and per_core_N.
            memory_config=(ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG if shard_wo_output else ttnn.DRAM_MEMORY_CONFIG),
        )
        ttnn.deallocate(attn_output_11SH)

        if self.num_devices > 1:
            output_all_reduce = self._all_reduce(output_11SH)
            ttnn.deallocate(output_11SH)
        else:
            output_all_reduce = output_11SH

        if self.bo is not None and not self.fuse_bo:
            output_after_bias = ttnn.add(output_all_reduce, self.bo)
            ttnn.deallocate(output_all_reduce)
        else:
            output_after_bias = output_all_reduce

        return output_after_bias

    def _all_reduce(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Sum the per-device partial products of wo, whose weight is sharded on its K axis."""
        gathered = ttnn.experimental.all_gather_async(
            input_tensor,
            persistent_output_buffer=None,
            dim=1,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=4 if self.configuration.is_galaxy else 1,
            topology=ttnn.Topology.Ring,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        reduced = ttnn.experimental.fast_reduce_nc(gathered, dims=[1], output=None, compute_kernel_config=None)
        ttnn.deallocate(gathered)
        return reduced
