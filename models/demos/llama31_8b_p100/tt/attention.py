# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from models.ttt_compat.tt.attention import *  # noqa: F401,F403  # isort: skip
import math

import models.ttt_compat.tt.attention as _tt
import ttnn
from models.demos.llama31_8b_p100.tt import wiring
from models.ttt_compat.tt.ccl import tt_all_gather, tt_all_reduce
from models.ttt_compat.tt.common import Mode


class Attention(_tt.Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._matmul_bh = wiring.on("matmul_bh", self.args)
        self._qkv_heads_decode = wiring.on("qkv_heads_decode", self.args)
        self._concat_heads_decode = wiring.on("concat_heads_decode", self.args)
        self._rotary_bf8 = wiring.on("rotary_bf8", self.args)
        self._sdpa_bh = wiring.on("sdpa_bh", self.args)

    def forward_decode(self, x: ttnn.Tensor, current_pos, rot_mats=None, page_table=None, kv_cache=None) -> ttnn.Tensor:
        if self.TG and self.prefetcher is None:
            x_interleaved = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            xqkv_fused_sharded = ttnn.linear(
                x_interleaved,
                self.wqkv,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=None,
                compute_kernel_config=self.li_qkv_decode_compute_kernel_cfg,
                dtype=self.ccl_dtype,
            )
            ttnn.deallocate(x_interleaved)
        else:
            xqkv_fused_sharded = wiring.linear(
                self._matmul_bh,
                x,
                self.wqkv,
                memory_config=self.args.get_attn_qkv_mm_mem_config(Mode.DECODE, self.prefetcher),
                program_config=self.args.get_attn_qkv_program_config(Mode.DECODE, 1, self.prefetcher),
                compute_kernel_config=self.li_qkv_decode_compute_kernel_cfg,
                dtype=self.ccl_dtype if self.TG else self.activation_dtype or ttnn.bfloat16,
                global_cb=self.prefetcher.global_cb if self.prefetcher is not None else None,
                sub_device_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
            )
        if self.wqkv_bias_decode:
            num_tiles = int(math.ceil(xqkv_fused_sharded.shape[-2] / self.tile_size))
            xqkv_fused_sharded = xqkv_fused_sharded + self.wqkv_bias_decode[num_tiles - 1]

        ttnn.deallocate(x)
        qkv_all_reduce_mem_cfg = self.args.get_attn_qkv_all_reduce_output_mem_config(
            Mode.DECODE, list(self.mesh_device.shape)[1], self.prefetcher
        )
        xqkv_fused = tt_all_reduce(
            xqkv_fused_sharded,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=1,
            memory_config=(
                qkv_all_reduce_mem_cfg if qkv_all_reduce_mem_cfg is not None else xqkv_fused_sharded.memory_config()
            ),
            sharded=not (self.TG and self.prefetcher is None),
            dtype=self.ccl_dtype,
            topology=self.ccl_topology,
            subdevice_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
        )
        if self.TG:
            xqkv_fused = ttnn.matmul(
                self.slice_mat,
                xqkv_fused,
                dtype=ttnn.bfloat16,
                memory_config=self.args.get_attn_create_head_input_mem_config(Mode.DECODE),
            )
        else:
            if self.prefetcher is None and not (
                self._qkv_heads_decode is not None
                and self._qkv_heads_decode.covers(xqkv_fused_sharded, self.n_local_heads, self.n_local_kv_heads)
            ):
                xqkv_fused = ttnn.sharded_to_interleaved(xqkv_fused_sharded, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
                ttnn.deallocate(xqkv_fused_sharded)
            else:
                xqkv_fused = xqkv_fused_sharded
        fqkv_shape = xqkv_fused.shape
        xqkv_fused = ttnn.reshape(
            xqkv_fused, (1, 1, self.batch_size_per_device_group, fqkv_shape[3]), (1, 1, 32, fqkv_shape[3])
        )

        (
            q_heads_pre_rot_1BQD,
            k_heads_pre_rot_1BKD,
            v_heads_1BKD,
        ) = (
            self._qkv_heads_decode.qkv_heads_decode(
                xqkv_fused,
                self.n_local_heads,
                self.n_local_kv_heads,
                memory_config=self.args.get_attn_create_head_output_mem_config(Mode.DECODE, self.prefetcher),
            )
            if self._qkv_heads_decode is not None
            and self._qkv_heads_decode.covers(xqkv_fused, self.n_local_heads, self.n_local_kv_heads)
            else ttnn.experimental.nlp_create_qkv_heads_decode(
                xqkv_fused,
                num_heads=self.n_local_heads,
                num_kv_heads=self.n_local_kv_heads,
                memory_config=self.args.get_attn_create_head_output_mem_config(Mode.DECODE, self.prefetcher),
            )
        )
        norm_config = self.args.get_norm_config("attn", Mode.DECODE, None)
        q_heads_pre_rot_1BQD = self.q_norm(q_heads_pre_rot_1BQD, mode=Mode.DECODE, norm_config=norm_config)
        k_heads_pre_rot_1BKD = self.k_norm(k_heads_pre_rot_1BKD, mode=Mode.DECODE, norm_config=norm_config)
        ttnn.deallocate(xqkv_fused)

        q_heads_1BQD, k_heads_1BKD = self.rotary_embedding_decode(
            q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats, current_pos
        )

        ttnn.deallocate(q_heads_pre_rot_1BQD)
        ttnn.deallocate(k_heads_pre_rot_1BKD)

        if self.k_bias_shift_decode is not None:
            k_heads_shifted = ttnn.subtract(
                k_heads_1BKD, self.k_bias_shift_decode, memory_config=k_heads_1BKD.memory_config()
            )
            ttnn.deallocate(k_heads_1BKD)
            k_heads_1BKD = k_heads_shifted
        if kv_cache:
            keys = kv_cache[0]
            values = kv_cache[1]
        else:
            keys = self.layer_past[0]
            values = self.layer_past[1]

        if self.use_qk_fused:
            ttnn.experimental.paged_fused_update_cache(
                keys, k_heads_1BKD, values, v_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
            )
        else:
            ttnn.experimental.paged_update_cache(
                keys, k_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
            )
            ttnn.experimental.paged_update_cache(
                values, v_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
            )
        ttnn.deallocate(k_heads_1BKD)
        ttnn.deallocate(v_heads_1BKD)
        sdpa_decode_prog_cfg = self.args.get_attn_sdpa_decode_program_config(self.prefetcher)
        if page_table is not None:
            attn_output_1G4D = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_heads_1BQD,
                keys,
                values,
                page_table_tensor=page_table,
                cur_pos_tensor=current_pos,
                scale=self.scale,
                sliding_window_size=self.sliding_window,
                program_config=sdpa_decode_prog_cfg,
                compute_kernel_config=self.sdpa_decode_compute_kernel_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            attn_output_1G4D = ttnn.transformer.scaled_dot_product_attention_decode(
                q_heads_1BQD,
                keys,
                values,
                cur_pos_tensor=current_pos,
                scale=self.scale,
                sliding_window_size=self.sliding_window,
                program_config=sdpa_decode_prog_cfg,
                compute_kernel_config=self.sdpa_decode_compute_kernel_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        ttnn.deallocate(q_heads_1BQD)
        if self._concat_heads_decode is not None and self._concat_heads_decode.covers(
            attn_output_1G4D, self.n_local_heads, self.prefetcher
        ):
            attn_output_cat = self._concat_heads_decode.concat_heads_decode(attn_output_1G4D, self.n_local_heads)
            ttnn.deallocate(attn_output_1G4D)
        else:
            attn_output_11BH = ttnn.to_memory_config(
                attn_output_1G4D,
                memory_config=self.args.get_attn_sdpa_output_mem_config(
                    Mode.DECODE, self.batch_size_per_device_group, self.prefetcher
                ),
            )

            attn_output_cat = ttnn.experimental.nlp_concat_heads_decode(
                attn_output_11BH,
                num_heads=self.n_local_heads,
                sub_core_grids=self.prefetcher.all_worker_cores_range_set if self.prefetcher is not None else None,
            )
            ttnn.deallocate(attn_output_11BH)
            ttnn.deallocate(attn_output_1G4D)

        if self.use_fused_all_gather_matmul or self.prefetcher is not None:
            attn_output_cat = ttnn.to_memory_config(
                attn_output_cat,
                self.args.get_attn_concat_heads_output_mem_config(Mode.DECODE, self.prefetcher),
            )

            if self.ccl_topology == ttnn.Topology.Ring and self.prefetcher is None:
                _, dense_out_sharded = ttnn.experimental.all_gather_matmul_async(
                    attn_output_cat,
                    self.wo,
                    persistent_output_buffer=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                    all_gather_core_grid_offset=(0, 4),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                    num_links=self.model_config["ATTN_AGMM_CONFIG"]["num_links"],
                    memory_config_ag=self.args.get_attn_all_gather_output_mem_config(Mode.DECODE, None),
                    memory_config_mm=self.args.get_attn_dense_output_mem_config(Mode.DECODE, None),
                    program_config=self.args.get_attn_all_gather_matmul_program_config(Mode.DECODE, None),
                    compute_kernel_config=self.compute_kernel_config_hifi2,
                    chunks_per_sync=self.model_config["ATTN_AGMM_CONFIG"]["chunks_per_sync"],
                    num_workers_per_link=self.model_config["ATTN_AGMM_CONFIG"]["num_workers_per_link"],
                    num_buffers_per_channel=2,
                    subdevice_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
                )
            else:
                all_gather_output = ttnn.experimental.all_gather_async(
                    attn_output_cat,
                    persistent_output_buffer=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                    num_links=1,
                    topology=self.ccl_topology,
                    memory_config=self.args.get_attn_all_gather_output_mem_config(Mode.DECODE, self.prefetcher),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                    subdevice_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
                )
                dense_out_sharded = ttnn.linear(
                    all_gather_output,
                    self.wo_sharded_ring if self.prefetcher is not None else self.wo,
                    memory_config=self.args.get_attn_dense_output_mem_config(Mode.DECODE, self.prefetcher),
                    program_config=self.args.get_attn_all_gather_matmul_program_config(Mode.DECODE, self.prefetcher),
                    compute_kernel_config=self.li_o_decode_compute_kernel_cfg,
                    global_cb=self.prefetcher.global_cb if self.prefetcher is not None else None,
                    sub_device_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
                )
                ttnn.deallocate(all_gather_output)
            ttnn.deallocate(attn_output_cat)
            dense_out_sharded = ttnn.to_memory_config(
                dense_out_sharded,
                self.args.get_attn_dense_output_mem_config(Mode.DECODE, self.prefetcher),
            )
            return dense_out_sharded

        else:
            attn_output = attn_output_cat
            if 1 not in self.mesh_device.shape:
                attn_output = tt_all_gather(
                    attn_output_cat,
                    self.mesh_device,
                    self.tt_ccl,
                    dim=2,
                    cluster_axis=1,
                    memory_config=self.args.get_attn_gather_users_mem_config(
                        Mode.DECODE, list(self.mesh_device.shape)[1], self.prefetcher
                    ),
                    sharded=True,
                    subdevice_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
                )
            if self.TG:
                attn_output = ttnn.to_memory_config(attn_output, ttnn.L1_MEMORY_CONFIG)
                attn_output = ttnn.matmul(
                    self.user_selection_matrix,
                    attn_output,
                    core_grid=ttnn.CoreGrid(y=4, x=8),
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                )

            dense_out_sharded = wiring.linear(
                self._matmul_bh,
                attn_output,
                self.wo,
                core_grid=ttnn.CoreGrid(y=4, x=8) if self.TG else None,
                program_config=self.args.get_attn_wo_program_config(Mode.DECODE, 1, self.prefetcher),
                memory_config=self.args.get_attn_wo_output_mem_config(Mode.DECODE, self.prefetcher),
                dtype=ttnn.bfloat8_b if self.TG else None,
                compute_kernel_config=self.li_o_decode_compute_kernel_cfg,
                global_cb=self.prefetcher.global_cb if self.prefetcher is not None else None,
                sub_device_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
            )

            ttnn.deallocate(attn_output_cat)

            dense_out_reduced = tt_all_reduce(
                dense_out_sharded,
                self.mesh_device,
                self.tt_ccl,
                cluster_axis=0,
                dim=0 if (self.TG and self.hidden_size < 8192) else 3,
                topology=self.ccl_topology,
                memory_config=self.args.get_attn_all_reduce_output_mem_config(
                    Mode.DECODE, self.hidden_size, list(self.mesh_device.shape)[0], self.prefetcher
                ),
                sharded=True,
                dtype=self.ccl_dtype,
                use_composite=True if self.hidden_size == 8192 else False,
                subdevice_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
            )

            if not self.TG:
                dense_out_reduced = ttnn.to_memory_config(
                    dense_out_reduced, self.args.get_attn_dense_output_mem_config(Mode.DECODE, None)
                )

            return dense_out_reduced

    def forward_prefill(
        self,
        x_11SH,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ):
        batch_size = x_11SH.shape[0]
        if batch_size > 1:
            x_11SH = ttnn.reshape(x_11SH, [1, 1, x_11SH.shape[-2] * x_11SH.shape[-3] * x_11SH.shape[-4], -1])

        seq_len = x_11SH.shape[-2]
        original_seq_len = seq_len
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"

        if seq_len > self.MAX_QKV_MM_SEQ_LEN and seq_len % self.MAX_QKV_MM_SEQ_LEN != 0:
            padded_seq_len = (
                (seq_len + self.MAX_QKV_MM_SEQ_LEN - 1) // self.MAX_QKV_MM_SEQ_LEN
            ) * self.MAX_QKV_MM_SEQ_LEN
            pad_len = padded_seq_len - seq_len
            x_11SH = ttnn.pad(x_11SH, padding=[(0, 0), (0, 0), (0, pad_len), (0, 0)], value=0.0)
            seq_len = padded_seq_len

        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // self.MAX_QKV_MM_SEQ_LEN, self.MAX_QKV_MM_SEQ_LEN, -1])

        qkv_heads = None
        if self.args.use_minimal_qkv_prefill_matmul(seq_len) and self.wqkv_bias_prefill is None and batch_size == 1:
            qkv_heads = wiring.qkv_heads(
                self._matmul_bh,
                x_11SH,
                self.wqkv,
                compute_kernel_config=self.li_qkv_prefill_compute_kernel_cfg,
                config=self.args.get_attn_qkv_program_config(Mode.PREFILL, seq_len, None),
                num_heads=self.n_local_heads,
                num_kv_heads=self.n_local_kv_heads,
            )
        if qkv_heads is not None:
            q_heads_1QSD_pre_rot, k_heads_1KSD_pre_rot, v_heads_1VSD = qkv_heads
            ttnn.deallocate(x_11SH)
        else:
            if self.args.use_minimal_qkv_prefill_matmul(seq_len):
                xqkv_fused = wiring.minimal_matmul(
                    self._matmul_bh,
                    x_11SH,
                    self.wqkv,
                    compute_kernel_config=self.li_qkv_prefill_compute_kernel_cfg,
                    config=self.args.get_attn_qkv_program_config(Mode.PREFILL, seq_len, None),
                )
            else:
                xqkv_fused = wiring.linear(
                    self._matmul_bh,
                    x_11SH,
                    self.wqkv,
                    dtype=self.ccl_dtype if self.TG else self.activation_dtype or ttnn.bfloat16,
                    memory_config=self.args.get_attn_qkv_mm_mem_config(Mode.PREFILL, None),
                    compute_kernel_config=self.li_qkv_prefill_compute_kernel_cfg,
                    program_config=self.args.get_attn_qkv_program_config(Mode.PREFILL, seq_len, None),
                )

            if self.wqkv_bias_prefill is not None:
                xqkv_fused = xqkv_fused + self.wqkv_bias_prefill

            xqkv_fused = tt_all_reduce(
                xqkv_fused,
                self.mesh_device,
                self.tt_ccl,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=self.ccl_dtype,
            )

            if seq_len > self.MAX_QKV_MM_SEQ_LEN:
                xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

            if original_seq_len != seq_len:
                xqkv_fused = xqkv_fused[:, :, :original_seq_len, :]
                seq_len = original_seq_len

            if batch_size > 1:
                xqkv_fused = ttnn.reshape(xqkv_fused, [batch_size, 1, seq_len // batch_size, -1])

            ttnn.deallocate(x_11SH)

            (
                q_heads_1QSD_pre_rot,
                k_heads_1KSD_pre_rot,
                v_heads_1VSD,
            ) = ttnn.experimental.nlp_create_qkv_heads(
                xqkv_fused,
                num_heads=self.n_local_heads,
                num_kv_heads=self.n_local_kv_heads,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        norm_config = self.args.get_norm_config("attn", Mode.PREFILL, None)
        q_heads_1QSD_pre_rot = self.q_norm(q_heads_1QSD_pre_rot, mode=Mode.PREFILL, norm_config=norm_config)
        k_heads_1KSD_pre_rot = self.k_norm(k_heads_1KSD_pre_rot, mode=Mode.PREFILL, norm_config=norm_config)

        if qkv_heads is None:
            ttnn.deallocate(xqkv_fused)

        rot8 = (
            self._rotary_bf8 is not None
            and self.rotary_embedding_prefill == self._mllama_rope_prefill
            and self.k_bias_shift_prefill is None
            and self._rotary_bf8.covers(
                q_heads_1QSD_pre_rot,
                k_heads_1KSD_pre_rot,
                rot_mats[0],
                rot_mats[1],
                self.transformation_mats["prefill"],
            )
        )
        if rot8:
            k_dtype = (kv_cache[0] if kv_cache else self.layer_past[0]).dtype
            q_heads_1QSD_8b = self._rotary_bf8.rotary_bf8(
                q_heads_1QSD_pre_rot,
                rot_mats[0],
                rot_mats[1],
                self.transformation_mats["prefill"],
                self.activation_dtype or ttnn.bfloat8_b,
            )
            k_heads_1KSD_8b = self._rotary_bf8.rotary_bf8(
                k_heads_1KSD_pre_rot, rot_mats[0], rot_mats[1], self.transformation_mats["prefill"], k_dtype
            )
        else:
            q_heads_1QSD, k_heads_1KSD = self.rotary_embedding_prefill(
                q_heads_1QSD_pre_rot, k_heads_1KSD_pre_rot, rot_mats
            )
        ttnn.deallocate(q_heads_1QSD_pre_rot)
        ttnn.deallocate(k_heads_1KSD_pre_rot)

        if not rot8 and self.k_bias_shift_prefill is not None:
            k_heads_shifted = ttnn.subtract(
                k_heads_1KSD, self.k_bias_shift_prefill, memory_config=k_heads_1KSD.memory_config()
            )
            ttnn.deallocate(k_heads_1KSD)
            k_heads_1KSD = k_heads_shifted

        if kv_cache:
            keys_BKSD, values_BKSD = kv_cache[0], kv_cache[1]
        else:
            keys_BKSD, values_BKSD = self.layer_past[0], self.layer_past[1]

        if not rot8:
            k_heads_1KSD_8b = ttnn.typecast(k_heads_1KSD, dtype=keys_BKSD.dtype)
            ttnn.deallocate(k_heads_1KSD)

        if seq_len >= self.min_kv_prefill_shard_seqlen and not self.TG and page_table is None:
            k_fill = ttnn.interleaved_to_sharded(k_heads_1KSD_8b, self.args.get_attn_kv_prefill_mem_config(seq_len))
        else:
            k_fill = k_heads_1KSD_8b

        v_heads_1VSD_8b = ttnn.typecast(v_heads_1VSD, dtype=values_BKSD.dtype)

        ttnn.deallocate(v_heads_1VSD)

        if seq_len >= self.min_kv_prefill_shard_seqlen and not self.TG and page_table is None:
            v_fill = ttnn.interleaved_to_sharded(v_heads_1VSD_8b, self.args.get_attn_kv_prefill_mem_config(seq_len))
        else:
            v_fill = v_heads_1VSD_8b

        if self.TG:
            k_fill = self.prefill_prepare_tensor_for_kv_cache(k_fill, user_id)
            v_fill = self.prefill_prepare_tensor_for_kv_cache(v_fill, user_id)
        if page_table is not None:
            block_size = keys_BKSD.shape[2]
            fill_page_table = chunk_page_table if chunk_page_table is not None else page_table

        if batch_size > 1:
            seq_len_per_user = k_fill.shape[2]
            page_len = fill_page_table.shape[1] * block_size

            for slot_idx in range(batch_size):
                k_user = k_fill[slot_idx : slot_idx + 1, :, :, :]
                v_user = v_fill[slot_idx : slot_idx + 1, :, :, :]

                k_user_sliced = k_user[:, :, :page_len, :] if page_len < seq_len_per_user else k_user
                v_user_sliced = v_user[:, :, :page_len, :] if page_len < seq_len_per_user else v_user

                ttnn.experimental.paged_fill_cache(keys_BKSD, k_user_sliced, fill_page_table, batch_idx=slot_idx)
                ttnn.experimental.paged_fill_cache(values_BKSD, v_user_sliced, fill_page_table, batch_idx=slot_idx)
        elif page_table is not None:
            page_len = fill_page_table.shape[1] * block_size
            k_fill_sliced = k_fill[:, :, :page_len, :] if page_len < k_fill.shape[2] else k_fill
            v_fill_sliced = v_fill[:, :, :page_len, :] if page_len < v_fill.shape[2] else v_fill
            ttnn.experimental.paged_fill_cache(keys_BKSD, k_fill_sliced, fill_page_table, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(values_BKSD, v_fill_sliced, fill_page_table, batch_idx=user_id)
        else:
            ttnn.fill_cache(
                keys_BKSD,
                k_fill,
                user_id % self.batch_size_per_device_group,
            )
            ttnn.fill_cache(
                values_BKSD,
                v_fill,
                user_id % self.batch_size_per_device_group,
            )
        if seq_len >= self.min_kv_prefill_shard_seqlen and not self.TG and page_table is None:
            ttnn.deallocate(k_fill)
            ttnn.deallocate(v_fill)

        if not rot8:
            q_heads_1QSD_8b = ttnn.typecast(q_heads_1QSD, dtype=self.activation_dtype or ttnn.bfloat8_b)
            ttnn.deallocate(q_heads_1QSD)

        attn_output_11SH = None
        if chunk_start_idx is not None:
            if self.sliding_window is not None:
                raise NotImplementedError("Sliding window not supported for chunked prefill SDPA")
            if isinstance(chunk_start_idx, ttnn.Tensor):
                attn_output_84SD = ttnn.transformer.chunked_scaled_dot_product_attention(
                    input_tensor_q=q_heads_1QSD_8b,
                    input_tensor_k=keys_BKSD,
                    input_tensor_v=values_BKSD,
                    page_table_tensor=page_table,
                    chunk_start_idx=None,
                    chunk_start_idx_tensor=chunk_start_idx,
                    compute_kernel_config=self.sdpa_prefill_compute_kernel_cfg,
                    program_config=self.args.get_attn_sdpa_program_config(Mode.PREFILL, seq_len, 0, None),
                )
            else:
                attn_output_84SD = ttnn.transformer.chunked_scaled_dot_product_attention(
                    input_tensor_q=q_heads_1QSD_8b,
                    input_tensor_k=keys_BKSD,
                    input_tensor_v=values_BKSD,
                    page_table_tensor=page_table,
                    chunk_start_idx=chunk_start_idx,
                    compute_kernel_config=self.sdpa_prefill_compute_kernel_cfg,
                    program_config=self.args.get_attn_sdpa_program_config(Mode.PREFILL, seq_len, chunk_start_idx, None),
                )
        else:
            sdpa_seq_len = seq_len // batch_size if batch_size > 1 else seq_len
            sdpa_program_config = self.args.get_attn_sdpa_program_config(Mode.PREFILL, sdpa_seq_len, None, None)
            attn_output_84SD = None
            if self._sdpa_bh is not None and self.sliding_window is None and batch_size == 1:
                attn_output_11SH = self._sdpa_bh.sdpa(
                    q_heads_1QSD_8b,
                    k_heads_1KSD_8b,
                    v_heads_1VSD_8b,
                    self.scale,
                    sdpa_program_config,
                    self.sdpa_prefill_compute_kernel_cfg,
                    output_concat_heads=True,
                )
            if attn_output_11SH is None:
                attn_output_84SD = ttnn.transformer.scaled_dot_product_attention(
                    q_heads_1QSD_8b,
                    k_heads_1KSD_8b,
                    v_heads_1VSD_8b,
                    is_causal=True,
                    sliding_window_size=self.sliding_window,
                    scale=self.scale,
                    compute_kernel_config=self.sdpa_prefill_compute_kernel_cfg,
                    program_config=sdpa_program_config,
                )

        ttnn.deallocate(q_heads_1QSD_8b)
        ttnn.deallocate(k_heads_1KSD_8b)
        ttnn.deallocate(v_heads_1VSD_8b)

        if attn_output_11SH is None:
            if batch_size == 1:
                attn_output_1QSD = ttnn.reshape(attn_output_84SD, [1, self.n_local_heads, -1, self.head_dim])
            else:
                attn_output_1QSD = attn_output_84SD

            attn_output_11SH = ttnn.experimental.nlp_concat_heads(
                attn_output_1QSD,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(attn_output_1QSD)

        if batch_size > 1:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, 1, seq_len, -1])

        if seq_len > 1024:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, seq_len // 1024, 1024, -1])

        if self.use_fused_all_gather_matmul:
            attn_output_11SH = ttnn.experimental.all_gather_async(
                attn_output_11SH,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=1,
                topology=self.ccl_topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

        output_11SH = wiring.linear(
            self._matmul_bh,
            attn_output_11SH,
            self.wo,
            compute_kernel_config=self.li_o_prefill_compute_kernel_cfg,
            dtype=self.activation_dtype or ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.args.get_attn_wo_program_config(Mode.PREFILL, seq_len, None),
        )

        if seq_len > 1024:
            output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_output_11SH)

        if not self.use_fused_all_gather_matmul:
            output_11SH = tt_all_reduce(
                output_11SH,
                self.mesh_device,
                self.tt_ccl,
                cluster_axis=0,
                dim=0 if self.TG else 3,
                topology=self.ccl_topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=self.ccl_dtype,
            )

        return output_11SH
