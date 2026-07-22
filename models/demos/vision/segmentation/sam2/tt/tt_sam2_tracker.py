# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI U.S. Corp., for the TTNN port. Reference code © Meta Platforms, Inc. (Apache-2.0).
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.demos.vision.segmentation.sam2.tt.tt_mask_decoder import TtMaskDecoder
from models.demos.vision.segmentation.sam2.tt.tt_memory_attention import ATTENTION_BANK_DTYPE, TtMemoryAttention
from models.demos.vision.segmentation.sam2.tt.tt_memory_encoder import TtMemoryEncoder
from models.demos.vision.segmentation.sam2.tt.tt_prompt_encoder import TtPromptEncoder

NO_OBJ_SCORE = -1024.0


class MaskBilinearUpsampler:
    def __init__(self, device):
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        coords = (torch.arange(1024).float() + 0.5) / 4 - 0.5
        coords = torch.clamp(coords, 0, 255)
        mh = self._interp_matrix(coords, 256)
        mw = self._interp_matrix(coords, 256)
        self.mw = ttnn.unsqueeze_to_4D(ttnn.from_torch(mw, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16))
        self.mh_t = ttnn.unsqueeze_to_4D(
            ttnn.from_torch(mh.transpose(-2, -1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        )

    @staticmethod
    def _interp_matrix(coords, size):
        floor = torch.floor(coords).long()
        ceil = torch.clamp(floor + 1, 0, size - 1)
        w = coords - floor.float()
        matrix = torch.zeros((size, len(coords)), dtype=torch.float32)
        matrix[floor, torch.arange(len(coords))] = 1 - w
        matrix[ceil, torch.arange(len(coords))] += w
        return matrix

    def __call__(self, low_nchw):
        source = low_nchw
        if low_nchw.layout != ttnn.TILE_LAYOUT:
            source = ttnn.to_layout(low_nchw, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        elif low_nchw.dtype != ttnn.bfloat16:
            source = ttnn.typecast(low_nchw, ttnn.bfloat16)
        try:
            width_upsampled = ttnn.matmul(
                source,
                self.mw,
                memory_config=source.memory_config(),
                compute_kernel_config=self.compute_kernel_config,
            )
        finally:
            if source is not low_nchw:
                ttnn.deallocate(source)
        try:
            return ttnn.matmul(
                self.mh_t,
                width_upsampled,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
            )
        finally:
            ttnn.deallocate(width_upsampled)


class TtSam2Tracker:
    def __init__(
        self,
        parameters,
        device,
        config,
        *,
        num_maskmem,
        max_obj_ptrs_in_encoder,
        output_cq_id,
    ):
        self.device = device
        self.p = parameters
        self.num_maskmem = num_maskmem
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        self.multimask_output_in_sam = config.multimask_output_in_sam
        self.multimask_min_pt_num = config.multimask_min_pt_num
        self.multimask_max_pt_num = config.multimask_max_pt_num
        self.multimask_output_for_tracking = config.multimask_output_for_tracking
        self.hidden_dim = 256
        self.mem_dim = 64
        self.feat_tokens = 4096
        self._output_cq_id = output_cq_id
        self._mask_upsampler = None
        self._mask_indices = None
        self._closed = False

        self.prompt_encoder = TtPromptEncoder(parameters.sam_prompt_encoder, device)
        self.sam_mask_decoder = TtMaskDecoder(parameters.sam_mask_decoder, device)
        self.memory_attention = TtMemoryAttention(parameters.memory_attention, device)
        self.memory_encoder = TtMemoryEncoder(parameters.memory_encoder, device)

        self.dense_pe_seq = self.p.sam_prompt_encoder.dense_pe_seq
        self.no_mask_dense_seq = self.p.sam_prompt_encoder.no_mask_dense_seq
        batch, channels, height, width = self.p.vision_pos_dev.shape
        vision_pos_nhwc = ttnn.permute(self.p.vision_pos_dev, (0, 2, 3, 1))
        self.vision_pos_seq = ttnn.reshape(vision_pos_nhwc, (batch, height * width, channels))
        self.no_mem_embed = self.p.no_mem_embed_dev
        self.no_obj_ptr = self.p.no_obj_ptr_dev
        batch, channels, height, width = self.p.memory_pos_dev.shape
        spatial_pe_nhwc = ttnn.permute(self.p.memory_pos_dev, (0, 2, 3, 1))
        spatial_pe_seq = ttnn.reshape(spatial_pe_nhwc, (batch, height * width, channels))
        spatial_pe_seq_tile = ttnn.to_layout(spatial_pe_seq, ttnn.TILE_LAYOUT)
        self.precomputed_proj_pos_rope = [
            self._precompute_layer_position_rope(
                layer.cross_attn,
                spatial_pe_seq_tile,
                self.p.maskmem_tpos_enc_dev,
            )
            for layer in self.memory_attention.layers
        ]
        self._proj_pos_rope_cat_cache = [{} for _ in self.memory_attention.layers]

    def _precompute_layer_position_rope(self, cross_attention, spatial_pe_seq, maskmem_tpos_enc):
        cosine, sine = cross_attention.rope.tables()
        projected_positions = []
        for temporal_position in range(self.num_maskmem):
            temporal_encoding = ttnn.slice(
                maskmem_tpos_enc,
                [self.num_maskmem - temporal_position - 1, 0, 0, 0],
                [self.num_maskmem - temporal_position, 1, 1, self.mem_dim],
            )
            temporal_encoding = ttnn.reshape(temporal_encoding, (1, 1, self.mem_dim))
            encoded_position = ttnn.add(
                spatial_pe_seq,
                temporal_encoding,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            projected_position = ttnn.linear(
                encoded_position,
                cross_attention.p.k_proj.weight,
                bias=None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            projected_position = ttnn.reshape(
                projected_position,
                (1, 1, self.feat_tokens, self.hidden_dim),
            )
            projected_positions.append(cross_attention.rope.apply(projected_position, cosine, sine))
            ttnn.deallocate(projected_position)
            ttnn.deallocate(temporal_encoding)
            ttnn.deallocate(encoded_position)
        return projected_positions

    def _mask_index_row(self):
        if self._mask_indices is None:
            indices = ttnn.arange(
                0,
                3,
                1,
                dtype=ttnn.uint32,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            self._mask_indices = ttnn.reshape(indices, (1, 3))
        return self._mask_indices

    def _project_bank_kv_stock(self, memory, *, apply_spatial_rope=False):
        projection = self.memory_attention.p.bank_k_proj
        compute_config = self.memory_attention.layers[0].cross_attn.qkv_split_compute_kernel_config
        packed_keys = None
        split_keys = ()
        keys = ()
        latent_value = None
        try:
            packed_keys = ttnn.linear(
                memory,
                projection.weight,
                bias=projection.bias,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=compute_config,
            )
            split_keys = tuple(
                ttnn.split(
                    packed_keys,
                    self.hidden_dim,
                    dim=-1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            token_count = int(memory.shape[-2])
            pages = tuple(ttnn.reshape(key, (1, 1, token_count, self.hidden_dim)) for key in split_keys)
            if apply_spatial_rope:
                cos, sin = self.memory_attention.rope.tables()
                keys = tuple(self.memory_attention.rope.apply(page, cos, sin) for page in pages)
                for page in pages:
                    ttnn.deallocate(page)
            else:
                keys = pages
            split_keys = ()
            latent_value = ttnn.clone(
                memory,
                dtype=ATTENTION_BANK_DTYPE,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return keys, latent_value
        except BaseException:
            for tensor in (*(keys or split_keys), latent_value):
                if tensor is not None and tensor.is_allocated():
                    ttnn.deallocate(tensor)
            raise
        finally:
            if packed_keys is not None:
                ttnn.deallocate(packed_keys)

    def _project_object_pointer_source(self, source):
        tiled = source if source.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(source, ttnn.TILE_LAYOUT)
        try:
            projected_k, latent_v = self._project_bank_kv_stock(tiled)
            compact_projected_k = tuple(
                ttnn.typecast(tensor, dtype=ATTENTION_BANK_DTYPE, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for tensor in projected_k
            )
            for tensor in projected_k:
                ttnn.deallocate(tensor)
        finally:
            if tiled is not source:
                ttnn.deallocate(tiled)
        num_tokens = int(source.shape[-2])
        return (
            tuple(ttnn.reshape(tensor, (1, 1, num_tokens, self.hidden_dim)) for tensor in compact_projected_k),
            ttnn.reshape(latent_v, (1, 1, num_tokens, self.mem_dim)),
        )

    def _select_memory_bank_pages(self, frame_idx, bank):
        cond_outputs = bank["cond_frame_outputs"]
        non_cond_outputs = bank["non_cond_frame_outputs"]
        temporal_outputs = [(0, output) for output in cond_outputs.values()]
        for temporal_position in range(1, self.num_maskmem):
            frame_offset = self.num_maskmem - temporal_position
            temporal_outputs.append((temporal_position, non_cond_outputs.get(frame_idx - frame_offset)))

        num_layers = len(self.memory_attention.layers)
        spatial_k = [[] for _ in range(num_layers)]
        spatial_v = []
        temporal_positions = []
        for temporal_position, previous in temporal_outputs:
            if previous is None:
                continue
            temporal_positions.append(temporal_position)
            spatial_v.append(previous["v_maskmem"])
            for layer_idx in range(num_layers):
                spatial_k[layer_idx].append(previous["k_maskmem_rope"][layer_idx])

        pointer_values = [output for frame, output in cond_outputs.items() if frame <= frame_idx]
        max_pointers = min(self.max_obj_ptrs_in_encoder, frame_idx + 1)
        for frame_offset in range(1, max_pointers):
            previous = non_cond_outputs.get(frame_idx - frame_offset)
            if previous is not None:
                pointer_values.append(previous)
        return spatial_k, spatial_v, temporal_positions, pointer_values

    def _assemble_pre_projected_memory(self, spatial_k, spatial_v, temporal_positions, pointer_values):
        projected_v = ttnn.concat(
            [*spatial_v, *(previous["obj_ptr_v"] for previous in pointer_values)],
            dim=2,
        )
        pre_projected_kv = []
        for layer_idx, layer_spatial_k in enumerate(spatial_k):
            owns_k_base = len(layer_spatial_k) > 1
            k_base = ttnn.concat(layer_spatial_k, dim=2) if owns_k_base else layer_spatial_k[0]
            key = tuple(temporal_positions)
            cache = self._proj_pos_rope_cat_cache[layer_idx]
            if key not in cache:
                pieces = [self.precomputed_proj_pos_rope[layer_idx][t_pos] for t_pos in key]
                cache[key] = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=2)
            k_memory = ttnn.add(
                k_base,
                cache[key],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if owns_k_base:
                ttnn.deallocate(k_base)
            projected_k = ttnn.concat(
                [k_memory, *(previous["obj_ptr_k"][layer_idx] for previous in pointer_values)],
                dim=2,
            )
            ttnn.deallocate(k_memory)
            pre_projected_kv.append((projected_k, projected_v))
        return pre_projected_kv, projected_v

    # ---- pipeline ------------------------------------------------------------
    def _prepare_memory_conditioned_features(self, frame_idx, enc, bank):
        batch, height, width, _ = enc.top_nhwc.shape
        vision_feat_seq = ttnn.reshape(enc.top_nhwc, (batch, height * width, self.hidden_dim))
        if frame_idx == 0:
            return ttnn.add(vision_feat_seq, self.no_mem_embed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        spatial_k, spatial_v, temporal_positions, pointer_values = self._select_memory_bank_pages(frame_idx, bank)
        pre_projected_kv, projected_v = self._assemble_pre_projected_memory(
            spatial_k, spatial_v, temporal_positions, pointer_values
        )
        try:
            out_seq = self.memory_attention(
                vision_feat_seq,
                self.vision_pos_seq,
                pre_projected_kv,
            )
        finally:
            for projected_k, _ in pre_projected_kv:
                ttnn.deallocate(projected_k)
            ttnn.deallocate(projected_v)
        return out_seq

    def _use_multimask(self, is_init_cond_frame, prompt_inputs):
        num_points = 0
        if prompt_inputs is not None:
            point_labels = prompt_inputs["point_labels"]
            if point_labels is not None:
                num_points += int(point_labels.shape[-1])
            if prompt_inputs["boxes"] is not None:
                num_points += 2
        return (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and self.multimask_min_pt_num <= num_points <= self.multimask_max_pt_num
        )

    def _select_best_mask(self, masks, ious_dev, sam_tokens, *, multimask_output):
        if not multimask_output:
            return masks, ttnn.reshape(sam_tokens, (1, self.hidden_dim)), [ious_dev]
        best_idx_dev = ttnn.argmax(ious_dev, dim=-1, keepdim=True)
        onehot_bool = ttnn.eq(self._mask_index_row(), best_idx_dev)
        onehot_row = ttnn.typecast(onehot_bool, ttnn.bfloat16)
        ttnn.deallocate(onehot_bool)
        if onehot_row.layout != ttnn.TILE_LAYOUT:
            row_major = onehot_row
            onehot_row = ttnn.to_layout(row_major, ttnn.TILE_LAYOUT)
            ttnn.deallocate(row_major)
        l1_onehot_row = ttnn.to_memory_config(onehot_row, ttnn.L1_MEMORY_CONFIG)
        if l1_onehot_row is not onehot_row:
            ttnn.deallocate(onehot_row)
        onehot_row = ttnn.reshape(l1_onehot_row, (1, 1, 3))
        tiled_masks = masks if masks.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(masks, ttnn.TILE_LAYOUT)
        masks_flat = ttnn.reshape(tiled_masks, (1, 3, 256 * 256))
        best_low = ttnn.matmul(onehot_row, masks_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        best_low = ttnn.reshape(best_low, (1, 1, 256, 256))
        tiled_sam_tokens = (
            sam_tokens if sam_tokens.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(sam_tokens, ttnn.TILE_LAYOUT)
        )
        best_token = ttnn.matmul(onehot_row, tiled_sam_tokens, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        best_token = ttnn.reshape(best_token, (1, self.hidden_dim))
        selection_to_free = [
            masks,
            masks_flat,
            sam_tokens,
            tiled_sam_tokens,
            ious_dev,
            best_idx_dev,
            onehot_row,
        ]
        return best_low, best_token, selection_to_free

    def _forward_sam_heads(self, pix_feat_seq, enc, prompt_inputs, *, is_init_cond_frame):
        sparse = None
        if prompt_inputs is not None:
            sparse = self.prompt_encoder.embed_sparse(
                prompt_inputs["point_coords"],
                prompt_inputs["point_labels"],
                prompt_inputs["boxes"],
            )
        multimask_output = self._use_multimask(is_init_cond_frame, prompt_inputs)
        masks, ious_dev, sam_tokens, obj_score = self.sam_mask_decoder(
            image_embeddings=pix_feat_seq,
            image_pe=self.dense_pe_seq,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=self.no_mask_dense_seq,
            multimask_output=multimask_output,
            high_res_features=enc.high_res,
        )
        best_low, best_token, selection_to_free = self._select_best_mask(
            masks,
            ious_dev,
            sam_tokens,
            multimask_output=multimask_output,
        )
        tiled_obj_score = (
            obj_score if obj_score.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(obj_score, ttnn.TILE_LAYOUT)
        )
        is_obj = ttnn.gt(tiled_obj_score, 0.0)  # [1,1]
        if tiled_obj_score is not obj_score:
            ttnn.deallocate(tiled_obj_score)
        is_obj_map = ttnn.reshape(is_obj, (1, 1, 1, 1))
        gated_low = ttnn.where(
            is_obj_map,
            best_low,
            NO_OBJ_SCORE,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(best_low)
        best_low = gated_low
        for tensor in (*selection_to_free, sparse):
            if tensor is not None and tensor.is_allocated():
                ttnn.deallocate(tensor)
        if self._mask_upsampler is None:
            self._mask_upsampler = MaskBilinearUpsampler(self.device)
        high = self._mask_upsampler(best_low)
        layers = self.p.obj_ptr_proj.layers
        obj_ptr = best_token
        for index in range(len(layers)):
            obj_ptr = ttnn.linear(
                obj_ptr,
                layers[str(index)].weight,
                bias=layers[str(index)].bias,
                activation="relu" if index < len(layers) - 1 else None,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        gated_obj_ptr = ttnn.where(
            is_obj,
            obj_ptr,
            self.no_obj_ptr,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(obj_ptr)
        obj_ptr = gated_obj_ptr
        for tensor in (best_token, is_obj_map, is_obj):
            if tensor.is_allocated():
                ttnn.deallocate(tensor)
        return {
            "low_res": best_low,
            "high_res": high,
            "obj_ptr": obj_ptr,
            "object_score_logits": obj_score,
        }

    def _encode_new_memory(self, enc, high_res_masks, *, is_mask_from_prompt):
        tiled_masks = (
            high_res_masks
            if high_res_masks.layout == ttnn.TILE_LAYOUT
            else ttnn.to_layout(high_res_masks, ttnn.TILE_LAYOUT)
        )
        if is_mask_from_prompt:
            positive = ttnn.gt(tiled_masks, 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            transformed = ttnn.where(
                positive,
                self.cfg_scale + self.cfg_bias,
                self.cfg_bias,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(positive)
        else:
            transformed = ttnn.unary_chain(
                tiled_masks,
                [
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID),
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, self.cfg_scale),
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.ADD_UNARY_SFPU, self.cfg_bias),
                ],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if tiled_masks is not high_res_masks:
            ttnn.deallocate(tiled_masks)
        transformed_rm = ttnn.to_layout(
            transformed,
            ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(transformed)
        m = ttnn.permute(
            transformed_rm,
            (0, 2, 3, 1),
            memory_config=self.memory_encoder.mask_input_memory_config,
        )
        ttnn.deallocate(transformed_rm)
        return self.memory_encoder(enc.top_nhwc, m)

    # sigmoid scale/bias from sam2_hiera_t.yaml
    cfg_scale = 20.0
    cfg_bias = -10.0

    def _project_new_memory(self, maskmem_features):
        batch, height, width, _ = maskmem_features.shape
        maskmem_features_seq = ttnn.reshape(maskmem_features, (batch, height * width, self.mem_dim))
        maskmem_features_seq_tile = (
            maskmem_features_seq
            if maskmem_features_seq.layout == ttnn.TILE_LAYOUT
            else ttnn.to_layout(maskmem_features_seq, ttnn.TILE_LAYOUT)
        )
        projected_k_maskmem, latent_v_maskmem = self._project_bank_kv_stock(
            maskmem_features_seq_tile,
            apply_spatial_rope=True,
        )
        k_maskmem_rope_list = []
        for k_maskmem_rope in projected_k_maskmem:
            compact_k_maskmem_rope_seq = ttnn.typecast(
                k_maskmem_rope,
                dtype=ATTENTION_BANK_DTYPE,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            k_maskmem_rope_list.append(
                ttnn.reshape(compact_k_maskmem_rope_seq, (1, 1, self.feat_tokens, self.hidden_dim))
            )
            ttnn.deallocate(k_maskmem_rope)
        latent_v_maskmem = ttnn.reshape(latent_v_maskmem, (1, 1, self.feat_tokens, self.mem_dim))
        if maskmem_features_seq_tile is not maskmem_features_seq:
            ttnn.deallocate(maskmem_features_seq_tile)
        ttnn.deallocate(maskmem_features)
        return k_maskmem_rope_list, latent_v_maskmem

    def _begin_mask_readback(self, mask):
        host_mask = ttnn.allocate_tensor_on_host(
            mask.shape,
            mask.dtype,
            mask.layout,
            self.device,
            mask.memory_config(),
        )
        producer_event = ttnn.record_event(self.device, 0)
        if self._output_cq_id != 0:
            ttnn.wait_for_event(self._output_cq_id, producer_event)
        ttnn.copy_device_to_host_tensor(
            mask,
            host_mask,
            blocking=False,
            cq_id=self._output_cq_id,
        )
        return host_mask, ttnn.record_event(self.device, self._output_cq_id)

    def _finish_track_step(self, frame_idx, enc, pix_feat, prompt_inputs):
        heads = self._forward_sam_heads(
            pix_feat,
            enc,
            prompt_inputs,
            is_init_cond_frame=frame_idx == 0,
        )
        high_res_host, high_res_read_event = self._begin_mask_readback(heads["high_res"])
        ttnn.deallocate(pix_feat)
        maskmem_features = self._encode_new_memory(
            enc,
            heads["high_res"],
            is_mask_from_prompt=prompt_inputs is not None,
        )
        k_maskmem_rope_list, latent_v_maskmem = self._project_new_memory(maskmem_features)
        obj_ptr_source = ttnn.reshape(heads["obj_ptr"], (1, self.hidden_dim // self.mem_dim, self.mem_dim))
        obj_ptr_k, obj_ptr_v = self._project_object_pointer_source(obj_ptr_source)
        ttnn.event_synchronize(high_res_read_event)
        return {
            "pred_masks": heads["low_res"],
            "pred_masks_high_res": high_res_host,
            "_pred_masks_high_res_device": heads["high_res"],
            "obj_ptr": heads["obj_ptr"],
            "object_score_logits": heads["object_score_logits"],
            "obj_ptr_k": obj_ptr_k,
            "obj_ptr_v": obj_ptr_v,
            "k_maskmem_rope": k_maskmem_rope_list,
            "v_maskmem": latent_v_maskmem,
        }

    def track_step(self, frame_idx, enc, prompt_inputs, bank):
        if self._closed:
            raise RuntimeError("SAM2 tracker is closed")
        pix_feat = self._prepare_memory_conditioned_features(frame_idx, enc, bank)
        return self._finish_track_step(frame_idx, enc, pix_feat, prompt_inputs)

    def close(self):
        if self._closed:
            return
        self.__dict__.clear()
        self._closed = True
