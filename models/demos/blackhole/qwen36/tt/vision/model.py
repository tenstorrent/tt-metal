# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tt.vision import vision_fast as _vf
from models.demos.blackhole.qwen36.tt.vision.functional import qwen3_5_vision_transformer_preprocess
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import get_rot_transformation_mat
from models.tt_transformers.tt.load_checkpoints import (
    convert_hf_to_meta,
    convert_rope_style_hf_to_meta,
    standardize_hf_keys_multimodal,
)

from .patch_merger import PatchMerger
from .vision_block import VisionBlock
from .vision_model_config import VisionModelArgs


class VisionTransformer(LightweightModule):
    """
    Vision Transformer model for Qwen 3 VL.
    This implements only the transformer blocks part of the vision transformer.
    Patch embedding and merging should be done outside this class.
    """

    def __init__(
        self,
        args,
        dtype,
        state_dict,
        weight_cache_path,
        tt_ccl=None,
    ):
        """
        Initialize the Vision Transformer model.

        Args:
            args (VisionModelArgs): Model arguments
            dtype (ttnn.dtype): Data type for computations
            state_dict (dict): State dictionary containing model weights
            weight_cache_path (str): Path to weight cache
            tt_ccl (TT_CCL, optional): CCL helper. The vision tower is always
                tensor-parallel, so one is constructed internally if not given.
        """
        super().__init__()
        self.args = args
        self.dtype = dtype
        self.weight_cache_path = weight_cache_path

        # The vision tower is always tensor-parallel and needs a TT_CCL on hand.
        self.tt_ccl = tt_ccl if tt_ccl is not None else TT_CCL(args.mesh_device)

        # Create transformation matrix for RoPE QK prefill
        transformation_mat_torch = get_rot_transformation_mat(
            args.head_dim
        )  # todo)) args.head_dim is ignored inside the function
        self.transformation_mats = {
            "prefill": ttnn.as_tensor(
                transformation_mat_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=args.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(args.mesh_device),
            )
        }

        # Create vision blocks
        self.blocks = []
        for i in range(args.hf_config.vision_config.depth):
            block = VisionBlock(
                mesh_device=args.mesh_device,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                dtype=dtype,
                transformation_mats=self.transformation_mats,
                args=args,
                tt_ccl=self.tt_ccl,
            )
            self.blocks.append(block)

        # The Megatron-style PatchMerger consumes a fractured-along-dim=3 tensor
        # directly (no pre-merger all-gather) and produces a fractured-along-dim=3
        # output, mirroring the LLM's DistributedNorm + LMHead final stretch in
        # `tt_transformers.tt.model`.
        self.patch_merger = PatchMerger(
            mesh_device=args.mesh_device,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            state_dict_prefix=args.get_state_dict_prefix("PatchMerger"),
            dtype=dtype,
            tt_ccl=self.tt_ccl,
        )

    def prepare_input(self, patch_input, seq_len=None):
        """Convert a patchified torch input to a ttnn tensor
        Args:
            patch_input (torch.Tensor): Patchified input tensor
            seq_len (int): Sequence length

        Returns:
            ttnn.Tensor: Prepared input tensor
        """
        patch_seq_len, _ = patch_input.shape
        x = patch_input
        seq_len = ((patch_seq_len // 128) + 1) * 128 if seq_len is None else seq_len
        x = torch.nn.functional.pad(x, (0, 0, 0, seq_len - patch_seq_len)).unsqueeze(0)
        x = self.args.prepare_residual_tensor_prefill(x)
        return x

    def forward(
        self,
        x,
        unpadded_seq_len,
        rot_mats,
        trim_before_merger=True,
        _mark=None,
    ):
        """
        Forward pass through the Vision Transformer blocks.

        _mark (QWEN36_VISION_TIMING): called with "tower" after the last block is dispatched and with "merger" after
        the merger; None (default) = no call.

        trim_before_merger=False keeps the merger on the full PADDED length (rows are independent, so the valid
        rows are unchanged) so every device op depends only on the padded sequence bucket, never on the exact
        request size; the caller trims the valid rows on host.

        Args:
            x (ttnn.Tensor): Input tensor [batch_size, 1, seq_len, hidden_dim].
                This arrives fractured along dim=3 (the hidden dim) because
                `prepare_residual_tensor_prefill` shards at load time.
            unpadded_seq_len (int): Real sequence length before padding.
            rot_mats (list): Rotation matrices for positional embeddings.

        Returns:
            ttnn.Tensor: Output tensor, fractured along dim=3 (each device owns
                out_hidden_size/TP), matching the LLM lm_head contract.
        """
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                rot_mats=rot_mats,
            )
        if _mark is not None:
            _mark("tower")

        # The PatchMerger consumes the block output fractured along dim=3
        # directly (its first op is a DistributedLayerNorm that all-gathers
        # internally) and produces a fractured-along-dim=3 output. No
        # pre-merger all-gather is needed.
        if trim_before_merger:
            x = x[:, :, :unpadded_seq_len, :]
        x = self.patch_merger(x)
        if _mark is not None:
            _mark("merger")
        return x


class DropInVisionTransformer(torch.nn.Module):
    """Wraps VisionTransformer to be a drop-in replacement for
    Qwen2_5_VisionTransformerPretrainedModel. It uses the reference model
    for certain preprocessing steps like patch embedding and index calculation.
    """

    def __init__(
        self,
        reference_model,
        model_args: VisionModelArgs,
        dtype=ttnn.bfloat8_b,
        debug=False,
        tt_ccl=None,
    ):
        """
        Initialize the TorchVisionTransformer wrapper.

        Args:
            reference_model (Qwen2_5_VisionTransformerPretrainedModel): Initialized reference HF model instance.
            model_args (VisionModelArgs): Model configuration arguments.
            dtype (ttnn.dtype): Compute dtype for weights.
            debug (bool): If True, run the reference path alongside and report PCC.
            tt_ccl (TT_CCL, optional): Reuse a CCL helper across multiple models.
        """
        super().__init__()
        self.reference_model = reference_model
        self.model_args = model_args
        self.debug = debug

        state_dict = standardize_hf_keys_multimodal(reference_model.state_dict())
        state_dict = convert_hf_to_meta(state_dict, model_args.head_dim)
        state_dict_prefix = model_args.get_state_dict_prefix("VisionTransformer")
        state_dict = {f"{state_dict_prefix}.{k}": v for k, v in state_dict.items()}

        # Initialize TT model
        self.tt_model = VisionTransformer(
            args=model_args,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            dtype=dtype,
            tt_ccl=tt_ccl,
        )

    @property
    def dtype(self):
        return self.reference_model.dtype

    @property
    def spatial_merge_size(self):
        return self.model_args.hf_config.vision_config.spatial_merge_size

    def upload_rows(self, rows):
        """Host bf16 [n_tokens, out_hidden] -> the packed device tensor host_trim returns (host tilize, hidden-sharded,
        DRAM): a host->device write only, never a new device program."""
        mesh = self.model_args.mesh_device
        host_rows = ttnn.from_torch(
            rows,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )
        t0 = time.perf_counter()
        dev = ttnn.to_device(host_rows, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # The host->device write may still be in flight when to_device returns; its host source (rows / host_rows) must
        # not be freed before it lands (rc1e d30u20 11-11: freed host memory reused under an in-flight write corrupted
        # the tail of the rows -> the LAST image read back as another image's / garbage colours). Record an event on the
        # same (default) command queue right after the write and wait for it: then the write is complete and the sources
        # may go. Not a new device program; the splice's next step (_set_vision_merge) reads these rows back to host on
        # the same queue anyway, so it would have waited for the same work.
        ttnn.event_synchronize(ttnn.record_event(mesh))
        rec = _vf.timing_current()
        if rec is not None:  # QWEN36_VISION_TIMING: the packed-row upload (tilize excluded; the write is synchronised above)
            rec.add("rows_upload_ms", 1e3 * (time.perf_counter() - t0))
            rec.add("rows_upload_bytes", int(rows.shape[0]) * int(rows.shape[1]) * 2)
        st = self.__dict__.setdefault("_upload_stats", [0, 0, 0.0])
        st[0] += 1
        st[1] += int(rows.shape[0])
        st[2] += time.perf_counter() - t0
        if st[0] in (1, 10, 100) or st[0] % 1000 == 0:
            logger.info(
                f"vision upload: {st[0]} uploads, {st[1]} rows, write+sync {1e3 * st[2] / st[0]:.2f} ms/upload avg, "
                f"{1e6 * st[2] / max(st[1], 1) * 1024:.0f} us per 1024 rows (1 MP image)"
            )
        return dev

    # ---- QWEN36_VISION_* levers (vision_fast.py; all default off) ----------------------------------------------------
    def _rot_width(self):
        """Columns of the uploaded rotary tables: head_dim (the release; vision_attention pads to padded_head_dim on
        device in every block), or padded_head_dim with QWEN36_VISION_LEAN=rotpad (padded once on the host)."""
        return self.model_args.padded_head_dim if _vf.lean("rotpad") else self.model_args.head_dim

    def _grid_lru(self):
        cap = _vf.grid_cache_cap()
        if cap <= 0:
            return None
        lru = self.__dict__.get("_grid_cache")
        if lru is None or lru.cap != cap:
            lru = self.__dict__["_grid_cache"] = _vf.LRU(cap)
        return lru

    def _fast_path_on(self):
        return bool(
            _vf.grid_cache_cap() > 0
            or self.__dict__.get("_resident_rot")
            or _vf.pipeline_enabled()
            or _vf.patch_linear_enabled()
            or _vf.lean("rotpad")
        )

    def _rot_host(self, grid_thw, unpadded_seq_len, seq_len, rot_w):
        """Host cos / sin [1, 1, seq_len, rot_w] of one grid: the release's tables (rows past the image = identity
        rotation), plus the identity column pad when rot_w > head_dim."""
        _, (cos, sin) = qwen3_5_vision_transformer_preprocess(
            seq_len=unpadded_seq_len,
            grid_thw=grid_thw,
            head_dim=self.model_args.head_dim,
            spatial_merge_size=self.model_args.hf_config.vision_config.spatial_merge_size,
        )
        cos, sin = convert_rope_style_hf_to_meta(cos, sin)
        return _vf.pad_rot_tables(cos, sin, unpadded_seq_len, seq_len, width=rot_w)

    def stage_resident_rot(self, grids):
        """QWEN36_VISION_RESIDENT_GRIDS: upload the rotary tables of these (t, h, w) grids once and keep them in DRAM.
        ONLY before trace capture (warmup_vision): a device tensor allocated after capture can share DRAM with trace
        intermediates and be overwritten by a replay, so nothing is made resident at request time."""
        mesh = self.model_args.mesh_device
        rot_w = self._rot_width()
        res = self.__dict__.setdefault("_resident_rot", {})
        for t, h, w in grids:
            g = torch.tensor([[int(t), int(h), int(w)]], dtype=torch.int32)
            n = int(t) * int(h) * int(w)
            seq_len = _vf.vision_bucket(n)
            key = _vf.grid_key(g, seq_len, rot_w)
            if key in res:
                continue
            res[key] = tuple(
                ttnn.from_torch(
                    x,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                for x in self._rot_host(g, n, seq_len, rot_w)
            )
        return sorted(res)

    def _read_rows(self, tt_out, n_valid, rec):
        """Merged rows of one image to host, valid rows only, owned bf16 (the release's read-back + trim)."""
        out_hidden_full = self.model_args.hf_config.vision_config.out_hidden_size
        mesh = self.model_args.mesh_device
        if rec is not None:
            rec.start()
        host = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3))
        ttnn.deallocate(tt_out)
        if rec is not None:
            rec.lap("readback")
        rows = host.reshape(-1, host.shape[-1])[:n_valid, :out_hidden_full].to(torch.bfloat16).clone()
        if rec is not None:
            rec.lap("trim")
        return rows

    def _forward_fast(self, all_pixel_values, all_grid_thw, rec):
        """The host_trim tower loop with the levers; returns the per-image host rows. Device work per image is the
        release's (same programs on the padded bucket: two table uploads unless resident, the patch upload, the tower,
        the read-back); what changes is where the host constants come from and when the read-back happens.

        - QWEN36_VISION_GRID_CACHE=N: per (grid, bucket, table width) the position-embedding interpolation (torch) and
          the cos / sin tables as tilized HOST tensors; a hit skips their CPU work and uploads the kept host tensors.
        - QWEN36_VISION_RESIDENT_GRIDS: tables staged in DRAM by stage_resident_rot are used as they are (no upload, no
          deallocate).
        - QWEN36_VISION_LEAN=rotpad: tables are uploaded padded_head_dim wide, so vision_attention skips its two device
          pads per block.
        - QWEN36_VISION_PATCH_LINEAR=1: patch embedding as one matmul.
        - QWEN36_VISION_PIPELINE=1 (two or more images): every image is dispatched before the first read-back, so the
          host pre-work of image k+1 runs while the device works on image k. The outputs wait in DRAM (bucket/4 x 5120
          bf16 each) until the loop ends; nothing replays a trace in between."""
        mesh = self.model_args.mesh_device
        vc = self.model_args.hf_config.vision_config
        merge_sq = vc.spatial_merge_size**2
        rot_w = self._rot_width()
        cache = self._grid_lru()
        resident = self.__dict__.get("_resident_rot") or {}
        pipeline = _vf.pipeline_enabled() and len(all_grid_thw) > 1
        wait = not pipeline  # timing: per-stage device waits would undo the overlap, so PIPELINE is clocked without them
        mark = None if rec is None else (lambda name: rec.lap(name, sync=wait, disp=True))
        n_dev = int(mesh.get_num_devices())
        outputs, pending, host_refs = [], [], []
        for grid_thw in all_grid_thw:
            pixel_values = all_pixel_values[: grid_thw.prod(), :]
            all_pixel_values = all_pixel_values[grid_thw.prod() :, :]
            grid_thw = grid_thw.unsqueeze(0)
            unpadded_seq_len = grid_thw.prod(dim=1).sum().item()
            seq_len = _vf.vision_bucket(unpadded_seq_len)
            if rec is not None:
                rec.image(grid_thw[0].tolist(), unpadded_seq_len, seq_len)
            key = _vf.grid_key(grid_thw, seq_len, rot_w)
            ent = cache.get(key) if cache is not None else None
            cos_host, sin_host, pos_embeds = ent if ent is not None else (None, None, None)
            dev_rot = resident.get(key)
            if dev_rot is None and cos_host is None:
                cos_host, sin_host = (
                    ttnn.from_torch(
                        x,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=None,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                    )
                    for x in self._rot_host(grid_thw, unpadded_seq_len, seq_len, rot_w)
                )
            if rec is not None:
                rec.lap("host_rot")  # with the levers this includes the tables' bf16 conversion + tilize
            if _vf.patch_linear_enabled():
                patch_input = _vf.patch_embed_linear(self.reference_model.patch_embed, pixel_values)
            else:
                patch_input = self.reference_model.patch_embed(pixel_values)
            if rec is not None:
                rec.lap("host_patch")
            if pos_embeds is None:
                pos_embeds = self.reference_model.fast_pos_embed_interpolate(grid_thw).detach()
            if cache is not None and ent is None:
                cache.put(key, (cos_host, sin_host, pos_embeds))
            patch_input = patch_input + pos_embeds
            if rec is not None:
                rec.lap("host_pos")
            if dev_rot is not None:
                cos, sin = dev_rot
            else:
                cos = ttnn.to_device(cos_host, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                sin = ttnn.to_device(sin_host, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                # the writes may still be in flight: their host sources stay referenced until every read-back of this
                # call is done (a cache eviction or the end of this iteration must not free them; see upload_rows)
                host_refs += [cos_host, sin_host]
            if rec is not None:
                nb = 0 if dev_rot is not None else 2 * seq_len * rot_w * 2 * n_dev
                rec.lap("up_rot", sync=wait, nbytes=nb)
            tt_input = self.tt_model.prepare_input(patch_input, seq_len)
            if rec is not None:
                rec.lap("up_patch", sync=wait, nbytes=seq_len * int(patch_input.shape[-1]) * 2)
            tt_out = self.tt_model(
                tt_input,
                unpadded_seq_len=unpadded_seq_len,
                rot_mats=[cos, sin],
                trim_before_merger=False,
                **({} if mark is None else dict(_mark=mark)),
            )
            ttnn.deallocate(tt_input)
            if dev_rot is None:
                ttnn.deallocate(cos)
                ttnn.deallocate(sin)
            n_valid = unpadded_seq_len // merge_sq
            if pipeline:
                pending.append((tt_out, n_valid))
                continue
            outputs.append(self._read_rows(tt_out, n_valid, rec))
        for tt_out, n_valid in pending:  # PIPELINE: the read-backs, in image order (clocked on the last image's entry)
            outputs.append(self._read_rows(tt_out, n_valid, rec))
        del host_refs
        return outputs

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, host_trim: bool = False, return_host_rows=False
    ):
        """
        Forward pass mimicking the Qwen3_5_VisionTransformerPretrainedModel interface.

        Args:
            pixel_values (torch.Tensor): Input pixel values tensor (equivalent to hidden_states for the ref model start).
                                         Shape typically [num_patches, hidden_size] or similar before patch_embed.
            grid_thw (torch.Tensor): Tensor describing the grid dimensions (time, height, width) for each image/video.
                                     Shape [num_images_or_videos, 3].

        return_host_rows (with host_trim): return the list of per-item host bf16 [n_tokens_i, out_hidden] rows instead of
            uploading them (the vision cache keeps host rows and uploads per request with upload_rows).

        host_trim: shape-stable mode for serving. The tower + merger run on the padded bucket only (no device op
            depends on the exact patch count, so nothing new is compiled for a new image size after the prefill
            traces were captured -- a program first dispatched after capture can be clobbered by a later trace replay
            and hang the next vision call), the valid merged rows are trimmed on HOST and uploaded as one 2D
            [num_merged_tokens, out_hidden] bf16 TILE tensor, hidden-sharded across the mesh.

        Returns:
            torch.Tensor: Output tensor with shape [1, B, seq_len, out_hidden_size] (host_trim: [n_tokens, hidden]).
        """
        # process pixel_values for each image/video separately
        all_pixel_values = pixel_values
        all_grid_thw = grid_thw
        final_outputs = []
        # QWEN36_VISION_TIMING=1 (default off: rec is None and nothing below changes): stage clock of this call, on the
        # record of the image request being prefilled, or a record of its own (warm-up, demos, the non-spec paths).
        rec = own_rec = None
        if _vf.timing_enabled():
            rec = _vf.timing_current()
            if rec is None:
                _mesh = self.model_args.mesh_device
                rec = own_rec = _vf.timing_begin(sync=lambda: ttnn.synchronize_device(_mesh), standalone=True)
        t_call = time.perf_counter()
        mark = None if rec is None else (lambda name: rec.lap(name, sync=True, disp=True))
        if host_trim and self._fast_path_on():
            # QWEN36_VISION_GRID_CACHE / _RESIDENT_GRIDS / _PIPELINE / _PATCH_LINEAR / _LEAN=rotpad (all default off): the
            # serving loop with those levers. The loop below stays the release's.
            final_outputs = self._forward_fast(all_pixel_values, all_grid_thw, rec)
            all_grid_thw = []
        # todo)) refactor this code to leverage tt-mesh's ttnn.ShardTensorToMesh(mesh_device, dim=batch_size_dim) for data parallelism
        for grid_thw in all_grid_thw:
            # --- pick out the pixel_values for this users' images (grid_thw.prod() pixels) ---
            pixel_values = all_pixel_values[: grid_thw.prod(), :]
            all_pixel_values = all_pixel_values[grid_thw.prod() :, :]
            # --- Preprocessing ---
            # 1. Calculate total unpadded sequence length
            grid_thw = grid_thw.unsqueeze(0)
            unpadded_seq_len = grid_thw.prod(dim=1).sum().item()
            # Calculate padded sequence length (divisible by 2048) required by models/tt_transformers/tt/attention.py::forward_prefill
            # (the release rule (n // 2048 + 1) * 2048 unless QWEN36_VISION_BUCKET_EXACT / _1024 is set: vision_fast.py)
            seq_len = _vf.vision_bucket(unpadded_seq_len)
            if rec is not None:
                rec.image(grid_thw[0].tolist(), unpadded_seq_len, seq_len)

            # 2. Use preprocessing function from reference/functional to get indices and embeddings
            cu_seqlens, position_embeddings = qwen3_5_vision_transformer_preprocess(
                seq_len=unpadded_seq_len,
                grid_thw=grid_thw,
                head_dim=self.model_args.head_dim,
                spatial_merge_size=self.model_args.hf_config.vision_config.spatial_merge_size,
            )
            if rec is not None:
                rec.lap("host_rot")

            # 3. Use reference model's patch embedding
            patch_input = self.reference_model.patch_embed(pixel_values)
            if rec is not None:
                rec.lap("host_patch")
            pos_embeds = self.reference_model.fast_pos_embed_interpolate(grid_thw)
            patch_input = patch_input + pos_embeds
            if rec is not None:
                rec.lap("host_pos")

            # 4. Prepare rotational embeddings (cos, sin) -> pad -> convert to TT tensors
            cos_orig, sin_orig = position_embeddings
            cos_orig, sin_orig = convert_rope_style_hf_to_meta(cos_orig, sin_orig)
            # pad sequence length with cos = 1, sin = 0 (identity rotation)
            cos_padded = (
                torch.nn.functional.pad(cos_orig, (0, 0, 0, seq_len - unpadded_seq_len), value=1)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            sin_padded = (
                torch.nn.functional.pad(sin_orig, (0, 0, 0, seq_len - unpadded_seq_len), value=0)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            if rec is not None:
                rec.lap("host_rot")  # the table conversion + padding, added to the rotary stage
            # Convert to TT tensors on the mesh device
            cos = ttnn.from_torch(
                cos_padded,
                dtype=ttnn.bfloat16,  # Use bfloat16 for RoPE
                layout=ttnn.TILE_LAYOUT,
                device=self.model_args.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.model_args.mesh_device),
            )
            sin = ttnn.from_torch(
                sin_padded,
                dtype=ttnn.bfloat16,  # Use bfloat16 for RoPE
                layout=ttnn.TILE_LAYOUT,
                device=self.model_args.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.model_args.mesh_device),
            )
            rot_mats = [cos, sin]
            if rec is not None:
                # host bf16 + tilize + the write to every chip; bytes = both tables x the chips they are replicated to
                n_dev = int(self.model_args.mesh_device.get_num_devices())
                rec.lap("up_rot", sync=True, nbytes=2 * seq_len * int(cos_padded.shape[-1]) * 2 * n_dev)

            # 5. Prepare input tensor for the TT model using window_index
            tt_input = self.tt_model.prepare_input(patch_input, seq_len)
            if rec is not None:
                # row pad + host bf16 + tilize + the hidden-sharded write
                rec.lap("up_patch", sync=True, nbytes=seq_len * int(patch_input.shape[-1]) * 2)

            # --- TT Model Execution ---
            tt_out = self.tt_model(
                tt_input,
                unpadded_seq_len=unpadded_seq_len,
                rot_mats=rot_mats,  # Use rot_mats generated in this forward pass
                trim_before_merger=not host_trim,
                **({} if mark is None else dict(_mark=mark)),
            )

            # deallocate device tensors that are not needed by decode
            ttnn.deallocate(tt_input)
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
            ttnn.deallocate(rot_mats[0])
            ttnn.deallocate(rot_mats[1])

            if host_trim:
                out_hidden_full = self.model_args.hf_config.vision_config.out_hidden_size
                merge_sq = self.model_args.hf_config.vision_config.spatial_merge_size**2
                n_valid = unpadded_seq_len // merge_sq
                mesh = self.model_args.mesh_device
                if rec is not None:
                    rec.start()  # the deallocations above are not part of the read-back
                host = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3))
                ttnn.deallocate(tt_out)
                if rec is not None:
                    rec.lap("readback")
                # .clone(): own the rows. to_torch's result (and the reshape / slice / same-dtype .to views of it) can
                # alias a host buffer ttnn frees and reuses; the vision cache keeps these rows across requests
                # (rc1e d30u20 11-11: cached rows later read back as another image's / garbage colours).
                final_outputs.append(
                    host.reshape(-1, host.shape[-1])[:n_valid, :out_hidden_full].to(torch.bfloat16).clone()
                )
                if rec is not None:
                    rec.lap("trim")
                continue

            # --- Postprocessing ---
            # 1. Extract the relevant output part and adjust shape (matching test logic).
            # The merger output is fractured along dim=3, so each device owns
            # out_hidden_size/TP channels.
            out_hidden_size = (
                self.model_args.hf_config.vision_config.out_hidden_size // self.model_args.cluster_shape[1]
            )
            # Output shape from TT is [1, B=1, S, H_out_padded], slice H and squeeze B, batch dims.
            # The slice/reshape args are GLOBAL shapes; the tensor is already
            # fractured along dim=3 and ttnn handles the per-device extents internally.
            final_output = ttnn.reshape(tt_out[:, 0:1, :, :out_hidden_size], (-1, out_hidden_size))
            # ttnn.deallocate(tt_out)

            if self.debug:
                logger.info(f"DropInVisionTransformer: Debug enabled, running reference model...")
                reference_output = self.reference_model.forward(pixel_values, grid_thw)
                _, pcc = comp_pcc(reference_output, final_output)
                logger.info(f"DropInVisionTransformer: PCC to reference model: {pcc}")

            # 2. The merger already produces a tensor fractured along the hidden
            # dim (dim=3 in 4D / dim=1 in the 2D-reshaped view), which is the
            # desired output sharding.
            # final_output_sharded = final_output

            # 3. Aggregate in batched users list
            final_outputs.append(tt_out)

        if rec is not None:
            rec.add("tower_call_ms", 1e3 * (time.perf_counter() - t_call))
            if own_rec is not None and (return_host_rows or not host_trim):
                _vf.timing_end(own_rec, logger.info)  # no row upload follows: the call's line is complete
                own_rec = None
        if host_trim:
            if return_host_rows:
                return final_outputs
            if own_rec is not None:
                try:
                    return self.upload_rows(torch.cat(final_outputs, dim=0))
                finally:
                    _vf.timing_end(own_rec, logger.info)
            # Tilize on the HOST (no device= here): the row count depends on the request's image size, and a device
            # tilize (Tilize / TilizeWithValPadding) is a program keyed by that shape, i.e. one NEW program compiled
            # after the prefill/decode traces are captured per distinct image size.
            return self.upload_rows(torch.cat(final_outputs, dim=0))

        # concatenate all the outputs. With a single image, ttnn.concat aliases
        # its lone input, so deallocating the sources would free the return.
        if len(final_outputs) == 1:
            return final_outputs[0]
        tt_out = ttnn.concat(final_outputs, dim=1)
        for t in final_outputs:
            ttnn.deallocate(t)
        return tt_out
