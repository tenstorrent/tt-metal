# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PaddleOCR-VL vision tower: qwen36 blocks on device, HF embeddings and permutation on the host, padded per VISION_BUCKETS."""

from __future__ import annotations

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule

# Encoder kernels are shared with Qwen3.5's Blackhole tower (identical dims),
# not duplicated; tests/test_vision_tower_pcc.py gates it, probe_vision_layers.py localizes a break.
from models.demos.blackhole.qwen36.tt.vision.patch_merger import PatchMerger
from models.demos.blackhole.qwen36.tt.vision.vision_block import VisionBlock
from models.demos.blackhole.qwen36.tt.vision.vision_distributed_layernorm import DistributedLayerNorm
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import get_rot_transformation_mat

from ..weight_mapping import PATCH_EMBED_BIAS, PATCH_EMBED_WEIGHT, POS_EMBED
from .functional import preprocess

# Padded patch counts the tower will compile for. Every bucket is a multiple
# of 128, 1024 and (above 2048) MAX_QKV_MM_SEQ_LEN. 6144 is the measured
# resolution ceiling (1204224 px); see commit history for the CER data and
# why higher needs tiling the page rather than a bigger single image.
VISION_BUCKETS = (1024, 2048, 4096, 6144)

# One grid per bucket, used to compile each program set during warmup. Both
# dimensions are even (spatial merge is 2); any count inside the bucket compiles
# the same programs, since the tower runs at the bucket size throughout.
WARMUP_GRIDS = {1024: (32, 32), 2048: (32, 64), 4096: (64, 64), 6144: (64, 80)}


def bucket_for(n_patches: int) -> int:
    for b in VISION_BUCKETS:
        if n_patches <= b:
            return b
    raise ValueError(
        f"{n_patches} patches exceeds the largest bucket {VISION_BUCKETS[-1]}; "
        f"the image processor should have clamped this to {VISION_BUCKETS[-1]} "
        f"({VISION_BUCKETS[-1] // 4} merged tokens, {VISION_BUCKETS[-1] * 196} pixels). "
        "Check max_pixels in mm_processor_kwargs."
    )


class HostEmbeddings(torch.nn.Module):
    """Patch and position embedding on the host, via the HuggingFace module so it cannot drift."""

    def __init__(self, hf_vision_config, host_weights: dict[str, torch.Tensor], dtype=torch.bfloat16):
        super().__init__()
        from transformers.models.paddleocr_vl.modeling_paddleocr_vl import PaddleOCRVisionEmbeddings

        self.embeddings = PaddleOCRVisionEmbeddings(hf_vision_config)
        # ``load_state_dict`` renames the conv to ``patch_embedding._linear`` and
        # the table to ``position_embedding.positional_embedding``; map back to
        # the names the HF module declares.
        self.embeddings.load_state_dict(
            {
                "patch_embedding.weight": host_weights[PATCH_EMBED_WEIGHT],
                "patch_embedding.bias": host_weights[PATCH_EMBED_BIAS],
                "position_embedding.weight": host_weights[POS_EMBED],
            },
            strict=True,
        )
        self.embeddings.to(dtype).eval()
        self.dtype = dtype

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """``[N, 3, p, p]`` patches -> ``[N, hidden]`` embeddings, position added."""
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(0)  # the HF module expects a batch dim
        return self.embeddings(pixel_values.to(self.dtype), grid_thw=grid_thw)


class VisionTransformer(LightweightModule):
    """Device half (blocks, ln_post, merger) over an already embedded, permuted, padded sequence."""

    def __init__(self, args, dtype, state_dict, weight_cache_path, tt_ccl=None):
        super().__init__()
        self.args = args
        self.dtype = dtype
        self.weight_cache_path = weight_cache_path
        self.tt_ccl = tt_ccl if tt_ccl is not None else TT_CCL(args.mesh_device)

        transformation_mat_torch = get_rot_transformation_mat(args.head_dim)
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

        self.blocks = [
            VisionBlock(
                mesh_device=args.mesh_device,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                dtype=dtype,
                transformation_mats=self.transformation_mats,
                args=args,
                tt_ccl=self.tt_ccl,
            )
            for i in range(args.n_vision_layers)
        ]

        # PaddleOCR normalizes the tower output before the projector; Qwen3.5 has
        # no equivalent, so this sits between the blocks and the reused merger.
        self.ln_post = DistributedLayerNorm(
            device=args.mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            state_dict_prefix="visual.ln_post",
            tt_ccl=self.tt_ccl,
            weight_cache_path=weight_cache_path,
            eps=args.vision_layer_norm_eps,
        )

        self.patch_merger = PatchMerger(
            mesh_device=args.mesh_device,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            state_dict_prefix=args.get_state_dict_prefix("PatchMerger"),
            dtype=dtype,
            tt_ccl=self.tt_ccl,
        )

    def prepare_input(self, patch_input: torch.Tensor, seq_len: int) -> ttnn.Tensor:
        """Pad the patch sequence to ``seq_len`` and shard it along hidden."""
        n = patch_input.shape[0]
        x = torch.nn.functional.pad(patch_input, (0, 0, 0, seq_len - n)).unsqueeze(0)
        return self.args.prepare_residual_tensor_prefill(x)

    def __call__(self, x, unpadded_seq_len, rot_mats):
        return self.forward(x, unpadded_seq_len, rot_mats)

    def forward(self, x, unpadded_seq_len: int, rot_mats):
        """``x`` is fractured along hidden; every op runs at bucket size so warmup covers all shapes."""
        # [0, n, S]: real patches never attend to padding; always 3 entries, so one program set per bucket.
        cu_window_seqlens = ttnn.from_torch(
            torch.tensor([0, unpadded_seq_len, x.shape[-2]], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.args.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
        )
        for block in self.blocks:
            x = block(x, rot_mats=rot_mats, cu_window_seqlens=cu_window_seqlens)
        ttnn.deallocate(cu_window_seqlens)

        x = self.ln_post(x)
        return self.patch_merger(x)


class DropInVisionTransformer(torch.nn.Module):
    """Host seam plus device tower; forward returns what PaddleOCRVLModel.get_image_features does."""

    def __init__(
        self,
        model_args,
        device_state_dict: dict[str, torch.Tensor],
        host_state_dict: dict[str, torch.Tensor],
        dtype=ttnn.bfloat8_b,
        tt_ccl=None,
    ):
        super().__init__()
        self.model_args = model_args
        self.host_embeddings = HostEmbeddings(model_args.hf_config.vision_config, host_state_dict)
        self.tt_model = VisionTransformer(
            args=model_args,
            dtype=dtype,
            state_dict=device_state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            tt_ccl=tt_ccl,
        )
        self._warmed: set[int] = set()

    @property
    def spatial_merge_size(self) -> int:
        return self.model_args.spatial_merge_size

    @torch.no_grad()
    def warmup_buckets(self) -> int:
        """Compile every bucket before trace capture; a later compile corrupts the parked trace (#48536)."""
        patch = self.model_args.patch_size
        for bucket, (h, w) in WARMUP_GRIDS.items():
            n = h * w
            assert bucket_for(n) == bucket, f"warmup grid {h}x{w} maps to {bucket_for(n)}, not {bucket}"
            if bucket in self._warmed:
                continue
            logger.info(f"vision tower: warming bucket {bucket} with a {h}x{w} grid ({n} patches)")
            self.forward(
                torch.zeros(n, 3, patch, patch, dtype=torch.bfloat16),
                torch.tensor([[1, h, w]], dtype=torch.int32),
            )
        return len(self._warmed)

    def _rot_mats(self, cos: torch.Tensor, sin: torch.Tensor):
        mesh = self.model_args.mesh_device
        out = []
        for t in (cos, sin):
            out.append(
                ttnn.from_torch(
                    t.unsqueeze(0).unsqueeze(0),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
            )
        return out

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """One tower pass per image (multi-image OCR prompts are rare); returns the concatenated embeddings."""
        outputs = []
        remaining = pixel_values

        for row in grid_thw:
            row = row.unsqueeze(0)
            n = int(row.prod())
            patches, remaining = remaining[:n], remaining[n:]

            embedded = self.host_embeddings(patches, row)  # [n, dim], raster order

            pre = preprocess(
                row,
                head_dim=self.model_args.head_dim,
                spatial_merge_size=self.model_args.spatial_merge_size,
                bucket=bucket_for(n),
                permute=True,
            )
            embedded = embedded[pre["perm"]]  # raster -> merge-block order

            if pre["seq_len"] not in self._warmed:
                logger.info(f"vision tower: first use of bucket {pre['seq_len']} (patches={n}), compiling")
                self._warmed.add(pre["seq_len"])

            tt_in = self.tt_model.prepare_input(embedded.float(), pre["seq_len"])
            tt_out = self.tt_model(tt_in, unpadded_seq_len=n, rot_mats=self._rot_mats(pre["cos"], pre["sin"]))

            merged = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(self.model_args.mesh_device, dim=3))
            out_dim = self.model_args.hf_config.vision_config.out_hidden_size
            merged = merged[:, 0, :, :out_dim].reshape(-1, out_dim)
            outputs.append(merged[: n // (self.spatial_merge_size**2)])

        return torch.cat(outputs, dim=0)
