# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from .pipeline_wan import WanPipeline


class WanPipeline21(WanPipeline):
    """
    Single-transformer Wan2.1 T2V pipeline.

    Identical transformer architecture to Wan2.2 but uses one model for all denoising
    steps (no MoE second transformer). Loads from Wan-AI/Wan2.1-T2V-14B-Diffusers by
    default.
    """

    @staticmethod
    def create_pipeline(
        mesh_device,
        *,
        checkpoint_name="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        scheduler=None,
        sp_axis=None,
        tp_axis=None,
        num_links=None,
        dynamic_load=None,
        topology=None,
        is_fsdp=None,
        vae_t_chunk_size=None,
        sdpa_t_fracture_w_only=None,
        target_height: int = 0,
        target_width: int = 0,
        num_frames: int = 81,
        run_warmup: bool = True,
    ):
        return WanPipeline.create_pipeline(
            mesh_device,
            checkpoint_name=checkpoint_name,
            scheduler=scheduler,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            num_links=num_links,
            dynamic_load=dynamic_load,
            topology=topology,
            is_fsdp=is_fsdp,
            pipeline_class=WanPipeline21,
            vae_t_chunk_size=vae_t_chunk_size,
            sdpa_t_fracture_w_only=sdpa_t_fracture_w_only,
            target_height=target_height,
            target_width=target_width,
            num_frames=num_frames,
            boundary_ratio=None,
            run_warmup=run_warmup,
        )
