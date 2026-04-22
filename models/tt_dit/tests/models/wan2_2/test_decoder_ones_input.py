# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full VAE decoder run on an all-ones latent input, for visual inspection.

Feeds the real (HF-weights) VAE decoder an all-ones latent of shape
[1, 16, 7, 60, 104] (480p, 7-frame cached) on a 2x4 mesh and saves the
decoded output as an MP4 so any W-boundary mosaic seam is immediately
visible.  Also prints a per-column mean profile so the seam can be
located numerically at W = 208, 416, 624 (the three W-device boundaries
in the 832-px-wide output on a 4-chip W split).

No PCC / torch-ref check.  The point is to run the end-to-end network
and look at the picture.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn

from ....models.vae.vae_wan2_1 import WanDecoder
from ....parallel.config import ParallelFactor, VaeHWParallelConfig
from ....parallel.manager import CCLManager
from ....utils.conv3d import conv_pad_height, conv_pad_in_channels
from ....utils.tensor import typed_tensor_2dshard


def _export_to_video(frames: np.ndarray, video_path: str, fps: int = 16) -> None:
    """Write uint8 [T, H, W, 3] frames to ``video_path`` via imageio_ffmpeg's bundled ffmpeg."""
    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    t, h, w = frames.shape[:3]
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        video_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode} when writing {video_path}")


def _print_column_seam_profile(
    frames_tchw: torch.Tensor,
    w_factor: int,
    tag: str,
) -> None:
    """Print per-column mean (T-averaged) and PER-FRAME raw-pixel seam jumps.

    A clean decoder output has a smooth column profile.  A W-seam shows up as
    a sharp spike/drop at columns w = W * d / w_factor for d = 1..w_factor-1.

    IMPORTANT: with conv-weights-to-1 and ones input, most frames converge to a
    uniform constant (std ~ 0).  Averaging over T washes out the seam because
    only a few frames (typically t=0 of each chunk) show it.  So we also print
    a per-t seam jump.
    """
    T, C, H, W = frames_tchw.shape
    col_profile = frames_tchw.float().mean(dim=(0, 1, 2))  # (W,)
    assert col_profile.shape == (W,)

    logger.info(f"=== {tag}: column profile (T={T}, C={C}, H={H}, W={W}, w_factor={w_factor}) ===")
    overall_mean = col_profile.mean().item()
    overall_std = col_profile.std().item()
    logger.info(f"  T-AVG col-mean: mean={overall_mean:.4f}  std={overall_std:.4f}")
    logger.info(f"  col[0]={col_profile[0].item():.4f}  col[W-1]={col_profile[W - 1].item():.4f}")

    for d in range(1, w_factor):
        s = W * d // w_factor
        left = col_profile[s - 1].item() if s - 1 >= 0 else float("nan")
        here = col_profile[s].item()
        right = col_profile[s + 1].item() if s + 1 < W else float("nan")
        jump_l = abs(here - left)
        jump_r = abs(here - right)
        logger.info(
            f"  [T-AVG] seam col w={s:4d}: col[{s - 1}]={left:.4f}  col[{s}]={here:.4f}  "
            f"col[{s + 1}]={right:.4f}  |jumpL|={jump_l:.4f}  |jumpR|={jump_r:.4f}"
        )

    # Per-frame seam magnitude: max_|jump| over H, channel-averaged.
    logger.info("=== per-frame seam jump |max over H| at each W-boundary (ch-averaged) ===")
    header = "  t  " + "  ".join([f"  s={W * d // w_factor:<4d} (|jL|,|jR|)" for d in range(1, w_factor)])
    logger.info(header)
    for t in range(T):
        row_parts = [f"{t:3d}"]
        for d in range(1, w_factor):
            s = W * d // w_factor
            # raw pixel jump at (t,h,s-1)->(t,h,s), channel-averaged
            jL = (frames_tchw[t, :, :, s] - frames_tchw[t, :, :, s - 1]).abs().mean(0).max().item()
            jR = (frames_tchw[t, :, :, s] - frames_tchw[t, :, :, s + 1]).abs().mean(0).max().item()
            row_parts.append(f"({jL:6.3f},{jR:6.3f})")
        logger.info("  " + "   ".join(row_parts))


@pytest.mark.parametrize(
    "B, C, T, H, W, target_height, target_width, t_chunk_size, cached, vae_call_chunk",
    [
        # 81-frame 480p production config (matches pipeline_wan.py defaults):
        #   num_frames = 81  ->  num_latent_frames = (81 - 1) // 4 + 1 = 21
        #   vae_t_chunk_size = 7   (WanDecoder ctor kwarg AND call-time kwarg)
        #   cached = True          (cached=(vae_t_chunk_size is not None))
        # So we feed a full 21-latent-frame input and the decoder internally
        # chunks it into 3 chunks of 7 latent frames with the temporal cache,
        # producing 81 output frames.
        (1, 16, 21, 60, 104, 480, 832, 7, True, 7),
    ],
    ids=["480p_81f_production"],
)
@pytest.mark.parametrize(
    "mesh_device, h_axis, w_axis, num_links",
    [
        ((2, 4), 0, 1, 1),
        ((2, 2), 0, 1, 1),  # fallback for 4-device (2x2) mesh
    ],
    ids=["2x4_h0_w1", "2x2_h0_w1"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "use_fused",
    [True, False],
    ids=["fused", "standalone_np_conv3d"],
)
@pytest.mark.parametrize(
    "single_block",
    [False, True],
    ids=["multi_block", "single_block"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.timeout(900)
def test_wan_decoder_ones_input(
    mesh_device,
    B,
    C,
    T,
    H,
    W,
    target_height,
    target_width,
    t_chunk_size,
    cached,
    vae_call_chunk,
    h_axis,
    w_axis,
    num_links,
    use_fused,
    single_block,
):
    """Run the full TT VAE decoder on an all-ones latent; save MP4 + column seam profile.

    With ``use_fused=True``  : default path (fused ``neighbor_pad_conv3d``).
    With ``use_fused=False`` : standalone ``neighbor_pad_persistent_buffer`` +
    separate ``conv3d`` dispatch.  Comparing the two MP4s tells us whether the
    W-seam is introduced by the fused op or by the halo exchange itself.
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan as TorchAutoencoderKLWan

    # single_block has no meaning for the standalone path (conv3d is dispatched
    # stand-alone, the W-halo is already materialized before the conv runs).
    if single_block and not use_fused:
        pytest.skip("single_block only affects use_fused=True; skipping redundant standalone run")

    logger.info(f"WanDecoder use_fused={use_fused} for every halo-needing conv; timeconv stays standalone")

    _run_decoder_ones(
        mesh_device=mesh_device,
        B=B,
        C=C,
        T=T,
        H=H,
        W=W,
        target_height=target_height,
        target_width=target_width,
        t_chunk_size=t_chunk_size,
        cached=cached,
        vae_call_chunk=vae_call_chunk,
        h_axis=h_axis,
        w_axis=w_axis,
        num_links=num_links,
        use_fused=use_fused,
        single_block=single_block,
        TorchAutoencoderKLWan=TorchAutoencoderKLWan,
    )


def _override_mid_block_full_cin(decoder) -> None:
    """Force mid_block's resblock convs to C_in_block=384 (full C_in, no channel splitting).

    mid_block has 2 WanResidualBlocks, each with 2 WanCausalConv3d (384->384, 3x3x3).
    Production blocking: (C_in_block=96, C_out_block=96, T=1, H=32, W=4).
    Override C_in_block to 384 so only 1 C_in block per core (no C_in parallelism).
    Keep spatial blocking at production values.
    """
    for name, child in decoder.decoder.mid_block._children.items():
        if name == "resnets":
            for _, resblock in child._children.items():
                for conv_name, conv in resblock._children.items():
                    cfg = getattr(conv, "conv_config", None)
                    if cfg is None or not getattr(conv, "_use_fused", False):
                        continue
                    before_cin = cfg.C_in_block
                    cfg.C_in_block = 384
                    logger.info(
                        f"full-C_in override on mid_block.{conv_name}: "
                        f"C_in_block {before_cin} -> {cfg.C_in_block}  "
                        f"(T,H,W)_out_block=({cfg.T_out_block},{cfg.H_out_block},{cfg.W_out_block})"
                    )


def _run_decoder_ones(
    *,
    mesh_device,
    B,
    C,
    T,
    H,
    W,
    target_height,
    target_width,
    t_chunk_size,
    cached,
    vae_call_chunk,
    h_axis,
    w_axis,
    num_links,
    use_fused,
    single_block,
    TorchAutoencoderKLWan,
):
    torch.manual_seed(0)
    dtype = ttnn.DataType.BFLOAT16

    base_dim = 96
    z_dim = 16
    dim_mult = [1, 2, 4, 4]
    num_res_blocks = 2
    attn_scales: list = []
    temperal_downsample = [False, True, True]
    out_channels = 3
    is_residual = False

    # Real weights from HF.
    logger.info("Loading real VAE weights from Wan-AI/Wan2.2-T2V-A14B-Diffusers ...")
    torch_model = TorchAutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="vae")
    torch_model.eval()

    # --- DEBUG (disabled): override conv / linear weights & biases so interior
    # output is a clean constant.  Useful to visually isolate W-seam jumps.
    # Re-enable by un-commenting when you want the ones-weights signal.
    #
    #   conv weights  -> 1.0     (input all-ones + weight all-ones => conv output = kT*kH*kW*C_in)
    #   conv biases   -> 0.0
    #   norm weights  -> 1.0     (RMS/GroupNorm gamma -> identity)
    #   linear/proj   -> 1/C_in  (keeps scale bounded through attn/mlp)
    # with torch.no_grad():
    #     sd = torch_model.state_dict()
    #     num_conv_w = num_conv_b = num_norm_w = 0
    #     for name, p in sd.items():
    #         if name.endswith(".weight"):
    #             if p.dim() >= 3:
    #                 p.fill_(1.0)
    #                 num_conv_w += 1
    #             elif p.dim() == 1:
    #                 p.fill_(1.0)
    #                 num_norm_w += 1
    #             elif p.dim() == 2:
    #                 _, C_in = p.shape
    #                 p.fill_(1.0 / float(C_in))
    #                 num_conv_w += 1
    #         elif name.endswith(".bias"):
    #             p.zero_()
    #             if p.dim() == 1:
    #                 num_conv_b += 1
    #     logger.info(
    #         f"weight override: conv/linear weights -> 1, biases -> 0, norm gamma -> 1 "
    #         f"(conv_w={num_conv_w}, conv_b={num_conv_b}, norm_w={num_norm_w})"
    #     )
    #     torch_model.load_state_dict(sd)

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[h_axis], mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[w_axis], mesh_axis=w_axis),
    )
    tt_model = WanDecoder(
        base_dim=base_dim,
        z_dim=z_dim,
        dim_mult=dim_mult,
        num_res_blocks=num_res_blocks,
        attn_scales=attn_scales,
        temperal_downsample=temperal_downsample,
        out_channels=out_channels,
        is_residual=is_residual,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        dtype=dtype,
        target_height=target_height,
        target_width=target_width,
        t_chunk_size=t_chunk_size,
        cached=cached,
        use_fused=use_fused,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    if single_block and use_fused:
        _override_mid_block_full_cin(tt_model)
    elif single_block and not use_fused:
        logger.warning("single_block=True has no effect for use_fused=False; skipping override")

    # ----- All-ones latent input -----
    torch_input = torch.ones(B, C, T, H, W, dtype=torch.float32)
    logger.info(f"Feeding all-ones latent, shape={tuple(torch_input.shape)}")

    tt_input_tensor = torch_input.permute(0, 2, 3, 4, 1)  # BTHWC
    tt_input_tensor = conv_pad_in_channels(tt_input_tensor)
    tt_input_tensor, logical_h = conv_pad_height(tt_input_tensor, parallel_config.height_parallel.factor)
    if logical_h != tt_input_tensor.shape[2]:
        logger.info(f"padded latent H {logical_h} -> {tt_input_tensor.shape[2]}")
    tt_input_tensor = typed_tensor_2dshard(
        tt_input_tensor,
        mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        shard_mapping={h_axis: 2, w_axis: 3},
        dtype=ttnn.bfloat16,
    )

    # ----- TT decoder -----
    # Call with vae_call_chunk (matches pipeline_wan.py: vae(..., t_chunk_size=vae_t_chunk_size)).
    # For 81-frame prod: T=21 latent, call-time t_chunk_size=7 -> 3 internal chunks using cache.
    logger.info(
        f"running TT decoder (ctor t_chunk_size={t_chunk_size}, cached={cached}, "
        f"call-time t_chunk_size={vae_call_chunk}, latent_T={T})"
    )
    start = time.time()
    tt_output, new_logical_h = tt_model(tt_input_tensor, logical_h, t_chunk_size=vae_call_chunk)

    concat_dims = [None, None]
    concat_dims[h_axis] = 3
    concat_dims[w_axis] = 4
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
    )
    logger.info(f"TT decoder time: {time.time() - start:.2f}s   output shape: {tuple(tt_output_torch.shape)}")

    # Trim to expected logical output (C_out=3, H=target_height, W=target_width).
    if tt_output_torch.shape[1] != out_channels:
        tt_output_torch = tt_output_torch[:, :out_channels]
    if new_logical_h != tt_output_torch.shape[3]:
        tt_output_torch = tt_output_torch[:, :, :, :new_logical_h, :]

    # [B, C, T_out, H_out, W_out] -> [T_out, C, H_out, W_out]
    frames_tchw = tt_output_torch[0].permute(1, 0, 2, 3).contiguous().float()
    T_out, C_out, H_out, W_out = frames_tchw.shape
    logger.info(f"Decoded frames: T={T_out} C={C_out} H={H_out} W={W_out}")

    # ----- Per-column seam profile (the numeric version of the visual check) -----
    w_factor = parallel_config.width_parallel.factor
    _print_column_seam_profile(frames_tchw, w_factor=w_factor, tag="TT decoder output")

    # ----- Save artifacts -----
    # Split filenames by (path, block-mode) so runs don't overwrite each other.
    path_tag = "fused" if use_fused else "standalone"
    block_tag = "singleblk" if single_block else "multiblk"
    repo_root = Path(__file__).resolve().parents[5]
    out_pt = repo_root / f"wan_decoder_ones_480p_81f_{path_tag}_{block_tag}.pt"
    out_mp4 = repo_root / f"wan_decoder_ones_480p_81f_{path_tag}_{block_tag}.mp4"

    torch.save(frames_tchw, out_pt)
    logger.info(f"Saved raw frames tensor -> {out_pt}")

    # Normalize to uint8 [T, H, W, 3] for mp4.  Wan VAE decoder output is roughly in [-1, 1];
    # clamp and rescale to [0, 255].
    frames_thwc = frames_tchw.permute(0, 2, 3, 1).clamp(-1.0, 1.0)
    frames_uint8 = ((frames_thwc + 1.0) * 127.5).round().clamp(0, 255).byte().cpu().numpy()
    _export_to_video(frames_uint8, str(out_mp4), fps=16)
    logger.info(f"Saved video -> {out_mp4}")

    # ----- Compare against standalone reference -----
    ref_pt = repo_root / "wan_decoder_ones_480p_81f_standalone_multiblk.pt"
    if not ref_pt.exists():
        ref_pt = repo_root / "wan_decoder_ones_480p_81f_standalone.pt"
    if ref_pt.exists() and use_fused:
        ref = torch.load(ref_pt, map_location="cpu").float()
        cur = frames_tchw.float()
        if ref.shape == cur.shape:
            diff = (cur - ref).abs()
            max_err = diff.max().item()
            mean_err = diff.mean().item()
            flat_ref = ref.reshape(-1)
            flat_cur = cur.reshape(-1)
            pcc = torch.corrcoef(torch.stack([flat_ref, flat_cur]))[0, 1].item()
            logger.info(f"=== vs standalone ref ({ref_pt.name}) ===")
            logger.info(f"  PCC      = {pcc:.6f}")
            logger.info(f"  max_err  = {max_err:.6f}")
            logger.info(f"  mean_err = {mean_err:.6f}")

            T, CC, HH, WW = cur.shape
            for d in range(1, w_factor):
                s = WW * d // w_factor
                cur_jump = (cur[:, :, :, s] - cur[:, :, :, s - 1]).abs().mean().item()
                ref_jump = (ref[:, :, :, s] - ref[:, :, :, s - 1]).abs().mean().item()
                logger.info(
                    f"  W-boundary col={s}: cur_jump={cur_jump:.4f}  ref_jump={ref_jump:.4f}  delta={abs(cur_jump - ref_jump):.4f}"
                )

            if pcc > 0.9999 and max_err < 0.05:
                logger.info("PASS: fused output matches standalone reference (no seam regression)")
            else:
                logger.warning(f"MISMATCH: PCC={pcc:.6f} max_err={max_err:.6f} — possible seam artifact")
        else:
            logger.warning(f"Shape mismatch: cur={cur.shape} ref={ref.shape}, skipping comparison")
    elif use_fused:
        logger.warning(f"No standalone reference found at {ref_pt}, skipping comparison")

    logger.info(
        "DONE. 81-frame 480p production config on 2x4 mesh. "
        "Open the MP4 and look for 3 vertical seam lines at x = 208, 416, 624. "
        "Per-frame seam table above shows which temporal frames exhibit the jump."
    )
