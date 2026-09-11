# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The T-sharded audio ENCODER across every (factor, axis) a mesh supports, checked at PSNR level.

Per mesh: the unsharded baseline, then one T-shard per mesh axis longer than 1, with the factor equal to that axis's
length (the pipeline's choice under MINIMAX_H3_AUDIO_T_SHARD=1). On 4x8 that is (4, axis 0) and (8, axis 1); on a
galaxy opened as 1x32 it is (32, axis 1), the quad's inter-host axis length; on the 4x32 quad both.

Each sharded run reports latency, PSNR of mean and logs vs the unsharded encode, the PSNR of the LAST 20 latents
(a shard-boundary defect shows up in the tail first), and a run-twice determinism check (a finite repeat PSNR means a
race or an uninitialized read, not a math error). A factor that hangs is caught by the op timeout and reported, and
the remaining factors still run; the asserts come at the end.

    TT_METAL_OPERATION_TIMEOUT_SECONDS=300 pytest models/tt_dit/tests/models/minimax_h3/test_audio_encode_tshard_matrix.py -k mesh4x8

The meshes open with 8 KB fabric packets like the pipeline: `neighbor_pad_async` corrupts rows wider than one packet,
and the encoder's final trunk conv exchanges 8192-byte rows (fp32, C=2048).
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

from ....models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
from ....models.audio_vae.minimax_h3.encoder_minimax_h3_audio import MiniMaxH3AudioEncoder
from ....parallel.config import ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.test import line_params_8k, ring_params_8k_req_exact_devices
from .common import load_config, psnr, weights_subdir
from .test_audio_minimax_h3 import HOP_LENGTH, NUM_LATENT_FRAMES, _best

_LINE_8K = {**line_params_8k, "require_exact_physical_num_devices": True, "l1_small_size": 65536}
MESH = [
    pytest.param((4, 8), _LINE_8K, id="mesh4x8"),
    pytest.param((1, 32), _LINE_8K, id="mesh1x32"),
    # The quad (4 hosts x 32 chips) under the ring fabric the quad pipeline opens it with.
    pytest.param((4, 32), {**ring_params_8k_req_exact_devices, "l1_small_size": 65536}, id="mesh4x32"),
]
TAIL_LATENTS = 20
PSNR_BAR_DB = 40.0


def _factors_for(shape):
    """Unsharded first, then one T-shard per mesh axis longer than 1 (the factor must equal the axis length)."""
    factors = [(1, 1)]
    for axis in (0, 1):
        if shape[axis] > 1:
            factors.append((shape[axis], axis))
    return factors


@pytest.mark.timeout(5400)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH, indirect=["mesh_device", "device_params"])
def test_audio_encode_tshard_matrix(mesh_device):
    weights_dir = weights_subdir("audio_vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_MODEL_PATH")
    from safetensors.torch import load_file

    config = load_config(weights_dir)
    converted = convert_minimax_h3_audio_state_dict(
        load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors"))
    )
    encoder_state = {
        k: v for k, v in converted.items() if k.startswith(("encoder.", "pre_block.", "mean_proj.", "logs_proj."))
    }
    torch.manual_seed(2)
    waveform = torch.randn(2, 1, NUM_LATENT_FRAMES * HOP_LENGTH) * 0.1
    shape = (mesh_device.shape[0], mesh_device.shape[1])

    baseline = None
    results = []  # (factor, axis, seconds, psnr_mean, psnr_logs, psnr_tail_mean); seconds None = failed
    for factor, axis in _factors_for(shape):
        pc = None if factor <= 1 else ParallelFactor(factor=factor, mesh_axis=axis)
        ccl = None if pc is None else CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
        logger.info(f"ENCODE MATRIX mesh={shape} t_factor={factor} axis={axis}: building")
        try:
            encoder = MiniMaxH3AudioEncoder(
                encoder_dim=config["encoder_dim"],
                encoder_rates=tuple(config["encoder_rates"]),
                latent_dim=config["latent_dim"],
                latent_channels=config["latent_channels"],
                num_attention_heads=config["num_attention_heads"],
                mesh_device=mesh_device,
                parallel_config=pc,
                ccl_manager=ccl,
            )
            encoder.load_torch_state_dict(dict(encoder_state))
            mean, logs = encoder(waveform)
            mean2, _ = encoder(waveform)
            repeat_db = psnr(mean, mean2)
            logger.info(f"REPEAT t_factor={factor} axis={axis}: run1 vs run2 mean PSNR {repeat_db:.1f} dB")
            seconds = _best(lambda: encoder(waveform))
        except Exception as exc:
            logger.warning(f"ENCODE MATRIX t_factor={factor} axis={axis} FAILED: {str(exc)[:200]}")
            results.append((factor, axis, None, None, None, None, None))
            continue
        assert mean.shape == (2, config["latent_channels"], NUM_LATENT_FRAMES), f"{tuple(mean.shape)}"
        if baseline is None:
            baseline = (mean, logs)
            g_mean = g_logs = t_mean = float("inf")
        else:
            g_mean = psnr(baseline[0], mean)
            g_logs = psnr(baseline[1], logs)
            t_mean = psnr(baseline[0][..., -TAIL_LATENTS:], mean[..., -TAIL_LATENTS:])
        results.append((factor, axis, seconds, g_mean, g_logs, t_mean, repeat_db))
        logger.info(
            f"PERF encode t_factor={factor} axis={axis}: {seconds:.4f} s  PSNR mean {g_mean:.1f} dB  "
            f"logs {g_logs:.1f} dB  last {TAIL_LATENTS} latents (mean) {t_mean:.1f} dB"
        )
        del encoder

    logger.info(f"=== audio ENCODE T-shard matrix summary, mesh {shape} ===")
    for factor, axis, seconds, g_mean, g_logs, t_mean, repeat_db in results:
        if seconds is None:
            logger.info(f"  t_factor={factor:2d} axis={axis}: FAILED (see warning above)")
        else:
            logger.info(
                f"  t_factor={factor:2d} axis={axis}: {seconds:.4f} s  mean {g_mean:6.1f} dB  logs {g_logs:6.1f} dB  "
                f"tail {t_mean:6.1f} dB  repeat {repeat_db:6.1f} dB"
            )

    assert results and results[0][0] == 1 and results[0][2] is not None, "the unsharded baseline must run"
    failed = [(f, a) for f, a, s, *_ in results if s is None]
    weak = [
        (f, a, gm, tm)
        for f, a, s, gm, gl, tm, _ in results
        if s is not None and f != 1 and (gm < PSNR_BAR_DB or gl < PSNR_BAR_DB or tm < PSNR_BAR_DB)
    ]
    nondeterministic = [(f, a, r) for f, a, s, *_, r in results if s is not None and r != float("inf")]
    assert not failed, f"sharded encode failed for {failed}"
    assert not weak, f"sharded encode below {PSNR_BAR_DB} dB (factor, axis, mean, tail): {weak}"
    assert (
        not nondeterministic
    ), f"sharded encode not bit-reproducible run to run (factor, axis, dB): {nondeterministic}"
