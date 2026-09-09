# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Where the audio decoder's time goes on a 4x8 Galaxy at T-shard factor 8 (the pipeline default).

Two benchmarks, both on the shipping decoder configuration (`build_audio_decoder` defaults), 8 KB fabric packets:

  test_audio_decode_wall_matrix   wall time per decode, eager and traced, for 5 s / 15 s clips at batch 1 / 2,
                                  factor 8 vs the unsharded reference (best of 3 after a warm call).
  test_audio_decode_stage_timing  eager decode split into stages by synchronizing the mesh at stage
                                  boundaries: latent projection (its own upload + readback), vocoder upload,
                                  conv_pre, each of the 7 upsample bands (transposed conv + 3 AMP blocks),
                                  act_post + conv_post, the final T gather, readback; and, nested inside the
                                  bands, the T-pad tail maintenance and the halo/gather CCL calls.

Sync-at-boundary timing in eager mode measures dispatch + device per stage (they are serialized by the sync),
so the shares are shares of eager wall time; the per-op device durations come from the Tracy harness
(tools/tracy_audio_decode_t8_harness.py). Nothing here asserts on speed.

    TT_METAL_OPERATION_TIMEOUT_SECONDS=600 pytest models/tt_dit/tests/models/minimax_h3/test_audio_decode_benchmark.py -s
"""

import os
import time
from collections import defaultdict

import pytest
import torch
from loguru import logger

import ttnn

from ....layers import audio_ops, audio_resample
from ....models.audio_vae import vocoder_ltx
from ....models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
from ....parallel.config import ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.test import line_params_8k
from .common import build_audio_decoder, load_config, weights_subdir

MESH = [
    pytest.param(
        (4, 8),
        {
            **line_params_8k,
            "require_exact_physical_num_devices": True,
            "l1_small_size": 65536,
            "trace_region_size": 1_200_000_000,
        },
        id="mesh4x8_8k",
    )
]
HOP_LENGTH = 800
# (latent frames, batch): 207 = the 5.17 s short clip, 600 = a 15 s clip; batch 1 = one request, 2 = stereo/test.
CLIPS = [(207, 2), (207, 1), (600, 1), (600, 2)]


def _load(mesh_device, *, factor: int, axis: int):
    weights_dir = weights_subdir("audio_vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 audio_vae not found; set MINIMAX_H3_MODEL_PATH")
    from safetensors.torch import load_file

    config = load_config(weights_dir)
    pc = None if factor <= 1 else ParallelFactor(factor=factor, mesh_axis=axis)
    ccl = None if pc is None else CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    decoder = build_audio_decoder(config, mesh_device, parallel_config=pc, ccl_manager=ccl)
    decoder.load_torch_state_dict(
        convert_minimax_h3_audio_state_dict(
            load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors"))
        ),
        strict=False,
    )
    return decoder, config


def _best(fn, mesh_device, n=3):
    fn()
    ttnn.synchronize_device(mesh_device)
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(mesh_device)
        best = min(best, time.perf_counter() - t0)
    return best


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH, indirect=["mesh_device", "device_params"])
def test_audio_decode_wall_matrix(mesh_device):
    rows = []
    for factor, axis in [(8, 1), (1, 1)]:
        decoder, config = _load(mesh_device, factor=factor, axis=axis)
        for frames, batch in CLIPS:
            torch.manual_seed(2)
            latents = torch.randn(batch, config["latent_channels"], frames)
            eager = _best(lambda: decoder(latents), mesh_device)
            try:
                traced = _best(lambda: decoder(latents, traced=True), mesh_device)
            except Exception as exc:  # trace region too small, or an untraceable op: report, keep going
                logger.warning(
                    f"traced decode failed for factor {factor} frames {frames} batch {batch}: {str(exc)[:160]}"
                )
                traced = float("nan")
            finally:
                decoder.release_trace()
            rows.append((factor, frames, batch, eager, traced))
            logger.info(
                f"WALL factor={factor} frames={frames} ({frames * HOP_LENGTH / 32000:.1f} s audio) batch={batch}: "
                f"eager {eager:.4f} s  traced {traced:.4f} s"
            )
        del decoder
    logger.info("=== audio decode wall matrix (4x8, best of 3) ===")
    logger.info(f"{'factor':>6s} {'frames':>6s} {'batch':>5s} {'eager s':>9s} {'traced s':>9s} {'eager/traced':>12s}")
    for factor, frames, batch, eager, traced in rows:
        logger.info(f"{factor:6d} {frames:6d} {batch:5d} {eager:9.4f} {traced:9.4f} {eager / traced:12.2f}")


class _StageTimer:
    """Wraps module forwards and module-level helpers with mesh syncs; accumulates wall time per stage."""

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.total = defaultdict(float)
        self.calls = defaultdict(int)
        self._restore = []

    def _wrap(self, name, fn):
        def wrapped(*args, **kwargs):
            ttnn.synchronize_device(self.mesh_device)
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            ttnn.synchronize_device(self.mesh_device)
            self.total[name] += time.perf_counter() - t0
            self.calls[name] += 1
            return out

        return wrapped

    def wrap_forward(self, module, name):
        # Module.__call__ dispatches to self.forward, so an instance attribute shadows the class method.
        module.forward = self._wrap(name, module.forward)

    def wrap_attr(self, owner, attr, name):
        original = getattr(owner, attr)
        setattr(owner, attr, self._wrap(name, original))
        self._restore.append((owner, attr, original))

    def restore(self):
        for owner, attr, original in self._restore:
            setattr(owner, attr, original)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize(("frames", "batch"), [(600, 1), (207, 2)], ids=["600lat_b1", "207lat_b2"])
def test_audio_decode_stage_timing(mesh_device, frames, batch):
    decoder, config = _load(mesh_device, factor=8, axis=1)
    torch.manual_seed(2)
    latents = torch.randn(batch, config["latent_channels"], frames)
    decoder(latents)  # warm: compile everything outside the timed pass
    ttnn.synchronize_device(mesh_device)

    voc = decoder.decoder
    timer = _StageTimer(mesh_device)
    # top-level stages (disjoint, in call order)
    timer.wrap_attr(decoder, "_project_latents_device", "0 latent projection (upload+conv+readback)")
    timer.wrap_attr(voc, "_upload_BCT", "1 vocoder upload")
    timer.wrap_forward(voc.conv_pre, "2 conv_pre")
    for i, up in enumerate(voc.ups):
        timer.wrap_forward(up, f"3.{i} band {i} transposed conv")
        for k in range(voc.num_kernels):
            timer.wrap_forward(voc.resblocks[i * voc.num_kernels + k], f"3.{i} band {i} AMP blocks")
    timer.wrap_forward(voc.act_post, "4 act_post")
    timer.wrap_forward(voc.conv_post, "4 conv_post")
    timer.wrap_attr(voc, "_device_to_host", "6 readback + host crop")
    # nested helpers (counted inside the stages above; reported as "of which")
    timer.wrap_attr(vocoder_ltx, "_set_tpad_tail", "of which: T-pad tail maintenance")
    timer.wrap_attr(vocoder_ltx, "_all_gather_t", "of which: T all-gather (ups + final)")
    timer.wrap_attr(audio_ops, "_t_neighbor_pad", "of which: halo exchange (convs)")
    timer.wrap_attr(audio_resample, "_t_neighbor_pad", "of which: halo exchange (resamplers)")

    t0 = time.perf_counter()
    decoder(latents)
    ttnn.synchronize_device(mesh_device)
    wall = time.perf_counter() - t0
    timer.restore()

    top = {k: v for k, v in timer.total.items() if not k.startswith("of which")}
    nested = {k: v for k, v in timer.total.items() if k.startswith("of which")}
    accounted = sum(top.values())
    logger.info(
        f"=== eager audio decode stage timing, 4x8 factor 8, {frames} latents ({frames * HOP_LENGTH / 32000:.1f} s) "
        f"batch {batch}: wall {wall:.3f} s, stages sum {accounted:.3f} s (sync overhead + unwrapped ops = rest) ==="
    )
    for name in sorted(top):
        logger.info(f"  {name:44s} {top[name]*1e3:9.1f} ms  {top[name]/wall*100:5.1f} %  ({timer.calls[name]} calls)")
    for name in sorted(nested):
        logger.info(
            f"  {name:44s} {nested[name]*1e3:9.1f} ms  {nested[name]/wall*100:5.1f} %  ({timer.calls[name]} calls)"
        )
    assert wall > 0
