# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""H3 audio trace capture hold on a 4x8 Galaxy, with the decoder built as the pipeline serves it.

A capture keeps only the DRAM it freed that nothing else uses when it ends, not every range freed during it.
Buffers made after the capture (filling most of the free DRAM) never land on the trace's intermediates:
replay still matches eager, and so does a second length captured beside the live trace. Releasing the
traces gives the held DRAM back.
"""

import pytest
import torch
from loguru import logger

import ttnn

from ....pipelines.minimax_h3.pipeline_minimax_h3 import _AUDIO_PACK_BANDS
from ....utils import tracing
from .common import psnr
from .test_audio_decode_squeeze import MESH, _load

MB = 1 << 20
REPLAY_FLOOR_DB = 60.0   # replay and eager run the same programs; anything below this is a corrupted replay
FILL_KEEP_MB = 1024      # free DRAM per bank left unfilled, for the eager ops that feed each replay


def _dram(mesh_device):
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return view.total_bytes_free_per_bank, view.total_bytes_allocated_per_bank, view.num_banks


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize(("num_latent_frames", "second_frames"), [(900, 600)], ids=["15s_then_10s"])
def test_audio_trace_capture_hold(mesh_device, num_latent_frames, second_frames, monkeypatch):
    holds = []
    record = tracing.Tracer._record

    def recording(self, **kwargs):
        record(self, **kwargs)
        if self._hold_capture_frees:
            holds.append(self.capture_hold_bytes)

    monkeypatch.setattr(tracing.Tracer, "_record", recording)
    decoder, _ = _load(
        mesh_device, split_mode="kernel", pack_bands=_AUDIO_PACK_BANDS, act_mode="fused", batch_shard_axis=0
    )
    torch.manual_seed(0)
    latents = torch.randn(1, decoder.latent_channels, num_latent_frames)
    second = torch.randn(1, decoder.latent_channels, second_frames)

    eager = decoder(latents)
    eager_second = decoder(second)
    ttnn.synchronize_device(mesh_device)
    free0, alloc0, banks = _dram(mesh_device)
    traced = decoder(latents, traced=True)
    ttnn.synchronize_device(mesh_device)
    free1, alloc1, _ = _dram(mesh_device)
    assert len(holds) == 1, f"one held capture expected, saw {len(holds)}"
    hold = holds[0]
    logger.info(
        f"HOLD capture {num_latent_frames} latents: DRAM per bank free {free0 // MB} -> {free1 // MB} MB, "
        f"allocated {alloc0 // MB} -> {alloc1 // MB} MB, trace holds {hold // MB} MB per bank "
        f"({hold * banks / 2**30:.2f} GiB per chip, {banks} banks)"
    )
    assert hold > 0, "the capture held nothing: its intermediates are not protected from later buffers"
    assert hold <= free1, f"the capture holds {hold // MB} MB per bank, more than the {free1 // MB} MB it leaves free"

    fill = []
    shape = [1, 1, 8192, 4096]   # 64 MiB bf16 per device, interleaved over its banks
    per_bank = 64 * MB // banks
    while _dram(mesh_device)[0] - per_bank > FILL_KEEP_MB * MB:
        fill.append(ttnn.full(shape, 7.0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG))
    ttnn.synchronize_device(mesh_device)
    logger.info(f"HOLD fill: {len(fill)} buffers, DRAM per bank free {_dram(mesh_device)[0] // MB} MB")

    replay = decoder(latents, traced=True)
    traced_second = decoder(second, traced=True)
    replay_after_second = decoder(latents, traced=True)
    eager_after = decoder(latents)
    ttnn.synchronize_device(mesh_device)
    assert len(holds) == 2, f"the second length captures once, saw {len(holds) - 1}"
    scores = {
        "capture": psnr(eager, traced),
        "replay_after_fill": psnr(eager, replay),
        "second_length_capture": psnr(eager_second, traced_second),
        "replay_after_second_capture": psnr(eager, replay_after_second),
        "eager_after_fill": psnr(eager, eager_after),
    }
    logger.info(
        "HOLD replay vs eager PSNR dB: " + ", ".join(f"{k} {v:.2f}" for k, v in scores.items())
        + f"; max abs diff replay {(eager.float() - replay.float()).abs().max().item():.3e}"
        + f"; second capture holds {holds[1] // MB} MB per bank"
    )
    for name, db in scores.items():
        assert db >= REPLAY_FLOOR_DB, f"{name}: {db:.2f} dB against eager, floor {REPLAY_FLOOR_DB} dB"

    for t in fill:
        ttnn.deallocate(t)
    decoder.release_trace()
    ttnn.synchronize_device(mesh_device)
    free2, alloc2, _ = _dram(mesh_device)
    logger.info(f"HOLD released: DRAM per bank free {free2 // MB} MB, allocated {alloc2 // MB} MB")
    assert free2 >= free0, f"release left {(free0 - free2) // MB} MB per bank held"
