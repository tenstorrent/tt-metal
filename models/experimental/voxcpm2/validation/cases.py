# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pinned native-CUDA oracle selections for the complete component accuracy gate.

The MLPerf locations below require maintainer provisioning; tests never download
or manufacture missing CUDA captures. Environment overrides relocate these exact
captures, rather than silently selecting a different oracle. Other captures can
be tested with the replay_component CLI.
"""

from dataclasses import dataclass
from pathlib import Path


ASSET_ROOT = Path("/mnt/MLPerf/tt_dnn-models/openbmb/VoxCPM2")
DEFAULT_CHECKPOINT = ASSET_ROOT / "checkpoint"
CAPTURES = {
    "text": (
        "VOXCPM2_CUDA_CAPTURE",
        ASSET_ROOT / "cuda-reference-torch210-20261007",
        "72a5bbbc9deb2b2f517bde44309bbdf978986d759a4625d6a29f49f98004d782",
    ),
    "voice": (
        "VOXCPM2_VOICE_CUDA_CAPTURE",
        ASSET_ROOT / "cuda-reference-with-ref-torch210-20261007",
        "971ea6c97fe5601f21442b1bb3e6a3a981b04c2c52767f0fd79b863f0fde3393",
    ),
}


@dataclass(frozen=True)
class ReplayCase:
    component: str
    event_index: int
    capture: str = "text"
    max_relative_rms: float = 0.02
    max_abs: float | None = None
    dtype: str = "bfloat16"

    @property
    def id(self):
        return f"{self.component}-{self.event_index:06d}-{self.capture}"


# The decoder's 3% RMS gate intentionally rejects the historical 7.6% result.
# PCC alone cannot detect a gain/DC error. Do not loosen this to hide a regression.
CASES = (
    ReplayCase("audio_vae.decode", 0, max_relative_rms=0.03, dtype="float32"),
    ReplayCase("audio_vae.encode", 0, capture="voice", max_relative_rms=0.05, dtype="float32"),
    ReplayCase("base_lm.forward", 0),
    ReplayCase("enc_to_lm_proj.forward", 1),
    ReplayCase("feat_decoder.estimator.forward", 0),
    ReplayCase("feat_decoder.estimator.forward", 4),
    ReplayCase("feat_decoder.estimator.forward", 8),
    ReplayCase("feat_encoder.forward", 1),
    ReplayCase("fsq_layer.forward", 1),
    ReplayCase("fusion_concat_proj.forward", 1),
    ReplayCase("lm_to_dit_proj.forward", 0),
    ReplayCase("res_to_dit_proj.forward", 0),
    ReplayCase("residual_lm.forward", 0),
    ReplayCase("stop_actn.forward", 0),
    ReplayCase("stop_head.forward", 0, max_abs=1e-3),
    ReplayCase("stop_proj.forward", 0),
)
