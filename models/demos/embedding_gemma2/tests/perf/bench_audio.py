# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Burst and sustained traced latency of the audio forward (subsampling -> 12 conformer layers ->
output_proj -> embed_audio).

Same harness, switches and definitions as `bench_vision.py` (`run_tower_bench`, read its docstring for
the option table and costs): a plain run is the headline shape (1 x 768, a 30 s clip), burst only, on the
12 x 10 grid; EMBEDDING_GEMMA2_BENCH_CONTRACT=1 adds every batch-1 contract shape (1 x 256 = 10 s,
1 x 768 = 30 s); EMBEDDING_GEMMA2_BENCH_SUSTAINED / _TELEMETRY / _CHECK / _ALL as there;
EMBEDDING_GEMMA2_GRID=13x10 for the P150 grid.

Device-only timing: inputs already resident, `execute_trace` + one synchronize per window.
Inputs are the real processor outputs of the audio test cases (`audio_common.load_case`).

Run:
    python models/demos/embedding_gemma2/tests/perf/bench_audio.py [--cases 1x256,1x768] [--json out.jsonl] [--label name]
"""

from __future__ import annotations

import torch

from models.demos.embedding_gemma2.tests.perf.bench_vision import TowerSpec, run_tower_bench

# (batch, token bucket) -> processor cases concatenated along the batch (as test_audio_perf.PERF_CASES)
CASE_INPUTS = {
    (1, 256): ("s10",),
    (1, 768): ("s30",),
    (2, 256): ("s10", "s10"),
    (4, 256): ("s10",) * 4,
    (2, 768): ("s30",) * 2,
}


def _case_inputs(snapshot, b: int, s: int):
    """(case names, (input_features, input_features_mask)) for shape b x s."""
    from models.demos.embedding_gemma2.tests.pcc.audio.audio_common import load_case

    names = CASE_INPUTS.get((b, s))
    if names is None:
        raise ValueError(f"no input case for {b}x{s}")
    feats, masks = zip(*(load_case(str(snapshot), c) for c in names))
    return names, (torch.cat(feats), torch.cat(masks))


def _inputs(snapshot, b: int, s: int):
    """prepare_inputs args for shape b x s (also used by dtype_sweep_towers.py)."""
    return _case_inputs(snapshot, b, s)[1]


def _audio_reference(snapshot, args):
    from models.demos.embedding_gemma2.reference.hf_audio_reference import load_reference_audio, run_audio_reference

    torch.set_num_threads(max(torch.get_num_threads(), 16))
    tower, embed, _ = load_reference_audio(snapshot)
    feats, mask = args
    return run_audio_reference(tower, embed, feats, mask, capture_layers=()).soft_tokens


def _audio_build(device, snapshot):
    from models.demos.embedding_gemma2.tt.audio.audio_encoder import AudioEncoder

    return AudioEncoder.from_pretrained(device, snapshot)


AUDIO = TowerSpec(
    name="audio",
    unit="clip",
    headline=("1x768",),
    contract=("1x256", "1x768"),
    burst_default=20,
    build=_audio_build,
    inputs=_case_inputs,
    counts=lambda host: host.num_tokens,
    reference=_audio_reference,
)


if __name__ == "__main__":
    run_tower_bench(AUDIO)
