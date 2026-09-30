# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Where the LTX-2.5 text-encode time goes on 4x8: host prep, device graph, readback, for the
eager path, the first traced call (capture) and replays; the Gemma stack vs feature extractor +
connectors; and what the 1024-token padding costs. Prints GEMMA4_TIMING lines; asserts only that
traced and eager embeddings agree.
"""

import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[6]))

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_dit.encoders.gemma4.encoder_pair import Gemma4TokenizerEncoderPair
from models.tt_dit.encoders.gemma4.model_gemma import Gemma4RotaryEmbedding
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.test import ring_params_8k
from models.tt_dit.utils.tracing import Tracer

LTX25 = os.environ.get("LTX25_ROOT", os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5"))
TEXT_ENCODER = os.environ.get(
    "GEMMA4_CHECKPOINT", f"{LTX25}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
)
TRANSFORMER = os.environ.get(
    "LTX25_TRANSFORMER", f"{LTX25}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
)

PROMPTS = (
    "A confident rapper in a black leather jacket and gold chain leans toward camera in a neon-lit studio, "
    "rapping a fast verse. Audio: punchy drums, deep bass, crisp vocals.",
    "A red paper boat drifts across a still pond at dusk, ripples spreading behind it. "
    "The camera holds a low steady shot near the waterline as the light fades. "
    "Audio: gentle water laps, distant crickets, soft evening air.",
    "An old fisherman mends a net on a wooden pier at sunrise, gulls wheeling overhead. "
    "The camera slowly dollies in on his weathered hands. Audio: waves against pilings, gull cries.",
    "A barista pours steamed milk into a latte, drawing a leaf pattern, in a busy cafe. "
    "Close-up, shallow depth of field. Audio: espresso machine hiss, low chatter, cups clinking.",
)


def _emit(name: str, seconds: float, extra: str = "") -> None:
    logger.info(f"GEMMA4_TIMING {name}={seconds * 1000:.1f}ms {extra}".rstrip())


def _prep(pair, prompt):
    input_ids, attention_mask = pair.tokenize(prompt)
    tt_ids = ttnn.from_torch(input_ids, device=pair.mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    seq = tt_ids.shape[-1]
    tt_mask = pair.gemma_encoder.build_attn_mask(attention_mask, seq)
    fe_mask = pair.feature_extractor.build_mask(attention_mask)
    src_idx, keep_mask = pair.video_connector.build_indices(attention_mask, seq)
    return (tt_ids, tt_mask, fe_mask, src_idx, keep_mask), attention_mask


def _readback(video_dev, audio_dev):
    return (
        ttnn.to_torch(ttnn.get_device_tensors(video_dev)[0]).float(),
        ttnn.to_torch(ttnn.get_device_tensors(audio_dev)[0]).float(),
    )


def _timed_encode(pair, prompt, *, traced: bool, label: str):
    md = pair.mesh_device
    ttnn.synchronize_device(md)
    t0 = time.perf_counter()
    inputs, attention_mask = _prep(pair, prompt)
    ttnn.synchronize_device(md)
    t1 = time.perf_counter()
    video_dev, audio_dev = pair._encode_device(*inputs, traced=traced)
    ttnn.synchronize_device(md)
    t2 = time.perf_counter()
    out = _readback(video_dev, audio_dev)
    t3 = time.perf_counter()
    real = int(attention_mask.sum())
    _emit(f"{label}.prep", t1 - t0, f"real_tokens={real}")
    _emit(f"{label}.device", t2 - t1)
    _emit(f"{label}.readback", t3 - t2)
    _emit(f"{label}.total", t3 - t0)
    return out


def _pcc(a, b) -> float:
    return float(comp_pcc(a, b, 0.0)[1])


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{**ring_params_8k, "trace_region_size": 200_000_000, "l1_small_size": 32768}],
    indirect=["device_params"],
)
def test_gemma4_encode_timing(*, mesh_device):
    for path in (TEXT_ENCODER, TRANSFORMER):
        if not Path(path).exists():
            pytest.skip(f"missing {path}")

    pair = Gemma4TokenizerEncoderPair(
        TEXT_ENCODER,
        mesh_device=mesh_device,
        ccl_manager=CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Linear),
        parallel_config=EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=mesh_device.shape[1], mesh_axis=1)),
        transformer_checkpoint=TRANSFORMER,
    )
    pair.ensure_loaded()
    enc = pair.gemma_encoder
    md = mesh_device

    # Compile pass, then eager encodes of prompts the device has not seen.
    _timed_encode(pair, "warmup", traced=False, label="eager_compile")
    eager = {p: _timed_encode(pair, p, traced=False, label=f"eager[{i}]") for i, p in enumerate(PROMPTS)}

    # Eager split: Gemma stack alone vs feature extractor + connectors.
    inputs, _ = _prep(pair, PROMPTS[1])
    tt_ids, tt_mask, fe_mask, src_idx, keep_mask = inputs
    for rep in range(2):
        ttnn.synchronize_device(md)
        t0 = time.perf_counter()
        hs = enc(tt_ids, tt_attn_mask=tt_mask)
        ttnn.synchronize_device(md)
        t1 = time.perf_counter()
        hs_list = list(hs[:-2]) + [hs[-1]]
        vf, af = pair.feature_extractor(hs_list, fe_mask)
        ttnn.synchronize_device(md)
        t2 = time.perf_counter()
        trans_mat = pair._prepare_trans_mat()
        pair.video_connector(vf, src_idx, keep_mask, trans_mat=trans_mat)
        pair.audio_connector(af, src_idx, keep_mask, trans_mat=trans_mat)
        ttnn.synchronize_device(md)
        t3 = time.perf_counter()
        _emit(f"eager_split[{rep}].gemma48", t1 - t0)
        _emit(f"eager_split[{rep}].feature_extractor", t2 - t1)
        _emit(f"eager_split[{rep}].connectors", t3 - t2)
        del hs, hs_list, vf, af

    # Traced: capture on the first prompt, replay the rest (and the first again).
    _timed_encode(pair, PROMPTS[0], traced=True, label="traced_capture")
    for i, p in enumerate(PROMPTS):
        got = _timed_encode(pair, p, traced=True, label=f"replay[{i}]")
        pv, pa = _pcc(eager[p][0], got[0]), _pcc(eager[p][1], got[1])
        logger.info(f"GEMMA4_PCC replay[{i}] vs eager: video={pv:.6f} audio={pa:.6f}")
        assert pv > 0.999 and pa > 0.999, f"traced embeddings drift from eager: {pv}, {pa}"

    # Gemma stack alone, traced, at 1024 and at a 256-token bucket: device time vs prompt padding.
    # The rope tables bind their first seq_len, so the 256 run gets fresh ones; the 1024 ones stay
    # referenced for the lifetime of the test so the earlier traces never read freed memory.
    gemma_only = Tracer(lambda ids, mask: enc(ids, tt_attn_mask=mask), device=md, clone_prep_inputs=False)
    gemma_only(tt_ids, tt_mask)
    for rep in range(3):
        ttnn.synchronize_device(md)
        t0 = time.perf_counter()
        gemma_only(tt_ids, tt_mask)
        ttnn.synchronize_device(md)
        _emit(f"gemma48_replay_1024[{rep}]", time.perf_counter() - t0)

    keep_rope = (enc.rotary_emb_global, enc.rotary_emb_local)
    cfg = enc.config
    enc.rotary_emb_global = Gemma4RotaryEmbedding(
        md,
        head_dim=cfg.global_head_dim,
        base=cfg.global_rope_theta,
        max_seq_len=256,
        partial_rotary_factor=cfg.partial_rotary_factor,
    )
    enc.rotary_emb_local = Gemma4RotaryEmbedding(md, head_dim=cfg.head_dim, base=cfg.rope_theta, max_seq_len=256)
    ids_1024, mask_1024 = pair.tokenize(PROMPTS[1])
    ids_256, mask_256 = ids_1024[:, -256:], mask_1024[:, -256:]
    assert int(mask_256.sum()) == int(mask_1024.sum()), "prompt longer than the 256 bucket"
    tt_ids_256 = ttnn.from_torch(ids_256, device=md, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_mask_256 = enc.build_attn_mask(mask_256, 256)
    hs_256_eager = [
        ttnn.to_torch(ttnn.get_device_tensors(h)[0]).float() for h in enc(tt_ids_256, tt_attn_mask=tt_mask_256)
    ]
    gemma_256 = Tracer(lambda ids, mask: enc(ids, tt_attn_mask=mask), device=md, clone_prep_inputs=False)
    gemma_256(tt_ids_256, tt_mask_256)
    for rep in range(3):
        ttnn.synchronize_device(md)
        t0 = time.perf_counter()
        gemma_256(tt_ids_256, tt_mask_256)
        ttnn.synchronize_device(md)
        _emit(f"gemma48_replay_256[{rep}]", time.perf_counter() - t0)

    # Real-token hidden states, 256 bucket vs the 1024 run, per aggregated layer.
    real = int(mask_256.sum())
    enc.rotary_emb_global, enc.rotary_emb_local = keep_rope
    hs_1024 = [ttnn.to_torch(ttnn.get_device_tensors(h)[0]).float() for h in enc(tt_ids, tt_attn_mask=tt_mask)]
    pccs = [_pcc(a[..., -real:, :], b[..., -real:, :]) for a, b in zip(hs_1024, hs_256_eager)]
    logger.info(
        f"GEMMA4_PCC bucket256 vs 1024 real-token hidden: min={min(pccs):.6f} "
        f"final={pccs[-1]:.6f} per-layer={[round(p, 5) for p in pccs]}"
    )
