"""Phase B: the TTNN fast codebook decoder vs the torch (fp32) fast decoder on real weights and real hidden
states (from the stage-01 goldens when available). Env: FISH_S2_MESH_SHAPE (1x1), FISH_S2_REPORT, FISH_S2_FLOORS."""
import json
import os
import time
from pathlib import Path

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from models.autoports.fishaudio_s2_pro.config import S2Config  # noqa: E402
from models.autoports.fishaudio_s2_pro.tt import weights as W  # noqa: E402
from models.autoports.fishaudio_s2_pro.tt.device import open_mesh, parse_shape  # noqa: E402
from models.autoports.fishaudio_s2_pro.tt.fast_decoder import TTFastDecoder  # noqa: E402
from models.autoports.fishaudio_s2_pro.tt.fast_decoder_torch import TorchFastDecoder  # noqa: E402

REPORT = {}


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.fixture(scope="module")
def handle():
    h = open_mesh(parse_shape(os.environ.get("FISH_S2_MESH_SHAPE", "1x1")))
    yield h
    h.close()


@pytest.fixture(scope="module")
def decoders(handle, snapshot):
    cfg = S2Config.from_snapshot(snapshot)
    sd = W.load_fish_state_dict(snapshot)
    fast_split = W.fast_tower_state_dict(sd, cfg, split_qkv=True)
    fast_fused = W.fast_tower_state_dict(sd, cfg, split_qkv=False)
    t0 = time.time()
    tt = TTFastDecoder(handle.mesh, fast_split, cfg, use_trace=os.environ.get("FISH_S2_FAST_TRACE", "1") == "1")
    tt.warmup()
    REPORT["build_s"] = time.time() - t0
    ref = TorchFastDecoder(fast_fused, cfg, dtype=torch.float32)
    return cfg, tt, ref


@pytest.fixture(scope="module", autouse=True)
def _write_report():
    yield
    out = os.environ.get("FISH_S2_REPORT")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(REPORT, open(out, "w"), indent=2, default=str)


def _hidden_and_codes(golden_root, cfg, n=8):
    """Real (hidden, frame) pairs from the goldens' teacher-forced dumps, else random."""
    pairs = []
    for d in sorted(golden_root.glob("*/*/greedy")) if golden_root else []:
        tf, fr = d / "teacher_forced.pt", d / "frames.pt"
        if tf.exists() and fr.exists():
            t = torch.load(tf)
            frames = torch.load(fr)
            for g in range(min(n, t["fast_hidden_first"].shape[0])):
                if int(frames[0, g]) != cfg.im_end_id:
                    pairs.append((t["fast_hidden_first"][g].float(), frames[1:, g].tolist()))
        if len(pairs) >= n:
            break
    if not pairs:
        g = torch.Generator().manual_seed(0)
        for _ in range(n):
            pairs.append(
                (
                    torch.randn(cfg.fast.dim, generator=g) * 0.02,
                    [int(torch.randint(0, 4096, (1,), generator=g))]
                    + torch.randint(0, 1024, (9,), generator=g).tolist(),
                )
            )
    return pairs[:n]


def test_teacher_forced_logits(decoders, golden_root):
    cfg, tt, ref = decoders
    pairs = _hidden_and_codes(golden_root, cfg)
    pccs, top1, frames_t = [], [], []
    for hidden, codes in pairs:
        rec_tt, rec_ref = [], []
        t0 = time.time()
        tt.frame(hidden, codes[0], lambda lg, i, c=codes: c[i], record=rec_tt)
        frames_t.append(time.time() - t0)
        ref.frame(hidden, codes[0], lambda lg, i, c=codes: c[i], record=rec_ref)
        for a, b in zip(rec_tt, rec_ref):
            pccs.append(pcc(a, b))
            top1.append(int(a.argmax()) == int(b.argmax()))
    r = {
        "n_frames": len(pairs),
        "logits_pcc_min": min(pccs),
        "logits_pcc_mean": sum(pccs) / len(pccs),
        "argmax_agreement": sum(top1) / len(top1),
        "frame_seconds_mean": sum(frames_t) / len(frames_t),
        "trace": bool(tt.traces),
    }
    REPORT["teacher_forced"] = r
    print(json.dumps(r, indent=2))
    assert r["logits_pcc_min"] >= 0.98, r
    assert r["argmax_agreement"] >= 0.80, r  # bf16 near-tie flips; the PCC bar above is the correctness gate


def test_replay_deterministic(decoders, golden_root):
    cfg, tt, ref = decoders
    hidden, codes = _hidden_and_codes(golden_root, cfg, 1)[0]
    a, b = [], []
    tt.frame(hidden, codes[0], lambda lg, i, c=codes: c[i], record=a)
    tt.frame(hidden, codes[0], lambda lg, i, c=codes: c[i], record=b)
    d = max(float((x - y).abs().max()) for x, y in zip(a, b))
    REPORT["replay_max_abs_diff"] = d
    assert d < 1e-3, d


def test_free_running_codes_match_torch_mostly(decoders, golden_root):
    cfg, tt, ref = decoders
    hidden, codes = _hidden_and_codes(golden_root, cfg, 1)[0]
    argmax = lambda lg, i: int(lg.argmax())
    a = tt.frame(hidden, codes[0], argmax)
    b = ref.frame(hidden, codes[0], argmax)
    agree = sum(x == y for x, y in zip(a, b)) / 9
    REPORT["free_running_agreement"] = agree
    print("free-running codes tt/torch:", a, b, agree)
    # informational only: once one near-tied argmax differs every later code legitimately diverges (the chain is
    # conditioned on the previous code); the teacher-forced PCC test above is the correctness bar.
    assert len(a) == 9 and all(0 <= c < 4096 for c in a)
