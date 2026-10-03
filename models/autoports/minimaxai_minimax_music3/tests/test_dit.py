"""Stage 05: the TTNN DiT vs the torch reference: single forwards (unit goldens) and a full 30-step window denoise
with the vocoder, against artifacts/golden/readme/tf/dit_reference.pt."""
import json
import os
import time

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from models.autoports.minimaxai_minimax_music3.config import Music3Config  # noqa: E402
from models.autoports.minimaxai_minimax_music3.reference.pipeline import Music3Reference  # noqa: E402
from models.autoports.minimaxai_minimax_music3.tests.tt_common import Report, pcc  # noqa: E402
from models.autoports.minimaxai_minimax_music3.tt import weights as W  # noqa: E402
from models.autoports.minimaxai_minimax_music3.tt.device import open_mesh, parse_shape  # noqa: E402
from models.autoports.minimaxai_minimax_music3.tt.dit import TTDiT  # noqa: E402

REPORT = Report()
DT = {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[os.environ.get("MUSIC3_DIT_DTYPE", "bf16")]


@pytest.fixture(scope="module")
def handle():
    h = open_mesh(parse_shape(os.environ.get("MUSIC3_MESH_SHAPE", "1x1")))
    yield h
    h.close()


@pytest.fixture(scope="module", autouse=True)
def _write_report():
    yield
    REPORT.write()


def test_two_layer_block_stack(handle, snapshot, golden_root):
    """Fast first check: 2 layers, real weights, production window length."""
    u = torch.load(golden_root / "unit" / "unit_goldens.pt")["dit2_L689_t0.5"]
    cfg = Music3Config.from_snapshot(snapshot)
    dit = TTDiT(handle.mesh, W.load_dit_state(snapshot, num_layers=2), cfg.dit, weights_dtype=DT, num_layers=2)
    y = dit.forward(u["x"], torch.full((2,), u["t"]), u["cond"])
    r = pcc(y, u["y"])
    REPORT["dit2_L689"] = {"pcc": r}
    print("2-layer PCC", r)
    dit.release()
    assert r >= 0.99, r


@pytest.fixture(scope="module")
def dit(handle, snapshot):
    cfg = Music3Config.from_snapshot(snapshot)
    t0 = time.time()
    d = TTDiT(
        handle.mesh,
        W.load_dit_state(snapshot),
        cfg.dit,
        weights_dtype=DT,
        use_trace=os.environ.get("MUSIC3_TRACE", "1") == "1",
    )
    REPORT.update(load_s=time.time() - t0, dit_dtype=os.environ.get("MUSIC3_DIT_DTYPE", "bf16"))
    yield d
    d.release()


def test_unit_goldens(dit, golden_root):
    u = torch.load(golden_root / "unit" / "unit_goldens.pt")
    rows = {}
    for k, g in u.items():
        if not k.startswith("dit_L"):
            continue
        t0 = time.time()
        y = dit.forward(g["x"], torch.full((2,), g["t"]), g["cond"])
        rows[k] = {
            "pcc": pcc(y, g["y"]),
            "pcc_cond_row": pcc(y[0], g["y"][0]),
            "pcc_uncond_row": pcc(y[1], g["y"][1]),
            "seconds": time.time() - t0,
        }
    # warmed timing of the production shape
    g = u["dit_L689_t0.5000"]
    t0 = time.time()
    for _ in range(5):
        dit.forward(g["x"], torch.full((2,), g["t"]), g["cond"])
    rows["warm_ms_per_forward_L689"] = 1000 * (time.time() - t0) / 5
    REPORT["unit"] = rows
    print(json.dumps(rows, indent=1))
    assert min(r["pcc"] for k, r in rows.items() if isinstance(r, dict)) >= 0.99


def test_window_denoise_and_vocoder(dit, snapshot, golden_root):
    g = torch.load(golden_root / "readme" / "golden.pt")
    ref_out = torch.load(golden_root / "readme" / "tf" / "dit_reference.pt")
    ref = Music3Reference(snapshot, load_llm=False, load_dit=False, load_vocoder=True)
    gen = torch.Generator("cpu").manual_seed(int(ref_out["seed"]))
    t0 = time.time()
    den = ref.denoise(g["frame_hiddens"], gen, record_steps=(0, 15, 29), dit_forward=dit.forward)
    dt = time.time() - t0
    rows = {"chunks": len(den["latent_chunks"]), "denoise_seconds": dt}
    for k, (a, b) in enumerate(zip(den["chunk_records"], ref_out["chunk_records"])):
        assert torch.equal(a["noise"], b["noise"]) and torch.allclose(a["condition"], b["condition"], atol=1e-5)
        s0a, s0b = a["steps"][0], b["steps"][0]
        rows[f"chunk{k}_step0_pred_cond_pcc"] = pcc(s0a["pred_cond"], s0b["pred_cond"])
        rows[f"chunk{k}_step0_pred_uncond_pcc"] = pcc(s0a["pred_uncond"], s0b["pred_uncond"])
        for sa, sb in zip(a["steps"][1:], b["steps"][1:]):
            rows[f"chunk{k}_step{sa['i']}_pred_cond_pcc_trajectory"] = pcc(sa["pred_cond"], sb["pred_cond"])
        rows[f"chunk{k}_latents_pcc"] = pcc(a["latents_out"], b["latents_out"])
    wav = ref.decode(den["latent_chunks"])
    rows["audio_pcc"] = pcc(wav, ref_out["audio"])
    x, y = wav.mean(1).squeeze(0), ref_out["audio"].mean(1).squeeze(0)
    sx = torch.stft(x, 2048, 512, window=torch.hann_window(2048), return_complex=True).abs()
    sy = torch.stft(y, 2048, 512, window=torch.hann_window(2048), return_complex=True).abs()
    rows["log_spectral_distance_db"] = float((20 * torch.log10((sx + 1e-5) / (sy + 1e-5))).pow(2).mean().sqrt())
    rows["spectral_convergence"] = float((sx - sy).norm() / sy.norm())
    REPORT["window"] = rows
    print(json.dumps(rows, indent=1))
    import soundfile as sf

    out = os.environ.get("MUSIC3_E2E_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        sf.write(os.path.join(out, "dit_tt_readme.wav"), wav.squeeze(0).T.numpy(), 44100, subtype="PCM_16")
    assert rows["chunk0_step0_pred_cond_pcc"] >= 0.99, rows["chunk0_step0_pred_cond_pcc"]
    assert rows["chunk0_latents_pcc"] >= 0.95, rows["chunk0_latents_pcc"]
    assert rows["spectral_convergence"] < 0.5, rows["spectral_convergence"]


def test_two_window_denoise_blues(dit, snapshot, golden_root):
    """Two windows (689 + 345 latents) with the overlap carry, against the CPU reference on the blues golden."""
    ref_p = golden_root / "blues" / "tf" / "dit_reference.pt"
    if not ref_p.exists():
        pytest.skip("blues DiT reference not computed")
    g = torch.load(golden_root / "blues" / "golden.pt")
    ref_out = torch.load(ref_p)
    ref = Music3Reference(snapshot, load_llm=False, load_dit=False, load_vocoder=True)
    gen = torch.Generator("cpu").manual_seed(int(ref_out["seed"]))
    den = ref.denoise(g["frame_hiddens"], gen, record_steps=(0,), dit_forward=dit.forward)
    rows = {"chunks": len(den["latent_chunks"])}
    for k, (a, b) in enumerate(zip(den["chunk_records"], ref_out["chunk_records"])):
        rows[f"chunk{k}_step0_pred_cond_pcc"] = pcc(a["steps"][0]["pred_cond"], b["steps"][0]["pred_cond"])
        rows[f"chunk{k}_latents_pcc"] = pcc(a["latents_out"], b["latents_out"])
    wav = ref.decode(den["latent_chunks"])
    rows["audio_pcc"] = pcc(wav, ref_out["audio"])
    rows["audio_rms"] = float(wav.pow(2).mean().sqrt())
    rows["ref_audio_rms"] = float(ref_out["audio"].pow(2).mean().sqrt())
    REPORT["window_blues"] = rows
    print(json.dumps(rows, indent=1))
    assert rows["chunks"] == 2 and rows["chunk1_latents_pcc"] >= 0.95, rows
