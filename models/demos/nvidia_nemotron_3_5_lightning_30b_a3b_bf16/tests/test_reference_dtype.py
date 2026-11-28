"""Unit tests for the reference-precision decision in tt/_hf_ref.py.

The full-depth fp32 reference for this 30B model needs ~217 GB and OOM-killed the optimize session
(2026-09-21). choose_reference_dtype() decides precision from the model's OWN recorded size versus
live host memory, at the single point every reference build funnels through, and is depth-aware so a
shallow gate build keeps fp32 while the full-depth build drops to bf16. No device required.
"""
import torch

from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt import _hf_ref


def _size(monkeypatch, bf16_gb, total_layers, avail_gb):
    monkeypatch.setattr(_hf_ref, "_checkpoint_bf16_bytes_and_layers", lambda: (int(bf16_gb * 1e9), total_layers))
    monkeypatch.setattr(_hf_ref, "_mem_available_bytes", lambda: int(avail_gb * 1e9))


def test_full_depth_fp32_that_does_not_fit_drops_to_bf16(monkeypatch):
    # 66 GB bf16 checkpoint => ~224 GB fp32 peak at margin 1.7; 200 GB free cannot hold it.
    _size(monkeypatch, bf16_gb=66, total_layers=52, avail_gb=200)
    dt, why = _hf_ref.choose_reference_dtype(None)
    assert dt is torch.bfloat16, why


def test_shallow_gate_build_keeps_fp32_on_the_same_box(monkeypatch):
    # THE DEPTH-AWARE POINT: same model, same 200 GB, but only 7 of 52 layers -> ~30 GB fp32, fits.
    _size(monkeypatch, bf16_gb=66, total_layers=52, avail_gb=200)
    dt, why = _hf_ref.choose_reference_dtype(7)
    assert dt is torch.float32, why


def test_full_depth_fits_on_a_big_box(monkeypatch):
    _size(monkeypatch, bf16_gb=66, total_layers=52, avail_gb=400)
    dt, why = _hf_ref.choose_reference_dtype(None)
    assert dt is torch.float32, why


def test_unsized_model_keeps_the_fp32_default(monkeypatch):
    monkeypatch.setattr(_hf_ref, "_checkpoint_bf16_bytes_and_layers", lambda: (None, None))
    monkeypatch.setattr(_hf_ref, "_mem_available_bytes", lambda: int(1e9))
    dt, why = _hf_ref.choose_reference_dtype(None)
    assert dt is torch.float32, why


def test_explicit_low_mem_signal_forces_bf16(monkeypatch):
    _size(monkeypatch, bf16_gb=66, total_layers=52, avail_gb=400)  # fp32 would fit
    monkeypatch.setenv("PERF_MCP_LOW_MEM_REFERENCE", "1")
    dt, why = _hf_ref.choose_reference_dtype(7)
    assert dt is torch.bfloat16, why


def test_explicit_force_fp32_wins_even_when_it_would_not_fit(monkeypatch):
    _size(monkeypatch, bf16_gb=66, total_layers=52, avail_gb=50)  # would not fit
    monkeypatch.setenv("PERF_MCP_FORCE_FP32_REFERENCE", "1")
    dt, why = _hf_ref.choose_reference_dtype(None)
    assert dt is torch.float32, why


def test_margin_is_tunable_not_a_fixed_gigabyte_number(monkeypatch):
    # A borderline box (200 GB free, 0.8 usable = 160 GB). Default margin 1.7 -> ~224 GB, no fit.
    _size(monkeypatch, bf16_gb=66, total_layers=52, avail_gb=200)
    assert _hf_ref.choose_reference_dtype(None)[0] is torch.bfloat16
    # Tighten the margin (a ratio, via env) and the same box now clears fp32 -- no GB constant baked in.
    monkeypatch.setenv("PERF_MCP_MEM_SAFETY_MARGIN", "1.0")
    assert _hf_ref.choose_reference_dtype(None)[0] is torch.float32
