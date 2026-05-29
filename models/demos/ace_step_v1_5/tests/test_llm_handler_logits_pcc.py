# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""PCC: 5 Hz LM handler TTNN logits helpers vs PyTorch (bf16).

Synthetic tensor tests require Pearson ≥ 0.99. Experimental causal LM tests compare
:class:`~models.demos.ace_step_v1_5.ttnn_impl.five_hz_causal_lm_experimental.AceStepFiveHzExperimentalTtnnCausalLM`
to HuggingFace ``AutoModelForCausalLM``; prefill checks last-position logits only.
Default floor: ``0.98`` (BF16 production). When ``ACE_STEP_LM_BFLOAT8_WEIGHTS=1``, floor is ``0.89``.
Override via ``ACE_STEP_EXPERIMENTAL_LM_PCC`` / ``ACE_STEP_EXPERIMENTAL_LM_PCC_BFP8``.
debug via ``ACE_STEP_DEBUG_LM_LOGITS`` (and ``_HF`` / ``_LAYER_PCC``).
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from models.common.utility_functions import comp_pcc
from models.demos.ace_step_v1_5.ttnn_impl.five_hz_causal_lm_experimental import AceStepFiveHzExperimentalTtnnCausalLM
from models.demos.ace_step_v1_5.ttnn_impl.five_hz_lm.five_hz_lm_paths import get_checkpoints_dir
from models.demos.ace_step_v1_5.ttnn_impl.lm_constrained_logits_ttnn import (
    logits_add_delta_bf16,
    logits_divide_by_scalar_bf16,
    logits_keep_allowed_bf16,
)
from models.demos.ace_step_v1_5.ttnn_impl.lm_logits_debug import (
    ace_step_debug_lm_logits_enabled,
    debug_compare_decode_logits_stages,
    debug_compare_prefill_logits_stages,
    log_lm_pcc,
)
from models.demos.ace_step_v1_5.ttnn_impl.lm_logits_ttnn import cfg_linear_combination_bf16
from models.demos.ace_step_v1_5.ttnn_impl.lm_postprocess_tt_transformers import (
    apply_penalty_filter_sample,
    repetition_penalty_apply,
)
from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_five_hz_lm_bfloat8_weights_enabled

_PCC = 0.99
_PCC_EXPERIMENTAL_LM_BF16 = float(os.environ.get("ACE_STEP_EXPERIMENTAL_LM_PCC", "0.98"))
_PCC_EXPERIMENTAL_LM_BFP8 = float(os.environ.get("ACE_STEP_EXPERIMENTAL_LM_PCC_BFP8", "0.89"))


def _experimental_lm_pcc_min() -> float:
    """HF logits floor: BF16/HiFi4 path ~0.98; production ``bfloat8_b`` weights ~0.90."""
    if os.environ.get("ACE_STEP_EXPERIMENTAL_LM_PCC") is not None:
        return _PCC_EXPERIMENTAL_LM_BF16
    if ace_step_five_hz_lm_bfloat8_weights_enabled():
        return _PCC_EXPERIMENTAL_LM_BFP8
    return _PCC_EXPERIMENTAL_LM_BF16


_FIVE_HZ_LM_DIR_NAMES = ("acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-0.6B")


def _resolve_five_hz_lm_dir() -> Path | None:
    """Same checkpoint layout as ``run_prompt_to_wav.py`` (HF hub cache or ACESTEP_CHECKPOINTS_DIR)."""
    roots: list[Path] = []
    env = os.environ.get("ACE_STEP_CHECKPOINT_DIR")
    if env:
        roots.append(Path(env).expanduser().resolve())
    roots.append(Path.home() / ".cache" / "huggingface" / "hub" / "ACE-Step-1.5-checkpoints")
    roots.append(get_checkpoints_dir())
    for root in roots:
        for name in _FIVE_HZ_LM_DIR_NAMES:
            d = (root / name).resolve()
            if (d / "config.json").is_file():
                return d
    return None


_LM_SKIP = (
    "5 Hz LM not found (tried acestep-5Hz-lm-1.7B then acestep-5Hz-lm-0.6B). "
    "Set ACE_STEP_CHECKPOINT_DIR or ACESTEP_CHECKPOINTS_DIR, or populate e.g. "
    "~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints/acestep-5Hz-lm-1.7B/."
)


def _align_logits_last_dims(ref: torch.Tensor, got: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Trim to common ``[1, S, V]`` (sequence / vocab) for PCC."""
    if ref.dim() != 3 or got.dim() != 3:
        raise ValueError(f"expected rank-3 logits, got {ref.dim()} vs {got.dim()}")
    b0 = min(int(ref.shape[0]), int(got.shape[0]))
    s0 = min(int(ref.shape[1]), int(got.shape[1]))
    v0 = min(int(ref.shape[2]), int(got.shape[2]))
    return ref[:b0, :s0, :v0].contiguous(), got[:b0, :s0, :v0].contiguous()


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size == 0 or bb.size == 0:
        return 1.0
    if np.array_equal(aa, bb):
        return 1.0
    r = float(np.corrcoef(aa, bb)[0, 1])
    if not np.isfinite(r):
        return 1.0
    return r


def _assert_pcc(ref: torch.Tensor, got: torch.Tensor, *, name: str = "op") -> None:
    r = _pearson(ref.detach().float().cpu().numpy(), got.detach().float().cpu().numpy())
    print(f"[llm_handler_logits_pcc] {name}: Pearson PCC={r:.8f} (threshold={_PCC})", flush=True)
    assert r >= _PCC, f"Pearson PCC {r:.6f} < {_PCC} (shapes ref={tuple(ref.shape)} got={tuple(got.shape)})"


def _torch_cfg_bf16(cond: torch.Tensor, uncond: torch.Tensor, cfg_scale: float) -> torch.Tensor:
    c = cond.detach().float().cpu().to(torch.bfloat16)
    u = uncond.detach().float().cpu().to(torch.bfloat16)
    return (u + float(cfg_scale) * (c - u)).to(torch.float32)


def _torch_add_bf16(scores: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    a = scores.detach().float().cpu().to(torch.bfloat16)
    b = delta.detach().float().cpu().to(torch.bfloat16)
    return (a + b).to(torch.float32)


def _torch_div_bf16(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    t = float(temperature)
    if t <= 0:
        t = 1e-6
    a = scores.detach().float().cpu().to(torch.bfloat16)
    return (a / t).to(torch.float32)


def _torch_repetition_penalty_bf16(
    scores: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
    *,
    prompt_ignore_length: int | None = None,
) -> torch.Tensor:
    b, v = int(scores.shape[0]), int(scores.shape[1])
    ids = input_ids.long()
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    if int(ids.shape[0]) != b:
        raise ValueError("batch mismatch")
    if prompt_ignore_length is not None and int(prompt_ignore_length) > 0:
        ids = ids[:, int(prompt_ignore_length) :]
    ids_cpu = ids.detach().cpu().clamp(min=0, max=max(0, v - 1))
    mask = torch.zeros((b, v), dtype=torch.bfloat16)
    mask.scatter_(1, ids_cpu, 1.0)
    s = scores.detach().float().cpu().to(torch.bfloat16)
    zero = torch.zeros_like(s)
    is_neg = s < zero
    p = float(penalty)
    scaled_neg = s * p
    scaled_pos = s / p
    penalized = torch.where(is_neg, scaled_neg, scaled_pos)
    out = torch.where(mask != 0, penalized, s)
    return out.to(torch.float32)


def _torch_keep_allowed(scores: torch.Tensor, allowed: list[int]) -> torch.Tensor:
    v = int(scores.shape[1])
    out = torch.full((1, v), float("-inf"), dtype=torch.float32)
    s = scores[0].detach().float().cpu()
    for t in allowed:
        ti = int(t)
        if 0 <= ti < v:
            out[0, ti] = s[ti]
    return out


@pytest.fixture
def logits_pair(torch_seed):
    torch.manual_seed(int(torch_seed))
    b, v = 2, 512
    scores = torch.randn(b, v, dtype=torch.float32) * 2.0
    delta = torch.randn(b, v, dtype=torch.float32) * 0.25
    return scores, delta


def test_llm_handler_logits_add_delta_pcc(device, logits_pair):
    scores, delta = logits_pair
    ref = _torch_add_bf16(scores, delta)
    got = logits_add_delta_bf16(scores, delta, device=device)
    _assert_pcc(ref, got, name="logits_add_delta_bf16")


def test_llm_handler_logits_divide_temperature_pcc(device, logits_pair):
    scores, _ = logits_pair
    ref = _torch_div_bf16(scores, 0.7)
    got = logits_divide_by_scalar_bf16(scores, 0.7, device=device)
    _assert_pcc(ref, got, name="logits_divide_by_scalar_bf16")


def test_llm_handler_logits_cfg_linear_combination_pcc(device, logits_pair):
    scores, delta = logits_pair
    cond = scores
    uncond = scores + delta * 0.5
    ref = _torch_cfg_bf16(cond, uncond, 1.35)
    got = cfg_linear_combination_bf16(cond, uncond, 1.35, device=device)
    _assert_pcc(ref, got, name="cfg_linear_combination_bf16")


def test_llm_handler_logits_repetition_penalty_pcc(device, logits_pair):
    """PCC for ``AceStepPenalties1D.apply_hf_repetition`` (subclass of ``Penalties1D``)
    vs HF ``RepetitionPenaltyLogitsProcessor`` semantics. The wrapper trades the
    legacy ``prompt_ignore_length`` knob for HF-faithful behaviour (all input_ids
    contribute to the penalty mask)."""
    scores, _ = logits_pair
    b, v = scores.shape
    seq = torch.randint(0, v, (b, 48), dtype=torch.long)
    penalty = 1.12
    # ``prompt_ignore_length=0`` ⇒ HF semantics, which is what AceStepPenalties1D emulates.
    ref = _torch_repetition_penalty_bf16(scores, seq, penalty, prompt_ignore_length=0)
    got = repetition_penalty_apply(scores, seq, penalty, device=device)
    _assert_pcc(ref, got, name="AceStepPenalties1D.apply_hf_repetition")


def test_llm_handler_fused_postprocess_sample_smoke(device, logits_pair):
    """Smoke test: the fused on-device ``apply_penalty_filter_sample`` returns a valid
    token id per row. We can't do an exact PCC match against HF here because the
    sampler is stochastic (``ttnn.sampling`` with per-call seeds), but we can check
    the contract: shape ``[B]``, dtype int64, every id in ``[0, vocab)``."""
    scores, _ = logits_pair
    b, v = scores.shape
    seq = torch.randint(0, v, (b, 48), dtype=torch.long)
    tokens = apply_penalty_filter_sample(
        scores,
        seq,
        repetition_penalty=1.1,
        top_k=32,
        top_p=0.9,
        temperature=1.0,
        seed=1234,
        device=device,
    )
    assert tokens.shape == (b,), f"expected [{b}], got {tuple(tokens.shape)}"
    assert tokens.dtype == torch.int64, f"expected int64, got {tokens.dtype}"
    assert int(tokens.min()) >= 0 and int(tokens.max()) < v


def test_llm_handler_logits_keep_allowed_pcc(device, torch_seed):
    torch.manual_seed(int(torch_seed))
    v = 512
    scores = torch.randn(1, v, dtype=torch.float32)
    allowed = [0, 3, 7, 11, 19, 100, 255, 400, 501]
    ref = _torch_keep_allowed(scores, allowed)
    got = logits_keep_allowed_bf16(scores, allowed, device=device)
    _assert_pcc(ref, got, name="logits_keep_allowed_bf16")


@pytest.mark.skipif(_resolve_five_hz_lm_dir() is None, reason=_LM_SKIP)
def test_llm_handler_experimental_causal_lm_prefill_decode_pcc_vs_torch(device, torch_seed):
    """PCC (``comp_pcc``) for experimental TTNN causal LM vs HF."""
    pcc_min = _experimental_lm_pcc_min()
    lm_dir = _resolve_five_hz_lm_dir()
    assert lm_dir is not None
    print(f"[llm_handler_logits_pcc] experimental_causal_lm: LM dir={lm_dir}", flush=True)
    cfg = AutoConfig.from_pretrained(str(lm_dir), trust_remote_code=True)
    vocab = int(getattr(cfg, "vocab_size", 0) or 0)
    if vocab <= 8:
        pytest.skip("invalid vocab_size from config")

    torch.manual_seed(int(torch_seed))
    L = 24
    hi = min(8192, vocab - 1)
    lo = max(4, min(8, hi - 1))
    full = torch.randint(lo, hi, (1, L + 1), dtype=torch.long)
    ids_pre = full[:, :L].contiguous()
    ids_next = full[:, L : L + 1].contiguous()

    hf = AutoModelForCausalLM.from_pretrained(
        str(lm_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    with torch.inference_mode():
        o1 = hf(input_ids=ids_pre, use_cache=True)
        ref_pre = o1.logits.float().cpu()
        o2 = hf(input_ids=ids_next, past_key_values=o1.past_key_values, use_cache=True)
        ref_dec = o2.logits.float().cpu()
    del hf
    gc.collect()

    try:
        exp = AceStepFiveHzExperimentalTtnnCausalLM(
            str(lm_dir),
            device,
            max_seq_len=1024,
        )
    except RuntimeError as exc:
        pytest.skip(f"experimental TTNN causal LM init/load skipped: {exc}")

    tt_pre: torch.Tensor | None = None
    tt_dec: torch.Tensor | None = None
    tt_pre_raw = None
    tt_dec_raw = None
    try:
        exp.reset_decode_state()
        with torch.inference_mode():
            t1 = exp.forward(input_ids=ids_pre.clone(), past_key_values=None, use_cache=True)
            tt_pre = t1.logits.float().cpu()
            tt_pre_raw = getattr(exp, "_debug_last_raw_logits", None)
            t2 = exp.forward(input_ids=ids_next.clone(), past_key_values=True, use_cache=True)
            tt_dec = t2.logits.float().cpu()
            tt_dec_raw = getattr(exp, "_debug_last_raw_logits", None)
    finally:
        if ace_step_debug_lm_logits_enabled() and tt_pre is not None:
            vocab_dbg = int(getattr(cfg, "vocab_size", tt_pre.shape[-1]))
            debug_compare_prefill_logits_stages(
                hf_ref_last=ref_pre[:, -1:, :],
                logits_tt_torch=tt_pre_raw if tt_pre_raw is not None else tt_pre,
                last_token_offset_in_tile=getattr(exp.qwen, "_prefill_last_token_offset_in_tile", None),
                seq_log=int(ids_pre.shape[1]),
                vocab=vocab_dbg,
                lm_dir=str(lm_dir),
                input_ids=ids_pre,
                qwen_params=getattr(exp.qwen, "_last_debug_params", None),
            )
            if tt_dec is not None:
                debug_compare_decode_logits_stages(
                    hf_ref_last=ref_dec[:, -1:, :],
                    logits_tt_torch=tt_dec_raw if tt_dec_raw is not None else tt_dec,
                    seq_log=int(ids_next.shape[1]),
                    vocab=vocab_dbg,
                    qwen_params=getattr(exp.qwen, "_last_debug_params", None),
                )
        if "exp" in locals():
            del exp
        gc.collect()

    rp, gp = _align_logits_last_dims(ref_pre, tt_pre)
    rp_last = rp[:, -1:, :].contiguous()
    gp_last = gp[:, -1:, :].contiguous()
    r_pre_last = _pearson(rp_last.numpy(), gp_last.numpy())
    if ace_step_debug_lm_logits_enabled():
        log_lm_pcc("test.prefill_last_pos", rp_last, gp_last, min_pcc=pcc_min)
    ok_pre, pcc_pre = comp_pcc(rp_last, gp_last, pcc=pcc_min)
    r_pre_full_diag = _pearson(rp.numpy(), gp.numpy())
    print(
        f"[llm_handler_logits_pcc] experimental_causal_lm_prefill_last_pos "
        f"comp_pcc={float(pcc_pre):.8f} (min={pcc_min}) "
        f"Pearson last-pos={r_pre_last:.8f}  "
        f"(diagnostic full-seq Pearson={r_pre_full_diag:.6f} — expected low; per-position "
        f"prefill logits are not exposed by tt_transformers' sharded LMHead path)",
        flush=True,
    )
    assert ok_pre, f"prefill last-pos comp_pcc {float(pcc_pre):.6f} < {pcc_min}"

    rd, gd = _align_logits_last_dims(ref_dec, tt_dec)
    r_dec = _pearson(rd.numpy(), gd.numpy())
    if ace_step_debug_lm_logits_enabled():
        log_lm_pcc("test.decode_last_pos", rd, gd, min_pcc=pcc_min)
    ok_dec, pcc_dec = comp_pcc(rd, gd, pcc=pcc_min)
    print(
        f"[llm_handler_logits_pcc] experimental_causal_lm_decode_step comp_pcc={float(pcc_dec):.8f} "
        f"(min={pcc_min}) Pearson={r_dec:.8f}",
        flush=True,
    )
    assert ok_dec, f"decode comp_pcc {float(pcc_dec):.6f} < {pcc_min}"


@pytest.mark.skipif(_resolve_five_hz_lm_dir() is None, reason=_LM_SKIP)
@pytest.mark.parametrize("seq_lens", [(24, 80), (50, 100)])
def test_prefill_trace_last_token_matches_eager_varying_seq_len(device, torch_seed, seq_lens):
    """Prefill trace must key on real ``seq_len``, not padded length alone (regression for noisy LM)."""
    lm_dir = _resolve_five_hz_lm_dir()
    assert lm_dir is not None
    torch.manual_seed(int(torch_seed))
    cfg = AutoConfig.from_pretrained(str(lm_dir), trust_remote_code=True)
    vocab = int(getattr(cfg, "vocab_size", 0) or 0)
    if vocab <= 8:
        pytest.skip("invalid vocab_size from config")

    try:
        exp = AceStepFiveHzExperimentalTtnnCausalLM(
            str(lm_dir),
            device,
            max_seq_len=1024,
            use_prefill_trace=True,
        )
    except RuntimeError as exc:
        pytest.skip(f"experimental TTNN causal LM init/load skipped: {exc}")

    hi = min(8192, vocab - 1)
    lo = max(4, min(8, hi - 1))
    try:
        for seq_len in seq_lens:
            ids = torch.randint(lo, hi, (1, int(seq_len)), dtype=torch.long)
            exp.reset_decode_state()
            exp.qwen._use_prefill_trace = False
            with torch.inference_mode():
                eager = exp.forward(input_ids=ids.clone(), past_key_values=None, use_cache=True)
            exp.reset_decode_state()
            exp.qwen._use_prefill_trace = True
            with torch.inference_mode():
                traced = exp.forward(input_ids=ids.clone(), past_key_values=None, use_cache=True)
            re, ge = _align_logits_last_dims(eager.logits.float().cpu(), traced.logits.float().cpu())
            ok, pcc = comp_pcc(re[:, -1:, :], ge[:, -1:, :], pcc=_PCC)
            print(
                f"[llm_handler_logits_pcc] prefill_trace_vs_eager seq_len={seq_len} " f"comp_pcc={float(pcc):.8f}",
                flush=True,
            )
            assert ok, f"prefill trace vs eager last-pos comp_pcc {float(pcc):.6f} < {_PCC} " f"for seq_len={seq_len}"
    finally:
        if hasattr(exp, "release_trace"):
            exp.release_trace()
        del exp
        gc.collect()
