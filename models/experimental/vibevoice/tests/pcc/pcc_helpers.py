# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the VibeVoice LM PCC tests: HuggingFace Qwen2 reference construction,
TT LM builder + a layer-level decode probe, and PCC comparison/reporting utilities. Used by the
full prefill/decode chain tests and the decoder-layer decode PCC sweep."""

from __future__ import annotations

from typing import NamedTuple

import torch
import ttnn
from transformers.cache_utils import DynamicCache

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.tt.ttnn_vibevoice_lm import (
    TTVibeVoiceLM,
    preprocess_lm_weights,
)
from tests.ttnn.utils_for_testing import assert_numeric_metrics


PCC_THRESHOLD = 0.99
# Per-position median gate for LM hidden: flattened PCC is dominated by a few
# massive-activation outlier positions (see ``per_position_pcc``). Seed-2 ISL sweep
# measured min median ≈ 0.963 at ISL=512; keep headroom under that floor.
HIDDEN_MEDIAN_THRESHOLD = 0.96


DECODE_GENERATION_LENGTH = 10


PREFILL_ISL_EXTENDED_SWEEP_LENGTHS = [
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
]


PREFILL_CHUNK_SIZE = 256


_HF_MODEL_CACHE: dict[tuple[int, str], torch.nn.Module] = {}


DEFAULT_HF_DECODE_ATTN_IMPLEMENTATION = "sdpa"


def _remap_lm_state_to_hf(lm_state: dict) -> dict:
    hf_state = {}
    for k, v in lm_state.items():
        hf_k = k
        hf_k = hf_k.replace("tok_embeddings.", "embed_tokens.")
        hf_k = hf_k.replace(".attention.wq", ".self_attn.q_proj")
        hf_k = hf_k.replace(".attention.wk", ".self_attn.k_proj")
        hf_k = hf_k.replace(".attention.wv", ".self_attn.v_proj")
        hf_k = hf_k.replace(".attention.wo", ".self_attn.o_proj")
        hf_k = hf_k.replace(".feed_forward.w1", ".mlp.gate_proj")
        hf_k = hf_k.replace(".feed_forward.w3", ".mlp.up_proj")
        hf_k = hf_k.replace(".feed_forward.w2", ".mlp.down_proj")
        hf_k = hf_k.replace(".attention_norm", ".input_layernorm")
        hf_k = hf_k.replace(".ffn_norm", ".post_attention_layernorm")
        hf_k = hf_k.replace("norm.weight", "norm.weight")
        hf_state[hf_k] = v
    return hf_state


def _get_hf_reference_model(
    lm_state: dict,
    vv_config,
    *,
    attn_implementation: str = DEFAULT_HF_DECODE_ATTN_IMPLEMENTATION,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    """Build and cache one HF Qwen2Model per ``(lm_state, attn_implementation, dtype)``."""
    cache_key = (id(lm_state), attn_implementation, dtype)
    if cache_key not in _HF_MODEL_CACHE:
        from transformers import Qwen2Config, Qwen2Model

        cfg_dec = vv_config.decoder
        hf_cfg = Qwen2Config(
            hidden_size=cfg_dec.hidden_size,
            num_hidden_layers=cfg_dec.num_hidden_layers,
            num_attention_heads=cfg_dec.num_attention_heads,
            num_key_value_heads=cfg_dec.num_key_value_heads,
            intermediate_size=cfg_dec.intermediate_size,
            vocab_size=cfg_dec.vocab_size,
            rope_theta=cfg_dec.rope_theta,
            rms_norm_eps=cfg_dec.rms_norm_eps,
            max_position_embeddings=cfg_dec.max_position_embeddings,
            attn_implementation=attn_implementation,
        )
        model = Qwen2Model(hf_cfg)
        model.load_state_dict(_remap_lm_state_to_hf(lm_state), strict=False)
        model.to(dtype)
        model.eval()
        _HF_MODEL_CACHE[cache_key] = model
    return _HF_MODEL_CACHE[cache_key]


class _TTVibeVoiceLMLayerProbe(TTVibeVoiceLM):
    """Test-only subclass: capture hidden states after embed and each transformer layer."""

    def forward_decoder_layer_hidden(
        self,
        hidden: torch.Tensor,
        start_pos: int,
        kv_cache,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Run one decoder layer on ``hidden`` [B, 1, H]; returns [B, 1, H] float32 (Devstral-style)."""
        x = hidden_torch_to_tt(hidden, self.device)
        x = self._transformer_layer(x, layer_idx, (self._cos_tt, self._sin_tt), kv_cache, start_pos)
        return _tt_tensor_to_hidden_torch(x)


def _tt_tensor_to_hidden_torch(x: ttnn.Tensor) -> torch.Tensor:
    """Convert [B, 1, S, H] device tensor to float32 torch [B, S, H]."""
    return ttnn.to_torch(ttnn.typecast(x, ttnn.float32)).to(torch.float32).squeeze(1)


def as_layer_probe(lm_tt: TTVibeVoiceLM) -> _TTVibeVoiceLMLayerProbe:
    """Rebind an existing LM instance as a layer probe (test-only, no extra weight copy)."""
    probe = _TTVibeVoiceLMLayerProbe.__new__(_TTVibeVoiceLMLayerProbe)
    probe.__dict__.update(lm_tt.__dict__)
    return probe


def hidden_torch_to_tt(hidden: torch.Tensor, device) -> ttnn.Tensor:
    """``hidden`` [B, 1, H] bf16 → TT ``[B, 1, 1, H]`` TILE."""
    if hidden.dim() != 3 or hidden.shape[1] != 1:
        raise ValueError(f"expected hidden [B, 1, H], got {tuple(hidden.shape)}")
    B, _, H = hidden.shape
    return ttnn.from_torch(
        hidden.reshape(B, 1, 1, H).to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def tt_weight_cache(submodule: str):
    """Default device-weight cache for one ``submodule``, keyed by the active checkpoint
    (``config.MODEL_PATH``, set by the conftest fixture). On by default and shared with
    ``from_checkpoint``'s cache files; ``VV_DISABLE_WEIGHT_CACHE=1`` disables it.
    """
    from models.experimental.vibevoice.common import config as _vv_config
    from models.experimental.vibevoice.common.weight_cache import resolve_weight_cache

    return resolve_weight_cache(_vv_config.MODEL_PATH, submodule)


def build_tt_lm(lm_state: dict, mesh_device, cfg) -> TTVibeVoiceLM:
    # Weight cache on by default (shares from_checkpoint's "lm" files); VV_DISABLE_WEIGHT_CACHE=1 off.
    weights = preprocess_lm_weights(lm_state, mesh_device, cfg, tt_weight_cache("lm"))
    return TTVibeVoiceLM(weights, mesh_device)


def compare_decode_hidden_pcc(ref_decode: torch.Tensor, tt_decode: torch.Tensor):
    """Compare single-step decode hidden states; returns (passed, pcc)."""
    ref_f = ref_decode.to(torch.float32)
    tt_f = tt_decode.to(torch.float32)
    if ref_f.shape != tt_f.shape:
        raise AssertionError(f"Decode hidden shape mismatch: ref={tuple(ref_f.shape)} tt={tuple(tt_f.shape)}")
    return comp_pcc(ref_f, tt_f, pcc=PCC_THRESHOLD)


def print_decode_pcc_summary(step_pccs: list[float]) -> None:
    """Print Step | PCC table for a multi-step decode sweep."""
    print("\nDecode PCC summary:")
    print("Step | PCC")
    print("-----|--------")
    for step, pcc in enumerate(step_pccs):
        print(f"{step:4d} | {pcc:.5f}")
    print(f"min={min(step_pccs):.5f}  mean={sum(step_pccs) / len(step_pccs):.5f}")


def compare_prefill_hidden_pcc(
    ref_prefill: torch.Tensor,
    tt_prefill: torch.Tensor,
    seq_len: int,
    *,
    per_token: bool = True,
):
    """Compare prefill hidden states; returns (passed, overall_pcc, per_position_pcc).

    Set ``per_token=False`` for long ISL sweeps — per-position ``comp_pcc`` in a Python loop
    scales linearly with sequence length and makes 2k+ runs impractically slow.
    """
    ref_f = ref_prefill.to(torch.float32)
    tt_f = tt_prefill.to(torch.float32)
    if ref_f.shape != tt_f.shape:
        raise AssertionError(
            f"Prefill hidden shape mismatch for seq_len={seq_len}: ref={tuple(ref_f.shape)} tt={tuple(tt_f.shape)}"
        )
    passed_p, pcc_p = comp_pcc(ref_f, tt_f, pcc=PCC_THRESHOLD)
    if not per_token:
        return passed_p, pcc_p, []
    per_pos = [comp_pcc(ref_f[:, p], tt_f[:, p], pcc=PCC_THRESHOLD)[1] for p in range(seq_len)]
    return passed_p, pcc_p, per_pos


def per_position_pcc(ref_prefill: torch.Tensor, tt_prefill: torch.Tensor) -> torch.Tensor:
    """Vectorized per-position PCC over the hidden dim; returns a 1-D ``[seq_len]`` tensor.

    Equivalent to ``comp_pcc(ref[:, p], tt[:, p])`` for every position but computed with
    tensor ops (no Python loop), so it stays fast for long ISL sweeps where the per-token
    loop in ``compare_prefill_hidden_pcc`` is impractical. Non-finite values are zeroed to
    match ``comp_pcc``.

    The per-position median of this is a length-stable accuracy metric: the overall flattened
    PCC is dominated by a few massive-activation positions/channels (|hidden| ~20x typical)
    whose bf16 rounding tanks the flattened correlation even reference-vs-reference, so it is
    not a reliable gate for long sequences.
    """
    ref_f = ref_prefill.to(torch.float32).reshape(-1, ref_prefill.shape[-1])
    tt_f = tt_prefill.to(torch.float32).reshape(-1, tt_prefill.shape[-1])
    if not bool((torch.isfinite(ref_f) & torch.isfinite(tt_f)).all()):
        ref_f = torch.nan_to_num(ref_f, nan=0.0, posinf=0.0, neginf=0.0)
        tt_f = torch.nan_to_num(tt_f, nan=0.0, posinf=0.0, neginf=0.0)
    ref_c = ref_f - ref_f.mean(dim=-1, keepdim=True)
    tt_c = tt_f - tt_f.mean(dim=-1, keepdim=True)
    cov = (ref_c * tt_c).sum(dim=-1, dtype=torch.float64)
    denom = torch.sqrt(ref_c.pow(2).sum(dim=-1, dtype=torch.float64) * tt_c.pow(2).sum(dim=-1, dtype=torch.float64))
    return (cov / denom.clamp_min(1e-30)).to(torch.float32)


def prefill_isl_sweep_effective_lengths(vv_config, isl_lengths=None) -> tuple[list[int], int]:
    """Return ISL list capped by ``decoder.max_position_embeddings`` and the model limit."""
    lengths = list(isl_lengths or PREFILL_ISL_EXTENDED_SWEEP_LENGTHS)
    max_pos = vv_config.decoder.max_position_embeddings
    effective = [n for n in lengths if n <= max_pos]
    return effective, max_pos


DECODE_LAYER_IDX = 0


DECODE_BATCH_SIZE = 1


class DecoderLayerPccContext(NamedTuple):
    hidden_size: int
    hf_layer: torch.nn.Module
    hf_rotary_emb: torch.nn.Module
    tt_probe: _TTVibeVoiceLMLayerProbe


def build_decoder_layer_pcc_context(mesh_device, lm_state, vv_config) -> DecoderLayerPccContext:
    """Layer-0 fixtures for decode PCC (empty KV cache, positions 0 … N-1)."""
    cfg = vv_config.decoder
    model = _get_hf_reference_model(lm_state, vv_config)
    lm_tt = build_tt_lm(lm_state, mesh_device, cfg)
    return DecoderLayerPccContext(
        hidden_size=cfg.hidden_size,
        hf_layer=model.layers[DECODE_LAYER_IDX],
        hf_rotary_emb=model.rotary_emb,
        tt_probe=as_layer_probe(lm_tt),
    )


def reference_decoder_layer_decode_forward(
    layer: torch.nn.Module,
    rotary_emb: torch.nn.Module,
    hidden: torch.Tensor,
    *,
    position: int,
    cache: DynamicCache,
) -> torch.Tensor:
    """Single decode step on HF ``Qwen2DecoderLayer`` with ``DynamicCache``."""
    pos = torch.tensor([[position]], dtype=torch.long, device=hidden.device)
    cache_position = torch.tensor([position], dtype=torch.long, device=hidden.device)
    cos, sin = rotary_emb(hidden, pos)
    with torch.no_grad():
        # HF decoder layers return a tuple (hidden_states, ...); cache kwarg is singular.
        out = layer(
            hidden,
            position_ids=pos,
            past_key_value=cache,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=(cos, sin),
        )
    hidden_out = out[0] if isinstance(out, (tuple, list)) else out
    if hidden_out.dim() == 2:
        hidden_out = hidden_out.unsqueeze(1)
    return hidden_out.float()


def tt_decoder_layer_decode_forward(
    probe: _TTVibeVoiceLMLayerProbe,
    hidden: torch.Tensor,
    *,
    position: int,
    kv_cache,
    layer_idx: int = DECODE_LAYER_IDX,
) -> torch.Tensor:
    """Single decode step on TT decoder layer (``hidden`` [B, 1, H] bf16)."""
    return probe.forward_decoder_layer_hidden(hidden, position, kv_cache, layer_idx=layer_idx)


# Decoder-layer decode numeric floors (seed-0 random hiddens, positions 0…9).
# Historical baseline (pre nlp_create_qkv_heads / pre flash-decode): min PCC≈0.999985.
# Current BH production path (fused or fp32 RoPE + bf16 flash-decode / fp32 manual) measures
# min PCC≈0.9982, max|Δ|≈0.20, rel-Frob≈0.060 on this random-hidden probe. Floors track that
# with headroom; e2e/decode PCC cover end-to-end quality.
DECODE_LAYER_PCC_THRESHOLD = 0.9975
DECODE_LAYER_RTOL = 0.09
DECODE_LAYER_ATOL = 0.22
DECODE_LAYER_FROBENIUS_THRESHOLD = 0.07


def run_decoder_layer_decode_pcc_sweep(
    mesh_device,
    lm_state,
    vv_config,
    *,
    num_steps: int = DECODE_GENERATION_LENGTH,
) -> list[float]:
    """10 decode steps at positions 0–9 with random hidden states and empty KV cache."""
    ctx = build_decoder_layer_pcc_context(mesh_device, lm_state, vv_config)
    hf_cache = DynamicCache()
    kv_cache = ctx.tt_probe.alloc_kv_cache(num_steps + 8)

    failures: list[str] = []
    step_pccs: list[float] = []

    print(
        f"[decoder layer decode PCC] layer={DECODE_LAYER_IDX} batch={DECODE_BATCH_SIZE} "
        f"hidden={ctx.hidden_size} steps={num_steps} positions=0–{num_steps - 1} "
        f"(no prefill, random hiddens)"
    )

    torch.manual_seed(0)
    for step in range(num_steps):
        hidden = (torch.rand(DECODE_BATCH_SIZE, 1, ctx.hidden_size, dtype=torch.bfloat16) * 2) - 1

        ref_out = reference_decoder_layer_decode_forward(
            ctx.hf_layer,
            ctx.hf_rotary_emb,
            hidden,
            position=step,
            cache=hf_cache,
        )
        tt_out = tt_decoder_layer_decode_forward(
            ctx.tt_probe,
            hidden,
            position=step,
            kv_cache=kv_cache,
        )

        # Align shapes to [B, 1, H] float32 (HF may be [B,1,H]; TT probe returns [B,S,H]).
        ref_f = ref_out.to(torch.float32)
        tt_f = tt_out.to(torch.float32)
        if ref_f.dim() == 2:
            ref_f = ref_f.unsqueeze(1)
        if tt_f.dim() == 2:
            tt_f = tt_f.unsqueeze(1)
        if ref_f.shape != tt_f.shape:
            raise AssertionError(f"Decode hidden shape mismatch: ref={tuple(ref_f.shape)} tt={tuple(tt_f.shape)}")

        passed_d, msg = assert_numeric_metrics(
            ref_f,
            tt_f,
            pcc_threshold=DECODE_LAYER_PCC_THRESHOLD,
            rtol=DECODE_LAYER_RTOL,
            atol=DECODE_LAYER_ATOL,
            frobenius_threshold=DECODE_LAYER_FROBENIUS_THRESHOLD,
            assert_on_fail=False,
        )
        # Keep a PCC scalar for the summary table.
        _, pcc_d = comp_pcc(ref_f, tt_f, pcc=DECODE_LAYER_PCC_THRESHOLD)
        step_pccs.append(float(pcc_d))
        print(f"Decode step {step}  position={step}  PCC={float(pcc_d):.5f}")
        print(msg)

        if not passed_d:
            failures.append(f"decode step={step} position={step}\n{msg}")

    print_decode_pcc_summary(step_pccs)
    if failures:
        raise AssertionError("Decoder layer decode numeric metrics below threshold:\n" + "\n".join(failures))

    return step_pccs
