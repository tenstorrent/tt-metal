# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
**Scale-depth smoke:** ``N`` × decoder text stack on mesh with **TTNN MoE** (device routing), up to
``EXPECTED_NUM_LAYERS`` (36).

This is **not** run in default CI: it pulls **N × decoder** keys from the hub checkpoint, allocates large
device state, and can exceed the repo default pytest timeout (~300s).

**DRAM (Blackhole P150×4):** initializing **36** layers of bf16 sharded MoE + attention can **OOM** during
``from_torch``/tilize for expert weights (~1 GiB peak per tensor plus cumulative weights). If init fails with
out-of-memory, the test **skips** with a hint. Reduce depth until it fits::

    export MISTRAL4_E2E_DEEP_N_LAYERS=10   # try 8–16 on 4×P150; raise toward 36 as memory allows

Run manually::

    export MISTRAL4_E2E_36L_TTNN_MOE=1
    export MISTRAL4_E2E_DEEP_N_LAYERS=36   # optional; default is 36
    export MISTRAL4_E2E_DEEP_PCC=1        # optional; N must be 1 or 2; PCC vs host-MoE Tt stack (HF routing on TT)
    export MESH_DEVICE=P150x4             # optional
    export MESH_DEVICE=single             # optional: 1×1 mesh; fabric disabled (no QSFP mesh required)
    pytest models/experimental/mistral_small_4_119b/tests/test_text_prefill_e2e_36l_ttnn_moe_smoke.py -v -s \\
        --timeout=1200

Checks: ``TtMistral4TextPrefillLogits`` init, one **short** prefill forward (``seq_len=2``), logits shape
``[1, S, V]``, all finite.

**PCC (opt-in):** set ``MISTRAL4_E2E_DEEP_PCC=1`` and ``MISTRAL4_E2E_DEEP_N_LAYERS`` to ``1`` or ``2`` only.
Builds a **host-MoE** ``TtMistral4TextPrefillLogits`` (``use_ttnn_moe=False``) first, records logits, frees it,
then builds the TTNN-MoE stack with ``moe_hf_torch_routing=True`` and asserts ``comp_pcc`` vs that reference
(same floors as ``test_text_prefill_e2e_logits_pcc.py``: **~0.92** for ``N=1``, **~0.76** for ``N=2``).
Pure **HF torch** logits vs TTNN MoE can sit near **~0.03** on short prefill (``lm_head`` amplifies device MoE
drift); this smoke therefore does **not** assert vs HF. For ``N > 2`` or default device routing, no PCC here.
"""

from __future__ import annotations

import gc
import os

import pytest
import torch
import torch.nn.functional as F
import ttnn
from loguru import logger

from models.common.utility_functions import comp_allclose, comp_pcc, run_for_wormhole_b0_or_blackhole
from models.experimental.mistral_small_4_119b.constants import (
    EXPECTED_NUM_LAYERS,
    EXPECTED_VOCAB_SIZE,
    HF_MODEL_ID,
    TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY,
    text_decoder_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tt.mistral4_self_attention import _torch_for_ttnn_upload
from models.experimental.mistral_small_4_119b.tt.mistral4_text_prefill import TtMistral4TextPrefillLogits
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered


pytest.importorskip("transformers")
pytest.importorskip("transformers.models.mistral4.modeling_mistral4", reason="Mistral4 required")


def _text_config_eager_attn():
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load HF config: {exc}")
    text = cfg.text_config
    if text is None:
        pytest.skip("Config has no text_config")
    if hasattr(text, "attn_implementation"):
        text.attn_implementation = "eager"
    if hasattr(text, "_attn_implementation"):
        text._attn_implementation = "eager"
    return text


def _prefixes_e2e(num_decoder_layers: int) -> tuple[str, ...]:
    p: list[str] = ["language_model.model.embed_tokens."]
    for i in range(num_decoder_layers):
        p.append(text_decoder_layer_state_dict_prefix(i))
    p.append("language_model.model.norm.")
    p.append("language_model.lm_head.")
    return tuple(p)


def _load_state_dict(num_decoder_layers: int) -> dict:
    try:
        return load_hf_state_dict_filtered(HF_MODEL_ID, _prefixes_e2e(num_decoder_layers))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Checkpoint load failed: {exc}")


def _e2e_deep_want_logits_pcc() -> bool:
    return os.environ.get("MISTRAL4_E2E_DEEP_PCC") == "1"


def _e2e_deep_num_layers() -> int:
    raw = os.environ.get("MISTRAL4_E2E_DEEP_N_LAYERS", str(EXPECTED_NUM_LAYERS)).strip()
    try:
        n = int(raw)
    except ValueError as exc:
        raise ValueError(f"MISTRAL4_E2E_DEEP_N_LAYERS must be an integer, got {raw!r}") from exc
    if n < 1 or n > EXPECTED_NUM_LAYERS:
        raise ValueError(f"MISTRAL4_E2E_DEEP_N_LAYERS must be in [1, {EXPECTED_NUM_LAYERS}], got {n}")
    return n


def _is_dram_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "not enough space" in msg or "oom" in msg


def _mistral4_e2e_mesh_and_device_params():
    """Mesh shape from ``MESH_DEVICE``; disable fabric for 1×1 so BH does not require Ethernet mesh links."""
    shape = mesh_device_request_param()
    base = {"trace_region_size": 30000000, "num_command_queues": 1}
    fabric = ttnn.FabricConfig.DISABLED if shape == (1, 1) else ttnn.FabricConfig.FABRIC_1D
    return [pytest.param(shape, {**base, "fabric_config": fabric}, id=f"mesh{shape[0]}x{shape[1]}")]


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.skipif(
    os.environ.get("MISTRAL4_E2E_36L_TTNN_MOE") != "1",
    reason="Set MISTRAL4_E2E_36L_TTNN_MOE=1 to run 36-layer TTNN MoE E2E smoke (large download, long runtime, high DRAM).",
)
@pytest.mark.parametrize("mesh_device, device_params", _mistral4_e2e_mesh_and_device_params(), indirect=True)
def test_mistral_small_4_text_prefill_e2e_36l_ttnn_moe_smoke(reset_seeds, mesh_device):
    """Prefill logits: embed → N× TTNN decoder (device MoE) → host norm+lm_head (``N`` from env, default 36)."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding

    n = _e2e_deep_num_layers()
    want_pcc = _e2e_deep_want_logits_pcc()
    if want_pcc and n > 2:
        pytest.skip(
            f"MISTRAL4_E2E_DEEP_PCC=1 with MISTRAL4_E2E_DEEP_N_LAYERS={n}: PCC path only supports "
            f"N in {{1, 2}} (memory / runtime). Unset MISTRAL4_E2E_DEEP_PCC for full-depth smoke, "
            f"or set MISTRAL4_E2E_DEEP_N_LAYERS to 1 or 2."
        )

    text = _text_config_eager_attn()
    logger.info(
        f"Loading filtered checkpoint for {n} decoder layers + embed + norm + lm_head "
        f"(MISTRAL4_E2E_DEEP_N_LAYERS, max={EXPECTED_NUM_LAYERS}) …"
    )
    state_dict = _load_state_dict(n)

    seq_len = 2
    w = state_dict[TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY]
    vocab = int(w.shape[0])
    assert vocab == EXPECTED_VOCAB_SIZE

    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab, (1, seq_len), dtype=torch.long)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    w_bf16 = _torch_for_ttnn_upload(w)
    hidden0 = F.embedding(input_ids, w_bf16).to(torch.bfloat16)
    rotary = Mistral4RotaryEmbedding(text).eval().to(torch.bfloat16)
    position_embeddings = rotary(hidden0, position_ids)

    ref_logits = None
    if want_pcc:
        try:
            ref_model = TtMistral4TextPrefillLogits(
                mesh_device,
                state_dict,
                text,
                num_decoder_layers=n,
                use_ttnn_moe=False,
            )
        except Exception as exc:
            if _is_dram_oom(exc):
                pytest.skip(
                    f"Host-MoE reference init OOM ({exc!r}); lower MISTRAL4_E2E_DEEP_N_LAYERS or disable DEEP_PCC."
                )
            pytest.skip(f"Host-MoE reference model init failed (MISTRAL4_E2E_DEEP_PCC=1): {exc}")
        ref_logits = ref_model(
            input_ids,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            mode="prefill",
        )
        del ref_model
        gc.collect()

    try:
        model = TtMistral4TextPrefillLogits(
            mesh_device,
            state_dict,
            text,
            num_decoder_layers=n,
            use_ttnn_moe=True,
            moe_hf_torch_routing=want_pcc,
        )
    except Exception as exc:
        if _is_dram_oom(exc):
            pytest.skip(
                f"Device DRAM OOM building {n} TTNN-MoE layers ({exc!r}). "
                f"Try export MISTRAL4_E2E_DEEP_N_LAYERS=8 (or 12–16) on P150×4, or more chips / bf8 MoE weights when available."
            )
        raise AssertionError(f"TtMistral4TextPrefillLogits({n} layers, TTNN MoE) init failed: {exc}") from exc

    logger.info("Running prefill forward …")
    try:
        tt_logits = model(
            input_ids,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            mode="prefill",
        )
    except Exception as exc:
        if _is_dram_oom(exc):
            pytest.skip(
                f"Device DRAM OOM during {n}-layer prefill forward ({exc!r}). "
                f"Try lowering MISTRAL4_E2E_DEEP_N_LAYERS or reducing activation memory (shorter seq, staging)."
            )
        raise

    assert tt_logits.ndim == 3
    assert tuple(tt_logits.shape) == (
        1,
        seq_len,
        EXPECTED_VOCAB_SIZE,
    ), f"logits shape {tuple(tt_logits.shape)} != (1, {seq_len}, {EXPECTED_VOCAB_SIZE})"
    assert tt_logits.dtype in (torch.bfloat16, torch.float32)
    assert torch.isfinite(tt_logits.to(torch.float32)).all(), "non-finite values in logits"

    if want_pcc:
        assert ref_logits is not None
        # Same bars as ``test_text_prefill_e2e_logits_pcc``: host-MoE Tt stack vs HF; here TTNN MoE vs host-MoE Tt.
        min_pcc = 0.76 if n >= 2 else 0.92
        passing, pcc_message = comp_pcc(ref_logits, tt_logits, pcc=min_pcc)
        logger.info(comp_allclose(ref_logits, tt_logits))
        logger.info(
            f"{n}-layer TTNN MoE vs host-MoE Tt prefill logits PCC (HF routing on TT, min={min_pcc}): {pcc_message}"
        )
        assert passing, f"PCC below {min_pcc}: {pcc_message}"

    logger.info(
        f"{n}-layer TTNN MoE E2E prefill smoke: OK (finite logits, expected shape"
        f"{'; PCC vs host-MoE Tt' if want_pcc else ''})"
    )
