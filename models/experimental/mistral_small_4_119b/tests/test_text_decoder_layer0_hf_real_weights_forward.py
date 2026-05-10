# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real Hugging Face ``Mistral4DecoderLayer`` (layer 0) forward with checkpoint weights.

This loads the **full** ``mistralai/Mistral-Small-4-119B-2603`` multimodal model via
``AutoModelForImageTextToText.from_pretrained`` so transformers applies the same
FP8 / scale handling as production.

**Why this often skips on a TTNN box**

Tenstorrent bring-up machines frequently have **no NVIDIA CUDA** in PyTorch.
Transformers still expects **CUDA or Intel XPU** to handle this hub’s **FP8**
language-model weights. On **CPU-only** it tries FP8→bf16 dequantization and hits
``ValueError: not enough values to unpack`` in ``finegrained_fp8.py`` (scalar
``weight_scale_inv``), so **full** ``from_pretrained`` + layer-0 forward cannot
run there today.

**What to use instead (same box, real BF16 shard weights, TTNN)**

Run the **input_layernorm** stub PCC (hub BF16 tensor, TTNN RMSNorm, silicon):

``pytest models/experimental/mistral_small_4_119b/tests/test_text_decoder_layer0_input_norm_stub_pcc.py -q``

**When this test can run**

* **Default device selection**: PyTorch **CUDA** if available, else **Intel XPU**
  if ``torch.xpu.is_available()``, else **skip** with a short explanation.
* ``MS4_HF_REAL_LAYER0_DEVICE=cuda`` / ``=xpu`` / ``=cpu`` to force (CPU usually
  still fails for FP8 as above).
* Peak memory is huge (119B); with CUDA, ``device_map="auto"`` lets ``accelerate``
  shard across multiple GPUs when present.

**Run**

.. code-block:: bash

   MS4_RUN_HF_REAL_LAYER0_FORWARD=1 pytest \\
     models/experimental/mistral_small_4_119b/tests/test_text_decoder_layer0_hf_real_weights_forward.py -q
"""

import os

import pytest
import torch

from models.experimental.mistral_small_4_119b.constants import HF_MODEL_ID


pytest.importorskip("transformers")
pytest.importorskip("transformers.models.mistral4.modeling_mistral4", reason="Mistral4 requires recent transformers")


def _mistral4_layer0_and_rotary(hf_model: torch.nn.Module):
    """
    Resolve ``(decoder_layer_0, rotary_emb)`` from a loaded Mistral3 multimodal model.

    Layout matches hub keys ``language_model.model.layers.0.*``: the text trunk is
    ``language_model`` (Mistral4ForCausalLM-style) with an inner ``.model`` (``Mistral4Model``)
    exposing ``layers`` and ``rotary_emb``.
    """
    root = getattr(hf_model, "model", hf_model)
    lm = getattr(root, "language_model", None)
    if lm is None:
        raise AssertionError(
            f"Expected hf_model.model.language_model; got root={type(root).__name__}, "
            f"attrs={sorted(a for a in dir(root) if not a.startswith('_'))}"
        )
    # Mistral4ForCausalLM: .model is Mistral4Model with .layers / .rotary_emb
    text_core = getattr(lm, "model", lm)
    layers = getattr(text_core, "layers", None)
    if layers is None or len(layers) == 0:
        raise AssertionError(
            f"No text layers on {type(text_core).__name__}; attrs="
            f"{sorted(a for a in dir(text_core) if not a.startswith('_'))}"
        )
    rotary = getattr(text_core, "rotary_emb", None)
    if rotary is None:
        raise AssertionError(f"No rotary_emb on {type(text_core).__name__}")
    return layers[0], rotary


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _hf_load_device() -> torch.device:
    """Pick torch device for ``from_pretrained`` (FP8 hub path: CUDA or XPU per transformers, not CPU)."""
    choice = os.getenv("MS4_HF_REAL_LAYER0_DEVICE", "").strip().lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            pytest.skip("MS4_HF_REAL_LAYER0_DEVICE=cuda but torch.cuda.is_available() is False.")
        return torch.device("cuda", torch.cuda.current_device())
    if choice == "xpu":
        if not _xpu_available():
            pytest.skip("MS4_HF_REAL_LAYER0_DEVICE=xpu but torch.xpu.is_available() is False.")
        return torch.device("xpu")

    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    if _xpu_available():
        return torch.device("xpu")

    pytest.skip(
        "No PyTorch CUDA or Intel XPU: this FP8 hub checkpoint does not load on CPU with current "
        "transformers (finegrained_fp8 scalar scale dequant bug). TTNN hosts without a CUDA GPU "
        "should use real BF16 norm PCC on silicon instead: "
        "pytest models/experimental/mistral_small_4_119b/tests/test_text_decoder_layer0_input_norm_stub_pcc.py -q. "
        "To force a CPU load attempt anyway: MS4_HF_REAL_LAYER0_DEVICE=cpu."
    )


def _from_pretrained_multimodal(local_only: bool, dev: torch.device):
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

    base = dict(trust_remote_code=True, local_files_only=local_only)

    def _one_call(**extra):
        kw = {**base, **extra}
        try:
            return AutoModelForImageTextToText.from_pretrained(HF_MODEL_ID, **kw)
        except TypeError:
            # Older transformers: ``torch_dtype`` instead of ``dtype``.
            if "dtype" in kw:
                kw = {k: v for k, v in kw.items() if k != "dtype"}
                kw["torch_dtype"] = torch.bfloat16
                return AutoModelForImageTextToText.from_pretrained(HF_MODEL_ID, **kw)
            raise

    attempts = []
    if dev.type == "cuda":
        cid = torch.cuda.current_device()
        # Prefer ``device_map="auto"`` so multiple CUDA devices can shard weights.
        attempts.append({"dtype": torch.bfloat16, "device_map": "auto"})
        attempts.append({"dtype": torch.bfloat16, "device_map": f"cuda:{cid}"})
    elif dev.type == "xpu":
        attempts.append({"dtype": torch.bfloat16, "device_map": "xpu"})
        attempts.append({"dtype": torch.bfloat16, "device_map": {"": "xpu"}})
    else:
        attempts.append({"dtype": torch.bfloat16, "low_cpu_mem_usage": True})

    last_exc = None
    for extra in attempts:
        try:
            return _one_call(**extra)
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("no load attempts configured")


@pytest.mark.slow
def test_mistral4_decoder_layer0_forward_real_checkpoint_weights():
    if os.environ.get("MS4_RUN_HF_REAL_LAYER0_FORWARD") != "1":
        pytest.skip(
            "Loads the full 119B HF checkpoint (very high RAM / VRAM). " "Set MS4_RUN_HF_REAL_LAYER0_FORWARD=1 to run."
        )

    dev = _hf_load_device()
    local_only = os.getenv("CI") == "true"

    try:
        model = _from_pretrained_multimodal(local_only, dev)
    except Exception as exc:
        pytest.skip(f"from_pretrained failed (hub, OOM, FP8 conversion, or deps): {exc}")

    model.eval()
    text_cfg = model.config.text_config
    if hasattr(text_cfg, "attn_implementation"):
        text_cfg.attn_implementation = "eager"
    if hasattr(text_cfg, "_attn_implementation"):
        text_cfg._attn_implementation = "eager"

    layer0, rotary = _mistral4_layer0_and_rotary(model)
    run_dev = next(layer0.parameters()).device

    batch, seq = 1, 8
    hidden = text_cfg.hidden_size
    torch.manual_seed(0)
    x = torch.randn(batch, seq, hidden, dtype=torch.bfloat16, device=run_dev)
    position_ids = torch.arange(seq, dtype=torch.long, device=run_dev).unsqueeze(0).expand(batch, -1)
    position_embeddings = rotary(x, position_ids)

    with torch.no_grad():
        y = layer0(
            x,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
        )

    assert y.shape == x.shape
    assert torch.isfinite(y).all(), "layer 0 forward produced non-finite values"
