# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Diagnostic: per-layer comparison of the TTNN GemmaEncoder vs HF reference.

Reproduces the "explodes after layer 1" symptom and localizes it: for each of
the first N layers we report NaN count, max-abs magnitude, and PCC over the real
(non-pad) token positions, for both the masked (left-pad causal+pad) and the
unmasked (pure causal) attention paths.

Run on a single Blackhole chip (1x1 mesh, tp=1, no CCL):
    pytest models/tt_dit/tests/encoders/gemma/test_gemma_device_diag.py -s
"""

import glob
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[6]))

import gc

import pytest
import torch
from loguru import logger
from safetensors.torch import load_file

import ttnn
from models.tt_dit.encoders.gemma.model_gemma import GemmaConfig, GemmaEncoder
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager

N_LAYERS = int(os.environ.get("N_LAYERS", "6"))
SEQ_LEN = 128  # must be >= SDPA q/k chunk size (128) in model_gemma to avoid a device hang
# Long enough to fill all SEQ_LEN slots with REAL tokens (no padding). With no
# padding the attention mask is all-ones, so the reference runs pure-causal and the
# device is_causal path matches exactly — clean per-layer numerics, no mask confound.
PROMPT = (
    "A plump orange tabby cat sits on a worn piano bench in a sunlit living room, "
    "carefully playing the keys with its soft paws while dust motes drift through "
    "the warm afternoon light. Outside the tall windows a gentle breeze stirs the "
    "garden, and the faint melody echoes down the hallway past framed photographs, "
    "old books stacked on the floor, and a sleeping dog curled on a faded rug near "
    "the fireplace, utterly content in the quiet rhythm of the unhurried day. "
    "Down the street a baker pulls warm loaves from the oven, their crust crackling "
    "as steam rises into the cool morning air, and children chase a bright red ball "
    "across the cobblestones while pigeons scatter and wheel above the rooftops. "
    "Far beyond the town the mountains stand blue and patient under a wide pale sky, "
    "their snowcaps catching the first gold of dawn as a river winds slowly through "
    "the valley, carrying leaves and the reflections of clouds toward the distant sea."
)


def _gemma_path() -> str:
    explicit = os.environ.get("GEMMA_PATH")
    if explicit:
        return explicit
    cands = glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/*/")
    )
    if cands:
        return cands[0].rstrip("/")
    return "google/gemma-3-12b-it-qat-q4_0-unquantized"


def pcc(a, b):
    a_f, b_f = a.flatten().float(), b.flatten().float()
    a_m, b_m = a_f - a_f.mean(), b_f - b_f.mean()
    d = (a_m.pow(2).sum() * b_m.pow(2).sum()).sqrt()
    return ((a_m * b_m).sum() / d).item() if d > 0 else 0.0


def _stats(name, t):
    t = t.float()
    n_nan = int(torch.isnan(t).sum().item())
    n_inf = int(torch.isinf(t).sum().item())
    finite = t[torch.isfinite(t)]
    mx = finite.abs().max().item() if finite.numel() else float("nan")
    return f"{name}: nan={n_nan} inf={n_inf} max|x|={mx:.3e}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_gemma_device_per_layer(*, mesh_device):
    gemma_path = _gemma_path()
    if not os.path.isdir(gemma_path):
        pytest.skip(f"Gemma not found: {gemma_path}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(gemma_path)
    tokens = tok(PROMPT, return_tensors="pt", padding="max_length", max_length=SEQ_LEN, truncation=True)
    n_real = int(tokens.attention_mask.sum().item())
    logger.info(f"tokens {tuple(tokens.input_ids.shape)}, real={n_real} (left-pad)")

    # --- HF reference hidden states (first N_LAYERS only, for speed) ---
    skip_ref = os.environ.get("SKIP_REF") == "1"
    ref_hs = None
    if skip_ref:
        logger.info("SKIP_REF=1 — device-only run, no PCC comparison")
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(gemma_path, torch_dtype=torch.bfloat16).eval()
        # Truncate the decoder stack to N_LAYERS so the CPU forward is ~8x cheaper.
        lm = ref_model
        for attr in ("model", "language_model", "layers"):
            if hasattr(lm, attr):
                if attr == "layers":
                    break
                lm = getattr(lm, attr)
        assert hasattr(lm, "layers"), "could not locate decoder layers on reference model"
        lm.layers = lm.layers[:N_LAYERS]
        if hasattr(lm, "config"):
            lm.config.num_hidden_layers = N_LAYERS
        logger.info(f"truncated reference to {len(lm.layers)} layers")
        with torch.no_grad():
            out = ref_model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
            )
        ref_hs = [h.float() for h in out.hidden_states]  # embed + N_LAYERS
        logger.info(f"ref hidden states: {len(ref_hs)}")
        del ref_model, out
        gc.collect()

    # --- device encoder: first N_LAYERS only (tp=1) ---
    weight_files = sorted(glob.glob(f"{gemma_path}/model-*.safetensors"))
    full_sd = {}
    for f in weight_files:
        full_sd.update(load_file(f))

    # keep embed_tokens, final norm, and first N_LAYERS decoder layers
    keep = {}
    for k, v in full_sd.items():
        kk = k
        if kk.startswith("language_model.model.layers."):
            li = int(kk.split("language_model.model.layers.")[1].split(".")[0])
            if li >= N_LAYERS:
                continue
        keep[k] = v
    del full_sd
    gc.collect()

    config = GemmaConfig(num_hidden_layers=N_LAYERS, max_position_embeddings=SEQ_LEN)
    enc_parallel = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=1))
    enc_ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    encoder = GemmaEncoder(config, mesh_device, enc_ccl, enc_parallel)
    inc = encoder.load_torch_state_dict(keep, strict=False)
    logger.info(f"load: missing={len(inc.missing_keys)} unexpected={len(inc.unexpected_keys)}")
    del keep
    gc.collect()

    tt_ids = ttnn.from_torch(tokens.input_ids, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    real_slice = slice(SEQ_LEN - n_real, SEQ_LEN)  # left-pad → real tokens at the end

    # Run only the pure-causal path (is_causal=True, no mask). The masked/-inf-row
    # path hangs the SDPA kernel and is investigated separately (RUN_MASKED=1).
    variants = [("causal-only", None)]
    if os.environ.get("RUN_MASKED") == "1":
        variants.insert(0, ("masked", tokens.attention_mask))

    for label, mask in variants:
        logger.info(f"================ {label} ================")
        hs = encoder(tt_ids, attention_mask=mask)
        # hs: [embed, layer0..layerN-1, final_norm]
        n_cmp = len(ref_hs) if ref_hs is not None else len(hs) - 1
        for i in range(min(len(hs) - 1, n_cmp)):
            dev = ttnn.to_torch(ttnn.get_device_tensors(hs[i])[0]).float()
            if dev.dim() == 4:
                dev = dev[0]
            if ref_hs is None:
                logger.info(f"  L{i:02d}  {_stats('dev', dev)}")
                sys.stderr.flush()
                continue
            ref = ref_hs[i]
            p_all = pcc(dev, ref)
            p_real = pcc(dev[:, real_slice, :], ref[:, real_slice, :])
            logger.info(
                f"  L{i:02d}  {_stats('dev', dev)}  ref_max|x|={ref.abs().max().item():.3e}"
                f"  | PCC_all={p_all:.4f} PCC_real={p_real:.4f}"
            )
            sys.stderr.flush()
        for h in hs:
            ttnn.deallocate(h)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])
