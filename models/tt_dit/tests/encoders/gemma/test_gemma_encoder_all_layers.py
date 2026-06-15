# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Per-layer parity: TTNN GemmaEncoder vs HF reference, real-token PCC over the first
``N_LAYERS`` decoder layers (the full 12B encoder doesn't fit one chip at tp=1).

    pytest models/tt_dit/tests/encoders/gemma/test_gemma_encoder_all_layers.py -s
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
from models.tt_dit.utils.check import assert_quality

N_LAYERS = int(os.environ.get("N_LAYERS", "6"))
SEQ_LEN = 128  # must be >= SDPA q/k chunk size (128) in model_gemma to avoid a device hang
# Per-state real-token PCC floor at the default N_LAYERS=6 (measured min ~0.9963 at the final
# norm; shallower states ride ~0.9998). bf16 drifts with depth, so a much larger N_LAYERS may
# need a lower floor.
PCC_BAR = 0.995
# Long enough to fill all SEQ_LEN slots with real tokens (no padding), so the reference
# runs pure-causal and the device is_causal path matches exactly — no mask confound.
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


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_gemma_layers_individually(*, mesh_device):
    gemma_path = _gemma_path()
    if not os.path.isdir(gemma_path):
        pytest.skip(f"Gemma not found: {gemma_path}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(gemma_path)
    tokens = tok(PROMPT, return_tensors="pt", padding="max_length", max_length=SEQ_LEN, truncation=True)
    n_real = int(tokens.attention_mask.sum().item())
    logger.info(f"tokens {tuple(tokens.input_ids.shape)}, real={n_real} (left-pad)")

    # --- HF reference hidden states (first N_LAYERS only, for speed) ---
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
    with torch.no_grad():
        out = ref_model(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            output_hidden_states=True,
        )
    ref_hs = [h.float() for h in out.hidden_states]  # embed + N_LAYERS
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
        if k.startswith("language_model.model.layers."):
            li = int(k.split("language_model.model.layers.")[1].split(".")[0])
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

    # Pure-causal path (no mask): all positions real, so the device is_causal path
    # matches the reference exactly.
    hs = encoder(tt_ids, attention_mask=None)  # device: [embed, L0..L_{N-1}, final_norm]
    # HF output_hidden_states is [embed, L0..L_{N-2}, final_norm(L_{N-1})] — its last entry is
    # post-final-norm, not the raw last layer. Align the shared prefix directly and the final
    # norm at [-1]; the device's raw last-layer state has no HF counterpart.
    device_states = list(hs[: len(ref_hs) - 1]) + [hs[-1]]
    for i, (ref, dev_tt) in enumerate(zip(ref_hs, device_states)):
        dev = ttnn.to_torch(ttnn.get_device_tensors(dev_tt)[0]).float()
        if dev.dim() == 4:
            dev = dev[0]
        assert torch.isfinite(dev).all(), f"state {i}: device output has NaN/Inf"
        logger.info("  final_norm" if i == len(ref_hs) - 1 else f"  L{i:02d}")
        assert_quality(ref[:, real_slice, :], dev[:, real_slice, :], pcc=PCC_BAR)
    for h in hs:
        ttnn.deallocate(h)
