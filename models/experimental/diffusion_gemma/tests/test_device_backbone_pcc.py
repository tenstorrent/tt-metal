# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""#47461 Stage-2 — DiffusionGemma causal backbone logits PCC on QB2.

This is #47461's *real* acceptance gate, distinct from a plain-gemma-on-QB2 run
(#47487 fit). It validates that the reused gemma4 backbone reproduces the HF
**DiffusionGemma** reference logits on the **fine-tuned** weights:

1. Load `DiffusionGemmaForBlockDiffusion` once (CPU, bf16).
2. Reference logits = its **causal text backbone** (`model.model.encoder`) ->
   `lm_head` -> final logit softcap. (The decoder is bidirectional + reads
   encoder KV; we use the *encoder* text model, which is the causal path.)
3. The encoder text weights are **tied** to `model.decoder.*`, so the same
   numerical weights are remapped into the gemma4 key namespace
   (`model.decoder.* -> model.language_model.*`) via `weight_mapping`.
4. Build the reused `Gemma4Model` on device from those weights
   (`tensor_cache_path=None` so no stale tensor cache shadows them) and run a
   causal prefill, then PCC TT-vs-HF.

Run on QB2 (4x Blackhole):

    source /home/zni/venvs/tt-diffusion-gemma/bin/activate
    export PYTHONPATH=/home/zni/tt-metal TT_METAL_HOME=/home/zni/tt-metal
    DG_RUN_DEVICE=1 MESH_DEVICE=P150x4 pytest \
      models/experimental/diffusion_gemma/tests/test_device_backbone_pcc.py -k "1x4" -v -s

Threshold is env-driven (`DG_BACKBONE_PCC`, default 0.83 — the established
Blackhole-1x4 baseline); the per-token argmax match is logged as the
text-quality signal.
"""

import gc
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import compare_tensors, parametrize_mesh_with_fabric

DG_CKPT = os.getenv(
    "DG_CKPT",
    "/home/zni/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/"
    "snapshots/0f28bc42f588fbd8f71e08102b1c3960298a1358",
)
GEMMA_CONFIG_DIR = os.getenv("GEMMA_CONFIG_DIR", "/home/zni/dg_models/gemma-4-26B-A4B-it")
PROMPT = os.getenv("DG_PROMPT", "The capital of France is")
PCC_THRESHOLD = float(os.getenv("DG_BACKBONE_PCC", "0.83"))

pytestmark = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (QB2, MESH_DEVICE=P150x4)",
)


def _reference_and_backbone_state(prompt):
    """Load DiffusionGemma once -> (tokenizer, input_ids, ref_logits, backbone_state).

    ref_logits[1, seq, vocab] = causal text backbone -> lm_head -> softcap.
    backbone_state = the same (tied) decoder weights remapped to gemma4 keys.
    """
    from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion

    from models.experimental.diffusion_gemma.weight_mapping import remap_state_dict

    tok = AutoTokenizer.from_pretrained(DG_CKPT)
    input_ids = tok.encode(prompt, return_tensors="pt")  # [1, seq]

    logger.info(f"Loading DiffusionGemmaForBlockDiffusion from {DG_CKPT} ...")
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        DG_CKPT, dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    softcap = float(model.config.text_config.final_logit_softcapping)

    with torch.no_grad():
        enc = model.model.encoder(input_ids=input_ids)  # builds causal mask internally
        hidden = enc.last_hidden_state  # [1, seq, hidden], final RMSNorm applied
        ref_logits = model.lm_head(hidden).float()
        ref_logits = torch.tanh(ref_logits / softcap) * softcap

    backbone_state, self_cond_state, ignored = remap_state_dict(model.state_dict())
    logger.info(f"remap: {len(backbone_state)} backbone keys, {len(self_cond_state)} self-cond, {len(ignored)} ignored")
    del model
    gc.collect()
    return tok, input_ids, ref_logits, backbone_state


@parametrize_mesh_with_fabric()
def test_diffusiongemma_backbone_logits_pcc(mesh_device, reset_seeds, request):
    import torch.nn.functional as F

    from models.common.utility_functions import comp_pcc
    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.ccl import CCLManager
    from models.demos.gemma4.tt.model import Gemma4Model
    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    if tp < 2:
        pytest.skip("26B-A4B backbone needs TP>=2 (use -k 1x4 on QB2)")

    # ── HF reference + remapped (tied) weights, one load ─────────────────────
    tok, input_ids, ref_logits, backbone_state = _reference_and_backbone_state(PROMPT)
    seq_len = input_ids.shape[1]
    padded_len = ((seq_len + 31) // 32) * 32
    input_ids_padded = F.pad(input_ids, (0, padded_len - seq_len), value=0) if padded_len > seq_len else input_ids
    logger.info(f"Prompt: '{PROMPT}' -> {seq_len} tokens (padded to {padded_len})")

    # ── gemma4 config (backbone arch is identical to plain Gemma-4 26B-A4B) ──
    hf_config = Gemma4ModelArgs.load_hf_config(GEMMA_CONFIG_DIR)
    model_args = Gemma4ModelArgs.from_hf_config(hf_config)
    model_args._hf_text_config = getattr(hf_config, "text_config", hf_config)

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device) if tp > 1 else None

    tt_model = Gemma4Model(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=backbone_state,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,  # in-memory weights only — never shadow with a stale cache
        mesh_config=mesh_config,
        max_seq_len=max(padded_len, 128),
        max_local_batch_size=1,
        create_kv_cache=True,
    )

    # ── TT causal prefill ────────────────────────────────────────────────────
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    tokens_tt = ttnn.from_torch(
        input_ids_padded.to(torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=replicate,
    )
    embeds = tt_model.embed_tokens(tokens_tt)
    embeds = ttnn.reshape(embeds, (1, 1, padded_len, model_args.hidden_size))
    embeds = ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)
    embeds_torch = (
        F.embedding(input_ids_padded.long(), backbone_state["model.language_model.embed_tokens.weight"])
        * tt_model.embed_scale
    ).float()

    tt_logits = tt_model.ttnn_prefill_forward(
        embeds,
        page_table=None,
        kv_cache=tt_model.tt_kv_cache,
        input_ids_torch=input_ids_padded,
        embeds_torch=embeds_torch,
    )
    tt_logits_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_logits)[0]).float()
    tt_logits.deallocate(True)
    if tt_logits_torch.dim() == 4:
        tt_logits_torch = tt_logits_torch.squeeze(1)

    # ── PCC (compare only the unpadded prefix) ───────────────────────────────
    hf_cmp = ref_logits[:, :seq_len, :]
    tt_cmp = tt_logits_torch[:, :seq_len, :]

    passing, pcc_msg = compare_tensors(tt_cmp, hf_cmp, pcc_threshold=PCC_THRESHOLD)
    logger.info(f"DiffusionGemma backbone logits PCC (seq_len={seq_len}, tp={tp}): {pcc_msg}")

    argmatch = 0
    for t in range(seq_len):
        _, pcc_t = comp_pcc(hf_cmp[0, t], tt_cmp[0, t], pcc=0.0)
        hf_tok = int(hf_cmp[0, t].argmax().item())
        tt_tok = int(tt_cmp[0, t].argmax().item())
        ok = hf_tok == tt_tok
        argmatch += int(ok)
        logger.info(
            f"  token[{t}] pcc={pcc_t:.5f} argmax HF={hf_tok} TT={tt_tok} "
            f"({'ok' if ok else 'MISMATCH'}) hf='{tok.decode([hf_tok])}' tt='{tok.decode([tt_tok])}'"
        )
    logger.info(f"argmax match: {argmatch}/{seq_len}  (greedy-decode equivalence)")

    assert passing, f"DiffusionGemma backbone PCC too low: {pcc_msg} (< {PCC_THRESHOLD})"
