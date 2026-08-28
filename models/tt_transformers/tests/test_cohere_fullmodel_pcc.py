# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal — bounty tt-metal#49307 track)
# SPDX-License-Identifier: Apache-2.0
#
# Command-R (c4ai-command-r-v01) FULL-MODEL PCC harness — P6.
#
# Chains ALL 40 CohereDecoderLayers sequentially (the TTNN output of layer LL feeds
# layer LL+1 — real error accumulation through the residual stream, not per-layer
# isolation), comparing each layer's output against the fp32 CPU reference dumps
# (tt-rd scripts/command-r/cpu_reference_capture.py, P4). Then applies the final
# norm + CohereLMHead (logit_scale 0.0625, tied embeddings) and compares against
# the reference `final_norm` / `logits` keys. Bounty gate: PCC >= 0.99.
#
# Reference dump contract (per prompt file prompt{NN}.npz, fp32 numpy):
#   embed            embed_tokens output           [1, S, 8192]  == input of decoder layer 0
#   layer{LL:02d}_out  decoder layer LL residual output [1, S, 8192]
#   final_norm       model.norm output             [1, S, 8192]
#   logits           lm_head output * logit_scale  [1, S, 256000]
#
# Env knobs:
#   COHERE_REF_DIR     dump directory (default /root/v4run/results/command-r/reference)
#   COHERE_REF_PROMPT  prompt index (default 0)
#   COHERE_LAYERS      layer range "A-B" (default "0-39") or comma list "0,1,5"
#   COHERE_GATE        PCC gate (default 0.99 — the bounty gate)
#
# Run on the QB2 host via the side-load pattern (see tt-rd
# results/2026-08-28-command-r-p5b-pcc-pass.md):
#   python_env/bin/python -m pytest models/tt_transformers/tests/test_cohere_fullmodel_pcc.py -v -s

import gc
import os

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.cohere.cohere_decoder import CohereDecoderLayer
from models.tt_transformers.tt.cohere.cohere_lm_head import CohereLMHead
from models.tt_transformers.tt.cohere.cohere_norm import build_cohere_final_norm
from models.tt_transformers.tt.common import Mode, get_rot_transformation_mat
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.rope import get_rot_mats

MESH_DEVICES = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}

# Cross-test handoff: the 40-layer sweep stashes its chained final hidden here so
# the logits test runs on the TRUE end-to-end residual stream (falls back to the
# reference layer39_out when the sweep did not run in this pytest process).
_CHAINED = {}


def _load_ref_npz():
    ref_dir = os.environ.get("COHERE_REF_DIR", "/root/v4run/results/command-r/reference")
    pidx = int(os.environ.get("COHERE_REF_PROMPT", "0"))
    path = os.path.join(ref_dir, f"prompt{pidx:02d}.npz")
    if not os.path.exists(path):
        pytest.skip(f"CPU reference dump {path} not present")
    logger.info(f"[cohere-fullmodel] reference dump: {path}")
    return np.load(path)


def _parse_layers():
    spec = os.environ.get("COHERE_LAYERS", "0-39").strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    if spec.lower() == "all":
        return list(range(40))
    return [int(x) for x in spec.split(",") if x.strip()]


def _pcc(a, b):
    """comp_pcc wrapper returning (passing, message, numeric_pcc)."""
    passing, msg = comp_pcc(a, b, pcc=float(os.environ.get("COHERE_GATE", "0.99")))
    # msg is the formatted pcc; recompute numerically for logging/summary
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    vx = a_f - a_f.mean()
    vy = b_f - b_f.mean()
    denom = vx.norm() * vy.norm()
    val = float((vx @ vy) / denom) if float(denom) > 0 else 0.0
    return passing, msg, val


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICES.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("max_seq_len", (128,))
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_cohere_fullmodel_40layer_pcc(max_seq_len, mesh_device, reset_seeds, ensure_gc):
    """Chain all 40 decoder layers from the captured `embed`; per-layer PCC vs the
    fp32 reference layer outputs. The chained hidden (TTNN out of layer LL) is the
    input of layer LL+1, so this measures true end-to-end error accumulation.
    Gate: EVERY layer >= 0.99 (bounty gate)."""
    dtype = ttnn.bfloat8_b  # stack default (test_decoder_prefill.py) — production dtype
    layers = _parse_layers()
    ref = _load_ref_npz()
    embed = torch.from_numpy(ref["embed"]).float()
    seq_len = embed.shape[1]
    assert seq_len <= max_seq_len, f"prompt seq {seq_len} > max_seq_len {max_seq_len}"

    model_args = ModelArgs(
        mesh_device, max_batch_size=1, max_seq_len=max_seq_len, cache_hf=True, use_hf_rope=False
    )
    state_dict = model_args.load_state_dict()  # full 40-layer checkpoint

    rot_mats = get_rot_mats(
        head_dim=model_args.head_dim,
        device=mesh_device,
        seq_len=max_seq_len,
        theta=model_args.rope_theta,
        rope_scaling=model_args.rope_scaling,
    )
    transformation_mats = {
        "prefill": ttnn.as_tensor(
            get_rot_transformation_mat(model_args.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
    }

    tt_ccl = TT_CCL(mesh_device)
    hidden = torch.nn.functional.pad(embed, (0, 0, 0, max_seq_len - seq_len)).to(torch.bfloat16)

    pccs = []
    worst = (None, 2.0)
    for layer_idx in layers:
        in_key = "embed" if layer_idx == 0 else f"layer{layer_idx - 1:02d}_out"
        out_key = f"layer{layer_idx:02d}_out"
        if out_key not in ref:
            pytest.skip(f"{out_key} not in reference dump")
        ref_in = torch.from_numpy(ref[in_key]).float() if in_key in ref else None
        ref_out = torch.from_numpy(ref[out_key]).float()

        # drift of the chained input vs the reference input for this layer
        in_pcc_msg = "n/a"
        if ref_in is not None:
            _, _, in_pcc = _pcc(ref_in, hidden[:, :seq_len].float())
            in_pcc_msg = f"{in_pcc:.6f}"

        tt_layer = CohereDecoderLayer(
            args=model_args,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            dtype=dtype,
            state_dict=state_dict,
            layer_num=layer_idx,
            weight_cache_path=model_args.weight_cache_path(dtype),
            transformation_mats=transformation_mats,
            paged_attention_config=None,
            prefetcher=None,
        )
        tt_input = model_args.prepare_residual_tensor_prefill(hidden)
        tt_out = tt_layer(
            tt_input,
            None,
            rot_mats_global=rot_mats,
            rot_mats_local=None,
            user_id=0,
            mode=Mode.PREFILL,
            page_table=None,
        )
        hidden_full = (
            ttnn.to_torch(
                tt_out,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape
                ),
            )[:, 0:1, :, : model_args.dim]
            .reshape(1, max_seq_len, -1)  # [1, S_pad, 8192] 3-D — keep padding for the next layer
        )

        passing, pcc_message, pcc_val = _pcc(ref_out, hidden_full[:, :seq_len].float())
        pccs.append((layer_idx, pcc_val, passing))
        if pcc_val < worst[1]:
            worst = (layer_idx, pcc_val)
        logger.info(
            f"[fullmodel] layer={layer_idx:02d} out_pcc={pcc_val:.6f} in_drift_pcc={in_pcc_msg} "
            f"gate={'PASS' if passing else 'FAIL'}"
        )

        # free device buffers before building the next layer (35B total does not fit)
        for t in (tt_input, tt_out):
            try:
                ttnn.deallocate(t)
            except Exception:
                pass
        del tt_layer, tt_input, tt_out
        gc.collect()

        hidden = hidden_full.to(torch.bfloat16)

    _CHAINED["final_hidden"] = hidden  # [1, max_seq_len, 8192] bf16, padded
    _CHAINED["seq_len"] = seq_len

    vals = [v for _, v, _ in pccs]
    mean_pcc = sum(vals) / len(vals)
    min_layer, min_pcc = worst
    logger.info(
        f"[fullmodel] layers={len(pccs)} min_pcc={min_pcc:.6f} (layer {min_layer:02d}) "
        f"mean_pcc={mean_pcc:.6f} final_layer_pcc={vals[-1]:.6f}"
    )
    failures = [f"layer{li:02d}={v:.6f}" for li, v, ok in pccs if not ok]
    assert not failures, (
        f"Full-model chained PCC < 0.99 (bounty gate) on {len(failures)}/{len(pccs)} layers: "
        + ", ".join(failures)
    )


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICES.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("max_seq_len", (128,))
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_cohere_final_norm_logits_pcc(max_seq_len, mesh_device, reset_seeds, ensure_gc):
    """Final norm (build_cohere_final_norm) + CohereLMHead (logit_scale 0.0625, tied
    embeddings) on the chained residual stream from the 40-layer sweep (or the
    reference layer39_out when run standalone). Compares vs reference `final_norm`
    and `logits`. Gate: PCC >= 0.99 (bounty gate)."""
    dtype = ttnn.bfloat8_b
    ref = _load_ref_npz()
    for key in ("final_norm", "logits"):
        if key not in ref:
            pytest.skip(f"{key} not in reference dump")
    ref_norm = torch.from_numpy(ref["final_norm"]).float()
    ref_logits = torch.from_numpy(ref["logits"]).float()
    seq_len = ref_norm.shape[1]

    if "final_hidden" in _CHAINED:
        hidden = _CHAINED["final_hidden"]
        src = "chained 40-layer output"
    else:
        hidden = torch.from_numpy(ref["layer39_out"]).float()
        hidden = torch.nn.functional.pad(hidden, (0, 0, 0, max_seq_len - seq_len)).to(torch.bfloat16)
        src = "reference layer39_out (standalone)"
    logger.info(f"[cohere-head] input: {src}")

    model_args = ModelArgs(
        mesh_device, max_batch_size=1, max_seq_len=max_seq_len, cache_hf=True, use_hf_rope=False
    )
    state_dict = model_args.load_state_dict()
    tt_ccl = TT_CCL(mesh_device)

    tt_norm = build_cohere_final_norm(
        args=model_args,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        tt_ccl=tt_ccl,
    )
    tt_head = CohereLMHead(
        args=model_args,
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        dtype=dtype,
        state_dict=state_dict,
        state_dict_prefix=model_args.get_state_dict_prefix("", None),
        weight_cache_path=model_args.weight_cache_path(dtype),
        max_columns_per_device=model_args.max_columns_per_device_lm_head,
        prefetcher=None,
    )

    tt_input = ttnn.from_torch(
        hidden.unsqueeze(1).to(torch.bfloat16),  # [1, 1, S_pad, 8192]
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=list(mesh_device.shape)),
    )
    tt_normed = tt_norm(tt_input, mode="prefill")

    # final-norm PCC (norm forward all-gathers to full width — DistributedNorm contract)
    normed_torch = ttnn.to_torch(
        tt_normed,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=model_args.cluster_shape),
    )[:1, :1].reshape(1, -1, model_args.dim)[:, :seq_len]
    passing_n, pcc_msg_n, pcc_n = _pcc(ref_norm, normed_torch.float())
    logger.info(f"[cohere-head] final_norm pcc={pcc_n:.6f} gate={'PASS' if passing_n else 'FAIL'} ({pcc_msg_n})")

    # logits PCC — mirror model.py prefill EXACTLY: slice to a 32-row window first
    # (model.py: ttnn.slice(x, (0,0,get_last_token,0), (1,1,get_last_token+32, ...)) —
    # get_lm_head_input_mem_config(PREFILL) shards (tile_padded_batch_rows=32, 128)
    # per core, so a 128-row input fails shard-grid fit — observed on-box run2
    # 2026-08-28: tensor_spec.cpp:161 !shard_grid_fit_error.has_value()). Rows 0:32
    # cover all real tokens for these prompts (seq_len <= 32 by construction).
    lm_head_input_mem_cfg = model_args.get_lm_head_input_mem_config(Mode.PREFILL, None)
    tt_head_in = ttnn.slice(tt_normed, (0, 0, 0, 0), (1, 1, 32, tt_normed.shape[-1]))
    if lm_head_input_mem_cfg.is_sharded():
        tt_head_in = ttnn.interleaved_to_sharded(tt_head_in, lm_head_input_mem_cfg)
    tt_logits = tt_head(tt_head_in)
    logits_torch = ttnn.to_torch(
        tt_logits,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=model_args.cluster_shape),
    )
    logger.info(f"[cohere-head] raw logits shape from device: {tuple(logits_torch.shape)}")
    logits_torch = logits_torch.reshape(1, -1, logits_torch.shape[-1])[:, :seq_len, : model_args.vocab_size]
    # rows are the 32-row lm-head window; rows :seq_len are the real prompt tokens
    passing_l, pcc_msg_l, pcc_l = _pcc(ref_logits, logits_torch.float())
    logger.info(f"[cohere-head] logits pcc={pcc_l:.6f} gate={'PASS' if passing_l else 'FAIL'} ({pcc_msg_l})")

    assert passing_n, f"final_norm PCC < 0.99 (bounty gate): {pcc_msg_n}"
    assert passing_l, f"logits PCC < 0.99 (bounty gate): {pcc_msg_l}"
