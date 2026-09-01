# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TEMP(#53533): REMOVE BEFORE PR -- runner-side MTP PCC against the composed CPU reference.

The runner does not measure PCC; that is the producer's job, against a golden trace. But the
GLM-5.2 golden trace stops at layer 77, so it carries NO truth for the four MTP KV slots -- the
e2e's only MTP check today is ``max|.|`` per slot, which proves the slots were written and nothing
about whether they were written *correctly*. This module is the stand-in until a trace with MTP
layers exists.

Why it is not redundant with ``tests/mtp_prefill/test_mtp.py::test_mtp_predictor_pcc``: that test
runs on RANDOM weights, deliberately. ``test_mtp_module_pcc``'s docstring records the measured
reason -- a trained GLM gate driven by a synthetic input picks different top-8 experts on device
than on CPU and isolated-block PCC collapses to ~0.1, while the same layer scores ~0.995 in
context. Here the input is the real trunk's h^0 on real tokens, so the gate is trained AND the
input is in distribution. This is the one place the real-weight MTP numbers can be measured.

Two constraints shape what is measured:

1. **Teacher forcing is mandatory.** The trunk's own KV sits at ~0.86 against the golden trace
   (a pre-existing, non-MTP floor). A reference chained from its own h^0 would inherit that and
   report an MTP PCC <= 0.86 by construction, proving nothing. So the reference is fed the
   DEVICE's own per-level hidden states via ``hiddens=``: each level is then measured in
   isolation and the number is about MTP's math, not the trunk's.

2. **Chunk 0 only.** A full 56320-token CPU reference is 11 chunks x 4 levels of a 256-expert MoE
   plus sparse MLA over a growing cache -- not tractable in the runner's process. Chunk 0 is
   self-contained (no attention prefix), and rows [0, chunk) of the MTP KV slots are written by
   chunk 0 and never rewritten, so reading them at the end is exact.

Removal is: delete this file, and delete the four lines in ``prefill_runner.py`` marked
``TEMP(#53533)``. Nothing in the model, runtime, or transport is touched -- the capture is a
monkeypatch installed from here.

Off unless ``PREFILL_MTP_REF_PCC=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from loguru import logger

import ttnn

ENV_FLAG = "PREFILL_MTP_REF_PCC"

# Captured during chunk 0 by the monkeypatch in install(). Host torch, already gathered.
_CAP: dict = {}


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0") == "1"


# ---------------------------------------------------------------------------
# Capture (chunk 0 only), by monkeypatch -- no model/runtime code is touched
# ---------------------------------------------------------------------------


def _act_composer(mesh_device, sp_axis: int, tp_axis: int):
    """Activation composer: SP concatenated over rows, TP over columns -- natural token order."""
    dims = [None, None]
    dims[sp_axis] = -2
    dims[tp_axis] = -1
    return ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=mesh_device.shape)


def _gather(t: ttnn.Tensor, composer) -> torch.Tensor:
    """[1, 1, seq, hidden] host bf16."""
    return ttnn.to_torch(t, mesh_composer=composer).to(torch.bfloat16)


class _EmbedSourceTap:
    """Delegates to the real MTPDeviceEmbedSource and records each level's embedding host-side.

    It records what the DEVICE consumed -- post position-0 masking -- so the reference is fed the
    identical tensor and a masking difference cannot hide inside the PCC.
    """

    def __init__(self, inner, composer, sink: list):
        self._inner, self._composer, self._sink = inner, composer, sink

    def __call__(self, k: int, prev_normed):
        embed = self._inner(k, prev_normed)
        self._sink.append(_gather(embed, self._composer))
        return embed

    def __getattr__(self, name):  # generated_tokens, real_len, ...
        return getattr(self._inner, name)


def install(runtime) -> None:
    """Patch the runtime so chunk 0's MTP inputs and per-level hidden states are captured."""
    predictor = getattr(getattr(runtime, "model", None), "mtp_predictor", None)
    if predictor is None:
        logger.warning(f"[{ENV_FLAG}] no mtp_predictor on this rank; capture not installed")
        return

    mesh_device = runtime.mesh_device
    composer = _act_composer(mesh_device, runtime.config.sp_axis, runtime.config.tp_axis)

    orig_chunk, orig_forward = runtime.prefill_chunk, predictor.forward

    def chunk_tap(input_tensor, kv_caches, slot_id, actual_start, actual_end, *a, **kw):
        if "chunk" not in _CAP:
            # Recorded before the call: the predictor tap below runs inside it.
            _CAP["chunk"] = {"slot_id": int(slot_id), "start": int(actual_start), "end": int(actual_end)}
        return orig_chunk(input_tensor, kv_caches, slot_id, actual_start, actual_end, *a, **kw)

    def forward_tap(source, hidden, *a, **kw):
        first = "h0" not in _CAP
        if first:
            _CAP["h0"] = _gather(hidden, composer)
            _CAP["embeds"] = []
            source = _EmbedSourceTap(source, composer, _CAP["embeds"])
        out = orig_forward(source, hidden, *a, **kw)
        if first:
            _CAP["normed"] = [_gather(t, composer) for t in out.out_head_normed]
            logger.info(
                f"[{ENV_FLAG}] captured chunk 0: h0={tuple(_CAP['h0'].shape)} "
                f"embeds={len(_CAP['embeds'])} normed={len(_CAP['normed'])}"
            )
        return out

    runtime.prefill_chunk = chunk_tap
    predictor.forward = forward_tap
    logger.info(f"[{ENV_FLAG}] capture installed (chunk 0 only); PCC report runs after the request loop")


# ---------------------------------------------------------------------------
# Layer-78 host weights (same recipe as tests/test_prefill_block.py::_glm_pretrained_weights,
# duplicated here so this file is deletable on its own)
# ---------------------------------------------------------------------------


def _load_layer_weights(config, model_dir: str, layer_idx: int, num_routed_experts: int):
    from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.reference import pretrained_mla_weights
    from models.demos.deepseek_v3_d_p.reference.tt.moe.moe import load_moe_weights_from_hf
    from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

    prefix = f"model.layers.{layer_idx}."
    weight_map = json.load(open(Path(model_dir) / "model.safetensors.index.json"))["weight_map"]
    shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
    mla_weights = pretrained_mla_weights(
        config, layer=layer_idx, checkpoint_path=[str(Path(model_dir) / s) for s in shards]
    )
    norms = load_hf_state_dict_filtered(model_dir, [f"{prefix}input_layernorm.", f"{prefix}post_attention_layernorm."])
    attn_norm_w = norms[f"{prefix}input_layernorm.weight"].to(torch.bfloat16)
    ffn_norm_w = norms[f"{prefix}post_attention_layernorm.weight"].to(torch.bfloat16)
    routed, shared = load_moe_weights_from_hf(model_dir, layer_idx, num_routed_experts)
    g = load_hf_state_dict_filtered(model_dir, [f"{prefix}mlp.gate."])
    moe_weights = {
        "gate_weights": {
            "weight": g[f"{prefix}mlp.gate.weight"].to(torch.bfloat16),
            "e_score_correction_bias": g[f"{prefix}mlp.gate.e_score_correction_bias"].float(),
        },
        "routed_expert_weights": routed,
        "shared_expert_weights": shared,
    }
    return mla_weights, attn_norm_w, ffn_norm_w, moe_weights


# ---------------------------------------------------------------------------
# Device KV readback
# ---------------------------------------------------------------------------


def _read_mtp_kv(runtime, kv_caches, *, slot: int, num_levels: int, first_slot: int, chunk_rows: int):
    """The K MTP KV slots for `slot`, chunk 0's rows only: [K, 1, chunk_rows, kv_lora + qk_rope].

    Sliced on device first (the whole 82-slot x 56320 cache would be tens of GB on host), then
    composed with ttMLA.kv_cache_to_host's composer so SP shards land in natural token order.
    DRAM_MEMORY_CONFIG on the slice is required -- the cache is ND-sharded ROUND_ROBIN_1D.
    """
    mesh_device = runtime.mesh_device
    sp_axis = runtime.config.sp_axis
    kvpe = kv_caches.kvpe
    depth = getattr(runtime.model, "num_kvpe_cache_layers", None)
    assert depth is not None, "model has no num_kvpe_cache_layers; cannot locate the MTP slots"
    base = slot * depth + first_slot

    s = list(kvpe.storage.shape)
    sl = ttnn.slice(
        kvpe.storage,
        [base, 0, 0, 0],
        [base + num_levels, s[1], s[2], s[3]],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    host = ttnn.to_torch(
        sl,
        mesh_composer=ttnn.create_mesh_composer(
            mesh_device,
            config=ttnn.MeshComposerConfig(
                dims=(2, -1),
                mesh_shape_override=ttnn.MeshShape(mesh_device.shape[sp_axis], 1),
            ),
        ),
    )
    ttnn.deallocate(sl)
    host = kvpe.unpack_host(host)  # [K, 1, sp * seq_local, kvpe]

    sp = mesh_device.shape[sp_axis]
    seq_local = host.shape[2] // sp
    rows_per_chip = chunk_rows // sp
    # Chip c holds this chunk's tokens [c*rows_per_chip, (c+1)*rows_per_chip) in its FIRST rows
    # (chunk 0 is the first write); concatenating chip-major is therefore natural token order.
    return torch.cat([host[:, :, c * seq_local : c * seq_local + rows_per_chip, :] for c in range(sp)], dim=2)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def report(runtime, kv_caches, hf_config, model_dir: str) -> None:
    """Compose the CPU reference for chunk 0 and log per-level MTP PCC. Never raises."""
    try:
        _report(runtime, kv_caches, hf_config, model_dir)
    except Exception as exc:  # bring-up scaffolding must not take the runner down
        logger.exception(f"[{ENV_FLAG}] MTP reference PCC failed: {exc}")


def _report(runtime, kv_caches, hf_config, model_dir: str) -> None:
    from models.demos.deepseek_v3_d_p.reference.glm_5_2.mtp import glm_mtp_predictor_reference
    from models.demos.deepseek_v3_d_p.tt.mtp_prefill.utils import load_mtp_state_dict
    from tests.ttnn.utils_for_testing import comp_pcc

    if "h0" not in _CAP:
        logger.warning(f"[{ENV_FLAG}] nothing captured (no MTP chunk ran on this rank); skipping")
        return
    predictor = runtime.model.mtp_predictor
    K = predictor.num_levels

    h0 = _CAP["h0"].squeeze(0)  # [1, seq, hidden]
    embeds = [e.squeeze(0) for e in _CAP["embeds"]]
    normed = [t.squeeze(0) for t in _CAP["normed"]]
    assert len(embeds) == K and len(normed) == K, f"captured {len(embeds)} embeds / {len(normed)} hiddens for K={K}"
    seq_len = int(h0.shape[1])
    chunk = _CAP["chunk"]
    logger.info(
        f"[{ENV_FLAG}] chunk 0: slot={chunk['slot_id']} positions=[{chunk['start']}, {chunk['end']}) "
        f"seq_len={seq_len} K={K} index_share={predictor.index_share} chain_from={predictor.chain_from}"
    )

    # Device work FIRST, and nothing device-side after it. The e2e harness SIGINTs the runner once the
    # producer exits and SIGKILLs 120 s later; a SIGKILL landing inside a device call wedges an eth core
    # (needs tt-smi -glx_reset_auto). Everything below this readback is pure host torch, so a kill there
    # is harmless. The optional dump makes the comparison re-runnable offline if the process does die.
    dev_kv = _read_mtp_kv(
        runtime,
        kv_caches,
        slot=chunk["slot_id"],
        num_levels=K,
        first_slot=predictor.first_cache_slot,
        chunk_rows=seq_len,
    )
    logger.info(f"[{ENV_FLAG}] device MTP KV read back: {tuple(dev_kv.shape)}")
    dump = os.environ.get("PREFILL_MTP_REF_PCC_DUMP")
    if dump:
        torch.save({"h0": h0, "embeds": embeds, "normed": normed, "dev_kv": dev_kv, "chunk": chunk}, dump)
        logger.info(f"[{ENV_FLAG}] capture + device KV saved to {dump}")

    layer_idx = predictor.mtp_config.mtp_layer_idx
    logger.info(f"[{ENV_FLAG}] loading layer-{layer_idx} host weights from {model_dir} (~10 s measured)")
    mla_weights, attn_norm_w, ffn_norm_w, moe_weights = _load_layer_weights(
        hf_config, model_dir, layer_idx, hf_config.n_routed_experts
    )
    mtp_state_dict = load_mtp_state_dict(model_dir, layer_idx=layer_idx)

    # Teacher forcing: hiddens[k] is level k+1's H^k, so hiddens = [device h^0] + device H^1..H^(K-1).
    # Without this the trunk's ~0.86 KV floor would cap every MTP level by construction.
    hiddens = [h0] + normed[: K - 1]

    logger.info(f"[{ENV_FLAG}] composing CPU reference: {K} levels x one 256-expert MoE block (slow)")
    _, _, _, ref_kv = glm_mtp_predictor_reference(
        hf_config,
        mla_weights,
        mtp_state_dict,
        attn_norm_w,
        ffn_norm_w,
        embeds,
        h0,
        seq_len,
        moe_weights=moe_weights,
        num_levels=K,
        index_share=predictor.index_share,
        chain_from=predictor.chain_from,
        hiddens=hiddens,
        positions=torch.arange(chunk["start"], chunk["start"] + seq_len),
        actual_start=chunk["start"],
        actual_end=chunk["end"],
    )

    assert dev_kv.shape == ref_kv.shape, f"device KV {tuple(dev_kv.shape)} vs reference {tuple(ref_kv.shape)}"

    # Split latent from RoPE, as tests/test_prefill_block.py does: they fail in different ways and a
    # merged PCC lets a healthy latent hide a broken k_pe.
    kv_lora = hf_config.kv_lora_rank
    lines = []
    for k in range(K):
        ref_slot, dev_slot = ref_kv[k : k + 1], dev_kv[k : k + 1]
        _, kv_pcc = comp_pcc(ref_slot[..., :kv_lora].float(), dev_slot[..., :kv_lora].float())
        _, pe_pcc = comp_pcc(ref_slot[..., kv_lora:].float(), dev_slot[..., kv_lora:].float())
        lines.append(f"  L{k + 1} (KV slot {predictor.first_cache_slot + k}): kv={kv_pcc:.6f} pe={pe_pcc:.6f}")
    logger.info(f"[{ENV_FLAG}] MTP KV vs composed CPU reference, chunk 0, teacher-forced:\n" + "\n".join(lines))
