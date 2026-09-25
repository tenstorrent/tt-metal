# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""V4-Flash golden KV trace ("v4_groups_v1") from the torch reference, one decoder layer at a time.

Layout (``<out_dir>/``):
  metadata.json           {"token_ids": [...], "layout": "v4_groups_v1", "seq_len": S, "layer_kinds": [...],
                           "rope_frame": "device"}
  kv_cache/layer_N.safetensors
      window_layer_N      [128, 512]   the last 128 tokens' K (== V) rows in RING order (row = token % 128)
      compressed_layer_N  [S // rate, 512]   HCA (rate 128) / CSA (rate 4) compressed entries       (not for SWA)
      index_k_layer_N     [S // 4, 128]      the lightning indexer's keys                            (CSA only)

Every tensor is what the prefill worker writes into the contract's unified caches (``tt/v4/kv_contract.py``):
window rows compare with unified rows [0, 128), entries with rows 128.., keys with the index cache -- same rope
frame as the device (the reference's interleaved ``apply_rotary_pos_emb`` == ``rotary_embedding_llama`` with the
transformation matrix; measured PCC 0.9999 on the M4 export test, no re-interleave).

The model is streamed: one ``DeepseekV4DecoderLayer`` at a time (weights via ``layer_weights(idx)`` in the
reference names -- the checkpoint loader ``tt/v4/weights/hf_names.py`` or a random init), the 4 residual streams
carried between layers, the caches tapped through the reference modules themselves (attention-input hook -> K
rows; ``compressor(...)`` -> entries; the indexer keys through ``csa_math.compress_chunk``, validated against the
reference at 1e-4). A whole prompt is one single-shot pass per layer (chunked and unchunked agree by construction
of the reference's caches). Host time is the reference's MoE + eager attention: INFERRED 2-3 min per layer for 5120
tokens on host 29; MEASURE and log the first real run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4RotaryEmbedding,
    apply_rotary_pos_emb,
)
from models.demos.deepseek_v3_d_p.tt.v4.attention import csa_math as C
from models.demos.deepseek_v3_d_p.tt.v4.layer_kinds import CSA, SLIDING, layer_kinds

LAYOUT = "v4_groups_v1"
WINDOW = 128


def load_reference_layer(cfg, layer_idx: int, w: dict) -> DeepseekV4DecoderLayer:
    """A reference decoder layer from the reference-named dict (``"__experts__"`` = per-expert HF-orientation
    dicts, packed back into the fused ``experts.gate_up_proj / down_proj``)."""
    layer = DeepseekV4DecoderLayer(cfg, layer_idx).eval()
    sd = {k: v for k, v in w.items() if not k.startswith("__")}
    experts = w.get("__experts__")
    if experts is not None:
        gu = torch.stack([torch.cat([e["gate_proj"], e["up_proj"]], 0) for e in experts], 0)
        dn = torch.stack([e["down_proj"] for e in experts], 0)
        sd["mlp.experts.gate_up_proj"] = gu
        sd["mlp.experts.down_proj"] = dn
    missing, unexpected = layer.load_state_dict({k: v.to(torch.float32) for k, v in sd.items()}, strict=False)
    missing = [m for m in missing if m != "mlp.gate.tid2eid" and m != "mlp.gate.e_score_correction_bias"]
    assert not missing and not unexpected, (missing, unexpected)
    return layer


def ring_rows(k_rows: torch.Tensor, total: int, sw: int = WINDOW) -> torch.Tensor:
    ring = torch.zeros(sw, k_rows.shape[-1])
    for t in range(max(0, total - sw), total):
        ring[t % sw] = k_rows[t]
    return ring


def layer_kv_golden(
    layer: DeepseekV4DecoderLayer, cfg, rot: DeepseekV4RotaryEmbedding, streams: torch.Tensor, input_ids: torch.Tensor
):
    """Run one layer on the streams ``[1, S, 4, D]`` (fp32) and return ``(new_streams, {tensors})``."""
    S = streams.shape[1]
    pos = torch.arange(S).unsqueeze(0)
    kind = layer_kinds(cfg)[layer.layer_idx]
    rope_type = "main" if kind == SLIDING else "compress"
    captured = {}

    def hook(mod, args, kwargs):
        captured["hidden"] = args[0] if args else kwargs["hidden_states"]

    h = layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True)
    i, j = torch.arange(S).view(S, 1), torch.arange(S).view(1, S)
    mask = torch.zeros(S, S).masked_fill(~((j <= i) & (i - j < cfg.sliding_window)), float("-inf")).view(1, 1, S, S)
    with torch.no_grad():
        pe = {
            "main": rot(streams, position_ids=pos, layer_type="main"),
            "compress": rot(streams, position_ids=pos, layer_type="compress"),
        }
        out = layer(
            streams,
            position_embeddings=pe,
            position_ids=pos,
            attention_mask=mask,
            input_ids=input_ids,
            past_key_values=None,
        )
    h.remove()
    attn = layer.self_attn
    hidden = captured["hidden"]  # [1, S, D] = input_layernorm(collapsed)
    with torch.no_grad():
        kv = attn.kv_norm(attn.kv_proj(hidden)).view(1, S, 1, -1).transpose(1, 2)
        cos, sin = pe[rope_type]
        k_rows = apply_rotary_pos_emb(kv, cos, sin)[0, 0]  # [S, 512]
        tensors = {f"window_layer_{layer.layer_idx}": ring_rows(k_rows, S).contiguous()}
        if kind != SLIDING:
            q_res = attn.q_a_norm(attn.q_a_proj(hidden))
            entries, _ = attn.compressor(hidden, q_res, pos, None, layer.layer_idx)
            tensors[f"compressed_layer_{layer.layer_idx}"] = entries[0, 0].contiguous()
        if kind == CSA:
            idx = attn.compressor.indexer
            keys, _ = C.compress_chunk(
                idx.kv_proj(hidden)[0],
                idx.gate_proj(hidden)[0],
                idx.position_bias,
                idx.kv_norm.weight,
                idx.kv_norm.variance_epsilon,
                idx.rotary_emb,
                C.empty_prior(idx.head_dim),
                0,
            )
            tensors[f"index_k_layer_{layer.layer_idx}"] = keys.contiguous()
    return out, tensors


def generate_golden(
    cfg,
    *,
    layer_weights: Callable[[int], dict],
    embed_weight: torch.Tensor,
    token_ids: Iterable[int],
    out_dir: str | Path,
    layers: Optional[Iterable[int]] = None,
    reference_layers: Optional[Callable[[int], DeepseekV4DecoderLayer]] = None,
) -> Path:
    """Write the golden for ``layers`` (default: all). ``reference_layers(idx)`` may hand over ready-made layers
    (random-weight tests); otherwise ``layer_weights(idx)`` feeds ``load_reference_layer``."""
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    (out_dir / "kv_cache").mkdir(parents=True, exist_ok=True)
    ids = torch.tensor(list(token_ids), dtype=torch.long).view(1, -1)
    S = ids.shape[1]
    kinds = layer_kinds(cfg)
    layer_list = list(range(len(kinds)) if layers is None else layers)
    rot = DeepseekV4RotaryEmbedding(cfg)
    with torch.no_grad():
        emb = torch.nn.functional.embedding(ids, embed_weight.float())  # [1, S, D]
        streams = emb.unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()
    written = []
    for li in layer_list:
        layer = (
            reference_layers(li) if reference_layers is not None else load_reference_layer(cfg, li, layer_weights(li))
        )
        streams, tensors = layer_kv_golden(layer, cfg, rot, streams, ids)
        save_file(
            {k: v.to(torch.float32).contiguous() for k, v in tensors.items()},
            str(out_dir / "kv_cache" / f"layer_{li}.safetensors"),
        )
        written.append(li)
        logger.info(f"[v4 golden] layer {li} ({kinds[li]}): {sorted(tensors)}")
        del layer
    meta = {
        "token_ids": ids[0].tolist(),
        "layout": LAYOUT,
        "seq_len": S,
        "layer_kinds": kinds,
        "rope_frame": "device",
        "layers": written,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta))
    return out_dir


def load_golden(trace_dir: str | Path, layer_idx: int) -> dict:
    from safetensors.torch import load_file

    return load_file(str(Path(trace_dir) / "kv_cache" / f"layer_{layer_idx}.safetensors"))


def trace_layout(trace_dir: str | Path) -> Optional[str]:
    md = json.loads((Path(trace_dir) / "metadata.json").read_text())
    return md.get("layout")
