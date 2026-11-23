# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 loop: pass-2 attention with window carry, the pass-2 decoder layer, the schedule and a looped stack.

Built on the GLM-5.3 reference: pass-1, prelude and coda layers are ``glm_decoder_layer_reference``; a pass-2
layer attends through the *same* ``SparseMLAReference`` instance as its pass-1 layer (shared weights, A2) and
runs the GLM-5.3 FFN half. Pass 2 runs no indexer: it reads pass-1 layer L's top-k (A5).
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import torch

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import SparseMLAReference, random_mla_weights
from models.demos.deepseek_v3_d_p.reference.glm_5_3.block import (
    glm_decoder_layer_reference,
    glm_ffn_half_reference,
    rms_norm,
)
from models.demos.deepseek_v3_d_p.reference.glm_5_4.ops import (
    damp_gate_reference,
    pass1_indices,
    two_bank_sparse_mla,
    window_indices,
)
from models.demos.deepseek_v3_d_p.reference.glm_5_4.schedule import LoopStep, loop_schedule  # noqa: F401 (re-export)


class Pass2Result(NamedTuple):
    """One pass-2 chunk: the output, the window carry for the next chunk, and this chunk's pass-2 KVPE rows."""

    output: torch.Tensor  # [1, S, dim]
    window_carry: torch.Tensor  # [n, D]: positions end-n .. end-1, n = min(window - 1, end)
    kvpe: torch.Tensor  # [S, D]: pass-2 KVPE of positions actual_start .. end-1


class LoopPass2MLAReference:
    """Loop pass-2 sparse MLA for one layer; the bank-B window carry is passed in and returned explicitly.

    ``mla_ref`` is the pass-1 layer's reference: its weights project the pass-2 queries and KVPE at the pass-1
    RoPE positions (A4); its caches are not read or written here (bank A is passed in explicitly). Between
    chunks the carry holds the last ``window - 1`` pass-2 KVPE rows, all that the next chunk's first row can see
    (A3). Holding no state, it can check any chunk given the device's carry (teacher forcing).
    """

    def __init__(self, mla_ref: SparseMLAReference, *, window: int, latent_dim: int) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._mla_ref = mla_ref
        self._window = window
        self._latent_dim = latent_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        kvpe_a: torch.Tensor,
        indices_a: torch.Tensor,
        actual_start: int = 0,
        window_carry: torch.Tensor | None = None,
    ) -> Pass2Result:
        """Pass-2 attention for normed hidden states [1, S, dim] at positions actual_start .. actual_start+S-1.

        Args:
            kvpe_a: bank A, the pass-1 KVPE cache of this layer [Ta, D] (or [1, 1, Ta, D]).
            indices_a: pass-1 top-k of this layer in device form [1, 1, S, k] (see ``ops.pass1_indices``).
            window_carry: pass-2 KVPE rows [n, D] of positions actual_start-n .. actual_start-1 with
                n = min(window - 1, actual_start), i.e. the previous chunk's ``Pass2Result.window_carry``.
                None means no earlier rows (only valid at actual_start 0 or window 1).
        """
        seq_len = hidden_states.shape[-2]
        q, kvpe_chunk = self._mla_ref.absorbed_query_and_kvpe(hidden_states, actual_start=actual_start)
        carry = kvpe_chunk[:0] if window_carry is None else window_carry.reshape(-1, kvpe_chunk.shape[-1])
        need = min(self._window - 1, actual_start)
        if carry.shape[0] != need:
            raise ValueError(
                f"window carry has {carry.shape[0]} rows; a chunk at {actual_start} with window {self._window} "
                f"needs positions {actual_start - need}..{actual_start - 1} ({need} rows)"
            )
        kvpe_b = torch.cat([carry.to(kvpe_chunk.dtype), kvpe_chunk], dim=0)
        indices_b = window_indices(actual_start, seq_len, self._window, kv_start=actual_start - need)
        kvpe_a = kvpe_a.reshape(-1, kvpe_a.shape[-1])
        latent = two_bank_sparse_mla(
            q, kvpe_a, indices_a, kvpe_b, indices_b, self._mla_ref.softmax_scale, latent_dim=self._latent_dim
        )
        new_carry = kvpe_b[kvpe_b.shape[0] - min(self._window - 1, kvpe_b.shape[0]) :]
        return Pass2Result(self._mla_ref.latent_attention_output(latent, hidden_states), new_carry, kvpe_chunk)


@torch.no_grad()
def glm54_pass2_layer_reference(
    config,
    attention: LoopPass2MLAReference,
    attn_norm_weight: torch.Tensor,
    ffn_norm_weight: torch.Tensor,
    hidden_states: torch.Tensor,
    *,
    kvpe_a: torch.Tensor,
    indices_a: torch.Tensor,
    actual_start: int = 0,
    window_carry: torch.Tensor | None = None,
    ffn_weights: dict | None = None,
    moe_weights: dict | None = None,
) -> Pass2Result:
    """One loop pass-2 decoder layer: the GLM-5.3 layer with two-bank attention in place of indexer sparse MLA.

    Returns the layer output with the attention's window carry and pass-2 KVPE (see ``LoopPass2MLAReference``).
    """
    x = hidden_states
    attn = attention.forward(
        rms_norm(x, attn_norm_weight, config.rms_norm_eps),
        kvpe_a=kvpe_a,
        indices_a=indices_a,
        actual_start=actual_start,
        window_carry=window_carry,
    )
    out = glm_ffn_half_reference(
        config, x + attn.output, ffn_norm_weight, ffn_weights=ffn_weights, moe_weights=moe_weights
    )
    return attn._replace(output=out)


def _ffn_kwargs(layer: dict) -> dict:
    if "ffn_weights" in layer:
        return {"ffn_weights": layer["ffn_weights"]}
    return {"moe_weights": {k: layer[k] for k in ("gate_weights", "routed_expert_weights", "shared_expert_weights")}}


def _mla_weights_with_indexer(config, mla_weights: dict) -> dict:
    """A ``shared`` layer's state dict carries no indexer weights; the CPU MLA still builds an indexer, whose
    selection is then overridden by the reused top-k. Fill the absent keys with unused placeholders."""
    if any(k.startswith("indexer.") for k in mla_weights):
        return mla_weights
    placeholder = random_mla_weights(config, seed=0)
    return {**mla_weights, **{k: v for k, v in placeholder.items() if k.startswith("indexer.")}}


@dataclass
class Glm54StackResult:
    """Stack output and the intermediates a device test needs to localize a wiring fault, each [1, S, ...]
    over the whole sequence (chunks concatenated).

    ``checkpoints`` keys: ``pass{k}.in`` (x_{k-1}), ``pass{k}.out`` (h_k), ``gate{k}.beta``, ``gate{k}.out`` (x_k);
    per loop layer L: ``L{L}.pass1.topk`` (device form [1, 1, S, k]), ``L{L}.pass1.kvpe``, ``L{L}.pass2.kvpe``.
    """

    output: torch.Tensor
    checkpoints: dict[str, torch.Tensor] = field(default_factory=dict)


@torch.no_grad()
def glm54_stack_reference(
    config,
    layer_weights: list[dict],
    gate_weights: list[dict],
    hidden_states: torch.Tensor,
    *,
    chunk_size: int | None = None,
) -> Glm54StackResult:
    """GLM-5.4 decoder stack on CPU: prelude, loop pass 1 + gate 1, loop pass 2 + gate 2, coda.

    Args:
        config: GLM-5.4 hf config (``glm_5_4_hf_config`` or a reduced copy) with ``num_hidden_layers``,
            ``indexer_types``, ``loop_first_layer``, ``loop_last_layer``, ``loop_count`` (must be 2) and
            ``loop_window_size``.
        layer_weights: one TtPrefillBlock-form state dict per layer (``attn_norm_weight``, ``mla_weights``,
            ``ffn_norm_weight`` and ``ffn_weights`` or ``gate_weights``/``routed_expert_weights``/
            ``shared_expert_weights``). Loop layers have one weight set, used by both passes.
        gate_weights: per loop pass, ``{"norm_weight", "proj_weight", "proj_bias"}`` (A1b).
        hidden_states: stack input [1, S, hidden].
        chunk_size: run chunked prefill with this many tokens per chunk; None = one chunk.
    """
    num_layers = config.num_hidden_layers
    first, last, count = config.loop_first_layer, config.loop_last_layer, config.loop_count
    types = config.indexer_types
    if count != 2:
        raise ValueError(f"the GLM-5.4 reference defines loop passes 1 and 2 only, got loop_count={count}")
    if len(layer_weights) != num_layers or len(gate_weights) != count or len(types) < num_layers:
        raise ValueError(f"need {num_layers} layers, {count} gates and indexer_types for each layer")
    if last + 1 < num_layers and types[last + 1] != "full":
        # Pass 2 runs no indexer, so a shared first coda layer has no top-k of its own pass to reuse.
        raise ValueError(f"first coda layer {last + 1} must own an indexer, got {types[last + 1]!r}")
    if types[0] != "full":
        raise ValueError("layer 0 must own an indexer")

    seq_len = hidden_states.shape[-2]
    chunk_size = seq_len if chunk_size is None else chunk_size
    mla = {
        i: SparseMLAReference(config, _mla_weights_with_indexer(config, w["mla_weights"]), seq_len=seq_len)
        for i, w in enumerate(layer_weights)
    }
    pass2 = {
        i: LoopPass2MLAReference(mla[i], window=config.loop_window_size, latent_dim=config.kv_lora_rank)
        for i in range(first, last + 1)
    }
    carries: dict[int, torch.Tensor | None] = {i: None for i in pass2}
    schedule = loop_schedule(num_layers, first, last, count)
    pieces: dict[str, list[torch.Tensor]] = {}

    def keep(name: str, t: torch.Tensor) -> None:
        pieces.setdefault(name, []).append(t)

    outputs = []
    for start in range(0, seq_len, chunk_size):
        x = hidden_states[:, start : start + chunk_size]
        end = start + x.shape[-2]
        pass_in = None
        full_topk = None  # top-k of the most recent full layer in the current pass-1 (or prelude/coda) sweep
        pass1_topk = {}
        for step in schedule:
            if step.kind == "gate":
                g = gate_weights[step.loop_pass - 1]
                x_gated, beta = damp_gate_reference(
                    pass_in,
                    x,
                    norm_weight=g["norm_weight"],
                    proj_weight=g["proj_weight"],
                    proj_bias=g["proj_bias"],
                    eps=config.rms_norm_eps,
                )
                keep(f"pass{step.loop_pass}.out", x)
                keep(f"gate{step.loop_pass}.beta", beta)
                x = x_gated.to(x.dtype)
                keep(f"gate{step.loop_pass}.out", x)
                continue
            i, w = step.layer, layer_weights[step.layer]
            if step.layer == first:
                pass_in = x
                keep(f"pass{step.loop_pass}.in", x)
            if step.loop_pass == 2:
                x, carries[i], kvpe2 = glm54_pass2_layer_reference(
                    config,
                    pass2[i],
                    w["attn_norm_weight"],
                    w["ffn_norm_weight"],
                    x,
                    kvpe_a=mla[i].kvpe_cache,
                    indices_a=pass1_topk[i],
                    actual_start=start,
                    window_carry=carries[i],
                    **_ffn_kwargs(w),
                )
                keep(f"L{i}.pass2.kvpe", kvpe2)
                continue
            shared = types[i] == "shared"
            x, kvpe, topk = glm_decoder_layer_reference(
                config,
                None,
                w["attn_norm_weight"],
                w["ffn_norm_weight"],
                x,
                seq_len,
                indexer_topk=full_topk if shared else None,
                return_indexer_topk=True,
                mla_ref=mla[i],
                actual_start=start,
                **_ffn_kwargs(w),
            )
            full_topk = topk
            if step.loop_pass == 1:
                pass1_topk[i] = pass1_indices(topk, start, config.index_topk)
                keep(f"L{i}.pass1.topk", pass1_topk[i])
                keep(f"L{i}.pass1.kvpe", kvpe[0, 0, start:end])
        outputs.append(x)

    checkpoints = {name: torch.cat(ts, dim=-2) for name, ts in pieces.items()}
    return Glm54StackResult(output=torch.cat(outputs, dim=-2), checkpoints=checkpoints)
