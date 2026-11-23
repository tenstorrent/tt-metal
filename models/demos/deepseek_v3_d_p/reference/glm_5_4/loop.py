# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 loop: pass-2 attention with window carry, the pass-2 decoder layer, the schedule and a looped stack.

Built on the GLM-5.3 reference: pass-1, prelude and coda layers are ``glm_decoder_layer_reference``; a pass-2
layer attends through the *same* ``SparseMLAReference`` instance as its pass-1 layer (shared weights, A2) and
runs the GLM-5.3 FFN half. Pass 2 runs no indexer: it reads pass-1 layer L's top-k (A5).
"""

from dataclasses import dataclass, field

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


class LoopPass2MLAReference:
    """Loop pass-2 sparse MLA for one layer, chunk by chunk, carrying the bank-B window across chunks.

    ``mla_ref`` is the pass-1 layer's reference: its weights project the pass-2 queries and KVPE at the pass-1
    RoPE positions (A4); its caches are not read or written here (bank A is passed in explicitly).
    Chunks must arrive in order. Between chunks the carry holds the last ``window - 1`` pass-2 KVPE rows, all
    that the next chunk's first row can see (A3).
    """

    def __init__(self, mla_ref: SparseMLAReference, *, window: int, latent_dim: int) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._mla_ref = mla_ref
        self._window = window
        self._latent_dim = latent_dim
        self._carry: torch.Tensor | None = None
        self._next_start = 0
        self.last_kvpe: torch.Tensor | None = None

    @property
    def window_carry(self) -> torch.Tensor:
        """Pass-2 KVPE rows [n, D] of positions next_start - n .. next_start - 1, n = min(window - 1, next_start)."""
        assert self._carry is not None, "call forward() first"
        return self._carry

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        kvpe_a: torch.Tensor,
        indices_a: torch.Tensor,
        actual_start: int = 0,
    ) -> torch.Tensor:
        """Attention output [1, S, dim] for normed hidden states [1, S, dim] at positions actual_start.. .

        Args:
            kvpe_a: bank A, the pass-1 KVPE cache of this layer [Ta, D] (or [1, 1, Ta, D]).
            indices_a: pass-1 top-k of this layer in device form [1, 1, S, k] (see ``ops.pass1_indices``).
        """
        if actual_start != self._next_start:
            raise ValueError(f"chunk starts at {actual_start}, expected {self._next_start} (chunks in order)")
        seq_len = hidden_states.shape[-2]
        q, kvpe_chunk = self._mla_ref.absorbed_query_and_kvpe(hidden_states, actual_start=actual_start)
        carry = kvpe_chunk[:0] if self._carry is None else self._carry
        kvpe_b = torch.cat([carry, kvpe_chunk], dim=0)
        indices_b = window_indices(actual_start, seq_len, self._window, kv_start=actual_start - carry.shape[0])
        kvpe_a = kvpe_a.reshape(-1, kvpe_a.shape[-1])
        latent = two_bank_sparse_mla(
            q, kvpe_a, indices_a, kvpe_b, indices_b, self._mla_ref.softmax_scale, latent_dim=self._latent_dim
        )
        self._carry = kvpe_b[kvpe_b.shape[0] - min(self._window - 1, kvpe_b.shape[0]) :]
        self._next_start = actual_start + seq_len
        self.last_kvpe = kvpe_chunk
        return self._mla_ref.latent_attention_output(latent, hidden_states)


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
    ffn_weights: dict | None = None,
    moe_weights: dict | None = None,
) -> torch.Tensor:
    """One loop pass-2 decoder layer: the GLM-5.3 layer with two-bank attention in place of indexer sparse MLA."""
    x = hidden_states
    attn_out = attention.forward(
        rms_norm(x, attn_norm_weight, config.rms_norm_eps),
        kvpe_a=kvpe_a,
        indices_a=indices_a,
        actual_start=actual_start,
    )
    return glm_ffn_half_reference(
        config, x + attn_out, ffn_norm_weight, ffn_weights=ffn_weights, moe_weights=moe_weights
    )


@dataclass(frozen=True)
class LoopStep:
    """One step of the GLM-5.4 schedule: a decoder layer, or the damp gate closing loop pass ``loop_pass``.

    ``loop_pass`` is 0 for prelude and coda layers and 1..loop_count inside the loop.
    """

    kind: str  # "layer" | "gate"
    loop_pass: int
    layer: int | None = None


def loop_schedule(num_layers: int, loop_first_layer: int, loop_last_layer: int, loop_count: int) -> list[LoopStep]:
    """Execution order: prelude, then loop_count x (loop layers, gate), then coda."""
    if not 0 <= loop_first_layer <= loop_last_layer < num_layers or loop_count < 1:
        raise ValueError(f"bad loop {loop_first_layer}..{loop_last_layer} x{loop_count} in {num_layers} layers")
    steps = [LoopStep("layer", 0, i) for i in range(loop_first_layer)]
    for k in range(1, loop_count + 1):
        steps += [LoopStep("layer", k, i) for i in range(loop_first_layer, loop_last_layer + 1)]
        steps.append(LoopStep("gate", k))
    steps += [LoopStep("layer", 0, i) for i in range(loop_last_layer + 1, num_layers)]
    return steps


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
                kvpe_a = mla[i].kvpe_cache
                x = glm54_pass2_layer_reference(
                    config,
                    pass2[i],
                    w["attn_norm_weight"],
                    w["ffn_norm_weight"],
                    x,
                    kvpe_a=kvpe_a,
                    indices_a=pass1_topk[i],
                    actual_start=start,
                    **_ffn_kwargs(w),
                )
                keep(f"L{i}.pass2.kvpe", pass2[i].last_kvpe)
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
