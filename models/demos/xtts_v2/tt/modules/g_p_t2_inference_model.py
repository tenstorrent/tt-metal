# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `g_p_t2_inference_model` of coqui/XTTS-v2.

Reference submodule: `gpt.gpt_inference`, a
`TTS.tts.layers.xtts.gpt_inference.GPT2InferenceModel` — the generation-step
wrapper around the GPT2 stack used inside `GPT.generate()`.

This port covers the deterministic PREFILL step the PCC harness drives
(`past_key_values=None`, `use_cache=False`), where the module reduces to a plain
causal forward + LM head:

    prefix_emb = cached_prefix_emb                       # stored via store_prefix_emb
    gen_inputs = input_ids[:, prefix_len:]
    gen_emb    = embeddings(gen_inputs) + pos_embedding(0..gen_len)
    emb        = cat([prefix_emb, gen_emb], dim=1)
    hidden     = transformer(inputs_embeds=emb)          # causal GPT2
    logits     = mel_head(final_norm(hidden))            # lm_head = Sequential(norm, linear)

The GPT2 transformer reuses the native `g_p_t2_model` port. Embeddings,
the final LayerNorm, the LM-head matmul, and all concat/slice run in ttnn. Only
the integer `input_ids[:, prefix_len:]` slice runs on host (index arithmetic).

The prefix embedding is stateful (set by `store_prefix_emb` on the torch module
before the ttnn port is built); we snapshot it at build time.

Harness note: the forward takes only kwargs, so the PCC harness supplies a
throwaway synthetic positional `primary` arg (ignored); real inputs arrive as
host torch tensors in **kwargs.
"""

from __future__ import annotations

import ttnn
from models.demos.xtts_v2.tt.modules.g_p_t2_model import build as _build_gpt2_model
from models.demos.xtts_v2.tt.modules.learned_position_embeddings import build as _build_lpe

_LN_EPS = 1e-5


def build(device, torch_module):
    """Bind trained weights + the stored prefix and return a native ttnn forward."""
    import torch

    m = torch_module

    # GPT2 transformer stack (native, port).
    gpt2_forward = _build_gpt2_model(device, m.transformer.float())

    # Token embedding (ROW_MAJOR for ttnn.embedding).
    emb_w = ttnn.as_tensor(
        m.embeddings.weight.detach().contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Absolute position-prefix lookup via the     # learned_position_embeddings leaf module (returns float32 [sl, D]).
    lpe = _build_lpe(device, m.pos_embedding)

    # lm_head = Sequential(final_norm: LayerNorm, mel_head: Linear).
    norm = m.lm_head[0]
    linear = m.lm_head[1]
    lnf_w = ttnn.as_tensor(
        norm.weight.detach().contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    lnf_b = ttnn.as_tensor(
        norm.bias.detach().contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # nn.Linear weight is [out, in]; ttnn matmul wants [in, out].
    head_w = ttnn.as_tensor(
        linear.weight.detach().t().contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    head_b = ttnn.as_tensor(
        linear.bias.detach().contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # The LM head is a 1024->1026 projection whose bf16-accumulated matmul
    # dominates the output error (PCC-sensitive over 1026 logits). Run it with
    # HiFi4 + fp32 accumulation so it matches the float32 reference.
    _head_kernel_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    model_dim = int(m.embeddings.weight.shape[1])

    # Snapshot the stateful prefix embedding stored via store_prefix_emb(). Kept in
    # a mutable state cell so a served pipeline can refresh it per utterance via
    # `forward.set_prefix(...)` without rebuilding the transformer (build_pipeline).
    state = {}

    def _upload_prefix(prefix_torch):
        prefix_tt = ttnn.as_tensor(
            prefix_torch.detach().contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        state["prefix_tt"] = prefix_tt
        state["prefix_len"] = int(prefix_tt.shape[1])

    # cached_prefix_emb only exists after the first compute_embeddings() call; when the
    # pipeline is built before any utterance (build_pipeline), the prefix arrives via
    # set_prefix from the per-utterance forward instead.
    if getattr(m, "cached_prefix_emb", None) is not None:
        _upload_prefix(m.cached_prefix_emb)

    def forward(_primary=None, *args, **kwargs):
        import torch as _torch

        if "prefix_tt" not in state:
            raise RuntimeError(
                "g_p_t2_inference_model: no prefix embedding uploaded — call "
                "forward.set_prefix(gpt_inference.cached_prefix_emb) after compute_embeddings"
            )
        # Host-free decode: when a device `gen_ids_tt` [1, gen_len] (uint32) is
        # supplied, embed it directly — no host slice, no host->device upload. The
        # ids are grown on device by the caller via ttnn.concat of the on-device
        # argmax token, so the whole autoregressive feed stays resident.
        gen_ids_tt = kwargs.get("gen_ids_tt")
        if gen_ids_tt is not None:
            ids_tt = gen_ids_tt
            gen_len = int(ids_tt.shape[1])
            pos_src = ids_tt  # lpe reads only .shape[1]
        else:
            input_ids = kwargs["input_ids"]
            gen_inputs = input_ids[:, state["prefix_len"] :]  # host int slice
            gen_len = int(gen_inputs.shape[1])
            ids_tt = ttnn.as_tensor(
                gen_inputs.to(_torch.int32).contiguous(),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            pos_src = gen_inputs
        tok = ttnn.embedding(ids_tt, emb_w, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        # lpe reads only pos_src.shape[1]=gen_len and returns float32
        # [gen_len, D]; cast to bf16 so ttnn.add operand dtypes match tok.
        pos = lpe(pos_src)  # float32 [gen_len, D]
        pos = ttnn.reshape(pos, [1, gen_len, model_dim])
        pos = ttnn.typecast(pos, ttnn.bfloat16)
        gen_emb = ttnn.add(tok, pos)  # [1, gen_len, D]

        emb = ttnn.concat([state["prefix_tt"], gen_emb], dim=1)  # [1, prefix_len+gen_len, D]

        hidden = gpt2_forward(emb)
        normed = ttnn.layer_norm(hidden, epsilon=_LN_EPS, weight=lnf_w, bias=lnf_b)
        logits = ttnn.linear(
            normed,
            head_w,
            bias=head_b,
            compute_kernel_config=_head_kernel_cfg,
        )  # [1, seq, num_audio_tokens]
        return logits

    # Per-utterance prefix refresh: re-uploads ONLY the prefix buffer (the rest of the
    # build — transformer, embeddings, LM head — is reused). Same as_tensor upload path
    # as the build-time snapshot, so numerics are identical.
    forward.set_prefix = _upload_prefix
    return forward


def g_p_t2_inference_model(*args, **kwargs):
    raise RuntimeError(
        "g_p_t2_inference_model requires build(device, torch_module) to bind "
        "trained weights and the stored prefix; the bare callable has no state."
    )
