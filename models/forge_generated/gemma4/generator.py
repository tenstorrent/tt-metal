"""Generator that wraps a Gemma4ForCausalLM for multi-token sampling.

`Generator.generate(prompt_ids, max_new_tokens)` is currently the
prefill-loop fallback: at each step the prompt + tokens-generated-so-
far are padded to seq_len and fed through the prefill body. KV caches
reset once at session start.

`Generator.fast_generate_experimental(...)` is a one-prefill-then-N-
decodes implementation that depends on the sliding-attention bodies
working with non-zero K cache. It currently produces wrong output:
the codegen-derived `_sliding_decode` body uses
`embedding(indices=var_190, weight=k_cache_reshaped)` to build the K
input to its where-mask SDPA chain, which is a circular shift of K
cache values. The reference test_decode logits were generated with
zero K cache; in that case the embedding output is well-defined
(every "shifted" row is also zero). With prefill K populating the
cache, the embedding pulls real K values, scrambling the SDPA
inputs in a way that doesn't correspond to causal attention over a
populated history.

Making fast_generate work requires rewriting `_sliding_decode` to
use `paged_update_cache(k_cache, new_K, current_pos)` +
`scaled_dot_product_attention_decode(... sliding_window_size=256)` —
mirroring tt_transformers/tt/attention.py:730-776. That's
substantial surgery (~400 lines of codegen-derived ops replaced with
~30 lines of canonical tt-metal calls), tracked as the natural next
step.

The cache-write groundwork is in place:
  - Full layers: existing `fill_cache(k_cache, ttnn_add_115, 0)` for
    prefill at rows 0..seq_len-1, `paged_update_cache(... current_pos)`
    for decode. Already canonical.
  - Sliding layers: prefill writes via `fill_cache(k_cache, padded, 0)`
    at rows 256-seq_len..255 (front-padded with zeros via concat).
    Decode captures `ttnn_where_8`/`ttnn_where_10` from the attention
    return tuple and reattaches them to `self.k_cache`/`self.v_cache`
    on the layer. PCC stays green.
"""
import torch
from gemma4 import synthesize_decode_inputs, synthesize_prefill_inputs

import ttnn

_DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def _build_prefill_token_slot(ids, seq_len, pad_id, device):
    """Build the int32 [1, seq_len] prefill token-IDs tensor (slot 7)."""
    if len(ids) > seq_len:
        ids = ids[:seq_len]
    elif len(ids) < seq_len:
        ids = ids + [pad_id] * (seq_len - len(ids))
    user_tokens = torch.tensor(ids, dtype=torch.int32).reshape(1, seq_len)
    return ttnn.as_tensor(
        user_tokens,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=device,
        memory_config=_DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def _build_decode_token_slot(token_id, device):
    """Build the int32 [1, 1] decode-step token-ID tensor (slot 7)."""
    return ttnn.as_tensor(
        torch.tensor([[int(token_id)]], dtype=torch.int32),
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=device,
        memory_config=_DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def _logits_to_torch(logits, device):
    """Bring device-side logits back to CPU as [1, seq, vocab] float."""
    host = ttnn.from_device(logits)
    return ttnn.to_torch(
        host,
        mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0),
    )[:1, :, :].float()


def _runtime_to_input_list(runtime):
    n_slots = max(runtime) + 1
    out = [None] * n_slots
    for slot, t in runtime.items():
        out[slot] = t
    return out


class Generator:
    """Owns a model + tokenizer and runs prefill→decode greedy sampling."""

    def __init__(self, model, tokenizer, *, seq_len, pad_id=None):
        self.model = model
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_id = pad_id if pad_id is not None else (tokenizer.pad_token_id or 0)
        self.device = model.mesh_device

    def reset(self):
        """Clear KV caches between independent generation sessions."""
        self.model.reset_kv_caches()

    def generate(self, prompt_ids, *, max_new_tokens, on_token=None):
        """Prefill-loop generation. Each step pads prompt+tokens-so-far
        to seq_len and runs the prefill body; argmax of the last-real-
        position logits is the next token. EOS short-circuits.

        This is the correctness path. See module docstring for why the
        fast prefill→decode path is still experimental.
        """
        if len(prompt_ids) >= self.seq_len:
            raise ValueError(
                f"prompt is {len(prompt_ids)} tokens; need < seq_len={self.seq_len} headroom for generation."
            )

        self.reset()

        sequence = list(prompt_ids)
        generated = []
        cap = min(max_new_tokens, self.seq_len - len(prompt_ids))

        for _ in range(cap):
            runtime = synthesize_prefill_inputs(self.device, seq_len=self.seq_len)
            runtime[7] = _build_prefill_token_slot(sequence, self.seq_len, self.pad_id, self.device)
            logits = self.model(_runtime_to_input_list(runtime), mode="prefill", current_pos=0)
            logits_torch = _logits_to_torch(logits, self.device)
            last_real = len(sequence) - 1
            next_id = int(logits_torch[0, last_real].argmax().item())
            sequence.append(next_id)
            generated.append(next_id)
            if on_token is not None:
                on_token(next_id)
            if next_id == self.tokenizer.eos_token_id:
                break

        return generated

    def fast_generate_experimental(self, prompt_ids, *, max_new_tokens, on_token=None):
        """One prefill + N decode steps. Currently produces incorrect
        output — see module docstring. Kept for follow-up work on the
        sliding-decode rewrite.
        """
        if len(prompt_ids) >= self.seq_len:
            raise ValueError(
                f"prompt is {len(prompt_ids)} tokens; need < seq_len={self.seq_len} headroom for generation."
            )

        self.reset()

        prefill_runtime = synthesize_prefill_inputs(self.device, seq_len=self.seq_len)
        prefill_runtime[7] = _build_prefill_token_slot(prompt_ids, self.seq_len, self.pad_id, self.device)
        prefill_logits = self.model(_runtime_to_input_list(prefill_runtime), mode="prefill", current_pos=0)
        prefill_torch = _logits_to_torch(prefill_logits, self.device)
        last_real = len(prompt_ids) - 1
        next_id = int(prefill_torch[0, last_real].argmax().item())

        generated = [next_id]
        if on_token is not None:
            on_token(next_id)
        if next_id == self.tokenizer.eos_token_id:
            return generated

        current_pos = len(prompt_ids)
        for _ in range(max_new_tokens - 1):
            decode_runtime = synthesize_decode_inputs(self.device)
            decode_runtime[7] = _build_decode_token_slot(next_id, self.device)
            logits = self.model(_runtime_to_input_list(decode_runtime), mode="decode", current_pos=current_pos)
            logits_torch = _logits_to_torch(logits, self.device)
            next_id = int(logits_torch[0, 0].argmax().item())
            current_pos += 1
            generated.append(next_id)
            if on_token is not None:
                on_token(next_id)
            if next_id == self.tokenizer.eos_token_id:
                break

        return generated
