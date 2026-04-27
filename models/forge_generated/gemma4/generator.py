"""Generator that wraps a Gemma4ForCausalLM for multi-token sampling.

`Generator.generate(prompt_ids, max_new_tokens)` runs the prefill body
in a loop — at each step the prompt + tokens-generated-so-far are
padded to `seq_len` and fed through the prefill orchestration; the
logits at the last "real" position give the next token. KV caches
are reset once at the start of each session.

The fast prefill→decode path (one prefill + N decodes, O(seq_len + N)
vs the loop's O(seq_len*N)) is the eventual goal. It needs the
sliding-attention cache geometry to be reworked: the existing
`_sliding_decode` body's masked write places the new K/V at a fixed
"last row" (row 255) of a 256-slot circular buffer, not at row
`pos_ids`. Adding `fill_cache(k_cache, new_K, 0)` in `_sliding_prefill`
populates rows 0..seq_len-1, which doesn't align with the decode
body's expected geometry — first decode step then produces the same
token as prefill, repeating. A real fix needs one of:
  - prefill writes to rows `256-seq_len..255` (most recent prompt
    token at row 255), or
  - decode replaces the where-mask with paged_update_cache at
    row pos_ids, or
  - introduce circular indexing into the where mask.
Each is non-trivial surgery on the codegen-derived ttnn op graph.
Leaving as a follow-up. Full layers already do paged_update_cache /
fill_cache and would work if sliding caught up.
"""
import torch
from gemma4 import synthesize_prefill_inputs

import ttnn

_DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def _build_token_slot(ids, seq_len, pad_id, device):
    """Build the int32 [1, seq_len] token-IDs tensor for runtime slot 7."""
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


def _logits_to_torch(logits, device):
    """Bring device-side logits back to CPU as [1, seq_len, vocab] float."""
    host = ttnn.from_device(logits)
    return ttnn.to_torch(
        host,
        mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0),
    )[:1, :, :].float()


class Generator:
    """Owns a model + tokenizer and runs prefill-loop generation."""

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
        """Generate up to `max_new_tokens` continuation tokens.

        Each step pads `prompt_ids + generated_so_far` to `seq_len` and
        runs prefill; the next token is the argmax of the logits at the
        last real prompt+gen position. EOS short-circuits the loop.

        `on_token` is an optional callback invoked with each generated
        token id (useful for streaming output in `demo.py`).
        """
        if len(prompt_ids) >= self.seq_len:
            raise ValueError(
                f"prompt is {len(prompt_ids)} tokens; need < seq_len={self.seq_len} " "headroom for generation."
            )

        self.reset()

        sequence = list(prompt_ids)
        generated = []
        cap = min(max_new_tokens, self.seq_len - len(prompt_ids))

        for _ in range(cap):
            runtime = synthesize_prefill_inputs(self.device, seq_len=self.seq_len)
            runtime[7] = _build_token_slot(sequence, self.seq_len, self.pad_id, self.device)
            n_slots = max(runtime) + 1
            input_list = [None] * n_slots
            for slot, t in runtime.items():
                input_list[slot] = t

            logits = self.model(input_list, mode="prefill", current_pos=0)
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
