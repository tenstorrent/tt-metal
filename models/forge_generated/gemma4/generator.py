"""Generator that wraps a Gemma4ForCausalLM for multi-token sampling.

`Generator.generate(prompt_ids, max_new_tokens)` runs the prefill body
in a loop — at each step the prompt + tokens-generated-so-far are
padded to `model.seq_len` and fed through the prefill orchestration;
the logits at the last "real" position give the next token. KV caches
are reset once at the start of each session.

This matches the correctness of the legacy `demo.py` loop. The
fast prefill→decode path (one prefill + N decodes, O(seq_len + N) vs
the loop's O(seq_len*N)) is the eventual goal; it requires the
sliding-attention bodies to write into `self.k_cache`/`self.v_cache`
in-place (full bodies already do, via paged_update_cache /
fill_cache, but sliding bodies still synthesize ephemeral K/V via
concat / where). That cache-write surgery is tracked as a follow-up.
"""
import torch
from gemma4 import synthesize_prefill_inputs

import ttnn


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
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
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
            # Fresh runtime inputs each step. The prefill body writes K/V
            # into self.k_cache/self.v_cache; for now those writes don't
            # persist across this loop because each step's prefill rebuilds
            # them from scratch — but that's exactly the legacy behavior.
            runtime = synthesize_prefill_inputs(self.device, seq_len=self.seq_len)
            runtime[7] = _build_token_slot(sequence, self.seq_len, self.pad_id, self.device)

            n_slots = max(runtime) + 1
            input_list = [None] * n_slots
            for slot, t in runtime.items():
                input_list[slot] = t

            logits = self.model(input_list, mode="prefill", current_pos=0)
            logits_torch = _logits_to_torch(logits, self.device)

            # Logits at the position of the last real token; padded
            # positions [len(sequence):] don't influence under causal masking.
            last_real = len(sequence) - 1
            next_id = int(logits_torch[0, last_real].argmax().item())
            sequence.append(next_id)
            generated.append(next_id)
            if on_token is not None:
                on_token(next_id)
            if next_id == self.tokenizer.eos_token_id:
                break

        return generated
