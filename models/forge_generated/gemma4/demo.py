"""Gemma4-31B-it multi-token generation demo.

Runs prefill in a loop: at each step the prompt + tokens-generated-so-far
are padded to seq_len, fed through the prefill model, and the logits at
the last "real" position give the next token. The generated tokens
accumulate until either max_new_tokens is reached or seq_len fills up.

This uses prefill-only (the codegen prefill body is what we have a
verified PCC for). True decode-mode continuation would need KV-cache
handoff between prefill and decode — out of scope for this demo.

Usage:
    python -m gemma4.demo "What is your favorite city?"
    python -m gemma4.demo "Why is the sky blue?" --seq-len 64 --max-new-tokens 32
"""
import argparse

import gemma4
import torch
from gemma4 import weights as gw
from transformers import AutoTokenizer

import ttnn


def _tokenize_prompt(prompt: str, tokenizer) -> list[int]:
    """Apply gemma's chat template + tokenize. Returns the un-padded
    list of token IDs (length = chat-template-wrapped prompt length).
    """
    messages = [{"role": "user", "content": prompt}]
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return enc["input_ids"][0].tolist()


def _build_token_slot(ids: list[int], seq_len: int, pad_id: int, device):
    """Pad/truncate `ids` to length `seq_len` and build the ttnn.Tensor
    for runtime input slot 7 (shape [1, seq_len], INT32, ROW_MAJOR,
    replicated across the (1,4) mesh).
    """
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
    """Bring the device-side logits back to CPU as a [1, seq_len, vocab]
    float tensor (concatenating shards along the vocab dim)."""
    host = ttnn.from_device(logits)
    return ttnn.to_torch(
        host,
        mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0),
    )[:1, :, :].float()


def run(prompt: str, *, seq_len: int = 64, max_new_tokens: int = 32) -> tuple[list[int], str]:
    print(f"\n>>> Prompt: {prompt!r}")
    print(f"    seq_len={seq_len}, max_new_tokens={max_new_tokens}\n")

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")
    pad_id = tokenizer.pad_token_id or 0
    prompt_ids = _tokenize_prompt(prompt, tokenizer)
    if len(prompt_ids) >= seq_len:
        print(
            f"[error] prompt is {len(prompt_ids)} tokens after chat-template, "
            f"≥ seq_len={seq_len}; nothing to generate."
        )
        return [], ""
    print(f"Prompt tokens (len={len(prompt_ids)}): {prompt_ids}")
    print(f"Prompt text: {tokenizer.decode(prompt_ids)!r}")

    device = gemma4.utils.DeviceGetter.get_device((1, 4))

    # Build the model once at the chosen seq_len. Re-running prefill at
    # different seq_lens would mean rebuilding (preludes / scalar tensors
    # bake the seq_len in at construction time).
    print("\nLoading HF weights and building prefill model...")
    hf = gw.load_hf_weights()
    model = gemma4.Gemma4ForCausalLM.from_state_dict(
        hf,
        device,
        is_decode=False,
        seq_len=seq_len,
    )

    sequence = list(prompt_ids)
    generated = []
    cap = min(max_new_tokens, seq_len - len(prompt_ids))
    if cap < max_new_tokens:
        print(f"[info] capping max_new_tokens at {cap} (seq_len - prompt_len).")

    print(f"\nGenerating up to {cap} tokens...")
    for step in range(cap):
        # Fresh runtime inputs each step — the KV caches start from zeros
        # and prefill writes them in-place. Override slot 7 with the
        # current sequence padded to seq_len.
        runtime = gemma4.synthesize_prefill_inputs(device, seq_len=seq_len)
        runtime[7] = _build_token_slot(sequence, seq_len, pad_id, device)

        n_slots = max(runtime) + 1
        input_list = [None] * n_slots
        for slot, t in runtime.items():
            input_list[slot] = t

        logits = model(input_list)
        logits_torch = _logits_to_torch(logits, device)

        # The next-token prediction is the logits at the position of the
        # last real token: index len(sequence) - 1 (positions [len:seq_len]
        # contain pad tokens; under causal masking they don't influence
        # the position-(len-1) logits).
        last_real = len(sequence) - 1
        next_id = int(logits_torch[0, last_real].argmax().item())
        next_text = tokenizer.decode([next_id])
        sequence.append(next_id)
        generated.append(next_id)
        print(f"  step {step:>2}: id={next_id:>6}  text={next_text!r}")

        if next_id == tokenizer.eos_token_id:
            print("  [eos reached]")
            break

    completion = tokenizer.decode(generated)
    print(f"\n<<< Generated ({len(generated)} tokens): {completion!r}\n")
    return generated, completion


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "prompt",
        nargs="?",
        default="What is your favorite city?",
        help="A short question. Will be wrapped by Gemma's chat template.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Prefill sequence length. Must be > prompt token count. "
        "Larger seq_len → more headroom for generation but more compute "
        "per step. Max ≈ 256 (KV cache buffer size).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Cap on generated tokens (also bounded by seq_len - prompt_len).",
    )
    args = parser.parse_args()
    run(args.prompt, seq_len=args.seq_len, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
