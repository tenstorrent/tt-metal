"""Gemma4-31B-it multi-token generation demo.

Wraps `gemma4.Generator` around a Gemma4ForCausalLM and a HF tokenizer.
The generator currently runs prefill-loop sampling: at each step the
prompt + tokens-so-far are padded to seq_len and fed through the
prefill body; argmax of the last-real-position logits is the next
token. The fast prefill→decode path (one prefill + N decodes) is the
follow-up that depends on sliding-attention KV cache writes.

Usage:
    python -m gemma4.demo "What is your favorite city?"
    python -m gemma4.demo "Why is the sky blue?" --seq-len 64 --max-new-tokens 32
"""
import argparse

import gemma4
from gemma4 import weights as gw
from transformers import AutoTokenizer


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


def run(prompt: str, *, seq_len: int = 64, max_new_tokens: int = 32) -> tuple[list[int], str]:
    print(f"\n>>> Prompt: {prompt!r}")
    print(f"    seq_len={seq_len}, max_new_tokens={max_new_tokens}\n")

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")
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

    print("\nLoading HF weights and building model...")
    hf = gw.load_hf_weights()
    model = gemma4.Gemma4ForCausalLM.from_state_dict(hf, device, seq_len=seq_len)
    generator = gemma4.Generator(model, tokenizer, seq_len=seq_len)

    cap = min(max_new_tokens, seq_len - len(prompt_ids))
    if cap < max_new_tokens:
        print(f"[info] capping max_new_tokens at {cap} (seq_len - prompt_len).")

    print(f"\nGenerating up to {cap} tokens...")

    step_counter = {"i": 0}

    def on_token(tok_id):
        i = step_counter["i"]
        text = tokenizer.decode([tok_id])
        print(f"  step {i:>2}: id={tok_id:>6}  text={text!r}")
        step_counter["i"] = i + 1

    generated = generator.generate(prompt_ids, max_new_tokens=cap, on_token=on_token)

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
