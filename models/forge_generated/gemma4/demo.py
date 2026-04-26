"""Gemma4-31B-it single-step demo — tokenize a question, run prefill,
print the predicted next token.

The codegen artifact bakes prefill seq_len=19 into the model
(the input slot 7 has shape [1, 19]). So the prompt — wrapped by
gemma's chat template — must fit in 19 tokens. Short questions like
"What is your favorite city?" wrap to exactly 19 tokens.

This is a *single-token* demo: it shows the model produces a sensible
next-token prediction for a user prompt. Full multi-token generation
needs a Generator class that threads KV cache state from prefill
output → decode input — see gap analysis #1 (HIGH priority).

Usage:
    python -m gemma4.demo "What is your favorite city?"
    python -m gemma4.demo "Why is the sky blue?"

Environment:
    Activated venv (`source venv/activate`), TT_MLIR_HOME set,
    PYTHONPATH including {TT_METAL_RUNTIME_ROOT}/ttnn (the legacy
    `gemma4_prefill/run` script set these for the test harness).
"""
import argparse

import gemma4
import torch
from gemma4 import weights as gw
from transformers import AutoTokenizer

import ttnn

PREFILL_SEQ_LEN = 19  # baked into the codegen


def _tokenize(prompt: str, tokenizer) -> list[int]:
    """Apply gemma's chat template + tokenize. Pad/truncate to PREFILL_SEQ_LEN."""
    messages = [{"role": "user", "content": prompt}]
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    ids = enc["input_ids"][0].tolist()
    if len(ids) > PREFILL_SEQ_LEN:
        print(f"[warn] prompt is {len(ids)} tokens after chat-template; " f"truncating to {PREFILL_SEQ_LEN}.")
        ids = ids[:PREFILL_SEQ_LEN]
    elif len(ids) < PREFILL_SEQ_LEN:
        pad_id = tokenizer.pad_token_id or 0
        print(f"[info] padding from {len(ids)} → {PREFILL_SEQ_LEN} tokens " f"(pad_id={pad_id}).")
        ids = ids + [pad_id] * (PREFILL_SEQ_LEN - len(ids))
    return ids


def _override_token_ids(runtime: dict, ids: list[int], device) -> None:
    """Replace runtime[7] with the user's tokenized prompt."""
    user_tokens = torch.tensor(ids, dtype=torch.int32).reshape(1, PREFILL_SEQ_LEN)
    runtime[7] = ttnn.as_tensor(
        user_tokens,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def run(prompt: str) -> tuple[int, str]:
    print(f"\n>>> Prompt: {prompt!r}\n")

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")
    ids = _tokenize(prompt, tokenizer)
    print(f"Tokenized (len={len(ids)}): {ids}")

    device = gemma4.utils.DeviceGetter.get_device((1, 4))

    runtime = gemma4.synthesize_prefill_inputs(device)
    _override_token_ids(runtime, ids, device)

    hf = gw.load_hf_weights()
    n_slots = max(runtime) + 1
    input_list = [None] * n_slots
    for slot, t in runtime.items():
        input_list[slot] = t

    model = gemma4.Gemma4ForCausalLM.from_state_dict(
        hf,
        device,
        is_decode=False,
    )
    logits = model(input_list)

    logits_host = ttnn.from_device(logits)
    logits_torch = ttnn.to_torch(
        logits_host,
        mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0),
    )[:1, :, :].float()

    last_pos = logits_torch.shape[1] - 1
    next_id = logits_torch[0, last_pos].argmax().item()
    next_text = tokenizer.decode([next_id])
    print(f"\n<<< Next token: id={next_id}, text={next_text!r}\n")
    return next_id, next_text


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "prompt",
        nargs="?",
        default="What is your favorite city?",
        help="A short question (must fit in 19 tokens after chat-template " "wrapping).",
    )
    args = parser.parse_args()
    run(args.prompt)


if __name__ == "__main__":
    main()
