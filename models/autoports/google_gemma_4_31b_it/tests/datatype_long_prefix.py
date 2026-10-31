"""Agent-shaped retained notes and structured reply across a long chat prefix."""

import json


def run(gen, root):
    tokenizer = gen.tokenizer
    notes = "\n".join(
        f"Completed task {i}: inspected source, ran checks, and recorded the unchanged result." for i in range(4000)
    )
    note_ids = tokenizer.encode(notes, add_special_tokens=False)[:32000]
    messages = [
        {
            "role": "user",
            "content": "Remember the deployment label ORCHID-742. We will refer to it after reviewing the work log.",
        },
        {"role": "assistant", "content": "The deployment label is ORCHID-742."},
        {
            "role": "user",
            "content": tokenizer.decode(note_ids)
            + '\nReturn only a JSON object with key "deployment_label" and the deployment label given at the beginning of our conversation.',
        },
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(rendered, add_special_tokens=False)
    outputs = gen.generate_batch([tokens] * gen.batch, 128) if gen.batch > 1 else [gen.generate(tokens, 128)]
    output = outputs[0]
    completion = tokenizer.decode(output)
    result = {
        "prompt_mode": "chat",
        "prompt_tokens": len(tokens),
        "rendered_prompt_file": "long_prefix_prompt.txt",
        "generated_tokens": output,
        "completion": completion,
        "contains_expected_label": "ORCHID-742" in completion,
        "expected_label": "ORCHID-742",
        "generation_cap": 128,
        "batch": gen.batch,
        "all_completions": [tokenizer.decode(o) for o in outputs],
        "all_slots_retrieve_label": all("ORCHID-742" in tokenizer.decode(o) for o in outputs),
    }
    (root / "long_prefix_prompt.txt").write_text(rendered)
    (root / "long_prefix.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
