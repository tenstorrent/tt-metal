#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create three long, distinct prompts from Frankenstein.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--words-per-prompt", type=int, default=16000)
    args = parser.parse_args()

    text = args.source.read_text()
    words = list(re.finditer(r"\S+", text))
    required = 3 * args.words_per_prompt
    if len(words) < required:
        raise ValueError(f"{args.source} has {len(words)} words; at least {required} are required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        first = index * args.words_per_prompt
        last = first + args.words_per_prompt
        prompt = text[words[first].start() : words[last - 1].end()]
        output = args.output_dir / f"prompt_{index + 1}.json"
        output.write_text(json.dumps({"prompt": prompt}, ensure_ascii=False, indent=2) + "\n")
        print(f"{output}: source words [{first}, {last})")


if __name__ == "__main__":
    main()
