#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test tokenization for PI0 simulation.

Compares SimpleRoboticsTokenizer vs Gemma tokenizer on common
robotics prompts to help debug trajectory issues.

Usage:
    python models/experimental/pi0/tests/demo/test_tokenization.py
"""

import sys
import os

sys.path.insert(0, os.environ.get("TT_METAL_HOME", os.path.join(os.path.dirname(__file__), *[".."] * 4)))

from models.experimental.pi0.tests.demo.run_pybullet_sim import SimpleRoboticsTokenizer

TEST_PROMPTS = [
    "pick up cube",
    "pick and place",
    "grasp object",
    "reach forward",
    "push block right",
    "lift the cube up",
    "move to the target",
]


def main():
    print("=" * 60)
    print("  PI0 Tokenization Comparison")
    print("=" * 60)

    simple = SimpleRoboticsTokenizer()
    print(f"\n--- SimpleRoboticsTokenizer (vocab_size={simple.vocab_size}) ---\n")

    for prompt in TEST_PROMPTS:
        tokens, mask = simple.encode(prompt, max_length=16)
        valid_count = sum(mask)
        print(f"  '{prompt}'")
        print(f"    Tokens: {tokens[:valid_count]}")
        print(f"    Count: {valid_count} (padded to {len(tokens)})")
        print()

    try:
        from models.experimental.pi0.tests.demo.run_pybullet_sim import GemmaTokenizerWrapper
        print("\n--- Gemma Tokenizer ---\n")
        gemma = GemmaTokenizerWrapper()
        if gemma.use_official:
            for prompt in TEST_PROMPTS:
                tokens, mask = gemma.encode(prompt, max_length=16)
                valid_count = sum(mask)
                print(f"  '{prompt}'")
                print(f"    Tokens: {tokens[:valid_count]}")
                print(f"    Count: {valid_count} (padded to {len(tokens)})")
                print()
        else:
            print("  Gemma tokenizer not available (fallback active).")
            print("  Run: huggingface-cli login")
    except Exception as e:
        print(f"\n  Could not load Gemma tokenizer: {e}")
        print("  Run: pip install transformers && huggingface-cli login")

    print("\nTip: If the robot takes wrong trajectories, the tokenizer")
    print("may produce token IDs the model doesn't recognize.")
    print("Use --use-gemma-tokenizer for best results.")


if __name__ == "__main__":
    main()
