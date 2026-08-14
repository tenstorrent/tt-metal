# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The shared qualitative suite, re-run against the optimized decode path.

Same harness as the full-model stage -- ``doc/full_model/bench/qualitative.py``,
loaded by path -- with artifacts under ``doc/optimized_full_model/qualitative/``.
The HF control is expensive (CPU bf16, 128 tokens per prompt), so the full-model
stage's committed control is copied in rather than regenerated when
``--reuse-hf-control`` is given; ``$qualitative-check`` accepts the previous-stage
control for a serving/optimization regression comparison, and the checkpoint,
tokenizer, prompt set and generation parameters are unchanged (the copied
``qualitative_prompt_format.json`` is the proof).

Usage::

    python doc/optimized_full_model/bench/qualitative.py --arm tt
    python doc/optimized_full_model/bench/qualitative.py --arm compare --reuse-hf-control
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

PREV = ROOT / "doc/full_model/qualitative"
OUT = ROOT / "doc/optimized_full_model/qualitative"


def main() -> int:
    path = ROOT / "doc/full_model/bench/qualitative.py"
    spec = importlib.util.spec_from_file_location("muse_glimmer_full_model_qualitative", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)

    if "--reuse-hf-control" in sys.argv:
        sys.argv.remove("--reuse-hf-control")
        for name in ("qualitative_hf_chat.json", "qualitative_prompt_format.json", "qualitative_prompts.json"):
            source = PREV / name
            if source.exists() and not (OUT / name).exists():
                shutil.copy2(source, OUT / name)
                module.say(f"QUAL reused the full-model stage control {name}")
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
