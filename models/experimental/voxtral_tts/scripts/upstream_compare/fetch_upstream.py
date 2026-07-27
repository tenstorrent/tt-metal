# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fetch the two vLLM-Omni model files this comparison runs against, at the PINNED commit.

Not vendored into the tree: they are live vLLM-Omni code (Apache-2.0) with a large import tree,
and pinning by SHA here keeps the comparison reproducible without taking on third-party source.
See ../../reference/PROVENANCE.md for the pin and the licence note.

    python models/experimental/voxtral_tts/scripts/upstream_compare/fetch_upstream.py
"""

import os
import urllib.request

# Keep in sync with reference/PROVENANCE.md
COMMIT = "8001bb155dae5798a1ae891ae2529a314c6ee99a"
BASE = f"https://raw.githubusercontent.com/vllm-project/vllm-omni/{COMMIT}"
FILES = [
    "vllm_omni/model_executor/models/voxtral_tts/voxtral_tts_audio_generation.py",
    "vllm_omni/model_executor/models/voxtral_tts/voxtral_tts_audio_tokenizer.py",
]
DEST = os.environ.get("VOXTRAL_UPSTREAM_SRC",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream_src"))


def main():
    os.makedirs(DEST, exist_ok=True)
    for rel in FILES:
        out = os.path.join(DEST, os.path.basename(rel))
        url = f"{BASE}/{rel}"
        with urllib.request.urlopen(url) as r:
            data = r.read()
        with open(out, "wb") as f:
            f.write(data)
        print(f"[fetch] {len(data):6d} B  {os.path.basename(rel)}")
    print(f"[fetch] -> {DEST} (commit {COMMIT[:12]})")


if __name__ == "__main__":
    main()
