# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The CPU reference golden of the e2e test (test_e2e_galaxy.py): the package's fp32 reference over a contiguous run
of layers on the first ``--seq`` tokens of a prompt, in chunks of ``--chunk``. No GPU golden of this model exists.

    python -m models.demos.deepseek_v3_d_p.tests.mistral_large_4.cpu_golden \
        --prompt prompt.json --out DIR [--seq 5120] [--layers 0-71] [--dump 0,1,...]

Writes DIR/chunk_<c>.safetensors per chunk (``L<i>``: the output of each dumped layer, ``final_norm``; bfloat16, rows
in position order) and, for the last chunk, ``logits`` (fp32, its last LOGITS_ROWS rows), then DIR/meta.json. With
``--state``, also the serving checks' golden in tt-metal's prefill trace layout: DIR/metadata.json (``token_ids``) and
DIR/kv_cache/layer_<i>.safetensors (``kv_latent_cache_layer_<i>``, ``k_rope_cache_layer_<i>``,
``index_key_cache_layer_<i>``: each layer's state over the whole sequence, ``--state-dtype``; [seq, 0] where it owns none). Holds
the dense weights in bf16 (about 44 GB for every layer) and streams the routed experts from the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

CHECKPOINT_DIR = "/mnt/models/mistralai/Mistral-Large-4-1T-A52B"
LOGITS_ROWS = 32
STATE_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
# layers dumped by default: every block variant early, then groups through the stack to the last layer
DUMP = (0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 18, 31, 35, 36, 39, 40, 47, 55, 63, 67, 70, 71)


class Spec:
    """The spec fields the reference reads (reference/model.py Reference)."""

    def __init__(self, hf_dir: str, max_seq: int):
        self.hf_dir = Path(hf_dir)
        self._v = {"reference.weight_dtype": "bfloat16", "shapes.max_seq_len": max_seq}

    def get(self, key, default=None):
        return self._v.get(key, default)


def parse_layers(text: str) -> list[int]:
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


def prompt_tokens(path: str) -> list[int]:
    return list(json.loads(Path(path).read_text())["token_ids"])


def main(argv=None):
    from safetensors.torch import save_file

    from models.demos.deepseek_v3_d_p.reference.mistral_large_4.model import Reference

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--prompt", required=True, help="a prompt.json with token_ids")
    p.add_argument("--out", required=True)
    p.add_argument("--seq", type=int, default=5120)
    p.add_argument("--chunk", type=int, default=5120)
    p.add_argument("--layers", default="0-71")
    p.add_argument("--dump", default=",".join(map(str, DUMP)))
    p.add_argument("--hf-dir", default=CHECKPOINT_DIR)
    p.add_argument("--state", action="store_true", help="also write the serving golden (kv_cache/, metadata.json)")
    p.add_argument("--state-dtype", default="float32", choices=sorted(STATE_DTYPES))
    a = p.parse_args(argv)
    assert a.seq % a.chunk == 0, (a.seq, a.chunk)
    layers = parse_layers(a.layers)
    dump = [i for i in map(int, a.dump.split(",")) if i in layers]
    if layers[-1] not in dump:
        dump.append(layers[-1])
    tokens = prompt_tokens(a.prompt)[: a.seq]
    assert len(tokens) == a.seq, f"the prompt has {len(tokens)} tokens, need {a.seq}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ref = Reference(Spec(a.hf_dir, a.seq), layers=layers, dtype=torch.float32)
    load_s = time.time() - t0
    state = ref.new_state(a.seq)
    chunks = a.seq // a.chunk
    times = []
    for c in range(chunks):
        got = {}

        def rec(name, t):
            if name.endswith(".out") and int(name[1:].split(".")[0]) in dump:
                got[name.split(".")[0]] = t.detach().to(torch.bfloat16).contiguous()

        t1 = time.time()
        final, logits = ref.forward_chunk(
            tokens[c * a.chunk : (c + 1) * a.chunk],
            c * a.chunk,
            state,
            rec=rec,
            logits_last_n=LOGITS_ROWS if c == chunks - 1 else 0,
        )
        got["final_norm"] = final.to(torch.bfloat16).contiguous()
        if logits is not None:
            got["logits"] = logits.float().contiguous()
        save_file(got, str(out / f"chunk_{c:02d}.safetensors"))
        times.append(round(time.time() - t1, 1))
        print(f"chunk {c}: {times[-1]} s", flush=True)
    if a.state:
        (out / "kv_cache").mkdir(exist_ok=True)
        for i in layers:
            st = ref.state_tensors(state, i, a.seq)
            save_file(
                {f"{n}_cache_layer_{i}": t.to(STATE_DTYPES[a.state_dtype]).contiguous() for n, t in st.items()},
                str(out / "kv_cache" / f"layer_{i}.safetensors"),
            )
        (out / "metadata.json").write_text(json.dumps({"token_ids": tokens, "seq_len": a.seq, "layers": layers}) + "\n")
    meta = {
        "hf_dir": a.hf_dir,
        "prompt": a.prompt,
        "tokens_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "seq": a.seq,
        "chunk": a.chunk,
        "layers": [layers[0], layers[-1]],
        "dump": dump,
        "logits_rows": LOGITS_ROWS,
        "reference": "fp32 compute, bf16 weights (reference/model.py)",
        "load_seconds": round(load_s, 1),
        "chunk_seconds": times,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
