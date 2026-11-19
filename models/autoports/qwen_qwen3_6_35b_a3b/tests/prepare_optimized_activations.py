# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prepare natural-text HF layer inputs for decoder-only precision probes.

This host-only helper never imports TTNN and never constructs a full model. It
reads only the requested embedding rows, then loads one checkpoint decoder layer
at a time. Each layer processes the prefix followed by its held-out next token
using that layer's causal cache. Its outputs become the following layer's inputs.
Thus ``xd`` is a real next-token hidden input, not random noise or an independently
processed token. The target layer itself need not run to capture its inputs.

Example (after sourcing the existing lane environment)::

    timeout 3600 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.prepare_optimized_activations \
        --length 129 --layers 0 3 34 --output <artifact-directory>/natural129.pt

``--length`` includes the held-out token: 129 produces x=[1,128,2048] and
xd=[1,1,2048]. Use 257 for a 256-token prefill. A supplied ``--text-file`` must
contain enough tokens; text is never repeated or padded to meet the length.

The torch.load(..., weights_only=True) payload contains ``input_ids``, ``layers``
(mapping string layer numbers to ``x``/``xd`` CPU BF16 tensors), and ``metadata``.
Consume payload["layers"][str(layer_idx)] and assert that x.shape[1] matches the
benchmark's requested prefix length. Recompute the target-layer HF reference
from these x/xd inputs, using a fresh cache and decode position x.shape[1]. The
sibling .metadata.json records exact selected token IDs, prompt format, checkpoint
and propagation settings. These are input-provenance artifacts, not model-quality,
full-model, device-correctness or performance results. No chat-quality verdict is
made: the prompt is raw natural-text teacher forcing with no chat template.
"""

import argparse
import gc
import importlib.metadata
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

# Original, nonrepeating prose used only as a deterministic teacher-forced input.
NATURAL_TEXT = """At the edge of the village, a narrow path followed the river toward an old stone bridge.
Mara walked there before breakfast to inspect the water gauge. Overnight rain had raised the river,
but the lower steps were still visible. She wrote the reading in a notebook, checked it a second time,
and marked the difference from yesterday. The notebook also held observations about wind, fallen
branches, and the time at which the first sunlight reached the bridge.

On her way back she met a carpenter carrying two lengths of wood. He was repairing the public notice
board beside the market. They discussed where to move it so people could read the announcements
without standing in the rain. A roof over the board would help, but it would also cast a shadow on the
small map. They agreed to make a cardboard model before cutting the final pieces.

Later that morning the library opened its doors for a workshop. Each table held a different collection:
old railway schedules, photographs of gardens, drawings of local birds, and letters describing how the
village had changed. The visitors were asked to choose one object and write down a question about it.
A child noticed that the station clock in a photograph had no hands. An older visitor remembered that
the clock had been removed for cleaning just before the picture was taken. Their conversation led to
a longer discussion about the difference between an observation and an explanation.

By noon the sky had cleared. The market sellers rolled back their canvas covers, and a baker placed
warm loaves on a wooden shelf. Mara bought bread and returned to the bridge with her notebook.
The river had risen again, though much less quickly than during the night. She compared the two
measurements and decided to take another reading before evening. A single number could describe the
water level at one moment, but several measurements would show whether the change was slowing.

At home she spread the village map across the kitchen table. There were three routes between the
bridge and the library. The shortest crossed a field that was often muddy after rain. The second
followed a paved road with a steep hill. The third wound through the orchard and passed a sheltered
bench. She drew a small symbol beside each route to record its advantages, leaving enough space to
add notes when the seasons changed. The map was useful because it kept the details visible without
pretending that one route would be best for every journey.
"""


def embedding_inputs(snapshot, input_ids, hidden_size):
    """Read unique embedding rows without materializing the vocabulary matrix."""
    import torch
    from safetensors import safe_open

    name = "model.language_model.embed_tokens.weight"
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    shard = snapshot / index[name]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        table = handle.get_slice(name)
        shape = table.get_shape()
        assert len(shape) == 2 and shape[1] == hidden_size, shape
        token_ids = input_ids[0].tolist()
        assert min(token_ids) >= 0 and max(token_ids) < shape[0]
        unique_rows = {token: table[token : token + 1, :] for token in sorted(set(token_ids))}
        result = torch.cat([unique_rows[token] for token in token_ids], dim=0).unsqueeze(0)
    assert result.dtype == torch.bfloat16, "Expected actual checkpoint BF16 embeddings; do not silently cast"
    return result, {"tensor": name, "shard": str(shard), "shape": shape, "unique_rows_read": len(unique_rows)}


def run(args):
    # Keep heavy imports inside run: --help and syntax checks are CPU-light.
    import torch
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache

    from models.autoports.qwen_qwen3_6_35b_a3b.tests import reference

    assert "ttnn" not in sys.modules, "This preparation is host-only"
    if args.length < 2 or args.threads < 1:
        raise ValueError("length must include a nonempty prefix and held-out token; threads must be positive")
    layers = sorted(set(args.layers))
    config = reference.config()
    if not layers or layers[0] < 0 or layers[-1] >= config.num_hidden_layers:
        raise ValueError("Requested layer input is outside the checkpoint")
    if args.length > config.max_position_embeddings:
        raise ValueError("Requested text exceeds the model context")
    output = args.output.resolve()
    metadata_path = output.with_suffix(".metadata.json")
    if output.suffix != ".pt":
        raise ValueError("--output must have a .pt suffix")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("Use fresh output and metadata paths to preserve prior evidence")
    torch.set_num_threads(args.threads)
    text = args.text_file.read_text() if args.text_file else NATURAL_TEXT
    tokenizer = AutoTokenizer.from_pretrained(reference.SNAPSHOT, local_files_only=True)
    all_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(all_ids) < args.length:
        raise ValueError(f"Text has {len(all_ids)} tokens but {args.length} requested; provide longer natural text")
    input_ids = torch.tensor([all_ids[: args.length]], dtype=torch.long)
    hidden, embedding_metadata = embedding_inputs(reference.SNAPSHOT, input_ids, config.hidden_size)
    prefix, last = hidden[:, :-1].contiguous(), hidden[:, -1:].contiguous()
    del hidden
    captured = {}
    upstream = []
    start_time = time.monotonic()
    for layer_idx in range(layers[-1] + 1):
        assert prefix.dtype == last.dtype == torch.bfloat16
        assert torch.isfinite(prefix).all() and torch.isfinite(last).all(), layer_idx
        if layer_idx in layers:
            captured[str(layer_idx)] = {"x": prefix.detach().clone(), "xd": last.detach().clone()}
        if layer_idx == layers[-1]:
            break
        state_dict = reference.weights(layer_idx)
        hf = reference.layer(state_dict, layer_idx)
        del state_dict
        cache = DynamicCache(config=config)
        next_prefix = reference.hf_forward(hf, prefix, cache, 0)
        next_last = reference.hf_forward(hf, last, cache, args.length - 1)
        assert isinstance(next_prefix, torch.Tensor) and isinstance(next_last, torch.Tensor)
        assert next_prefix.shape == prefix.shape and next_last.shape == last.shape
        prefix, last = next_prefix, next_last
        upstream.append({"layer": layer_idx, "kind": config.layer_types[layer_idx]})
        del next_prefix, next_last, cache, hf
        gc.collect()
        print(f"HF_UPSTREAM_LAYER_DONE {layer_idx} elapsed_s={time.monotonic() - start_time:.3f}", flush=True)
    assert "ttnn" not in sys.modules, "HF input preparation unexpectedly imported TTNN"
    metadata = {
        "schema_version": 1,
        "purpose": "decoder-only real-activation precision probe; no model-quality or performance verdict",
        "model": reference.MODEL,
        "checkpoint_revision": reference.REVISION,
        "snapshot": str(reference.SNAPSHOT),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors")},
        "command": sys.argv,
        "prompt_format": "raw natural-text teacher forcing; no chat template or special tokens",
        "chat_template_applied": False,
        "generation_performed": False,
        "text_source": str(args.text_file.resolve()) if args.text_file else "embedded original natural prose",
        "source_text": text,
        "selected_text_decoded": tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False),
        "input_ids": input_ids[0].tolist(),
        "total_token_count": args.length,
        "prefill_length": args.length - 1,
        "held_out_decode_position": args.length - 1,
        "held_out_token_id": int(input_ids[0, -1]),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_local_files_only": True,
        "embedding": embedding_metadata,
        "layers": layers,
        "layer_kinds": {str(idx): config.layer_types[idx] for idx in layers},
        "propagated_upstream_layers": upstream,
        "propagation": "At each upstream layer: prefix with fresh per-layer cache, then held-out token with that cache",
        "attention_implementation": config._attn_implementation,
        "dtype": str(prefix.dtype),
        "device": "CPU",
        "cpu_threads": args.threads,
        "source_files": [str(Path(__file__).resolve()), str(Path(reference.__file__).resolve())],
        "hf_forward_source": inspect.getsourcefile(reference.Qwen3_5MoeDecoderLayer),
        "ttnn_imported": False,
        "elapsed_seconds": time.monotonic() - start_time,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_ids": input_ids, "layers": captured, "metadata": metadata}, output)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"HF_ACTIVATIONS_READY {output} layers={layers} prefix={args.length - 1}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--length", type=int, default=129, help="Total text tokens, including one held-out token")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 3, 34])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--text-file", type=Path, help="Optional UTF-8 natural text; no chat formatting is applied")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
