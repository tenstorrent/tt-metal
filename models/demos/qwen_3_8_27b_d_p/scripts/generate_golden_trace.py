#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generate the golden trace: the CPU reference forward, run **once** and saved.

Output layout follows the shared-store convention ``<weights>/golden/<prompt>_<isl>/``:

    <out>/
      metadata.json                    {"token_ids": [...], ...} — the exact input tokens
      kv_cache/layer_N.safetensors     the reference carried state for layer N
      final_hidden.safetensors         the post-final-norm hidden states (the e2e artifact)

What a layer file holds depends on the layer type, and a hybrid model needs both:

* **full_attention** (16 layers): ``key_cache_layer_N`` / ``value_cache_layer_N``, each
  ``[1, num_kv_heads, isl, head_dim]`` — post-RoPE K and raw V, the usual graded artifact.
* **linear_attention** (48 layers): ``recurrent_state_layer_N`` ``[1, num_v_heads, dk, dv]`` and
  ``conv_state_layer_N`` ``[1, kernel-1, conv_dim]`` — the Gated DeltaNet's carried state, which
  is what "the KV cache" *means* for those layers. Grading only the 16 attention layers would
  leave three quarters of the model unmeasured.

The reference computes in fp16 (recipe section 4) and the trace is written fp16. It needs real
weights: this refuses to run without a safetensors checkpoint, because a trace from synthetic
weights proves only that two random-valued pipelines agree.

    python3 models/demos/qwen_3_8_27b_d_p/scripts/generate_golden_trace.py \\
        --out $QWEN35_GOLDEN_ROOT/longbook_5120 --isl 5120
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from models.demos.qwen_3_8_27b_d_p.reference.config import Qwen35TextConfig  # noqa: E402
from models.demos.qwen_3_8_27b_d_p.reference.modeling import (  # noqa: E402
    REF_DTYPE,
    AttentionCapture,
    GdnCapture,
    Qwen35RotaryEmbedding,
    Qwen35TextModel,
)
from models.demos.qwen_3_8_27b_d_p.tt.weights import (  # noqa: E402
    assert_unquantized,
    load_text_backbone_state_dict,
    resolve_checkpoint_path,
)

DEFAULT_PROMPT = (
    "The history of computing hardware spans the development of machines that perform "
    "calculations, from mechanical aids through electronic digital computers. "
)


def _raise_cpu_time_limit() -> None:
    """RLIMIT_CPU counts CPU-seconds summed over every thread, so a 64-thread matmul drains a
    24-CPU-hour budget in ~22 wall-minutes and the run dies with SIGXCPU mid-forward — which looks
    like a crash rather than a limit. Raise the soft limit to the hard one (allowed unprivileged)."""
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))
        except (ValueError, OSError) as e:
            print(f"[limit] WARNING: could not raise RLIMIT_CPU (soft={soft}s): {e}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="trace directory to create")
    ap.add_argument("--isl", type=int, required=True, help="input sequence length in tokens")
    ap.add_argument("--model-path", type=str, default=None, help="checkpoint dir (default: resolved)")
    ap.add_argument("--prompt", type=str, default=None, help="prompt text (default: a built-in paragraph)")
    ap.add_argument("--prompt-json", type=Path, default=None, help='JSON {"prompt": "..."}')
    ap.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="REDUCED depth — a diagnostic, never a grade. Recorded in metadata.json as reduced=true.",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Generate the trace by chunked forward instead of one-shot. The result must be "
        "identical either way; used to check the reference's own chunking, not the device's.",
    )
    ap.add_argument("--threads", type=int, default=None)
    return ap.parse_args()


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_json:
        data = json.loads(args.prompt_json.read_text())
        return data["prompt"] if isinstance(data, dict) else data
    return args.prompt or DEFAULT_PROMPT


def tokenize(model_path: Path, prompt: str, isl: int) -> list[int]:
    """Tile the prompt up to exactly ``isl`` real tokens — no padding, so every graded position
    carries signal."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    ids: list[int] = []
    while len(ids) < isl:
        ids.extend(tok(prompt)["input_ids"])
    return ids[:isl]


def main() -> int:
    args = parse_args()
    _raise_cpu_time_limit()
    torch.set_num_threads(args.threads or (torch.get_num_threads()))
    torch.set_grad_enabled(False)

    model_path = Path(args.model_path) if args.model_path else resolve_checkpoint_path()
    assert_unquantized(model_path)
    print(f"[weights] REAL checkpoint: {model_path}", flush=True)

    cfg = Qwen35TextConfig.from_json()
    if args.num_layers:
        cfg = cfg.reduced(args.num_layers)
        print(f"[reduced] depth {args.num_layers}/64 — this trace is a DIAGNOSTIC, not a grade", flush=True)

    token_ids = tokenize(model_path, load_prompt(args), args.isl)
    print(f"[tokens] {len(token_ids)} tokens", flush=True)

    t0 = time.time()
    layers = range(cfg.num_hidden_layers) if args.num_layers else None
    state_dict = load_text_backbone_state_dict(model_path, layers=layers, dtype=REF_DTYPE)
    # Build on the meta device and load with assign=True: constructing 27B fp16 parameters for
    # real would run nn.Linear's kaiming init over every one of them and then immediately throw
    # the values away, at 54 GB of pointless writes.
    with torch.device("meta"):
        model = Qwen35TextModel(cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    assert not missing, f"checkpoint is missing reference parameters: {missing[:5]}"
    assert not unexpected, f"checkpoint has parameters the reference does not: {unexpected[:5]}"
    # inv_freq is a non-persistent buffer, so it is not in the state dict and is still on meta.
    model.rotary_emb = Qwen35RotaryEmbedding(cfg)
    model.eval()
    del state_dict
    print(f"[weights] reference built in {time.time() - t0:.0f}s", flush=True)

    out_dir = args.out
    kv_dir = out_dir / "kv_cache"
    kv_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(total=cfg.num_hidden_layers, desc="prefill (streaming per-layer state)", unit="layer")
    shapes: dict[str, list[int]] = {}

    def save_layer(idx: int, state, _hidden: torch.Tensor) -> None:
        if isinstance(state, AttentionCapture):
            tensors = {
                f"key_cache_layer_{idx}": state.key.to(REF_DTYPE).contiguous(),
                f"value_cache_layer_{idx}": state.value.to(REF_DTYPE).contiguous(),
            }
            shapes["key"] = list(state.key.shape)
        else:
            assert isinstance(state, GdnCapture)
            tensors = {
                f"recurrent_state_layer_{idx}": state.recurrent_state.to(REF_DTYPE).contiguous(),
                f"conv_state_layer_{idx}": state.conv_state.to(REF_DTYPE).contiguous(),
            }
            shapes["recurrent"] = list(state.recurrent_state.shape)
        save_file(tensors, str(kv_dir / f"layer_{idx}.safetensors"))
        progress.update(1)

    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    t0 = time.time()
    if args.chunk_size:
        assert args.isl % args.chunk_size == 0
        states = None
        outs = []
        for start in range(0, args.isl, args.chunk_size):
            hidden, states = model(
                input_ids=input_ids[:, start : start + args.chunk_size],
                start_pos=start,
                states=states,
                skip_lm_head=True,
                on_layer=save_layer if start + args.chunk_size >= args.isl else None,
            )
            outs.append(hidden)
        final_hidden = torch.cat(outs, dim=1)
    else:
        final_hidden, _states = model(input_ids=input_ids, skip_lm_head=True, on_layer=save_layer)
    progress.close()
    elapsed = time.time() - t0

    save_file({"final_hidden": final_hidden.to(REF_DTYPE).contiguous()}, str(out_dir / "final_hidden.safetensors"))
    metadata = {
        "model_path": str(model_path),
        "reference": "models.demos.qwen_3_8_27b_d_p.reference.modeling (fp16)",
        "token_ids": token_ids,
        "n_tokens": len(token_ids),
        "num_layers": cfg.num_hidden_layers,
        "reduced": bool(args.num_layers),
        "layer_types": list(cfg.layer_types),
        "num_kv_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "linear_num_value_heads": cfg.linear_num_value_heads,
        "linear_key_head_dim": cfg.linear_key_head_dim,
        "conv_kernel": cfg.linear_conv_kernel_dim,
        "dtype": "float16",
        "generated_chunked": bool(args.chunk_size),
        "forward_seconds": elapsed,
        "key_cache_shape": shapes.get("key"),
        "recurrent_state_shape": shapes.get("recurrent"),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\n[done] {out_dir}  ({elapsed / 60:.1f} min, {len(token_ids) / elapsed:.1f} tok/s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
