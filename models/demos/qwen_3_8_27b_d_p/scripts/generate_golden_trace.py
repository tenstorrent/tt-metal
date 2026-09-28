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

The reference computes in fp16 (recipe section 4). The trace's STORAGE dtype is separate and set
by ``--trace-dtype`` — fp16 by default for ``--backend reference``, bf16 for ``--backend hf``.
Storing bf16 keeps 8 mantissa bits against fp16's 11, so it is a coarser golden; it also rounds the
oracle toward the device's own bf16, which raises measured PCC without anything having improved.

It needs real weights: this refuses to run without a safetensors checkpoint, because a trace from
synthetic weights proves only that two random-valued pipelines agree.

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
    ap.add_argument(
        "--backend",
        choices=("reference", "hf"),
        default="reference",
        help="'reference' (default): this package's vendored fp16 reference. 'hf': upstream "
        "transformers Qwen3_5TextModel, bypassing reference/modeling.py entirely.",
    )
    ap.add_argument(
        "--trace-dtype",
        choices=("float16", "bfloat16"),
        default=None,
        help="dtype the trace is WRITTEN in, independent of the compute dtype. Defaults to "
        "bfloat16 for --backend hf and float16 for --backend reference (recipe section 4). "
        "Note bfloat16 has 8 mantissa bits against float16's 11, so it stores a coarser golden.",
    )
    ap.add_argument(
        "--hf-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="--backend hf only, the COMPUTE dtype. NOT fp16: torch has no vectorised fp16 GEMM on "
        "x86, so a literal fp16 HF forward is ~3500x slower. Storage is --trace-dtype.",
    )
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


def build_hf_backend(cfg: Qwen35TextConfig, model_path: Path, dtype: torch.dtype):
    """Upstream ``Qwen3_5TextModel`` on the real checkpoint — the reference-free variant.

    Same weight loader as the reference path, because the reference was *trimmed* from upstream
    rather than rewritten: the key names match one-for-one, so one state dict feeds both.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel, Qwen3_5TextRotaryEmbedding

    from models.demos.qwen_3_8_27b_d_p.tests.host.hf_bridge import to_hf_text_config

    hf_cfg = to_hf_text_config(cfg)
    hf_cfg._attn_implementation = "eager"
    hf_cfg.dtype = dtype

    with torch.device("meta"):
        model = Qwen3_5TextModel(hf_cfg)
    state_dict = load_text_backbone_state_dict(model_path, dtype=dtype)
    state_dict.pop("lm_head.weight", None)  # Qwen3_5TextModel is the backbone, no head
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    assert not missing, f"HF wants parameters the checkpoint lacks: {missing[:5]}"
    assert not unexpected, f"checkpoint has parameters HF does not: {unexpected[:5]}"
    # Same meta trap as the reference path: inv_freq is a non-persistent buffer.
    model.rotary_emb = Qwen3_5TextRotaryEmbedding(hf_cfg)
    return model.eval()


def forward_hf(model, cfg: Qwen35TextConfig, input_ids: torch.Tensor, save_layer) -> torch.Tensor:
    """One-shot HF forward, re-emitting its cache as the same captures ``save_layer`` expects.

    Not streamed: HF returns the whole cache at once, so peak memory holds all 64 layers rather
    than one. Chunked generation is not supported here — use ``--backend reference`` for that.
    """
    out = model(input_ids=input_ids, use_cache=True)
    keep = cfg.linear_conv_kernel_dim - 1
    for idx in range(cfg.num_hidden_layers):
        layer = out.past_key_values.layers[idx]
        if cfg.is_full_attention(idx):
            state = AttentionCapture(key=layer.keys, value=layer.values)
        else:
            # Upstream caches `kernel` columns and keeps kernel-1 of them; the reference stores the
            # minimal kernel-1 directly (see reference/modeling.py next_conv_state).
            state = GdnCapture(
                conv_state=layer.conv_states[..., -keep:].contiguous(),
                recurrent_state=layer.recurrent_states.reshape(
                    1, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
                ),
            )
        save_layer(idx, state, None)
    return out.last_hidden_state


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
    if args.backend == "hf":
        assert not args.chunk_size, "--backend hf is one-shot only; use --backend reference"
        assert not args.num_layers, "--backend hf does not support --num-layers"
        model = build_hf_backend(cfg, model_path, getattr(torch, args.hf_dtype))
    else:
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
    print(f"[weights] {args.backend} built in {time.time() - t0:.0f}s", flush=True)

    # The trace's storage dtype is independent of what the forward computed in.
    trace_dtype_name = args.trace_dtype or ("bfloat16" if args.backend == "hf" else "float16")
    trace_dtype = getattr(torch, trace_dtype_name)
    print(f"[trace] writing {trace_dtype_name}", flush=True)

    out_dir = args.out
    kv_dir = out_dir / "kv_cache"
    kv_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(total=cfg.num_hidden_layers, desc="prefill (streaming per-layer state)", unit="layer")
    shapes: dict[str, list[int]] = {}

    def save_layer(idx: int, state, _hidden: torch.Tensor) -> None:
        if isinstance(state, AttentionCapture):
            tensors = {
                f"key_cache_layer_{idx}": state.key.to(trace_dtype).contiguous(),
                f"value_cache_layer_{idx}": state.value.to(trace_dtype).contiguous(),
            }
            shapes["key"] = list(state.key.shape)
        else:
            assert isinstance(state, GdnCapture)
            tensors = {
                f"recurrent_state_layer_{idx}": state.recurrent_state.to(trace_dtype).contiguous(),
                f"conv_state_layer_{idx}": state.conv_state.to(trace_dtype).contiguous(),
            }
            shapes["recurrent"] = list(state.recurrent_state.shape)
        save_file(tensors, str(kv_dir / f"layer_{idx}.safetensors"))
        progress.update(1)

    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    t0 = time.time()
    if args.backend == "hf":
        final_hidden = forward_hf(model, cfg, input_ids, save_layer)
    elif args.chunk_size:
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

    save_file({"final_hidden": final_hidden.to(trace_dtype).contiguous()}, str(out_dir / "final_hidden.safetensors"))
    metadata = {
        "model_path": str(model_path),
        "backend": args.backend,
        "reference": (
            f"transformers Qwen3_5TextModel ({args.hf_dtype} compute, {trace_dtype_name} trace)"
            if args.backend == "hf"
            else "models.demos.qwen_3_8_27b_d_p.reference.modeling (fp16)"
        ),
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
        "dtype": trace_dtype_name,
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
