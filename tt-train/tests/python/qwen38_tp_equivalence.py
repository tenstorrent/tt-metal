# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TP=1 vs TP=4 equivalence for the Qwen3.8 backbone.

Tensor parallelism must be a pure implementation detail: the same checkpoint
loaded onto one chip and onto four has to produce the same logits.  This is the
test that actually pins down the sharding, and in particular the fused-QKV row
permutation -- get that wrong and each chip mixes another chip's Q with its own
K, which still runs and still trains, just on a different model.

Run as two processes (one mesh per process), then compare:

    ./ttenv.sh tests/python/qwen38_tp_equivalence.py --tp 1 --out /tmp/tp1.npz
    ./ttenv.sh tests/python/qwen38_tp_equivalence.py --tp 4 --out /tmp/tp4.npz
    ./ttenv.sh tests/python/qwen38_tp_equivalence.py --compare /tmp/tp1.npz /tmp/tp4.npz

The checkpoint is synthetic but laid out exactly like the real one (HF names,
HF shapes, fused QKV, ``[C, 1, K]`` conv) so the loader is exercised for real.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ttml.models.qwen38.parallel import qkv_shard_permutation

# Small enough to run in seconds, but every head count still divides 4 so the
# TP=4 path is fully exercised: 4 key / 12 value / 8 query / 4 KV heads.
TINY = dict(
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=4,
    vocab_size=1024,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=128,
    linear_num_key_heads=4,
    linear_num_value_heads=12,
    linear_key_head_dim=64,
    linear_value_head_dim=64,
    delta_chunk_size=32,
    max_position_embeddings=512,
)

BATCH, SEQ, SEED = 2, 64, 1234
_HF = "model.language_model."


def build_checkpoint(cfg, directory: Path) -> None:
    """Write a synthetic checkpoint with the real HF names and shapes."""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(SEED)

    def r(*shape, scale=0.02):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    hidden, inter = cfg.hidden_size, cfg.intermediate_size
    tensors = {
        "lm_head.weight": r(cfg.vocab_size, hidden),
        f"{_HF}embed_tokens.weight": r(cfg.vocab_size, hidden),
        f"{_HF}norm.weight": r(hidden),
    }

    for i in range(cfg.num_hidden_layers):
        p = f"{_HF}layers.{i}."
        tensors[p + "input_layernorm.weight"] = r(hidden)
        tensors[p + "post_attention_layernorm.weight"] = r(hidden)
        tensors[p + "mlp.gate_proj.weight"] = r(inter, hidden)
        tensors[p + "mlp.up_proj.weight"] = r(inter, hidden)
        tensors[p + "mlp.down_proj.weight"] = r(hidden, inter)

        if cfg.is_full_attention(i):
            gate = 2 if cfg.attn_output_gate else 1
            tensors[p + "self_attn.q_proj.weight"] = r(cfg.num_attention_heads * cfg.head_dim * gate, hidden)
            tensors[p + "self_attn.k_proj.weight"] = r(cfg.num_key_value_heads * cfg.head_dim, hidden)
            tensors[p + "self_attn.v_proj.weight"] = r(cfg.num_key_value_heads * cfg.head_dim, hidden)
            tensors[p + "self_attn.o_proj.weight"] = r(hidden, cfg.num_attention_heads * cfg.head_dim)
            tensors[p + "self_attn.q_norm.weight"] = r(cfg.head_dim)
            tensors[p + "self_attn.k_norm.weight"] = r(cfg.head_dim)
        else:
            n_v = cfg.linear_num_value_heads
            tensors[p + "linear_attn.in_proj_qkv.weight"] = r(cfg.qkv_proj_dim, hidden)
            tensors[p + "linear_attn.in_proj_z.weight"] = r(cfg.value_proj_dim, hidden)
            tensors[p + "linear_attn.in_proj_a.weight"] = r(n_v, hidden)
            tensors[p + "linear_attn.in_proj_b.weight"] = r(n_v, hidden)
            tensors[p + "linear_attn.out_proj.weight"] = r(hidden, cfg.value_proj_dim)
            tensors[p + "linear_attn.conv1d.weight"] = r(cfg.qkv_proj_dim, 1, cfg.linear_conv_kernel_dim)
            tensors[p + "linear_attn.A_log"] = r(n_v, scale=0.5)
            tensors[p + "linear_attn.dt_bias"] = r(n_v, scale=0.5)
            tensors[p + "linear_attn.norm.weight"] = r(cfg.linear_value_head_dim)

    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / "model.safetensors"))


def check_qkv_shard_layout(model, cfg, checkpoint: Path, tp: int) -> bool:
    """Assert each chip physically holds the QKV rows the permutation promises.

    This is the load-bearing check, and it is deliberately not a comparison of
    logits.  At small weight scales the DeltaNet's contribution to the logits is
    swamped by the embedding and MLP path, so an end-to-end PCC stays high even
    with contiguous (wrong) sharding -- it cannot tell the two apart.  Reading
    the sharded weight back and comparing it against the expected permutation
    separates them by three orders of magnitude.
    """
    import ttnn
    import ttml
    from safetensors.numpy import load_file

    tensors = load_file(str(checkpoint / "model.safetensors"))
    ref = tensors[f"{_HF}layers.0.linear_attn.in_proj_qkv.weight"].astype(np.float32)
    perm = qkv_shard_permutation(cfg, tp)

    params = model.parameters()
    name = next(n for n in params if n.endswith("layers/0/linear_attn/in_proj_qkv/weight"))
    dev = ttml.autograd.AutoContext.get_instance().get_device()
    # Gather the shards along the output-feature dim; the DP axis just replicates.
    composer = ttnn.create_mesh_composer(dev, ttnn.MeshComposerConfig([0, 2]))
    got = np.split(params[name].tensor.to_numpy(composer=composer), 2, axis=0)[0]
    got = got.reshape(-1, cfg.hidden_size)

    permuted = float(np.abs(got - ref[perm]).max())
    contiguous = float(np.abs(got - ref).max())
    print(f"[tp={tp}] qkv shard vs permuted {permuted:.6f}, vs contiguous {contiguous:.6f}")
    ok = permuted < 0.1 * contiguous
    print(f"[tp={tp}] QKV_SHARD_LAYOUT", "OK" if ok else "WRONG")
    return ok


def run(tp: int, out: Path, checkpoint: Path) -> None:
    import ttnn
    import ttml
    from ttml.models.qwen38 import Qwen38Config, Qwen38Transformer
    from ttml.models.qwen38.loading import load_from_safetensors

    # The physical mesh here is 2x4 and a [1, 4] submesh is rejected, so the TP
    # run opens the full [2, 4]. Data parallelism is left idle by feeding the
    # same tokens to both DP groups; they then compute identical logits, and the
    # duplicate replica is dropped on read-back. So this still compares TP
    # against a single-chip baseline with nothing else varying.
    dp = 2 if tp > 1 else 1
    ttml.open_device_mesh(ttml.Mesh((dp, tp), ("dp", "tp")))
    if tp > 1:
        # A 2D mesh requires every axis to be an enabled parallelism axis, so DDP
        # is switched on even though it does nothing here: only a forward pass is
        # run, and DDP would only act at gradient-sync time.
        ttml.autograd.AutoContext.get_instance().initialize_parallelism_context(
            ttml.autograd.DistributedConfig(enable_ddp=True, enable_tp=True)
        )

    cfg = Qwen38Config(use_tp=(tp > 1), **TINY)
    if not checkpoint.exists():
        build_checkpoint(cfg, checkpoint)

    model = Qwen38Transformer(cfg)
    stats = load_from_safetensors(model, checkpoint, cfg)
    print(f"[tp={tp}] loaded {stats['loaded']} tensors, {len(stats['unexpected'])} unexpected")

    if tp > 1 and not check_qkv_shard_layout(model, cfg, checkpoint, tp):
        raise SystemExit(f"QKV rows are not sharded as qkv_shard_permutation specifies (tp={tp})")

    rng = np.random.default_rng(SEED)
    ids_np = rng.integers(0, cfg.vocab_size, (BATCH, 1, 1, SEQ)).astype(np.uint32)
    ids = ttml.autograd.Tensor.from_numpy(ids_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)

    logits = model(ids, None)
    dev = ttml.autograd.AutoContext.get_instance().get_device()
    # The vocab dim is sharded over the tp axis, so concatenate along it to
    # reassemble the full logits; at tp=1 this is a plain read-back.
    composer = ttnn.create_mesh_composer(dev, ttnn.MeshComposerConfig([0, 3]))
    full = logits.to_numpy(composer=composer)
    if dp > 1:
        # The DP axis concatenated identical replicas onto the batch dim.
        replicas = np.split(full, dp, axis=0)
        spread = max(float(np.abs(replicas[0] - r).max()) for r in replicas[1:])
        print(f"[tp={tp}] dp replicas agree to {spread:.6f}")
        full = replicas[0]
    print(f"[tp={tp}] logits {full.shape}")
    np.savez(out, logits=full)


def compare(a: Path, b: Path) -> int:
    x = np.load(a)["logits"].astype(np.float32)
    y = np.load(b)["logits"].astype(np.float32)
    if x.shape != y.shape:
        print(f"SHAPE_MISMATCH {x.shape} vs {y.shape}")
        return 1

    # bf16 activations, so exact equality is not expected; correlation is the
    # discriminating statistic. A wrong QKV permutation lands near 0, while
    # differing only in accumulation order stays above ~0.999.
    xf, yf = x.ravel(), y.ravel()
    pcc = float(np.corrcoef(xf, yf)[0, 1])
    denom = np.maximum(np.abs(xf).max(), 1e-6)
    print(f"SHAPE {x.shape}")
    print(f"PCC {pcc:.6f}")
    print(f"MAX_ABS_DIFF {np.abs(xf - yf).max():.6f}  (logit range +/-{denom:.3f})")
    print(f"REL_MAX_DIFF {np.abs(xf - yf).max() / denom:.6f}")
    ok = pcc > 0.999
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=Path("/tmp/qwen38_tiny_ckpt"))
    ap.add_argument("--compare", nargs=2, type=Path)
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)
    run(args.tp, args.out, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
