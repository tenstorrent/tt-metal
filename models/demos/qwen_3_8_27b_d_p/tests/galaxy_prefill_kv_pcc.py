#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P1/P2: per-layer carried state after a real-weights prefill, against the CPU golden trace.

The graded artifact. Both stages run this script; the only difference is whether the prompt goes
through in one shot or in chunks:

    PREFILL_CHUNKED=0 PREFILL_TRACE_DIR=.../longbook_5120  python3 .../galaxy_prefill_kv_pcc.py
    PREFILL_CHUNKED=1 PREFILL_TRACE_DIR=.../longbook_10240 python3 .../galaxy_prefill_kv_pcc.py

**Every layer is graded, not just the 16 with a KV cache.** For a full-attention layer that means
the usual post-RoPE K and raw V. For a Gated DeltaNet layer it means the recurrent state and the
conv history — the state a later chunk continues from, and the only thing that makes those 48
layers observable at all. Grading K/V alone would leave three quarters of this model unmeasured
and every number in the table misleading about coverage.

The e2e final hidden states are compared too; KV/state PCC is a proxy for correctness, not a
substitute for the model's actual output.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import ttnn  # noqa: E402
from models.common.utility_functions import comp_pcc  # noqa: E402
from models.demos.qwen_3_8_27b_d_p.config import MeshConfig  # noqa: E402
from models.demos.qwen_3_8_27b_d_p.reference.config import Qwen35TextConfig  # noqa: E402
from models.demos.qwen_3_8_27b_d_p.spec import load_spec, ttnn_dtype  # noqa: E402
from models.demos.qwen_3_8_27b_d_p.tt.caches import PrefillCaches  # noqa: E402
from models.demos.qwen_3_8_27b_d_p.tt.ccl import CCLManager  # noqa: E402
from models.demos.qwen_3_8_27b_d_p.tt.gdn.weights import device_conv_state_to_hf  # noqa: E402
from models.demos.qwen_3_8_27b_d_p.tt.model import Qwen35Model  # noqa: E402
from models.demos.qwen_3_8_27b_d_p.tt.tt_prefill_runtime import PrefillRuntimeConfig, TtPrefillRuntime  # noqa: E402
from models.demos.qwen_3_8_27b_d_p.tt.weights import (  # noqa: E402
    assert_unquantized,
    load_text_backbone_state_dict,
    resolve_checkpoint_path,
)
from models.demos.qwen_3_8_27b_d_p.utils.general_utils import get_default_num_links  # noqa: E402

SPEC = load_spec()


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def read_kv_from_cache(
    caches: PrefillCaches,
    cfg: Qwen35TextConfig,
    mesh_config: MeshConfig,
    layer_idx: int,
    isl: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose a full-attention layer's on-device K/V back into ``[1, n_kv, isl, head_dim]``.

    Two layout facts have to be undone, and getting the second wrong is what a scrambled-token PCC
    of ~0.44 looks like:

    1. The packed cache is an ``NdShard``/``ROUND_ROBIN_1D`` tensor. ``ttnn.slice`` on it corrupts
       the round-robin bank mapping, so it is converted to plain DRAM-interleaved first.
    2. **The sequence is block-cyclic per chunk, not contiguous per row.** Within one chunk row
       ``r`` owns the contiguous token block ``[r*chunk_local, (r+1)*chunk_local)``, and each new
       chunk appends another such block below it. So a row's local rows read
       ``chunk 0's block, chunk 1's block, ...`` — concatenating rows end to end is only correct
       for a single chunk, and interleaves chunk boundaries into the middle of the sequence for
       any more than that.
    """
    kv = caches.kv
    slot = kv.slot(0, cfg.kv_slot(layer_idx))
    sp, tp = mesh_config.sp, mesh_config.tp
    chunk_local = chunk_size // sp
    n_chunks = isl // chunk_size
    n_kv = cfg.num_key_value_heads
    kv_local = n_kv // tp

    out = []
    for cache in (kv.k, kv.v):
        interleaved = ttnn.to_memory_config(cache, ttnn.DRAM_MEMORY_CONFIG)
        dev = ttnn.get_device_tensors(interleaved)
        composed = torch.zeros(1, n_kv, isl, cfg.head_dim)
        for c in range(tp):
            for r in range(sp):
                shard = ttnn.to_torch(dev[r * tp + c])[slot : slot + 1]  # [1, kv_local, seq_local, d]
                for chunk in range(n_chunks):
                    local = slice(chunk * chunk_local, (chunk + 1) * chunk_local)
                    start = chunk * chunk_size + r * chunk_local
                    composed[:, c * kv_local : (c + 1) * kv_local, start : start + chunk_local] = shard[
                        :, :, local, :
                    ].float()
        out.append(composed)
        interleaved.deallocate(True)
    return out[0], out[1]


def read_gdn_state(
    caches: PrefillCaches, cfg: Qwen35TextConfig, mesh_config: MeshConfig, layer_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose a Gated DeltaNet layer's recurrent + conv state. Both are replicated across SP (every
    row runs the same full-chunk scan) and value-head sharded across TP, so row 0's columns are the
    whole thing."""
    state = caches.gdn[layer_idx]
    hv_local = cfg.linear_num_value_heads // mesh_config.tp
    rec_dev = ttnn.get_device_tensors(state.recurrent)
    recurrent = torch.cat(
        [
            ttnn.to_torch(rec_dev[c]).reshape(1, hv_local, cfg.linear_key_head_dim, cfg.linear_value_head_dim)
            for c in range(mesh_config.tp)
        ],
        dim=1,
    )
    conv_dev = ttnn.get_device_tensors(state.conv_state)
    conv = torch.cat([ttnn.to_torch(conv_dev[c]) for c in range(mesh_config.tp)], dim=-1)
    conv = conv.reshape(1, cfg.linear_conv_kernel_dim - 1, cfg.gdn_conv_dim)
    # -> [1, conv_dim, kernel-1] in checkpoint channel order; see device_conv_state_to_hf.
    return device_conv_state_to_hf(conv, cfg, mesh_config.tp), recurrent


def main() -> int:
    trace_dir = os.environ.get("PREFILL_TRACE_DIR")
    if not trace_dir:
        print("ERROR: set PREFILL_TRACE_DIR to a golden trace directory", file=sys.stderr)
        return 3
    trace = Path(trace_dir)
    metadata = json.loads((trace / "metadata.json").read_text())
    token_ids = torch.tensor(metadata["token_ids"], dtype=torch.int64).reshape(1, -1)
    isl = token_ids.numel()
    chunked = os.environ.get("PREFILL_CHUNKED", "0") == "1"
    chunk_size = _env_int("PREFILL_CHUNK_SIZE", SPEC.chunk_size) if chunked else isl
    num_layers = _env_int("PREFILL_NUM_LAYERS", metadata["num_layers"])

    cfg = Qwen35TextConfig.from_json()
    reduced = num_layers != cfg.num_hidden_layers
    if reduced:
        cfg = cfg.reduced(num_layers)
    assert num_layers == metadata["num_layers"], (
        f"the trace has {metadata['num_layers']} layers but this run asks for {num_layers}; "
        f"a golden trace is per (model, prompt, ISL, depth)"
    )
    if reduced or metadata.get("reduced"):
        logger.warning("REDUCED RUN: every number below is a diagnostic, not a grade (recipe section 4)")

    # DIAGNOSTIC ONLY. The spec binds activations to bfloat16; this override exists to attribute
    # residual-stream drift to the activation dtype rather than assert it. Any run that uses it is
    # labelled a spec deviation in the log line below and is not a grade.
    activation_dtype = os.environ.get("QWEN35_ACTIVATION_DTYPE", SPEC.activation_dtype)
    weight_dtype = os.environ.get("QWEN35_WEIGHT_DTYPE", SPEC.weight_dtype)
    kv_cache_dtype = os.environ.get("QWEN35_KV_CACHE_DTYPE", SPEC.kv_cache_dtype)
    for name, used, spec_value in (
        ("activations", activation_dtype, SPEC.activation_dtype),
        ("weights", weight_dtype, SPEC.weight_dtype),
        ("kv_cache", kv_cache_dtype, SPEC.kv_cache_dtype),
    ):
        if used != spec_value:
            logger.warning(
                f"SPEC DEVIATION (diagnostic): {name} {used}, spec says {spec_value}. " f"This run is NOT a grade."
            )

    rows, cols = SPEC.mesh_shape
    logger.info(
        f"{'CHUNKED' if chunked else 'ONE-SHOT'} prefill: isl={isl} chunk={chunk_size} layers={num_layers} "
        f"mesh={rows}x{cols} trace={trace}"
    )

    model_path = resolve_checkpoint_path()
    assert_unquantized(model_path)
    t0 = time.time()
    state_dict = load_text_backbone_state_dict(
        model_path, layers=range(num_layers) if reduced else None, dtype=torch.bfloat16
    )
    logger.info(f"checkpoint loaded in {time.time() - t0:.0f}s")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    failures: list[str] = []
    try:
        mesh_config = MeshConfig((rows, cols), tp=cols)
        ccl = CCLManager(mesh, num_links=get_default_num_links(mesh), topology=ttnn.Topology.Linear)
        t0 = time.time()
        model = Qwen35Model(
            mesh,
            cfg,
            state_dict,
            mesh_config=mesh_config,
            ccl_manager=ccl,
            weight_dtype=ttnn_dtype(weight_dtype),
            activation_dtype=ttnn_dtype(activation_dtype),
            cache_dtype=ttnn_dtype(kv_cache_dtype),
            tensor_cache_path=os.environ.get("TT_CACHE_PATH"),
        )
        del state_dict
        logger.info(f"device model built in {time.time() - t0:.0f}s")

        runtime = TtPrefillRuntime(
            mesh,
            model,
            cfg,
            mesh_config=mesh_config,
            config=PrefillRuntimeConfig(chunk_size=chunk_size, max_seq_len=isl, num_users=1),
        )
        caches = model.allocate_caches(max_seq_len=isl)

        t0 = time.time()
        outputs = runtime.prefill_sequence(token_ids, caches, skip_lm_head=True)
        ttnn.synchronize_device(mesh)
        elapsed = time.time() - t0
        logger.info(f"prefill: {isl} tokens in {elapsed:.1f}s ({isl / elapsed:.0f} tok/s)")

        lower, target = SPEC.acceptance.pcc_lower_bound, SPEC.acceptance.pcc_target
        worst = 1.0
        for layer_idx in range(num_layers):
            golden = load_file(str(trace / "kv_cache" / f"layer_{layer_idx}.safetensors"))
            if cfg.is_full_attention(layer_idx):
                k, v = read_kv_from_cache(caches, cfg, mesh_config, layer_idx, isl, chunk_size)
                pairs = [
                    ("k", golden[f"key_cache_layer_{layer_idx}"], k),
                    ("v", golden[f"value_cache_layer_{layer_idx}"], v),
                ]
            else:
                conv, recurrent = read_gdn_state(caches, cfg, mesh_config, layer_idx)
                pairs = [
                    ("recurrent", golden[f"recurrent_state_layer_{layer_idx}"], recurrent),
                    ("conv", golden[f"conv_state_layer_{layer_idx}"], conv),
                ]
            for name, expected, got in pairs:
                assert tuple(got.shape) == tuple(
                    expected.shape
                ), f"layer {layer_idx} {name}: device {tuple(got.shape)} != golden {tuple(expected.shape)}"
                ok, pcc = comp_pcc(expected.float(), got.float(), lower)
                worst = min(worst, float(pcc))
                flag = "" if ok else "  <-- BELOW LOWER BOUND"
                logger.info(f"layer {layer_idx:2d} {cfg.layer_types[layer_idx][:6]} {name:9s} PCC {pcc}{flag}")
                if not ok:
                    failures.append(f"layer {layer_idx} {name}: {pcc}")

        # e2e: the model's actual output, not a proxy for it.
        final_golden = load_file(str(trace / "final_hidden.safetensors"))["final_hidden"]
        pieces = []
        for out in outputs:
            dev = ttnn.get_device_tensors(out)
            pieces.append(torch.cat([ttnn.to_torch(dev[r * cols]) for r in range(rows)], dim=2))
        final_device = torch.cat(pieces, dim=2).reshape(final_golden.shape)
        ok, pcc = comp_pcc(final_golden.float(), final_device.float(), lower)
        logger.info(f"e2e final hidden PCC {pcc}{'' if ok else '  <-- BELOW LOWER BOUND'}")
        if not ok:
            failures.append(f"e2e: {pcc}")

        logger.info(
            f"worst per-layer PCC across {num_layers} layers: {worst} " f"(lower bound {lower}, target {target})"
        )
    finally:
        ttnn.close_mesh_device(mesh)

    if failures:
        logger.error(f"{len(failures)} comparison(s) below the spec's lower bound:")
        for f in failures[:20]:
            logger.error(f"  {f}")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
