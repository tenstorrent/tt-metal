# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared harness for GLM-4.7-REAP e2e PCC + perf tests.

Sets the full production decode env config (including the knobs added during the
decode-optimization work) as DEFAULTS, then builds the TT model once per session
and exposes a `generate()` helper. Env vars use setdefault, so an outer environment
(e.g. the tt_hw_planner `optimize` run, or a manual A/B) always overrides them.
"""
from __future__ import annotations

import os

# --- Full production decode config, defaulted BEFORE any model import. -------------
# (setdefault: explicit env from the caller/tool wins.)
_ENV_DEFAULTS = {
    # reduce / CCL
    "GLM4_MOE_REDUCE_IMPL": "native",
    "GLM4_MOE_EP_REDUCE_DEVICE": "1",
    "GLM4_MOE_CCL_NUM_LINKS": "4",
    "GLM4_MOE_CCL_TOPOLOGY": "ring",
    # dtypes / precision
    "GLM4_MOE_EXPERTS_TT_DTYPE": "bf4",
    "GLM4_MOE_DISTRIBUTED_QK_NORM": "1",
    "GLM4_MOE_ROUTER_USE_BIASED_TOPK_VALUES": "1",
    # memory / sharding
    "GLM4_MOE_DRAM_SHARD": "1",
    "GLM4_MOE_PACKER_L1_ACC": "1",
    "GLM4_MOE_EP_L1": "1",
    "GLM4_MOE_SDPA_L1": "1",
    "GLM4_MOE_NORM_L1": "1",
    # prefill MoE chunking
    "GLM4_MOE_MOE_SPARSE_PREFILL_PCM": "1",
    "GLM4_MOE_MOE_SPARSE_PREFILL_CHUNK_TOKENS": "4096",
    # decode-winner fidelity + fusion (FUSE accuracy fix has landed)
    "GLM4_MOE_MOE_SPARSE_FIDELITY": "lofi",
    "GLM4_MOE_ATTN_FIDELITY": "lofi",
    "GLM4_MOE_FUSE_EXPERTS_GATE_UP": "1",
    # --- knobs added during decode-opt work (defaults == current production behavior) ---
    "GLM4_MOE_MOE_SPARSE_DOWN_FIDELITY": "lofi",  # == gate/up (no-op); set hifi2 to recover down precision
    "GLM4_MOE_MOE_RING_REDUCE": "0",  # validated-neutral ring TP reduce; opt-in
    "GLM4_MOE_MOE_SPARSE_BLOCK_SIZE": "32",  # sparse token tile height (tile-locked)
    "GLM4_MOE_MOE_SPARSE_PER_CORE_M": "1",  # decode single token-block
}
for _k, _v in _ENV_DEFAULTS.items():
    os.environ.setdefault(_k, _v)

# Test-tunable knobs (env-overridable), not model env vars.
DEFAULT_MODEL_ID = os.environ.get("GLM4_MOE_HF_MODEL") or "cerebras/GLM-4.7-REAP-218B-A32B"
DEFAULT_PROMPT = (
    "Explain in two short paragraphs how mixture-of-experts models route tokens to "
    "experts, and why that can improve efficiency."
)
DEFAULT_MESH_ROWS = int(os.environ.get("GLM4_MOE_TEST_MESH_ROWS", "8"))
DEFAULT_MESH_COLS = int(os.environ.get("GLM4_MOE_TEST_MESH_COLS", "4"))
# max_new used to size the KV cache at build time (perf uses up to this many).
BUILD_MAX_NEW = int(os.environ.get("GLM4_MOE_TEST_BUILD_MAX_NEW", "128"))
MIN_CACHE_TOKENS = int(os.environ.get("GLM4_MOE_TEST_MIN_CACHE_TOKENS", "256"))
BLOCK_SIZE = int(os.environ.get("GLM4_MOE_TEST_BLOCK_SIZE", "64"))
KV_CACHE_DTYPE = os.environ.get("GLM4_MOE_TEST_KV_DTYPE", "bf8")

import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402


class Glm4Harness:
    """Holds a built TT model + mesh; `generate()` runs prefill+decode."""

    def __init__(self, mesh_device, runner, tokenizer, snap, prompt_ids, prompt_len, block_size, max_seq_len):
        self.mesh_device = mesh_device
        self.runner = runner
        self.tokenizer = tokenizer
        self.snap = snap
        self._prompt_ids_single = prompt_ids  # [1, S]
        self.prompt_len = prompt_len
        self.block_size = block_size
        self.max_seq_len = max_seq_len

    def generate(self, max_new_tokens: int, *, enable_trace: bool, sampling: bool, warmup: bool = True) -> dict:
        import ttnn

        from models.experimental.glm4_moe.scripts.debug_run_full_tt_greedy import (
            _alloc_contiguous_page_table,
            _alloc_gqa_kv_cache,
            _parse_tt_dtype,
        )

        runner = self.runner
        dev = self.mesh_device
        block_size = self.block_size
        batch = 1
        prompt_ids = self._prompt_ids_single.repeat(batch, 1)
        prompt_lens = [self.prompt_len] * batch
        blocks_per_seq = max(1, ((self.max_seq_len + block_size - 1) // block_size))
        num_blocks = batch * blocks_per_seq

        kv_cache = _alloc_gqa_kv_cache(
            device=dev,
            num_key_value_heads=int(runner.hparams.num_key_value_heads),
            head_dim=int(runner.hparams.head_dim),
            num_layers=int(runner.num_layers_to_run),
            num_blocks=num_blocks,
            block_size=block_size,
            tt_dtype=_parse_tt_dtype(KV_CACHE_DTYPE),
        )
        page_table = _alloc_contiguous_page_table(batch=batch, blocks_per_seq=blocks_per_seq)

        use_sampling = enable_trace and sampling
        t_pf = time.perf_counter()
        logits = runner.prefill(
            tokens=prompt_ids,
            prompt_lens=prompt_lens,
            page_table=page_table,
            kv_cache=kv_cache,
            seq_pad_multiple=block_size,
        )
        prefill_s = time.perf_counter() - t_pf
        logits_flat = logits.reshape(batch, -1)
        first = int(torch.argmax(logits_flat[0]).item())
        generated = [first]

        if warmup and enable_trace:
            _ = runner.decode(
                tokens=torch.tensor([[first]], dtype=torch.int32),
                start_pos=torch.tensor([self.prompt_len], dtype=torch.int32),
                page_table=page_table,
                kv_cache=kv_cache,
                enable_trace=True,
                sampling_params=True if use_sampling else None,
            )

        step_ms: list[float] = []
        cur = first
        for step in range(max_new_tokens - 1):
            pos = torch.tensor([self.prompt_len + step], dtype=torch.int32)
            tok_in = torch.tensor([[cur]], dtype=torch.int32)
            t0 = time.perf_counter()
            out = runner.decode(
                tokens=tok_in,
                start_pos=pos,
                page_table=page_table,
                kv_cache=kv_cache,
                enable_trace=enable_trace,
                sampling_params=True if use_sampling else None,
            )
            if use_sampling:
                cur = int(out.reshape(-1).to(torch.int32).cpu()[0].item())
            else:
                cur = int(torch.argmax(out.reshape(batch, -1)[0]).item())
            step_ms.append((time.perf_counter() - t0) * 1000.0)
            generated.append(cur)

        for pair in kv_cache:
            for t in pair:
                ttnn.deallocate(t, force=False)

        rest = step_ms[1:] if len(step_ms) >= 2 else step_ms
        text = self.tokenizer.decode(
            torch.cat([self._prompt_ids_single[0], torch.tensor(generated, dtype=torch.int32)]).tolist(),
            skip_special_tokens=True,
        )
        return {
            "generated": generated,
            "prefill_s": prefill_s,
            "decode_mean_ms": statistics.mean(rest) if rest else float("nan"),
            "decode_min_ms": min(rest) if rest else float("nan"),
            "decode_max_ms": max(rest) if rest else float("nan"),
            "text": text,
        }


@pytest.fixture(scope="session")
def glm4_model():
    """Build the GLM-4.7-REAP TT model once for the session (heavy: 218B weights)."""
    import ttnn
    from transformers import AutoTokenizer

    from models.experimental.glm4_moe.scripts.debug_run_full_tt_greedy import (
        _round_up,
        _set_default_fabric_config,
        _resolve_snapshot,
    )
    from models.experimental.glm4_moe.tt.model_tt import Glm4MoeTT
    from models.experimental.glm4_moe.tt.weights import find_missing_shards

    model_id = DEFAULT_MODEL_ID
    os.environ.setdefault("HF_MODEL", model_id)
    os.environ.setdefault("GLM4_MOE_HF_MODEL", model_id)
    snap = _resolve_snapshot(model_id)
    missing = find_missing_shards(snap)
    if missing:
        pytest.skip(f"weights missing {len(missing)} shards (e.g. {missing[0]}); download first")

    tokenizer = AutoTokenizer.from_pretrained(str(snap), local_files_only=True, use_fast=True)
    enc = tokenizer(DEFAULT_PROMPT, return_tensors="pt", add_special_tokens=True)
    prompt_ids = enc["input_ids"].to(dtype=torch.int32)
    prompt_len = int(prompt_ids.shape[1])

    total = max(prompt_len + BUILD_MAX_NEW, MIN_CACHE_TOKENS, _round_up(prompt_len, 128))
    blocks_per_seq = max(1, _round_up(total, BLOCK_SIZE) // BLOCK_SIZE)
    max_seq_len = int(blocks_per_seq * BLOCK_SIZE)

    n_dev = DEFAULT_MESH_ROWS * DEFAULT_MESH_COLS
    _set_default_fabric_config(n_dev)
    is_galaxy = ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY
    dispatch_cfg = (
        ttnn.DispatchCoreConfig(axis=ttnn.DispatchCoreAxis.ROW)
        if is_galaxy
        else ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH)
    )
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(DEFAULT_MESH_ROWS, DEFAULT_MESH_COLS),
        dispatch_core_config=dispatch_cfg,
    )
    try:
        # Reuse the debug greedy runner's warm weight-conversion cache to avoid a slow
        # from-scratch reconvert on every session (override via GLM4_MOE_TEST_CACHE_DIR).
        cache_dir = Path(
            os.path.expanduser(os.environ.get("GLM4_MOE_TEST_CACHE_DIR", "~/.cache/ttnn/models/glm4_moe/debug_greedy"))
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        runner = Glm4MoeTT.create(
            device=mesh_device,
            snapshot_dir=snap,
            cache_dir=cache_dir,
            max_seq_len=max_seq_len,
            max_batch_size=32,
        )
        yield Glm4Harness(mesh_device, runner, tokenizer, snap, prompt_ids, prompt_len, BLOCK_SIZE, max_seq_len)
    finally:
        ttnn.close_mesh_device(mesh_device)
