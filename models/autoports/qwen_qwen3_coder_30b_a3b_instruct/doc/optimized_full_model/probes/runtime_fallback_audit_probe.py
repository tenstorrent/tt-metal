# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The runtime fallback audit **for the path stage 06 actually measures**.

``Qwen3CoderModel.runtime_fallback_audit()`` is asserted field by field by
``test_runtime_fallback_audit_is_clean`` every run, so its contents are already
gated. What it does *not* carry is the two things stage 06 changed -- the paged
SDPA-decode program config and the sampler's live-row count -- because adding
fields to it would change a dict the test suite pins. This probe therefore
prints the audit **and** reads those two properties off the modules that own
them, so the published audit describes the shipped path rather than the
stage-05 one.

It also records, on the real 48-layer model:

* the SDPA program config the model builds **for its own KV cache**, including
  whether the depth clamp bound;
* that prefill is still at the op default (``sdpa_program_config=None``), i.e.
  the prefill lever is wired but unadopted;
* the generator's counters across two steady-state traced decode tokens -- the
  host-work boundary in numbers rather than in prose;
* which sampling strategy the greedy path dispatched to.

    python runtime_fallback_audit_probe.py --layers 48
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt import multichip_decoder as MC  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.generator import build_generator  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.model import DEFAULT_TRACE_REGION_SIZE  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--out", type=Path, default=HERE / "runtime_fallback_audit.json")
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MC.MESH_SHAPE), trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    report: dict = {"layers": args.layers, "context": args.context, "prompt_len": args.prompt_len}
    try:
        t0 = time.perf_counter()
        gen = build_generator(
            str(MODEL_DIR), mesh, override_num_layers=args.layers, max_context_len=args.context, max_batch_size=1
        )
        print(f"weight load {time.perf_counter() - t0:.1f} s ({args.layers} layers)", flush=True)
        model = gen.model

        report["audit"] = {k: str(v) for k, v in model.runtime_fallback_audit().items()}

        kv_cache = gen._ensure_kv_cache()

        # -- the host-work boundary, in counters ------------------------------
        prompt = list(range(1000, 1000 + args.prompt_len))
        horizon = len(prompt) + 8
        page_table = gen.make_page_table([horizon])
        gen.reset()
        gen.prefill_forward(
            torch.tensor([prompt]),
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=[len(prompt)],
            sampling_mode="device",
        )
        gen.decode_forward(
            None,
            torch.tensor([len(prompt)]),
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_mode="device",
            enable_trace=True,
            active_batch=1,
            decode_horizon=horizon,
        )
        ttnn.synchronize_device(mesh)
        before = dict(gen.trace_stats)
        gen.decode_forward(None, None, page_table=None, kv_cache=kv_cache, sampling_mode="device", enable_trace=True)
        gen.decode_forward(None, None, page_table=None, kv_cache=kv_cache, sampling_mode="device", enable_trace=True)
        ttnn.synchronize_device(mesh)
        after = dict(gen.trace_stats)
        moved = {k: (before[k], after[k]) for k in after if before.get(k) != after[k]}
        report["steady_state_two_tokens"] = {
            "counters_before": before,
            "counters_after": after,
            "counters_that_moved": moved,
            "only_replays_moved": set(moved) == {"replays"},
        }

        # ``KVCache.page_table`` is bound by the first prefill, and the k_chunk
        # clamp reads it -- so this has to come after a real pass, not before.
        cache0 = kv_cache[0]
        cfg = MC._sdpa_program_config(mesh, cache0)
        # ``repr()`` on an ``SDPAProgramConfig`` raises out of the nanobind
        # binding ("Unable to convert function return value to a Python type"),
        # so the config is described from the values that built it. Minor, but
        # it is why this is not simply ``repr(cfg)``.
        grid = mesh.compute_with_storage_grid_size()
        report["stage06_measured_path"] = {
            "sampler_class": type(model.sampler).__name__,
            "sampler_dist_active_rows": model.sampler._dist_active_rows,
            "sampler_dist_local_vocab": getattr(model.sampler, "_dist_local_vocab", None),
            "sampler_distributed_argmax_taken": getattr(model.sampler, "_dist_die_offset", None) is not None,
            "sdpa_decode_paged": bool(cache0.is_paged),
            "sdpa_decode_k_chunk_tuned": MC._SDPA_PAGED_K_CHUNK,
            "sdpa_decode_k_chunk_used": MC._sdpa_k_chunk(cache0),
            "sdpa_decode_k_chunk_clamped": MC._sdpa_k_chunk(cache0) != MC._SDPA_PAGED_K_CHUNK,
            "sdpa_decode_cache_depth_per_user": MC._paged_cache_depth(cache0),
            "sdpa_decode_max_cores_per_head_batch": MC._SDPA_PAGED_MAX_CORES_PER_HEAD,
            "sdpa_decode_q_chunk": 32,
            "sdpa_decode_program_config": (
                f"SDPAProgramConfig(compute_with_storage_grid_size=({grid.x}, {grid.y}), "
                f"q_chunk_size=32, k_chunk_size={MC._sdpa_k_chunk(cache0)}, "
                f"max_cores_per_head_batch={MC._SDPA_PAGED_MAX_CORES_PER_HEAD})"
            ),
            "sdpa_decode_program_config_type": type(cfg).__name__,
            "sdpa_decode_config_cache_entries": len(MC._SDPA_CONFIG_CACHE),
            # The prefill seam: built and measured, deliberately not wired.
            "sdpa_prefill_program_config_passed": (
                "None"
                if "sdpa_program_config=None" in inspect.getsource(MC.decoder_layer_prefill_multichip)
                else "WIRED -- the documented state is None"
            ),
            "sdpa_prefill_crossover_if_ever_adopted": MC._SDPA_PREFILL_CROSSOVER,
        }

        gen.teardown()
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
