# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real 36-layer, 1k GPT-OSS prefill proof for the DFlash handoff.

This is a Galaxy-gated sibling of ``galaxy_prefill_kv_pcc.py``. It keeps that
test's target-KV gate and additionally checks the true-prefill pre-norm
``reduced_hidden`` and final-token logits/y0 against the Blaze seed reference.
It also emits a consumer fixture that the P/D/reload owner can bulk-prime.

Required:
  PREFILL_TRACE_DIR          target KV golden; its first 1k tokens drive smoke mode
  TT_HF_DRAFT_MODEL          gpt-oss-120b-DFlash checkpoint
  HF_MODEL                   real GPT-OSS target checkpoint

Optional:
  BLAZE_DFLASH_SEED_REF      generate_seed.py output; enables feature/y0 PCC gates
  PREFILL_DFLASH_HANDOFF_OUT output .pt for the reload/DFlash compatibility gate
  PREFILL_DFLASH_COMPAT_CONTROL prefill-by-decode control result .pt
  PREFILL_DFLASH_COMPAT_RESULT  reload consumer result .pt
  GPT_OSS_DFLASH_PCC_MIN     aggregate pre-norm PCC floor (default 0.90)
  GPT_OSS_DFLASH_POS_PCC_MIN optional per-position PCC floor
  PREFILL_TPS_ITERS          disabled/enabled timing repetitions (default 1)
  PREFILL_TSU_MIN            synchronized enabled-path tokens/s floor
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

import ttnn
from models.common.utility_functions import comp_pcc

REAL_TOKENS = 1000
PADDED_TOKENS = 1024
ROWS, COLS = 4, 8
NUM_LAYERS = 36
TARGET_LAYERS = (1, 9, 17, 25, 33)
HANDOFF_FORMAT_VERSION = 1


def _pcc(reference: torch.Tensor, actual: torch.Tensor) -> float:
    return float(comp_pcc(reference.float(), actual.float(), 0.0)[1])


def _require_path(env: str) -> Path:
    raw = os.environ.get(env)
    if not raw:
        raise RuntimeError(f"{env} is required")
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"{env}={path} does not exist")
    return path


def _load_inputs(seed_path: Path | None, trace_dir: Path):
    kv_tokens = list(json.loads((trace_dir / "metadata.json").read_text())["token_ids"])
    if len(kv_tokens) < REAL_TOKENS:
        raise ValueError(f"{trace_dir} needs at least {REAL_TOKENS} token ids")
    selected = [int(x) for x in kv_tokens[:REAL_TOKENS]]
    if seed_path is None:
        return None, selected
    seed = torch.load(seed_path, map_location="cpu", weights_only=False)
    seed_tokens = seed.get("seed_token_ids") or seed.get("prompt_token_ids")
    if not seed_tokens:
        raise ValueError(f"{seed_path} needs at least one seed token id")
    reference_positions = min(len(seed_tokens), REAL_TOKENS)
    if "reduced_hidden_prenorm_ref" not in seed or seed["reduced_hidden_prenorm_ref"].shape[0] < reference_positions:
        raise ValueError(f"{seed_path} needs reduced_hidden_prenorm_ref[{reference_positions}, H]")
    if [int(x) for x in seed_tokens[:reference_positions]] != selected[:reference_positions]:
        raise ValueError("PREFILL_TRACE_DIR and BLAZE_DFLASH_SEED_REF contain different token prefixes")
    return seed, selected


def _to_host_reduced(mesh, result) -> torch.Tensor:
    host = ttnn.to_torch(
        result.reduced_hidden,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh,
            mesh_shape=tuple(mesh.shape),
            dims=(2, 3),
        ),
    )
    return host.reshape(-1, host.shape[-1]).float()


def _write_consumer_fixture(
    path: Path, seed: dict | None, token_ids: list[int], result, reduced_pre: torch.Tensor
) -> Path:
    """Emit the native handoff plus a directly runnable proposal trace.

    The prompt's first 1k rows are always the true-prefill output. If the
    independent seed contains a post-prompt margin, append it so proposal
    acceptance can commit future target rows. Otherwise one duplicate tail row
    still permits a first-proposal-only run.
    """

    requested_margin = int(os.getenv("PREFILL_DFLASH_TRACE_MARGIN", "8"))
    extra_reduced = reduced_pre[-1:]
    if seed is not None and seed["reduced_hidden_prenorm_ref"].shape[0] > REAL_TOKENS:
        extra_reduced = seed["reduced_hidden_prenorm_ref"][REAL_TOKENS : REAL_TOKENS + requested_margin].float()
    trace_reduced = torch.cat([reduced_pre, extra_reduced], dim=0).contiguous()
    next_token = torch.full((trace_reduced.shape[0],), int(result.y0), dtype=torch.int64)
    if seed is not None and seed.get("next_token_ref") is not None:
        available = min(len(seed["next_token_ref"]), len(next_token))
        next_token[:available] = torch.as_tensor(seed["next_token_ref"][:available], dtype=torch.int64)
    next_token[REAL_TOKENS - 1] = int(result.y0)
    trace = {
        "prompt_token_ids": list(token_ids),
        "reduced_hidden": trace_reduced,
        "next_token": next_token,
        "block_size": int(os.getenv("PREFILL_DFLASH_BLOCK_SIZE", "8")),
        "hidden_dim": int(reduced_pre.shape[-1]),
        "target_layer_ids": TARGET_LAYERS,
    }
    payload = {
        "format_version": HANDOFF_FORMAT_VERSION,
        "producer": {
            "model": "gpt-oss-120b",
            "path": "true_prefill",
            "feature_contract": "fc_only_prenorm",
            "reference": "independent_hf_teacher_forced" if seed is not None else None,
        },
        "handoff": {
            "slot_id": result.slot_id,
            "actual_start": result.actual_start,
            "actual_end": result.actual_end,
            "chunk_size": result.chunk_size,
            "layout": {
                "mesh_shape": result.layout.mesh_shape,
                "sp_axis": result.layout.sp_axis,
                "tp_axis": result.layout.tp_axis,
                "sequence": result.layout.sequence,
                "feature": result.layout.feature,
            },
            "reduced_hidden_prenorm": reduced_pre.contiguous(),
            "last_token_logits": result.logits,
            "y0": result.y0,
        },
        "trace": trace,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    trace_path = path.with_name(f"{path.stem}.trace.pt")
    torch.save(trace, trace_path)
    return trace_path


def _gate_consumer_compatibility() -> None:
    """Compare the reload consumer against the current prefill-by-decode control.

    Producing these two files belongs to the reload/DFlash stack. Keeping the
    comparison here makes the final cross-stack acceptance criterion executable
    without coupling true prefill to consumer implementation details.
    """

    control_raw = os.environ.get("PREFILL_DFLASH_COMPAT_CONTROL")
    result_raw = os.environ.get("PREFILL_DFLASH_COMPAT_RESULT")
    if not control_raw and not result_raw:
        return
    if not control_raw or not result_raw:
        raise ValueError("set both PREFILL_DFLASH_COMPAT_CONTROL and PREFILL_DFLASH_COMPAT_RESULT")
    control = torch.load(control_raw, map_location="cpu", weights_only=False)
    consumed = torch.load(result_raw, map_location="cpu", weights_only=False)

    def decode_contract(blob):
        # Native run_spec_decode --result-out schema: one dataclass-as-dict per turn.
        if isinstance(blob, list):
            turns = blob
            return {
                "first_proposal": turns[0]["rounds"][0]["proposals"],
                "accepted_tokens": [token for turn in turns for rnd in turn["rounds"] for token in rnd["accepted"]],
                "prompt_kv_pcc": None,
            }
        return blob

    control = decode_contract(control)
    consumed = decode_contract(consumed)
    for key in ("first_proposal", "accepted_tokens"):
        if list(consumed[key]) != list(control[key]):
            raise AssertionError(f"DFlash compatibility mismatch for {key}")
    kv_pcc = consumed.get("prompt_kv_pcc")
    if kv_pcc is None:
        return
    kv_pcc = float(kv_pcc)
    kv_floor = float(os.getenv("GPT_OSS_DFLASH_DRAFT_KV_PCC_MIN", "0.99"))
    if kv_pcc < kv_floor:
        raise AssertionError(f"prompt-primed drafter KV PCC {kv_pcc:.5f} < {kv_floor}")


def main() -> int:
    if ttnn.get_num_devices() < ROWS * COLS:
        print("[dflash-prefill] SKIP: requires a 4x8 Blackhole Galaxy", flush=True)
        return 0
    trace_dir = _require_path("PREFILL_TRACE_DIR")
    seed_raw = os.environ.get("BLAZE_DFLASH_SEED_REF")
    seed_path = _require_path("BLAZE_DFLASH_SEED_REF") if seed_raw else None
    draft_path = _require_path("TT_HF_DRAFT_MODEL")
    seed, token_ids = _load_inputs(seed_path, trace_dir)

    from models.demos.gpt_oss_d_p.tt.model_config import ModelArgs
    from models.demos.gpt_oss_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    linear = os.getenv("PREFILL_TOPOLOGY", "ring") == "linear"
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D if linear else ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(ROWS, COLS))
    try:
        model_args = ModelArgs(mesh_device=mesh)
        hf_config = model_args.hf_config
        if hf_config.num_hidden_layers != NUM_LAYERS:
            raise ValueError(f"expected the 36-layer GPT-OSS target, got {hf_config.num_hidden_layers}")
        state_dict = (
            {} if os.getenv("GPT_OSS_WEIGHTS_FROM_CACHE") == "1" else ModelArgs.load_state_dict(model_args.weights_path)
        )
        runtime = TtPrefillRuntime(
            mesh,
            hf_config,
            state_dict,
            TtPrefillRuntimeConfig(
                num_layers=NUM_LAYERS,
                max_seq_len=PADDED_TOKENS,
                mesh_shape=(ROWS, COLS),
                default_chunk_size=PADDED_TOKENS,
                num_users=2,
                weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
                topology=ttnn.Topology.Linear if linear else ttnn.Topology.Ring,
                dflash_checkpoint_path=draft_path,
            ),
        )
        del state_dict
        runtime.compile()
        padded = token_ids + [0] * (PADDED_TOKENS - REAL_TOKENS)

        def run(*, handoff: bool, kv_tail: bool):
            inp = runtime.make_chunk_input(padded, PADDED_TOKENS)
            out = runtime.prefill_chunk(
                inp,
                slot_id=0,
                actual_start=0,
                actual_end=REAL_TOKENS,
                chunk_size=PADDED_TOKENS,
                dflash_handoff=handoff,
                dflash_kv_tail=kv_tail,
            )
            ttnn.synchronize_device(mesh)
            return out

        iters = int(os.getenv("PREFILL_TPS_ITERS", "1"))
        disabled_times = []
        for _ in range(iters):
            started = time.perf_counter()
            run(handoff=False, kv_tail=False)
            disabled_times.append(time.perf_counter() - started)
        feature_only_times = []
        for _ in range(iters):
            started = time.perf_counter()
            feature_result = run(handoff=True, kv_tail=False)
            feature_only_times.append(time.perf_counter() - started)
            ttnn.deallocate(feature_result.reduced_hidden)
        enabled_times = []
        result = None
        for _ in range(iters):
            started = time.perf_counter()
            next_result = run(handoff=True, kv_tail=True)
            enabled_times.append(time.perf_counter() - started)
            if result is not None:
                ttnn.deallocate(result.reduced_hidden)
            result = next_result
        assert result is not None
        enabled_median = statistics.median(enabled_times)
        enabled_tsu = REAL_TOKENS / enabled_median
        tsu_min = os.environ.get("PREFILL_TSU_MIN")

        # Existing target-KV proof remains the authoritative target-cache gate.
        kv_pcc = runtime.kv_cache_pcc_check(
            slot_id=0,
            n_chunks=1,
            trace_dir=trace_dir,
            real_len=REAL_TOKENS,
            chunk_size=PADDED_TOKENS,
        )
        gate_failures = []
        kv_min = os.environ.get("GPT_OSS_KV_PCC_MIN")
        if kv_min is not None and kv_pcc < float(kv_min):
            gate_failures.append(f"target KV PCC {kv_pcc:.5f} < {kv_min}")

        export_started = time.perf_counter()
        reduced = _to_host_reduced(mesh, result)[:REAL_TOKENS]
        export_ms = (time.perf_counter() - export_started) * 1000.0
        dflash_kv_pcc = runtime.dflash_kv_cache_pcc_check(
            reduced,
            slot_id=0,
            chunk_size=PADDED_TOKENS,
        )
        dflash_kv_min = float(os.getenv("GPT_OSS_DFLASH_DRAFT_KV_PCC_MIN", "0.999"))
        if dflash_kv_pcc < dflash_kv_min:
            gate_failures.append(f"drafter KV PCC {dflash_kv_pcc:.5f} < {dflash_kv_min}")
        aggregate = per_position = None
        golden_token = None
        if seed is not None:
            reference_positions = min(len(seed["seed_token_ids"]), REAL_TOKENS)
            reference = seed["reduced_hidden_prenorm_ref"][:reference_positions].float()
            reduced_reference = reduced[:reference_positions]
            aggregate = _pcc(reference, reduced_reference)
            per_position = torch.tensor([_pcc(reference[i], reduced_reference[i]) for i in range(reference_positions)])
            pcc_min = float(os.getenv("GPT_OSS_DFLASH_PCC_MIN", "0.90"))
            if aggregate < pcc_min:
                gate_failures.append(f"reduced_hidden aggregate PCC {aggregate:.5f} < {pcc_min}")
            pos_min = os.environ.get("GPT_OSS_DFLASH_POS_PCC_MIN")
            if pos_min is not None and float(per_position.min()) < float(pos_min):
                gate_failures.append(f"reduced_hidden min per-position PCC {per_position.min():.5f} < {pos_min}")

            if len(seed["next_token_ref"]) >= REAL_TOKENS:
                golden_token = int(seed["next_token_ref"][REAL_TOKENS - 1])
                topk_ids = seed.get("golden_topk_token_ids")
                if result.y0 != golden_token and (
                    topk_ids is None or result.y0 not in [int(x) for x in topk_ids[REAL_TOKENS - 1]]
                ):
                    gate_failures.append(
                        f"y0={result.y0} differs from golden={golden_token} and is outside golden top-k"
                    )

        handoff_out = Path(os.getenv("PREFILL_DFLASH_HANDOFF_OUT", "generated/gptoss_dflash_handoff_1k.pt"))
        spec_trace = _write_consumer_fixture(handoff_out, seed, token_ids, result, reduced)
        _gate_consumer_compatibility()
        accuracy = f"KV_PCC={kv_pcc:.5f} DFLASH_KV_PCC={dflash_kv_pcc:.5f} y0={result.y0}"
        if aggregate is not None and per_position is not None:
            accuracy += (
                f" RH_REF_POS={len(per_position)} RH_PCC={aggregate:.5f} "
                f"RH_POS(min/mean)={per_position.min():.5f}/{per_position.mean():.5f}"
            )
            if len(per_position) >= REAL_TOKENS:
                accuracy += f" RH_LAST_PCC={per_position[REAL_TOKENS - 1]:.5f}"
        if golden_token is not None:
            accuracy += f" y0_ref={golden_token}"
        print(f"[dflash-prefill] {accuracy}", flush=True)
        print(
            f"[dflash-prefill] latency target/feature-only/feature+KV median="
            f"{statistics.median(disabled_times) * 1000:.1f}/"
            f"{statistics.median(feature_only_times) * 1000:.1f}/"
            f"{enabled_median * 1000:.1f} ms ({enabled_tsu:.1f} tokens/s); "
            f"feature_export={export_ms:.1f} ms; enqueue_breakdown={result.timings_ms}",
            flush=True,
        )
        print(f"[dflash-prefill] consumer fixture={handoff_out} spec_trace={spec_trace}", flush=True)
        if tsu_min is not None and enabled_tsu < float(tsu_min):
            gate_failures.append(f"enabled prefill throughput {enabled_tsu:.1f} tokens/s < {tsu_min}")
        if gate_failures:
            raise AssertionError("; ".join(gate_failures))
        ttnn.deallocate(result.reduced_hidden)
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
