# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit-level, model-free probe of ``models.common.sampling.TTPenalties``.

Question it answers: *does the presence penalty actually reach the logits?*

The vLLM sampling suite can only see a penalty when it flips a greedy argmax.
``TestPresencePenalty`` on this checkpoint does not, so "presence penalty has no
effect" and "presence penalty has an effect too small to change the argmax" look
identical from the server.  This probe removes the model entirely: it drives
``TTPenalties.apply()`` on device with a *known* logits tensor and a *known*
token history, reads the result back, and checks it against the closed-form
reference

    out = logits - presence*output_mask - frequency*output_counts   (repetition == 1)

element-wise, separately for the columns whose token appeared and the columns
whose token did not.  A presence penalty that never reaches the logits shows up
as ``delta == 0`` on appeared columns; a working one shows ``delta == penalty``.

Two histories are exercised, because serving uses both:
  * ``reset_output_tokens(...)``   -- the host-staged reset (prefill / reset_batch)
  * ``update_output_tokens(...)``  -- the on-device incremental scatter_add that
                                     every traced decode step runs

Run (needs the mesh, ~1 min, no weights):

    timeout 600 python models/autoports/meta_models_muse_glimmer_30b/doc/vllm_integration/bench/penalty_probe.py \
        --out models/autoports/meta_models_muse_glimmer_30b/doc/vllm_integration/penalty_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import ttnn
from models.common.sampling.tt_penalties import TTPenalties

# The shipped serving geometry: see tt/generator.py::_SamplingArgs.
VOCAB_SIZE = 202048
PADDED_VOCAB_SIZE = 202752
MAX_BATCH_SIZE = 32
MESH_SHAPE = (1, 4)


class _Args:
    """Minimal attribute bag; TTPenalties reads exactly these."""

    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.padded_vocab_size = PADDED_VOCAB_SIZE
        self.max_batch_size = MAX_BATCH_SIZE
        self.sub_core_grids = None
        self.sampling_dp = 1


def _to_device_logits(mesh, logits: torch.Tensor) -> ttnn.Tensor:
    """[B, padded_vocab] -> vocab-sharded bf16 tile tensor, the sampler's layout."""
    return ttnn.from_torch(
        logits,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 1), mesh_shape=MESH_SHAPE),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _from_device_logits(mesh, tt: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(
        tt,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 1), mesh_shape=MESH_SHAPE),
    ).float()


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def _stats(name, got: torch.Tensor, want: torch.Tensor):
    err = (got - want).abs()
    return {
        "name": name,
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
    }


def run_case(
    mesh,
    pen: TTPenalties,
    *,
    label: str,
    presence: list[float],
    frequency: list[float],
    repetition: list[float],
    prompt_tokens: torch.Tensor,
    output_tokens: torch.Tensor,
    logits_host: torch.Tensor,
    incremental: bool,
) -> dict:
    """One measurement.  ``incremental`` drives the decode-step scatter_add path."""
    pen.reset_params(presence, frequency, repetition)
    pen.reset_prompt_tokens(prompt_tokens)

    if incremental:
        # Decode path: start empty, then feed one token per step exactly as
        # SamplingGenerator does after each sampled token.
        pen.reset_output_tokens(None)
        for step in range(output_tokens.shape[-1]):
            col = output_tokens[:, step].reshape(1, 1, 1, MAX_BATCH_SIZE).to(torch.int32)
            tt_tok = ttnn.from_torch(
                col,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, None), mesh_shape=MESH_SHAPE),
            )
            pen.update_output_tokens(tt_tok)
    else:
        pen.reset_output_tokens(output_tokens)

    tt_logits = _to_device_logits(mesh, logits_host)
    out = pen.apply(tt_logits)
    got = _from_device_logits(mesh, out)[:MAX_BATCH_SIZE, :PADDED_VOCAB_SIZE]
    tt_logits.deallocate()

    # ---- closed-form reference (repetition kept at 1.0 in every case here) ----
    counts = torch.zeros((MAX_BATCH_SIZE, PADDED_VOCAB_SIZE), dtype=torch.float32)
    counts.scatter_add_(1, output_tokens.long(), torch.ones_like(output_tokens, dtype=torch.float32))
    mask = (counts > 0).float()
    p = torch.tensor(presence, dtype=torch.float32).view(-1, 1)
    f = torch.tensor(frequency, dtype=torch.float32).view(-1, 1)
    want = _bf16(_bf16(logits_host) - _bf16(p * mask) - _bf16(f * counts))

    delta = _bf16(logits_host) - got  # what the device actually subtracted

    # Per-row: the drop on a column whose token appeared, vs one that did not.
    rows = []
    for r in range(len(presence)):
        appeared = torch.nonzero(mask[r]).flatten()
        absent = torch.nonzero(mask[r] == 0).flatten()[:4096]
        rows.append(
            {
                "row": r,
                "presence": presence[r],
                "frequency": frequency[r],
                "n_appeared_cols": int(appeared.numel()),
                "delta_on_appeared": sorted({round(float(v), 4) for v in delta[r, appeared]}),
                "counts_on_appeared": sorted({int(v) for v in counts[r, appeared]}),
                "max_abs_delta_on_absent": float(delta[r, absent].abs().max()) if absent.numel() else None,
            }
        )

    res = {
        "case": label,
        "incremental_update_output_tokens": incremental,
        "vs_reference": _stats(label, got, want),
        "rows": rows,
    }
    print(f"\n=== {label} (incremental={incremental}) ===", flush=True)
    print(f"    max|got-reference| = {res['vs_reference']['max_abs_err']:.4f}", flush=True)
    for r in rows:
        print(
            f"    row {r['row']:>2} presence={r['presence']:>5} frequency={r['frequency']:>5} "
            f"appeared_cols={r['n_appeared_cols']} counts={r['counts_on_appeared']} "
            f"delta_on_appeared={r['delta_on_appeared']} "
            f"max|delta|_elsewhere={r['max_abs_delta_on_absent']}",
            flush=True,
        )
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*MESH_SHAPE), l1_small_size=6144, trace_region_size=0)
    report: dict = {"geometry": {"mesh": list(MESH_SHAPE), "vocab": VOCAB_SIZE, "padded_vocab": PADDED_VOCAB_SIZE}}
    try:
        pen = TTPenalties(mesh_device=mesh, args=_Args())

        # A flat, known logits field.  Flat on purpose: any change is the penalty,
        # nothing else.  0.5 so the repetition branch (logits > 0) is the taken one,
        # matching what a real peaked argmax column looks like.
        logits_host = torch.full((MAX_BATCH_SIZE, PADDED_VOCAB_SIZE), 0.5, dtype=torch.float32)

        # Token history: 3 distinct ids, one of them repeated 4x, so presence
        # (once) and frequency (per occurrence) are separable in one shot.
        hist = [11, 11, 11, 11, 4242, 90210]
        output_tokens = torch.tensor([hist] * MAX_BATCH_SIZE, dtype=torch.int32)
        prompt_tokens = torch.tensor([[7, 8, 9]] * MAX_BATCH_SIZE, dtype=torch.int32)

        # The exact sweep TestPresencePenalty::test_different_presence_penalties uses.
        sweep = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        presence = (sweep * 4)[:MAX_BATCH_SIZE]
        zero = [0.0] * MAX_BATCH_SIZE
        one = [1.0] * MAX_BATCH_SIZE

        report["cases"] = [
            run_case(
                mesh,
                pen,
                label="presence_only_reset_path",
                presence=presence,
                frequency=zero,
                repetition=one,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                logits_host=logits_host,
                incremental=False,
            ),
            run_case(
                mesh,
                pen,
                label="presence_only_decode_scatter_path",
                presence=presence,
                frequency=zero,
                repetition=one,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                logits_host=logits_host,
                incremental=True,
            ),
            run_case(
                mesh,
                pen,
                label="frequency_control_decode_scatter_path",
                presence=zero,
                frequency=presence,
                repetition=one,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                logits_host=logits_host,
                incremental=True,
            ),
        ]

        # Verdict: on the presence cases the drop on an appeared column must equal
        # the row's presence penalty, and must be zero everywhere else.
        ok = True
        for case in report["cases"][:2]:
            for r in case["rows"]:
                want = round(r["presence"], 4)
                got = r["delta_on_appeared"]
                if got != [want] and not (want == 0.0 and got in ([0.0], [-0.0])):
                    ok = False
                    print(f"    MISMATCH {case['case']} row {r['row']}: want [{want}] got {got}", flush=True)
                if r["max_abs_delta_on_absent"] not in (0.0, None):
                    ok = False
                    print(f"    LEAK {case['case']} row {r['row']}: {r['max_abs_delta_on_absent']}", flush=True)
        report["presence_penalty_reaches_logits"] = ok
        print(f"\nPRESENCE_PENALTY_REACHES_LOGITS = {ok}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}", flush=True)
    return 0 if report.get("presence_penalty_reaches_logits") else 1


if __name__ == "__main__":
    sys.exit(main())
