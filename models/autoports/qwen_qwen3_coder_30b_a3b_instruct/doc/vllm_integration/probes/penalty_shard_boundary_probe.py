# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Do sampling penalties land on the right die, the right column and the right row?

The failure mode this probe exists for produces **valid-looking wrong output**,
not an error. Logits are column-parallel: die ``d`` holds global vocabulary ids
``d*37984 .. d*37984+37983`` and nothing else. A penalty is keyed by a *global*
token id, so penalising id ``t`` means touching local column ``t % 37984`` on
die ``t // 37984`` **and no column on the other three dies**. Get that wrong and
the model still emits fluent text -- it just penalises three tokens nobody asked
about, and the one that was asked about not at all.

``tt/model.py::_WatcherCleanSampling1D`` never does that index arithmetic in a
kernel. The penalty operands are built on the host at full vocabulary width and
handed down through ``ttnn.ShardTensorToMesh(dim=-1)``, the same even 4-way split
the column-parallel LM head produced the logits under, so column ``t`` lands on
the die that holds logit ``t`` by construction. This probe checks that claim on
the device rather than asserting it, and checks four more:

* **cross-die reach** -- a token in die 0's range and a token in die 3's range are
  both penalised, in the same step, on the same row;
* **no aliasing** -- for a penalised id ``t``, the *same local index* on the other
  three dies (``t ± k*37984``) is **bit-identical** to the un-penalised logit.
  Bit-identical, not close: an unpenalised column gets ``x * 1.0 - 0.0``, which is
  exact in bf16, so this is a property of the arithmetic and not of a tolerance;
* **boundary columns** -- local column 0 and local column 37983 of a die, i.e. the
  ids either side of every shard seam, are penalised correctly;
* **row isolation** -- rows with different penalties, and rows with *no* penalty,
  in one batch; row *i*'s history must not move row *j*'s logits.

Everything is compared against a torch transcription of vLLM's
``model_executor/layers/utils.py::apply_penalties`` (repetition first over
prompt+output, then frequency over output counts, then presence over the output
mask), and the last leg runs the *whole* shipped sampler --
``_WatcherCleanSampling1D.decode_forward`` -- so the check covers the ordering
requirement too: penalties must be applied **before** the argmax selection.

The host staging is not re-implemented here: the probe binds the shipped
``Qwen3CoderGenerator.set_penalty_params`` onto a minimal shim, so a bug in the
host-side scatter is a bug this probe sees.

Standalone: opens its own 1x4 mesh with synthetic logits at the shipped shape,
no 48-layer model, no weights. Nothing here writes into ``doc/full_model/``,
``doc/optimized_full_model/`` or ``doc/datatype_sweep/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.generator import Qwen3CoderGenerator  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.model import _WatcherCleanSampling1D  # noqa: E402
from models.common.modules.sampling.sampling_1d import Sampling1DConfig  # noqa: E402
from models.common.modules.tt_ccl import TT_CCL  # noqa: E402

HERE = Path(__file__).resolve().parent
VOCAB = 151936
DEVICES = 4
LOCAL_VOCAB = VOCAB // DEVICES  # 37984
SLOTS = 32
TOPOLOGY = ttnn.Topology.Ring


# ---------------------------------------------------------------------------
# Reference: vLLM's apply_penalties, transcribed
# ---------------------------------------------------------------------------


def reference_penalised(logits: torch.Tensor, rows: list[dict]) -> torch.Tensor:
    """``vllm/model_executor/layers/utils.py::apply_penalties`` on ``[R, V]``.

    Order is vLLM's and is load-bearing: repetition multiplies the *raw* logit,
    then frequency and presence subtract.
    """
    out = logits.to(torch.float32).clone()
    for row, spec in enumerate(rows):
        prompt = torch.as_tensor(spec.get("prompt", []), dtype=torch.int64)
        output = torch.as_tensor(spec.get("output", []), dtype=torch.int64)
        p = float(spec.get("repetition", 1.0))
        f = float(spec.get("frequency", 0.0))
        q = float(spec.get("presence", 0.0))
        if p != 1.0:
            seen = torch.unique(torch.cat((prompt, output))) if (prompt.numel() + output.numel()) else prompt
            if seen.numel():
                values = out[row, seen]
                out[row, seen] = torch.where(values > 0, values / p, values * p)
        if output.numel() and (f != 0.0 or q != 0.0):
            unique, counts = torch.unique(output, return_counts=True)
            out[row, unique] -= f * counts.to(torch.float32) + q
    return out


# ---------------------------------------------------------------------------
# Host staging: the shipped generator methods, on a device-only shim
# ---------------------------------------------------------------------------


class _HostStage:
    """Just enough of ``Qwen3CoderGenerator`` to drive ``set_penalty_params``.

    The three staging methods are taken **from the shipped class**, not copied,
    so the host-side scatter this probe checks is the one that runs in serving.
    """

    set_penalty_params = Qwen3CoderGenerator.set_penalty_params
    _ensure_penalty_host = Qwen3CoderGenerator._ensure_penalty_host
    _upload_penalty_tensor = Qwen3CoderGenerator._upload_penalty_tensor
    _penalty_split = Qwen3CoderGenerator._penalty_split
    # ``@staticmethod`` on the original: accessing it through the class unwraps
    # the descriptor, so it has to be re-wrapped or it would rebind as a method.
    _row_token_ids = staticmethod(Qwen3CoderGenerator._row_token_ids)

    def __init__(self, mesh, sampler):
        self.mesh_device = mesh
        self.model = type("_M", (), {"sampler": sampler})()
        self.trace_stats = {"penalty_host_copies": 0}
        self._trace_model_id = None
        self._trace_sampling_id = None
        self._decode_warm_key = None
        self._penalty_mode = 0
        self._penalty_host = None
        self._penalty_local_vocab = None
        self._penalty_prev_add: list = []
        self._penalty_prev_rep: list = []

    def _release_decode_traces(self):  # no trace in this probe
        pass


def pad_history(rows: list[dict], key: str) -> torch.Tensor | None:
    """vLLM's ``[rows, L]`` int32 history tensor, **-1 padded**, or None."""
    lists = [list(spec.get(key, [])) for spec in rows]
    width = max((len(item) for item in lists), default=0)
    if width == 0:
        return None
    out = torch.full((len(lists), width), -1, dtype=torch.int32)
    for row, ids in enumerate(lists):
        if ids:
            out[row, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
    return out


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------


def build_sampler(mesh, ccl):
    sampler = _WatcherCleanSampling1D.from_config(
        Sampling1DConfig(
            vocab_size=VOCAB,
            valid_vocab_size=VOCAB,
            mesh_device=mesh,
            tt_ccl=ccl,
            max_batch_size=SLOTS,
            max_top_k=32,
            num_gather_links=1,
            sampling_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            allow_force_argmax=True,
            num_argmax_gather_links=1,
            ag_topology=TOPOLOGY,
            pad_to_power_of_2=False,
        )
    )
    sampler._dist_active_rows = SLOTS
    sampler.load_device_buffers()
    return sampler


def to_device(mesh, host: torch.Tensor):
    return ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
    )


def from_device(mesh, tensor) -> torch.Tensor:
    return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))


#: The batch under test. Deliberately mixed: a repetition-only row whose tokens
#: straddle every shard seam, an *unpenalised* row, an additive-only row, a row
#: with all three, and a row whose history is the same local index in a different
#: die from row 0's -- the aliasing trap.
def make_rows() -> list[dict]:
    return [
        {
            # die 0 first column, die 0 last column, die 1 first column,
            # die 3 last column -- both sides of three shard seams.
            "prompt": [0, LOCAL_VOCAB - 1, LOCAL_VOCAB, VOCAB - 1],
            "output": [7, 7, 3 * LOCAL_VOCAB + 11],
            "repetition": 2.0,
        },
        {},  # no penalty at all, alongside penalised rows
        {"output": [5, 5, 5, LOCAL_VOCAB + 5, 2 * LOCAL_VOCAB + 5], "frequency": 0.75, "presence": 1.25},
        {
            "prompt": [123, 2 * LOCAL_VOCAB + 123],
            "output": [3 * LOCAL_VOCAB + 4, 3 * LOCAL_VOCAB + 4],
            "repetition": 0.5,
            "frequency": -0.5,
            "presence": 2.0,
        },
        # Same *local* index 5 as row 2's die-0 token, but only on die 3. If the
        # stage ever computed a local index and broadcast it, this row and row 2
        # would contaminate each other.
        {"prompt": [3 * LOCAL_VOCAB + 5], "repetition": 3.0},
    ]


def run(mesh, ccl, *, seed: int) -> dict:
    torch.manual_seed(seed)
    sampler = build_sampler(mesh, ccl)
    stage = _HostStage(mesh, sampler)
    rows = make_rows()
    active = len(rows)

    # Logits well away from zero on both signs, so the sign-dependent repetition
    # branch is exercised in both directions.
    host_logits = (torch.randn(1, 1, SLOTS, VOCAB) * 4.0).to(torch.bfloat16)
    baseline = host_logits[0, 0].to(torch.float32).clone()

    live, changed = stage.set_penalty_params(
        presence=[r.get("presence", 0.0) for r in rows],
        frequency=[r.get("frequency", 0.0) for r in rows],
        repetition=[r.get("repetition", 1.0) for r in rows],
        prompt_tokens=pad_history(rows, "prompt"),
        output_tokens=pad_history(rows, "output"),
        active_batch=active,
    )
    assert live and changed, (live, changed)
    assert sampler._penalty_mode == 3, sampler._penalty_mode

    logits = to_device(mesh, host_logits)
    penalised, is_new = sampler._apply_penalties(logits)
    assert is_new
    got = from_device(mesh, penalised)[0, 0].to(torch.float32)
    ttnn.deallocate(penalised)

    want = reference_penalised(host_logits[0, 0], rows + [{}] * (SLOTS - active))

    results: dict = {"seed": seed, "vocab": VOCAB, "devices": DEVICES, "local_vocab": LOCAL_VOCAB}

    # -- 1. matches the reference everywhere -------------------------------
    # bf16 has an 8-bit significand; the reference is computed in fp32, so the
    # tolerance is one bf16 ulp of the value, not an arbitrary epsilon.
    tolerance = 1e-2 * torch.maximum(want.abs(), torch.tensor(1.0))
    delta = (got - want).abs()
    results["max_abs_error"] = float(delta.max())
    results["max_ulp_ratio"] = float((delta / tolerance).max())
    results["matches_vllm_reference"] = bool((delta <= tolerance).all())

    # -- 2. only the intended columns moved --------------------------------
    moved = (got != baseline).nonzero()
    expected_moved: set[tuple[int, int]] = set()
    for row, spec in enumerate(rows):
        touched: set[int] = set()
        if spec.get("repetition", 1.0) != 1.0:
            touched |= set(spec.get("prompt", [])) | set(spec.get("output", []))
        if spec.get("frequency", 0.0) or spec.get("presence", 0.0):
            touched |= set(spec.get("output", []))
        expected_moved |= {(row, int(t)) for t in touched}
    actual_moved = {(int(r), int(c)) for r, c in moved.tolist()}
    # A penalty can round to a no-op in bf16 (e.g. a value whose penalised form
    # is the same bf16), so the check is "nothing unexpected moved", plus the
    # reference comparison above for "everything expected moved".
    results["unexpected_columns_moved"] = sorted(actual_moved - expected_moved)[:20]
    results["no_unexpected_columns_moved"] = not (actual_moved - expected_moved)
    results["expected_columns_that_moved"] = len(actual_moved & expected_moved)
    results["expected_columns_total"] = len(expected_moved)

    # -- 3. cross-die reach and non-aliasing -------------------------------
    #
    # For every penalised id ``t``, the same *local* index on the other three
    # dies must be bit-identical to the input. This is the check that would fail
    # if the stage ever replicated a per-die tensor instead of sharding one.
    aliases_intact = True
    alias_failures = []
    cross_die = {}
    for row, token in sorted(expected_moved):
        die = token // LOCAL_VOCAB
        local = token % LOCAL_VOCAB
        cross_die.setdefault(int(die), 0)
        cross_die[int(die)] += 1
        for other in range(DEVICES):
            if other == die:
                continue
            alias = other * LOCAL_VOCAB + local
            if (row, alias) in expected_moved:
                # Row 2's history deliberately contains 5, 37984+5 and 2*37984+5:
                # three *different* global ids that share a local index. Each is
                # penalised on its own die and that is correct, so an alias that
                # is itself a requested token is not an aliasing failure -- it is
                # the cross-die coverage this row exists to provide.
                continue
            if got[row, alias] != baseline[row, alias]:
                aliases_intact = False
                alias_failures.append({"row": row, "token": int(token), "alias": int(alias)})
    results["penalised_tokens_per_die"] = cross_die
    results["reaches_die_0"] = cross_die.get(0, 0) > 0
    results["reaches_die_3"] = cross_die.get(3, 0) > 0
    results["boundary_columns_covered"] = sorted(
        {int(t % LOCAL_VOCAB) for _, t in expected_moved} & {0, LOCAL_VOCAB - 1}
    )
    results["same_local_index_on_other_dies_untouched"] = aliases_intact
    results["alias_failures"] = alias_failures[:20]

    # -- 4. row isolation ---------------------------------------------------
    unpenalised_rows = [r for r in range(SLOTS) if r not in {row for row, _ in expected_moved}]
    results["unpenalised_rows"] = len(unpenalised_rows)
    results["unpenalised_rows_bit_identical"] = bool(torch.equal(got[unpenalised_rows], baseline[unpenalised_rows]))

    # -- 5. the whole sampler, penalties before the selection ---------------
    #
    # ``decode_forward`` is the shipped entry point; running it proves the
    # penalty stage sits ahead of the argmax rather than beside it.
    token_out = ttnn.from_torch(
        torch.zeros((1, 1, 1, SLOTS), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    sampled, _ = sampler.decode_forward(logits, tt_out_tok=token_out)
    device_tokens = ttnn.to_torch(ttnn.get_device_tensors(sampled)[0]).reshape(-1)[:active].to(torch.int64)
    want_tokens = want[:active].argmax(dim=-1)
    unpenalised_tokens = baseline[:active].argmax(dim=-1)
    results["sampler_tokens"] = [int(v) for v in device_tokens]
    results["reference_penalised_tokens"] = [int(v) for v in want_tokens]
    results["unpenalised_tokens"] = [int(v) for v in unpenalised_tokens]
    results["sampler_matches_penalised_reference"] = bool(torch.equal(device_tokens, want_tokens))

    # -- 5b. a penalty that must *change* the winner ------------------------
    #
    # Leg 5 above is a null result on its own: with random logits, penalising 13
    # of 151936 columns almost never moves the argmax, so "sampler == reference"
    # there only shows the two agree, not that the penalty reached the selection.
    # This leg penalises each row's **current winner** -- which for this seed
    # lives on die 0 for one row and die 3 for another -- hard enough that the
    # winner must change, and leaves one row unpenalised as the control.
    winners = [int(baseline[row].argmax()) for row in range(active)]
    forced = [
        {"prompt": [winners[row]], "repetition": 8.0, "presence": 40.0, "output": [winners[row]]} if row != 1 else {}
        for row in range(active)
    ]
    stage.set_penalty_params(
        presence=[r.get("presence", 0.0) for r in forced],
        frequency=[0.0] * active,
        repetition=[r.get("repetition", 1.0) for r in forced],
        prompt_tokens=pad_history(forced, "prompt"),
        output_tokens=pad_history(forced, "output"),
        active_batch=active,
    )
    forced_want = reference_penalised(host_logits[0, 0], forced + [{}] * (SLOTS - active))
    forced_sampled, _ = sampler.decode_forward(logits, tt_out_tok=token_out)
    forced_tokens = ttnn.to_torch(ttnn.get_device_tensors(forced_sampled)[0]).reshape(-1)[:active].to(torch.int64)
    results["forced_winner_dies"] = [int(w // LOCAL_VOCAB) for w in winners]
    results["forced_prior_winners"] = winners
    results["forced_tokens"] = [int(v) for v in forced_tokens]
    results["forced_reference_tokens"] = [int(v) for v in forced_want[:active].argmax(dim=-1)]
    results["forced_matches_reference"] = bool(torch.equal(forced_tokens, forced_want[:active].argmax(dim=-1)))
    results["forced_winner_changed"] = [int(forced_tokens[row]) != winners[row] for row in range(active)]
    # Every penalised row must have moved off its old winner; row 1, the control,
    # must not have.
    results["forced_penalty_changed_the_winner"] = all(
        results["forced_winner_changed"][row] == (row != 1) for row in range(active)
    )

    # -- 5c. the fast staging path is the mesh mapper's own partition --------
    #
    # The operands are staged as four contiguous per-die buffers and assembled
    # with ``ttnn.from_host_shards``, because handing a full-width tensor to
    # ``ShardTensorToMesh(dim=-1)`` re-slices a 9.7 MB strided view on every
    # decode step (6.601 ms of a 6.897 ms upload). That moves the global ->
    # (die, local) split into host Python, so it is pinned here: build the same
    # operand both ways and require the two device tensors to be bit-identical.
    # If the split ever disagreed with the mapper the logits are sharded under,
    # every penalty would land on the wrong die and nothing else would notice.
    # Re-stage the *original* batch: leg 5b left the forced-winner config in the
    # host buffers, and this leg asserts against ``rows``.
    stage.set_penalty_params(
        presence=[r.get("presence", 0.0) for r in rows],
        frequency=[r.get("frequency", 0.0) for r in rows],
        repetition=[r.get("repetition", 1.0) for r in rows],
        prompt_tokens=pad_history(rows, "prompt"),
        output_tokens=pad_history(rows, "output"),
        active_batch=active,
    )
    staged = ttnn.from_torch(
        torch.zeros((1, 1, SLOTS, VOCAB), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
    )
    stage._upload_penalty_tensor(stage._penalty_host["rep_neg"], staged)
    fast = from_device(mesh, staged)[0, 0]
    # The same content, assembled the way the rest of the tree does it.
    reference_full = torch.cat([shard[0, 0] for shard in stage._penalty_host["rep_neg"]], dim=-1)
    mapped = ttnn.from_torch(
        reference_full.reshape(1, 1, SLOTS, VOCAB),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
    )
    results["fast_staging_matches_shard_mapper"] = bool(torch.equal(fast, from_device(mesh, mapped)[0, 0]))
    # ... and the operand really does carry p at the ids row 0 asked for, on the
    # die that owns each of them.
    row0 = rows[0]
    seen0 = sorted(set(row0["prompt"]) | set(row0["output"]))
    results["staged_operand_carries_p_on_the_owning_die"] = all(float(fast[0, t]) == row0["repetition"] for t in seen0)
    ttnn.deallocate(staged)
    ttnn.deallocate(mapped)

    # -- 6. the fast path is really a different graph -----------------------
    live0, changed0 = stage.set_penalty_params(active_batch=active)
    results["neutral_request_is_fast_path"] = (not live0) and changed0 and sampler._penalty_mode == 0
    neutral, neutral_new = sampler._apply_penalties(logits)
    results["fast_path_is_identity"] = (not neutral_new) and neutral is logits

    results["passed"] = all(
        (
            results["matches_vllm_reference"],
            results["no_unexpected_columns_moved"],
            results["reaches_die_0"],
            results["reaches_die_3"],
            results["boundary_columns_covered"] == [0, LOCAL_VOCAB - 1],
            results["same_local_index_on_other_dies_untouched"],
            results["unpenalised_rows_bit_identical"],
            results["sampler_matches_penalised_reference"],
            results["forced_matches_reference"],
            results["forced_penalty_changed_the_winner"],
            results["fast_staging_matches_shard_mapper"],
            results["staged_operand_carries_p_on_the_owning_die"],
            results["neutral_request_is_fast_path"],
            results["fast_path_is_identity"],
        )
    )
    ttnn.deallocate(logits)
    ttnn.deallocate(token_out)
    return results


def time_modes(mesh, ccl, *, reps: int, history: int = 0) -> dict:
    """What the penalty stage costs, trace-captured, against the same sampler.

    Three graphs, one per ``_penalty_mode``: 0 (the ops are not in the trace at
    all -- this is what an unpenalised request runs), 1 (repetition only) and 3
    (both stages). Each is the *whole* sampler -- penalty stage plus the
    distributed argmax -- captured and replayed, so the delta is the stage's real
    price in the decode loop and not a standalone op benchmark.

    The host staging cost is timed separately, because it is host time and does
    not overlap the replay.
    """
    import statistics
    import time

    torch.manual_seed(0)
    sampler = build_sampler(mesh, ccl)
    stage = _HostStage(mesh, sampler)
    rows = make_rows()
    if history:
        # A *serving-sized* history. The correctness batch has 2-5 tokens per
        # row; a real decode step at 128/128 carries 128-256, and the per-step
        # host work is O(history). Timing only the tiny batch would understate
        # the staging cost and hide where the in-situ overhead actually is.
        filler = list(range(4096, 4096 + history))
        rows = [dict(r, prompt=list(r.get("prompt", [])) + filler) if r else r for r in rows]
    active = len(rows)
    logits = to_device(mesh, (torch.randn(1, 1, SLOTS, VOCAB) * 4.0).to(torch.bfloat16))
    token_out = ttnn.from_torch(
        torch.zeros((1, 1, 1, SLOTS), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    def stage_args(mode: int) -> dict:
        return {
            "presence": [r.get("presence", 0.0) if mode & 2 else 0.0 for r in rows],
            "frequency": [r.get("frequency", 0.0) if mode & 2 else 0.0 for r in rows],
            "repetition": [r.get("repetition", 1.0) if mode & 1 else 1.0 for r in rows],
            "prompt_tokens": pad_history(rows, "prompt"),
            "output_tokens": pad_history(rows, "output"),
            "active_batch": active,
        }

    out: dict = {"reps": reps, "history_tokens_per_row": history}
    for mode in (0, 1, 3):
        stage.set_penalty_params(**stage_args(mode))
        assert sampler._penalty_mode == mode, (mode, sampler._penalty_mode)
        sampler.decode_forward(logits, tt_out_tok=token_out)  # compile
        ttnn.synchronize_device(mesh)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        sampler.decode_forward(logits, tt_out_tok=token_out)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh)
        samples = []
        for _ in range(reps):
            start = time.perf_counter()
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            samples.append((time.perf_counter() - start) * 1e3)
        ttnn.release_trace(mesh, trace_id)
        out[f"sampler_ms_mode{mode}"] = round(statistics.median(samples), 4)

        if mode:
            host = []
            for _ in range(reps):
                start = time.perf_counter()
                stage.set_penalty_params(**stage_args(mode))
                ttnn.synchronize_device(mesh)
                host.append((time.perf_counter() - start) * 1e3)
            out[f"host_staging_ms_mode{mode}"] = round(statistics.median(host), 4)

    out["device_cost_repetition_only_ms"] = round(out["sampler_ms_mode1"] - out["sampler_ms_mode0"], 4)
    out["device_cost_both_ms"] = round(out["sampler_ms_mode3"] - out["sampler_ms_mode0"], 4)
    out["fast_path_is_free"] = True  # mode 0 *is* the shipped graph; nothing is added
    ttnn.deallocate(logits)
    ttnn.deallocate(token_out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time", action="store_true", help="also measure the stage's cost")
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument(
        "--history", type=int, default=256, help="history tokens per row for the serving-sized cost leg"
    )
    parser.add_argument("--json", type=str, default=str(HERE / "penalty_shard_boundary_probe.json"))
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, DEVICES), trace_region_size=90_000_000)
    try:
        ccl = TT_CCL(mesh)
        results = run(mesh, ccl, seed=args.seed)
        if args.time:
            results["cost"] = time_modes(mesh, ccl, reps=args.reps)
            results["cost_serving_history"] = time_modes(mesh, ccl, reps=args.reps, history=args.history)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    Path(args.json).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
