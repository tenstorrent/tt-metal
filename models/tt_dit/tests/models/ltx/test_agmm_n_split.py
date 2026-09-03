# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Correctness gate for the non-uniform N split in all_gather_minimal_matmul_async.

The split moves N columns OFF the two fabric-relay cores and onto the other seven, to buy the relay
cores slack to hide the ring relay behind a shorter matmul. It changes only WHICH core owns an output
column — never the K-accumulation order (every core still runs one N block, K walked in the same
direction) — so a correct split is BIT-EXACT against the uniform baseline. That makes bit-exactness,
not a PCC threshold, the gate: a dropped, duplicated or misplaced column cannot hide inside it.

ONE ARM PER PROCESS, on purpose. An in-process A/B cannot be trusted here: the split changes only
runtime args, so the two arms are the same cached program, and disable_and_clear_program_cache() was
observed NOT to force the factory to re-run for every op (ff1 silently kept the baseline program, and
the gate then "passed" without ever applying the split). Each arm therefore gets a fresh process,
dumps its outputs, and `--compare` checks the dumps:

  base    TT_AGMM_FABRIC_N_PCT=0       -> the uniform split; vs torch it is the numerical floor
  base2   TT_AGMM_FABRIC_N_PCT=0       -> vs base: pins a BIT-EXACT noise floor
  split   (no knob = the DEFAULT)      -> vs base: MUST be bit-exact
  mutant  + TT_AGMM_N_SPLIT_MUTANT=1   -> vs base: MUST go red (proves the gate bites)

The program factory logs one "AGMM N split: ... -> SPLIT / uniform" line per op, and --compare
asserts the split arm actually took SPLIT on the sites that have room — so an arm cannot pass by
quietly failing to apply the thing under test.

RMSE/sigma is reported next to PCC and is the limb that catches wrong-tensor bugs: the qkv mutant
lands at PCC 0.9896, which a naive 0.98 PCC bound would wave through.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.layers.linear import ColParallelLinear
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.test import ring_params

VIDEO_DIM = 4096
V_ROWS_S1 = 9728 // 8  # 1216 rows/device at SP=8

# The production AG-matmul sites. gate's N is a SINGLE tile (32 head logits / TP=4), so the 9-way N
# axis has no room to re-split at all: it must fall back to uniform, and the gate asserts that rather
# than pretending otherwise.
SITES = {
    "gate": (VIDEO_DIM, 32, None),  # N_local =   32 ->   1 tile  -> NO split room
    "qkv": (VIDEO_DIM, 3 * VIDEO_DIM, 3),  # N_local = 3072 ->  96 tiles
    "out": (VIDEO_DIM, VIDEO_DIM, None),  # N_local = 1024 ->  32 tiles
    "ff1": (VIDEO_DIM, 4 * VIDEO_DIM, None),  # N_local = 4096 -> 128 tiles
}
SPLITTABLE = ["qkv", "out", "ff1"]  # gate has no headroom; see above

DUMP_DIR = "opt/nsplit_dumps"


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [pytest.param((4, 8), {**ring_params}, id="ring_bh_4x8")],
    indirect=True,
)
def test_agmm_n_split(mesh_device: ttnn.MeshDevice) -> None:
    """Run ONE arm (selected by the env the process was launched with) and dump its outputs."""
    arm = os.environ["NSPLIT_ARM"]
    tp_axis, sp_axis = 0, 1
    tp = tuple(mesh_device.shape)[tp_axis]
    sp = tuple(mesh_device.shape)[sp_axis]
    assert (tp, sp) == (4, 8), f"expected 4x8 with tp on axis0, got tp={tp} sp={sp}"

    ccl = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    pc = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp, mesh_axis=sp_axis),
    )

    # Same seed in every arm: the arms differ ONLY by the knob.
    torch.manual_seed(0)
    x_torch = torch.randn(1, 1, V_ROWS_S1, VIDEO_DIM // tp) * 0.5
    x = ttnn.from_torch(x_torch, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=mesh_device)

    def dev0(t: ttnn.Tensor) -> torch.Tensor:
        """Device 0's local shard, as torch — no mesh-composer sharding convention to get wrong."""
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()

    # x is replicated (no mesh_mapper), so the op's TP gather along K concatenates tp identical
    # shards: the reference contracts against exactly that.
    x_full = x_torch.repeat(1, 1, 1, tp).squeeze()

    dump: dict[str, torch.Tensor] = {}
    for name, (in_f, out_f, chunks) in SITES.items():
        m = ColParallelLinear(
            in_f, out_f, bias=True, mesh_device=mesh_device, mesh_axis=tp_axis, ccl_manager=ccl, chunks=chunks
        )
        m.load_torch_state_dict({"weight": torch.randn(out_f, in_f) * 0.02, "bias": torch.randn(out_f) * 0.05})

        # Read the COLUMN shard back off device 0 rather than re-deriving how ColParallelLinear split
        # it: the reference then matches device 0's output whatever the sharding convention is.
        w_local = dev0(m.weight.data).squeeze()  # [K, N_local]
        b_local = dev0(m.bias.data).squeeze()  # [N_local]
        dump[f"ref:{name}"] = x_full @ w_local + b_local

        out = m(x, parallel_config=pc)
        outs = out if isinstance(out, (list, tuple)) else [out]
        # chunks split the local N; re-join them in order to recover [M, N_local].
        dump[f"got:{name}"] = torch.cat([dev0(o).squeeze() for o in outs], dim=-1)
        for o in outs:
            ttnn.deallocate(o)

    os.makedirs(DUMP_DIR, exist_ok=True)
    path = f"{DUMP_DIR}/{arm}.pt"
    torch.save(dump, path)
    logger.info(f"NSPLIT arm={arm} dumped -> {path}")


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    if torch.equal(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _rmse_sigma(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref, got = ref.flatten().float(), got.flatten().float()
    denom = ref.std()
    if denom == 0:
        return float("inf") if not torch.equal(ref, got) else 0.0
    return ((ref - got).pow(2).mean().sqrt() / denom).item()


def compare(split_log: str) -> int:
    """Check the four dumps against each other. Returns a process exit code."""
    arms = {a: torch.load(f"{DUMP_DIR}/{a}.pt") for a in ("base", "base2", "split", "mutant")}

    # The split arm must have ACTUALLY applied the split on every site with headroom — otherwise
    # "bit-exact" would just mean "the knob did nothing".
    with open(split_log) as f:
        decisions = [ln.split("AGMM N split: ")[1].strip() for ln in f if "AGMM N split: " in ln]
    split_ntiles = {d.split("N_tiles=")[1].split()[0] for d in decisions if "-> SPLIT" in d}
    expected_ntiles = {"96", "32", "128"}  # qkv, out, ff1
    failures = []
    if not expected_ntiles.issubset(split_ntiles):
        failures.append(
            f"split arm did not apply the split to all splittable sites: "
            f"took SPLIT on N_tiles={sorted(split_ntiles)}, expected superset of {sorted(expected_ntiles)}"
        )

    for name in SITES:
        ref = arms["base"][f"ref:{name}"]
        base = arms["base"][f"got:{name}"]
        p_ref, r_ref = _pcc(ref, base), _rmse_sigma(ref, base)
        exact_rep = torch.equal(base, arms["base2"][f"got:{name}"])
        split = arms["split"][f"got:{name}"]
        exact_split = torch.equal(base, split)
        mut = arms["mutant"][f"got:{name}"]
        exact_mut = torch.equal(base, mut)
        p_mut, r_mut = _pcc(base, mut), _rmse_sigma(base, mut)

        print(
            f"NSPLIT {name:5s} | vs_torch pcc={p_ref:.6f} rmse/s={r_ref:.5f} "
            f"| repeat_bitexact={exact_rep} | split_bitexact={exact_split} "
            f"| mutant_bitexact={exact_mut} pcc={p_mut:.6f} rmse/s={r_mut:.5f}"
        )

        if p_ref < 0.99 or r_ref > 0.05:
            failures.append(f"{name}: uniform baseline is wrong vs torch (pcc={p_ref:.6f} rmse/s={r_ref:.5f})")
        if not exact_rep:
            failures.append(f"{name}: same path twice is not bit-exact — no stable floor to gate against")
        if not exact_split:
            failures.append(f"{name}: SPLIT IS NOT BIT-EXACT vs uniform")
        if name in SPLITTABLE and exact_mut:
            failures.append(f"{name}: MUTANT WAS NOT CAUGHT — the gate does not bite")
        if name not in SPLITTABLE and not exact_mut:
            failures.append(f"{name}: expected fallback-to-uniform (no split room), but output moved")

    if failures:
        print("\nN-SPLIT GATE: FAIL\n  " + "\n  ".join(failures))
        return 1
    print("\nN-SPLIT GATE: PASS (split bit-exact on all sites; mutant caught on every splittable site)")
    return 0


if __name__ == "__main__":
    sys.exit(compare(sys.argv[1]))
