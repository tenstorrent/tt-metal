# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-only (one Galaxy row, TP=8; no weights loaded): the packed speculative inputs' derive programs
(tt/spec_inputs.py) write a verify's and an MTP draft's trace inputs exactly as the host built them input by input.

For each step the host copies the packed upload; the derive program runs eagerly, then replayed from a trace of it.
Every derived buffer is read back from every chip of the row and compared bit for bit with the host values
(test_spec_inputs_host.py's expectations: TtMiMoModel.prepare_decode_inputs_host, kv.owned_positions / owner_mask,
serial_positions, the MTP's plan_mtp / mtp_cache_position inputs). With the decode RoPE tables (whole 192-wide rows,
or compact: the 64 rotated dims plus the 1 / 0 tail, as past 128K context), the verify's and the draft's table rows
at the derived indices (TtMiMoModel.rope_table_rows) must equal the host cos / sin they used to upload.

test_rope_table_rows_1m_tp8: the same rows of 1M-position tables (the MTP serving entry's context) near 0, 128K and
1M - 1, past it into the table margin, and at random positions with idle slots, eager and traced."""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.reference.spec_decode import SPEC_SLOTS, DraftSource, plan_mtp
from models.experimental.mimo_v2_6_flash.tests.common import GALAXY_DEVICE_PARAMS, dp_submeshes, model_path
from models.experimental.mimo_v2_6_flash.tests.test_spec_inputs_host import (
    B,
    mtp_host_values,
    random_positions,
    verify_host_values,
)
from models.experimental.mimo_v2_6_flash.tt import kv as kv_lib
from models.experimental.mimo_v2_6_flash.tt.common import MeshPlan
from models.experimental.mimo_v2_6_flash.tt.model import ROPE_TABLE_MARGIN, TtMiMoModel
from models.experimental.mimo_v2_6_flash.tt.rope import MiMoRotarySetup
from models.experimental.mimo_v2_6_flash.tt.spec_inputs import SpecInputs

MAX_SEQ_LEN = 4096
DEVICE_PARAMS = {**GALAXY_DEVICE_PARAMS, "trace_region_size": 20_000_000}


def chips(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def same(t, want, what):
    """Every chip's logical values of device tensor ``t`` equal ``want`` (one tensor, or one per chip)."""
    for c, got in enumerate(chips(t)):
        w = want[c] if isinstance(want, list) else want
        assert torch.equal(
            got.reshape(w.shape).to(w.dtype), w
        ), f"{what}: chip {c}\n{got.flatten()[:8]}\n{w.flatten()[:8]}"


def model_stub(plan, cfg, rope_tables, max_seq_len=MAX_SEQ_LEN):
    """The TtMiMoModel attributes prepare_decode_inputs_host, setup_decode_rope_tables and rope_table_rows read;
    ``rope_tables`` None, "whole" or "compact"."""
    rope_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
    stub = SimpleNamespace(
        plan=plan,
        cfg=cfg,
        mesh_device=plan.mesh_device,
        max_batch=B,
        max_seq_len=max_seq_len,
        decode_trace=None,
        rope_tables=None,
        rope_table_tails=None,
        rope_setups={
            "swa": MiMoRotarySetup(plan.mesh_device, B, cfg.head_dim, rope_dim, cfg.swa_rope_theta),
            "full": MiMoRotarySetup(plan.mesh_device, B, cfg.head_dim, rope_dim, cfg.rope_theta),
        },
    )
    if rope_tables is not None:
        TtMiMoModel.setup_decode_rope_tables(stub, compact=rope_tables == "compact")
    return stub


def same_rows(stub, idx, tables, host, what):
    """The table rows at ``idx`` (TtMiMoModel.rope_table_rows) equal the host cos / sin ``host`` once uploaded."""
    mesh = stub.mesh_device
    for k, (row, h) in enumerate(zip(TtMiMoModel.rope_table_rows(stub, idx, tables), host)):
        uploaded = ttnn.to_device(h, mesh)
        same(row, chips(uploaded)[0], f"{what} table {k}")
        row.deallocate(True)
        uploaded.deallocate(True)


def replayed(mesh, run):
    """Run ``run`` eagerly (compiling it), capture it, and return a function replaying the capture."""
    run()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    run()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    return lambda: ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)


@pytest.mark.parametrize("rope_tables", [None, "whole", "compact"])
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_spec_inputs_tp8(mesh_device, rope_tables):
    cfg = MiMoConfig.from_model_path(model_path())
    plan = MeshPlan(dp_submeshes(mesh_device)[0])
    mesh = plan.mesh_device
    rep = kv_lib.kv_replication(plan.tp, cfg.num_key_value_heads)
    stub = model_stub(plan, cfg, rope_tables)
    rope = rope_tables is not None
    spec = SpecInputs(plan, B, rep)
    g = torch.Generator().manual_seed(0)

    # Verify: the trace's inputs, allocated as capture_verify_trace does.
    idle = torch.full((B,), -1, dtype=torch.int32)
    host = TtMiMoModel.prepare_decode_inputs_host(stub, torch.zeros(B, dtype=torch.int32), idle, serial=True)
    inputs = [ttnn.to_device(t, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in host]
    replay = replayed(mesh, lambda: spec.derive_verify(inputs, rope))
    cases = [(torch.zeros(B, dtype=torch.int32), idle)]
    cases += [
        (torch.randint(0, 152576, (B,), generator=g, dtype=torch.int32), random_positions(g, a, 0, MAX_SEQ_LEN))
        for a in (1, 4, 17, B)
    ]
    for k, (tokens, positions) in enumerate(cases):
        for run in (lambda: spec.derive_verify(inputs, rope), replay):
            ttnn.copy_host_to_device_tensor(spec.verify_host(tokens, positions), spec.verify_packed)
            run()
            want = verify_host_values(tokens, positions, rope)
            same(inputs[0], want[0]["tokens"], f"case {k} tokens")
            same(inputs[1], want[0]["cur_pos"], f"case {k} cur_pos")
            same(inputs[6], (positions >= 0).to(torch.bfloat16).reshape(1, 1, -1, 1), f"case {k} mask")
            same(inputs[7], [w["pos_owned"] for w in want], f"case {k} pos_owned")
            owner = kv_lib.owner_mask(positions, plan.tp, rep)
            same(inputs[8], [owner[c] for c in range(plan.tp)], f"case {k} owner_mask")
            if rope:
                same(inputs[9], want[0]["rope_pos"], f"case {k} rope_pos")
                # The four rows the verify reads equal the cos / sin the host would upload for these positions.
                host_rope = TtMiMoModel.prepare_decode_inputs_host(stub, tokens, positions)[2:6]
                same_rows(stub, inputs[9], stub.rope_tables, host_rope, f"case {k} verify RoPE")
            for j, s in enumerate(want[0]["serial"]):
                same(inputs[len(inputs) - SPEC_SLOTS + j], s, f"case {k} serial {j}")

    # MTP draft: tokens, fill, SPEC_SLOTS serial, step2, step3 (TtMiMoMTP._idle_inputs), and the RoPE indices.
    def buf(dtype, shape, value):
        return kv_lib.replicated(plan, torch.full(shape, value, dtype=torch.int32), dtype)

    tokens_tt = buf(ttnn.uint32, (1, B), 0)
    ints = [buf(ttnn.int32, (B,), -1) for _ in range(SPEC_SLOTS + 3)]
    fill, serial, step2, step3 = ints[0], ints[1 : 1 + SPEC_SLOTS], ints[-2], ints[-1]
    idx = [buf(ttnn.uint32, (1, B), 0) for _ in range(3)] if rope else None
    run_mtp = lambda: spec.derive_mtp(tokens_tt, fill, serial, step2, step3, idx)  # noqa: E731
    replay = replayed(mesh, run_mtp)
    drafts = [(None, None)]
    for bonus in (1, 255, 256, 257, 511, 512, MAX_SEQ_LEN - 4):
        drafts.append(
            ([11, 12, 13, 14], plan_mtp(DraftSource(row=0, slots=[0, 1, 2, 3]), [bonus - 2, bonus - 1, bonus, -1], 3))
        )
        drafts.append(([0, 0, 15, 0], plan_mtp(DraftSource(row=0, slots=[2]), [bonus, -1, -1, -1], 1)))
    for k, (toks, mplan) in enumerate(drafts):
        for run in (run_mtp, replay):
            ttnn.copy_host_to_device_tensor(spec.mtp_host(toks, mplan), spec.mtp_packed)
            run()
            want = mtp_host_values(toks, mplan)
            same(tokens_tt, want["tokens"], f"draft {k} tokens")
            same(fill, want["fill"], f"draft {k} fill")
            for j in range(SPEC_SLOTS):
                same(serial[j], want["serial"][j], f"draft {k} serial {j}")
            same(step2, want["step2"], f"draft {k} step2")
            same(step3, want["step3"], f"draft {k} step3")
            if rope:
                for p, (i, w) in enumerate(zip(idx, want["rope_idx"])):
                    same(i, w, f"draft {k} rope index {p}")
                    # The table rows the draft reads equal the host cos / sin it uploaded before.
                    host_pos = w.reshape(-1).to(torch.int32)
                    host_rope = stub.rope_setups["swa"].decode_host_inputs(host_pos)
                    same_rows(stub, i, stub.rope_tables[:2], host_rope, f"draft {k} RoPE pass {p}")


@pytest.mark.parametrize("compact", [True, False], ids=["compact", "whole"])
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_rope_table_rows_1m_tp8(mesh_device, compact):
    """1M-position tables (537 MB per chip compact, 1.61 GB whole): the rows a step reads at its RoPE index
    (max(position, 0)) equal the cos / sin the host uploads for its positions, all four tables, eager and traced."""
    max_seq_len = 1 << 20
    cfg = MiMoConfig.from_model_path(model_path())
    plan = MeshPlan(dp_submeshes(mesh_device)[0])
    mesh = plan.mesh_device
    stub = model_stub(plan, cfg, "compact" if compact else "whole", max_seq_len)
    end = max_seq_len + ROPE_TABLE_MARGIN
    g = torch.Generator().manual_seed(0)
    cases = [torch.arange(base, base + B, dtype=torch.int32) for base in (0, 131072 - 16, max_seq_len - B, end - B)]
    cases += [random_positions(g, a, 0, max_seq_len) for a in (1, 4, 17, B)]
    idx = kv_lib.replicated(plan, torch.zeros(1, B, dtype=torch.int32), ttnn.uint32)
    made = []

    def gather():
        made[:] = TtMiMoModel.rope_table_rows(stub, idx, stub.rope_tables)

    gather()
    for t in made:
        t.deallocate(True)
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    gather()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    traced = list(made)
    for k, positions in enumerate(cases):
        host_idx = kv_lib.replicated(plan, positions.clamp(min=0).reshape(1, B), ttnn.uint32, device=False)
        ttnn.copy_host_to_device_tensor(host_idx, idx)
        # The host cos / sin of these positions (decode_host_inputs: what the step would upload without the tables).
        swa, full = stub.rope_setups["swa"], stub.rope_setups["full"]
        host = [*swa.decode_host_inputs(positions), *full.decode_host_inputs(positions)]
        same_rows(stub, idx, stub.rope_tables, host, f"case {k} eager")
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        for j, (row, h) in enumerate(zip(traced, host)):
            uploaded = ttnn.to_device(h, mesh)
            same(row, chips(uploaded)[0], f"case {k} traced table {j}")
            uploaded.deallocate(True)
    ttnn.release_trace(mesh, tid)
