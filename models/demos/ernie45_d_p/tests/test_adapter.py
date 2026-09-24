# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gate P2.16: ERNIE through the prefill-server engine contract (adapter + runtime), in one process.

Drives the model exactly as common/prefill's engine does: get_adapter("ernie45_d_p") -> allocate_kv_cache ->
build_runtime -> set_layer_completion_sink -> compile -> prefill_chunk(slot, actual_start, actual_end, request_id)
per chunk -> build_kv_chunk_table. Then runs the PRODUCER's own KV read-back + PCC
(prefill_producer._read_slot_kv_and_check_pcc: read_dram_umd through the published table) against the golden.
Uses user slot 1 of 2 so slot addressing is exercised.
"""

import os
import tempfile

TASK = "P2.16"
SEQ, CHUNK, SLOT, USERS = 4096, 2048, 1, 2
os.environ.update(
    PREFILL_MODEL="ernie45_d_p",
    PREFILL_SP="1",
    PREFILL_TP="4",
    PREFILL_CHUNK_SIZE=str(CHUNK),
    PREFILL_MAX_SEQ_LEN=str(SEQ),
    PREFILL_NUM_LAYERS="28",
)

import ttnn  # noqa: E402
from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter  # noqa: E402
from models.demos.ernie45_d_p.bringup import metrics  # noqa: E402
from models.demos.ernie45_d_p.tests.conftest import mesh_1x4  # noqa: E402
from models.demos.ernie45_d_p.tt.common import Golden  # noqa: E402


def _device_map(mesh):
    out = {}
    for c in range(mesh.shape[1]):
        fid = mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, c))
        out[(int(fid.mesh_id), int(fid.chip_id))] = int(
            ttnn.cluster.get_chip_unique_id_from_fabric_node_id(int(fid.mesh_id), int(fid.chip_id))
        )
    return out


@mesh_1x4
def test_engine_contract(mesh_device, record):
    from models.demos.common.prefill.runners import prefill_producer as producer

    task = record.task
    G = Golden(SEQ, CHUNK)
    adapter = get_adapter("ernie45_d_p")
    hf = adapter.load_hf_config()
    params = PrefillRunParams(
        mesh_shape=(1, 4),
        num_layers=adapter.model_config.NUM_LAYERS,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=SEQ,
        chunk_size=CHUNK,
        num_users=USERS,
        capacity_factor=1,
        num_links=1,
        gate_mode_name=adapter.default_gate_mode,
        kv_only_last_layer=False,
        weight_cache_path=adapter.weight_cache_path((1, 4)),
    )
    kv = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf, params=params)
    rt = adapter.build_runtime(mesh_device=mesh_device, hf_config=hf, params=params)
    acks = []
    rt.set_layer_completion_sink(lambda layer, rid: acks.append((layer, rid)))
    rt.compile(kv)
    acks.clear()

    tokens = G.tokens().tolist()
    for c in range(SEQ // CHUNK):
        inp = rt.make_chunk_input(tokens[c * CHUNK : (c + 1) * CHUNK])
        out = rt.prefill_chunk(inp, kv, slot_id=SLOT, actual_start=c * CHUNK, actual_end=(c + 1) * CHUNK, request_id=c)
        assert out is None
    ttnn.synchronize_device(mesh_device)

    failed = []
    want_acks = [(layer, c) for c in range(SEQ // CHUNK) for layer in range(28)]
    if acks != want_acks:
        failed.append(f"layer acks {acks[:5]}... != global in-order per chunk")

    with tempfile.TemporaryDirectory() as td:
        path = rt.build_kv_chunk_table(kv, os.path.join(td, "ernie_kv_table.pb"))
        table = ttnn.experimental.disaggregation.import_from_protobuf_file(path)
    if rt.kv_migration_base_address(kv) != table.lookup(0, 0, 0, 0).noc_addr & 0xFFFFFFFF:
        failed.append("migration base address != table entry (layer 0, pos 0, slot 0, K head 0)")

    # Cross-check the two read paths on the same chunks: in-process table read vs the producer's read_dram_umd.
    import torch

    from models.demos.ernie45_d_p.reference.ernie_ref import pcc

    dmap = _device_map(mesh_device)
    D = ttnn.experimental.disaggregation
    for layer in (0, 27):
        gk, _ = G.kv(layer)
        for h in range(4):
            a = torch.cat(
                [
                    ttnn.to_torch(
                        D.tensor_from_bfp8_bytes(
                            table.read_device_chunk(layer=layer, position=p, slot=SLOT, config_id=h), [1, 1, 32, 128]
                        )
                    ).float()[0, 0]
                    for p in range(0, SEQ, 32)
                ]
            )
            b = producer._read_kv_slice(table, dmap, h, layer, SLOT, SEQ, 128, producer._decode_bfp8_chunk).float()
            print(
                f"L{layer} K head {h}: in-process vs golden {pcc(a, gk[h]):.6f} | producer-path vs golden {pcc(b, gk[h]):.6f} "
                f"| paths equal {torch.equal(a, b)} maxdiff {(a - b).abs().max():.3e}"
            )
            if not torch.allclose(a, b):
                failed.append(f"producer read path != in-process table read (L{layer} K h{h})")

    mins = producer._read_slot_kv_and_check_pcc(table, dmap, SLOT, SEQ, str(G.dir))
    # Independent score of the same producer read path. Guards against a stubbed/lying comp_pcc: under
    # run_safe_pytest's precompile mode tests/plugins/up_front_collect.py rebinds comp_pcc -> (True, 0.999999)
    # and the stub leaks into lazily-imported call sites in the real pass (hence --no-precompile on this gate).
    ind = {"k": 1.0, "v": 1.0}
    for layer in range(28):
        gk, gv = G.kv(layer)
        for kind, base, gold in (("k", 0, gk), ("v", 4, gv)):
            dev = torch.stack(
                [
                    producer._read_kv_slice(table, dmap, base + h, layer, SLOT, SEQ, 128, producer._decode_bfp8_chunk)
                    for h in range(4)
                ]
            )
            ind[kind] = min(ind[kind], pcc(dev.float(), gold.float()))
    metrics.record(task, "pcc_kv_independent_min", min(ind.values()))
    agree = max(abs(mins["k"] - ind["k"]), abs(mins["v"] - ind["v"]))
    metrics.record(task, "producer_vs_independent_pcc_gap", agree)
    if agree > 1e-4:
        failed.append(f"producer comp_pcc {mins} disagrees with independent PCC {ind}")
    metrics.record(task, "pcc_producer_kv_k", mins["k"])
    metrics.record(task, "pcc_producer_kv_v", mins["v"])
    metrics.record(task, "engine_checks_failed", len(failed))
    print(f"producer read-back KV PCC: K={mins['k']:.6f} V={mins['v']:.6f}; independent {ind}; failed: {failed}")
    assert not failed and min(mins.values()) >= 0.97
