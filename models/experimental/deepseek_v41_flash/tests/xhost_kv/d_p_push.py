# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Phase 2 of the fabric push (``PREFILL_D_P_FABRIC_PUSH.md``): metal_2's prefill state pushed from its device caches
into a decode session over TT-Fabric (``tt/prefill/d_p_push.py``), checked against the host path. Two ranks of one
``tt-run`` job, as the servers will run:

* rank 0 (host 40, prefill): ``DPPrefill`` over the 8 x 4 mesh, a seeded random prompt of ``--tokens`` prefilled
  into slot 0; on the decode side's targets it pushes the slot into session A, then replies with the same slot's
  ``host_states`` (the host path's input) over TCP.
* rank 1 (host 41, decode): ``DeepSeekV41Model`` (layers 0-20 by default), two sessions; sends session A's
  ``remote_commit_targets``, runs ``finish_remote_commit`` on the reply, commits the host states into session B
  through ``commit_host_prefill_state`` and compares A with the states and with B byte for byte on every chip of
  every stage (``decode_commit.check``).

Launch from the repo root on host 40 (host 41 has the same checkout at the same path)::

    DEEPSEEK_V4_FABRIC_PAYLOAD=6208 tt-run \\
      --rank-binding tests/tt_metal/distributed/config/dual_bh_galaxy_8x4_hosts_40_41_two_mesh_rank_bindings.yaml \\
      --mpi-args "--host DC16-2-BG-0202-u20-40,DC16-2-BG-0202-u13-41 \\
        --map-by rankfile:file=$PWD/tests/tt_metal/distributed/config/dual_8x4_hosts_40_41.rankfile \\
        --mca btl self,tcp --mca btl_tcp_if_include ens5f0np0.1 --bind-to none --tag-output" \\
      bash -c 'cd $TT_METAL_HOME && export PYTHONPATH=$TT_METAL_HOME LD_LIBRARY_PATH=$TT_METAL_HOME/build/lib \\
        BRINGUP_SPEC=models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml BRINGUP_HF=... && \\
        exec python_env/bin/python models/experimental/deepseek_v41_flash/tests/xhost_kv/d_p_push.py --tokens 16384'

(``BRINGUP_*`` are not forwarded by ``tt-run``: set them inside the rank.) Rank 1 exits non-zero on a mismatch.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time

import torch

import ttnn
from models.experimental.deepseek_v41_flash.tests.xhost_kv.common import (
    MESH_SHAPE,
    PORT,
    RECEIVER_RANK,
    SENDER_RANK,
    connect,
    init_rank,
    log,
)
from models.experimental.deepseek_v41_flash.tests.xhost_kv.decode_commit import check
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR
from models.experimental.deepseek_v41_flash.tt.prefill.d_p_wire import _states, recv_msg, send_msg, send_states


def run_prefill(args) -> int:
    from models.demos.common.bringup.testing.harness import close_mesh, open_mesh
    from models.experimental.deepseek_v41_flash.tt.prefill.d_p_bridge import DPPrefill
    from models.experimental.deepseek_v41_flash.tt.prefill.d_p_push import push

    init_rank(SENDER_RANK, "prefill")
    mesh, fab = open_mesh()
    try:
        log("prefill", f"mesh {tuple(mesh.shape)} open, packet payload {ttnn.get_tt_fabric_max_payload_size_bytes()} B")
        d_p = DPPrefill(mesh, max_seq=args.tokens, slots=1)
        gen = torch.Generator().manual_seed(args.seed)
        ids = torch.randint(0, d_p.cfg.vocab_size, (args.tokens,), generator=gen).tolist()
        t0 = time.perf_counter()
        d_p.prefill(ids)
        log("prefill", f"{args.tokens} tokens prefilled in {time.perf_counter() - t0:.1f} s")
        with socket.create_server(("0.0.0.0", args.port)) as listener:
            sock, _ = listener.accept()
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            req, _ = recv_msg(sock)
            assert req["op"] == "push", req
            targets = req["targets"]
            if targets["seq_len"] != args.tokens:
                sys.exit(f"prefill: decode wants {targets['seq_len']} tokens, the slot holds {args.tokens}")
            times = {}
            t0 = time.perf_counter()
            done = push(d_p, targets, times=times)
            done["total"] = time.perf_counter() - t0
            done["times"] = times
            send_msg(sock, {"done": True, **done})
            log(
                "prefill",
                f"pushed {done['pages']} pages, {done['bytes'] / 2**20:.1f} MiB: device {done['seconds'] * 1e3:.1f} ms "
                f"({done['bytes'] / done['seconds'] / 1e9:.2f} GB/s incl. launch), total {done['total'] * 1e3:.1f} ms, "
                + ", ".join(f"{k} {v * 1e3:.1f} ms" for k, v in times.items()),
            )
            t0 = time.perf_counter()
            host = d_p.host_states(args.tokens)
            send_states(sock, host, time.perf_counter() - t0)
            log("prefill", f"host states read back in {time.perf_counter() - t0:.1f} s")
            recv_msg(sock)  # the decode side's bye: keep the job (and this mesh's fabric) up until it has checked
            sock.close()
    finally:
        close_mesh(mesh, fab)
    return 0


def run_decode(args) -> int:
    import contextlib

    from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import (
        _build_rope,
        _construct_model,
        _traced_max_seq,
    )
    from models.experimental.deepseek_v4_flash.demo.chat_cli import open_mesh_device
    from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config, set_active_system_config
    from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

    init_rank(RECEIVER_RANK, "decode")
    os.environ.setdefault("DEEPSEEK_V4_DECODE_LAYERS", "21")
    mesh_cm = open_mesh_device(None, mesh_shape=MESH_SHAPE, l1_small_size=args.l1_small_size)
    with mesh_cm as mesh, contextlib.ExitStack() as stack:
        system_config = load_system_config(mesh_device=mesh)
        set_active_system_config(system_config)
        t0 = time.perf_counter()
        model, lm_head, loader, config = _construct_model(
            mesh, stack, loader=DeepseekV4WeightLoader(args.model_dir), system_config=system_config
        )
        max_seq = _traced_max_seq(config, args.tokens + model.sliding_window, model.num_layers)
        model.prepare_static_decode(
            _build_rope(config, loader.snapshot_dir, max_seq), max_seq, lm_head=lm_head, num_sessions=2
        )
        log("decode", f"{model.num_layers}-layer decode model ready in {time.perf_counter() - t0:.0f} s")

        ring_layers = [li for li in range(model.num_layers) if li < config.candidate_source_layer_id]
        sid_b = model.open_session()
        sid_a = model.open_session()
        targets = model.remote_commit_targets(sid_a, args.tokens, ring_layers)
        log("decode", f"session {sid_a} (slot {targets['slot']}): {len(targets['stages'])} stage(s) to write")
        sock = connect(args.prefill, args.port)
        send_msg(sock, {"op": "push", "targets": targets})
        done, _ = recv_msg(sock)
        assert done.get("done"), done
        t0 = time.perf_counter()
        model.finish_remote_commit(sid_a)
        finish = time.perf_counter() - t0
        t0 = time.perf_counter()
        host = _states(*recv_msg(sock))
        t1 = time.perf_counter()
        model.commit_host_prefill_state(host, sid_b)
        t2 = time.perf_counter()
        log(
            "decode",
            f"push path: push {done['total']:.2f} s + finish_remote_commit {finish:.2f} s; host path: readback "
            f"{host['seconds']:.2f} s + send {t1 - t0:.2f} s + commit_host_prefill_state {t2 - t1:.2f} s "
            f"(session {sid_b}, {host['bytes'] / 2**20:.1f} MiB)",
        )
        bad = check(model, host, sid_a, sid_b, set(ring_layers))
        send_msg(sock, {"op": "bye"})
        sock.close()
        log(
            "decode",
            f"pushed {done['bytes'] / 2**20:.1f} MiB, device {done['seconds'] * 1e3:.1f} ms, "
            f"total {done['total'] * 1e3:.1f} ms; " + ("PASS: byte-exact" if not bad else f"FAIL: {bad} bad row(s)"),
        )
        return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefill", default="10.82.97.40", help="rank 0's address (host 40)")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--tokens", type=int, default=16384, help="prompt length (a multiple of 128)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="the decode model's checkpoint (rank 1)")
    parser.add_argument("--l1-small-size", type=int, default=10240, help="as the decode server's default")
    args = parser.parse_args()
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    rank = int(ttnn.distributed_context_get_rank())
    return run_prefill(args) if rank == SENDER_RANK else run_decode(args)


if __name__ == "__main__":
    sys.exit(main())
