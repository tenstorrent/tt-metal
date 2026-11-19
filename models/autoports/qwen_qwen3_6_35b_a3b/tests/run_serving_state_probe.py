# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Focused reload/remap/readback probe; device execution must be serialized.

Use --cpu-only for command validation without importing Torch or TTNN.
Device run: timeout 1800 python -m
models.autoports.qwen_qwen3_6_35b_a3b.tests.run_serving_state_probe
This checks primitives, not full-model serving readiness. Source-only at creation.
"""

import argparse
import ast
import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tt/serving_state.py"


def cpu_checks():
    class HostList:
        def __init__(self, value):
            self.value = value

        def reshape(self, *shape):
            return self

        def tolist(self):
            return list(self.value)

    class TorchStub:
        as_tensor = staticmethod(HostList)

    names = {"ReloadCommands", "normalize_remap", "ReloadExecutor"}
    tree = ast.parse(SOURCE.read_text())
    namespace = {"dataclass": dataclass, "torch": TorchStub()}
    isolated = ast.Module(body=[node for node in tree.body if getattr(node, "name", None) in names], type_ignores=[])
    exec(compile(isolated, str(SOURCE), "exec"), namespace)
    Commands, Executor, normalize = [
        namespace[name] for name in ("ReloadCommands", "ReloadExecutor", "normalize_remap")
    ]
    legal = illegal = 0
    for bits in itertools.product((False, True), repeat=4):
        valid = not (bits[0] and bits[1]) and (not bits[3] or bits[0])
        try:
            Commands(*bits)
        except ValueError:
            assert not valid
            illegal += 1
        else:
            assert valid
            legal += 1
    for key in ("reset_batch", "decode_layout_changed"):
        try:
            Commands.from_kwargs({key: False})
        except TypeError:
            pass
        else:
            raise AssertionError(key)
    for bad in ([0, 0], [0.5, 1.5], [-1, 1], [True, False], [0]):
        try:
            normalize(bad, 2)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)

    events = []

    class Slots:
        capacity = 2
        poisoned = False

        def apply(self, order):
            events.append(("remap", order))

    executor = Executor(Slots(), require_drained=lambda: events.append("drain"))
    executor.execute(
        Commands(True, False, True, True),
        slot_remap=[1, 0],
        reload_forward=lambda: events.append("inputs"),
        reload_parameters=lambda: events.append("params"),
        reset_sampling=lambda: events.append("reset"),
        forward=lambda: events.append("forward"),
    )
    assert events == ["drain", ("remap", (1, 0)), "inputs", "params", "reset", "forward"]
    events.clear()
    executor.execute(
        Commands(False, True, False, False),
        reload_pages=lambda: events.append("pages"),
        forward=lambda: events.append("forward"),
    )
    assert events == [("remap", (0, 1)), "pages", "forward"]
    events.clear()
    executor.execute(Commands(False, False, False, False), forward=lambda: events.append("forward"))
    assert events == [("remap", (0, 1)), "forward"]
    try:
        executor.execute(
            Commands(), reload_forward=lambda: (_ for _ in ()).throw(RuntimeError("mutation")), forward=lambda: None
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Failed mutation accepted")
    assert executor.poisoned and executor.slots.poisoned
    return {
        "status": "PASS",
        "legal_commands": legal,
        "illegal_commands": illegal,
        "checks": ["invalid_remaps", "legacy_rejection", "ordering", "page_only", "resident", "failure_poisoning"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "doc/full_model/serving_state_probe")
    args = parser.parse_args()
    cpu = cpu_checks()
    if args.cpu_only:
        print(json.dumps(cpu, indent=2))
        return
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
    assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
    assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
    import torch

    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.serving_state import DecodeOutputPool, PrefillLogitRows, SlotState

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "RUNNING",
        "scope": "serving state primitives only",
        "cpu": cpu,
        "mesh": [1, 4],
        "tracker": True,
        "skip_program_cache": False,
        "captures": 0,
        "releases": 0,
        "steps": [],
        "sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), SOURCE, ROOT / "tt/slot_state.py")
        },
    }
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=10_000_000)
    trace = None

    def upload(value, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, sharded=False):
        mapper = ttnn.ShardTensorToMesh(mesh, dim=-1) if sharded else ttnn.ReplicateTensorToMesh(mesh)
        return ttnn.from_torch(
            value.contiguous(),
            dtype=dtype,
            layout=layout,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def refresh(value, target, sharded=False):
        mapper = ttnn.ShardTensorToMesh(mesh, dim=-1) if sharded else ttnn.ReplicateTensorToMesh(mesh)
        host = ttnn.from_torch(value.contiguous(), dtype=target.dtype, layout=target.layout, mesh_mapper=mapper)
        ttnn.copy_host_to_device_tensor(host, target)

    def equal_all(tensor, expected, label):
        locals_ = ttnn.get_device_tensors(tensor)
        assert len(locals_) == 4
        for chip, local in enumerate(locals_):
            actual = ttnn.to_torch(local).contiguous()
            expected_ = expected.contiguous()
            assert actual.shape == expected_.shape, (label, chip, actual.shape, expected_.shape)
            if expected_.dtype == torch.bfloat16:
                assert torch.equal(actual.view(torch.int16), expected_.view(torch.int16)), (label, chip)
            elif expected_.dtype == torch.float32:
                assert torch.equal(actual.view(torch.int32), expected_.view(torch.int32)), (label, chip)
            else:
                assert torch.equal(actual.to(torch.int64), expected_.to(torch.int64)), (label, chip)

    try:
        state = SlotState(mesh, 4)
        group_hosts = [
            (torch.arange(64, dtype=torch.int32) + 0x3F800001 + i * 65536).view(torch.float32).reshape(1, 64)
            for i in range(4)
        ]
        groups = [upload(value, ttnn.float32) for value in group_hosts]
        state.register_group("recurrent_toy", groups)
        packed = []
        for name, value, dtype, layout in (
            ("seed", torch.arange(32, dtype=torch.int32) + 0x12345000, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            ("parameter", torch.arange(32).to(torch.bfloat16) / 16, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            (
                "penalty",
                (torch.arange(32 * 96, dtype=torch.int32) * 1009).reshape(32, 96),
                ttnn.int32,
                ttnn.TILE_LAYOUT,
            ),
            (
                "mask",
                (torch.arange(32 * 64, dtype=torch.int32) % 1000).to(torch.bfloat16).reshape(1, 1, 32, 64),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
            ),
        ):
            tensor = upload(value, dtype, layout)
            state.register_packed(name, tensor)
            packed.append([name, tensor, value.clone()])
        metadata = {"request": ["r0", "r1", "r2", "r3"], "seed": list(range(32))}
        for name in metadata:
            state.register_host(
                name, lambda name=name: metadata[name], lambda value, name=name: metadata.__setitem__(name, value)
            )
        token_host = torch.arange(32, dtype=torch.int32).reshape(1, 1, 1, 32)
        token_source = upload(token_host, ttnn.uint32)
        token_output = upload(torch.zeros_like(token_host), ttnn.uint32)
        logits_host = (torch.arange(32 * 128).reshape(1, 1, 32, 128) % 997).to(torch.bfloat16)
        logits_source = upload(logits_host, ttnn.bfloat16, ttnn.TILE_LAYOUT, True)
        logits_output = upload(torch.zeros_like(logits_host), ttnn.bfloat16, ttnn.TILE_LAYOUT, True)
        singleton_host = logits_host[:, :, :1, :].clone()
        singleton = upload(singleton_host, ttnn.bfloat16, ttnn.TILE_LAYOUT, True)
        singleton_3d = ttnn.reshape(singleton, (1, 1, singleton.shape[-1]))
        row_staging = PrefillLogitRows(mesh)
        pool = DecodeOutputPool(
            mesh, token_template=token_output, logits_template=logits_output, vocab_size=123, capacity=4, depth=2
        )
        state.prepare()
        pool.prepare(token_template=token_output, logits_template=logits_output)
        row_staging.prepare(source=singleton, destination=logits_output)
        ttnn.copy(token_source, token_output)
        ttnn.copy(logits_source, logits_output)
        ttnn.synchronize_device(mesh)
        prepared = mesh.num_program_cache_entries()
        assert prepared > 0

        def check_cache():
            assert mesh.num_program_cache_entries() == prepared, (prepared, mesh.num_program_cache_entries())

        def check_staging():
            refresh(torch.zeros_like(logits_host), logits_output, True)
            expected_rows = torch.zeros_like(logits_host)
            for slot, source in ((7, singleton), (30, singleton_3d), (0, singleton)):
                row_staging.copy(source, logits_output, slot)
                expected_rows[:, :, slot : slot + 1, :] = singleton_host
                for chip, local in enumerate(ttnn.get_device_tensors(logits_output)):
                    assert torch.equal(ttnn.to_torch(local), expected_rows[..., chip * 32 : (chip + 1) * 32])
                check_cache()
            result["steps"].append({"prefill_logit_rows": [7, 30, 0], "all_four_chips": True})

        check_staging()

        for order in ((1, 0, 3, 2), (2, 0, 3, 1), (0, 1, 2, 3)):
            old = group_hosts
            state.apply(order)
            group_hosts = [old[i].clone() for i in order]
            padded = list(order) + list(range(4, 32))
            for tensor, expected in zip(groups, group_hosts):
                equal_all(tensor, expected, "group")
            for item in packed:
                name, tensor, value = item
                rows = value.reshape(32, -1)[padded].clone().reshape(value.shape)
                equal_all(tensor, rows, name)
                item[2] = rows
            assert metadata["request"] == ["r" + str(i) for i in metadata["seed"][:4]]
            check_cache()
            result["steps"].append({"remap": list(order), "all_four_chips": True})
        # Replays write preallocated outputs; output snapshots were all allocated
        # and their copy/read variants warmed before this first capture.
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        ttnn.copy(token_source, token_output)
        ttnn.copy(logits_source, logits_output)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        result["captures"] = 1
        rows = (3, 0, 2)
        for mode in (True, False):
            handles, expected = [], []
            for step in range(2):
                tokens = token_host + step * 1000
                logits = logits_host + step * 16
                refresh(tokens, token_source)
                refresh(logits, logits_source, True)
                handle = pool.reserve(rows, is_tokens=mode)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                pool.publish(handle, token_output if mode else logits_output)
                returned, events = pool.read(handle, async_read=True, sample_rows=None if mode else rows)
                assert returned is handle and events
                handles.append(handle)
                expected.append(
                    tokens.flatten()[list(rows)].to(torch.int64).reshape(-1, 1)
                    if mode
                    else logits.reshape(32, 128)[list(rows), :123].float()[:, None, :]
                )
            try:
                pool.reserve(rows, is_tokens=mode)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Full output ring accepted a third submission")
            try:
                pool.require_drained()
            except RuntimeError:
                pass
            else:
                raise AssertionError("Pending outputs allowed a state transition")
            outputs = []
            for handle, want in zip(handles, expected):
                for event in handle.events:
                    ttnn.event_synchronize(event)
                if mode:
                    equal_all(
                        handle.host,
                        (token_host + (1000 if handle is handles[1] else 0)).to(torch.uint32),
                        "asynchronous_token_snapshot",
                    )
                else:
                    local_hosts = ttnn.get_device_tensors(handle.host)
                    assert len(local_hosts) == 4
                    full = logits_host + (16 if handle is handles[1] else 0)
                    for chip, local in enumerate(local_hosts):
                        assert torch.equal(ttnn.to_torch(local), full[..., chip * 32 : (chip + 1) * 32])
                    try:
                        pool.process(handle, is_tokens=False, sample_rows=(0, 3, 2))
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("Changed read row order accepted")
                got = pool.process(handle, is_tokens=mode, sample_rows=None if mode else rows)
                assert torch.equal(got, want), (mode, got, want)
                outputs.append(got)
                try:
                    pool.process(handle, is_tokens=mode)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("Double consumption accepted")
            outputs[0].fill_(-123)
            assert torch.equal(outputs[1], expected[1]), "Returned writable CPU outputs alias"
            pool.require_drained()
            check_cache()
            result["steps"].append(
                {
                    "output_mode": "tokens" if mode else "logits",
                    "overlap": 2,
                    "all_four_chips": True,
                    "events": True,
                    "rows": list(rows),
                }
            )
        # Exercise changed state ownership while output trace remains live.
        state.apply((3, 2, 1, 0))
        check_staging()
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        check_cache()
        result.update(status="PASS", prepared_program_entries=prepared)
    except BaseException as error:
        result.update(status="FAIL", error=repr(error))
        raise
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
            result["releases"] += 1
        ttnn.close_mesh_device(mesh)
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"SERVING_STATE_PROBE_PASS {args.output_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
