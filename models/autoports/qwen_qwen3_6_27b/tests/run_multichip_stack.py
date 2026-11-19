# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Consecutive real layers cross both boundary kinds without host/layout adapters."""

import argparse
import hashlib
import json
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc
from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import load_layer
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder


class BoundaryGuard:
    """Reject boundary adapters, while retaining legitimate inside-layer ops."""

    layout_names = {
        "to_memory_config",
        "reshard",
        "to_layout",
        "reshape",
        "permute",
        "transpose",
        "interleaved_to_sharded",
        "sharded_to_interleaved",
        "from_torch",
        "to_torch",
    }

    def __init__(self, api):
        self.api, self.depth = api, 0
        self.inside, self.outside = Counter(), []
        self.patched = []
        self.stack = ExitStack()
        self.positive_control = False

    def __enter__(self):
        for prefix, namespace in (("ttnn", self.api), ("ttnn.experimental", self.api.experimental)):
            for name in dir(namespace):
                if name not in self.layout_names and not any(
                    token in name for token in ("all_gather", "all_reduce", "reduce_scatter")
                ):
                    continue
                original = getattr(namespace, name)
                if not callable(original):
                    continue
                label = prefix + "." + name

                def guarded(*args, _original=original, _label=label, **kwargs):
                    if self.depth == 0:
                        self.outside.append(_label)
                        raise AssertionError("inter-layer boundary operation: " + _label)
                    self.inside[_label] += 1
                    return _original(*args, **kwargs)

                self.stack.enter_context(patch.object(namespace, name, guarded))
                self.patched.append(label)
        assert "ttnn.to_memory_config" in self.patched
        # Deliberately rejected before forwarding invalid arguments or touching
        # a device; this proves the boundary interception is active.
        try:
            self.api.to_memory_config(None, None)
        except AssertionError as error:
            assert str(error) == "inter-layer boundary operation: ttnn.to_memory_config"
            assert self.outside == ["ttnn.to_memory_config"]
            self.outside.clear()
            self.positive_control = True
        else:
            raise AssertionError("Boundary guard positive control was not rejected")
        return self

    @contextmanager
    def layer(self):
        assert self.depth == 0
        self.depth = 1
        try:
            yield
        finally:
            self.depth = 0

    def __exit__(self, *error):
        return self.stack.__exit__(*error)

    def contract(self, value):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "layout": str(value.layout),
            "memory_config": str(value.memory_config()),
            "rank_buffer_addresses": [int(v.buffer_address()) for v in self.api.get_device_tensors(value)],
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--length", type=int, default=65)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layers", default="0,1,2,3,4")
    p.add_argument("--policy-file", type=Path)
    p.add_argument("--activation-file", type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(8)
    torch.manual_seed(74)
    cfg = Qwen3_5TextConfig(**json.loads((root / "doc/hf_config.json").read_text())["text_config"])
    indices = list(map(int, a.layers.split(",")))
    weights = {i: load_layer(i) for i in set(indices)}
    x = torch.randn(a.batch, a.length, 5120, dtype=torch.bfloat16) * 0.1
    d = torch.randn(a.batch, 1, 5120, dtype=torch.bfloat16) * 0.1
    if a.activation_file:
        recorded = torch.load(a.activation_file, weights_only=True)
        x = recorded["x"][: a.batch, : a.length].contiguous()
        d = (
            recorded["x"][: a.batch, a.length : a.length + 1]
            if recorded["x"].shape[1] > a.length
            else recorded["d"][: a.batch]
        ).contiguous()
        assert tuple(x.shape) == (a.batch, a.length, 5120) and tuple(d.shape) == (a.batch, 1, 5120)
    pages = (a.length + 32) // 32
    table = torch.randperm(a.batch * pages).reshape(a.batch, pages)
    rotary = Qwen3_5TextRotaryEmbedding(cfg)
    if not a.baseline:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1 if a.baseline else 4),
        physical_device_ids=[0] if a.baseline else [],
        trace_region_size=32 * 1024 * 1024,
    )
    try:
        cls = OptimizedDecoder if a.baseline else MultichipDecoder
        layers = [
            cls.from_state_dict(
                weights[i],
                hf_config=cfg,
                layer_idx=i,
                mesh_device=mesh,
                **({"policy": json.loads(a.policy_file.read_text())} if a.policy_file and not a.baseline else {}),
            )
            for i in indices
        ]
        states = [
            m.create_recurrent_state(a.batch) if m.kind == "linear_attention" else m.create_paged_state(table)
            for m in layers
        ]
        upload = layers[0]._upload if a.baseline else layers[0].upload_residual
        tx = upload(x)
        td = upload(d) if a.baseline else layers[0].upload_decode_residual(d)
        ropes = [
            tuple(layers[0]._upload(t.unsqueeze(1)) for t in rotary(v, pos))
            for v, pos in [(x, torch.arange(a.length).expand(a.batch, -1)), (d, torch.full((a.batch, 1), a.length))]
        ]
        position = ttnn.from_torch(
            torch.full((a.batch,), a.length, dtype=torch.int32),
            device=mesh,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        boundary_runs = []

        def chain(value, prefill):
            record = {"phase": "prefill" if prefill else "decode", "boundaries": []}
            with BoundaryGuard(ttnn) as guard:
                previous_value, previous_contract = None, None
                for index, (m, state) in enumerate(zip(layers, states)):
                    before = tuple(value.shape)
                    incoming = guard.contract(value)
                    if index:
                        entry = {
                            "from_layer": indices[index - 1],
                            "to_layer": indices[index],
                            "same_object": value is previous_value,
                            "producer": previous_contract,
                            "consumer": incoming,
                        }
                        record["boundaries"].append(entry)
                        assert (
                            entry["same_object"] and previous_contract == incoming
                        ), "Inter-layer residual owner/layout changed"
                    with guard.layer():
                        if prefill:
                            value = m.prefill_forward(
                                value, state=state, rope=ropes[0] if m.kind == "full_attention" else None
                            )
                        else:
                            value = m.decode_forward(
                                value,
                                state=state,
                                **({"rope": ropes[1], "current_pos": position} if m.kind == "full_attention" else {}),
                            )
                    assert tuple(value.shape) == before and value.dtype == ttnn.bfloat16
                    previous_value, previous_contract = value, guard.contract(value)
                record.update(
                    positive_control_rejected=guard.positive_control,
                    patched_apis=guard.patched,
                    inside_layer_calls=dict(guard.inside),
                    boundary_calls=guard.outside,
                )
                assert guard.positive_control and not guard.outside
            boundary_runs.append(record)
            return value

        def read(t):
            vals = [ttnn.to_torch(v) for v in ttnn.get_device_tensors(t)]
            return (vals[0] if a.baseline else torch.cat(vals, -1)).reshape(a.batch, -1, 5120)

        pref = read(chain(tx, True))
        saved = [
            (
                state,
                [
                    ttnn.clone(t)
                    for t in (
                        [state.conv, state.recurrent] if m.kind == "linear_attention" else [state.key, state.value]
                    )
                ],
            )
            for m, state in zip(layers, states)
        ]

        def restore():
            for m, (state, tensors) in zip(layers, saved):
                for src, dst in zip(
                    tensors, [state.conv, state.recurrent] if m.kind == "linear_attention" else [state.key, state.value]
                ):
                    ttnn.copy(src, dst)

        eager = read(chain(td, False))
        restore()
        warm = chain(td, False)
        del warm
        restore()
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = chain(td, False)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        restore()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        got = read(out)
        assert torch.equal(eager, got)
        ttnn.release_trace(mesh, tid)
        result = {"prefill": pref, "decode": got}
        metadata = {
            "batch": a.batch,
            "length": a.length,
            "seed": 74,
            "layers": indices,
            "baseline_sha256": hashlib.sha256((root / "tt/optimized_decoder.py").read_bytes()).hexdigest(),
            "hf_revision": "8f33b98937c6031bb49e713dee3e8e185aafd26f",
        }
        assert all(torch.isfinite(v).all() for v in result.values())
        report = dict(
            batch=a.batch,
            length=a.length,
            layers=indices,
            boundary_shape=list(td.shape),
            no_boundary_conversion=all(
                r["positive_control_rejected"]
                and not r["boundary_calls"]
                and all(b["same_object"] and b["producer"] == b["consumer"] for b in r["boundaries"])
                for r in boundary_runs
            ),
            boundary_instrumentation={
                "schema_version": 1,
                "runs": boundary_runs,
                "scope": "between public decoder forward calls; inside-layer CCL/layout operations allowed",
            },
            trace_bit_equal=True,
            source_sha256=hashlib.sha256(
                (root / ("tt/optimized_decoder.py" if a.baseline else "tt/multichip_decoder.py")).read_bytes()
            ).hexdigest(),
        )
        metadata["activation_file_sha256"] = (
            hashlib.sha256(a.activation_file.read_bytes()).hexdigest() if a.activation_file else None
        )
        report["metadata"] = metadata
        report["selected_policies"] = [dict(m.policy) for m in layers]
        from models.autoports.qwen_qwen3_6_27b.tests.check_optimized_multichip_evidence import loaded_native_snapshot

        report["native_loaded"] = loaded_native_snapshot(root.parents[2])
        if a.baseline:
            torch.save(dict(result, metadata=metadata), a.reference)
        else:
            ref = torch.load(a.reference, weights_only=True)
            assert ref["metadata"] == metadata
            report["metadata"] = metadata
            report["pcc"] = {k: pcc(v, ref[k]) for k, v in result.items()}
            assert min(report["pcc"].values()) >= 0.995, report
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        print(report, flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
