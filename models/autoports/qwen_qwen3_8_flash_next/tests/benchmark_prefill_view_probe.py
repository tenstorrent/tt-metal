# SPDX-License-Identifier: Apache-2.0
"""Reduced real-weight B32/S8 versus B1/S128 experiment; never benchmark evidence.

Requires exclusive device ownership and tracking enabled before Python starts.
Each layer uses a fresh mesh. Public state is reset identically before variants;
only one variant trace exists at a time. No device work occurs on import.
"""

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import time
import traceback
from pathlib import Path

import torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.benchmark_prefill_view_candidate import (
    assert_prefill_view_contract,
    make_prefill_decoder_view,
)
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics
from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_capacity_probe import view as memory_view
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.precision import apply_layer, for_layer, load_policy
from models.autoports.qwen_qwen3_8_flash_next.tt.state_ops import zero_inplace

ROOT = Path(__file__).resolve().parents[1]
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
BATCH, ACTIVE_ROW, SLOT, CONTEXT = 32, 5, 7, 262144
PUBLIC_TOKENS = CONTEXT + BATCH * 32
SCRATCH_BASE = PUBLIC_TOKENS // 32
ACTIVE_PAGES = [7, 1, 12, 0, 15, 4, 9, 2, 11]
SENTINEL_PAGES = [31, 101, 4095, 8191]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, payload):
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def read_shards(tensor):
    if tensor.dtype == ttnn.fp8_e4m3:
        # FP8 conversion is implemented on the mesh-composer path, not on
        # individual-device to_torch. Preserve native cache bytes/dtype.
        host = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(tensor.device(), dim=0))
        return [shard.clone() for shard in host.chunk(4, dim=0)]
    return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tensor)]


def tensor_identity(tensor):
    return dict(
        shape=list(tensor.shape),
        padded_shape=list(tensor.padded_shape),
        dtype=str(tensor.dtype),
        layout=str(tensor.layout),
        memory_config=str(tensor.memory_config()),
        buffer_address=tensor.buffer_address(),
    )


def host_copy(value, target, mesh, *, sharded=False):
    host = ttnn.from_torch(
        value.contiguous(),
        dtype=target.dtype,
        layout=target.layout,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if sharded else ttnn.ReplicateTensorToMesh(mesh),
    )
    ttnn.copy_host_to_device_tensor(host, target)


def page_table(batch):
    table = torch.stack([SCRATCH_BASE + lane * 16 + torch.arange(CONTEXT // 32) % 16 for lane in range(BATCH)])
    table[ACTIVE_ROW] = torch.arange(CONTEXT // 32)
    table[ACTIVE_ROW, : len(ACTIVE_PAGES)] = torch.tensor(ACTIVE_PAGES)
    return table.int() if batch == BATCH else table[ACTIVE_ROW : ACTIVE_ROW + 1].int()


class Segment:
    """Persistent inputs and output for one128-token composite device graph."""

    def __init__(self, layer, native_size, hidden, tokens):
        self.layer, self.mesh = layer, layer.mesh
        self.batch, self.native_size = layer.batch_size, native_size
        self.row = ACTIVE_ROW if self.batch == BATCH else 0
        self.hidden, self.tokens = hidden, tokens
        self.page = layer.ops.upload(page_table(self.batch), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        slots = torch.zeros(self.batch, 32, dtype=torch.int32)
        slots[:, 0] = torch.arange(self.batch) + BATCH
        slots[self.row, 0] = SLOT
        self.slots = layer.ops.upload(slots, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.inputs = []
        for offset in range(0, 128, native_size):
            h, ids, pos = self.host_inputs(0, offset)
            self.inputs.append(
                dict(
                    offset=offset,
                    hidden=layer.ops.shards(list(h.chunk(4, dim=0))),
                    tokens=layer.ops.upload(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                    positions=layer.ops.upload(pos, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                )
            )
        self.output = ttnn.empty((1, 1, 128, 2560), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh)

    def host_inputs(self, start, offset):
        n, row = self.native_size, self.row
        h = torch.zeros(4, self.batch, n, 2560, dtype=torch.bfloat16)
        h[:, row : row + 1] = self.hidden[:, :, start + offset : start + offset + n]
        ids = torch.zeros(self.batch, n, dtype=torch.int32)
        ids[row] = self.tokens[start + offset : start + offset + n]
        pos = torch.zeros(self.batch, 32, dtype=torch.int32)
        pos[row, 0] = start + offset
        return h, ids, pos

    def refresh(self, start):
        for item in self.inputs:
            h, ids, pos = self.host_inputs(start, item["offset"])
            host_copy(h, item["hidden"], self.mesh, sharded=True)
            host_copy(ids, item["tokens"], self.mesh)
            host_copy(pos, item["positions"], self.mesh)

    def forward(self):
        outputs = []
        for item in self.inputs:
            output = self.layer.prefill_forward(
                item["hidden"],
                current_positions=item["positions"],
                slots=self.slots,
                tokens=item["tokens"],
                page_table=self.page,
            )
            outputs.append(output[:, self.row : self.row + 1])
        joined = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)
        ttnn.copy(joined, self.output)


class Decode:
    def __init__(self, owner, hidden, tokens, *, position=256):
        self.owner = owner
        h = torch.zeros(4, BATCH, 1, 2560, dtype=torch.bfloat16)
        h[:, ACTIVE_ROW : ACTIVE_ROW + 1] = hidden[:, :, position : position + 1]
        self.hidden = owner.ops.shards(list(h.chunk(4, dim=0)))
        ids = torch.zeros(BATCH, 1, dtype=torch.int32)
        ids[ACTIVE_ROW, 0] = tokens[position]
        self.tokens = owner.ops.upload(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        pos = torch.zeros(BATCH, 32, dtype=torch.int32)
        pos[ACTIVE_ROW, 0] = position
        self.positions = owner.ops.upload(pos, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.page = owner.ops.upload(page_table(BATCH), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        slots = torch.zeros(BATCH, 32, dtype=torch.int32)
        slots[:, 0] = torch.arange(BATCH) + BATCH
        slots[ACTIVE_ROW, 0] = SLOT
        self.slots = owner.ops.upload(slots, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.output = ttnn.empty((1, 1, 1, 2560), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=owner.mesh)

    def forward(self):
        out = self.owner.decode_forward(
            self.hidden,
            current_positions=self.positions,
            tokens=self.tokens,
            slots=self.slots,
            page_table=self.page,
        )
        ttnn.copy(out[:, ACTIVE_ROW : ACTIVE_ROW + 1], self.output)


def reset(owner, candidate):
    names = (
        ("recurrent_bank", "conv_bank", "recurrent_state", "conv_state")
        if owner.kind == "linear_attention"
        else ("packed_cache", "raw_key_cache", "pooled_cache", "pooled_keys")
    )
    seen = set()
    for module in (owner, candidate):
        values = [getattr(module.attention, name) for name in names]
        if module.ple is not None:
            values += list(vars(module.ple_cache).values())
        for tensor in values:
            if id(tensor) not in seen:
                zero_inplace(owner.mesh, tensor)
                seen.add(id(tensor))


def public_state(owner):
    """Keep active state; check all unused GDN/PLE slots and selected QSA sentinels."""
    saved, checks = {}, {}

    def bank(name, tensor, rows_per_slot=1):
        values = read_shards(tensor[: BATCH * rows_per_slot])
        selected, unused = [], []
        for value in values:
            value = value.reshape(BATCH, rows_per_slot, *value.shape[1:])
            selected.append(value[SLOT].clone())
            rest = torch.cat((value[:SLOT], value[SLOT + 1 :]), dim=0)
            unused.append(
                dict(nonzero=int(torch.count_nonzero(rest.float())), finite=bool(torch.isfinite(rest.float()).all()))
            )
        saved[name] = selected
        checks[name] = dict(unused_public_slots=unused, coverage="all31unselected public slots")

    attention = owner.attention
    if owner.kind == "linear_attention":
        bank("recurrent_bank", attention.recurrent_bank)
        bank("conv_bank", attention.conv_bank)
    else:
        for name in ("packed_cache", "raw_key_cache", "pooled_cache"):
            tensor = getattr(attention, name)
            per_page = []
            for page in ACTIVE_PAGES + SENTINEL_PAGES:
                if name == "raw_key_cache":
                    part = tensor[page : page + 1]
                else:
                    stride = 8 if name == "pooled_cache" else 32
                    part = tensor[:, :, page * stride : (page + 1) * stride]
                per_page.append(read_shards(part))
            saved[name] = [
                torch.cat([page[chip].reshape(-1) for page in per_page[: len(ACTIVE_PAGES)]]) for chip in range(4)
            ]
            checks[name] = dict(
                sentinel_pages=SENTINEL_PAGES,
                sentinel_nonzero=[
                    sum(int(torch.count_nonzero(page[chip].float())) for page in per_page[len(ACTIVE_PAGES) :])
                    for chip in range(4)
                ],
                coverage="four unallocated public sentinel pages; not every untouched public page",
            )
    if owner.ple is not None:
        bank("ple_token_history", owner.ple_cache.token_history)
        bank("ple_conv_history", owner.ple_cache.conv_history, 9)
    return saved, checks


def compare(left, right):
    results = []
    for a, b in zip(left, right):
        if not torch.isfinite(a.float()).all() or not torch.isfinite(b.float()).all():
            results.append(dict(finite=False))
        elif a.dtype in (torch.int32, torch.int64, torch.uint32) or torch.equal(a.double(), b.double()):
            results.append(
                dict(
                    finite=True,
                    exact_equal=torch.equal(a.double(), b.double()),
                    max_abs_error=float((a.double() - b.double()).abs().max()),
                )
            )
        elif torch.std(a.double()) == 0 or torch.std(b.double()) == 0:
            results.append(dict(finite=True, pcc=None, max_abs_error=float((a.double() - b.double()).abs().max())))
        else:
            results.append(dict(finite=True, **pcc_metrics(a, b)))
    return results


def small_shape_checks(owner, candidate, hidden, tokens, layer_dir):
    """Same-S controls isolate batch-view changes from S8-to-S128 chunking.

    Call only after every trace has been released. These eager diagnostics may
    prepare new shapes and allocate buffers; they are not timing measurements.
    """
    candidate.prepare([1, 8])
    rows = {}
    for size in (1, 8):
        raw = {}
        decode = Decode(owner, hidden, tokens, position=size)
        for name, layer in (("owner_b32", owner), ("view_b1", candidate)):
            reset(owner, candidate)
            batch = layer.batch_size
            active_row = ACTIVE_ROW if batch == BATCH else 0
            h = torch.zeros(4, batch, size, 2560, dtype=torch.bfloat16)
            h[:, active_row : active_row + 1] = hidden[:, :, :size]
            ids = torch.zeros(batch, size, dtype=torch.int32)
            ids[active_row] = tokens[:size]
            slots = torch.zeros(batch, 32, dtype=torch.int32)
            slots[:, 0] = torch.arange(batch) + BATCH
            slots[active_row, 0] = SLOT
            uploaded = dict(
                hidden=layer.ops.shards(list(h.chunk(4, dim=0))),
                tokens=layer.ops.upload(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                positions=layer.ops.upload(
                    torch.zeros(batch, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
                ),
                slots=layer.ops.upload(slots, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                pages=layer.ops.upload(page_table(batch), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            )
            output = layer.prefill_forward(
                uploaded["hidden"],
                tokens=uploaded["tokens"],
                current_positions=uploaded["positions"],
                slots=uploaded["slots"],
                page_table=uploaded["pages"],
            )
            selected = read_shards(output[:, active_row : active_row + 1])
            state, checks = public_state(owner)
            decode.forward()
            handoff = read_shards(decode.output)
            next_state, next_checks = public_state(owner)
            raw[name] = dict(
                output=selected,
                state=state,
                state_checks=checks,
                owner_decode=handoff,
                owner_decode_state=next_state,
                owner_decode_state_checks=next_checks,
            )
            torch.save(raw, layer_dir / f"same_shape_s{size}_raw.pt")
        left, right = raw.values()
        path = layer_dir / f"same_shape_s{size}_raw.pt"
        rows[str(size)] = dict(
            output=compare(left["output"], right["output"]),
            public_state={key: compare(value, right["state"][key]) for key, value in left["state"].items()},
            owner_decode=compare(left["owner_decode"], right["owner_decode"]),
            owner_decode_public_state={
                key: compare(value, right["owner_decode_state"][key])
                for key, value in left["owner_decode_state"].items()
            },
            state_checks={name: value["state_checks"] for name, value in raw.items()},
            handoff_state_checks={name: value["owner_decode_state_checks"] for name, value in raw.items()},
            raw_artifact=dict(path=str(path), sha256=sha(path)),
            execution="eager after all traces released; fresh zero state per variant; same sequence length",
        )
    return rows


def run_layer(args, cfg, policy, layer_idx, result, save):
    layer_dir = args.output / f"layer{layer_idx}"
    layer_dir.mkdir()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_region)
    trace = None
    row = dict(layer=layer_idx, context=CONTEXT, public_pool_tokens=PUBLIC_TOKENS, variants={}, comparisons={})
    result["layers"].append(row)
    try:
        row.update(
            device_ids=list(mesh.get_device_ids()), architecture=str(mesh.arch()), initial_memory=memory_view(mesh)
        )
        owner = MultichipDecoder.from_state_dict(
            CheckpointState(args.checkpoint),
            hf_config=cfg,
            layer_idx=layer_idx,
            mesh_device=mesh,
            batch_size=BATCH,
            max_requests=2 * BATCH,
            chunk_size=128,
            attention_pool_tokens=PUBLIC_TOKENS + BATCH * 512,
            cache_dir=args.weight_cache,
        )
        apply_layer(owner, for_layer(policy, layer_idx))
        owner.prepare([1, 8])
        candidate = make_prefill_decoder_view(owner, chunk_size=128, lengths=(128,))
        row["ownership"] = assert_prefill_view_contract(owner, candidate)
        row["kind"] = owner.kind
        row["selected_layer_policy"] = owner.precision_policy
        weights = dict(
            hc=owner.attn_hyper.read_weight,
            router=owner.mlp.router,
            output=owner.attention.output_projection.row_weight,
        )
        weights["input"] = (
            owner.attention.packed_weight if owner.kind == "linear_attention" else owner.attention.packed_projection
        )
        row["weight_identity"] = {name: tensor_identity(value) for name, value in weights.items()}
        row["compute_identity"] = dict(
            output_fidelity=str(owner.attention.output_projection.compute.math_fidelity),
            routed_fidelity=str(owner.mlp.routed_compute.math_fidelity),
            ccl=owner.ops.precision_ccl,
        )
        rng = torch.Generator().manual_seed(89321 + layer_idx)
        hidden = (torch.randn(4, 1, 257, 2560, generator=rng) * 0.05).bfloat16()
        tokens = torch.randint(0, cfg.vocab_size, (257,), generator=rng).int()
        input_path = layer_dir / "inputs.pt"
        torch.save(
            dict(
                hidden_shards=hidden,
                tokens=tokens,
                page_table=page_table(BATCH),
                active_row=ACTIVE_ROW,
                public_slot=SLOT,
            ),
            input_path,
        )
        row["input_artifact"] = dict(path=str(input_path), sha256=sha(input_path))
        baseline = Segment(owner, 8, hidden, tokens)
        proposed = Segment(candidate, 128, hidden, tokens)
        decode = Decode(owner, hidden, tokens)
        segments = {"owner_b32_s8": baseline, "view_b1_s128": proposed}

        # Complete both variants, resets, continuation, owner decode, and every
        # diagnostic slice/read before ANY capture. Program-cache checks stay on.
        for segment in segments.values():
            reset(owner, candidate)
            for start in (0, 128):
                segment.refresh(start)
                segment.forward()
                read_shards(segment.output)
                public_state(owner)
            decode.forward()
            read_shards(decode.output)
            public_state(owner)
        row["prepared_memory"] = memory_view(mesh)
        save()
        retained = {}
        for name, segment in segments.items():
            reset(owner, candidate)
            segment.refresh(0)
            read_shards(segment.inputs[0]["positions"])
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            segment.forward()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            measurements, raw = [], {}
            row["variants"][name] = dict(
                trace_id=str(trace),
                native_calls_per_trace=128 // segment.native_size,
                physical_batch=segment.batch,
                trace_memory=memory_view(mesh, ttnn.BufferType.TRACE),
                timings=measurements,
            )
            save()
            for repeat in range(args.repeats):
                reset(owner, candidate)
                for start in (0, 128):
                    segment.refresh(start)
                    # Existing blocking read drains reset and input uploads;
                    # exclude this preparation boundary from diagnostic timing.
                    read_shards(segment.inputs[0]["positions"])
                    begin = time.perf_counter_ns()
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    output = read_shards(segment.output)
                    elapsed = (time.perf_counter_ns() - begin) / 1e9
                    measurements.append(dict(repeat=repeat, phase="prefill", start=start, tokens=128, seconds=elapsed))
                    # Only blocking reads of preallocated outputs occur with
                    # a live trace. No slicing/state diagnostics/eager decode.
                    raw[f"prefill_{start}"] = dict(output=output, state={}, repeat=repeat)
                    torch.save(raw, layer_dir / f"{name}_raw.pt")
                    save()
            ttnn.release_trace(mesh, trace)
            trace = None
            # Inspect state from the final traced256-token prefix only after
            # retiring the capture, then exercise actual B32-owner handoff.
            state, checks = public_state(owner)
            raw["prefill_128"]["state"] = state
            row["variants"][name]["state_checks_after256"] = checks
            decode.forward()
            handoff = read_shards(decode.output)
            state, checks = public_state(owner)
            raw["owner_decode_handoff"] = dict(output=handoff, state=state)
            row["variants"][name]["state_checks_handoff"] = checks
            path = layer_dir / f"{name}_raw.pt"
            torch.save(raw, path)
            row["variants"][name].update(
                raw_artifact=dict(path=str(path), sha256=sha(path)),
                prefill_median_seconds=statistics.median(m["seconds"] for m in measurements),
                trace_released=True,
            )
            retained[name] = raw
            save()
        left, right = retained.values()
        for phase in left:
            row["comparisons"][phase] = dict(
                output=compare(left[phase]["output"], right[phase]["output"]),
                public_state={
                    key: compare(value, right[phase]["state"][key]) for key, value in left[phase]["state"].items()
                },
            )
        row["same_shape_controls"] = small_shape_checks(owner, candidate, hidden, tokens, layer_dir)
        row["status"] = "measurements_complete_review_numerics"
        row["final_memory"] = memory_view(mesh)
        save()
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        row["mesh_closed"] = True
        save()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(f"/home/vkovacevic/qwen38-flash/hf-local/{REVISION}"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0,1,3")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trace-region", type=int, default=768000000)
    parser.add_argument(
        "--weight-cache", default="/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/de4b8e4_ple"
    )
    args = parser.parse_args()
    if (
        os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1"
        or os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0"
    ):
        raise ValueError("Launch with TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0")
    if args.repeats < 1:
        raise ValueError("At least one repeat is required")
    layers = [int(item) for item in args.layers.split(",")]
    if not layers or any(layer not in (0, 1, 3) for layer in layers):
        raise ValueError("This probe is scoped to representative real layers0,1,3")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    cfg = Qwen4ExpConfig.from_pretrained(args.checkpoint, local_files_only=True).text_config
    policy_path = ROOT / "doc/datatype_sweep/selected_precision_config.json"
    policy = load_policy(policy_path)
    if cfg.max_position_embeddings != CONTEXT or policy["config_id"] != "combined_head4_gdn4_lofi":
        raise ValueError("Full context and selected precision identity are mandatory")
    result = dict(
        status="running",
        checkpoint=str(args.checkpoint.resolve()),
        expected_revision=REVISION,
        policy=policy,
        policy_sha256=sha(policy_path),
        config_sha256=sha(args.checkpoint / "config.json"),
        checkpoint_index_sha256=sha(args.checkpoint / "model.safetensors.index.json"),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_hashes={
            str(path): sha(path)
            for path in [
                Path(__file__),
                Path(__file__).with_name("benchmark_prefill_view_candidate.py"),
                *sorted((ROOT / "tt").glob("*.py")),
            ]
        },
        trace_tracking=dict(enabled=True, skip_program_cache=False, acknowledged_tensors=[]),
        timing_contract="Reduced-layer composite trace replay through blocking readback of all four output shards; reset/input uploads excluded. Baseline trace contains16nativeS8 calls, candidate contains1nativeS128 call. Not full-model/server or roofline evidence.",
        input_contract="Seeded synthetic BF16 residual shards and token IDs with real selected-policy weights; not HF hidden-state or accuracy validation.",
        layers=[],
    )

    def save():
        dump(args.output / "summary.json", result)

    save()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    try:
        for layer in layers:
            run_layer(args, cfg, policy, layer, result, save)
        result["status"] = "measurements_complete_review_numerics"
    except BaseException:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
