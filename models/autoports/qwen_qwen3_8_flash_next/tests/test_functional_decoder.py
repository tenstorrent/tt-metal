# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full target decoder contract, including PLE and sparse QSA selection.

Serialized real-weight evidence invocation, cheap layer first:
QWEN38_WEIGHT_SOURCE=real TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m pytest -s \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_functional_decoder.py -k 'B1-layer0'

Default CI weights are deterministic synthetic weights derived from committed
real-weight statistics, with the unmodified full target config. No HF checkpoint
or download is needed by the default path. Real-weight acceptance evidence must
explicitly set QWEN38_WEIGHT_SOURCE=real.
QWEN38_DECODER_LENGTHS=1,31,32,33,127,128,129,259 overrides smoke lengths.
QWEN38_DECODER_LONG=1 additionally streams the full advertised context and its
non-aligned predecessor. The long run is required stage evidence, not a smaller
supported-context claim. QSA sparse selection is separately tested at 2053.
"""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer, Qwen4ExpTextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics
from models.autoports.qwen_qwen3_8_flash_next.tests.weight_stats import DEFAULT_CHECKPOINT, SyntheticState
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "doc/functional_decoder"
CHECKPOINT = Path(os.environ.get("QWEN38_CHECKPOINT", DEFAULT_CHECKPOINT))


class LazyPLETable(torch.nn.Module):
    """Exact HF-only lookup, following probe_ple_precision.SelectedTable.

    Read referenced checkpoint rows lazily, without constructing a 95 GiB HF
    table. TT's weight loader still loads its complete resident representation.
    This oracle never prepares/selects TT runtime table rows.
    """

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.weight = torch.empty(0, dtype=torch.bfloat16)

    def forward(self, ids):
        result = []
        for row in ids.flatten().tolist():
            shard, offset = divmod(row, 2500012)
            key = f"ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
            if isinstance(self.state, CheckpointState):
                with safe_open(self.state.path / self.state.index[key], framework="pt") as handle:
                    value = handle.get_slice(self.state.prefix + key)[offset : offset + 1].clone()
            else:
                value = self.state.rows(key, offset, offset + 1).clone()
            result.append(value)
        return torch.cat(result).reshape(*ids.shape, 160)


def load_reference(config, state, layer_idx):
    with torch.device("meta"):
        layer = Qwen4ExpTextDecoderLayer(config, layer_idx)
    weights = {key: state[key] for key in state if "ngram_embedding.shard_" not in key}
    missing, extra = layer.load_state_dict(weights, strict=False, assign=True)
    expected = ["ple.ple_embedding.ngram_embedding.weight"] if layer.ple is not None else []
    assert missing == expected and not extra, (missing, extra)
    if layer.ple is not None:
        layer.ple.ple_embedding.ngram_embedding = LazyPLETable(state)
    return layer.eval()


class HFReference:
    def __init__(self, config, state, layer_idx):
        self.config, self.layer_idx = config, layer_idx
        self.layer = load_reference(config, state, layer_idx)
        self.rope = Qwen4ExpTextRotaryEmbedding(config)
        self.caches, self.positions = {}, {}

    @torch.no_grad()
    def forward(self, x, tokens, *, slots, positions):
        outputs = []
        for lane, (slot, position) in enumerate(zip(slots, positions)):
            if position == 0:
                self.caches[slot] = DynamicCache(config=self.config)
                self.positions[slot] = 0
            assert self.positions[slot] == position, (slot, position, self.positions.get(slot))
            length = x.shape[1]
            total = position + length
            lane_x = x[lane : lane + 1]
            kwargs = {"past_key_values": self.caches[slot]}
            if self.layer.ple is not None:
                kwargs["ple_input_ids"] = tokens[lane : lane + 1].long()
            if self.layer.layer_type == "indexed_attention":
                full_positions = torch.arange(total).reshape(1, 1, total)
                kwargs["position_embeddings"] = self.rope(lane_x, full_positions)
                query_positions = torch.arange(position, total)[:, None]
                visible = torch.arange(total)[None, :] <= query_positions
                kwargs["attention_mask"] = torch.where(visible, 0.0, torch.finfo(torch.bfloat16).min).to(
                    torch.bfloat16
                )[None, None]
            else:
                kwargs["position_embeddings"] = (None, None)
            outputs.append(self.layer(lane_x, **kwargs))
            self.positions[slot] = total
        return torch.cat(outputs)


@contextmanager
def no_runtime_host_conversion():
    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime decoder attempted a host tensor boundary")

    with patch.object(ttnn, "from_torch", forbidden), patch.object(ttnn, "to_torch", forbidden), patch.object(
        torch, "tensor", forbidden
    ), patch.object(torch, "as_tensor", forbidden):
        yield


def record_pcc(expected, actual, *, layer_idx, batch, source, mode, position, length, extra=None):
    assert expected.shape == actual.shape
    metrics = pcc_metrics(expected, actual)
    score = metrics["pcc"]
    row = {
        "time_unix": time.time(),
        "layer_idx": layer_idx,
        "batch": batch,
        "weights": source,
        "mode": mode,
        "position": position,
        "length": length,
        **metrics,
        "threshold": 0.995,
        **(extra or {}),
    }
    print(json.dumps(row), flush=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with (EVIDENCE / "decoder_results.jsonl").open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    assert score >= 0.995, row


@pytest.mark.parametrize("layer_idx", [0, 3, 1], ids=["layer0", "layer3", "layer1"])
@pytest.mark.parametrize(
    "batch", [int(x) for x in os.environ.get("QWEN38_DECODER_BATCHES", "1,2").split(",")], ids=lambda value: f"B{value}"
)
def test_functional_decoder_contract(layer_idx, batch):
    torch.set_num_threads(8)
    source = os.environ.get("QWEN38_WEIGHT_SOURCE", "synthetic")
    if source == "real":
        config = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True).text_config
        state = CheckpointState(CHECKPOINT, f"model.language_model.layers.{layer_idx}.")
    elif source == "synthetic":
        state = SyntheticState(EVIDENCE / "weight_stats" / f"layer_{layer_idx}.json", seed=42)
        config = Qwen4ExpTextConfig(**state.stats["text_config"])
    else:
        raise ValueError(f"Unknown weight source {source}")
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    reference = HFReference(config, state, layer_idx)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        default_cache = (
            Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "tt-metal" / "autoports" / ROOT.name
        )
        cache_root = Path(os.environ.get("QWEN38_WEIGHT_CACHE") or default_cache).expanduser()
        cache_dir = cache_root / ("de4b8e4_ple" if source == "real" else "synthetic_seed42_block32_v1_ple")
        chunk_size = int(os.environ.get("QWEN38_DECODER_CHUNK_SIZE", "128"))
        model = FunctionalDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=layer_idx,
            mesh_device=mesh,
            batch_size=batch,
            max_requests=batch + 2,
            chunk_size=chunk_size,
            cache_dir=cache_dir,
        )
        lengths = [int(x) for x in os.environ.get("QWEN38_DECODER_LENGTHS", "1,31,32,33,127,128,129,259,3").split(",")]
        qsa_batch32_short = batch == 32 and os.environ.get("QWEN38_QSA_BATCH32_SHORT") == "1"
        if layer_idx == 3 and not qsa_batch32_short:
            lengths.append(2053)  # 513 complete blocks: indexer must exclude one.
        elif layer_idx == 3:
            print("QSA_BATCH32_SHORT_DIAGNOSTIC: sparse-selection coverage remains required at B1/B2", flush=True)
        model.prepare(lengths + [1, 2, 128, 129])

        def drain_profiler():
            # Drain finite profiler storage only between complete passes. This
            # preserves live trace metadata and is never inside a timed window.
            if os.environ.get("QWEN38_PROFILE") == "1":
                ttnn.ReadDeviceProfiler(mesh)

        def upload(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(x, dtype=dtype, layout=layout, device=mesh)

        def lane(values):
            out = torch.zeros(batch, 32, dtype=torch.int32)
            out[:, 0] = torch.tensor(values)
            return out

        # Physical KV pages are independently permuted for each request. Changing
        # page ownership across fresh requests detects stale cache/page handling.
        pages_per_request = config.max_position_embeddings // 32

        def pages(seed):
            rng = torch.Generator().manual_seed(seed)
            return torch.stack(
                [
                    torch.randperm(pages_per_request, generator=rng) + lane_idx * pages_per_request
                    for lane_idx in range(batch)
                ]
            ).to(torch.int32)

        page_host = pages(7)
        page_device = upload(page_host, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT) if layer_idx == 3 else None
        slot_ids = list(range(1, batch + 1))
        slot_device = upload(lane(slot_ids), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

        def inputs(length, seed):
            generator = torch.Generator().manual_seed(seed)
            tokens = torch.randint(0, config.vocab_size, (batch, length), generator=generator, dtype=torch.int32)
            eos = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
            if length > 2:
                tokens[:, 1] = eos
            # Four streams share a common component, approximating repeated
            # embeddings while retaining unequal streams to exercise mixing.
            base = torch.randn(batch, length, 2560, generator=generator) * 0.05
            noise = torch.randn(batch, length, 10240, generator=generator) * 0.005
            x = (base.repeat(1, 1, 4) + noise).to(torch.bfloat16)
            return x, tokens

        def run_prefill(x, tokens, position, *, label="prefill"):
            expected = reference.forward(x, tokens, slots=slot_ids, positions=[position] * batch)
            x_device = upload(x.unsqueeze(0))
            token_device = upload(tokens, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if layer_idx == 1 else None
            pos_device = upload(lane([position] * batch), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            with no_runtime_host_conversion():
                output = model.prefill_forward(
                    x_device,
                    current_positions=pos_device,
                    slots=slot_device,
                    tokens=token_device,
                    page_table=page_device,
                )
            record_pcc(
                expected,
                ttnn.to_torch(output).squeeze(0),
                layer_idx=layer_idx,
                batch=batch,
                source=source,
                mode=label,
                position=position,
                length=x.shape[1],
            )
            drain_profiler()
            return output

        drain_profiler()
        for sequence in lengths:
            x, tokens = inputs(sequence + 3, 1000 + sequence)
            if layer_idx == 3:
                page_host = pages(1000 + sequence)
                host_pages = ttnn.from_torch(page_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
                ttnn.copy_host_to_device_tensor(host_pages, page_device)
            run_prefill(x[:, :sequence], tokens[:, :sequence], 0)
            # Explicit prefix continuation, including a partial final chunk.
            run_prefill(
                x[:, sequence : sequence + 2], tokens[:, sequence : sequence + 2], sequence, label="prefix_continuation"
            )
            position = sequence + 2
            expected = reference.forward(x[:, -1:], tokens[:, -1:], slots=slot_ids, positions=[position] * batch)
            xd = upload(x[:, -1:].unsqueeze(0))
            td = upload(tokens[:, -1:], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if layer_idx == 1 else None
            pd = upload(lane([position] * batch), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            with no_runtime_host_conversion():
                out = model.decode_forward(
                    xd, current_positions=pd, slots=slot_device, tokens=td, page_table=page_device
                )
            record_pcc(
                expected,
                ttnn.to_torch(out).squeeze(0),
                layer_idx=layer_idx,
                batch=batch,
                source=source,
                mode="follow_decode",
                position=position,
                length=1,
            )
            drain_profiler()

        if os.environ.get("QWEN38_DECODER_LONG") == "1":
            # Stream full context without allocating full-context activations or
            # an SxS reference mask. Keep all real target weights and state.
            for final_mode in ("prefill", "decode"):
                position = 0
                while position < config.max_position_embeddings - 1:
                    length = min(128, config.max_position_embeddings - 1 - position)
                    model.prepare([length])
                    x, tokens = inputs(length, 9000 + position)
                    run_prefill(x, tokens, position, label=f"long_stream_{final_mode}_prefix")
                    position += length
                # Separate fresh streams establish full-context prefill and
                # decode against HF. The penultimate prefix is non-aligned.
                x, tokens = inputs(1, 9000 + position)
                expected = reference.forward(x, tokens, slots=slot_ids, positions=[position] * batch)
                xd = upload(x.unsqueeze(0))
                td = upload(tokens, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if layer_idx == 1 else None
                pd = upload(lane([position] * batch), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                final_forward = model.prefill_forward if final_mode == "prefill" else model.decode_forward
                with no_runtime_host_conversion():
                    out = final_forward(xd, current_positions=pd, slots=slot_device, tokens=td, page_table=page_device)
                record_pcc(
                    expected,
                    ttnn.to_torch(out).squeeze(0),
                    layer_idx=layer_idx,
                    batch=batch,
                    source=source,
                    mode=f"full_context_{final_mode}",
                    position=position,
                    length=1,
                    extra={"total_context": config.max_position_embeddings},
                )
                drain_profiler()

        if os.environ.get("QWEN38_PROFILE") == "1":
            from tracy import signpost

            profile_x, profile_tokens = inputs(128, 4400)
            profile_xd = upload(profile_x.unsqueeze(0))
            profile_td = upload(profile_tokens, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if layer_idx == 1 else None
            profile_pd = upload(lane([0] * batch), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            profile_kwargs = dict(
                current_positions=profile_pd, slots=slot_device, tokens=profile_td, page_table=page_device
            )
            drain_profiler()
            with no_runtime_host_conversion():
                model.prefill_forward(profile_xd, **profile_kwargs)
            ttnn.synchronize_device(mesh)
            drain_profiler()
            signpost("PERF_PREFILL")
            before = time.perf_counter()
            with no_runtime_host_conversion():
                profile_output = model.prefill_forward(profile_xd, **profile_kwargs)
            ttnn.synchronize_device(mesh)
            elapsed = time.perf_counter() - before
            signpost("PERF_PREFILL_END")
            drain_profiler()
            print(
                json.dumps(
                    {
                        "profile": "warmed_prefill",
                        "layer_idx": layer_idx,
                        "batch": batch,
                        "seq_len": 128,
                        "host_seconds": elapsed,
                        "latency_source": "host wall clock; device time requires tt-perf-report",
                    }
                ),
                flush=True,
            )

        # Preallocate and warm the non-aligned prefill used while a decode
        # trace is live. Its final one-token chunk exercises the same prepared
        # tail workspace used by decode, with persistent input buffers.
        interleave_x, interleave_tokens = inputs(129, 8100)
        interleave_xd = upload(interleave_x.unsqueeze(0))
        interleave_td = upload(interleave_tokens, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if layer_idx == 1 else None
        interleave_pd = upload(lane([0] * batch), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        interleave_kwargs = dict(
            current_positions=interleave_pd, slots=slot_device, tokens=interleave_td, page_table=page_device
        )
        with no_runtime_host_conversion():
            interleave_warm_output = model.prefill_forward(interleave_xd, **interleave_kwargs)
        ttnn.synchronize_device(mesh)
        del interleave_warm_output
        drain_profiler()

        # Warm every decode path, then capture once. Fresh requests reset on
        # device; changed input, token IDs, slots, positions and pages refresh
        # persistent buffers without creating new device tensors after capture.
        x, tokens = inputs(1, 600)
        xd = upload(x.unsqueeze(0))
        td = upload(tokens, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if layer_idx == 1 else None
        pd = upload(lane([0] * batch), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        forward_kwargs = dict(current_positions=pd, slots=slot_device, tokens=td, page_table=page_device)
        with no_runtime_host_conversion():
            model.decode_forward(xd, **forward_kwargs)
        ttnn.synchronize_device(mesh)
        drain_profiler()
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with no_runtime_host_conversion():
            traced_output = model.decode_forward(xd, **forward_kwargs)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        drain_profiler()
        replay_outputs = []
        for step in range(6):
            x, tokens = inputs(1, 600 + step % 3)
            order = slot_ids if step % 3 != 1 else list(reversed(slot_ids))
            expected = reference.forward(x, tokens, slots=order, positions=[step % 3] * batch)
            for destination, host, dtype, layout in (
                (xd, x.unsqueeze(0), ttnn.bfloat16, ttnn.TILE_LAYOUT),
                (pd, lane([step % 3] * batch), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                (slot_device, lane(order), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            ):
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=dtype, layout=layout), destination)
            if td is not None:
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), td
                )
            if layer_idx == 3:
                if step % 3 == 0:
                    trace_pages = pages(700 + step)
                # Request-owned physical pages move with their slot, while
                # lane order changes. Fresh requests receive a new permutation.
                page_rows = torch.tensor([slot_ids.index(slot) for slot in order])
                active_pages = trace_pages.index_select(0, page_rows)
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(active_pages, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), page_device
                )
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            actual = ttnn.to_torch(traced_output).squeeze(0)
            record_pcc(
                expected,
                actual,
                layer_idx=layer_idx,
                batch=batch,
                source=source,
                mode="traced_decode",
                position=step % 3,
                length=1,
                extra={"trace_id": str(trace), "capture_count": 1, "slots": order},
            )
            drain_profiler()
            if step >= 3:
                torch.testing.assert_close(actual, replay_outputs[step - 3], rtol=0, atol=0)
            replay_outputs.append(actual.clone())
        # Reuse the exact captured decode after a non-aligned prefill on the
        # same loaded instance. No new persistent device inputs or compilation
        # paths are introduced while the trace is live.
        interleaved_results = []
        for repeat in range(2):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(lane(slot_ids), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), slot_device
            )
            if layer_idx == 3:
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(pages(8200), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), page_device
                )
            prefix_expected = reference.forward(interleave_x, interleave_tokens, slots=slot_ids, positions=[0] * batch)
            with no_runtime_host_conversion():
                prefix_output = model.prefill_forward(interleave_xd, **interleave_kwargs)
            record_pcc(
                prefix_expected,
                ttnn.to_torch(prefix_output).squeeze(0),
                layer_idx=layer_idx,
                batch=batch,
                source=source,
                mode="prefill_with_live_decode_trace",
                position=0,
                length=129,
                extra={"trace_id": str(trace), "repeat": repeat},
            )
            drain_profiler()
            del prefix_output
            x, tokens = inputs(1, 8300)
            expected = reference.forward(x, tokens, slots=slot_ids, positions=[129] * batch)
            for destination, host, dtype, layout in (
                (xd, x.unsqueeze(0), ttnn.bfloat16, ttnn.TILE_LAYOUT),
                (pd, lane([129] * batch), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            ):
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=dtype, layout=layout), destination)
            if td is not None:
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), td
                )
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            actual = ttnn.to_torch(traced_output).squeeze(0)
            record_pcc(
                expected,
                actual,
                layer_idx=layer_idx,
                batch=batch,
                source=source,
                mode="traced_decode_after_interleaved_prefill",
                position=129,
                length=1,
                extra={"trace_id": str(trace), "capture_count": 1, "repeat": repeat},
            )
            drain_profiler()
            if interleaved_results:
                torch.testing.assert_close(actual, interleaved_results[0], rtol=0, atol=0)
            interleaved_results.append(actual.clone())

        if os.environ.get("QWEN38_PROFILE") == "1":
            # Position zero repeats a fresh one-token request deterministically.
            # Input copies and host comparison remain outside measured windows.
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(lane([0] * batch), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), pd
            )
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
            drain_profiler()
            signpost("PERF_DECODE")
            before = time.perf_counter()
            for _ in range(10):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            elapsed = time.perf_counter() - before
            signpost("PERF_DECODE_END")
            drain_profiler()
            print(
                json.dumps(
                    {
                        "profile": "traced_warmed_decode",
                        "layer_idx": layer_idx,
                        "batch": batch,
                        "seq_len": 1,
                        "position": 0,
                        "iterations": 10,
                        "host_seconds_per_iteration": elapsed / 10,
                        "latency_source": "host wall clock; device time requires tt-perf-report",
                    }
                ),
                flush=True,
            )
        print(f"FUNCTIONAL_DECODER_PASS layer={layer_idx} B={batch} source={source}", flush=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
