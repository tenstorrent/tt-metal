# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Experimental B1 prefill staging; not selected by the production generator.

This addresses H2 program signatures, not canonical resumed-prefill accuracy.
Two persistent BF16 RM [1,1,context+32768,4096] buffers preserve the layer-major
schedule and 32768/16384 chunk caps. Reads use device embedding as runtime row
lookup. Ordinary RM slice CANNOT be used here: SliceDeviceOperation's hash
(slice/device/slice_device_operation.cpp:459+) includes slice_start and slice_end.
Embedding accepts the same BF16 RM storage as its table and runtime UINT32 row
indices (embedding/device/embedding_device_operation.cpp); fixed shape indices
make the lookup program independent of absolute start. No host residual readback
or model computation is substituted.

Writes use experimental.slice_write with both input AND output already RM.
SliceWriteRMInterleavedProgramFactory excludes outer offsets from its hash and
updates them on hits (experimental/slice_write/device/slice_write_device_operation.cpp:86+).
The wrapper converts/rebinds tiled output storage, so passing tiled staging would
break the invariant (experimental/slice_write/slice_write.cpp:119+).

Call DiagnosticPrefillStaging.prepare(gen, warmup_buckets={"linear": [...],
"full": [...]}) before any capture. Then prefill(segment_ids, start=absolute)
returns host diagnostic logits, cropped only on host. start>0 requires the cache
from exactly that prefix. This owns no cache and supports only gen.cache slot 0.
Preparation restores cache contents. Existing decode/sampling preparation must
also finish before capture; this helper does not replace generator preparation.

Allocation tracking and numerical equivalence require device execution. No
allocation exemption, profile retirement, or per-request trace recapture is
introduced. The standalone CLI can seal compilation and optionally interleave
existing split decode traces with normalized prefill.
"""

LINEAR_HEIGHTS = (1, 32, 64, 128, 511, 512, 640, 768, 960, 1024, 4096, 8192, 32768)
FULL_HEIGHTS = (32, 64, 128, 480, 512, 768, 1024, 4096, 8192, 16384)


def chunk_plan(kind, start, count):
    """Return (absolute, valid, physical, decode_prologue) with original caps."""
    result = []
    end = start + count
    while start < end:
        if kind == "full_attention" and start % 32:
            result.append((start, 1, 1, True))
            start += 1
            continue
        cap, heights = (32768, LINEAR_HEIGHTS) if kind == "linear_attention" else (16384, FULL_HEIGHTS)
        valid = min(end - start, cap)
        physical = next(height for height in heights if height >= valid)
        result.append((start, valid, physical, False))
        start += valid
    return result


class DiagnosticPrefillStaging:
    @classmethod
    def prepare(cls, gen, warmup_buckets):
        """Allocate and warm finite profiles before the first model/sampler trace."""
        spec = {key: tuple(sorted(set(warmup_buckets.get(key, ())))) for key in ("linear", "full")}
        existing = getattr(gen, "_diagnostic_prefill_staging", None)
        if existing is not None:
            if existing.profile_spec != spec:
                raise ValueError("Prepared physical profiles cannot change during this model lifetime")
            return existing
        if any(state.model_trace is not None for state in gen.states.values()):
            raise RuntimeError("Staging and every eager helper must be prepared before capture")
        obj = cls(gen, spec)
        obj._warm()
        gen._diagnostic_prefill_staging = obj
        return obj

    def __init__(self, gen, profile_spec):
        import torch

        import ttnn
        from models.autoports.ornith_ai_ornith_1_5_9b.tests.full_model_normalized_attention import (
            NormalizedAttentionPrefill,
        )
        from models.autoports.ornith_ai_ornith_1_5_9b.tests.full_model_normalized_linear import NormalizedLinearPrefill

        if gen.max_batch_size != 1:
            raise ValueError("This diagnostic only supports the owned B1 cache")
        if any(layer.hidden_width != 4096 or layer.residual_layout != "replicated" for layer in gen.model.layers):
            raise ValueError("Expected the reviewed prefill residual layout")
        for key, allowed in (("linear", LINEAR_HEIGHTS), ("full", FULL_HEIGHTS)):
            if any(height not in allowed for height in profile_spec[key]):
                raise ValueError(f"Unknown {key} physical profile")
        self.gen, self.model, self.device = gen, gen.model, gen.device
        self.context, self.profile_spec = gen.max_seq_len, profile_spec
        self.capacity = self.context + 32768
        self.staging = [
            ttnn.zeros(
                (1, 1, self.capacity, 4096),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        ]
        self.tokens = self._upload(torch.zeros(1, 1024, dtype=torch.int32), dtype=ttnn.uint32)
        self.position = self._upload(torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32)
        self.table = self._upload(gen.page_table[:1], dtype=ttnn.int32)
        heights = {1, 32, 1024, *profile_spec["linear"], *profile_spec["full"]}
        self.row_indices = {
            height: self._upload(torch.arange(height, dtype=torch.int32)[None], dtype=ttnn.uint32) for height in heights
        }
        self.layers = []
        for decoder, states in zip(self.model.layers, gen.cache, strict=True):
            if decoder.kind == "linear_attention":
                if decoder.chunk_size != 32768:
                    raise ValueError("Do not replace the reviewed linear chunk cap")
                self.layers.append(NormalizedLinearPrefill(decoder, buckets=profile_spec["linear"]))
            else:
                if decoder.chunk_size != 16384:
                    raise ValueError("Do not replace the reviewed full-attention chunk cap")
                self.layers.append(
                    {
                        height: NormalizedAttentionPrefill(
                            decoder, state=states[0], physical_bucket=height, supported_context=self.context
                        )
                        for height in profile_spec["full"]
                    }
                )
        self.prepared = False
        self.warm_program_counts = None

    def _upload(self, value, *, dtype, device=True):
        import ttnn

        return self.model.upload(value, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    def _copy_integer(self, value, destination, *, signed=False):
        import ttnn

        host = self._upload(value, dtype=ttnn.int32 if signed else ttnn.uint32, device=False)
        ttnn.copy_host_to_device_tensor(host, destination)

    def _read(self, source, absolute, height):
        import torch

        import ttnn

        if absolute < 0 or absolute + height > self.capacity:
            raise ValueError("Physical lookup exceeds configured staging slack")
        self._copy_integer(torch.arange(absolute, absolute + height, dtype=torch.int32)[None], self.row_indices[height])
        rows = ttnn.embedding(self.row_indices[height], source, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(rows, (1, 1, height, 4096))

    @staticmethod
    def _write(value, destination, absolute):
        import ttnn

        value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)
        height = value.shape[2]
        ttnn.experimental.slice_write(
            value, destination, (0, 0, absolute, 0), (1, 1, absolute + height, 4096), (1, 1, 1, 1)
        )

    def _layer_chunk(self, index, source, destination, absolute, valid, height, prologue=False):
        import torch

        import ttnn

        layer, state, adapter = self.model.layers[index], self.gen.cache[index][0], self.layers[index]
        hidden = self._read(source, absolute, height)
        if prologue:
            self._copy_integer(torch.tensor([absolute], dtype=torch.int32), self.position, signed=True)
            result = layer.decode_forward(hidden, state=state, page_table=self.table, current_position=self.position)
        elif layer.kind == "linear_attention":
            selected = adapter.refresh(valid)
            if selected != height:
                raise AssertionError("Physical linear profile selection mismatch")
            result = adapter.forward(hidden, state=state)
        else:
            adapter[height].refresh(start=absolute, valid_count=valid, page_mapping=self.gen.page_table[0].tolist())
            result = adapter[height].forward(hidden)
        self._write(ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG), destination, absolute)

    def _embed(self, ids, start):
        import torch

        for offset in range(0, len(ids), 1024):
            physical = torch.zeros(1, 1024, dtype=torch.int32)
            chunk = ids[offset : offset + 1024]
            physical[0, : len(chunk)] = torch.as_tensor(chunk, dtype=torch.int32)
            self._copy_integer(physical, self.tokens)
            self._write(self.model.embed(self.tokens), self.staging[0], start + offset)

    def _head(self, source, start, count, *, last_only):
        import torch

        results = []
        positions = [start + count - 1] if last_only else range(start, start + count, 32)
        for absolute in positions:
            logits = self.model.logits(self._read(source, absolute, 32))
            host = self.model.host_logits(logits).reshape(1, 32, self.model.vocab_size)
            real = 1 if last_only else min(32, start + count - absolute)
            results.append(host[:, :real].clone())
        return torch.cat(results, dim=1)

    def _warm(self):
        import ttnn

        saved = []
        for states in self.gen.cache:
            for name in ("conv", "recurrent", "key", "value"):
                tensor = getattr(states[0], name)
                if tensor is not None:
                    saved.append((tensor, ttnn.clone(tensor)))
        before = self.device.num_program_cache_entries()
        try:
            self.gen.reset(clear_kv=True)
            self._embed([0], 0)
            # Physical source/destination specs are identical on both sides;
            # changing their backing address is a runtime argument on cache hits.
            for index, layer in enumerate(self.model.layers):
                key = "linear" if layer.kind == "linear_attention" else "full"
                for height in self.profile_spec[key]:
                    valid = min(height, self.context)
                    if layer.kind == "linear_attention" and next(h for h in LINEAR_HEIGHTS if h >= valid) != height:
                        raise ValueError("Configured context cannot select this linear profile")
                    self._layer_chunk(index, self.staging[0], self.staging[1], 0, valid, height)
                if layer.kind == "full_attention":
                    self._layer_chunk(index, self.staging[0], self.staging[1], 1, 1, 1, prologue=True)
            self._head(self.staging[1], 0, 1, last_only=True)
            self.gen.reset(clear_kv=True)
            for destination, snapshot in saved:
                ttnn.copy(snapshot, destination)
            ttnn.synchronize_device(self.device)
            self.prepared = True
            self.warm_program_counts = {"before": before, "after": self.device.num_program_cache_entries()}
        finally:
            # Restore also on a failed warmup; do not leave a live request with
            # warmup-mutated recurrence or K/V contents.
            if not self.prepared:
                for destination, snapshot in saved:
                    ttnn.copy(snapshot, destination)

    def prefill(self, ids, *, start=0, last_only=False):
        if not self.prepared:
            raise RuntimeError("Preparation must finish before request admission")
        ids = [int(token) for token in ids]
        if not ids or start < 0 or start + len(ids) > self.context:
            raise ValueError("Invalid logical prefill interval")
        if any(token < 0 or token >= self.model.vocab_size for token in ids):
            raise ValueError("Invalid token ID")
        # Check every profile before changing request state.
        plans = [chunk_plan(layer.kind, start, len(ids)) for layer in self.model.layers]
        for layer, plan in zip(self.model.layers, plans, strict=True):
            key = "linear" if layer.kind == "linear_attention" else "full"
            if any(height not in self.profile_spec[key] for _, _, height, prologue in plan if not prologue):
                raise ValueError("Diagnostic profile not prepared; configure it before capture")
        self._embed(ids, start)
        source, destination = self.staging
        for index, plan in enumerate(plans):
            for absolute, valid, height, prologue in plan:
                self._layer_chunk(index, source, destination, absolute, valid, height, prologue)
            source, destination = destination, source
        return self._head(source, start, len(ids), last_only=last_only)


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="33,34,65,513")
    parser.add_argument("--starts", default="0,32,68")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    lengths = [int(value) for value in args.lengths.split(",")]
    starts = [int(value) for value in args.starts.split(",")]
    if any(start < 0 or start + length + 1 > args.context for start in starts for length in lengths):
        parser.error("All prefix/segment/decode intervals must fit the configured context")
    if args.context % 32 or any(length < 1 for length in lengths):
        parser.error("Context must be page aligned and lengths positive")
    profiles = {"linear": set(), "full": set()}
    for kind, key in (("linear_attention", "linear"), ("full_attention", "full")):
        for start in starts:
            intervals = [(start, length) for length in lengths] + ([(0, start)] if start else [])
            for absolute, count in intervals:
                profiles[key].update(
                    height for _, _, height, prologue in chunk_plan(kind, absolute, count) if not prologue
                )
    profiles = {key: sorted(value) for key, value in profiles.items()}

    import torch

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator

    torch.set_num_threads(8)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"experimental": True, "profiles": profiles, "context": args.context, "cases": []}
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = build_generator(None, mesh, layer_indices=[0, 3], max_seq_len=args.context)
        helper = DiagnosticPrefillStaging.prepare(gen, profiles)
        report["warm_program_counts"] = helper.warm_program_counts
        report["staging_bytes_per_chip"] = 2 * helper.capacity * 4096 * 2
        if args.trace:
            state = gen.prepare_decode(kv_cache=gen.cache, page_table=gen.page_table[:1], slots=(0,))
            gen.capture(state)
        baseline = mesh.num_program_cache_entries()
        report["sealed_program_count"] = baseline
        mesh.set_program_cache_misses_allowed(False)
        for start in starts:
            # Return to a previously seen shape after different logical lengths.
            for length in [*lengths, lengths[0]]:
                gen.reset(clear_kv=True)
                if start:
                    helper.prefill(list(range(100, 100 + start)), last_only=True)
                before = mesh.num_program_cache_entries()
                logits = helper.prefill(list(range(200, 200 + length)), start=start, last_only=True)
                record = {
                    "start": start,
                    "length": length,
                    "before": before,
                    "after_prefill": mesh.num_program_cache_entries(),
                    "finite": bool(torch.isfinite(logits).all()),
                    "last_argmax_diagnostic": int(logits[0, -1].argmax()),
                }
                if args.trace:
                    predicted = gen.decode_forward(
                        torch.tensor([[record["last_argmax_diagnostic"]]]),
                        [start + length],
                        page_table=gen.page_table[:1],
                        kv_cache=gen.cache,
                    )
                    record["decode_token"] = int(predicted[0])
                    record["trace_ids"] = [state.model_trace, state.sample_trace]
                record["after"] = mesh.num_program_cache_entries()
                record["passed"] = record["finite"] and record["before"] == record["after"] == baseline
                report["cases"].append(record)
                path.write_text(json.dumps(report, indent=2, default=str) + "\n")
                print(json.dumps(record, default=str), flush=True)
                if not record["passed"]:
                    raise AssertionError("Post-prepare program count changed or output is not finite")
        report["passed"] = True
    except BaseException as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        mesh.set_program_cache_misses_allowed(True)
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
